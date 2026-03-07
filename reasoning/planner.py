"""
Planner — decomposes high-level goals into structured step sequences.

Two planning modes:
  1. **LLM planner** (default) — asks a language model to break the goal
     into an ordered list of abstract steps.
  2. **Template planner** — matches known goal patterns to pre-built
     step templates (zero LLM calls, instant).

The plan is a first-pass *sketch*.  The ReasoningEngine may deviate
from it at runtime when the environment doesn't match expectations.

Plan schema::

    Plan {
        goal:       str
        steps:      [Step, ...]
        status:     "pending" | "in_progress" | "completed" | "failed"
        metadata:   dict          # timing, model used, etc.
    }

    Step {
        index:      int
        action:     str           # e.g. "open_app", "navigate", "click", …
        target:     str           # argument / description
        status:     "pending" | "in_progress" | "done" | "failed" | "skipped"
        result:     str | None    # outcome once executed
        depends_on: list[int]     # indices this step depends on (default [])
    }
"""

import heapq
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Actions the planner is allowed to emit ───────────────────────

VALID_ACTIONS = frozenset({
    "open_app", "open_browser", "navigate", "search", "click",
    "type", "press_key", "scroll", "hotkey",
    "download", "run_command", "draw_plan",
    "mouse_click_xy", "mouse_drag",
    "move_file", "create_folder", "delete_file",
    "wait", "done",
})


_PHASE_PREP = frozenset({
    "open_app", "open_browser", "navigate", "run_command", "create_folder"
})
_PHASE_EXEC = frozenset({
    "search", "click", "type", "press_key", "scroll", "hotkey",
    "download", "draw_plan", "mouse_click_xy", "mouse_drag"
})
_PHASE_FINAL = frozenset({
    "move_file", "delete_file", "wait", "done"
})


# ── Data classes ─────────────────────────────────────────────────

class Step:
    """One atomic step inside a Plan."""

    __slots__ = ("index", "action", "target", "status", "result", "depends_on")

    def __init__(self, index: int, action: str, target: str = "",
                 depends_on: list[int] | None = None):
        self.index = index
        self.action = action
        self.target = target
        self.status = "pending"        # pending → in_progress → done/failed/skipped
        self.result: str | None = None
        self.depends_on: list[int] = depends_on or []

    # -- Lifecycle helpers --

    def start(self) -> None:
        self.status = "in_progress"

    def complete(self, result: str = "") -> None:
        self.status = "done"
        self.result = result

    def fail(self, reason: str = "") -> None:
        self.status = "failed"
        self.result = reason

    def skip(self, reason: str = "") -> None:
        self.status = "skipped"
        self.result = reason

    # -- Serialisation --

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    def __repr__(self) -> str:
        return (f"Step({self.index}, {self.action!r}, "
                f"{self.target!r}, {self.status})")


class TaskNode:
    """Hierarchical planning node. Leaf nodes map to executable actions."""

    __slots__ = (
        "id", "title", "action", "target", "status", "result",
        "depends_on", "children", "parent_id", "metadata",
    )

    def __init__(
        self,
        title: str,
        *,
        action: str = "",
        target: str = "",
        depends_on: list[str] | None = None,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        node_id: str | None = None,
    ):
        self.id = node_id or str(uuid.uuid4())
        self.title = title
        self.action = action
        self.target = target
        self.status = "pending"
        self.result: str | None = None
        self.depends_on = depends_on or []
        self.children: list[TaskNode] = []
        self.parent_id = parent_id
        self.metadata = metadata or {}

    def add_child(self, child: "TaskNode") -> None:
        child.parent_id = self.id
        self.children.append(child)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "action": self.action,
            "target": self.target,
            "status": self.status,
            "result": self.result,
            "depends_on": self.depends_on,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


class ScheduledTask:
    """Represents a delayed or recurring task goal."""

    __slots__ = (
        "id", "goal", "run_at", "priority", "status",
        "created_at", "context", "recurrence_seconds", "last_run_at",
    )

    def __init__(
        self,
        goal: str,
        run_at: float,
        *,
        priority: int = 0,
        context: str = "",
        recurrence_seconds: float | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.goal = goal
        self.run_at = run_at
        self.priority = priority
        self.status = "scheduled"
        self.created_at = time.time()
        self.context = context
        self.recurrence_seconds = recurrence_seconds
        self.last_run_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "run_at": self.run_at,
            "run_at_iso": datetime.fromtimestamp(self.run_at).isoformat(),
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "context": self.context,
            "recurrence_seconds": self.recurrence_seconds,
            "last_run_at": self.last_run_at,
        }


class TaskScheduler:
    """Persistent scheduler for one-shot and recurring goals.

    Tasks are kept in-memory for fast access and flushed to a JSON file
    on every mutation so they survive restarts.
    """

    def __init__(self, persist_path: Path | str | None = None):
        self._lock = threading.Lock()
        self._heap: list[tuple[float, int, int, str]] = []
        self._tasks: dict[str, ScheduledTask] = {}
        self._seq = 0
        self._persist_path: Path | None = (
            Path(persist_path) if persist_path else None
        )
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    def schedule(
        self,
        goal: str,
        run_at: float,
        *,
        context: str = "",
        priority: int = 0,
        recurrence_seconds: float | None = None,
    ) -> ScheduledTask:
        task = ScheduledTask(
            goal,
            run_at,
            priority=priority,
            context=context,
            recurrence_seconds=recurrence_seconds,
        )
        with self._lock:
            self._tasks[task.id] = task
            self._seq += 1
            heapq.heappush(self._heap, (task.run_at, -task.priority, self._seq, task.id))
        self._save()
        return task

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            task.status = "cancelled"
            self._tasks.pop(task_id, None)
            self._save()
            return True

    def list_tasks(self) -> list[ScheduledTask]:
        with self._lock:
            tasks = list(self._tasks.values())
        return sorted(tasks, key=lambda t: (t.run_at, -t.priority))

    def peek_next(self) -> ScheduledTask | None:
        tasks = self.list_tasks()
        return tasks[0] if tasks else None

    def pop_due(self, now_ts: float | None = None) -> list[ScheduledTask]:
        now_ts = now_ts or time.time()
        due: list[ScheduledTask] = []

        with self._lock:
            while self._heap:
                run_at, _neg_prio, _seq, task_id = self._heap[0]
                if run_at > now_ts:
                    break
                heapq.heappop(self._heap)
                task = self._tasks.get(task_id)
                if not task:
                    continue
                if task.status == "cancelled":
                    self._tasks.pop(task_id, None)
                    continue

                task.last_run_at = now_ts
                due.append(task)

                if task.recurrence_seconds and task.recurrence_seconds > 0:
                    task.run_at = now_ts + task.recurrence_seconds
                    task.status = "scheduled"
                    self._seq += 1
                    heapq.heappush(
                        self._heap,
                        (task.run_at, -task.priority, self._seq, task.id),
                    )
                else:
                    task.status = "dispatched"
                    self._tasks.pop(task.id, None)

        if due:
            self._save()
        return due

    def _save(self) -> None:
        """Flush current tasks to disk."""
        if not self._persist_path:
            return
        with self._lock:
            payload = [t.to_dict() for t in self._tasks.values()]
        tmp = self._persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._persist_path)

    def _load(self) -> None:
        """Restore tasks from disk (called once in __init__)."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[TaskScheduler] Failed to load {self._persist_path}: {exc}")
            return
        for entry in raw:
            task = ScheduledTask(
                goal=entry["goal"],
                run_at=entry["run_at"],
                priority=entry.get("priority", 0),
                context=entry.get("context", ""),
                recurrence_seconds=entry.get("recurrence_seconds"),
            )
            task.id = entry.get("id", task.id)
            task.status = entry.get("status", "scheduled")
            task.created_at = entry.get("created_at", task.created_at)
            task.last_run_at = entry.get("last_run_at")
            if task.status in ("scheduled",):
                self._tasks[task.id] = task
                self._seq += 1
                heapq.heappush(
                    self._heap,
                    (task.run_at, -task.priority, self._seq, task.id),
                )
        if self._tasks:
            print(f"[TaskScheduler] Loaded {len(self._tasks)} scheduled tasks from disk.")


class Plan:
    """An ordered sequence of Steps toward a goal."""

    def __init__(
        self,
        goal: str,
        steps: list[Step] | None = None,
        task_tree: list[TaskNode] | None = None,
    ):
        self.goal = goal
        self.steps: list[Step] = steps or []
        self.task_tree: list[TaskNode] = task_tree or []
        self.status = "pending"        # pending → in_progress → completed/failed
        self.created_at = time.time()
        self.metadata: dict[str, Any] = {}

        if not self.task_tree and self.steps:
            self.task_tree = self._build_linear_task_tree(self.steps)

    # -- Step access --

    @property
    def current_step(self) -> Step | None:
        """Return the first non-terminal step, or None if all done."""
        for s in self.steps:
            if s.status in ("pending", "in_progress"):
                return s
        return None

    @property
    def current_index(self) -> int:
        s = self.current_step
        return s.index if s else len(self.steps)

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.status in ("done", "skipped"))
        return done / len(self.steps)

    @property
    def is_complete(self) -> bool:
        return all(s.status in ("done", "skipped") for s in self.steps)

    @property
    def has_failed(self) -> bool:
        return any(s.status == "failed" for s in self.steps)

    # -- Mutation --

    def advance(self, result: str = "") -> Step | None:
        """Mark current step done and return the next one."""
        cur = self.current_step
        if cur:
            cur.complete(result)
        nxt = self.current_step
        if nxt is None:
            self.status = "completed"
        return nxt

    def fail_current(self, reason: str = "") -> None:
        cur = self.current_step
        if cur:
            cur.fail(reason)
        self.status = "failed"

    def insert_step(self, after_index: int, action: str,
                    target: str = "") -> Step:
        """Dynamically insert a new step (re-planning at runtime)."""
        new_idx = after_index + 1
        # Shift subsequent indices
        for s in self.steps:
            if s.index >= new_idx:
                s.index += 1
        step = Step(new_idx, action, target)
        insert_pos = 0
        for i, s in enumerate(self.steps):
            if s.index > after_index:
                insert_pos = i
                break
        else:
            insert_pos = len(self.steps)
        self.steps.insert(insert_pos, step)
        return step

    # -- Serialisation --

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "status": self.status,
            "progress": round(self.progress, 2),
            "steps": [s.to_dict() for s in self.steps],
            "task_tree": [n.to_dict() for n in self.task_tree],
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Compact text summary for LLM context injection."""
        lines = [f"Plan for: {self.goal}  ({self.status}, {self.progress:.0%} done)"]
        if self.task_tree:
            lines.append("Hierarchy:")
            for root in self.task_tree:
                self._append_tree_summary(lines, root, depth=0, max_depth=2)
        for s in self.steps:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "done": "[x]",
                "failed": "[!]",
                "skipped": "[-]",
            }.get(s.status, "[?]")
            line = f"  {marker} [{s.index}] {s.action}"
            if s.target:
                line += f" {s.target}"
            if s.result:
                line += f"  → {s.result}"
            lines.append(line)
        return "\n".join(lines)

    def set_task_tree(self, roots: list[TaskNode]) -> None:
        self.task_tree = roots
        self.steps = self._flatten_tree_to_steps(roots)

    def _build_linear_task_tree(self, steps: list[Step]) -> list[TaskNode]:
        root = TaskNode("Task Plan")
        for s in steps:
            root.add_child(
                TaskNode(
                    title=f"Step {s.index}: {s.action}",
                    action=s.action,
                    target=s.target,
                    metadata={"step_index": s.index},
                )
            )
        return [root]

    def _flatten_tree_to_steps(self, roots: list[TaskNode]) -> list[Step]:
        flat: list[Step] = []

        def walk(node: TaskNode) -> None:
            if node.is_leaf and node.action:
                flat.append(Step(len(flat), node.action, node.target))
                return
            for child in node.children:
                walk(child)

        for r in roots:
            walk(r)

        if not flat or flat[-1].action != "done":
            flat.append(Step(len(flat), "done", ""))
        return flat

    def _append_tree_summary(
        self,
        lines: list[str],
        node: TaskNode,
        *,
        depth: int,
        max_depth: int,
    ) -> None:
        indent = "  " + ("  " * depth)
        title = node.title or node.action or "task"
        lines.append(f"{indent}- {title}")
        if depth >= max_depth:
            return
        for child in node.children:
            self._append_tree_summary(lines, child, depth=depth + 1, max_depth=max_depth)

    def __repr__(self) -> str:
        return f"Plan({self.goal!r}, {len(self.steps)} steps, {self.status})"


# ── Planner ──────────────────────────────────────────────────────

_LLM_PLAN_PROMPT = """\
You are a task-planning AI for an autonomous computer agent running on Windows.

Given a goal, decompose it into a minimal ordered list of concrete steps.

Each step must be one of these actions:
  open_app, open_browser, navigate, search, click, type, press_key,
  scroll, hotkey, download, run_command, draw_plan,
  mouse_click_xy, mouse_drag, move_file, create_folder, delete_file, wait, done

Output ONLY a JSON array of step objects.  No markdown, no explanation.
Each step: {"action": "...", "target": "description / argument"}

Rules:
- Use open_browser if the goal requires the internet.
- End every plan with {"action": "done", "target": ""}.
- Keep it minimal — fewest steps possible.
- For file operations, use run_command with Windows commands.
- For drawing, open_app mspaint then draw_plan.

Example — Goal: "Search for cute cat pictures on Google"
[
  {"action": "open_browser", "target": "cute cat pictures"},
  {"action": "done", "target": ""}
]

Example — Goal: "Create a folder called Projects on the Desktop"
[
  {"action": "run_command", "target": "mkdir C:\\\\Users\\\\user\\\\Desktop\\\\Projects"},
  {"action": "done", "target": ""}
]
"""


class Planner:
    """Decomposes a goal string into a Plan object."""

    def __init__(self, model: str = "llama-3.1-8b-instant",
                 store_dir: str = "memory_store"):
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = model

        # Template bank for zero-LLM-call plans
        self._templates: dict[str, list[dict]] = {}
        self.scheduler = TaskScheduler(
            persist_path=Path(store_dir) / "scheduled_tasks.json",
        )

    # ── Public API ───────────────────────────────────────────────

    def create_plan(self, goal: str, context: str = "") -> Plan:
        """Generate a Plan for *goal*.

        Tries template match first, falls back to LLM.
        """
        # 1. Try template
        plan = self._try_template(goal)
        if plan:
            plan.metadata["source"] = "template"
            self._attach_hierarchy(plan)
            return plan

        # 2. LLM planning
        plan = self._llm_plan(goal, context)
        plan.metadata["source"] = "llm"
        plan.metadata["model"] = self._model
        self._attach_hierarchy(plan)
        return plan

    def schedule_task(
        self,
        goal: str,
        run_at: float | datetime | str,
        *,
        context: str = "",
        priority: int = 0,
        recurrence_seconds: float | None = None,
    ) -> ScheduledTask:
        """Schedule a goal for future dispatch.

        ``run_at`` supports:
        - epoch timestamp (float/int)
        - datetime
        - ISO-8601 string ("YYYY-MM-DDTHH:MM[:SS]")
        """
        run_ts = self._normalize_run_at(run_at)
        return self.scheduler.schedule(
            goal,
            run_ts,
            context=context,
            priority=priority,
            recurrence_seconds=recurrence_seconds,
        )

    def pop_due_scheduled_tasks(self, now_ts: float | None = None) -> list[ScheduledTask]:
        return self.scheduler.pop_due(now_ts)

    def list_scheduled_tasks(self) -> list[ScheduledTask]:
        return self.scheduler.list_tasks()

    def cancel_scheduled_task(self, task_id: str) -> bool:
        return self.scheduler.cancel(task_id)

    def replan(self, plan: Plan, reason: str,
               world_context: str = "") -> Plan:
        """Generate a new plan when the current one fails mid-way.

        Preserves completed steps and replans from the failure point."""
        completed = [s for s in plan.steps if s.status == "done"]
        completed_summary = ", ".join(
            f"{s.action} {s.target}" for s in completed
        ) or "(none)"

        prompt = (
            f"The original goal was: {plan.goal}\n"
            f"Steps already completed: {completed_summary}\n"
            f"Reason for replanning: {reason}\n\n"
            f"World context:\n{world_context}\n\n"
            f"Generate the REMAINING steps (do not repeat completed ones)."
        )
        new_plan = self._llm_plan(plan.goal, prompt)
        new_plan.metadata["source"] = "replan"
        new_plan.metadata["completed_before_replan"] = len(completed)

        # Prepend completed steps for continuity
        offset = len(completed)
        for s in new_plan.steps:
            s.index += offset
        new_plan.steps = [
            Step(s.index, s.action, s.target) for s in completed
        ] + new_plan.steps
        # Mark the carried-over steps as done
        for i in range(offset):
            new_plan.steps[i].status = "done"

        self._attach_hierarchy(new_plan)

        return new_plan

    def validate(self, plan: Plan) -> list[str]:
        """Return a list of validation warnings (empty = OK)."""
        warnings = []
        for s in plan.steps:
            if s.action not in VALID_ACTIONS:
                warnings.append(
                    f"Step {s.index}: unknown action '{s.action}'"
                )
        if plan.steps and plan.steps[-1].action != "done":
            warnings.append("Plan does not end with a 'done' step")
        return warnings

    # ── Templates ────────────────────────────────────────────────

    def register_template(self, pattern: str,
                          steps: list[dict]) -> None:
        """Register a goal-pattern → steps template.

        ``pattern`` is matched case-insensitively against the goal.
        """
        self._templates[pattern.lower()] = steps

    def _try_template(self, goal: str) -> Plan | None:
        g = goal.lower()
        for pattern, raw_steps in self._templates.items():
            if pattern in g:
                steps = [
                    Step(i, s["action"], s.get("target", ""))
                    for i, s in enumerate(raw_steps)
                ]
                return Plan(goal, steps)
        return None

    # ── LLM planning ────────────────────────────────────────────

    def _llm_plan(self, goal: str, extra_context: str = "") -> Plan:
        user_msg = f"Goal: {goal}"
        if extra_context:
            user_msg += f"\n\nContext:\n{extra_context}"

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _LLM_PLAN_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=500,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            raw_steps: list[dict] = json.loads(raw)
            steps = [
                Step(i, s.get("action", "done"),
                     s.get("target", ""))
                for i, s in enumerate(raw_steps)
            ]
        except Exception as e:
            print(f"[Planner] LLM planning failed: {e}")
            # Fallback: single-step plan — let the reasoning engine
            # figure out the steps at runtime (original behavior)
            steps = [Step(0, "done", "")]

        plan = Plan(goal, steps)
        return plan

    def _attach_hierarchy(self, plan: Plan) -> None:
        """Build a practical task hierarchy from linear steps."""
        roots = self._build_hierarchy_from_steps(plan.steps)
        if roots:
            plan.set_task_tree(roots)

    def _build_hierarchy_from_steps(self, steps: list[Step]) -> list[TaskNode]:
        if not steps:
            return [TaskNode("Task Plan")]

        root = TaskNode("Task Plan")
        prep = TaskNode("Preparation")
        execute = TaskNode("Execution")
        finalize = TaskNode("Finalization")

        has_prep = False
        has_exec = False
        has_final = False

        for s in steps:
            leaf = TaskNode(
                title=f"Step {s.index}: {s.action}",
                action=s.action,
                target=s.target,
                metadata={"step_index": s.index},
            )
            if s.action in _PHASE_PREP:
                prep.add_child(leaf)
                has_prep = True
            elif s.action in _PHASE_EXEC:
                execute.add_child(leaf)
                has_exec = True
            elif s.action in _PHASE_FINAL:
                finalize.add_child(leaf)
                has_final = True
            else:
                execute.add_child(leaf)
                has_exec = True

        if has_prep:
            root.add_child(prep)
        if has_exec:
            root.add_child(execute)
        if has_final:
            root.add_child(finalize)

        if not root.children:
            root.add_child(TaskNode("Execution"))

        return [root]

    @staticmethod
    def _normalize_run_at(run_at: float | datetime | str) -> float:
        if isinstance(run_at, (int, float)):
            return float(run_at)
        if isinstance(run_at, datetime):
            return run_at.timestamp()
        if isinstance(run_at, str):
            dt = datetime.fromisoformat(run_at)
            return dt.timestamp()
        raise TypeError("run_at must be float, datetime, or ISO datetime string")
