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

import json
import os
import time
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


class Plan:
    """An ordered sequence of Steps toward a goal."""

    def __init__(self, goal: str, steps: list[Step] | None = None):
        self.goal = goal
        self.steps: list[Step] = steps or []
        self.status = "pending"        # pending → in_progress → completed/failed
        self.created_at = time.time()
        self.metadata: dict[str, Any] = {}

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
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Compact text summary for LLM context injection."""
        lines = [f"Plan for: {self.goal}  ({self.status}, "
                 f"{self.progress:.0%} done)"]
        for s in self.steps:
            marker = {"pending": "○", "in_progress": "▶",
                      "done": "✓", "failed": "✗",
                      "skipped": "—"}.get(s.status, "?")
            line = f"  {marker} [{s.index}] {s.action}"
            if s.target:
                line += f" {s.target}"
            if s.result:
                line += f"  → {s.result}"
            lines.append(line)
        return "\n".join(lines)

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

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self._client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._model = model

        # Template bank for zero-LLM-call plans
        self._templates: dict[str, list[dict]] = {}

    # ── Public API ───────────────────────────────────────────────

    def create_plan(self, goal: str, context: str = "") -> Plan:
        """Generate a Plan for *goal*.

        Tries template match first, falls back to LLM.
        """
        # 1. Try template
        plan = self._try_template(goal)
        if plan:
            plan.metadata["source"] = "template"
            return plan

        # 2. LLM planning
        plan = self._llm_plan(goal, context)
        plan.metadata["source"] = "llm"
        plan.metadata["model"] = self._model
        return plan

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
