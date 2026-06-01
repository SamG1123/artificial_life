"""
SkillGraph — tracks the agent's evolving capabilities as a directed
graph of skills with proficiency levels, dependencies, and usage stats.

Structure::

    SkillNode {
        name:           str           — unique skill identifier
        category:       str           — grouping (web, system, drawing, …)
        proficiency:    float 0-1     — mastery level (updated on use)
        xp:             int           — raw experience points
        uses:           int           — times executed
        successes:      int           — successful executions
        failures:       int           — failed executions
        last_used:      float | None  — epoch timestamp
        dependencies:   list[str]     — skills this skill depends on
        unlocked:       bool          — all dependencies met
        metadata:       dict          — extra info (source, …)
    }

Proficiency is computed from the success rate smoothed with a minimum
sample size (Bayesian average):  ``(successes + prior) / (uses + 2*prior)``
where ``prior = 2``.  This starts at 0.5 and moves toward the true success
rate as more data is collected.

Persistence:
    ``memory_store/skill_graph.json.gz`` via ``compressed_storage``.
"""

import threading
import time
from typing import Any

from memory.compressed_storage import save_json, load_json


# ── Defaults ─────────────────────────────────────────────────────

_PRIOR = 2          # Bayesian smoothing prior
_XP_SUCCESS = 10    # XP gained on success
_XP_FAILURE = 2     # XP gained on failure (still learned something)
_XP_LEVEL_BASE = 50 # XP per proficiency level

# Action → Skill mapping
_ACTION_SKILL_MAP: dict[str, tuple[str, str]] = {
    # action: (skill_name, category)
    "open_app":        ("app_management",       "system"),
    "open_browser":    ("web_navigation",       "web"),
    "navigate":        ("web_navigation",       "web"),
    "search":          ("web_search",           "web"),
    "click":           ("ui_interaction",       "general"),
    "type":            ("text_input",           "general"),
    "press_key":       ("keyboard_control",     "general"),
    "scroll":          ("ui_interaction",       "general"),
    "hotkey":          ("keyboard_control",     "general"),
    "download":        ("file_download",        "web"),
    "run_command":     ("command_execution",    "system"),
    "draw_plan":       ("drawing",             "creative"),
    "mouse_click_xy":  ("precision_clicking",   "general"),
    "mouse_drag":      ("precision_clicking",   "general"),
    "move_file":       ("file_management",      "system"),
    "create_folder":   ("file_management",      "system"),
    "delete_file":     ("file_management",      "system"),
}

# Skill dependency edges (child → parents)
_DEFAULT_DEPENDENCIES: dict[str, list[str]] = {
    "web_search":       ["web_navigation"],
    "file_download":    ["web_navigation", "file_management"],
    "drawing":          ["app_management", "precision_clicking"],
    "command_execution": ["keyboard_control"],
}


# ── SkillNode ────────────────────────────────────────────────────

class SkillNode:
    """Single skill in the graph."""

    __slots__ = (
        "name", "category", "xp", "uses", "successes", "failures",
        "last_used", "dependencies", "unlocked", "metadata", "reward_history",
    )

    def __init__(
        self,
        name: str,
        category: str = "general",
        *,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.name = name
        self.category = category
        self.xp = 0
        self.uses = 0
        self.successes = 0
        self.failures = 0
        self.last_used: float | None = None
        self.dependencies = dependencies or []
        self.unlocked = len(self.dependencies) == 0
        self.metadata = metadata or {}
        self.reward_history: list[float] = []

    @property
    def proficiency(self) -> float:
        """Bayesian-smoothed success rate in [0, 1]."""
        return (self.successes + _PRIOR) / (self.uses + 2 * _PRIOR)

    @property
    def level(self) -> int:
        """Discrete level derived from XP."""
        return self.xp // _XP_LEVEL_BASE

    @property
    def avg_reward(self) -> float:
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)

    def record(self, success: bool, reward: float = 0.0) -> None:
        self.uses += 1
        if success:
            self.successes += 1
            self.xp += _XP_SUCCESS
        else:
            self.failures += 1
            self.xp += _XP_FAILURE
        self.reward_history.append(float(reward))
        self.reward_history = self.reward_history[-100:]
        self.last_used = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "proficiency": round(self.proficiency, 3),
            "level": self.level,
            "xp": self.xp,
            "uses": self.uses,
            "successes": self.successes,
            "failures": self.failures,
            "last_used": self.last_used,
            "dependencies": self.dependencies,
            "unlocked": self.unlocked,
            "metadata": self.metadata,
            "avg_reward": round(self.avg_reward, 3),
            "reward_history": [round(r, 3) for r in self.reward_history[-20:]],
        }

    def __repr__(self) -> str:
        return (f"SkillNode({self.name!r}, prof={self.proficiency:.0%}, "
                f"lv{self.level}, uses={self.uses})")


# ── SkillGraph ───────────────────────────────────────────────────

class SkillGraph:
    """Directed graph of the agent's skills with persistence.

    Parameters
    ----------
    store_dir : str
        Directory for persistence (default ``"memory_store"``).
    """

    _FILE = "skill_graph.json"

    def __init__(self, store_dir: str = "memory_store"):
        self._lock = threading.Lock()
        self._store_path = f"{store_dir}/{self._FILE}"
        self._nodes: dict[str, SkillNode] = {}
        self._dirty = False

        self._load()
        self._ensure_defaults()

    # ── Public API ───────────────────────────────────────────────

    def record_action(self, action: str, success: bool, goal: str = "",
                      reward: float = 0.0) -> SkillNode | None:
        """Record an action execution, updating the corresponding skill.

        Returns the updated SkillNode, or None if the action has no
        mapped skill.
        """
        mapping = _ACTION_SKILL_MAP.get(action)
        if not mapping:
            return None

        skill_name, category = mapping
        with self._lock:
            node = self._get_or_create(skill_name, category)
            node.record(success, reward=reward)
            self._check_unlocks()
            self._dirty = True

        self._save()
        return node

    def record_goal(self, goal: str, actions: list[dict], success: bool,
                    reward_by_action: list[dict] | None = None) -> dict:
        """Record the outcome of a full goal execution.

        Parameters
        ----------
        goal : str
            The goal text.
        actions : list[dict]
            List of action dicts (each with "action" key and optional "result").
        success : bool
            Whether the overall goal succeeded.

        Returns a summary dict of skills touched.
        """
        reward_map: dict[str, list[float]] = {}
        for ra in (reward_by_action or []):
            reward_map.setdefault(ra.get("action", ""), []).append(float(ra.get("reward", 0.0)))

        touched: dict[str, dict] = {}
        for act_dict in actions:
            act_name = act_dict.get("action", "")
            act_result = act_dict.get("result", "")
            act_success = success or not str(act_result).startswith("FAILED")
            reward_vals = reward_map.get(act_name, [])
            act_reward = reward_vals.pop(0) if reward_vals else 0.0
            node = self.record_action(act_name, act_success, goal, reward=act_reward)
            if node:
                touched[node.name] = node.to_dict()
        return touched

    def get_skill(self, name: str) -> SkillNode | None:
        with self._lock:
            return self._nodes.get(name)

    def all_skills(self) -> list[SkillNode]:
        with self._lock:
            return list(self._nodes.values())

    def skills_by_category(self, category: str) -> list[SkillNode]:
        with self._lock:
            return [n for n in self._nodes.values() if n.category == category]

    def categories(self) -> list[str]:
        with self._lock:
            return sorted({n.category for n in self._nodes.values()})

    def top_skills(self, n: int = 5) -> list[SkillNode]:
        with self._lock:
            return sorted(
                self._nodes.values(),
                key=lambda s: (s.proficiency, s.xp),
                reverse=True,
            )[:n]

    def weakest_skills(self, n: int = 5) -> list[SkillNode]:
        """Skills with lowest proficiency among those actually used."""
        with self._lock:
            used = [s for s in self._nodes.values() if s.uses > 0]
            return sorted(used, key=lambda s: (s.proficiency, -s.uses))[:n]

    def unlocked_skills(self) -> list[SkillNode]:
        with self._lock:
            return [n for n in self._nodes.values() if n.unlocked]

    def locked_skills(self) -> list[SkillNode]:
        with self._lock:
            return [n for n in self._nodes.values() if not n.unlocked]

    def get_dependencies(self, skill_name: str) -> list[SkillNode]:
        """Return the dependency SkillNodes for a skill."""
        node = self._nodes.get(skill_name)
        if not node:
            return []
        with self._lock:
            return [self._nodes[d] for d in node.dependencies if d in self._nodes]

    def get_dependents(self, skill_name: str) -> list[SkillNode]:
        """Return skills that depend on the given skill."""
        with self._lock:
            return [
                n for n in self._nodes.values()
                if skill_name in n.dependencies
            ]

    def add_skill(
        self,
        name: str,
        category: str = "general",
        dependencies: list[str] | None = None,
    ) -> SkillNode:
        """Manually register a new skill."""
        with self._lock:
            node = self._get_or_create(name, category)
            if dependencies:
                for dep in dependencies:
                    if dep not in node.dependencies:
                        node.dependencies.append(dep)
                    self._get_or_create(dep, "general")
            self._check_unlocks()
            self._dirty = True
        self._save()
        return node

    def stats(self) -> dict[str, Any]:
        """Aggregate statistics."""
        with self._lock:
            nodes = list(self._nodes.values())
        total = len(nodes)
        used = [n for n in nodes if n.uses > 0]
        return {
            "total_skills": total,
            "used_skills": len(used),
            "unlocked": sum(1 for n in nodes if n.unlocked),
            "locked": sum(1 for n in nodes if not n.unlocked),
            "total_xp": sum(n.xp for n in nodes),
            "total_uses": sum(n.uses for n in nodes),
            "avg_proficiency": (
                round(sum(n.proficiency for n in used) / len(used), 3)
                if used else 0.0
            ),
            "categories": {
                cat: len([n for n in nodes if n.category == cat])
                for cat in sorted({n.category for n in nodes})
            },
        }

    def get_context_for_llm(self, top_n: int = 5) -> str:
        """Compact text summary for injection into LLM prompts."""
        lines: list[str] = []
        top = self.top_skills(top_n)
        weak = self.weakest_skills(3)
        locked = self.locked_skills()

        if top:
            lines.append("Best skills: " + ", ".join(
                f"{s.name} ({s.proficiency:.0%} lv{s.level}, r={s.avg_reward:+.2f})" for s in top
            ))
        if weak:
            lines.append("Weakest skills: " + ", ".join(
                f"{s.name} ({s.proficiency:.0%})" for s in weak
            ))
        if locked:
            lines.append("Locked skills: " + ", ".join(s.name for s in locked))

        st = self.stats()
        lines.append(
            f"Total XP: {st['total_xp']}  |  "
            f"Skills used: {st['used_skills']}/{st['total_skills']}  |  "
            f"Avg proficiency: {st['avg_proficiency']:.0%}"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "skills": {n.name: n.to_dict() for n in self._nodes.values()},
                "stats": self.stats(),
            }

    # ── Internal ─────────────────────────────────────────────────

    def _get_or_create(self, name: str, category: str) -> SkillNode:
        """Must be called under self._lock."""
        if name not in self._nodes:
            deps = _DEFAULT_DEPENDENCIES.get(name, [])
            self._nodes[name] = SkillNode(
                name, category, dependencies=deps,
            )
        return self._nodes[name]

    def _check_unlocks(self) -> None:
        """Update unlocked flags based on dependency proficiency."""
        for node in self._nodes.values():
            if node.unlocked:
                continue
            if not node.dependencies:
                node.unlocked = True
                continue
            # Unlock when all deps have been used at least once
            node.unlocked = all(
                self._nodes.get(d) and self._nodes[d].uses > 0
                for d in node.dependencies
            )

    def _ensure_defaults(self) -> None:
        """Seed the graph with all known actions/skills."""
        with self._lock:
            for action, (skill, cat) in _ACTION_SKILL_MAP.items():
                self._get_or_create(skill, cat)
            self._check_unlocks()
        self._save()

    def _save(self) -> None:
        data = {}
        with self._lock:
            if not self._dirty:
                return
            data = {
                name: node.to_dict()
                for name, node in self._nodes.items()
            }
            self._dirty = False
        save_json(self._store_path, data)

    def _load(self) -> None:
        raw = load_json(self._store_path, default=None)
        if not raw or not isinstance(raw, dict):
            return
        with self._lock:
            for name, d in raw.items():
                node = SkillNode(
                    name=d.get("name", name),
                    category=d.get("category", "general"),
                    dependencies=d.get("dependencies", []),
                )
                node.xp = d.get("xp", 0)
                node.uses = d.get("uses", 0)
                node.successes = d.get("successes", 0)
                node.failures = d.get("failures", 0)
                node.last_used = d.get("last_used")
                node.unlocked = d.get("unlocked", True)
                node.metadata = d.get("metadata", {})
                self._nodes[name] = node
            print(f"[SkillGraph] Loaded {len(self._nodes)} skills from disk.")
