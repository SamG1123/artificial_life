"""
ReasoningEngine — the cognitive core that interprets goals, analyses
the environment, and decides actions.

Architecture::

    ┌───────────────────────────────────────────────────┐
    │               ReasoningEngine                      │
    │                                                    │
    │  goal + world state                                │
    │       ↓                                            │
    │  ┌──────────┐   plan   ┌────────────┐  action     │
    │  │ Planner  │ ──────→ │   Policy   │ ──────→ out │
    │  └──────────┘          │ (LLM / ML) │             │
    │       ↑ replan         └────────────┘             │
    │       └──────── feedback ←───── executor          │
    └───────────────────────────────────────────────────┘

The engine sits between the Brain (decides *what* to do) and the
Executor (carries out actions).  It manages:

  1. **Goal interpretation** — understands what the agent should achieve.
  2. **Planning** — breaks goals into step sequences via the Planner.
  3. **Action selection** — for each step, picks a concrete action via
     the PolicyModel.
  4. **Self-monitoring** — detects stuck loops, failures, and triggers
     replanning when needed.

Backward compatibility
~~~~~~~~~~~~~~~~~~~~~~
The engine exposes ``query_model()`` and ``query_model_with_vision()``
with the *exact same signatures* as the old ``ReasoningModel`` class,
so the executor can use it as a drop-in replacement.
"""

from __future__ import annotations

import time
import threading
from typing import Any

from .planner import Planner, Plan, Step
from .policy import BasePolicy, LLMPolicy


# ── Goal wrapper ─────────────────────────────────────────────────

class Goal:
    """Parsed representation of a goal with inferred attributes."""

    __slots__ = ("raw", "domain", "needs_browser", "needs_drawing",
                 "needs_system", "priority", "created_at")

    _WEB_KEYWORDS = frozenset({
        "find", "search", "look up", "lookup", "google", "browse",
        "recipe", "weather", "news", "website", "download", "pdf",
        "how to", "what is", "who is", "where is", "wiki",
        "buy", "price", "shop", "order", "book", "watch",
        "youtube", "video", "article", "blog", "reddit",
        "stackoverflow", "github", "online",
    })
    _DRAW_KEYWORDS = frozenset({
        "draw", "paint", "sketch", "doodle", "illustrate",
    })
    _SYS_KEYWORDS = frozenset({
        "shutdown", "restart", "folder", "directory", "delete file",
        "move file", "rename", "volume", "brightness", "wifi",
    })

    def __init__(self, raw: str, priority: int = 1):
        self.raw = raw
        self.created_at = time.time()
        self.priority = priority

        g = raw.lower()
        self.needs_browser = any(kw in g for kw in self._WEB_KEYWORDS)
        self.needs_drawing = any(kw in g for kw in self._DRAW_KEYWORDS)
        self.needs_system = any(kw in g for kw in self._SYS_KEYWORDS)
        self.domain = (
            "web" if self.needs_browser else
            "drawing" if self.needs_drawing else
            "system" if self.needs_system else
            "general"
        )

    def __repr__(self) -> str:
        return f"Goal({self.raw!r}, domain={self.domain})"


# ── ReasoningEngine ──────────────────────────────────────────────

class ReasoningEngine:
    """Orchestrates planning + policy for goal execution.

    Parameters
    ----------
    policy : BasePolicy | None
        Action-selection policy.  Defaults to ``LLMPolicy()``.
    planner : Planner | None
        Goal decomposer.  Defaults to ``Planner()``.
    behavior : BehaviorController | None
        Emotion/personality system for plan-aware emotional context.
    """

    MAX_REPLAN_ATTEMPTS = 2
    CONSECUTIVE_FAIL_REPLAN = 3   # replan after N consecutive failures

    def __init__(self, policy: BasePolicy | None = None,
                 planner: Planner | None = None,
                 behavior: Any = None):
        self.policy: BasePolicy = policy or LLMPolicy()
        self.planner: Planner = planner or Planner()
        self.behavior = behavior

        self._lock = threading.Lock()
        self._current_goal: Goal | None = None
        self._current_plan: Plan | None = None
        self._replan_count = 0
        self._consecutive_failures = 0

        # Expose groq_client for executor's draw_plan
        # (needs raw client for image generation prompt)
        if hasattr(self.policy, "groq_client"):
            self.groq_client = self.policy.groq_client

    # ── Goal lifecycle ───────────────────────────────────────────

    def set_goal(self, goal_text: str, world_context: str = "") -> Plan:
        """Interpret a goal and produce an initial plan.

        Called by the executor at the start of ``execute_goal()``.
        """
        with self._lock:
            self._current_goal = Goal(goal_text)
            self._replan_count = 0
            self._consecutive_failures = 0

            plan = self.planner.create_plan(goal_text, world_context)
            plan.status = "in_progress"
            self._current_plan = plan

            warnings = self.planner.validate(plan)
            if warnings:
                print(f"[ReasoningEngine] Plan warnings: {warnings}")

            return plan

    @property
    def current_plan(self) -> Plan | None:
        return self._current_plan

    @property
    def current_goal(self) -> Goal | None:
        return self._current_goal

    # ── Action selection (main interface for executor) ───────────

    def next_action(
        self,
        goal: str,
        elements: list[dict],
        history_context: str,
        *,
        screenshot_b64: str | None = None,
        screen_resolution: tuple[int, int] = (1920, 1080),
    ) -> dict:
        """Select the next action given the current environment.

        This is the primary method the executor calls each step.
        It enriches the policy query with plan context + emotion.
        """
        # Build plan context for the LLM
        plan_ctx = ""
        if self._current_plan:
            plan_ctx = self._current_plan.summary()

        # Build emotional context
        emo_ctx = ""
        if self.behavior:
            try:
                emo_ctx = self.behavior.get_emotional_context()
            except Exception:
                pass

        action = self.policy.select_action(
            goal=goal,
            elements=elements,
            history_context=history_context,
            screenshot_b64=screenshot_b64,
            screen_resolution=screen_resolution,
            plan_context=plan_ctx,
            emotional_context=emo_ctx,
        )

        # Advance plan tracking
        self._track_plan_step(action)

        return action

    # ── Backward-compatible API ──────────────────────────────────
    # So executor.py can use engine as a drop-in for ReasoningModel.

    def query_model(self, goal: str, elements: list,
                    user_context: str = "") -> dict:
        """Drop-in replacement for ``ReasoningModel.query_model()``."""
        return self.next_action(
            goal=goal,
            elements=elements,
            history_context=user_context,
        )

    def query_model_with_vision(self, goal: str, elements: list,
                                screenshot_b64: str,
                                user_context: str = "",
                                screen_resolution: tuple = (1920, 1080)) -> dict:
        """Drop-in replacement for ``ReasoningModel.query_model_with_vision()``."""
        return self.next_action(
            goal=goal,
            elements=elements,
            history_context=user_context,
            screenshot_b64=screenshot_b64,
            screen_resolution=screen_resolution,
        )

    # ── Feedback / monitoring ────────────────────────────────────

    def report_step_result(self, action: dict, success: bool,
                           result: str = "",
                           world_context: str = "") -> None:
        """Called after each executed step.  Updates plan tracking
        and trigger replanning if needed."""
        if success:
            self._consecutive_failures = 0
            if self._current_plan:
                cur = self._current_plan.current_step
                if cur:
                    cur.complete(result)
        else:
            self._consecutive_failures += 1
            if self._current_plan:
                cur = self._current_plan.current_step
                if cur:
                    cur.fail(result)

            # Auto-replan after consecutive failures
            if (self._consecutive_failures >= self.CONSECUTIVE_FAIL_REPLAN
                    and self._replan_count < self.MAX_REPLAN_ATTEMPTS
                    and self._current_plan):
                self._do_replan(
                    f"Last {self._consecutive_failures} actions failed",
                    world_context,
                )

    def request_replan(self, reason: str,
                       world_context: str = "") -> Plan | None:
        """Externally triggered replan (e.g. by the brain)."""
        if self._current_plan and self._replan_count < self.MAX_REPLAN_ATTEMPTS:
            return self._do_replan(reason, world_context)
        return None

    # ── Analysis helpers ─────────────────────────────────────────

    def analyze_goal(self, goal_text: str) -> dict:
        """Quick goal analysis without creating a full plan.

        Returns domain classification, browser/drawing/system needs.
        Used by the brain for high-level decision-making.
        """
        g = Goal(goal_text)
        return {
            "raw": g.raw,
            "domain": g.domain,
            "needs_browser": g.needs_browser,
            "needs_drawing": g.needs_drawing,
            "needs_system": g.needs_system,
        }

    def get_state(self) -> dict:
        """Serialisable snapshot of engine state."""
        return {
            "goal": self._current_goal.raw if self._current_goal else None,
            "goal_domain": self._current_goal.domain if self._current_goal else None,
            "plan": self._current_plan.to_dict() if self._current_plan else None,
            "replan_count": self._replan_count,
            "consecutive_failures": self._consecutive_failures,
        }

    # ── Internal ─────────────────────────────────────────────────

    def _track_plan_step(self, action: dict) -> None:
        """Keep plan steps in sync with actual actions."""
        if not self._current_plan:
            return
        cur = self._current_plan.current_step
        if cur and cur.status == "pending":
            cur.start()

    def _do_replan(self, reason: str, world_context: str) -> Plan:
        self._replan_count += 1
        self._consecutive_failures = 0
        print(f"[ReasoningEngine] Replanning ({self._replan_count}/"
              f"{self.MAX_REPLAN_ATTEMPTS}): {reason}")

        new_plan = self.planner.replan(
            self._current_plan, reason, world_context
        )
        new_plan.status = "in_progress"
        self._current_plan = new_plan
        return new_plan
