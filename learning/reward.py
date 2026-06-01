"""Reward Engine — multi-dimensional reward signals for adaptation.

Computes rewards from goal outcomes and action history using:
- task completion
- efficiency
- user satisfaction (implicit)
- side effects
"""

from __future__ import annotations

from typing import Any


class RewardEngine:
    """Computes and records reward signals for learning subsystems."""

    def __init__(self, skill_graph=None):
        self.skill_graph = skill_graph

    def evaluate_goal(
        self,
        *,
        goal: str,
        action_history: list[dict],
        success: bool,
        max_steps: int = 30,
    ) -> dict[str, Any]:
        """Compute reward dimensions and per-action rewards for one goal run."""
        steps = max(1, len(action_history))

        # 1) Completion: binary success signal.
        task_completion = 1.0 if success else -1.0

        # 2) Efficiency: fewer steps for successful goals is better.
        if success:
            efficiency = max(-0.2, 1.0 - (steps / max(1, max_steps)))
        else:
            efficiency = -0.4

        # 3) User satisfaction proxy: penalize blocked/failed actions.
        failed_or_blocked = 0
        for h in action_history:
            res = str(h.get("result", "")).upper()
            if res.startswith("FAILED") or res.startswith("BLOCKED"):
                failed_or_blocked += 1
        user_satisfaction = 0.6 if success else -0.6
        user_satisfaction -= min(0.5, failed_or_blocked * 0.08)

        # 4) Side effects proxy: heavy penalty for blocked/destructive attempts.
        side_effects = -min(0.7, failed_or_blocked * 0.1)

        total = (
            0.45 * task_completion
            + 0.20 * efficiency
            + 0.25 * user_satisfaction
            + 0.10 * side_effects
        )

        # Per-action reward assignment for skill-level updates.
        per_action: list[dict[str, Any]] = []
        for h in action_history:
            action = (h.get("action") or {}).get("action", "unknown")
            res = str(h.get("result", ""))
            r = 0.08
            if res.upper().startswith("FAILED"):
                r = -0.25
            elif res.upper().startswith("BLOCKED"):
                r = -0.35
            elif "SUCCESS" in res.upper() or success:
                r = 0.12
            per_action.append({"action": action, "reward": round(r, 3)})

        return {
            "goal": goal,
            "success": success,
            "steps": steps,
            "dimensions": {
                "task_completion": round(task_completion, 3),
                "efficiency": round(efficiency, 3),
                "user_satisfaction": round(user_satisfaction, 3),
                "side_effects": round(side_effects, 3),
            },
            "total": round(total, 3),
            "per_action": per_action,
        }
