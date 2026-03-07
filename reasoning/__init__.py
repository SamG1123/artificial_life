"""
Reasoning package — goal interpretation, planning, and action selection.

Modules:
  planner  — Decomposes goals into structured step sequences (Plan/Step).
  policy   — Pluggable action-selection strategies (LLM / ML).
  engine   — Orchestrates planner + policy into a unified ReasoningEngine.

The ReasoningEngine is backward-compatible with the old ReasoningModel
(exposes query_model / query_model_with_vision).
"""

from .planner import (
  Planner,
  Plan,
  Step,
  TaskNode,
  ScheduledTask,
  TaskScheduler,
)
from .policy import BasePolicy, LLMPolicy, MLPolicy
from .engine import ReasoningEngine, Goal

__all__ = [
  "Planner", "Plan", "Step", "TaskNode", "ScheduledTask", "TaskScheduler",
    "BasePolicy", "LLMPolicy", "MLPolicy",
    "ReasoningEngine", "Goal",
]
