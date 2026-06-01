"""learning — the learning subsystem for the AI agent.

Modules:
    ExperienceLogger  — logs (world_state, action, result) tuples to JSONL
    DatasetBuilder    — transforms logged experiences into training datasets
    ModelTrainer      — trains models (action classifier, reasoning stub)
    SelfImprover      — analyses performance and generates new strategies
    SkillGraph        — tracks skill proficiency, XP, and dependency graph
"""

from .experience import ExperienceLogger
from .dataset import DatasetBuilder
from .trainer import ModelTrainer
from .self_improver import SelfImprover
from .skill_graph import SkillGraph, SkillNode
from .nightly_trainer import NightlyTrainer
from .reward import RewardEngine

__all__ = [
    "ExperienceLogger",
    "DatasetBuilder",
    "ModelTrainer",
    "SelfImprover",
    "SkillGraph",
    "SkillNode",
    "NightlyTrainer",
    "RewardEngine",
]
