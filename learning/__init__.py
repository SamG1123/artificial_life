"""learning — the learning subsystem for the AI agent.

Modules:
    ExperienceLogger  — logs (world_state, action, result) tuples to JSONL
    DatasetBuilder    — transforms logged experiences into training datasets
    ModelTrainer      — trains models (action classifier, reasoning stub)
"""

from .experience import ExperienceLogger
from .dataset import DatasetBuilder
from .trainer import ModelTrainer
from .self_improver import SelfImprover

__all__ = [
    "ExperienceLogger",
    "DatasetBuilder",
    "ModelTrainer",
    "SelfImprover",
]
