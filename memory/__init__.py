"""
Memory package — multi-layered memory architecture for the autonomous AI.

Sub-modules:
  short_term  — rolling buffer of recent events / observations / actions
  episodic    — past tasks, actions taken, outcomes
  semantic    — learned facts, preferences, persistent knowledge
  vector      — embedding-based storage for faces, objects, concepts
  manager     — MemoryManager that coordinates all sub-systems

The top-level MemorySystem class is the main public interface
(backwards-compatible with the previous monolithic memory.py).
"""

from .short_term import ShortTermMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .vector import VectorMemory
from .manager import MemoryManager, MemorySystem
from .compressor import MemoryCompressor

__all__ = [
    "ShortTermMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "VectorMemory",
    "MemoryManager",
    "MemorySystem",
    "MemoryCompressor",
]
