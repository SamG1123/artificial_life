"""
Memory Manager — coordinates all memory sub-systems and provides
the unified public API (``MemorySystem``).

Architecture:
  ┌─────────────────────────────────────────────────┐
  │              MemoryManager                       │
  │  ┌───────────┬───────────┬──────────┬─────────┐ │
  │  │ ShortTerm │ Episodic  │ Semantic │ Vector  │ │
  │  │  (RAM)    │ (disk)    │ (disk)   │ (disk)  │ │
  │  └───────────┴───────────┴──────────┴─────────┘ │
  └─────────────────────────────────────────────────┘

``MemorySystem`` is a thin subclass that preserves the exact same
method names used by brain.py so all 26 call-sites keep working
with zero changes.
"""

import os
from threading import Lock

from .short_term import ShortTermMemory
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .vector import VectorMemory

MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "memory_store",
)


class MemoryManager:
    """Low-level coordinator — exposes each sub-system directly."""

    def __init__(self, store_dir: str | None = None):
        self._store_dir = store_dir or MEMORY_DIR
        os.makedirs(self._store_dir, exist_ok=True)

        self.short_term = ShortTermMemory()
        self.episodic = EpisodicMemory(self._store_dir)
        self.semantic = SemanticMemory(self._store_dir)
        self.vector = VectorMemory(self._store_dir)

    def cross_search(self, keyword: str, limit: int = 10) -> dict:
        """Search across all memory types for *keyword*.

        Returns:
            {
              "facts": [str, ...],
              "episodes": [dict, ...],
              "events": [dict, ...],
            }
        """
        return {
            "facts": self.semantic.search_facts(keyword),
            "episodes": self.episodic.search(keyword, limit=limit),
            "events": [
                e for e in self.short_term.get_recent(50)
                if keyword.lower() in e.get("content", "").lower()
            ][-limit:],
        }

    def stats(self) -> dict:
        """Return a summary of all memory sub-system sizes."""
        return {
            "short_term_events": self.short_term.size,
            "episodic_total": self.episodic.total_episodes,
            "semantic_facts": len(self.semantic.get_facts()),
            "vector_items": self.vector.size,
            "vector_categories": self.vector.category_counts(),
        }


class MemorySystem(MemoryManager):
    """Backwards-compatible public interface.

    Every method that brain.py (and world_state.py) already calls is
    preserved with the exact same signature.  Internally each call
    delegates to the appropriate sub-system.
    """

    # ── Short-term (delegates to self.short_term) ────────────────

    def add_event(self, event_type: str, content: str) -> None:
        self.short_term.add_event(event_type, content)

    def get_recent_events(self, n: int = 15, event_type: str = None) -> list:
        return self.short_term.get_recent(n, event_type)

    def get_context_summary(self, max_events: int = 20) -> str:
        return self.short_term.get_context_summary(max_events)

    # ── Episodic (delegates to self.episodic) ────────────────────

    def log_episode(self, goal: str, outcome: str, steps: int,
                    success: bool) -> None:
        self.episodic.log_episode(goal, outcome, steps, success)

    def get_recent_episodes(self, n: int = 5) -> list:
        return self.episodic.get_recent(n)

    def get_episodes_summary(self, n: int = 5) -> str:
        return self.episodic.get_summary(n)

    # ── Semantic / long-term (delegates to self.semantic) ────────

    def learn_fact(self, fact: str) -> None:
        self.semantic.learn_fact(fact)

    def learn_preference(self, preference: str) -> None:
        self.semantic.learn_preference(preference)

    def add_self_note(self, note: str) -> None:
        self.semantic.add_self_note(note)

    def set_user_name(self, name: str) -> None:
        self.semantic.set_user_name(name)

    def get_long_term_summary(self) -> str:
        return self.semantic.get_summary()
