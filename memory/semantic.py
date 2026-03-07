"""
Semantic Memory — persistent knowledge store.

Stores:
  * Learned facts about the world
  * User preferences and profile
  * Self-knowledge (AI's own notes about itself)
  * Personality traits

Characteristics:
  * Persisted to disk (JSON)
  * Categorised storage (facts, preferences, self-notes, user profile)
  * Deduplication on insert
  * Bounded capacity per category
  * Thread-safe
"""

import json
import os
import time
from threading import Lock

from .compressed_storage import save_json, load_json, migrate_if_needed


FACTS_CAPACITY = 200
PREFERENCES_CAPACITY = 50
SELF_NOTES_CAPACITY = 50


class SemanticMemory:
    """Stores and retrieves persistent factual knowledge."""

    def __init__(self, store_dir: str):
        self._file = os.path.join(store_dir, "long_term.json")
        self._lock = Lock()

        migrate_if_needed(self._file)
        self._data: dict = load_json(self._file, default={
            "user_name": None,
            "user_preferences": [],
            "learned_facts": [],
            "self_notes": [],
            "personality_traits": [
                "curious", "helpful", "calm", "slightly playful"
            ],
        })

    # ── Facts ────────────────────────────────────────────────────

    def learn_fact(self, fact: str) -> None:
        """Store a new fact (deduplicated)."""
        with self._lock:
            facts = self._data["learned_facts"]
            if fact not in facts:
                facts.append(fact)
                if len(facts) > FACTS_CAPACITY:
                    self._data["learned_facts"] = facts[-FACTS_CAPACITY:]
        self._save()

    def get_facts(self, n: int | None = None) -> list[str]:
        with self._lock:
            facts = list(self._data["learned_facts"])
        if n is not None:
            return facts[-n:]
        return facts

    def search_facts(self, keyword: str) -> list[str]:
        """Return facts containing *keyword*."""
        kw = keyword.lower()
        with self._lock:
            return [f for f in self._data["learned_facts"] if kw in f.lower()]

    # ── User profile ─────────────────────────────────────────────

    def set_user_name(self, name: str) -> None:
        with self._lock:
            self._data["user_name"] = name
        self._save()

    def get_user_name(self) -> str | None:
        with self._lock:
            return self._data.get("user_name")

    def learn_preference(self, preference: str) -> None:
        with self._lock:
            prefs = self._data["user_preferences"]
            if preference not in prefs:
                prefs.append(preference)
                if len(prefs) > PREFERENCES_CAPACITY:
                    self._data["user_preferences"] = prefs[-PREFERENCES_CAPACITY:]
        self._save()

    def get_preferences(self) -> list[str]:
        with self._lock:
            return list(self._data["user_preferences"])

    # ── Self-knowledge ───────────────────────────────────────────

    def add_self_note(self, note: str) -> None:
        """AI's own observation about itself or its behaviour."""
        with self._lock:
            notes = self._data["self_notes"]
            notes.append(note)
            if len(notes) > SELF_NOTES_CAPACITY:
                self._data["self_notes"] = notes[-SELF_NOTES_CAPACITY:]
        self._save()

    def get_self_notes(self) -> list[str]:
        with self._lock:
            return list(self._data["self_notes"])

    def get_personality(self) -> list[str]:
        with self._lock:
            return list(self._data.get("personality_traits", []))

    # ── Summary for LLM ─────────────────────────────────────────

    def get_summary(self) -> str:
        """Text summary of long-term memory for LLM context."""
        with self._lock:
            lt = self._data
        parts = []
        if lt.get("user_name"):
            parts.append(f"User's name: {lt['user_name']}")
        if lt.get("user_preferences"):
            parts.append(
                f"User preferences: {', '.join(lt['user_preferences'][-10:])}"
            )
        if lt.get("learned_facts"):
            parts.append(
                f"Known facts: {'; '.join(lt['learned_facts'][-10:])}"
            )
        if lt.get("self_notes"):
            parts.append(
                f"Self-notes: {'; '.join(lt['self_notes'][-5:])}"
            )
        return "\n".join(parts) if parts else "(no long-term memories yet)"

    # ── Raw access (for WorldState sync) ─────────────────────────

    def get_all(self) -> dict:
        """Return a copy of the entire semantic store."""
        with self._lock:
            return dict(self._data)

    # ── Persistence ──────────────────────────────────────────────

    def _save(self) -> None:
        with self._lock:
            data = dict(self._data)
        try:
            save_json(self._file, data)
        except Exception as e:
            print(f"[SemanticMemory] Failed to save: {e}")
