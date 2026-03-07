"""
Episodic Memory — experience log of past tasks and outcomes.

Stores:
  * Past goals attempted
  * Actions taken and step counts
  * Success / failure outcomes
  * Compressed summaries of older episodes

Characteristics:
  * Persisted to disk (JSON)
  * Bounded capacity with automatic compression of old episodes
  * Thread-safe
  * Queryable by recency, success, or keyword
"""

import json
import os
import time
from threading import Lock

from .compressed_storage import save_json, load_json, migrate_if_needed

EPISODIC_CAPACITY = 200       # max detailed episodes
COMPRESSED_CAPACITY = 500     # max compressed (summary-only) episodes


class EpisodicMemory:
    """Logs and retrieves past goal-execution experiences."""

    def __init__(self, store_dir: str):
        self._episodes_file = os.path.join(store_dir, "episodes.json")
        self._compressed_file = os.path.join(store_dir, "episodes_compressed.json")
        self._lock = Lock()

        # Auto-migrate old plain JSON to compressed
        migrate_if_needed(self._episodes_file)
        migrate_if_needed(self._compressed_file)

        self._episodes: list[dict] = load_json(self._episodes_file, [])
        self._compressed: list[dict] = load_json(self._compressed_file, [])

        # Enforce capacity on load
        if len(self._episodes) > EPISODIC_CAPACITY:
            self._compress_overflow()

    # ── Write ────────────────────────────────────────────────────

    def log_episode(
        self,
        goal: str,
        outcome: str,
        steps: int,
        success: bool,
        actions: list[str] | None = None,
    ) -> None:
        """Record a completed goal execution."""
        episode = {
            "goal": goal,
            "outcome": outcome,
            "steps": steps,
            "success": success,
            "actions": actions or [],
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M"),
        }
        with self._lock:
            self._episodes.append(episode)
            if len(self._episodes) > EPISODIC_CAPACITY:
                self._compress_overflow()

        self._save(self._episodes_file, self._episodes)

    # ── Read ───────────────────────────────────────────────────────────

    def get_recent(self, n: int = 5) -> list[dict]:
        """Return the *n* most recent detailed episodes."""
        with self._lock:
            return list(self._episodes[-n:])

    def get_summary(self, n: int = 5) -> str:
        """Text summary of recent episodes for LLM context."""
        episodes = self.get_recent(n)
        if not episodes:
            return "(no past experiences)"
        lines = []
        for ep in episodes:
            status = "✓" if ep["success"] else "✗"
            lines.append(
                f"[{ep['time_str']}] {status} \"{ep['goal']}\" "
                f"→ {ep['outcome']} ({ep['steps']} steps)"
            )
        return "\n".join(lines)

    def search(self, keyword: str, limit: int = 10) -> list[dict]:
        """Find episodes whose goal or outcome mentions *keyword*."""
        kw = keyword.lower()
        with self._lock:
            all_eps = self._episodes + self._compressed
        results = [
            ep for ep in all_eps
            if kw in ep.get("goal", "").lower()
            or kw in ep.get("outcome", "").lower()
        ]
        return results[-limit:]

    def get_success_rate(self, last_n: int = 20) -> float:
        """Success rate over the last *n* episodes."""
        recent = self.get_recent(last_n)
        if not recent:
            return 0.0
        return sum(1 for ep in recent if ep["success"]) / len(recent)

    def get_compressed(self, n: int = 20) -> list[dict]:
        """Return older compressed episodes."""
        with self._lock:
            return list(self._compressed[-n:])

    # ── Compression ──────────────────────────────────────────────

    def _compress_overflow(self) -> None:
        """Move oldest episodes into compressed storage.
        Caller must hold self._lock."""
        # Keep the most recent EPISODIC_CAPACITY, compress the rest
        overflow = self._episodes[:-EPISODIC_CAPACITY]
        self._episodes = self._episodes[-EPISODIC_CAPACITY:]

        for ep in overflow:
            self._compressed.append({
                "goal": ep["goal"],
                "success": ep["success"],
                "steps": ep["steps"],
                "time_str": ep.get("time_str", ""),
                "timestamp": ep.get("timestamp", 0),
            })

        # Trim compressed store
        if len(self._compressed) > COMPRESSED_CAPACITY:
            self._compressed = self._compressed[-COMPRESSED_CAPACITY:]

        self._save(self._compressed_file, self._compressed)

    # ── Persistence ──────────────────────────────────────────────

    @staticmethod
    def _save(path: str, data: list) -> None:
        try:
            save_json(path, data)
        except Exception as e:
            print(f"[EpisodicMemory] Failed to save {path}: {e}")

    @property
    def total_episodes(self) -> int:
        with self._lock:
            return len(self._episodes) + len(self._compressed)
