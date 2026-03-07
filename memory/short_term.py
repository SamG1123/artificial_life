"""
Short-Term Memory — working memory for the AI's immediate context.

Stores:
  * Recent observations (what the AI sees/hears)
  * Recent actions (what the AI did)
  * Current goals and thoughts
  * System events

Characteristics:
  * Fixed-capacity rolling buffer (FIFO eviction)
  * Fast append and query
  * No persistence — lost on restart (ephemeral by design)
  * Thread-safe
"""

import time
from collections import deque
from threading import Lock
from typing import Any


DEFAULT_CAPACITY = 50

# Prefixes used when rendering events for LLM context
_EVENT_PREFIXES = {
    "observation": "👁 SAW",
    "thought":     "💭 THOUGHT",
    "user_speech": "🗣 USER",
    "action":      "⚡ DID",
    "action_result": "📋 RESULT",
    "system":      "⚙ SYS",
}


class ShortTermMemory:
    """Rolling buffer of recent events / observations / actions."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        self._capacity = capacity
        self._buffer: deque[dict] = deque(maxlen=capacity)
        self._lock = Lock()
        self._current_goals: list[str] = []

    # ── Write ────────────────────────────────────────────────────

    def add_event(self, event_type: str, content: str) -> None:
        """Append a timestamped event to working memory.

        event_type: 'observation', 'thought', 'user_speech',
                    'action', 'action_result', 'system'
        """
        entry = {
            "type": event_type,
            "content": content,
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S"),
        }
        with self._lock:
            self._buffer.append(entry)

    def set_current_goal(self, goal: str) -> None:
        """Track the currently active goal."""
        with self._lock:
            if goal and goal not in self._current_goals:
                self._current_goals.append(goal)

    def clear_current_goal(self, goal: str) -> None:
        with self._lock:
            self._current_goals = [
                g for g in self._current_goals if g != goal
            ]

    # ── Read ─────────────────────────────────────────────────────

    def get_recent(self, n: int = 15, event_type: str | None = None) -> list[dict]:
        """Return the most recent *n* events, optionally filtered."""
        with self._lock:
            items = list(self._buffer)
        if event_type:
            items = [e for e in items if e["type"] == event_type]
        return items[-n:]

    def get_current_goals(self) -> list[str]:
        with self._lock:
            return list(self._current_goals)

    def get_context_summary(self, max_events: int = 20) -> str:
        """Build a text summary of recent working memory for LLM context."""
        events = self.get_recent(max_events)
        if not events:
            return "(no recent events)"
        lines = []
        for e in events:
            prefix = _EVENT_PREFIXES.get(e["type"], e["type"].upper())
            lines.append(f"[{e['time_str']}] {prefix}: {e['content']}")
        return "\n".join(lines)

    # ── Maintenance ──────────────────────────────────────────────

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._current_goals.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)
