"""
World State Manager — the AI's unified model of reality.

Combines all perception inputs, memory context, and emotional state
into a single queryable, serialisable representation that the brain
and planner can consume.

Schema (abbreviated):
  {
    "timestamp": float,
    "screen":    { ...PerceptionData... },
    "camera":    { ...PerceptionData... },
    "audio":     { ...PerceptionData... },
    "system":    { ...PerceptionData... },
    "memory": {
        "recent_events":  [ ... ],
        "long_term":      str,
        "episodes":       str,
    },
    "emotion": {
        "mood":           str,
        "valence":        float,   # -1 (negative) .. +1 (positive)
        "arousal":        float,   #  0 (calm) .. 1 (excited)
    },
    "summary": str,
  }
"""

import copy
import json
import time
import threading
from collections import deque
from typing import Any, Sequence


# ── Emotion helpers ──────────────────────────────────────────────

_MOOD_VECTORS = {
    # mood → (valence, arousal)
    "calm":       ( 0.2,  0.1),
    "curious":    ( 0.4,  0.4),
    "focused":    ( 0.3,  0.6),
    "amused":     ( 0.7,  0.5),
    "satisfied":  ( 0.8,  0.3),
    "concerned":  (-0.3,  0.5),
    "frustrated": (-0.5,  0.7),
    "bored":      (-0.2,  0.1),
    "excited":    ( 0.8,  0.8),
    "neutral":    ( 0.0,  0.2),
}

def _mood_to_vector(mood: str) -> tuple[float, float]:
    return _MOOD_VECTORS.get(mood, (0.0, 0.2))


# ── WorldStateManager ───────────────────────────────────────────

class WorldStateManager:
    """Maintains a continuously-updated, queryable model of the world.

    Thread-safe.  All mutation goes through ``update_*`` methods;
    all reads go through ``query_*`` / ``get_*`` methods.
    """

    HISTORY_SIZE = 50  # rolling history of past snapshots

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict = self._empty_state()
        self._history: deque[dict] = deque(maxlen=self.HISTORY_SIZE)

    # ── Schema ───────────────────────────────────────────────────

    @staticmethod
    def _empty_state() -> dict:
        return {
            "timestamp": 0.0,
            "screen":  {},
            "camera":  {},
            "audio":   {},
            "system":  {},
            "memory": {
                "recent_events": [],
                "long_term": "",
                "episodes": "",
            },
            "emotion": {
                "mood": "calm",
                "valence": 0.2,
                "arousal": 0.1,
            },
            "summary": "",
        }

    # ── Bulk update (from PerceptionManager.capture output) ──────

    def update_perception(self, perception_snapshot: dict) -> None:
        """Merge a full perception snapshot into the world state.

        ``perception_snapshot`` is the dict returned by
        ``PerceptionManager.capture_all()`` or ``.capture()``.
        """
        with self._lock:
            self._archive_current()
            now = time.time()
            for key in ("screen", "camera", "audio", "system"):
                if key in perception_snapshot:
                    self._state[key] = perception_snapshot[key]
            self._state["timestamp"] = now
            self._rebuild_summary()

    # ── Individual channel updates ───────────────────────────────

    def update_screen(self, data: dict) -> None:
        with self._lock:
            self._archive_current()
            self._state["screen"] = data
            self._state["timestamp"] = time.time()
            self._rebuild_summary()

    def update_camera(self, data: dict) -> None:
        with self._lock:
            self._archive_current()
            self._state["camera"] = data
            self._state["timestamp"] = time.time()
            self._rebuild_summary()

    def update_audio(self, data: dict) -> None:
        with self._lock:
            self._archive_current()
            self._state["audio"] = data
            self._state["timestamp"] = time.time()
            self._rebuild_summary()

    def update_system(self, data: dict) -> None:
        with self._lock:
            self._archive_current()
            self._state["system"] = data
            self._state["timestamp"] = time.time()
            self._rebuild_summary()

    def update_memory_context(
        self,
        recent_events: list | None = None,
        long_term: str | None = None,
        episodes: str | None = None,
    ) -> None:
        """Refresh the memory slice inside the world state."""
        with self._lock:
            mem = self._state["memory"]
            if recent_events is not None:
                mem["recent_events"] = recent_events
            if long_term is not None:
                mem["long_term"] = long_term
            if episodes is not None:
                mem["episodes"] = episodes
            self._state["timestamp"] = time.time()

    def update_emotion(self, mood: str, full_state: dict | None = None) -> None:
        """Set the current mood and derive valence/arousal.

        If ``full_state`` is provided (from BehaviorController.get_full_state()),
        the richer emotional data is stored instead of the simple mood vector.
        """
        if full_state:
            mood_data = full_state.get("mood", {})
            with self._lock:
                self._state["emotion"] = {
                    "mood": full_state.get("mood_label", mood),
                    "valence": mood_data.get("valence", 0.0),
                    "arousal": mood_data.get("arousal", 0.0),
                    "focus": mood_data.get("focus", 0.4),
                    "emotions": full_state.get("emotions", {}),
                    "dominant_emotion": full_state.get("dominant_emotion"),
                    "behavior": full_state.get("behavior", {}),
                }
        else:
            v, a = _mood_to_vector(mood)
            with self._lock:
                self._state["emotion"] = {
                    "mood": mood,
                    "valence": v,
                    "arousal": a,
                }

    # ── Queries ──────────────────────────────────────────────────

    def get_full_state(self) -> dict:
        """Return a deep copy of the entire world state."""
        with self._lock:
            return copy.deepcopy(self._state)

    def get_channel(self, channel: str) -> dict:
        """Return data for one perception channel.

        ``channel``: "screen" | "camera" | "audio" | "system" |
                     "memory" | "emotion"
        """
        with self._lock:
            return copy.deepcopy(self._state.get(channel, {}))

    def get_summary(self) -> str:
        """One-line human-readable summary of the current world state."""
        with self._lock:
            return self._state.get("summary", "")

    def get_emotion(self) -> dict:
        with self._lock:
            return dict(self._state["emotion"])

    def get_context_for_llm(self, max_events: int = 10) -> str:
        """Build a compact text block suitable for injecting into an
        LLM prompt.  Combines perception summaries, memory context,
        and emotional state."""
        with self._lock:
            s = self._state

        parts = []

        # Emotion
        emo = s.get("emotion", {})
        parts.append(f"Mood: {emo.get('mood', 'unknown')}")

        # Perception summaries
        for ch in ("screen", "system", "camera"):
            summary = (s.get(ch) or {}).get("summary", "")
            if summary:
                parts.append(f"{ch.title()}: {summary}")

        # Audio
        audio = s.get("audio", {})
        audio_summary = audio.get("summary", "")
        if audio_summary:
            parts.append(f"Audio: {audio_summary}")

        # Memory
        mem = s.get("memory", {})
        lt = mem.get("long_term", "")
        if lt:
            parts.append(f"Long-term memory:\n{lt}")
        episodes = mem.get("episodes", "")
        if episodes:
            parts.append(f"Recent episodes:\n{episodes}")
        events = mem.get("recent_events", [])
        if events:
            recent = events[-max_events:]
            lines = [f"  [{e.get('type','?')}] {e.get('content','')}"
                     for e in recent if isinstance(e, dict)]
            if lines:
                parts.append("Recent events:\n" + "\n".join(lines))

        return "\n\n".join(parts)

    def query(self, path: str) -> Any:
        """Dot-notation query into the world state.

        Examples:
            query("screen.data.active_window.title")
            query("emotion.mood")
            query("system.data.resources.cpu_percent")
        """
        with self._lock:
            obj = self._state
        for key in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                return None
            if obj is None:
                return None
        return copy.deepcopy(obj)

    # ── History ──────────────────────────────────────────────────

    def get_history(self, n: int = 10) -> list[dict]:
        """Return the last *n* archived snapshots (oldest first)."""
        with self._lock:
            items = list(self._history)
        return items[-n:]

    def get_state_age(self) -> float:
        """Seconds since the world state was last updated."""
        with self._lock:
            ts = self._state.get("timestamp", 0)
        return time.time() - ts if ts else float("inf")

    # ── Serialisation ────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialise the current world state to a JSON string."""
        return json.dumps(self.get_full_state(), default=str)

    def to_dict(self) -> dict:
        """Alias for get_full_state()."""
        return self.get_full_state()

    # ── Internals ────────────────────────────────────────────────

    def _archive_current(self) -> None:
        """Push a lightweight snapshot onto the history ring buffer.
        Caller must hold self._lock."""
        if self._state.get("timestamp"):
            slim = {
                "timestamp": self._state["timestamp"],
                "summary": self._state.get("summary", ""),
                "emotion": dict(self._state.get("emotion", {})),
            }
            self._history.append(slim)

    def _rebuild_summary(self) -> None:
        """Recompute the top-level summary string.
        Caller must hold self._lock."""
        parts = []
        for ch in ("screen", "camera", "audio", "system"):
            s = (self._state.get(ch) or {}).get("summary", "")
            if s:
                parts.append(s)
        emo = self._state.get("emotion", {})
        if emo.get("mood"):
            parts.append(f"Mood: {emo['mood']}")
        self._state["summary"] = " | ".join(parts)
