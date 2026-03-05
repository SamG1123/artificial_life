"""
EmotionEngine — short-term emotional reactions to events.

Emotions are instantaneous spikes that:
  1. fire in response to categorised events,
  2. push MoodEngine (persistent emotional drift),
  3. decay rapidly back to zero each tick.

Each active emotion has an *intensity* [0 .. 1] that fades by a fixed
decay factor every tick.  Personality traits modulate the initial spike
so the same event produces different emotions depending on personality.

Supported emotions (extensible):
    satisfaction, frustration, surprise, curiosity, amusement,
    concern, boredom, pride, disappointment, gratitude
"""

import time
import threading
from typing import Any

from .personality import PersonalityModel
from .mood import MoodEngine


# ── Event → emotion mapping ──────────────────────────────────────

# Each rule: (emotion_name, base_intensity, mood_push)
#   mood_push = (d_valence, d_arousal, d_focus)
_EVENT_RULES: dict[str, list[tuple[str, float, tuple[float, float, float]]]] = {
    # Positive events
    "goal_success": [
        ("satisfaction", 0.7,  ( 0.20,  0.05, -0.05)),
        ("pride",        0.4,  ( 0.10,  0.10,  0.00)),
    ],
    "user_praise": [
        ("satisfaction", 0.5,  ( 0.15,  0.05,  0.00)),
        ("gratitude",    0.4,  ( 0.10,  0.00,  0.00)),
    ],
    "user_greeting": [
        ("amusement",    0.3,  ( 0.10,  0.05,  0.00)),
    ],
    "new_discovery": [
        ("curiosity",    0.6,  ( 0.10,  0.15,  0.10)),
        ("surprise",     0.4,  ( 0.05,  0.20,  0.05)),
    ],
    "interesting_observation": [
        ("curiosity",    0.4,  ( 0.05,  0.10,  0.10)),
    ],

    # Negative events
    "goal_failure": [
        ("frustration",    0.6,  (-0.15,  0.15, -0.10)),
        ("disappointment", 0.4,  (-0.10,  0.00, -0.05)),
    ],
    "repeated_failure": [
        ("frustration",    0.8,  (-0.25,  0.20, -0.15)),
    ],
    "user_frustration": [
        ("concern",       0.5,  (-0.10,  0.10,  0.05)),
    ],
    "error": [
        ("surprise",      0.5,  (-0.05,  0.20,  0.00)),
        ("concern",       0.3,  (-0.10,  0.10,  0.00)),
    ],

    # Neutral / ambient
    "idle_long": [
        ("boredom",       0.4,  (-0.05, -0.10, -0.10)),
    ],
    "user_return": [
        ("amusement",     0.3,  ( 0.10,  0.10,  0.00)),
    ],
}


class EmotionEngine:
    """Short-lived emotional reactions that feed into the mood layer."""

    DECAY_FACTOR = 0.70  # emotions lose 30 % intensity each tick

    def __init__(self, personality: PersonalityModel, mood: MoodEngine):
        self.personality = personality
        self.mood = mood
        self._lock = threading.Lock()

        # Active emotions: {name: intensity}
        self._active: dict[str, float] = {}
        self._history: list[dict] = []   # last N reactions for debugging
        self._history_cap = 50

    # ── React to an event ────────────────────────────────────────

    def react(self, event_type: str, details: str = "") -> list[str]:
        """Process a categorised event.  Returns list of triggered emotion names.

        ``event_type`` should be one of the keys in ``_EVENT_RULES``.
        Unknown event types are silently ignored (no crash).
        """
        rules = _EVENT_RULES.get(event_type)
        if not rules:
            return []

        triggered: list[str] = []

        for emotion_name, base_intensity, mood_push in rules:
            intensity = self._apply_personality(emotion_name, base_intensity)

            with self._lock:
                # Stack: take the max of current and new intensity
                current = self._active.get(emotion_name, 0.0)
                self._active[emotion_name] = max(current, intensity)

            # Push mood
            self.mood.push(*mood_push)
            triggered.append(emotion_name)

        # Log to history
        with self._lock:
            self._history.append({
                "time": time.strftime("%H:%M:%S"),
                "event": event_type,
                "details": details[:120],
                "triggered": triggered,
            })
            if len(self._history) > self._history_cap:
                self._history = self._history[-self._history_cap:]

        return triggered

    # ── Tick / decay ─────────────────────────────────────────────

    def tick(self) -> None:
        """Decay all active emotions.  Call once per cognitive tick."""
        with self._lock:
            dead = []
            for name in self._active:
                self._active[name] *= self.DECAY_FACTOR
                if self._active[name] < 0.01:
                    dead.append(name)
            for name in dead:
                del self._active[name]

    # ── Queries ──────────────────────────────────────────────────

    def get_active(self) -> dict[str, float]:
        """Return currently active emotions with their intensities."""
        with self._lock:
            return {k: round(v, 3) for k, v in self._active.items() if v >= 0.01}

    def dominant(self) -> str | None:
        """Return the name of the strongest active emotion, or None."""
        active = self.get_active()
        if not active:
            return None
        return max(active, key=active.get)

    def intensity_of(self, emotion: str) -> float:
        with self._lock:
            return self._active.get(emotion, 0.0)

    def summary(self) -> str:
        """Human-readable summary for LLM context."""
        active = self.get_active()
        if not active:
            return "Emotions: (neutral — no active emotions)"
        parts = [f"{name}({intensity:.2f})" for name, intensity in
                 sorted(active.items(), key=lambda x: -x[1])]
        return "Active emotions: " + ", ".join(parts)

    def recent_reactions(self, n: int = 5) -> list[dict]:
        with self._lock:
            return list(self._history[-n:])

    # ── Internal ─────────────────────────────────────────────────

    def _apply_personality(self, emotion: str, base: float) -> float:
        """Modulate an emotion's base intensity by personality traits."""
        p = self.personality

        if emotion == "curiosity":
            base *= (1.0 + 0.4 * p.modifier("curiosity"))
        elif emotion == "frustration":
            # Patient personalities feel less frustration
            base *= max(0.2, 1.0 - 0.5 * p.modifier("patience"))
        elif emotion == "amusement":
            base *= (1.0 + 0.4 * p.modifier("playfulness"))
        elif emotion == "concern":
            base *= (1.0 + 0.3 * p.modifier("empathy"))
        elif emotion == "satisfaction":
            base *= (1.0 + 0.2 * p.modifier("friendliness"))
        elif emotion == "surprise":
            # Risk-tolerant personalities are less surprised
            base *= max(0.3, 1.0 - 0.3 * p.modifier("risk_tolerance"))
        elif emotion == "boredom":
            # Curious personalities get bored faster
            base *= (1.0 + 0.3 * p.modifier("curiosity"))
        elif emotion == "pride":
            base *= (1.0 + 0.2 * p.modifier("assertiveness"))

        return max(0.0, min(1.0, base))
