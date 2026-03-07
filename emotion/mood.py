"""
MoodEngine — long-term emotional state that drifts slowly.

Mood is a 3-dimensional vector:
    valence  [-1 .. +1]  negative ← → positive
    arousal  [ 0 ..  1]  calm ← → excited
    focus    [ 0 ..  1]  scattered ← → concentrated

Mood decays toward a *baseline* each tick (pulled by personality).
Events push it away from baseline; personality traits control how
strongly events affect mood and how fast it recovers.
"""

import time
import threading
from typing import Any

from .personality import PersonalityModel


class MoodEngine:
    """Continuously-decaying mood vector influenced by personality."""

    # Baseline mood — personality shifts it later
    _BASE_VALENCE = 0.15
    _BASE_AROUSAL = 0.20
    _BASE_FOCUS   = 0.40

    # How quickly mood decays back toward baseline per tick
    _DECAY_RATE = 0.05       # 5 % per tick (applied multiplicatively)

    def __init__(self, personality: PersonalityModel):
        self.personality = personality
        self._lock = threading.Lock()

        # Current mood values
        self.valence: float = self._baseline_valence()
        self.arousal: float = self._baseline_arousal()
        self.focus:   float = self._baseline_focus()

        self._last_tick = time.time()

    # ── Baselines (personality-shifted) ──────────────────────────

    def _baseline_valence(self) -> float:
        # Friendly / playful personalities rest at a slightly positive valence
        return self._BASE_VALENCE + 0.15 * self.personality.modifier("friendliness") \
                                  + 0.10 * self.personality.modifier("playfulness")

    def _baseline_arousal(self) -> float:
        # Curious personalities have a slightly elevated resting arousal
        return self._BASE_AROUSAL + 0.15 * self.personality.modifier("curiosity")

    def _baseline_focus(self) -> float:
        return self._BASE_FOCUS + 0.20 * self.personality.modifier("patience")

    # ── Tick / decay ─────────────────────────────────────────────

    def tick(self) -> None:
        """Called once per cognitive tick.  Decays mood toward baseline."""
        now = time.time()
        with self._lock:
            dt = now - self._last_tick
            self._last_tick = now

            # Number of decay steps (normalised to ~5 s ticks)
            steps = max(1, dt / 5.0)
            factor = (1.0 - self._DECAY_RATE) ** steps

            bv = self._baseline_valence()
            ba = self._baseline_arousal()
            bf = self._baseline_focus()

            self.valence = bv + (self.valence - bv) * factor
            self.arousal = ba + (self.arousal - ba) * factor
            self.focus   = bf + (self.focus   - bf) * factor

            self._clamp()

    # ── Push mood (called by EmotionEngine) ──────────────────────

    def push(self, d_valence: float = 0.0,
             d_arousal: float = 0.0,
             d_focus: float = 0.0) -> None:
        """Nudge mood away from baseline.  Personality modulates the push."""
        # Patient personalities dampen negative valence pushes
        patience_mod = self.personality.modifier("patience")
        if d_valence < 0:
            d_valence *= max(0.3, 1.0 - patience_mod)
        # Empathetic personalities amplify user-related pushes
        empathy_mod = self.personality.modifier("empathy")
        d_valence *= (1.0 + 0.3 * empathy_mod)

        with self._lock:
            self.valence += d_valence
            self.arousal += d_arousal
            self.focus   += d_focus
            self._clamp()

    # ── Queries ──────────────────────────────────────────────────

    def get_state(self) -> dict[str, float]:
        with self._lock:
            return {
                "valence": round(self.valence, 3),
                "arousal": round(self.arousal, 3),
                "focus":   round(self.focus, 3),
            }

    def get_mood_label(self) -> str:
        """Map the current vector to the closest named mood for
        backward compatibility with world_state / brain.py."""
        with self._lock:
            v, a = self.valence, self.arousal

        if v >= 0.6 and a >= 0.6:
            return "excited"
        if v >= 0.5 and a < 0.4:
            return "satisfied"
        if v >= 0.3 and a >= 0.3:
            return "amused"
        if v >= 0.2 and a >= 0.3:
            return "curious"
        if v >= 0.1 and a >= 0.5:
            return "focused"
        if v >= -0.1:
            return "calm"
        if v >= -0.3 and a < 0.3:
            return "bored"
        if v >= -0.3:
            return "concerned"
        return "frustrated"

    def summary(self) -> str:
        s = self.get_state()
        label = self.get_mood_label()
        return (f"Mood: {label} "
                f"(valence={s['valence']:.2f}, "
                f"arousal={s['arousal']:.2f}, "
                f"focus={s['focus']:.2f})")

    # ── Internal ─────────────────────────────────────────────────

    def _clamp(self) -> None:
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max( 0.0, min(1.0, self.arousal))
        self.focus   = max( 0.0, min(1.0, self.focus))
