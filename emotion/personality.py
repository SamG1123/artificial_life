"""
PersonalityModel — stable traits that shape how the AI reacts to the world.

Each trait is a float in [0, 1]. Traits rarely change; small drift is
allowed over long time-spans to simulate gradual personality development.

The model provides *modifiers* that downstream engines (MoodEngine,
EmotionEngine, BehaviorController) multiply against their raw values
so the same event feels different depending on personality.
"""

import json
import os
import threading
import time
from typing import Any

from memory.compressed_storage import save_json, load_json, migrate_if_needed

# Default trait values — balanced personality
_DEFAULTS: dict[str, float] = {
    "curiosity":       0.85,   # desire to explore / observe — high for self-directed AI
    "patience":        0.60,   # tolerance for slow progress or failures
    "friendliness":    0.75,   # warmth toward the user
    "risk_tolerance":  0.55,   # willingness to self-initiate risky actions
    "playfulness":     0.55,   # humour, lightheartedness
    "assertiveness":   0.60,   # how strongly the AI voices its own opinions
    "empathy":         0.65,   # sensitivity to user emotion / frustration
    "persistence":     0.70,   # determination to retry failed goals
}

# How much a trait can drift per nudge (capped per call)
_MAX_DRIFT = 0.02


class PersonalityModel:
    """Stable personality layer — rarely mutated, frequently queried."""

    def __init__(self, store_dir: str | None = None):
        self._lock = threading.Lock()
        self._store_path: str | None = None
        if store_dir:
            os.makedirs(store_dir, exist_ok=True)
            self._store_path = os.path.join(store_dir, "personality.json")

        self.traits: dict[str, float] = dict(_DEFAULTS)
        self._load()

    # ── Queries ──────────────────────────────────────────────────

    def get(self, trait: str) -> float:
        """Return a trait value (0 .. 1).  Unknown traits return 0.5."""
        with self._lock:
            return self.traits.get(trait, 0.5)

    def get_all(self) -> dict[str, float]:
        with self._lock:
            return dict(self.traits)

    def modifier(self, trait: str) -> float:
        """Return a [-0.5 .. +0.5] modifier centred around the trait.

        Useful for scaling emotional reactions:
            reaction_strength = base_strength + personality.modifier("patience")
        """
        return self.get(trait) - 0.5

    def summary(self) -> str:
        """Human-readable summary for LLM prompts."""
        with self._lock:
            lines = []
            for t, v in self.traits.items():
                label = (
                    "very low" if v < 0.2 else
                    "low" if v < 0.4 else
                    "moderate" if v < 0.6 else
                    "high" if v < 0.8 else
                    "very high"
                )
                lines.append(f"  {t}: {label} ({v:.2f})")
            return "Personality traits:\n" + "\n".join(lines)

    # ── Mutation (very rare — slow personality drift) ────────────

    def nudge(self, trait: str, delta: float) -> None:
        """Shift a trait by a tiny amount. Clamped to [0, 1] and
        rate-limited to ``_MAX_DRIFT`` per call."""
        clamped_delta = max(-_MAX_DRIFT, min(_MAX_DRIFT, delta))
        with self._lock:
            if trait in self.traits:
                old = self.traits[trait]
                self.traits[trait] = max(0.0, min(1.0, old + clamped_delta))
        self._save()

    # ── Persistence ──────────────────────────────────────────────

    def _load(self) -> None:
        if not self._store_path:
            return
        migrate_if_needed(self._store_path)
        data = load_json(self._store_path)
        if data is None:
            return
        try:
            with self._lock:
                for k, v in data.items():
                    if k in self.traits and isinstance(v, (int, float)):
                        self.traits[k] = max(0.0, min(1.0, float(v)))
        except Exception as e:
            print(f"[Personality] Failed to load: {e}")

    def _save(self) -> None:
        if not self._store_path:
            return
        try:
            with self._lock:
                data = dict(self.traits)
            save_json(self._store_path, data)
        except Exception as e:
            print(f"[Personality] Failed to save: {e}")
