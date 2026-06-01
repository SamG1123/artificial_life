"""Shared avatar state model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AvatarState:
    mode: str = "idle"
    mood: str = "calm"
    dominant_emotion: str = "neutral"
    valence: float = 0.0
    arousal: float = 0.2
    focus: float = 0.4
    speaking: bool = False
    speech_text: str = ""
    viseme: str = "rest"
    expression_strength: float = 0.0
    fps: int = 30
    metadata: dict = field(default_factory=dict)
