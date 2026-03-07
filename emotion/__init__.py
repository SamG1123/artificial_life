"""
Emotion package — three-layer emotional system for the autonomous AI.

Layers (stable → volatile):
  PersonalityModel  — stable traits that rarely change
  MoodEngine        — long-term emotional state, decays toward baseline
  EmotionEngine     — short-term reactions to events
  BehaviorController — translates emotional state into behavioral params
"""

from .personality import PersonalityModel
from .mood import MoodEngine
from .emotion import EmotionEngine
from .controller import BehaviorController

__all__ = [
    "PersonalityModel",
    "MoodEngine",
    "EmotionEngine",
    "BehaviorController",
]
