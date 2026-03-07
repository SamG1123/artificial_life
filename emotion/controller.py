"""
BehaviorController — translates emotional + personality state into
concrete behavioral parameters that the brain's decision-making uses.

This is the bridge between the emotional system and the cognitive loop.
Instead of the brain reading raw floats, it asks the controller
questions like:
    "should I speak right now?"
    "how verbose should I be?"
    "should I self-initiate a goal?"
    "what temperature should the LLM use?"

The controller inspects personality, mood, and active emotions to
produce actionable answers.
"""

import threading
from typing import Any

from .personality import PersonalityModel
from .mood import MoodEngine
from .emotion import EmotionEngine


class BehaviorController:
    """Decision-support layer driven by emotion + personality."""

    def __init__(self, personality: PersonalityModel,
                 mood: MoodEngine, emotion: EmotionEngine):
        self.personality = personality
        self.mood = mood
        self.emotion = emotion

    # ── Tick (cascades decay through all layers) ─────────────────

    def tick(self) -> None:
        """One cognitive tick — decay mood and emotions."""
        self.mood.tick()
        self.emotion.tick()

    # ── Behavioral queries ───────────────────────────────────────

    def should_speak(self) -> bool:
        """Should the AI say something proactively right now?

        More likely when: amused, excited, curious, or concerned.
        Less likely when: bored, calm, or focused."""
        ms = self.mood.get_state()
        dom = self.emotion.dominant()

        # Active emotions that encourage speaking
        if dom in ("amusement", "curiosity", "concern", "surprise"):
            return True

        # High arousal + positive valence → chatty
        if ms["arousal"] > 0.5 and ms["valence"] > 0.3:
            return True

        # Otherwise quiet
        return False

    def should_self_initiate(self) -> bool:
        """Should the AI start a goal on its own?

        Gated by personality (risk_tolerance, curiosity) and mood.
        Boosted when the curiosity emotion is active."""
        risk = self.personality.get("risk_tolerance")
        curiosity = self.personality.get("curiosity")
        ms = self.mood.get_state()

        score = (risk * 0.4 + curiosity * 0.3
                 + ms["valence"] * 0.15 + ms["focus"] * 0.15)

        # Active curiosity emotion lowers the threshold
        curiosity_intensity = self.emotion.intensity_of("curiosity")
        if curiosity_intensity > 0.2:
            score += curiosity_intensity * 0.20

        # Frustration raises the threshold
        frustration = self.emotion.intensity_of("frustration")
        if frustration > 0.4:
            score -= frustration * 0.15

        return score > 0.40

    def verbosity(self) -> str:
        """How verbose should the AI be? Returns 'brief', 'normal', or 'detailed'."""
        ms = self.mood.get_state()
        playful = self.personality.get("playfulness")

        # Frustrated or bored → brief
        if ms["valence"] < -0.2:
            return "brief"
        # Excited / amused + playful → detailed
        if ms["arousal"] > 0.5 and playful > 0.6:
            return "detailed"
        return "normal"

    def llm_temperature(self) -> float:
        """Suggest an LLM temperature based on emotional state.

        Calmer / more focused → lower temperature (precise).
        Excited / playful → higher temperature (creative)."""
        ms = self.mood.get_state()
        playful = self.personality.get("playfulness")

        base = 0.5
        base += 0.15 * ms["arousal"]
        base += 0.10 * (playful - 0.5)
        base -= 0.10 * ms["focus"]

        return max(0.2, min(0.9, round(base, 2)))

    def retry_willingness(self) -> float:
        """How willing is the AI to retry a failed goal? [0 .. 1]"""
        persistence = self.personality.get("persistence")
        frustration = self.emotion.intensity_of("frustration")
        ms = self.mood.get_state()

        score = persistence * 0.5 + ms["valence"] * 0.25 + (1.0 - frustration) * 0.25
        return max(0.0, min(1.0, score))

    def thought_interval_multiplier(self) -> float:
        """Multiplier on the idle-thought interval.

        < 1 → think more often (curious / stimulated)
        > 1 → think less often (bored / calm)
        """
        curiosity = self.personality.get("curiosity")
        ms = self.mood.get_state()

        # Curious + aroused → faster thoughts
        if curiosity > 0.6 and ms["arousal"] > 0.4:
            return 0.6
        # Bored → slower
        if ms["valence"] < -0.1 and ms["arousal"] < 0.2:
            return 1.4
        return 1.0

    # ── Context for LLM prompts ──────────────────────────────────

    def get_emotional_context(self) -> str:
        """Build a compact emotional-state block for injection into
        LLM prompts.  Replaces the previous bare ``Mood: calm`` line."""
        parts = [
            self.mood.summary(),
            self.emotion.summary(),
        ]
        v = self.verbosity()
        if v != "normal":
            parts.append(f"Response style: {v}")
        return "\n".join(parts)

    def get_full_state(self) -> dict:
        """Return the complete emotional state as a serialisable dict."""
        return {
            "personality": self.personality.get_all(),
            "mood": self.mood.get_state(),
            "mood_label": self.mood.get_mood_label(),
            "emotions": self.emotion.get_active(),
            "dominant_emotion": self.emotion.dominant(),
            "behavior": {
                "should_speak": self.should_speak(),
                "should_self_initiate": self.should_self_initiate(),
                "verbosity": self.verbosity(),
                "llm_temperature": self.llm_temperature(),
                "retry_willingness": self.retry_willingness(),
                "thought_interval_mult": self.thought_interval_multiplier(),
            },
        }

    def summary(self) -> str:
        """One-line summary combining mood + dominant emotion."""
        label = self.mood.get_mood_label()
        dom = self.emotion.dominant()
        if dom:
            return f"{label} (feeling {dom})"
        return label
