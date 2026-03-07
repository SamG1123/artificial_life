"""
Cognitive Attention System — biologically-inspired focus management.

Models human-like attention with:
  - Salience scoring (novelty, urgency, relevance, intensity)
  - Habituation (repeated identical stimuli lose salience)
  - Focus stack (current focus + background monitoring)
  - Change detection (only surface what actually changed)
  - Resource gating (throttle low-priority channels when busy)
  - Explicit focus directives ("pay attention to X")

Sits between WorldStateManager and Brain, filtering the raw
perception stream so the brain processes only what matters.

    Perception → WorldState → [ ATTENTION ] → Brain
                                   │
                      salience / habituation / focus
"""

import time
import hashlib
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from memory.compressed_storage import save_json, load_json


# ── Data structures ──────────────────────────────────────────────

@dataclass
class AttentionFocus:
    """A single item the system is attending to."""
    channel: str          # "screen" | "camera" | "audio" | "system" | "goal"
    label: str            # short description of what's being attended
    salience: float       # 0.0 – 1.0 overall score
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0 # seconds this has been attended to

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "label": self.label,
            "salience": round(self.salience, 3),
            "timestamp": self.timestamp,
            "duration": round(self.duration, 1),
        }


# ── Salience weights per cognitive state ─────────────────────────

_STATE_WEIGHTS = {
    # channel →  IDLE   EXECUTING  CONVERSING
    "screen":  (0.4,    0.6,       0.2),
    "camera":  (0.3,    0.2,       0.2),
    "audio":   (0.7,    0.5,       0.9),
    "system":  (0.3,    0.4,       0.1),
}

_STATE_IDX = {"IDLE": 0, "EXECUTING": 1, "CONVERSING": 2}


# ── AttentionSystem ──────────────────────────────────────────────

class AttentionSystem:
    """Filters and prioritises perception data for the brain."""

    # Salience threshold — items below this are dropped
    SALIENCE_FLOOR = 0.15

    # How many ticks an identical summary must repeat before habituation kicks in
    HABITUATION_THRESHOLD = 3

    # Max items on the focus stack
    MAX_FOCUS = 5

    # Habituation decay: how much salience drops per repeated observation
    HABITUATION_DECAY = 0.25

    def __init__(self, store_dir: str = "memory_store"):
        self._store_dir = store_dir
        self._lock = threading.Lock()

        # Focus stack — ordered by salience (highest first)
        self._focus_stack: list[AttentionFocus] = []

        # Habituation memory: channel → (content_hash, repeat_count)
        self._habituation: dict[str, tuple[str, int]] = {}

        # Previous perception summaries for change detection
        self._prev_summaries: dict[str, str] = {}

        # Explicit focus directives (channel → weight boost 0-1)
        self._directives: dict[str, float] = {}

        # Current goal context for relevance scoring
        self._current_goal: str | None = None

        # Stats
        self._total_processed = 0
        self._total_suppressed = 0

    # ── Public API ───────────────────────────────────────────────

    def process(self, world_state: dict, cognitive_state: str = "IDLE") -> dict:
        """Score and filter a world-state snapshot.

        Args:
            world_state: full state dict from WorldStateManager.get_full_state()
            cognitive_state: "IDLE" | "EXECUTING" | "CONVERSING"

        Returns:
            Filtered dict with only the channels that passed the
            attention gate, plus an "attention" metadata key.
        """
        state_idx = _STATE_IDX.get(cognitive_state, 0)
        scored: list[tuple[str, float, dict]] = []  # (channel, salience, data)

        for channel in ("screen", "camera", "audio", "system"):
            ch_data = world_state.get(channel)
            if not ch_data:
                continue

            summary = ""
            if isinstance(ch_data, dict):
                summary = ch_data.get("summary", "")

            # 1. Base weight from cognitive state
            weights = _STATE_WEIGHTS.get(channel, (0.3, 0.3, 0.3))
            base_weight = weights[state_idx]

            # 2. Novelty — did the content actually change?
            novelty = self._compute_novelty(channel, summary)

            # 3. Urgency — keywords that signal immediate importance
            urgency = self._compute_urgency(channel, summary)

            # 4. Relevance — alignment with current goal
            relevance = self._compute_relevance(channel, summary)

            # 5. Directive boost
            directive_boost = self._directives.get(channel, 0.0)

            # 6. Habituation penalty
            habituation_penalty = self._compute_habituation(channel, summary)

            # Composite salience
            salience = (
                0.25 * base_weight
                + 0.30 * novelty
                + 0.20 * urgency
                + 0.15 * relevance
                + 0.10 * directive_boost
                - habituation_penalty
            )
            salience = max(0.0, min(1.0, salience))

            scored.append((channel, salience, ch_data))

        # Sort by salience descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build filtered output
        filtered = {}
        new_focus: list[AttentionFocus] = []

        with self._lock:
            for channel, salience, ch_data in scored:
                self._total_processed += 1
                if salience < self.SALIENCE_FLOOR:
                    self._total_suppressed += 1
                    continue
                filtered[channel] = ch_data
                summary = ch_data.get("summary", "") if isinstance(ch_data, dict) else ""
                new_focus.append(AttentionFocus(
                    channel=channel,
                    label=summary[:80] if summary else channel,
                    salience=salience,
                ))

            # Keep goal focus if present
            if self._current_goal:
                goal_focus = AttentionFocus(
                    channel="goal",
                    label=self._current_goal[:80],
                    salience=0.9,
                )
                new_focus.insert(0, goal_focus)

            # Trim to max
            self._focus_stack = new_focus[:self.MAX_FOCUS]

        # Carry over non-perception keys
        filtered["timestamp"] = world_state.get("timestamp", time.time())
        filtered["memory"] = world_state.get("memory", {})
        filtered["emotion"] = world_state.get("emotion", {})
        filtered["summary"] = world_state.get("summary", "")

        # Attach attention metadata
        filtered["attention"] = {
            "focus": [f.to_dict() for f in self._focus_stack],
            "suppressed_channels": [
                ch for ch, sal, _ in scored if sal < self.SALIENCE_FLOOR
            ],
        }

        return filtered

    def set_goal(self, goal: str | None) -> None:
        """Tell the attention system what the agent is currently working on."""
        with self._lock:
            self._current_goal = goal

    def add_directive(self, channel: str, boost: float = 0.5) -> None:
        """Manually boost attention on a channel ("pay attention to screen")."""
        with self._lock:
            self._directives[channel] = max(0.0, min(1.0, boost))

    def remove_directive(self, channel: str) -> None:
        """Remove an explicit focus directive."""
        with self._lock:
            self._directives.pop(channel, None)

    def clear_directives(self) -> None:
        """Remove all explicit focus directives."""
        with self._lock:
            self._directives.clear()

    def get_focus(self) -> list[dict]:
        """Return what the system is currently attending to."""
        with self._lock:
            return [f.to_dict() for f in self._focus_stack]

    def get_primary_focus(self) -> dict | None:
        """Return the highest-salience focus item."""
        with self._lock:
            if self._focus_stack:
                return self._focus_stack[0].to_dict()
            return None

    def get_context_for_llm(self) -> str:
        """Build a compact attention summary suitable for LLM prompts."""
        with self._lock:
            if not self._focus_stack:
                return "(no active focus)"
            lines = []
            for f in self._focus_stack[:3]:
                lines.append(f"  [{f.channel}] {f.label} (salience={f.salience:.2f})")
            suppressed = self._total_suppressed
            processed = self._total_processed
        parts = ["Current attention:"]
        parts.extend(lines)
        if processed > 0:
            parts.append(f"  (filtered {suppressed}/{processed} low-priority inputs)")
        return "\n".join(parts)

    def stats(self) -> dict:
        """Return attention statistics."""
        with self._lock:
            return {
                "focus_count": len(self._focus_stack),
                "total_processed": self._total_processed,
                "total_suppressed": self._total_suppressed,
                "suppression_rate": (
                    round(self._total_suppressed / self._total_processed, 3)
                    if self._total_processed > 0 else 0.0
                ),
                "active_directives": dict(self._directives),
                "current_goal": self._current_goal,
            }

    # ── Scoring helpers ──────────────────────────────────────────

    def _compute_novelty(self, channel: str, summary: str) -> float:
        """How different is this observation from the last one on the same channel?"""
        prev = self._prev_summaries.get(channel, "")
        self._prev_summaries[channel] = summary

        if not prev and not summary:
            return 0.0
        if not prev or not summary:
            return 1.0  # first observation or content appeared/vanished
        if prev == summary:
            return 0.0  # identical — nothing new

        # Rough word-overlap ratio → novelty
        prev_words = set(prev.lower().split())
        curr_words = set(summary.lower().split())
        if not prev_words and not curr_words:
            return 0.0
        union = prev_words | curr_words
        intersection = prev_words & curr_words
        overlap = len(intersection) / len(union) if union else 1.0
        return 1.0 - overlap

    def _compute_urgency(self, channel: str, summary: str) -> float:
        """Detect keywords / patterns that signal high urgency."""
        if not summary:
            return 0.0
        lower = summary.lower()

        # Audio is inherently urgent (user might be talking)
        if channel == "audio" and summary.strip():
            return 0.8

        urgency = 0.0
        urgent_keywords = [
            "error", "crash", "fail", "critical", "alert", "warning",
            "user", "said", "speaking", "asking", "urgent", "stopped",
            "not responding", "high cpu", "low memory", "disk full",
        ]
        for kw in urgent_keywords:
            if kw in lower:
                urgency = max(urgency, 0.6)
                break

        # System resource pressure
        if channel == "system":
            if any(w in lower for w in ("high cpu", "low memory", "disk full", "90%", "95%", "99%")):
                urgency = max(urgency, 0.7)

        return min(1.0, urgency)

    def _compute_relevance(self, channel: str, summary: str) -> float:
        """How relevant is this perception to the current goal?"""
        with self._lock:
            goal = self._current_goal
        if not goal or not summary:
            return 0.0

        # Simple word overlap between goal and summary
        goal_words = set(goal.lower().split())
        summary_words = set(summary.lower().split())
        if not goal_words:
            return 0.0
        overlap = len(goal_words & summary_words)
        return min(1.0, overlap / max(1, len(goal_words)) * 1.5)

    def _compute_habituation(self, channel: str, summary: str) -> float:
        """Penalise repeated identical observations (habituation)."""
        content_hash = hashlib.md5(summary.encode()).hexdigest()[:12] if summary else ""

        prev = self._habituation.get(channel)
        if prev and prev[0] == content_hash:
            count = prev[1] + 1
        else:
            count = 1

        self._habituation[channel] = (content_hash, count)

        if count <= self.HABITUATION_THRESHOLD:
            return 0.0
        # Progressive penalty
        excess = count - self.HABITUATION_THRESHOLD
        return min(0.5, excess * self.HABITUATION_DECAY * 0.1)
