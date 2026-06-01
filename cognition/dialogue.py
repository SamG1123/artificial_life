"""Dialogue state tracking for multi-turn, persistent conversations.

Tracks:
- active topic
- unresolved user questions
- assistant commitments/promises

State is persisted locally so conversation continuity survives restarts.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass


_TOPIC_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so",
    "i", "you", "we", "they", "it", "this", "that", "to", "of",
    "for", "on", "in", "at", "with", "is", "are", "was", "were",
    "do", "does", "did", "can", "could", "would", "should", "will",
    "please", "me", "my", "your", "our", "their",
}

_QUESTION_PREFIXES = (
    "what", "why", "how", "when", "where", "who", "which",
    "can", "could", "would", "will", "should", "is", "are", "do", "did",
)


@dataclass
class _DialogueItem:
    text: str
    created_at: float
    status: str = "open"  # open | resolved
    reminded_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "created_at": self.created_at,
            "status": self.status,
            "reminded_at": self.reminded_at,
        }


class DialogueStateTracker:
    """Tracks conversation state and persists it per local user profile."""

    def __init__(self, store_dir: str = "memory_store", user_id: str = "default"):
        self._store_dir = store_dir
        self._user_id = user_id
        self._path = os.path.join(store_dir, f"dialogue_state_{user_id}.json")

        self.active_topic = ""
        self.open_questions: list[_DialogueItem] = []
        self.commitments: list[_DialogueItem] = []
        self.mood_arc: list[str] = []

        os.makedirs(store_dir, exist_ok=True)
        self._load()

    # ── Conversation ingestion ──────────────────────────────────

    def ingest_user(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        topic = self._infer_topic(text)
        if topic:
            self.active_topic = topic

        if self._is_question(text):
            self.open_questions.append(_DialogueItem(text=text, created_at=time.time()))

        self._save()

    def ingest_assistant(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        lowered = text.lower()
        if any(p in lowered for p in ("i will", "i'll", "let me", "i can", "i can do that")):
            self.commitments.append(_DialogueItem(text=text, created_at=time.time()))

        self._save()

    def resolve_by_goal(self, goal: str, success: bool) -> None:
        """Resolve likely related open items after a successful execution."""
        if not success:
            return
        goal_terms = set(self._tokenize(goal))
        if not goal_terms:
            return

        for item in self.open_questions:
            if item.status != "open":
                continue
            if goal_terms.intersection(self._tokenize(item.text)):
                item.status = "resolved"

        for item in self.commitments:
            if item.status != "open":
                continue
            if goal_terms.intersection(self._tokenize(item.text)):
                item.status = "resolved"

        self._save()

    # ── Proactive follow-up ─────────────────────────────────────

    def next_follow_up(self, min_interval_sec: float = 120.0) -> str:
        """Return a single follow-up line for unresolved questions/commitments."""
        now = time.time()

        for item in self.open_questions:
            if item.status == "open" and (now - item.reminded_at) >= min_interval_sec:
                item.reminded_at = now
                self._save()
                return f"Earlier you asked: '{item.text}'. Want me to continue on that now?"

        for item in self.commitments:
            if item.status == "open" and (now - item.reminded_at) >= min_interval_sec:
                item.reminded_at = now
                self._save()
                return f"I still have this pending promise: '{item.text}'. Should I handle it now?"

        return ""

    # ── Prompt context ──────────────────────────────────────────

    def get_context_summary(self, max_items: int = 3) -> str:
        open_q = [q.text for q in self.open_questions if q.status == "open"][:max_items]
        open_c = [c.text for c in self.commitments if c.status == "open"][:max_items]

        lines = []
        if self.active_topic:
            lines.append(f"Active topic: {self.active_topic}")
        if open_q:
            lines.append("Open user questions:")
            for q in open_q:
                lines.append(f"- {q}")
        if open_c:
            lines.append("Outstanding commitments:")
            for c in open_c:
                lines.append(f"- {c}")
        return "\n".join(lines)

    # ── Internals ───────────────────────────────────────────────

    def _is_question(self, text: str) -> bool:
        t = text.strip().lower()
        if "?" in t:
            return True
        return t.startswith(_QUESTION_PREFIXES)

    def _infer_topic(self, text: str) -> str:
        tokens = [t for t in self._tokenize(text) if t not in _TOPIC_STOPWORDS]
        if not tokens:
            return ""
        return " ".join(tokens[:3])

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        self.active_topic = raw.get("active_topic", "")
        self.mood_arc = raw.get("mood_arc", [])

        self.open_questions = [
            _DialogueItem(**q)
            for q in raw.get("open_questions", [])
            if isinstance(q, dict)
        ]
        self.commitments = [
            _DialogueItem(**c)
            for c in raw.get("commitments", [])
            if isinstance(c, dict)
        ]

    def _save(self) -> None:
        payload = {
            "active_topic": self.active_topic,
            "open_questions": [q.to_dict() for q in self.open_questions][-50:],
            "commitments": [c.to_dict() for c in self.commitments][-50:],
            "mood_arc": self.mood_arc[-100:],
        }
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            pass
