"""Preference learning and personalization helper.

Learns user preferences from conversation and activity patterns with
confidence scoring so weak signals are not acted upon.
"""

from __future__ import annotations

import json
import os
import re
import time
from threading import Lock


class PreferenceLearner:
    """Extracts and tracks user preference signals over time."""

    def __init__(self, memory, store_dir: str = "memory_store"):
        self.memory = memory
        self._path = os.path.join(store_dir, "preference_signals.json")
        self._lock = Lock()
        self._signals: dict[str, dict] = {}
        self._last_suggestion_ts = 0.0
        os.makedirs(store_dir, exist_ok=True)
        self._load()

    # -- Signal ingestion -----------------------------------------

    def observe_user_text(self, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        lower = t.lower()

        # Explicit positive preferences
        for pat in (r"\bi like\s+(.+)", r"\bi love\s+(.+)", r"\bi prefer\s+(.+)"):
            m = re.search(pat, lower)
            if m:
                self._add_signal(f"likes {m.group(1)[:80].strip()}", weight=1.0)

        # Explicit negative preferences
        for pat in (r"\bi (?:don't|do not) like\s+(.+)", r"\bi hate\s+(.+)", r"\bi dislike\s+(.+)"):
            m = re.search(pat, lower)
            if m:
                self._add_signal(f"dislikes {m.group(1)[:80].strip()}", weight=1.0)

        # Correction signal
        corr = re.search(r"\bno\b.*\bi meant\s+(.+)", lower)
        if corr:
            self._add_signal(f"corrected intent to: {corr.group(1)[:80].strip()}", weight=0.8)

        self._promote_high_confidence()

    def observe_goal(self, goal: str) -> None:
        g = (goal or "").lower()
        if not g:
            return

        # Simple app-usage preference heuristics
        if "vs code" in g or "vscode" in g:
            self._add_signal("frequently uses VS Code", weight=0.4)
        if "chrome" in g or "browser" in g or "search" in g:
            self._add_signal("frequently uses web browser", weight=0.3)

        # Time-of-day pattern (lightweight)
        hour = time.localtime().tm_hour
        if hour < 11:
            self._add_signal("active in mornings", weight=0.2)
        elif hour >= 20:
            self._add_signal("active in evenings", weight=0.2)

        self._promote_high_confidence()

    # -- Suggestions ----------------------------------------------

    def maybe_suggestion(self, min_confidence: float = 2.0,
                         cooldown_sec: float = 900.0) -> str:
        """Return a personalized suggestion only when confidence is strong."""
        now = time.time()
        if now - self._last_suggestion_ts < cooldown_sec:
            return ""

        with self._lock:
            if not self._signals:
                return ""
            top_key, top = max(self._signals.items(), key=lambda kv: kv[1].get("score", 0.0))

        score = float(top.get("score", 0.0))
        if score < min_confidence:
            return ""

        self._last_suggestion_ts = now

        if "VS Code" in top_key:
            return "You usually open VS Code first. Should I launch it now?"
        if "web browser" in top_key:
            return "You often start with web browsing. Want me to open the browser?"
        if "mornings" in top_key:
            return "You are usually active in the morning. Want me to prep your common tools?"
        if "evenings" in top_key:
            return "You tend to work in the evening. Should I queue up your usual setup?"
        return f"I have noticed a pattern: {top_key}. Want me to personalize around that?"

    def summary(self, top_n: int = 5) -> str:
        with self._lock:
            ranked = sorted(self._signals.items(), key=lambda kv: kv[1].get("score", 0.0), reverse=True)
        if not ranked:
            return ""
        lines = ["Preference signals:"]
        for key, val in ranked[:top_n]:
            lines.append(f"- {key} (confidence {val.get('score', 0.0):.1f})")
        return "\n".join(lines)

    # -- Internals -------------------------------------------------

    def _add_signal(self, key: str, weight: float) -> None:
        if not key:
            return
        with self._lock:
            rec = self._signals.setdefault(key, {"score": 0.0, "count": 0, "updated_at": 0.0})
            rec["score"] = float(rec.get("score", 0.0)) + float(weight)
            rec["count"] = int(rec.get("count", 0)) + 1
            rec["updated_at"] = time.time()
            self._save()

    def _promote_high_confidence(self) -> None:
        with self._lock:
            ranked = sorted(self._signals.items(), key=lambda kv: kv[1].get("score", 0.0), reverse=True)
        for key, rec in ranked[:10]:
            if float(rec.get("score", 0.0)) >= 2.5:
                try:
                    self.memory.learn_preference(key)
                except Exception:
                    pass

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self._signals = raw
        except (OSError, json.JSONDecodeError):
            self._signals = {}

    def _save(self) -> None:
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._signals, f, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            pass
