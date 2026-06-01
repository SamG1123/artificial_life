"""Simple text-driven viseme timing helpers."""

from __future__ import annotations

import re

_VISEME_TABLE = {
    "a": "aa", "e": "ee", "i": "ih", "o": "oh", "u": "uh",
    "m": "mbp", "b": "mbp", "p": "mbp",
    "f": "fv", "v": "fv",
    "l": "l", "r": "r",
    "s": "sz", "z": "sz", "c": "sz", "x": "sz",
    "t": "td", "d": "td", "n": "td",
    "k": "kg", "g": "kg", "q": "kg",
    "w": "wq", "y": "wq",
}


def estimate_duration_seconds(text: str, speed: float = 1.0) -> float:
    words = max(1, len(re.findall(r"\w+", text or "")))
    wps = max(1.2, 2.8 * max(0.6, min(1.6, speed)))
    return max(0.35, words / wps)


def viseme_at_progress(text: str, progress: float) -> str:
    if not text:
        return "rest"
    chars = [c.lower() for c in text if c.isalpha()]
    if not chars:
        return "rest"
    idx = min(len(chars) - 1, max(0, int(progress * len(chars))))
    ch = chars[idx]
    return _VISEME_TABLE.get(ch, "rest")
