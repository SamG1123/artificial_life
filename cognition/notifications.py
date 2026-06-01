"""Notification engine for proactive status updates.

Supports:
- priority channels (urgent, normal, low)
- do-not-disturb while executing goals
- user-configurable notification preferences (local JSON)
- system-event ingestion (battery and resource warnings)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from threading import Lock


@dataclass
class Notification:
    message: str
    priority: str  # urgent | normal | low
    source: str = "system"
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "priority": self.priority,
            "source": self.source,
            "timestamp": self.timestamp,
        }


class NotificationEngine:
    """Queues and dispatches proactive notifications."""

    def __init__(self, store_dir: str = "memory_store"):
        self._lock = Lock()
        self._path = os.path.join(store_dir, "notification_prefs.json")
        self._queue: list[Notification] = []
        self._history: list[dict] = []

        self._prefs = {
            "allow_urgent": True,
            "allow_normal": True,
            "allow_low": True,
            "dnd_during_executing": True,
        }
        self._last_event_ts: dict[str, float] = {}
        os.makedirs(store_dir, exist_ok=True)
        self._load_prefs()

    # -- Public API -------------------------------------------------

    def publish(self, message: str, priority: str = "normal", source: str = "system") -> None:
        if not message:
            return
        note = Notification(
            message=message.strip(),
            priority=priority,
            source=source,
            timestamp=time.time(),
        )
        with self._lock:
            self._queue.append(note)
            self._queue.sort(key=lambda n: {"urgent": 0, "normal": 1, "low": 2}.get(n.priority, 1))

    def next_for_state(self, cognitive_state: str) -> Notification | None:
        """Return the next notification allowed in the current state."""
        with self._lock:
            if not self._queue:
                return None

            for i, note in enumerate(self._queue):
                if not self._is_allowed(note, cognitive_state):
                    continue
                self._queue.pop(i)
                self._history.append(note.to_dict())
                self._history = self._history[-200:]
                return note
        return None

    def ingest_system_snapshot(self, system_data: dict) -> None:
        """Generate notifications from system perception payload."""
        if not isinstance(system_data, dict):
            return

        raw = system_data.get("data", {}) if "data" in system_data else system_data
        battery = raw.get("battery")
        resources = raw.get("resources", {})

        # Battery low event (cooldown 10 minutes)
        if isinstance(battery, dict):
            pct = battery.get("percent")
            plugged = battery.get("plugged_in", True)
            if pct is not None and pct <= 20 and not plugged:
                self._publish_with_cooldown(
                    key="battery_low",
                    cooldown_sec=600,
                    message=f"Battery is low at {pct}%. Consider plugging in.",
                    priority="urgent",
                    source="system",
                )

        # High CPU event (cooldown 5 minutes)
        cpu = resources.get("cpu_percent")
        if isinstance(cpu, (int, float)) and cpu >= 90:
            self._publish_with_cooldown(
                key="cpu_high",
                cooldown_sec=300,
                message=f"CPU usage is high at {cpu:.0f}%.",
                priority="normal",
                source="system",
            )

    def update_preferences(self, **prefs: bool) -> None:
        with self._lock:
            for k, v in prefs.items():
                if k in self._prefs:
                    self._prefs[k] = bool(v)
            self._save_prefs()

    def stats(self) -> dict:
        with self._lock:
            return {
                "queued": len(self._queue),
                "history": len(self._history),
                "prefs": dict(self._prefs),
            }

    # -- Internals --------------------------------------------------

    def _is_allowed(self, note: Notification, state: str) -> bool:
        if note.priority == "urgent":
            return self._prefs.get("allow_urgent", True)

        if note.priority == "normal" and not self._prefs.get("allow_normal", True):
            return False
        if note.priority == "low" and not self._prefs.get("allow_low", True):
            return False

        if state == "EXECUTING" and self._prefs.get("dnd_during_executing", True):
            return False
        return True

    def _publish_with_cooldown(self, key: str, cooldown_sec: float,
                               message: str, priority: str, source: str) -> None:
        now = time.time()
        last = self._last_event_ts.get(key, 0.0)
        if now - last < cooldown_sec:
            return
        self._last_event_ts[key] = now
        self.publish(message, priority=priority, source=source)

    def _load_prefs(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self._prefs.update({k: bool(v) for k, v in raw.items() if k in self._prefs})
        except (OSError, json.JSONDecodeError):
            pass

    def _save_prefs(self) -> None:
        tmp = self._path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._prefs, f, indent=2)
            os.replace(tmp, self._path)
        except OSError:
            pass
