"""Lightweight in-process event bus for avatar runtime."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Callable


EventHandler = Callable[[dict], None]


class AvatarEventBus:
    def __init__(self):
        self._lock = threading.RLock()
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        with self._lock:
            self._handlers[event_name].append(handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> None:
        with self._lock:
            handlers = self._handlers.get(event_name)
            if not handlers:
                return
            try:
                handlers.remove(handler)
            except ValueError:
                return

    def publish(self, event_name: str, payload: dict[str, Any] | None = None) -> None:
        event = {
            "type": event_name,
            "ts": time.time(),
            "payload": payload or {},
        }
        with self._lock:
            handlers = list(self._handlers.get(event_name, []))
            wildcard = list(self._handlers.get("*", []))
        for handler in handlers + wildcard:
            try:
                handler(event)
            except Exception:
                # Event sinks must not crash the producer pipeline.
                continue
