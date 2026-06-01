"""Continuous desktop frame stream for low-latency awareness.

Captures the display on a background thread and stores the latest frame.
Consumers can fetch the most recent screenshot without blocking on capture.
"""

from __future__ import annotations

import threading
import time

import pyautogui as pag
from PIL import Image


class ContinuousScreenStream:
    def __init__(self, fps: float = 4.0):
        self._fps = max(1.0, float(fps))
        self._lock = threading.Lock()
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

        self._latest: Image.Image | None = None
        self._latest_ts: float = 0.0
        self._frames_captured: int = 0
        self._last_error: str = ""

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="screen-stream")
        self._thread.start()

    def stop(self, timeout: float = 1.5) -> None:
        if not self.is_running:
            return
        assert self._stop_event is not None
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None
        self._stop_event = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_latest(self) -> tuple[Image.Image | None, float]:
        with self._lock:
            if self._latest is None:
                return None, 0.0
            return self._latest.copy(), self._latest_ts

    def stats(self) -> dict:
        with self._lock:
            age_ms = max(0.0, (time.time() - self._latest_ts) * 1000.0) if self._latest_ts else 0.0
            return {
                "running": self.is_running,
                "fps_target": self._fps,
                "frames_captured": self._frames_captured,
                "latest_age_ms": round(age_ms, 1),
                "last_error": self._last_error,
            }

    def _loop(self) -> None:
        assert self._stop_event is not None
        interval = 1.0 / self._fps

        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                img = pag.screenshot()
                with self._lock:
                    self._latest = img
                    self._latest_ts = time.time()
                    self._frames_captured += 1
                    self._last_error = ""
            except Exception as e:
                with self._lock:
                    self._last_error = str(e)

            elapsed = time.time() - t0
            remaining = interval - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)
