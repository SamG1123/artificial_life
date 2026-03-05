"""
Perception Manager — orchestrates all perception modules and merges
their outputs into a single world-state snapshot.

                ┌────────────┐
                │  Manager   │
                └─────┬──────┘
        ┌─────────┬───┴────┬───────────┐
        ▼         ▼        ▼           ▼
     Screen    Camera    Audio      System
    (OCR+LLM)  (YOLO)   (STT)    (OS state)

Usage:
    mgr = PerceptionManager(vision_model=existing_obj_detection)
    mgr.initialise()
    snapshot = mgr.capture_all()           # full world state
    snapshot = mgr.capture(["screen"])      # selective

The manager can also run a continuous background loop that keeps a
fresh world-state dict available for the brain at any time.
"""

import time
import threading
from typing import Sequence

from .screen import ScreenPerception
from .camera import CameraPerception
from .audio import AudioPerception
from .system import SystemPerception
from .base import PerceptionData


class PerceptionManager:
    """Gathers structured data from every perception module and
    produces a unified world-state dict."""

    def __init__(self, vision_model=None, camera_index: int = 0,
                 world_state_manager=None):
        """
        Args:
            vision_model: image_processing.ObjectDetection instance
                          shared by screen + camera modules.
            camera_index: OpenCV camera device index.
            world_state_manager: WorldStateManager instance — if given,
                                 every capture automatically pushes data
                                 into the world state.
        """
        self.screen = ScreenPerception(vision_model=vision_model)
        self.camera = CameraPerception(
            vision_model=vision_model, camera_index=camera_index
        )
        self.audio = AudioPerception()
        self.system = SystemPerception()

        self._modules = {
            "screen": self.screen,
            "camera": self.camera,
            "audio": self.audio,
            "system": self.system,
        }

        # Latest world-state (updated by background loop or manual capture)
        self._world_state: dict = {}
        self._lock = threading.Lock()
        self._bg_thread: threading.Thread | None = None
        self._wsm = world_state_manager

    # ── Lifecycle ────────────────────────────────────────────────

    def initialise(self) -> None:
        """Initialise all perception modules."""
        for name, mod in self._modules.items():
            try:
                mod.initialise()
                print(f"[Perception] {name} initialised.")
            except Exception as e:
                print(f"[Perception] {name} init failed: {e}")

    def shutdown(self) -> None:
        """Shut down all perception modules."""
        for name, mod in self._modules.items():
            try:
                mod.shutdown()
            except Exception:
                pass

    # ── Capture ──────────────────────────────────────────────────

    def capture_all(self) -> dict:
        """Run every module and return a merged world-state dict.

        Returns:
            {
              "timestamp": float,
              "screen":  { ...PerceptionData... },
              "camera":  { ...PerceptionData... },
              "audio":   { ...PerceptionData... },
              "system":  { ...PerceptionData... },
              "summary": "combined one-liner",
            }
        """
        return self.capture(list(self._modules.keys()))

    def capture(self, sources: Sequence[str]) -> dict:
        """Run only the named modules and return a world-state dict.

        Args:
            sources: subset of ["screen", "camera", "audio", "system"]
        """
        snapshot = {"timestamp": time.time()}
        summaries = []

        for name in sources:
            mod = self._modules.get(name)
            if mod is None:
                continue
            try:
                data = mod.capture()
                snapshot[name] = data
                if data.get("summary"):
                    summaries.append(data["summary"])
            except Exception as e:
                snapshot[name] = PerceptionData(
                    source=name,
                    data={"error": str(e)},
                    confidence=0.0,
                    summary=f"{name} error: {e}",
                )

        snapshot["summary"] = " | ".join(summaries) if summaries else ""

        with self._lock:
            self._world_state = snapshot

        # Push into WorldStateManager if available
        if self._wsm is not None:
            try:
                self._wsm.update_perception(snapshot)
            except Exception:
                pass

        return snapshot

    # ── Quick accessors ──────────────────────────────────────────

    def get_world_state(self) -> dict:
        """Return the most recent world-state snapshot (thread-safe)."""
        with self._lock:
            return dict(self._world_state)

    # ── Background loop ──────────────────────────────────────────

    def start_background_loop(
        self,
        stop_event: threading.Event,
        interval: float = 5.0,
        sources: Sequence[str] | None = None,
    ):
        """Start a daemon thread that refreshes the world state
        at the given interval.

        Args:
            stop_event: threading.Event — set to stop the loop.
            interval: seconds between captures.
            sources: which modules to poll (default: all except audio,
                     since audio has its own listen loop).
        """
        if sources is None:
            sources = ["screen", "system"]  # lightweight defaults

        def _loop():
            while not stop_event.is_set():
                try:
                    self.capture(sources)
                except Exception as e:
                    print(f"[Perception] Background capture error: {e}")
                stop_event.wait(interval)

        self._bg_thread = threading.Thread(target=_loop, daemon=True)
        self._bg_thread.start()
        print(f"[Perception] Background loop started (every {interval}s, sources={list(sources)}).")
