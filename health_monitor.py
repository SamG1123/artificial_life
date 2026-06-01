"""
Health Monitor — periodic system health snapshots for diagnostics.

Runs on a background thread, collecting:
  * Tick latency (time between cognitive ticks)
  * Memory usage (RAM)
  * Queue depths (TTS, goal)
  * Thread liveness
  * Brain state & energy

Exposes a `.snapshot()` method that returns the latest health data
(used by the status API and future remote dashboard).
"""

import os
import time
import threading

from logging_config import get_logger

log = get_logger("health")

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class HealthMonitor:
    """Collects and exposes system health metrics."""

    def __init__(self, brain=None, tts_queue=None, goal_queue=None,
                 threads: list[threading.Thread] | None = None,
                 interval: float = 30.0):
        """
        Args:
            brain: CognitiveBrain reference (for state/energy).
            tts_queue: The TTS output queue.
            goal_queue: The goal input queue.
            threads: List of managed threads to monitor.
            interval: Seconds between health snapshots.
        """
        self._brain = brain
        self._tts_queue = tts_queue
        self._goal_queue = goal_queue
        self._threads = threads or []
        self._interval = interval
        self._last_snapshot: dict = {}
        self._lock = threading.Lock()
        self._process = psutil.Process(os.getpid()) if _HAS_PSUTIL else None

    # ── Public API ───────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return the latest health snapshot (thread-safe)."""
        with self._lock:
            return dict(self._last_snapshot)

    def run(self, stop_event: threading.Event) -> None:
        """Background loop — call from a daemon thread."""
        log.info("Health monitor started (interval=%.0fs)", self._interval)
        while not stop_event.is_set():
            try:
                snap = self._collect()
                with self._lock:
                    self._last_snapshot = snap
                log.debug(
                    "Health: state=%s energy=%.0f ram=%.1fMB cpu=%.1f%% threads=%d/%d",
                    snap.get("brain_state", "?"),
                    snap.get("energy", 0),
                    snap.get("ram_mb", 0),
                    snap.get("cpu_percent", 0),
                    snap.get("threads_alive", 0),
                    snap.get("threads_total", 0),
                )
            except Exception as e:
                log.warning("Health collection error: %s", e)
            stop_event.wait(self._interval)
        log.info("Health monitor stopped.")

    # ── Internals ────────────────────────────────────────────────

    def _collect(self) -> dict:
        snap: dict = {"ts": time.time()}

        # Brain metrics
        if self._brain is not None:
            snap["brain_state"] = self._brain.state.name
            snap["energy"] = self._brain._energy
            snap["current_goal"] = self._brain.current_goal
            snap["idle_cycles"] = self._brain._idle_cycles

        # Queue depths
        if self._tts_queue is not None:
            snap["tts_queue_size"] = self._tts_queue.qsize()
        if self._goal_queue is not None:
            snap["goal_queue_size"] = self._goal_queue.qsize()

        # Thread liveness
        alive = sum(1 for t in self._threads if t.is_alive())
        snap["threads_alive"] = alive
        snap["threads_total"] = len(self._threads)
        snap["threads"] = {t.name: t.is_alive() for t in self._threads}

        # System resources (if psutil available)
        if self._process is not None:
            try:
                mem = self._process.memory_info()
                snap["ram_mb"] = mem.rss / (1024 * 1024)
                snap["cpu_percent"] = self._process.cpu_percent(interval=0.1)
            except Exception:
                snap["ram_mb"] = 0
                snap["cpu_percent"] = 0
        else:
            snap["ram_mb"] = 0
            snap["cpu_percent"] = 0

        return snap
