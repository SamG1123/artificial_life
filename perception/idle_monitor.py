"""
IdleMonitor — tracks user activity (keyboard/mouse input).

Windows-specific implementation using ctypes to track last input time.
Provides idle duration in seconds and utilities to check if user has been
inactive for a minimum duration.

Typical usage:
    monitor = IdleMonitor()
    monitor.start()
    
    # Check if user is idle >= 1 hour
    if monitor.idle_seconds() >= 3600:
        agent.work_in_foreground()
    else:
        agent.work_in_background()
    
    monitor.stop()
"""

import time
import threading
import logging
from threading import Event, Thread

try:
    import ctypes
    from ctypes import wintypes
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False

log = logging.getLogger("idle_monitor")


class IdleMonitor:
    """Tracks user input activity on Windows."""

    def __init__(self):
        """Initialize the idle monitor.
        
        On non-Windows systems, always report 0 idle time (user active).
        """
        self._stop_event = Event()
        self._idle_ms_cache = 0
        self._cache_update_time = 0
        self._cache_lock = threading.Lock()
        self._thread: Thread | None = None
        self._started = False

        if not WINDOWS_AVAILABLE:
            log.warning("IdleMonitor: ctypes not available, always reporting active")
            return

        # Windows API setup
        try:
            self._GetLastInputInfo = ctypes.windll.kernel32.GetLastInputInfo
            self._GetTickCount = ctypes.windll.kernel32.GetTickCount

            # Success — API is available
            self._windows_ready = True
        except (AttributeError, OSError) as e:
            log.warning("IdleMonitor: Windows API unavailable: %s", e)
            self._windows_ready = False

    def start(self) -> None:
        """Start background thread to periodically cache idle time."""
        if self._started:
            return
        if not WINDOWS_AVAILABLE:
            log.debug("IdleMonitor: skipped (non-Windows)")
            return

        self._started = True
        self._stop_event.clear()
        self._thread = Thread(
            target=self._update_loop,
            daemon=True,
            name="idle-monitor",
        )
        self._thread.start()
        log.info("IdleMonitor started")

    def stop(self) -> None:
        """Stop the background thread."""
        if not self._started:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._started = False
        log.info("IdleMonitor stopped")

    def idle_seconds(self) -> float:
        """Return seconds since last user input (keyboard or mouse).
        
        Returns:
            Idle time in seconds. 0 if user is active or on non-Windows.
        """
        if not WINDOWS_AVAILABLE or not self._started:
            return 0.0

        with self._cache_lock:
            return self._idle_ms_cache / 1000.0

    def is_idle_for(self, seconds: float) -> bool:
        """Check if user has been idle for at least the given duration.
        
        Args:
            seconds: Threshold in seconds (e.g., 3600 for 1 hour).
        
        Returns:
            True if idle_seconds() >= seconds, False otherwise.
        """
        return self.idle_seconds() >= seconds

    def is_active(self) -> bool:
        """Check if user is actively using the computer (idle < 5 seconds)."""
        return self.idle_seconds() < 5.0

    def _update_loop(self) -> None:
        """Background thread: periodically refresh idle time cache."""
        while not self._stop_event.is_set():
            try:
                idle_ms = self._get_idle_ms()
                with self._cache_lock:
                    self._idle_ms_cache = idle_ms
                    self._cache_update_time = time.time()
            except Exception as e:
                log.warning("Failed to query idle time: %s", e)

            # Update every 500ms
            self._stop_event.wait(0.5)

    def _get_idle_ms(self) -> int:
        """Query Windows API for idle milliseconds.
        
        Uses GetLastInputInfo() to retrieve the last input tick count,
        then compares with current tick to compute idle duration.
        """
        if not hasattr(self, "_windows_ready") or not self._windows_ready:
            return 0

        try:
            # Get the last input tick count
            last_input_info = wintypes.DWORD(0)
            if not self._GetLastInputInfo(ctypes.byref(last_input_info)):
                log.warning("GetLastInputInfo failed")
                return 0

            last_input_tick = last_input_info.value
            current_tick = self._GetTickCount()

            # Calculate idle time (handle tick overflow)
            if current_tick >= last_input_tick:
                idle_ms = current_tick - last_input_tick
            else:
                # Tick counter wrapped around (rare but possible)
                idle_ms = (0xFFFFFFFF - last_input_tick) + current_tick

            return idle_ms

        except Exception as e:
            log.error("GetLastInputInfo query failed: %s", e)
            return 0

    def stats(self) -> dict:
        """Return diagnostics dict."""
        return {
            "idle_seconds": self.idle_seconds(),
            "is_active": self.is_active(),
            "is_idle_1hr": self.is_idle_for(3600),
            "windows_available": WINDOWS_AVAILABLE,
            "running": self._started,
        }
