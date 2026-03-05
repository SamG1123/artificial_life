"""
System Perception — perceives the operating-system environment.

Responsibilities:
  * List running applications / foreground window
  * Gather OS resource usage (CPU, RAM, disk)
  * Detect system events (low battery, new USB device, etc.)
  * Output a structured snapshot of the system state

This is new functionality — no existing module covers it.
"""

import os
import time
import platform
import subprocess
import threading

from .base import PerceptionModule, PerceptionData

try:
    import pygetwindow as gw
except ImportError:
    gw = None

try:
    import psutil
except ImportError:
    psutil = None


class SystemPerception(PerceptionModule):
    """Perceives the state of the host operating system."""

    def __init__(self):
        self._lock = threading.Lock()
        self._os_name = platform.system()

    # ── PerceptionModule interface ───────────────────────────────

    def initialise(self) -> None:
        pass  # no special resources to acquire

    def capture(self) -> dict:
        """Gather a snapshot of the system state."""
        data = {
            "os": self._os_name,
            "active_window": self._active_window(),
            "open_windows": self._open_windows(),
            "resources": self._resources(),
            "battery": self._battery(),
        }
        summary = self._build_summary(data)
        return PerceptionData(
            source="system",
            data=data,
            confidence=1.0,
            summary=summary,
        )

    def shutdown(self) -> None:
        pass

    # ── Data collectors ──────────────────────────────────────────

    @staticmethod
    def _active_window() -> dict:
        if gw is None:
            return {}
        try:
            win = gw.getActiveWindow()
            if win and win.title.strip():
                return {
                    "title": win.title,
                    "position": {"x": win.left, "y": win.top},
                    "size": {"w": win.width, "h": win.height},
                }
        except Exception:
            pass
        return {}

    @staticmethod
    def _open_windows() -> list:
        if gw is None:
            return []
        try:
            return [
                w.title
                for w in gw.getAllWindows()
                if w.title.strip() and w.visible and len(w.title.strip()) > 1
            ][:20]
        except Exception:
            return []

    @staticmethod
    def _resources() -> dict:
        if psutil is None:
            return {}
        try:
            mem = psutil.virtual_memory()
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "ram_total_gb": round(mem.total / (1024 ** 3), 1),
                "ram_used_gb": round(mem.used / (1024 ** 3), 1),
                "ram_percent": mem.percent,
                "disk_percent": psutil.disk_usage("/").percent
                if os.name != "nt"
                else psutil.disk_usage("C:\\").percent,
            }
        except Exception:
            return {}

    @staticmethod
    def _battery() -> dict | None:
        if psutil is None:
            return None
        try:
            bat = psutil.sensors_battery()
            if bat is None:
                return None  # desktop — no battery
            return {
                "percent": bat.percent,
                "plugged_in": bat.power_plugged,
                "seconds_left": bat.secsleft if bat.secsleft >= 0 else None,
            }
        except Exception:
            return None

    # ── Summary ──────────────────────────────────────────────────

    @staticmethod
    def _build_summary(data: dict) -> str:
        parts = []
        win = data.get("active_window", {})
        if win.get("title"):
            parts.append(f"Active: {win['title']}")
        n = len(data.get("open_windows", []))
        if n:
            parts.append(f"{n} windows open")
        res = data.get("resources", {})
        if res.get("cpu_percent") is not None:
            parts.append(f"CPU {res['cpu_percent']}%")
        if res.get("ram_percent") is not None:
            parts.append(f"RAM {res['ram_percent']}%")
        bat = data.get("battery")
        if bat:
            plug = "plugged" if bat["plugged_in"] else "battery"
            parts.append(f"{bat['percent']}% ({plug})")
        return " | ".join(parts) if parts else "System OK"
