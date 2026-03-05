"""AppController — launch, close, and switch desktop applications.

Wraps pyautogui + pygetwindow for window management and os.startfile /
subprocess for launching apps.
"""

import os
import time
import subprocess

import pyautogui as pag

try:
    import pygetwindow as gw
except ImportError:
    gw = None

from .base import BaseController


class AppController(BaseController):
    """Manages desktop applications: launch, close, switch, list."""

    def __init__(self):
        self._last_result: str = ""

    # ── BaseController ───────────────────────────────────────────

    def is_available(self) -> bool:
        return True  # always available on a desktop OS

    # ── Public API ───────────────────────────────────────────────

    def open_app(self, app_name: str) -> str:
        """Launch an application by name. Returns a status string."""
        if not app_name:
            return self._fail("open_app: empty app_name")

        app = app_name.strip().lower()
        print(f"  [AppCtrl] Opening: {app_name}")
        try:
            if os.name == "nt":
                os.startfile(app)
            else:
                subprocess.Popen([app])
            time.sleep(2)

            # Try to maximise + bring to front
            self._focus_window(app)
            self._last_result = f"SUCCESS: opened {app_name}"
            return self._last_result
        except Exception as e:
            return self._fail(f"Failed to open {app_name}: {e}")

    def close_app(self, app_name: str) -> str:
        """Close a window whose title contains *app_name*."""
        if gw is None:
            return self._fail("pygetwindow not available")

        wins = self._find_windows(app_name)
        if not wins:
            return self._fail(f"No window matching '{app_name}'")

        try:
            wins[0].close()
            time.sleep(0.5)
            self._last_result = f"Closed '{wins[0].title}'"
            return self._last_result
        except Exception as e:
            return self._fail(f"close_app error: {e}")

    def switch_app(self, app_name: str) -> str:
        """Bring a running application to the foreground."""
        if gw is None:
            return self._fail("pygetwindow not available")

        wins = self._find_windows(app_name)
        if not wins:
            return self._fail(f"No window matching '{app_name}'")

        try:
            win = wins[0]
            if win.isMinimized:
                win.restore()
            win.activate()
            time.sleep(0.3)
            self._last_result = f"Switched to '{win.title}'"
            return self._last_result
        except Exception as e:
            return self._fail(f"switch_app error: {e}")

    def list_windows(self) -> list[str]:
        """Return titles of all visible windows."""
        if gw is None:
            return []
        return [w.title for w in gw.getAllWindows() if w.title.strip()]

    @property
    def last_result(self) -> str:
        return self._last_result

    # ── Internals ────────────────────────────────────────────────

    def _find_windows(self, name: str) -> list:
        """Case-insensitive partial-match on window titles."""
        if gw is None:
            return []
        target = name.strip().lower()
        exact = gw.getWindowsWithTitle(name)
        if exact:
            return exact
        return [w for w in gw.getAllWindows()
                if target in w.title.lower() and w.title.strip()]

    def _focus_window(self, app: str):
        """Best-effort maximise + activate."""
        if gw is None:
            pag.hotkey("win", "up")
            return
        try:
            time.sleep(1)
            wins = self._find_windows(app)
            if wins:
                wins[0].maximize()
                wins[0].activate()
        except Exception:
            pag.hotkey("win", "up")

    def _fail(self, msg: str) -> str:
        print(f"  [AppCtrl] {msg}")
        self._last_result = f"FAILED: {msg}"
        return self._last_result
