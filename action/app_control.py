"""AppController — launch, close, and switch desktop applications.

Wraps pyautogui + pygetwindow for window management and os.startfile /
subprocess for launching apps.
"""

import os
import time
import subprocess
import shutil

import pyautogui as pag

try:
    import pygetwindow as gw
except ImportError:
    gw = None

from .base import BaseController, UndoEntry
from logging_config import get_logger

log = get_logger("app_ctrl")


class AppController(BaseController):
    """Manages desktop applications: launch, close, switch, list."""

    def __init__(self):
        self._last_result: str = ""
        self._undo_stack: list[UndoEntry] = []

    # ── BaseController ───────────────────────────────────────────

    def is_available(self) -> bool:
        return True  # always available on a desktop OS

    # ── Public API ───────────────────────────────────────────────

    def open_app(self, app_name: str) -> str:
        """Launch an application by name. Returns a status string."""
        if not app_name:
            return self._fail("open_app: empty app_name")

        app = app_name.strip()
        log.info("Opening: %s", app_name)
        try:
            launched = self._launch_windows_app(app) if os.name == "nt" else self._launch_posix_app(app)
            if not launched:
                return self._fail(f"Failed to resolve {app_name}")
            time.sleep(2)

            # Try to maximise + bring to front
            self._focus_window(app)
            self._last_result = f"SUCCESS: opened {app_name}"
            # Register undo: close the app we just opened
            _name = app_name
            self._undo_stack.append(UndoEntry(
                f"close {_name}",
                lambda: self.close_app(_name),
            ))
            return self._last_result
        except Exception as e:
            return self._fail(f"Failed to open {app_name}: {e}")

    def _launch_posix_app(self, app: str) -> bool:
        try:
            subprocess.Popen([app])
            return True
        except Exception:
            return False

    def _launch_windows_app(self, app: str) -> bool:
        resolved = self._resolve_windows_app(app)
        try:
            if resolved.startswith(("steam://", "http://", "https://")):
                os.startfile(resolved)
            elif os.path.isfile(resolved):
                subprocess.Popen([resolved], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif shutil.which(resolved):
                subprocess.Popen([resolved], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.startfile(resolved)
            return True
        except Exception as e:
            log.warning("Windows launch failed for %s -> %s: %s", app, resolved, e)
            return False

    def _resolve_windows_app(self, app: str) -> str:
        target = app.strip().lower()

        if target in {"steam", "steam app", "steam application"}:
            candidates = [
                os.path.expandvars(r"%ProgramFiles(x86)%\Steam\Steam.exe"),
                os.path.expandvars(r"%ProgramFiles%\Steam\Steam.exe"),
                os.path.expandvars(r"%LocalAppData%\Steam\Steam.exe"),
            ]
            for candidate in candidates:
                if os.path.isfile(candidate):
                    return candidate
            return "steam://open/main"

        if target.endswith(".exe") or os.path.sep in target:
            return app

        # Try PATH first, then rely on shell launch for registered app aliases.
        which = shutil.which(app)
        if which:
            return which

        return app

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
        log.warning("%s", msg)
        self._last_result = f"FAILED: {msg}"
        return self._last_result

    # ── Rollback helpers ─────────────────────────────────────────

    def pop_undo(self) -> UndoEntry | None:
        """Pop the most recent undo entry."""
        return self._undo_stack.pop() if self._undo_stack else None

    def clear_undo(self) -> None:
        """Discard all undo entries."""
        self._undo_stack.clear()
