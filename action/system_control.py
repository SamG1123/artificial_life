"""SystemController — execute shell commands and manage the OS.

All commands pass through the SafetyValidator in the decision layer
before reaching this controller, so dangerous patterns (rm -rf,
shutdown, format, etc.) are already blocked or require confirmation.
This controller only handles execution.
"""

import os
import shutil
import subprocess

from .base import BaseController, UndoEntry
from logging_config import get_logger

log = get_logger("system_ctrl")


class SystemController(BaseController):
    """Runs shell commands and provides basic file / power operations."""

    COMMAND_TIMEOUT = 30  # seconds

    def __init__(self):
        self._last_result: str = ""
        self._undo_stack: list[UndoEntry] = []

    # ── BaseController ───────────────────────────────────────────

    def is_available(self) -> bool:
        return True

    # ── Shell commands ───────────────────────────────────────────

    def run_command(self, command: str) -> str:
        """Execute a shell command and return its output or error."""
        if not command.strip():
            return self._fail("Empty command")

        log.info("Running: %s", command)
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.COMMAND_TIMEOUT,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode == 0:
                msg = stdout or "(success, no output)"
                self._last_result = f"SUCCESS: {msg[:500]}"
            else:
                msg = stderr or stdout or f"exit code {result.returncode}"
                self._last_result = f"FAILED: {msg[:500]}"
        except subprocess.TimeoutExpired:
            self._last_result = "FAILED: command timed out"
        except Exception as e:
            self._last_result = f"FAILED: {e}"

        log.info("%s", self._last_result[:200])
        return self._last_result

    # ── Power management ─────────────────────────────────────────

    def shutdown(self, delay: int = 30) -> str:
        """Schedule a system shutdown (default 30 s delay for safety)."""
        if os.name == "nt":
            cmd = f"shutdown /s /t {delay}"
        else:
            cmd = f"shutdown -h +{max(delay // 60, 1)}"
        return self.run_command(cmd)

    def restart(self, delay: int = 30) -> str:
        """Schedule a system restart."""
        if os.name == "nt":
            cmd = f"shutdown /r /t {delay}"
        else:
            cmd = f"shutdown -r +{max(delay // 60, 1)}"
        return self.run_command(cmd)

    def cancel_shutdown(self) -> str:
        """Cancel a pending shutdown / restart."""
        cmd = "shutdown /a" if os.name == "nt" else "shutdown -c"
        return self.run_command(cmd)

    # ── File operations ──────────────────────────────────────────

    def list_directory(self, path: str = ".") -> list[str]:
        """Return names of items in *path*."""
        try:
            return os.listdir(path)
        except Exception as e:
            self._last_result = f"FAILED: {e}"
            return []

    def read_file(self, path: str, max_chars: int = 5000) -> str:
        """Read a text file (truncated to *max_chars*)."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(max_chars)
            self._last_result = f"Read {len(content)} chars from {path}"
            return content
        except Exception as e:
            return self._fail(f"read_file: {e}")

    def write_file(self, path: str, content: str) -> str:
        """Write *content* to a text file."""
        try:
            existed = os.path.exists(path)
            old_content = None
            if existed:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    old_content = f.read()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            self._last_result = f"Wrote {len(content)} chars to {path}"
            # Register undo
            _path, _old, _existed = path, old_content, existed
            if _existed and _old is not None:
                self._undo_stack.append(UndoEntry(
                    f"restore {_path}",
                    lambda: self._restore_file(_path, _old),
                ))
            else:
                self._undo_stack.append(UndoEntry(
                    f"delete created {_path}",
                    lambda: self._delete_path(_path),
                ))
            return self._last_result
        except Exception as e:
            return self._fail(f"write_file: {e}")

    def copy_file(self, src: str, dst: str) -> str:
        """Copy a file or directory."""
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            else:
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                shutil.copy2(src, dst)
            self._last_result = f"Copied {src} → {dst}"
            return self._last_result
        except Exception as e:
            return self._fail(f"copy_file: {e}")

    def move_file(self, src: str, dst: str) -> str:
        """Move / rename a file or directory."""
        try:
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            shutil.move(src, dst)
            self._last_result = f"Moved {src} → {dst}"
            # Register undo (reverse move)
            _src, _dst = src, dst
            self._undo_stack.append(UndoEntry(
                f"move {_dst} back to {_src}",
                lambda: (shutil.move(_dst, _src), "moved back")[1],
            ))
            return self._last_result
        except Exception as e:
            return self._fail(f"move_file: {e}")

    def file_exists(self, path: str) -> bool:
        return os.path.exists(path)

    @property
    def last_result(self) -> str:
        return self._last_result

    # ── Internals ────────────────────────────────────────────────

    def _fail(self, msg: str) -> str:
        log.warning("%s", msg)
        self._last_result = f"FAILED: {msg}"
        return self._last_result

    # ── Rollback helpers ─────────────────────────────────────────

    def pop_undo(self) -> UndoEntry | None:
        """Pop the most recent undo entry (used by executor on failure)."""
        return self._undo_stack.pop() if self._undo_stack else None

    def clear_undo(self) -> None:
        """Discard all undo entries (goal succeeded, no rollback needed)."""
        self._undo_stack.clear()

    @staticmethod
    def _restore_file(path: str, old_content: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(old_content)
        return f"restored {path}"

    @staticmethod
    def _delete_path(path: str) -> str:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
        return f"deleted {path}"
