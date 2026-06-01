"""ExperienceLogger — records (world_state, action, result) tuples.

Every action the agent executes in the environment gets logged as a
structured experience entry.  Entries are written to a JSONL file
(one JSON object per line) for fast append-only I/O and easy streaming
into DatasetBuilder later.

Schema per line:
{
    "ts":           float        — unix timestamp
    "goal":         str          — high-level goal being pursued
    "step":         int          — step index within the goal
    "world_state":  dict         — compact world snapshot at decision time
    "action":       dict         — the action dict sent to dispatch
    "result":       str          — outcome string from the executor
    "success":      bool         — whether the action succeeded
    "domain":       str          — action domain (browser/app/system/desktop/…)
    "risk":         str          — risk level from SafetyValidator
    "plan_step":    dict | None  — plan Step.to_dict() if plan exists
}
"""

import json
import os
import time
import threading
from pathlib import Path

from memory.compressed_storage import append_line, iter_lines, migrate_if_needed


class ExperienceLogger:
    """Append-only logger for state→action→result experiences."""

    def __init__(self, store_dir: str = "memory_store"):
        self._dir = Path(store_dir) / "experiences"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Active log file — rotated per day
        self._current_file: Path | None = None
        self._current_day: str = ""

        # Counters
        self._total_logged = 0
        self._session_logged = 0

    # ── Public API ───────────────────────────────────────────────

    def log(
        self,
        goal: str,
        step: int,
        world_state: dict,
        action: dict,
        result: str,
        success: bool,
        domain: str = "",
        risk: str = "",
        plan_step: dict | None = None,
    ):
        """Append one experience entry to the log file."""
        entry = {
            "ts": time.time(),
            "goal": goal,
            "step": step,
            "world_state": self._compact_state(world_state),
            "action": action,
            "result": result[:500],
            "success": success,
            "domain": domain,
            "risk": risk,
            "plan_step": plan_step,
        }
        self._write(entry)

    def log_episode_summary(
        self,
        goal: str,
        outcome: str,
        total_steps: int,
        success: bool,
        action_history: list,
    ):
        """Write a summary entry marking end-of-episode."""
        entry = {
            "ts": time.time(),
            "type": "episode_summary",
            "goal": goal,
            "outcome": outcome,
            "total_steps": total_steps,
            "success": success,
            "action_types": [a.get("action", {}).get("action", "?")
                             for a in action_history],
        }
        self._write(entry)

    def log_user_correction(self, input_text: str,
                            previous_intent: str,
                            corrected_intent: str):
        """Log explicit user corrections for supervised correction learning."""
        entry = {
            "ts": time.time(),
            "type": "user_correction",
            "input_text": (input_text or "")[:400],
            "previous_intent": (previous_intent or "")[:300],
            "corrected_intent": (corrected_intent or "")[:300],
        }
        self._write(entry)

    def get_log_files(self) -> list[Path]:
        """Return all .jsonl.gz log files sorted newest-first."""
        files = sorted(self._dir.glob("*.jsonl.gz"), reverse=True)
        # Also pick up any un-migrated plain .jsonl files
        plain = sorted(self._dir.glob("*.jsonl"), reverse=True)
        seen = {f.stem for f in files}  # stem strips .gz only — need name minus .gz
        for p in plain:
            if p.name not in seen:
                files.append(p)
        return files

    def iter_entries(self, days: int = 7):
        """Yield entries from the last *days* log files."""
        files = self.get_log_files()
        for f in files[:days]:
            yield from iter_lines(str(f))

    @property
    def stats(self) -> dict:
        return {
            "total_logged": self._total_logged,
            "session_logged": self._session_logged,
            "log_files": len(self.get_log_files()),
        }

    def close(self):
        pass  # No persistent handle with compressed storage

    # ── Internals ────────────────────────────────────────────────

    def _write(self, entry: dict):
        with self._lock:
            today = time.strftime("%Y-%m-%d")
            if today != self._current_day:
                self._current_day = today
                self._current_file = self._dir / f"exp_{today}.jsonl"
            append_line(str(self._current_file), entry)
            self._total_logged += 1
            self._session_logged += 1

    def _get_handle(self):
        pass  # Kept for API compat; no longer used

    @staticmethod
    def _compact_state(state: dict) -> dict:
        """Trim world_state to essential fields for training data size."""
        if not state:
            return {}
        compact = {}
        # Keep screen summary but not raw element lists
        if "screen" in state:
            screen = state["screen"]
            if isinstance(screen, dict):
                compact["screen_summary"] = screen.get("summary", "")
        # Keep camera summary
        if "camera" in state:
            camera = state["camera"]
            if isinstance(camera, dict):
                compact["camera_summary"] = camera.get("summary", "")
        # Keep emotion state
        if "emotion" in state:
            compact["emotion"] = state["emotion"]
        # Keep system info
        if "system" in state:
            system = state["system"]
            if isinstance(system, dict):
                compact["system_summary"] = system.get("summary", "")
        # Keep top-level summary
        if "summary" in state:
            compact["summary"] = state["summary"]

        return compact
