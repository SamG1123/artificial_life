"""
State Checkpoint — persist and recover brain cognitive state across restarts.

Saves a JSON snapshot of the brain's volatile state (energy, mood,
conversation buffer, current goal, idle cycles, etc.) to disk every
N ticks.  On startup the AgentController checks for a checkpoint and
restores the brain to where it left off.

File layout:
    memory_store/checkpoint.json   — latest snapshot (atomic write via rename)
"""

import os
import json
import time
import tempfile

from logging_config import get_logger

log = get_logger("checkpoint")

_DEFAULT_PATH = os.path.join("memory_store", "checkpoint.json")


class StateCheckpoint:
    """Serialize / restore CognitiveBrain state."""

    def __init__(self, path: str = _DEFAULT_PATH, interval_ticks: int = 6):
        """
        Args:
            path: File path for the checkpoint JSON.
            interval_ticks: How often (in ticks) to auto-save.  At IDLE
                            tick rate of 5 s this means every ~30 s.
        """
        self._path = path
        self._interval = interval_ticks
        self._ticks = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # ── Save ─────────────────────────────────────────────────────

    def tick(self, brain) -> None:
        """Call once per cognitive tick; auto-saves when interval reached."""
        self._ticks += 1
        if self._ticks >= self._interval:
            self._ticks = 0
            self.save(brain)

    def save(self, brain) -> None:
        """Write a snapshot to disk (atomic via tmp + rename)."""
        try:
            snap = {
                "ts": time.time(),
                "state": brain.state.name,
                "energy": brain._energy,
                "idle_cycles": brain._idle_cycles,
                "conversation_buffer": list(brain._conversation_buffer),
                "current_goal": brain.current_goal,
                "ticks_since_compression": brain._ticks_since_compression,
                "ticks_since_improvement": brain._ticks_since_improvement,
                "nightly_trained_this_sleep": brain._nightly_trained_this_sleep,
            }
            dir_name = os.path.dirname(self._path) or "."
            fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(snap, f)
                os.replace(tmp, self._path)
            except BaseException:
                # Clean up the temp file on failure
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            log.debug("Checkpoint saved (energy=%.0f, state=%s)", brain._energy, brain.state.name)
        except Exception as e:
            log.warning("Checkpoint save failed: %s", e)

    # ── Restore ──────────────────────────────────────────────────

    def load(self) -> dict | None:
        """Load the latest checkpoint from disk, or None if absent / corrupt."""
        if not os.path.exists(self._path):
            return None
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            age = time.time() - data.get("ts", 0)
            log.info("Checkpoint found (%.0f s old, state=%s, energy=%.0f)",
                     age, data.get("state"), data.get("energy", 0))
            return data
        except Exception as e:
            log.warning("Checkpoint load failed: %s", e)
            return None

    def restore(self, brain, data: dict) -> None:
        """Apply a loaded checkpoint dict onto a CognitiveBrain."""
        from brain import CognitiveState

        brain._energy = data.get("energy", brain._energy)
        brain._idle_cycles = data.get("idle_cycles", 0)
        brain._conversation_buffer = data.get("conversation_buffer", [])
        brain._ticks_since_compression = data.get("ticks_since_compression", 0)
        brain._ticks_since_improvement = data.get("ticks_since_improvement", 0)
        brain._nightly_trained_this_sleep = data.get("nightly_trained_this_sleep", False)

        # Restore cognitive state (but don't resume EXECUTING — the goal is lost)
        saved_state = data.get("state", "IDLE")
        if saved_state == "SLEEPING":
            brain.state = CognitiveState.SLEEPING
        else:
            brain.state = CognitiveState.IDLE

        log.info("Brain state restored: energy=%.0f, state=%s, conv_buffer=%d entries",
                 brain._energy, brain.state.name, len(brain._conversation_buffer))

    # ── Cleanup ──────────────────────────────────────────────────

    def clear(self) -> None:
        """Delete the checkpoint file (e.g. after clean shutdown)."""
        try:
            if os.path.exists(self._path):
                os.remove(self._path)
                log.debug("Checkpoint cleared.")
        except OSError:
            pass
