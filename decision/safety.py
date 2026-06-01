"""SafetyValidator — gates every action before execution.

Risk levels:
    SAFE      – no risk (e.g. done)
    LOW       – normal UI interaction
    MEDIUM    – app launch, chained commands
    HIGH      – downloads, elevated commands
    CRITICAL  – destructive system commands (blocked)

Verdicts:
    ALLOW   – proceed immediately
    CONFIRM – ask user for voice/console confirmation first
    BLOCK   – refuse to execute
"""

import json
import os
import re
import time
from enum import Enum


class RiskLevel(Enum):
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SafetyVerdict(Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    BLOCK = "block"


# ── Dangerous shell-command patterns ────────────────────────────────
_DANGEROUS_COMMANDS = [
    r"\brm\s+(-rf?|--recursive)",
    r"\bdel\s+/[sfq]",
    r"\bformat\b",
    r"\bshutdown\b",
    r"\brestart\b",
    r"\breboot\b",
    r"\bmkfs\b",
    r"\bdd\b\s+if=",
    r"\brmdir\s+/s",
    r"\breg\s+delete",
    r"\bnet\s+(user|stop)\b",
    r"\btakeown\b",
    r"\bicacls\b.*(/grant|/deny)",
    r"Remove-Item.*-Recurse",
    r"Stop-Service",
    r"Stop-Process",
    r"Restart-Computer",
    r"Clear-Content",
]

# ── Dangerous hotkey combos ─────────────────────────────────────────
_DANGEROUS_HOTKEYS = [
    ({"alt", "f4"}, "close window"),
    ({"win", "l"}, "lock computer"),
    ({"ctrl", "shift", "delete"}, "security screen"),
]

# Auto-allow threshold: after N user confirmations of the same pattern,
# bypass the confirmation prompt.
_AUTO_ALLOW_THRESHOLD = 3


class SafetyValidator:
    """Validates actions before execution to prevent dangerous operations.

    Supports context-aware risk assessment and remembers user confirmations
    so frequently-approved patterns don't nag repeatedly.
    """

    def __init__(self, store_dir: str = "memory_store"):
        self._blocked = 0
        self._confirmed = 0
        self._patterns = [re.compile(p, re.IGNORECASE) for p in _DANGEROUS_COMMANDS]

        # Confirmation memory: {action_signature: confirmation_count}
        self._confirm_cache: dict[str, int] = {}
        self._decision_log: list[dict] = []

        # Persistent confirmation cache path
        self._cache_path = os.path.join(store_dir, "safety_confirmations.json")
        self._load_cache()

    # ── public API ──────────────────────────────────────────────────

    def validate(self, action_dict: dict, *,
                 goal_context: str = "",
                 world_context: str = "") -> tuple:
        """Check an action for safety.

        Parameters
        ----------
        action_dict : dict
            The action to validate.
        goal_context : str
            The current goal being pursued (for context-aware decisions).
        world_context : str
            Summary of the current world state.

        Returns (SafetyVerdict, RiskLevel, reason_str).
        """
        action = action_dict.get("action", "done")

        if action == "done":
            return SafetyVerdict.ALLOW, RiskLevel.SAFE, ""

        if action == "run_command":
            verdict, risk, reason = self._check_command(action_dict.get("command", ""))
            # Context-aware downgrade: if the goal clearly involves
            # the action type, reduce risk for CONFIRM verdicts
            if verdict == SafetyVerdict.CONFIRM and goal_context:
                sig = self._action_signature(action_dict)
                if self._is_auto_allowed(sig):
                    self._log_decision(action_dict, SafetyVerdict.ALLOW,
                                       risk, "auto-allowed (confirmed before)",
                                       goal_context)
                    return SafetyVerdict.ALLOW, risk, "auto-allowed"
            return verdict, risk, reason

        if action == "download":
            sig = self._action_signature(action_dict)
            if self._is_auto_allowed(sig):
                self._log_decision(action_dict, SafetyVerdict.ALLOW,
                                   RiskLevel.HIGH, "auto-allowed download",
                                   goal_context)
                return SafetyVerdict.ALLOW, RiskLevel.HIGH, "auto-allowed download"
            return SafetyVerdict.CONFIRM, RiskLevel.HIGH, "File download requires confirmation"

        if action == "hotkey":
            return self._check_hotkey(action_dict.get("keys", []))

        if action == "open_app":
            return SafetyVerdict.ALLOW, RiskLevel.MEDIUM, "Opening application"

        # Standard low-risk UI actions
        if action in (
            "click", "type", "press_key", "scroll", "open_browser",
            "navigate", "go_back", "mouse_click_xy", "mouse_drag", "draw_plan",
        ):
            return SafetyVerdict.ALLOW, RiskLevel.LOW, ""

        # Unknown action — allow but flag
        return SafetyVerdict.ALLOW, RiskLevel.MEDIUM, f"Unknown action: {action}"

    def record_confirmation(self, action_dict: dict, confirmed: bool) -> None:
        """Record a user confirmation/rejection for learning.

        After _AUTO_ALLOW_THRESHOLD confirmations of the same pattern,
        future identical actions will auto-allow.
        """
        sig = self._action_signature(action_dict)
        if confirmed:
            self._confirmed += 1
            self._confirm_cache[sig] = self._confirm_cache.get(sig, 0) + 1
            self._save_cache()
        else:
            # User rejected — remove from cache if present
            self._confirm_cache.pop(sig, None)
            self._save_cache()

    @property
    def stats(self) -> dict:
        return {
            "blocked": self._blocked,
            "confirmed": self._confirmed,
            "auto_allowed_patterns": sum(
                1 for v in self._confirm_cache.values()
                if v >= _AUTO_ALLOW_THRESHOLD
            ),
            "recent_decisions": len(self._decision_log),
        }

    @property
    def decision_log(self) -> list[dict]:
        """Recent safety decisions (for learning/analysis)."""
        return list(self._decision_log[-100:])

    # ── private helpers ─────────────────────────────────────────────

    def _check_command(self, command: str) -> tuple:
        if not command.strip():
            return SafetyVerdict.BLOCK, RiskLevel.SAFE, "Empty command"

        for pat in self._patterns:
            if pat.search(command):
                self._blocked += 1
                return (
                    SafetyVerdict.BLOCK,
                    RiskLevel.CRITICAL,
                    f"Blocked dangerous command: {pat.pattern}",
                )

        # Chained commands are riskier
        if "|" in command or "&&" in command or ";" in command:
            return (
                SafetyVerdict.CONFIRM,
                RiskLevel.MEDIUM,
                "Command uses chaining/pipes",
            )

        return SafetyVerdict.ALLOW, RiskLevel.LOW, ""

    def _check_hotkey(self, keys: list) -> tuple:
        if not keys:
            return SafetyVerdict.BLOCK, RiskLevel.SAFE, "Empty hotkey"

        normalised = {k.lower() for k in keys}
        for combo, description in _DANGEROUS_HOTKEYS:
            if combo == normalised:
                self._blocked += 1
                return (
                    SafetyVerdict.BLOCK,
                    RiskLevel.HIGH,
                    f"Blocked dangerous hotkey: {description}",
                )

        return SafetyVerdict.ALLOW, RiskLevel.LOW, ""

    @staticmethod
    def _action_signature(action_dict: dict) -> str:
        """Create a stable signature for an action type (ignoring volatile fields)."""
        action = action_dict.get("action", "")
        # For commands, normalize whitespace
        if action == "run_command":
            cmd = action_dict.get("command", "").strip()
            # Extract the command verb (first token)
            verb = cmd.split()[0] if cmd else ""
            return f"run_command:{verb}"
        if action == "download":
            return "download"
        if action == "hotkey":
            keys = sorted(k.lower() for k in action_dict.get("keys", []))
            return f"hotkey:{'+'.join(keys)}"
        return action

    def _is_auto_allowed(self, signature: str) -> bool:
        """Check if this action signature has enough confirmations to auto-allow."""
        return self._confirm_cache.get(signature, 0) >= _AUTO_ALLOW_THRESHOLD

    def _log_decision(self, action_dict: dict, verdict: SafetyVerdict,
                      risk: RiskLevel, reason: str,
                      goal_context: str = "") -> None:
        """Record a safety decision for analysis/learning."""
        entry = {
            "timestamp": time.time(),
            "action": action_dict.get("action", ""),
            "verdict": verdict.value,
            "risk": risk.name,
            "reason": reason,
            "goal": goal_context[:100] if goal_context else "",
        }
        self._decision_log.append(entry)
        # Keep bounded
        if len(self._decision_log) > 500:
            self._decision_log = self._decision_log[-300:]

    def _load_cache(self) -> None:
        if not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                self._confirm_cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._confirm_cache = {}

    def _save_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            tmp = self._cache_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._confirm_cache, f, indent=2)
            os.replace(tmp, self._cache_path)
        except OSError:
            pass
