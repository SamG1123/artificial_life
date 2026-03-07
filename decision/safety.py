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

import re
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


class SafetyValidator:
    """Validates actions before execution to prevent dangerous operations."""

    def __init__(self):
        self._blocked = 0
        self._confirmed = 0
        self._patterns = [re.compile(p, re.IGNORECASE) for p in _DANGEROUS_COMMANDS]

    # ── public API ──────────────────────────────────────────────────

    def validate(self, action_dict: dict) -> tuple:
        """Check an action for safety.

        Returns (SafetyVerdict, RiskLevel, reason_str).
        """
        action = action_dict.get("action", "done")

        if action == "done":
            return SafetyVerdict.ALLOW, RiskLevel.SAFE, ""

        if action == "run_command":
            return self._check_command(action_dict.get("command", ""))

        if action == "download":
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

    @property
    def stats(self) -> dict:
        return {"blocked": self._blocked, "confirmed": self._confirmed}

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
