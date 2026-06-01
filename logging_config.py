"""
Centralized logging configuration for the Artificial Life project.

Replaces scattered print() calls with structured, rotating log files.
Each subsystem gets its own logger that writes to both console and file.

Usage:
    from logging_config import get_logger
    logger = get_logger("brain")
    logger.info("Cognitive loop starting...")
    logger.warning("Energy low: %d%%", energy)
    logger.error("Goal execution failed: %s", e)
"""

import os
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Shared format
_FORMAT = "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Max 5 MB per file, keep 3 backups → ~20 MB max per subsystem
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

_formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

# Cache so we don't create duplicate handlers
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Get (or create) a named logger with console + rotating file output.

    Args:
        name:  Subsystem name (e.g. "brain", "executor", "memory").
               Becomes the log filename and the logger name.
        level: Minimum log level (default DEBUG — everything is captured
               on disk; console shows INFO+).
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(f"al.{name}")
    logger.setLevel(level)
    logger.propagate = False

    # ── File handler (DEBUG+) — captures everything ──────────────
    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, f"{name}.log"),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_formatter)
    logger.addHandler(fh)

    # ── Console handler (INFO+) — keeps terminal readable ────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_formatter)
    logger.addHandler(ch)

    _loggers[name] = logger
    return logger
