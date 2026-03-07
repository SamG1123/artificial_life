"""
Base class and data structures for all perception modules.

Every perception module inherits from PerceptionModule and outputs
PerceptionData dicts so downstream consumers (WorldState, Brain)
get a uniform interface regardless of the source.
"""

import time
from abc import ABC, abstractmethod
from typing import Any


# ── Standardised perception output ───────────────────────────────

def PerceptionData(
    source: str,
    data: dict,
    confidence: float = 1.0,
    summary: str = "",
) -> dict:
    """Factory for the canonical perception dict.

    Every perception module emits dicts with this shape so the
    world-state model and brain can consume them uniformly.

    Fields
    ------
    source      : "screen" | "camera" | "audio" | "system"
    timestamp   : epoch float (auto-filled)
    data        : module-specific structured payload
    confidence  : 0-1 float (how reliable this reading is)
    summary     : short human-readable description
    """
    return {
        "source": source,
        "timestamp": time.time(),
        "data": data,
        "confidence": confidence,
        "summary": summary,
    }


# ── Abstract base ────────────────────────────────────────────────

class PerceptionModule(ABC):
    """Interface that every perception module must implement."""

    @abstractmethod
    def initialise(self) -> None:
        """Load models, open devices, allocate resources."""

    @abstractmethod
    def capture(self) -> dict:
        """Collect one snapshot of data and return a PerceptionData dict.

        Must be safe to call from any thread.  Should NOT block
        longer than a few seconds.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources (cameras, handles, etc.)."""
