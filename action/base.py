"""Base interface for all action-layer controllers.

Provides rollback support: each controller can register an undo callback
after performing an action, allowing the executor to unwind on failure.
"""

from abc import ABC, abstractmethod
from typing import Callable


class UndoEntry:
    """One reversible action with its undo callback."""
    __slots__ = ("description", "undo_fn")

    def __init__(self, description: str, undo_fn: Callable[[], str]):
        self.description = description
        self.undo_fn = undo_fn


class BaseController(ABC):
    """Every domain controller inherits from this."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when the controller is ready to accept commands."""

    def status(self) -> dict:
        """Optional: return controller health / metrics."""
        return {"available": self.is_available()}
