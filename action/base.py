"""Base interface for all action-layer controllers."""

from abc import ABC, abstractmethod


class BaseController(ABC):
    """Every domain controller inherits from this."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when the controller is ready to accept commands."""

    def status(self) -> dict:
        """Optional: return controller health / metrics."""
        return {"available": self.is_available()}
