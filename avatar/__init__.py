"""Avatar runtime package.

Provides desktop overlay + VTube Studio bridge plumbing.
"""

from .coordinator import AvatarCoordinator
from .settings import AvatarSettings

__all__ = ["AvatarCoordinator", "AvatarSettings"]
