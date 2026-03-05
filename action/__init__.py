"""action — the action layer for the AI agent.

Exposes domain controllers that execute commands in the environment:
    BrowserController  — click, scroll, type, navigate (via Selenium)
    AppController      — open, close, switch desktop apps
    SystemController   — shell commands, file ops, power management
    CameraController   — capture images, track objects
"""

from .app_control import AppController
from .system_control import SystemController
from .camera_control import CameraController
from .base import BaseController

__all__ = [
    "BaseController",
    "AppController",
    "SystemController",
    "CameraController",
]
