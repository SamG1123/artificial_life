"""DecisionRouter — maps action dicts to domain controllers.

Domains:
    BROWSER  — web browser interactions (navigate, search, click in browser, etc.)
    APP      — launching / switching applications
    SYSTEM   — shell commands, hotkeys
    DESKTOP  — mouse / keyboard on the OS desktop (pyautogui)
    DRAWING  — LLM-driven drawing sequences
    META     — control flow (done)
"""

from enum import Enum


class ActionDomain(Enum):
    BROWSER = "browser"
    APP = "app"
    SYSTEM = "system"
    DESKTOP = "desktop"
    DRAWING = "drawing"
    META = "meta"


# Actions whose domain is fixed regardless of context
_STATIC_ROUTES: dict[str, ActionDomain] = {
    "navigate": ActionDomain.BROWSER,
    "open_browser": ActionDomain.BROWSER,
    "download": ActionDomain.BROWSER,
    "go_back": ActionDomain.BROWSER,
    "open_app": ActionDomain.APP,
    "run_command": ActionDomain.SYSTEM,
    "hotkey": ActionDomain.SYSTEM,
    "draw_plan": ActionDomain.DRAWING,
    "done": ActionDomain.META,
}

# Actions whose domain depends on whether the browser is currently active
_CONTEXT_DEPENDENT: set[str] = {"click", "type", "press_key", "scroll"}

# Always desktop (raw coordinate actions)
_DESKTOP_ONLY: set[str] = {"mouse_click_xy", "mouse_drag"}


class DecisionRouter:
    """Routes an action dict to the correct domain controller."""

    def __init__(self):
        self._route_counts: dict[str, int] = {}

    def route(self, action_dict: dict, browser_mode: bool = False) -> ActionDomain:
        """Determine the domain for *action_dict*.

        Args:
            action_dict:  Action produced by the reasoning model.
            browser_mode: True when the executor currently has a browser open.

        Returns:
            The :class:`ActionDomain` this action belongs to.
        """
        action = action_dict.get("action", "done")

        if action in _DESKTOP_ONLY:
            domain = ActionDomain.DESKTOP
        elif action in _CONTEXT_DEPENDENT:
            domain = ActionDomain.BROWSER if browser_mode else ActionDomain.DESKTOP
        elif action in _STATIC_ROUTES:
            domain = _STATIC_ROUTES[action]
        else:
            domain = ActionDomain.DESKTOP  # fallback for unknown actions

        self._route_counts[domain.value] = self._route_counts.get(domain.value, 0) + 1
        return domain

    @property
    def stats(self) -> dict:
        """Cumulative count of actions routed per domain."""
        return dict(self._route_counts)
