"""Avatar configuration surface."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_json_map(name: str, default: dict[str, str]) -> dict[str, str]:
    raw = os.getenv(name)
    if not raw:
        return dict(default)
    try:
        import json
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v is not None}
    except Exception:
        pass
    return dict(default)


@dataclass(slots=True)
class AvatarSettings:
    enabled: bool = _env_bool("AVATAR_ENABLED", True)
    overlay_enabled: bool = _env_bool("AVATAR_OVERLAY_ENABLED", True)
    vts_enabled: bool = _env_bool("AVATAR_VTS_ENABLED", True)
    topmost: bool = _env_bool("AVATAR_OVERLAY_TOPMOST", True)
    click_through: bool = _env_bool("AVATAR_OVERLAY_CLICK_THROUGH", True)
    adaptive_fps: bool = _env_bool("AVATAR_ADAPTIVE_FPS", True)
    target_fps: int = int(os.getenv("AVATAR_TARGET_FPS", "60"))
    low_fps: int = int(os.getenv("AVATAR_LOW_FPS", "30"))
    min_fps: int = int(os.getenv("AVATAR_MIN_FPS", "24"))
    overlay_size: int = int(os.getenv("AVATAR_OVERLAY_SIZE", "300"))
    overlay_x: int = int(os.getenv("AVATAR_OVERLAY_X", "40"))
    overlay_y: int = int(os.getenv("AVATAR_OVERLAY_Y", "40"))
    overlay_state_path: str = os.getenv("AVATAR_OVERLAY_STATE_PATH", "memory_store/avatar_overlay_state.json")
    asset_dir: str = os.getenv("AVATAR_ASSET_DIR", "avatar_assets")
    hotkey_toggle: str = os.getenv("AVATAR_HOTKEY_TOGGLE", "CTRL+ALT+V")
    vts_host: str = os.getenv("AVATAR_VTS_HOST", "127.0.0.1")
    vts_port: int = int(os.getenv("AVATAR_VTS_PORT", "8001"))
    vts_plugin_name: str = os.getenv("AVATAR_VTS_PLUGIN_NAME", "ArtificialLifeAvatar")
    vts_developer: str = os.getenv("AVATAR_VTS_DEVELOPER", "ArtificialLife")
    vts_expression_map: dict[str, str] = field(default_factory=lambda: _env_json_map(
        "AVATAR_VTS_EXPRESSION_MAP",
        {},
    ))
