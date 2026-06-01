"""Windows desktop avatar overlay (always-on-top + optional click-through)."""

from __future__ import annotations

import ctypes
import json
import math
import os
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except Exception:
    Image = None
    ImageTk = None
    _HAS_PIL = False

from logging_config import get_logger

log = get_logger("avatar.overlay")

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080


@dataclass(slots=True)
class OverlaySnapshot:
    mood: str = "calm"
    emotion: str = "neutral"
    speaking: bool = False
    viseme: str = "rest"
    expression_strength: float = 0.0


class DesktopOverlay:
    def __init__(self, *, size: int, x: int, y: int, topmost: bool,
                 click_through: bool, asset_dir: str = "avatar_assets",
                 state_path: str = "memory_store/avatar_overlay_state.json"):
        self.size = size
        self.x = x
        self.y = y
        self.topmost = topmost
        self._click_through = click_through
        self._snapshot = OverlaySnapshot()
        self._lock = threading.Lock()
        self._interactive = False
        self._target_fps = 30
        self._asset_dir = Path(asset_dir)
        self._state_path = state_path
        self._drag_start: tuple[int, int] | None = None

        self._root: tk.Tk | None = None
        self._canvas: tk.Canvas | None = None
        self._stop_event: threading.Event | None = None
        self._asset_cache: dict[str, object] = {}
        self._photo_ref = None

    def update(self, *, mood: str, emotion: str, speaking: bool, viseme: str, strength: float) -> None:
        with self._lock:
            self._snapshot = OverlaySnapshot(
                mood=mood,
                emotion=emotion,
                speaking=speaking,
                viseme=viseme,
                expression_strength=max(0.0, min(1.0, strength)),
            )

    def run(self, stop_event: threading.Event, target_fps: int = 30) -> None:
        self._stop_event = stop_event
        self.set_target_fps(target_fps)
        self._load_position()
        try:
            self._root = tk.Tk()
            self._root.title("AvatarOverlay")
            self._root.overrideredirect(True)
            self._root.geometry(f"{self.size}x{self.size}+{self.x}+{self.y}")
            self._root.configure(bg="#00ff00")
            self._root.wm_attributes("-topmost", self.topmost)
            self._root.wm_attributes("-transparentcolor", "#00ff00")

            self._canvas = tk.Canvas(self._root, width=self.size, height=self.size,
                                     bg="#00ff00", highlightthickness=0, bd=0)
            self._canvas.pack(fill="both", expand=True)

            self._root.bind("<Button-3>", self._toggle_interactive)
            self._root.bind("<Button-1>", self._on_drag_start)
            self._root.bind("<B1-Motion>", self._on_drag_move)
            self._root.bind("<ButtonRelease-1>", self._on_drag_end)
            self._apply_click_through(self._click_through)

            self._schedule()
            self._poll_stop()
            self._root.mainloop()
        except Exception as e:
            log.warning("Overlay loop ended: %s", e)

    def set_target_fps(self, fps: int) -> None:
        with self._lock:
            self._target_fps = max(12, min(60, int(fps)))

    def toggle_click_through(self) -> None:
        self._interactive = not self._interactive
        self._apply_click_through(not self._interactive and self._click_through)

    def _on_drag_start(self, event) -> None:
        if not self._interactive:
            return
        self._drag_start = (event.x_root, event.y_root)

    def _on_drag_move(self, event) -> None:
        if not self._interactive or self._root is None or self._drag_start is None:
            return
        last_x, last_y = self._drag_start
        dx = event.x_root - last_x
        dy = event.y_root - last_y
        self.x += dx
        self.y += dy
        self._root.geometry(f"{self.size}x{self.size}+{self.x}+{self.y}")
        self._drag_start = (event.x_root, event.y_root)

    def _on_drag_end(self, _event) -> None:
        if not self._interactive:
            return
        self._drag_start = None
        self._save_position()

    def _schedule(self) -> None:
        if self._root is None:
            return
        self._draw()
        with self._lock:
            fps = self._target_fps
        interval_ms = max(12, int(1000 / max(1, fps)))
        self._root.after(interval_ms, self._schedule)

    def _poll_stop(self) -> None:
        if self._root is None:
            return
        if self._stop_event and self._stop_event.is_set():
            self._root.destroy()
            return
        self._root.after(100, self._poll_stop)

    def _draw(self) -> None:
        if self._canvas is None:
            return
        with self._lock:
            snap = self._snapshot

        if self._draw_assets(snap):
            return

        self._draw_fallback(snap)

    def _draw_assets(self, snap: OverlaySnapshot) -> bool:
        if not _HAS_PIL:
            return False
        if not self._asset_dir.exists() or self._canvas is None:
            return False

        base = self._load_layer("base/head.png") or self._load_layer("base/body.png")
        if base is None:
            return False

        eye_name = self._eye_for_emotion(snap.emotion)
        brow_name = self._brow_for_emotion(snap.emotion)
        mouth_name = f"mouth/{snap.viseme}.png"

        frame = Image.new("RGBA", (self.size, self.size), (0, 0, 0, 0))
        bob = int(4 * math.sin(time.time() * 2.0))

        for rel in [
            "base/body.png",
            "base/head.png",
            f"eyes/{eye_name}",
            f"brows/{brow_name}",
            mouth_name,
        ]:
            layer = self._load_layer(rel)
            if layer is None:
                continue
            comp = layer.copy()
            if rel.startswith("mouth/") or rel.startswith("eyes/") or rel.startswith("brows/"):
                frame.alpha_composite(comp, (0, bob))
            else:
                frame.alpha_composite(comp)

        photo = ImageTk.PhotoImage(frame)
        self._photo_ref = photo
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=photo)
        return True

    def _draw_fallback(self, snap: OverlaySnapshot) -> None:
        if self._canvas is None:
            return

        c = self._canvas
        c.delete("all")

        cx = self.size // 2
        cy = self.size // 2
        bob = int(4 * math.sin(time.time() * 2.0))

        face_color = "#ffe4c4"
        if snap.emotion in ("frustration", "disappointment"):
            face_color = "#ffd1c1"
        elif snap.emotion in ("amusement", "satisfaction", "pride"):
            face_color = "#fff2cc"

        c.create_oval(cx - 90, cy - 90 + bob, cx + 90, cy + 90 + bob,
                      fill=face_color, outline="#e7b58a", width=2)

        # Eyes react to speaking/expression intensity.
        eye_w = 18
        eye_h = 6 if snap.speaking else 10
        if snap.viseme == "mbp":
            eye_h = 8

        c.create_oval(cx - 45 - eye_w, cy - 25 + bob - eye_h,
                      cx - 45 + eye_w, cy - 25 + bob + eye_h,
                      fill="#2e2e2e", outline="")
        c.create_oval(cx + 45 - eye_w, cy - 25 + bob - eye_h,
                      cx + 45 + eye_w, cy - 25 + bob + eye_h,
                      fill="#2e2e2e", outline="")

        # Mouth viseme morphing.
        mouth_w = 34
        mouth_h = 8
        if snap.viseme in {"aa", "oh", "uh"}:
            mouth_h = 22
        elif snap.viseme in {"ee", "ih", "sz"}:
            mouth_h = 5
        elif snap.viseme in {"mbp", "fv"}:
            mouth_h = 3

        c.create_oval(cx - mouth_w, cy + 38 + bob - mouth_h,
                      cx + mouth_w, cy + 38 + bob + mouth_h,
                      fill="#a43d3d", outline="")

        # Debug tag: right-click toggles click-through mode.
        c.create_text(10, self.size - 12, anchor="w",
                      text="RMB: toggle overlay interaction",
                      fill="#304050", font=("Segoe UI", 8))

    def _toggle_interactive(self, _event=None):
        self.toggle_click_through()

    def _load_layer(self, rel_path: str):
        if not _HAS_PIL:
            return None
        key = rel_path.replace("\\", "/")
        cached = self._asset_cache.get(key)
        if cached is not None:
            return cached

        full = self._asset_dir / rel_path
        if not full.exists() or not full.is_file():
            return None
        try:
            img = Image.open(full).convert("RGBA")
            if img.size != (self.size, self.size):
                img = img.resize((self.size, self.size), Image.LANCZOS)
            self._asset_cache[key] = img
            return img
        except Exception:
            return None

    @staticmethod
    def _eye_for_emotion(emotion: str) -> str:
        if emotion in ("amusement", "satisfaction", "pride"):
            return "happy.png"
        if emotion in ("frustration", "disappointment"):
            return "angry.png"
        return "neutral.png"

    @staticmethod
    def _brow_for_emotion(emotion: str) -> str:
        if emotion in ("surprise", "curiosity"):
            return "raised.png"
        if emotion in ("frustration", "disappointment"):
            return "frown.png"
        return "neutral.png"

    def _apply_click_through(self, enabled: bool) -> None:
        if self._root is None:
            return
        hwnd = self._root.winfo_id()
        exstyle = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if enabled:
            exstyle |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW
        else:
            exstyle = (exstyle | WS_EX_LAYERED | WS_EX_TOOLWINDOW) & ~WS_EX_TRANSPARENT
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, exstyle)

    def _load_position(self) -> None:
        if not self._state_path or not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.x = int(data.get("x", self.x))
            self.y = int(data.get("y", self.y))
        except Exception:
            return

    def _save_position(self) -> None:
        if not self._state_path:
            return
        try:
            os.makedirs(os.path.dirname(self._state_path) or ".", exist_ok=True)
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump({"x": self.x, "y": self.y}, f)
        except Exception:
            return
