"""Avatar runtime coordinator.

Bridges brain/world/TTS signals into desktop overlay + VTube Studio bridge.
"""

from __future__ import annotations

import ctypes
import threading
import time
from queue import Queue, Empty

from logging_config import get_logger

from .events import AvatarEventBus
from .overlay import DesktopOverlay
from .settings import AvatarSettings
from .state import AvatarState
from .viseme import estimate_duration_seconds, viseme_at_progress
from .vts_bridge import VTubeStudioBridge

log = get_logger("avatar.coordinator")


class AvatarCoordinator:
    def __init__(self, brain, mouth, world_state, settings: AvatarSettings | None = None):
        self.brain = brain
        self.mouth = mouth
        self.world_state = world_state
        self.settings = settings or AvatarSettings()

        self.events = AvatarEventBus()
        self.state = AvatarState()

        self._overlay: DesktopOverlay | None = None
        self._vts: VTubeStudioBridge | None = None
        self._stop_event: threading.Event | None = None
        self._threads: list[threading.Thread] = []

        self._speech_queue: Queue[dict] = Queue(maxsize=16)
        self._last_hotkey_press_ts = 0.0
        self._last_fps_eval_ts = 0.0

    def start(self, stop_event: threading.Event) -> list[threading.Thread]:
        if not self.settings.enabled:
            return []
        self._stop_event = stop_event

        self._patch_tts_hooks()

        worker = threading.Thread(target=self._state_loop, args=(stop_event,), daemon=True, name="avatar-state")
        worker.start()
        self._threads.append(worker)

        if self.settings.overlay_enabled:
            self._overlay = DesktopOverlay(
                size=self.settings.overlay_size,
                x=self.settings.overlay_x,
                y=self.settings.overlay_y,
                topmost=self.settings.topmost,
                click_through=self.settings.click_through,
                asset_dir=self.settings.asset_dir,
                state_path=self.settings.overlay_state_path,
            )
            t = threading.Thread(
                target=self._overlay.run,
                args=(stop_event, self.settings.target_fps),
                daemon=True,
                name="avatar-overlay",
            )
            t.start()
            self._threads.append(t)

        if self.settings.vts_enabled:
            self._vts = VTubeStudioBridge(
                host=self.settings.vts_host,
                port=self.settings.vts_port,
                plugin_name=self.settings.vts_plugin_name,
                plugin_developer=self.settings.vts_developer,
                expression_map=self.settings.vts_expression_map,
            )
            t = threading.Thread(target=self._vts.run, args=(stop_event,), daemon=True, name="avatar-vts")
            t.start()
            self._threads.append(t)

        return list(self._threads)

    def stop(self) -> None:
        # stop event is managed by controller; no-op placeholder for symmetry
        return

    def _patch_tts_hooks(self) -> None:
        # We keep this backwards-compatible by adding optional hook slots.
        if not hasattr(self.mouth, "speech_hooks"):
            self.mouth.speech_hooks = []

        def _on_speech_event(evt: dict) -> None:
            if self._speech_queue.full():
                try:
                    self._speech_queue.get_nowait()
                except Empty:
                    pass
            try:
                self._speech_queue.put_nowait(evt)
            except Exception:
                pass

        self.mouth.speech_hooks.append(_on_speech_event)

    def _state_loop(self, stop_event: threading.Event) -> None:
        active_speech: dict | None = None

        while not stop_event.is_set():
            self._pull_emotion_state()
            self._maybe_toggle_overlay_hotkey()
            self._apply_adaptive_fps()

            try:
                while True:
                    evt = self._speech_queue.get_nowait()
                    if evt.get("type") == "speech_start":
                        active_speech = evt
                    elif evt.get("type") == "speech_end":
                        active_speech = None
                        self.state.speaking = False
                        self.state.viseme = "rest"
            except Empty:
                pass

            if active_speech is not None:
                self._update_viseme(active_speech)

            if self._overlay:
                self._overlay.update(
                    mood=self.state.mood,
                    emotion=self.state.dominant_emotion,
                    speaking=self.state.speaking,
                    viseme=self.state.viseme,
                    strength=self.state.expression_strength,
                )

            if self._vts:
                self._vts.submit_state(
                    speaking=self.state.speaking,
                    valence=self.state.valence,
                    arousal=self.state.arousal,
                    viseme=self.state.viseme,
                    emotion=self.state.dominant_emotion,
                    expression_strength=self.state.expression_strength,
                )

            # Default loop cadence is fast enough for mouth sync and expression updates.
            stop_event.wait(0.05)

    def _pull_emotion_state(self) -> None:
        try:
            emo = self.world_state.get_channel("emotion") if self.world_state else {}
            self.state.mood = str(emo.get("mood", "calm"))
            self.state.dominant_emotion = str(emo.get("dominant_emotion") or "neutral")
            self.state.valence = float(emo.get("valence", 0.0))
            self.state.arousal = float(emo.get("arousal", 0.2))
            self.state.focus = float(emo.get("focus", 0.4))
            self.state.expression_strength = min(1.0, max(0.0, abs(self.state.valence) * 0.7 + self.state.arousal * 0.3))
            if self.brain is not None:
                self.state.mode = self.brain.state.name.lower()
        except Exception:
            pass

    def _update_viseme(self, evt: dict) -> None:
        payload = evt.get("payload", {}) if isinstance(evt, dict) else {}
        text = str(payload.get("text", ""))
        speed = float(payload.get("speed", 1.0))
        start_ts = float(payload.get("start_ts", time.time()))
        elapsed = max(0.0, time.time() - start_ts)
        total = estimate_duration_seconds(text, speed=speed)
        progress = min(0.999, elapsed / max(total, 0.001))

        self.state.speaking = True
        self.state.speech_text = text
        self.state.viseme = viseme_at_progress(text, progress)

    def _apply_adaptive_fps(self) -> None:
        if not self._overlay:
            return

        now = time.time()
        if now - self._last_fps_eval_ts < 0.8:
            return
        self._last_fps_eval_ts = now

        if self.settings.adaptive_fps:
            cpu = self._cpu_percent()
            if self.state.speaking:
                target = self.settings.target_fps if cpu < 85 else self.settings.low_fps
            elif cpu >= 85:
                target = self.settings.min_fps
            elif cpu >= 65:
                target = self.settings.low_fps
            else:
                target = self.settings.target_fps
        else:
            target = self.settings.low_fps

        self.state.fps = int(target)
        self._overlay.set_target_fps(target)

    @staticmethod
    def _cpu_percent() -> float:
        try:
            import psutil  # type: ignore
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return 30.0

    def _maybe_toggle_overlay_hotkey(self) -> None:
        if not self._overlay:
            return
        if not hasattr(ctypes, "windll"):
            return

        if not self._is_hotkey_pressed(self.settings.hotkey_toggle):
            return

        now = time.time()
        if now - self._last_hotkey_press_ts < 0.45:
            return
        self._last_hotkey_press_ts = now
        self._overlay.toggle_click_through()
        log.info("Avatar overlay click-through toggled via hotkey: %s", self.settings.hotkey_toggle)

    @staticmethod
    def _is_hotkey_pressed(hotkey: str) -> bool:
        if not hotkey:
            return False
        user32 = ctypes.windll.user32

        vk_map = {
            "CTRL": 0x11,
            "ALT": 0x12,
            "SHIFT": 0x10,
            "WIN": 0x5B,
        }

        parts = [p.strip().upper() for p in hotkey.split("+") if p.strip()]
        if not parts:
            return False

        vks: list[int] = []
        for part in parts:
            if part in vk_map:
                vks.append(vk_map[part])
                continue
            if len(part) == 1 and "A" <= part <= "Z":
                vks.append(ord(part))
                continue
            if len(part) == 1 and "0" <= part <= "9":
                vks.append(ord(part))
                continue
            if part.startswith("F") and part[1:].isdigit():
                idx = int(part[1:])
                if 1 <= idx <= 24:
                    vks.append(0x70 + idx - 1)
                    continue
            return False

        for vk in vks:
            if not (user32.GetAsyncKeyState(vk) & 0x8000):
                return False
        return True
