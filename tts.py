"""
TTS module using Kokoro — a human-like neural TTS model (82M params).

Runs locally on CPU via ONNX Runtime.  Produces natural, consistent
speech with proper prosody, breathing pauses, and intonation.
Deterministic — same input always produces the same output.

Voices (prefix = accent/gender):
  af_heart   — American female (warm, default)
  af_bella   — American female (soft)
  af_sarah   — American female (clear)
  af_nova    — American female (bright)
  am_adam    — American male
  am_michael — American male (deep)
  bf_emma    — British female
  bm_george  — British male
"""

import re
import time
import numpy as np
import sounddevice as sd
from queue import Empty
from threading import Event
from config import global_command_queue

# ── Lazy-loaded global (heavy model loaded once on first speak) ──
_kokoro = None
_anime_voices: dict = {}   # pre-blended emotion → voice embeddings

SAMPLE_RATE = 24000

# Blend recipes: (english_voice, japanese_voice, en_weight)
# Higher en_weight = more intelligible English, lower = more anime feel
_DEFAULT_BLEND = ("af_sky", "jf_alpha", 0.6)

# Emotion → (english_voice, jp_voice, en_weight) for variety
_EMOTION_BLEND = {
    "satisfaction":   ("af_heart",  "jf_alpha",      0.6),
    "pride":          ("af_bella",  "jf_gongitsune", 0.6),
    "gratitude":      ("af_heart",  "jf_alpha",      0.6),
    "amusement":      ("af_nova",   "jf_nezumi",     0.55),
    "curiosity":      ("af_sky",    "jf_alpha",      0.6),
    "surprise":       ("af_nova",   "jf_nezumi",     0.5),
    "frustration":    ("af_sarah",  "jf_tebukuro",   0.65),
    "disappointment": ("af_bella",  "jf_tebukuro",   0.65),
    "concern":        ("af_heart",  "jf_alpha",      0.6),
    "boredom":        ("af_bella",  "jf_gongitsune", 0.6),
}

# Emotion → speed multiplier (subtle adjustments)
_EMOTION_SPEED = {
    "surprise":       1.08,
    "amusement":      1.05,
    "frustration":    0.92,
    "disappointment": 0.90,
    "boredom":        0.88,
    "concern":        0.95,
}


def _add_pauses(text: str) -> str:
    """Insert silence markers at natural pause points for Kokoro."""
    # Ellipsis → long pause
    text = re.sub(r'\.{3,}', '... ', text)
    # Em-dash / double-dash → comma pause
    text = re.sub(r'[—–]|--', ', ', text)
    return text


def _blend(en_name: str, jp_name: str, en_weight: float):
    """Blend an English and Japanese voice style."""
    en = _kokoro.get_voice_style(en_name)
    jp = _kokoro.get_voice_style(jp_name)
    return en_weight * en + (1.0 - en_weight) * jp


def _ensure_model():
    """Lazy-load the Kokoro ONNX model and pre-blend anime voices."""
    global _kokoro, _anime_voices
    if _kokoro is not None:
        return
    from kokoro_onnx import Kokoro
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
    voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")
    _kokoro = Kokoro(model_path, voices_path)

    # Pre-blend all emotion voices + default
    en, jp, w = _DEFAULT_BLEND
    _anime_voices["_default"] = _blend(en, jp, w)
    for emotion, (en, jp, w) in _EMOTION_BLEND.items():
        _anime_voices[emotion] = _blend(en, jp, w)
    print("[TTS] Kokoro model loaded (anime voice)")


class TextToSpeech:
    def __init__(self, speed: float = 1.05):
        self.speed = speed
        self.speech_hooks: list = []

    def text_to_speech_loop(self, stop_event: Event):
        _ensure_model()
        while not stop_event.is_set():
            try:
                item = global_command_queue.get(timeout=0.1)
            except Empty:
                continue
            text, emotion, style = self._unpack_item(item)
            self.speak(text, emotion=emotion, style=style)
            global_command_queue.task_done()

    def speak(self, text: str, emotion: str | None = None,
              style: dict | None = None):
        if not text or not text.strip():
            return
        _ensure_model()

        style = style or {}
        text = _add_pauses(text)
        voice = _anime_voices.get(emotion, _anime_voices["_default"])
        speed = self.speed * _EMOTION_SPEED.get(emotion, 1.0) if emotion else self.speed
        speed *= float(style.get("speed", 1.0))

        # Keep voice transitions smooth by avoiding huge jumps.
        speed = max(0.78, min(1.28, speed))

        # Optional style-driven text shaping.
        text = self._apply_style(text, style)
        start_ts = time.time()

        self._emit_speech_event("speech_start", {
            "text": text,
            "emotion": emotion,
            "style": style,
            "speed": speed,
            "start_ts": start_ts,
        })

        try:
            samples, sr = _kokoro.create(text, voice=voice, speed=speed)
            sd.play(samples, samplerate=sr)
            sd.wait()
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            self._emit_speech_event("speech_end", {
                "text": text,
                "emotion": emotion,
                "style": style,
                "speed": speed,
                "start_ts": start_ts,
            })

    def _emit_speech_event(self, event_type: str, payload: dict) -> None:
        if not self.speech_hooks:
            return
        event = {
            "type": event_type,
            "payload": payload,
        }
        for hook in list(self.speech_hooks):
            try:
                hook(event)
            except Exception:
                continue

    @staticmethod
    def _unpack_item(item):
        """Backwards-compatible queue item parsing.

        Supported forms:
            "text"
            (text, emotion)
            (text, emotion, style_dict)
        """
        if not isinstance(item, tuple):
            return item, None, None
        if len(item) == 2:
            return item[0], item[1], None
        if len(item) >= 3:
            return item[0], item[1], item[2]
        return "", None, None

    @staticmethod
    def _apply_style(text: str, style: dict) -> str:
        verbosity = style.get("verbosity", "normal")
        formality = style.get("formality", "neutral")

        out = text
        if verbosity == "brief":
            parts = re.split(r"(?<=[.!?])\s+", out)
            out = parts[0] if parts else out

        if formality == "formal":
            out = out.replace("I'm", "I am").replace("can't", "cannot")
        elif formality == "casual":
            out = out.replace("I am", "I'm")

        return out