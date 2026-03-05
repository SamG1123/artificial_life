"""
Screen Perception — captures what is visible on the display.

Responsibilities:
  * Take a screenshot
  * Run OCR to extract visible text + bounding boxes
  * Detect UI layout / interactive elements via vision model
  * Output a structured snapshot of the screen state

Wraps the existing `image_processing.ObjectDetection` for OCR and
vision-language analysis — no functionality is duplicated.
"""

import time
import base64
import threading
from io import BytesIO

import pyautogui as pag
import numpy as np
from PIL import Image

from .base import PerceptionModule, PerceptionData

# Lazy imports so the module loads fast even when heavy deps are slow
_easyocr_reader = None
_easyocr_lock = threading.Lock()

OCR_MAX_WIDTH = 960  # downscale before OCR for speed


def _get_ocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        with _easyocr_lock:
            if _easyocr_reader is None:
                import easyocr
                _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


class ScreenPerception(PerceptionModule):
    """Perceives the desktop screen via screenshots + OCR."""

    def __init__(self, vision_model=None):
        """
        Args:
            vision_model: an existing image_processing.ObjectDetection
                          instance.  If supplied, its ocr_infer / screenshot_infer
                          methods are reused for vision-language queries.
                          Pass None to skip vision-LLM analysis.
        """
        self._vision = vision_model
        self._width, self._height = 0, 0

    # ── PerceptionModule interface ───────────────────────────────

    def initialise(self) -> None:
        self._width, self._height = pag.size()

    def capture(self) -> dict:
        """Take a screenshot and return structured screen data."""
        screenshot = pag.screenshot()
        screen_data = self._extract_screen_data(screenshot)
        summary = self._build_summary(screen_data)
        return PerceptionData(
            source="screen",
            data=screen_data,
            confidence=0.85,
            summary=summary,
        )

    def shutdown(self) -> None:
        pass  # nothing to release

    # ── Internals ────────────────────────────────────────────────

    def _extract_screen_data(self, screenshot: Image.Image) -> dict:
        """Run OCR + optional vision-LLM on the screenshot."""
        # 1. OCR — extract text regions with bounding boxes
        ocr_results = self._run_ocr(screenshot)

        # 2. Active-window context
        window_info = self._get_active_window()

        # 3. Screenshot dimensions
        data = {
            "resolution": {"w": self._width, "h": self._height},
            "active_window": window_info,
            "ocr_texts": ocr_results,
            "screenshot_b64": self._to_b64(screenshot),
        }

        # 4. Optional vision-LLM description
        if self._vision is not None:
            try:
                desc = self._vision.screenshot_infer(
                    screenshot,
                    query="Briefly describe what is visible on screen: "
                          "the application, main content, and any focused elements.",
                )
                data["vision_description"] = desc
            except Exception:
                data["vision_description"] = ""

        return data

    def _run_ocr(self, screenshot: Image.Image) -> list:
        """Run EasyOCR on the screenshot, return list of
        {text, bbox, confidence} dicts."""
        img = np.array(screenshot)

        # Downscale for speed
        h, w = img.shape[:2]
        if w > OCR_MAX_WIDTH:
            scale = OCR_MAX_WIDTH / w
            img = np.array(
                screenshot.resize(
                    (OCR_MAX_WIDTH, int(h * scale)), Image.LANCZOS
                )
            )
        else:
            scale = 1.0

        try:
            reader = _get_ocr_reader()
            raw = reader.readtext(img)
        except Exception:
            return []

        results = []
        for bbox, text, conf in raw:
            # Scale bbox back to original resolution
            inv = 1.0 / scale
            x1 = int(bbox[0][0] * inv)
            y1 = int(bbox[0][1] * inv)
            x2 = int(bbox[2][0] * inv)
            y2 = int(bbox[2][1] * inv)
            results.append({
                "text": text,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 3),
            })
        return results

    @staticmethod
    def _get_active_window() -> dict:
        try:
            import pygetwindow as gw
            active = gw.getActiveWindow()
            if active and active.title.strip():
                return {
                    "title": active.title,
                    "position": {"x": active.left, "y": active.top},
                    "size": {"w": active.width, "h": active.height},
                }
        except Exception:
            pass
        return {}

    @staticmethod
    def _to_b64(img: Image.Image, max_w: int = 800) -> str:
        """Encode a PIL image to a JPEG base64 string."""
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _build_summary(data: dict) -> str:
        parts = []
        win = data.get("active_window", {})
        if win.get("title"):
            parts.append(f"Active: {win['title']}")
        n = len(data.get("ocr_texts", []))
        if n:
            parts.append(f"{n} text regions detected")
        desc = data.get("vision_description", "")
        if desc:
            parts.append(desc[:120])
        return " | ".join(parts) if parts else "Screen captured"
