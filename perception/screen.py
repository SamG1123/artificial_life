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
import cv2
import re

from .base import PerceptionModule, PerceptionData
from image_processing import detect_changed_regions

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
        self._prev_bgr = None
        self._prev_window_title = ""
        self._prev_app_state_sig = ""

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
        frame_bgr = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # 1. Compute changed regions vs previous frame
        diff_info = detect_changed_regions(self._prev_bgr, frame_bgr)
        changed_regions = diff_info.get("regions", [])
        change_ratio = diff_info.get("change_ratio", 0.0)

        # 1. OCR — extract text regions with bounding boxes
        if not changed_regions or change_ratio >= 0.75:
            ocr_results = self._run_ocr(screenshot)
            ocr_mode = "full"
        else:
            ocr_results = self._run_ocr_regions(screenshot, changed_regions)
            ocr_mode = "changed_regions"

        # 2. Active-window context
        window_info = self._get_active_window()
        curr_title = (window_info.get("title") or "").strip()

        # 2a. App-state heuristics from window title + OCR
        app_state = self._extract_app_state(window_info, ocr_results, change_ratio)
        app_state_sig = self._app_state_signature(app_state)

        # 2b. Change events
        change_events = self._build_change_events(
            prev_title=self._prev_window_title,
            curr_title=curr_title,
            changed_regions=changed_regions,
            change_ratio=change_ratio,
            prev_app_sig=self._prev_app_state_sig,
            curr_app_sig=app_state_sig,
            app_state=app_state,
        )

        # 3. Screenshot dimensions
        data = {
            "resolution": {"w": self._width, "h": self._height},
            "active_window": window_info,
            "ocr_texts": ocr_results,
            "ocr_mode": ocr_mode,
            "changed_regions": changed_regions,
            "change_ratio": change_ratio,
            "change_events": change_events,
            "app_state": app_state,
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

        # Update frame history for next capture.
        self._prev_bgr = frame_bgr
        self._prev_window_title = curr_title
        self._prev_app_state_sig = app_state_sig

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

    def _run_ocr_regions(self, screenshot: Image.Image, regions: list[dict]) -> list:
        """Run OCR only on changed regions and map bboxes back to full frame."""
        out: list[dict] = []
        full = np.array(screenshot)
        reader = _get_ocr_reader()

        for r in regions:
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            roi = full[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            h, w = roi.shape[:2]
            if w > OCR_MAX_WIDTH:
                scale = OCR_MAX_WIDTH / w
                roi_small = np.array(Image.fromarray(roi).resize((OCR_MAX_WIDTH, int(h * scale)), Image.LANCZOS))
            else:
                scale = 1.0
                roi_small = roi

            try:
                raw = reader.readtext(roi_small)
            except Exception:
                continue

            inv = 1.0 / scale
            for bbox, text, conf in raw:
                bx1 = int(bbox[0][0] * inv) + x1
                by1 = int(bbox[0][1] * inv) + y1
                bx2 = int(bbox[2][0] * inv) + x1
                by2 = int(bbox[2][1] * inv) + y1
                out.append({
                    "text": text,
                    "bbox": [bx1, by1, bx2, by2],
                    "confidence": round(conf, 3),
                })

        # De-duplicate near-identical OCR snippets.
        dedup: dict[tuple, dict] = {}
        for item in out:
            key = (
                item["text"].strip().lower(),
                item["bbox"][0] // 10,
                item["bbox"][1] // 10,
            )
            prev = dedup.get(key)
            if prev is None or item["confidence"] > prev["confidence"]:
                dedup[key] = item
        return list(dedup.values())

    @staticmethod
    def _build_change_events(
        *,
        prev_title: str,
        curr_title: str,
        changed_regions: list[dict],
        change_ratio: float,
        prev_app_sig: str,
        curr_app_sig: str,
        app_state: dict,
    ) -> list[str]:
        events: list[str] = []
        if curr_title and prev_title and curr_title != prev_title:
            events.append(f"new window appeared: {curr_title}")
        if change_ratio > 0.5:
            events.append("major screen update")
        elif changed_regions:
            events.append(f"{len(changed_regions)} region(s) changed")
        if prev_app_sig and curr_app_sig and prev_app_sig != curr_app_sig:
            app = app_state.get("app", "application")
            status = app_state.get("status", "updated")
            events.append(f"{app} state changed: {status}")
        return events

    @staticmethod
    def _extract_app_state(window_info: dict, ocr_results: list[dict], change_ratio: float) -> dict:
        """Infer high-level application state from screen cues.

        Heuristics use:
        - window title markers (e.g., '*' unsaved changes)
        - OCR text patterns (saving/loading/progress)
        - change ratio for animation/loading hints
        """
        title = (window_info.get("title") or "").strip()
        app = ScreenPerception._app_from_title(title)
        lower_title = title.lower()

        text_blob = " ".join((o.get("text") or "") for o in ocr_results).lower()

        unsaved = "*" in title or "unsaved" in lower_title
        if "notepad" in lower_title and title.startswith("*"):
            unsaved = True

        progress = ScreenPerception._extract_progress_percent(title + " " + text_blob)

        loading_tokens = (
            "loading", "syncing", "saving", "processing", "please wait",
            "installing", "updating", "buffering", "rendering",
        )
        loading = any(t in text_blob for t in loading_tokens)
        loading = loading or (progress is not None and progress < 100)

        if loading:
            status = "loading"
        elif unsaved:
            status = "editing"
        elif change_ratio > 0.35:
            status = "active"
        else:
            status = "idle"

        return {
            "app": app,
            "title": title,
            "status": status,
            "unsaved_changes": bool(unsaved),
            "loading": bool(loading),
            "progress_percent": progress,
            "change_ratio": round(float(change_ratio), 4),
        }

    @staticmethod
    def _app_from_title(title: str) -> str:
        if not title:
            return "unknown"
        # Common Windows title form: "Document - AppName"
        parts = [p.strip() for p in title.split("-") if p.strip()]
        if parts:
            return parts[-1][:40]
        return title[:40]

    @staticmethod
    def _extract_progress_percent(text: str):
        m = re.search(r"(\d{1,3})\s*%", text)
        if not m:
            return None
        pct = int(m.group(1))
        return max(0, min(100, pct))

    @staticmethod
    def _app_state_signature(app_state: dict) -> str:
        return "|".join([
            str(app_state.get("app", "")),
            str(app_state.get("status", "")),
            str(app_state.get("unsaved_changes", False)),
            str(app_state.get("loading", False)),
            str(app_state.get("progress_percent", None)),
        ])

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
        ocr_mode = data.get("ocr_mode", "")
        if ocr_mode:
            parts.append(f"OCR: {ocr_mode}")
        ev = data.get("change_events", [])
        if ev:
            parts.append("; ".join(ev[:2]))
        app_state = data.get("app_state", {})
        if app_state.get("app"):
            app_bits = [app_state.get("app", "")]
            if app_state.get("status"):
                app_bits.append(app_state["status"])
            if app_state.get("unsaved_changes"):
                app_bits.append("unsaved")
            if app_state.get("progress_percent") is not None:
                app_bits.append(f"{app_state['progress_percent']}%")
            parts.append("App: " + ", ".join(app_bits))
        desc = data.get("vision_description", "")
        if desc:
            parts.append(desc[:120])
        return " | ".join(parts) if parts else "Screen captured"
