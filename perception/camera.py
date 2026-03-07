"""
Camera Perception — perceives the physical world through the webcam.

Responsibilities:
  * Capture frames from the camera
  * Detect objects with YOLO (via existing image_processing.ObjectDetection)
  * Optionally recognise faces
  * Output a structured snapshot of what the camera sees

This module wraps `image_processing.ObjectDetection` — it does not
duplicate the YOLO model or inference code.
"""

import time
import threading

import cv2
import numpy as np

from .base import PerceptionModule, PerceptionData


class CameraPerception(PerceptionModule):
    """Perceives the physical environment through the webcam."""

    def __init__(self, vision_model=None, camera_index: int = 0):
        """
        Args:
            vision_model: an existing image_processing.ObjectDetection
                          instance whose YOLO model is reused for detection.
            camera_index: OpenCV camera device index.
        """
        self._vision = vision_model
        self._cam_idx = camera_index
        self._cap = None
        self._lock = threading.Lock()
        self._last_frame = None

    # ── PerceptionModule interface ───────────────────────────────

    def initialise(self) -> None:
        self._cap = cv2.VideoCapture(self._cam_idx)
        if not self._cap.isOpened():
            print(f"[CameraPerception] Warning: camera {self._cam_idx} failed to open.")

    def capture(self) -> dict:
        """Grab one frame, run detection, return structured data."""
        frame = self._grab_frame()
        if frame is None:
            return PerceptionData(
                source="camera",
                data={"objects": [], "frame_available": False},
                confidence=0.0,
                summary="No camera frame available",
            )

        detections = self._detect_objects(frame)
        summary = self._build_summary(detections)

        return PerceptionData(
            source="camera",
            data={
                "objects": detections,
                "frame_shape": list(frame.shape),
                "frame_available": True,
            },
            confidence=0.9 if detections else 0.5,
            summary=summary,
        )

    def shutdown(self) -> None:
        with self._lock:
            if self._cap and self._cap.isOpened():
                self._cap.release()
            self._cap = None

    # ── Public helpers ───────────────────────────────────────────

    def feed_frame(self, frame: np.ndarray):
        """Accept a frame from an external camera loop (e.g. the main
        thread's camera_infer) so capture() can use it without opening
        the device a second time."""
        with self._lock:
            self._last_frame = frame.copy()

    # ── Internals ────────────────────────────────────────────────

    def _grab_frame(self) -> np.ndarray | None:
        # Prefer a frame fed from the main camera loop
        with self._lock:
            if self._last_frame is not None:
                return self._last_frame.copy()

        # Otherwise try to read directly
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                return frame
        return None

    def _detect_objects(self, frame: np.ndarray) -> list:
        """Run YOLO detection and return a list of structured dicts."""
        if self._vision is None:
            return []

        try:
            results = self._vision.model.predict(
                source=frame, conf=0.4, verbose=False
            )
            boxes_raw = results[0].boxes
            names = results[0].names

            detections = []
            for b in boxes_raw:
                cls_id = int(b.cls[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                detections.append({
                    "label": names[cls_id],
                    "confidence": round(float(b.conf[0]), 3),
                    "bbox": [x1, y1, x2, y2],
                    "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                })
            return detections

        except Exception as e:
            print(f"[CameraPerception] Detection error: {e}")
            return []

    @staticmethod
    def _build_summary(detections: list) -> str:
        if not detections:
            return "Nothing detected"
        labels = [d["label"] for d in detections]
        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        parts = [f"{v}x {k}" if v > 1 else k for k, v in counts.items()]
        return f"Detected: {', '.join(parts)}"
