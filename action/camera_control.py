"""CameraController — capture images and track objects via the webcam.

Wraps the existing perception/camera.py CameraPerception and
image_processing.ObjectDetection so no model or inference code is
duplicated.  This controller is the *action* interface (take a photo,
start/stop tracking) vs. the perception module which is the *sensing*
interface (passive frame capture for world-state).
"""

import time
import threading

import cv2
import numpy as np

from .base import BaseController


class CameraController(BaseController):
    """Controls the webcam for intentional actions: snapshots, tracking."""

    def __init__(self, vision_model=None, camera_index: int = 0):
        """
        Args:
            vision_model: image_processing.ObjectDetection instance.
            camera_index: OpenCV device index.
        """
        self._vision = vision_model
        self._cam_idx = camera_index
        self._cap: cv2.VideoCapture | None = None
        self._lock = threading.Lock()
        self._last_frame: np.ndarray | None = None

        # Tracking state
        self._tracking = False
        self._track_label: str | None = None
        self._track_thread: threading.Thread | None = None
        self._track_stop = threading.Event()
        self._track_results: list[dict] = []

    # ── BaseController ───────────────────────────────────────────

    def is_available(self) -> bool:
        """True if we can grab a frame (camera open or frame is being fed)."""
        with self._lock:
            if self._last_frame is not None:
                return True
        if self._cap and self._cap.isOpened():
            return True
        return False

    # ── Public API ───────────────────────────────────────────────

    def open(self) -> str:
        """Open the camera device."""
        with self._lock:
            if self._cap and self._cap.isOpened():
                return "Camera already open"
            self._cap = cv2.VideoCapture(self._cam_idx)
            if not self._cap.isOpened():
                self._cap = None
                return "FAILED: could not open camera"
        return "Camera opened"

    def close(self) -> str:
        """Release the camera device."""
        self.stop_tracking()
        with self._lock:
            if self._cap and self._cap.isOpened():
                self._cap.release()
            self._cap = None
        return "Camera closed"

    def feed_frame(self, frame: np.ndarray):
        """Accept a frame from an external camera loop (main thread)."""
        with self._lock:
            self._last_frame = frame.copy()

    def capture_image(self, save_path: str | None = None) -> dict:
        """Take a single snapshot.

        Returns:
            dict with keys: success, detections, frame_shape, saved_to
        """
        frame = self._grab_frame()
        if frame is None:
            return {"success": False, "error": "No frame available"}

        detections = self._detect(frame)

        saved_to = None
        if save_path:
            cv2.imwrite(save_path, frame)
            saved_to = save_path

        return {
            "success": True,
            "detections": detections,
            "frame_shape": list(frame.shape),
            "saved_to": saved_to,
        }

    def track_object(self, label: str) -> str:
        """Start tracking objects matching *label* in a background thread.

        Results accumulate in track_results.  Call stop_tracking() when done.
        """
        if self._tracking:
            return f"Already tracking '{self._track_label}'. Stop first."

        self._track_label = label.lower()
        self._track_stop.clear()
        self._track_results.clear()
        self._tracking = True

        self._track_thread = threading.Thread(
            target=self._tracking_loop, daemon=True
        )
        self._track_thread.start()
        return f"Started tracking '{label}'"

    def stop_tracking(self) -> list[dict]:
        """Stop the tracking loop and return accumulated results."""
        if not self._tracking:
            return self._track_results

        self._track_stop.set()
        if self._track_thread:
            self._track_thread.join(timeout=3)
        self._tracking = False
        return self._track_results

    @property
    def track_results(self) -> list[dict]:
        """Snapshot of accumulated tracking detections."""
        return list(self._track_results)

    # ── Internals ────────────────────────────────────────────────

    def _grab_frame(self) -> np.ndarray | None:
        with self._lock:
            if self._last_frame is not None:
                return self._last_frame.copy()
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                return frame
        return None

    def _detect(self, frame: np.ndarray) -> list[dict]:
        """Run YOLO detection and return structured dicts."""
        if self._vision is None:
            return []
        try:
            results = self._vision.model.predict(
                source=frame, conf=0.4, verbose=False
            )
            boxes = results[0].boxes
            names = results[0].names
            detections = []
            for b in boxes:
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
            print(f"  [CamCtrl] Detection error: {e}")
            return []

    def _tracking_loop(self):
        """Background loop: detect + filter for target label."""
        print(f"  [CamCtrl] Tracking loop started for '{self._track_label}'")
        while not self._track_stop.is_set():
            frame = self._grab_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            detections = self._detect(frame)
            matches = [d for d in detections
                       if d["label"].lower() == self._track_label]
            if matches:
                self._track_results.append({
                    "timestamp": time.time(),
                    "matches": matches,
                })

            time.sleep(0.2)  # ~5 fps tracking
        print(f"  [CamCtrl] Tracking loop stopped")
