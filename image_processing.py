from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import os
from groq import Groq
import base64
from transformers import pipeline, AutoTokenizer, AutoModel
from threading import Thread, Event
from dotenv import load_dotenv
load_dotenv()


def detect_changed_regions(
    prev_bgr,
    curr_bgr,
    *,
    threshold: int = 24,
    min_area: int = 1400,
    max_regions: int = 10,
):
    """Return changed screen regions between two BGR frames.

    Args:
        prev_bgr: Previous frame as BGR ndarray or None.
        curr_bgr: Current frame as BGR ndarray.
        threshold: Pixel threshold for binary diff.
        min_area: Minimum contour area to keep.
        max_regions: Cap on returned regions.

    Returns:
        {
          "regions": [{"x1":int,"y1":int,"x2":int,"y2":int,"area":int}, ...],
          "change_ratio": float,
        }
    """
    if curr_bgr is None:
        return {"regions": [], "change_ratio": 0.0}

    h, w = curr_bgr.shape[:2]
    if prev_bgr is None or prev_bgr.shape[:2] != (h, w):
        # First frame or resolution mismatch -> treat as full change
        return {
            "regions": [{"x1": 0, "y1": 0, "x2": w, "y2": h, "area": int(w * h)}],
            "change_ratio": 1.0,
        }

    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

    # Blur suppresses tiny rendering noise between frames.
    prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)

    diff = cv2.absdiff(prev_blur, curr_blur)
    _, diff_bin = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    diff_bin = cv2.morphologyEx(diff_bin, cv2.MORPH_OPEN, kernel)
    diff_bin = cv2.dilate(diff_bin, kernel, iterations=2)

    changed_pixels = int(cv2.countNonZero(diff_bin))
    total_pixels = float(h * w) if h and w else 1.0
    change_ratio = changed_pixels / total_pixels

    contours, _ = cv2.findContours(diff_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for c in contours:
        area = int(cv2.contourArea(c))
        if area < min_area:
            continue
        x, y, rw, rh = cv2.boundingRect(c)
        regions.append({
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + rw),
            "y2": int(y + rh),
            "area": area,
        })

    regions.sort(key=lambda r: r["area"], reverse=True)
    regions = regions[:max_regions]

    # If many tiny regions were filtered out but diff ratio is still meaningful,
    # fallback to a single full-frame region.
    if not regions and change_ratio > 0.15:
        regions = [{"x1": 0, "y1": 0, "x2": w, "y2": h, "area": int(w * h)}]

    return {
        "regions": regions,
        "change_ratio": round(float(change_ratio), 4),
    }

class ObjectDetection:
    def __init__(self):
        self.model = YOLO("yolo26n.pt", verbose=False)
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.frame_buffer = []

    
    def model_train(self, dataset : str, model_save_path : str):
        self.model.train(data=dataset, epochs=100, imgsz=640, save_path=model_save_path)
    
    def model_infer(self, image_path : str):
        results = self.model.predict(source=image_path, conf=0.4, verbose=False)
        return results
    
    def get_labels(self, frame):
        results = self.model.predict(source=frame, conf=0.4, verbose=False)
        box = results[0].boxes
        class_label = results[0].names
        labels = []
        for b in box:
            cls_id = int(b.cls[0])
            label = class_label[cls_id]
            labels.append(label)
        return labels

    def get_boxes(self, frame):
        results = self.model.predict(source=frame, conf=0.4, verbose=False)
        box = results[0].boxes
        boxes = []
        for b in box:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            boxes.append((x1, y1, x2, y2))
        return boxes
    
    def screenshot_infer(self, screenshot, query : str = "extract elements from the image"):
        image = np.array(screenshot)
        elements = self.ocr_infer(image, query=query)
        return elements
        

    
    def ocr_infer(self, image, query: str = "Extract text from the image"):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        message = self.groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{query}:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=512
        )
        text = message.choices[0].message.content
        return text
    
    def speech_command_infer(self, text : str):
        self.set_command = text.lower()

    def camera_infer(self, stop_event):
        cap = cv2.VideoCapture(0)
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            box = self.get_boxes(frame)
            class_label = self.get_labels(frame)
            self.frame_buffer.append(frame)

            for i, (x1, y1, x2, y2) in enumerate(box):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_label[i], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

        cap.release()
        cv2.destroyAllWindows()

    def ocr_test(self, image_path : str):
        image = cv2.imread(image_path)
        text = self.ocr_infer(image)
        return text

if __name__ == "__main__":
    detector = ObjectDetection()
    detector.camera_infer()