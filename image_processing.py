import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from groq import Groq
import base64


class ObjectDetection:
    def __init__(self):
        self.model = YOLO("yolo26n.pt")
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.frame_buffer = []
    
    def model_train(self, dataset : str, model_save_path : str):
        self.model.train(data=dataset, epochs=100, imgsz=640, save_path=model_save_path)
    
    def model_infer(self, image_path : str):
        results = self.model.predict(source=image_path, conf=0.85, save=True, save_txt=True)
        return results
    
    def get_labels(self, frame):
        results = self.model.predict(source=frame, conf=0.4)
        box = results[0].boxes
        class_label = results[0].names
        labels = []
        for b in box:
            cls_id = int(b.cls[0])
            label = class_label[cls_id]
            labels.append(label)
        return labels

    def get_boxes(self, frame):
        results = self.model.predict(source=frame, conf=0.4)
        box = results[0].boxes
        boxes = []
        for b in box:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            boxes.append((x1, y1, x2, y2))
        return boxes

    def ocr_infer(self, image):
        # Convert image to base64 for Groq API
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.standard_b64encode(buffer).decode('utf-8')
        
        message = self.groq_client.chat.completions.create(
            model="llava-2-7b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": "Recognize and extract all text from this image."
                        }
                    ],
                }
            ],
            max_tokens=1024,
        )
        return message.choices[0].message.content

    def camera_infer(self):
        cap = cv2.VideoCapture(0)
        while True:
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
                break

            if cv2.waitKey(1) & 0xFF == ord('s'):
                text = self.ocr_infer(self.frame_buffer[-1])
                print("OCR Result:", text)



        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetection()
    detector.camera_infer()