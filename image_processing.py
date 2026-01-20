import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self):
        #self.model_face = YOLO("yolov8n.pt")
        self.model = YOLO("yolo26n.pt")
        #self.model = self.model_object

    def model_train(self, dataset : str, model_save_path : str):
        self.model.train(data=dataset, epochs=100, imgsz=640, save_path=model_save_path)
    
    def model_infer(self, image_path : str):
        results = self.model.predict(source=image_path, conf=0.85, save=True, save_txt=True)
        return results

    def camera_infer(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(source=frame, conf=0.3, save=True, save_txt=True)
            box = results[0].boxes
            class_label = results[0].names

            for b in box:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cls_id = int(b.cls[0])
                label = class_label[cls_id]
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(results[0])
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetection()
    #detector.model_train("data/dataset.yaml", "models/yolov8n_trained")
    detector.camera_infer()