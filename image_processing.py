import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


class ObjectDetection:
    def __init__(self):
        self.model = YOLO("yolo26n.pt")

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
            #results = self.model.predict(source=frame, conf=0.85, save=True, save_txt=True)
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetection()
    detector.camera_infer()