import app_control
import browser_control
import file_control
import system_control
import sys
import os
from groq import Groq
import easyocr
from transformers import LayoutLMv3Processor, LayoutLMv3Model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import planner_model
import pyautogui as pag
from config import global_ss_queue
from image_processing import ObjectDetection
import numpy as np





class AutomationExecutor:
    def __init__(self):
        self.width, self.height = pag.size()
        self.planner = planner_model.Planner()
        self.app_controller = app_control.AppControl()
        self.browser_controller = browser_control.BrowserController()
        #self.file_controller = file_control.FileController()
        #self.system_controller = system_control.SystemController()
        self.goal = False
        self.vision = ObjectDetection()
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)
        self.element_id = 0
        self.screen_state = {"resolution": (self.width, self.height), "elements": [] }

        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        self.model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
    
    def execute_goal(self, goal: str):
        plan = self.planner.create_plan(goal)
        print("Generated Plan:", plan)
        if not self.planner.validate_plan(plan):
            raise ValueError("Generated plan contains invalid actions.")
        
        for step in plan.get("steps", []):
            action = step["action"]
            target = step.get("target", "")
            self.execute_step(action, target)
    
    def guess_type(self, text):
        t = text.lower()

        if t in ["search", "submit", "login", "sign in", "next"]:
            return "button"

        if len(t) > 25:
            return "link"

        return "text"


    def normalize_boxes(boxes, image_shape):
        """Normalize bounding boxes to [0, 1] range based on image shape."""
        height, width = image_shape[:2]
        normalized = []
        for box in boxes:
            normalized_box = [[x / width, y / height] for x, y in box]
            normalized.append(normalized_box)
        return normalized


    def execute_step(self):
        while not self.goal:
            if global_ss_queue.empty():
                screenshot = pag.screenshot()
                global_ss_queue.put(screenshot)
            
                text_elements = self.ocr_reader.readtext(np.array(screenshot))
                
                for box, text, conf in text_elements:
                    if conf < 0.5:
                        continue
                    
                    x = int(sum([point[0] for point in box]) / 4)
                    y = int(sum([point[1] for point in box]) / 4)

                    self.element_id += 1
                    element = {
                        "id": self.element_id,
                        "type": self.guess_type(text),
                        "text": text,
                        "bbox": box,
                        "center": (x, y),
                        "confidence": conf
                    }
                    self.screen_state["elements"].append(element)
                
                print("Extracted Text Elements:", self.screen_state["elements"])

                #outputs = self.prepare_input(text_elements, np.array(screenshot))

                self.goal = True

            else:
                global_ss_queue.task_done()
                self.screen_state["elements"].clear()
                self.element_id = 0

    
    def prepare_input(self, ocr_results, screenshot):
        words = [e['text'] for e in ocr_results]
        boxes = self.normalize_boxes([e['bbox'] for e in ocr_results], screenshot.shape)
        encoding = self.processor(
            images = screenshot,
            text = words,
            boxes = boxes,
            return_tensors="pt"
        )

        outputs = self.model(**encoding)

    def reasoning_model(self, goal, query):
        pass



if __name__ == "__main__":
    executor = AutomationExecutor()
    executor.execute_step()
        