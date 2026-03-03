import app_control
import browser_control
import file_control
import system_control
import sys
import os
import json
import time
import hashlib
import warnings
from queue import Full
import easyocr

warnings.filterwarnings("ignore", message=".*pin_memory.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import planner_model
import pyautogui as pag
from config import global_ss_queue
from image_processing import ObjectDetection
import numpy as np
import reasoning_model
from PIL import Image

pag.PAUSE = 0.15  # small pause between pyautogui calls for reliability

MAX_STEPS = 30
MAX_REPEATS = 2  # auto-recover after this many identical actions
STEP_DELAY = 1.0  # seconds between actions
OCR_MAX_WIDTH = 960  # downscale screenshots for faster OCR


class AutomationExecutor:
    def __init__(self):
        self.width, self.height = pag.size()
        self.planner = planner_model.Planner()
        self.app_controller = app_control.AppControl()
        self.browser_controller = browser_control.BrowserController()
        self.goal = None
        self.goal_completed = False
        self.vision = ObjectDetection()
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)
        self.element_id = 0
        self.screen_state = {"resolution": (self.width, self.height), "elements": []}
        self.reason = reasoning_model.ReasoningModel()
        self.action_history = []  # track past actions for context
        self._last_screen_hash = None  # for screen-change detection
        self._failed_ids = set()  # element IDs that were clicked but didn't change anything

        # LayoutLMv3 loaded lazily — only when prepare_input() is called
        self._layout_processor = None
        self._layout_model = None

    @property
    def processor(self):
        if self._layout_processor is None:
            from transformers import LayoutLMv3Processor
            self._layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        return self._layout_processor

    @property
    def model(self):
        if self._layout_model is None:
            from transformers import LayoutLMv3Model
            self._layout_model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
        return self._layout_model

    # ── Screen analysis ──────────────────────────────────────────────

    def capture_screen_state(self):
        """Take a screenshot, run OCR, and populate self.screen_state."""
        self.screen_state["elements"].clear()
        self.element_id = 0

        screenshot = pag.screenshot()

        # Non-blocking put — don't hang if nobody is consuming
        try:
            global_ss_queue.put_nowait(screenshot)
        except Full:
            pass  # queue full, skip — old screenshot is still there

        # Downscale for faster OCR while keeping coordinate mapping
        img_array = np.array(screenshot)
        orig_h, orig_w = img_array.shape[:2]

        if orig_w > OCR_MAX_WIDTH:
            scale = OCR_MAX_WIDTH / orig_w
            small_w = OCR_MAX_WIDTH
            small_h = int(orig_h * scale)
            small_img = np.array(screenshot.resize((small_w, small_h), Image.LANCZOS))
        else:
            scale = 1.0
            small_img = img_array

        text_elements = self.ocr_reader.readtext(small_img)

        for box, text, conf in text_elements:
            if conf < 0.4:
                continue

            # Map coordinates back to original resolution
            if scale != 1.0:
                box = [[pt[0] / scale, pt[1] / scale] for pt in box]

            x = int(sum([point[0] for point in box]) / 4)
            y = int(sum([point[1] for point in box]) / 4)

            self.element_id += 1
            element = {
                "id": self.element_id,
                "type": self.guess_type(text),
                "text": text,
                "bbox": box,
                "center": (x, y),
                "confidence": conf,
            }
            self.screen_state["elements"].append(element)

        return screenshot

    def guess_type(self, text):
        t = text.lower().strip()
        # Common button labels
        button_words = {"search", "submit", "login", "sign in", "next", "ok",
                        "cancel", "close", "download", "accept", "continue",
                        "save", "open", "yes", "no", "apply", "confirm",
                        "back", "skip", "retry", "install", "update"}
        if t in button_words:
            return "button"
        # URLs are links
        if t.startswith(("http://", "https://", "www.")):
            return "link"
        # Short phrases that look like navigation / menu items
        word_count = len(t.split())
        if word_count <= 5 and len(t) <= 40:
            return "link"
        # Everything else is body text (not clickable)
        return "text"

    @staticmethod
    def normalize_boxes(boxes, image_shape):
        """Normalize bounding boxes to [0, 1] range based on image shape."""
        height, width = image_shape[:2]
        normalized = []
        for box in boxes:
            normalized_box = [[x / width, y / height] for x, y in box]
            normalized.append(normalized_box)
        return normalized

    # ── Action dispatch ──────────────────────────────────────────────

    def dispatch_action(self, action_dict: dict) -> bool:
        """Execute a single action returned by the reasoning model.
        
        Returns True if the goal is complete (action == 'done').
        """
        action = action_dict.get("action", "done")
        print(f"  → Executing action: {action_dict}")

        if action == "click":
            self._do_click(action_dict.get("target_id"))

        elif action == "type":
            self._do_type(action_dict.get("text", ""))

        elif action == "scroll":
            self._do_scroll(action_dict.get("direction", "down"))

        elif action == "open_app":
            self._do_open_app(action_dict.get("app_name", ""))

        elif action == "open_browser":
            self._do_open_browser(action_dict.get("query", ""))

        elif action == "done":
            print("  ✓ Goal marked as done by reasoning model.")
            return True

        else:
            print(f"  ⚠ Unknown action '{action}', skipping.")

        return False

    def _do_click(self, target_id):
        if target_id is None:
            print("  ⚠ click action missing target_id")
            return
        element = self._find_element(target_id)
        if element:
            x, y = element["center"]
            # Clamp to screen bounds
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            print(f"    Clicking element [{target_id}] \"{element['text']}\" at ({x}, {y})")
            # Move first, then click — more reliable than instant click
            pag.moveTo(x, y, duration=0.15)
            time.sleep(0.1)
            pag.click()
            # Wait for any page navigation / UI change
            time.sleep(0.8)
        else:
            print(f"  ⚠ Element with id {target_id} not found on screen")

    def _do_type(self, text: str):
        if not text:
            print("  ⚠ type action has empty text")
            return
        print(f"    Typing: \"{text}\"")
        pag.typewrite(text, interval=0.03)

    def _do_scroll(self, direction: str):
        amount = -3 if direction == "down" else 3
        print(f"    Scrolling {direction}")
        pag.scroll(amount)

    def _do_open_app(self, app_name: str):
        if not app_name:
            print("  ⚠ open_app action missing app_name")
            return
        print(f"    Opening app: {app_name}")
        self.app_controller.open_app(app_name)

    def _do_open_browser(self, query: str):
        if not query:
            print("  ⚠ open_browser action missing query")
            return
        print(f"    Opening browser and searching: {query}")
        # Only start the browser if it isn't already running
        if self.browser_controller.browser is None:
            self.browser_controller.start_browser()
        self.browser_controller.search(query)
        # Give the page time to load before next screenshot
        time.sleep(2)

    def _find_element(self, target_id: int):
        for el in self.screen_state["elements"]:
            if el["id"] == target_id:
                return el
        return None

    # ── Main loop ────────────────────────────────────────────────────

    def execute_goal(self, goal: str):
        """Run the full autonomy loop: observe → reason → act → repeat."""
        self.goal = goal
        self.goal_completed = False
        self.action_history.clear()
        self._failed_ids.clear()
        self._last_screen_hash = None
        step = 0

        print(f"\n{'='*60}")
        print(f"  GOAL: {goal}")
        print(f"{'='*60}\n")

        while not self.goal_completed and step < MAX_STEPS:
            step += 1
            print(f"\n── Step {step}/{MAX_STEPS} ──")

            # 1. Observe
            print("  Capturing screen state...")
            screenshot = self.capture_screen_state()
            screen_hash = self._hash_screenshot(screenshot)
            screen_changed = (screen_hash != self._last_screen_hash)
            self._last_screen_hash = screen_hash
            print(f"  Found {len(self.screen_state['elements'])} elements  "
                  f"(screen {'changed' if screen_changed else 'unchanged'})")

            # Mark last-clicked element as failed if screen didn't change
            if not screen_changed and self.action_history:
                last = self.action_history[-1]["action"]
                if last.get("action") == "click":
                    failed_id = last.get("target_id")
                    if failed_id is not None:
                        self._failed_ids.add(failed_id)
                        print(f"  ⚠ Click on [{failed_id}] had no effect, marking as non-clickable")

            # 2. Detect repetition and auto-recover
            if self._is_stuck():
                print("  ⚠ Stuck — same action repeated. Auto-scrolling down.")
                pag.scroll(-5)
                time.sleep(0.8)
                continue

            # 3. Reason
            print("  Querying reasoning model...")
            # Filter out elements the model shouldn't try clicking again
            filtered_elements = [
                e for e in self.screen_state["elements"]
                if e["id"] not in self._failed_ids
            ]

            history_context = ""
            if self.action_history:
                recent = self.action_history[-5:]
                lines = []
                for h in recent:
                    lines.append(f"  Step {h['step']}: {json.dumps(h['action'])}")
                    if h.get("result"):
                        lines.append(f"    Result: {h['result']}")
                history_context = "Previous actions:\n" + "\n".join(lines)
            if self._failed_ids:
                history_context += f"\n\nDo NOT click these IDs (already tried, no effect): {list(self._failed_ids)}"
            if not screen_changed and step > 1:
                history_context += "\n\nIMPORTANT: The screen did NOT change after the last action. Try a DIFFERENT element or scroll."

            action_dict = self.reason.query_model(
                goal=self.goal,
                elements=filtered_elements,
                user_context=history_context,
            )

            # 4. Act
            result_note = ""
            self.goal_completed = self.dispatch_action(action_dict)
            if not screen_changed and action_dict.get("action") == "click":
                result_note = "screen_unchanged_after_click"
            self.action_history.append({"step": step, "action": action_dict, "result": result_note})

            # Brief pause so the UI can react
            time.sleep(STEP_DELAY)

        if self.goal_completed:
            print(f"\n✓ Goal completed in {step} steps.")
        else:
            print(f"\n⚠ Reached max steps ({MAX_STEPS}) without completing the goal.")

    def _hash_screenshot(self, screenshot) -> str:
        """Compute a fast perceptual hash of a screenshot to detect changes."""
        small = screenshot.resize((64, 64)).convert("L")
        return hashlib.md5(small.tobytes()).hexdigest()

    def _is_stuck(self) -> bool:
        """Return True if the last MAX_REPEATS actions are identical."""
        if len(self.action_history) < MAX_REPEATS:
            return False
        recent = [json.dumps(h["action"], sort_keys=True)
                  for h in self.action_history[-MAX_REPEATS:]]
        return len(set(recent)) == 1

    # ── Legacy helpers (kept for compatibility) ──────────────────────

    def prepare_input(self, ocr_results, screenshot):
        words = [e['text'] for e in ocr_results]
        boxes = self.normalize_boxes([e['bbox'] for e in ocr_results], screenshot.shape)
        encoding = self.processor(
            images=screenshot,
            text=words,
            boxes=boxes,
            return_tensors="pt",
        )
        outputs = self.model(**encoding)
        return outputs


if __name__ == "__main__":
    executor = AutomationExecutor()
    user_goal = input("Enter your goal: ").strip()
    if user_goal:
        executor.execute_goal(user_goal)
    else:
        print("No goal provided.")
        