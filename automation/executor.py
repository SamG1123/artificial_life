from automation import browser_control
import sys
import os
import json
import math
import time
import hashlib
import subprocess
import base64
import warnings
from io import BytesIO
from queue import Full
import easyocr
import pyttsx3
import speech_recognition as sr

try:
    import pygetwindow as gw
except ImportError:
    gw = None

warnings.filterwarnings("ignore", message=".*pin_memory.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import planner_model
import pyautogui as pag
from config import global_ss_queue
from image_processing import ObjectDetection
import numpy as np
import reasoning_model
from reasoning import ReasoningEngine
from decision import SafetyValidator, SafetyVerdict, DecisionRouter
from action import AppController, SystemController, CameraController
from learning import ExperienceLogger
from PIL import Image

pag.PAUSE = 0.15  # small pause between pyautogui calls for reliability

MAX_STEPS = 30
MAX_REPEATS = 2  # auto-recover after this many identical actions
MAX_AUTO_SCROLLS = 3  # cap consecutive auto-scrolls before forcing a reason step
STEP_DELAY = 0.8  # seconds between actions
OCR_MAX_WIDTH = 960  # downscale screenshots for faster OCR


class AutomationExecutor:
    def __init__(self, reasoning_engine: ReasoningEngine | None = None,
                 experience_logger: ExperienceLogger | None = None):
        self.width, self.height = pag.size()
        self.planner = planner_model.Planner()
        self.browser = browser_control.BrowserController()
        self.goal = None
        self.goal_completed = False
        self.vision = ObjectDetection()
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)
        self.element_id = 0
        self.screen_state = {"resolution": (self.width, self.height), "elements": []}
        # Reasoning engine (new) with backward-compat fallback
        self.engine = reasoning_engine or ReasoningEngine()
        self.reason = self.engine  # backward compat — exposes query_model*()
        self.action_history = []
        self._last_screen_hash = None
        self._failed_ids = set()
        self._browser_mode = False  # True once a browser is opened
        self._page_elements = []   # raw Playwright elements (used for clicking)
        self._consecutive_unchanged = 0
        self._auto_scroll_count = 0  # consecutive auto-scroll recoveries
        self._current_screenshot_b64 = None  # base64 screenshot for vision model
        self._last_action_result = ""  # feedback from last action (success/failure/output)

        # Decision system
        self.safety = SafetyValidator()
        self.router = DecisionRouter()

        # Action-layer controllers
        self.app_ctrl = AppController()
        self.sys_ctrl = SystemController()
        self.cam_ctrl = CameraController(vision_model=self.vision)

        # Learning — experience logger
        self.exp_logger = experience_logger or ExperienceLogger()

        # OS info for reasoning model
        self._os_info = f"Windows ({os.name})" if os.name == 'nt' else os.name
        self._username = os.environ.get('USERNAME', os.environ.get('USER', 'user'))
        self._home_dir = os.path.expanduser('~').replace('/', os.sep)
        self._desktop_dir = os.path.join(self._home_dir, 'Desktop')

        # ── Voice support ────────────────────────────────────────────
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 170)
        self.tts_engine.setProperty('volume', 1.0)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # LayoutLMv3 loaded lazily
        self._layout_processor = None
        self._layout_model = None

    @property
    def processor(self):
        if self._layout_processor is None:
            from transformers import LayoutLMv3Processor
            self._layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        return self._layout_processor

    @property
    def layout_model(self):
        if self._layout_model is None:
            from transformers import LayoutLMv3Model
            self._layout_model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
        return self._layout_model

    # ── Screen analysis ──────────────────────────────────────────────

    def capture_screen_state(self):
        """Populate self.screen_state with visible elements.
        
        In browser mode: extract DOM elements via Playwright (fast & accurate).
        Otherwise: take a screenshot and run OCR.
        """
        self.screen_state["elements"].clear()
        self.element_id = 0
        self._page_elements = []

        if self._browser_mode and self.browser.is_running:
            return self._capture_browser_state()
        else:
            return self._capture_desktop_state()

    def _capture_browser_state(self):
        """Use Playwright DOM to get elements — no OCR needed."""
        self._page_elements = self.browser.get_page_elements(max_elements=60)

        for i, el in enumerate(self._page_elements):
            self.element_id += 1
            element = {
                "id": self.element_id,
                "type": el["type"],
                "text": el["text"],
                "href": el.get("href", ""),
                "center": (el["center_x"], el["center_y"]),
                "confidence": 1.0,
            }
            self.screen_state["elements"].append(element)

        # Also grab page context
        self.screen_state["url"] = self.browser.get_current_url()
        self.screen_state["title"] = self.browser.get_page_title()
        self.screen_state["page_text"] = self.browser.get_page_text(max_chars=1500)

        # Return a PIL screenshot for hashing
        try:
            png_bytes = self.browser.take_screenshot()
            return Image.open(__import__('io').BytesIO(png_bytes))
        except Exception:
            return pag.screenshot()

    def _capture_desktop_state(self):
        """Fallback: screenshot + OCR for non-browser screens."""
        screenshot = pag.screenshot()

        try:
            global_ss_queue.put_nowait(screenshot)
        except Full:
            pass

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
            if scale != 1.0:
                box = [[pt[0] / scale, pt[1] / scale] for pt in box]
            x = int(sum([point[0] for point in box]) / 4)
            y = int(sum([point[1] for point in box]) / 4)

            self.element_id += 1
            element = {
                "id": self.element_id,
                "type": self._guess_type_desktop(text),
                "text": text,
                "bbox": box,
                "center": (x, y),
                "confidence": conf,
            }
            self.screen_state["elements"].append(element)

        # Store base64 screenshot for vision model
        self._current_screenshot_b64 = self._screenshot_to_b64(screenshot)

        # Store window context
        self.screen_state["window_context"] = self._get_window_context()

        return screenshot

    def _screenshot_to_b64(self, screenshot) -> str:
        """Resize screenshot and encode as base64 for vision API."""
        if isinstance(screenshot, Image.Image):
            img = screenshot
        else:
            img = Image.fromarray(np.array(screenshot))
        max_w = 1024
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=75)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def _get_window_context(self) -> str:
        """Get info about open windows for screen awareness."""
        if gw is None:
            return "Window information unavailable."
        try:
            active = gw.getActiveWindow()
            if active:
                lines = [
                    f"Active window: \"{active.title}\" "
                    f"(pos: {active.left},{active.top}  size: {active.width}x{active.height})"
                ]
            else:
                lines = ["No active window detected."]

            all_wins = gw.getAllWindows()
            visible = [w.title for w in all_wins
                       if w.title.strip() and w.visible and len(w.title.strip()) > 1]
            if visible:
                lines.append(f"Open windows: {visible[:12]}")
            return "\n".join(lines)
        except Exception:
            return ""

    def _guess_type_desktop(self, text):
        t = text.lower().strip()
        button_words = {"search", "submit", "login", "sign in", "next", "ok",
                        "cancel", "close", "download", "accept", "continue",
                        "save", "open", "yes", "no", "apply", "confirm",
                        "back", "skip", "retry", "install", "update"}
        if t in button_words:
            return "button"
        if t.startswith(("http://", "https://", "www.")):
            return "link"
        word_count = len(t.split())
        if word_count <= 5 and len(t) <= 40:
            return "link"
        return "text"

    # ── Voice helpers ─────────────────────────────────────────────

    def speak(self, message: str):
        """Speak a message aloud via TTS."""
        print(f"  [VOICE] {message}")
        try:
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"  [!] TTS error: {e}")

    def listen_for_confirmation(self, prompt: str, timeout: int = 8) -> bool:
        """Speak a prompt, then listen for 'yes'/'no' via microphone.
        
        Falls back to console input if speech recognition fails.
        Returns True if user confirms, False otherwise.
        """
        self.speak(prompt)
        print(f"  [Waiting for voice response... say 'yes' or 'no']")

        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=5)
            response = self.recognizer.recognize_google(audio).lower().strip()
            print(f"  [Heard] '{response}'")
            if any(w in response for w in ["yes", "yeah", "yep", "sure", "confirm", "go ahead", "do it"]):
                self.speak("Confirmed.")
                return True
            else:
                self.speak("Cancelled.")
                return False
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            print("  [!] Couldn't hear response, falling back to console.")
        except sr.RequestError as e:
            print(f"  [!] Speech recognition error: {e}, falling back to console.")

        # Fallback to console
        try:
            answer = input("  Confirm? (yes/no): ").strip().lower()
            return answer in ("yes", "y", "yeah", "sure")
        except EOFError:
            return False

    @staticmethod
    def normalize_boxes(boxes, image_shape):
        height, width = image_shape[:2]
        normalized = []
        for box in boxes:
            normalized_box = [[x / width, y / height] for x, y in box]
            normalized.append(normalized_box)
        return normalized

    # ── Action dispatch ──────────────────────────────────────────────

    def dispatch_action(self, action_dict: dict) -> bool:
        """Execute a single action. Returns True if goal is complete."""
        action = action_dict.get("action", "done")

        # ── Safety gate ─────────────────────────────────────────
        verdict, risk, reason = self.safety.validate(action_dict)
        domain = self.router.route(action_dict, self._browser_mode)
        print(f"  -> Executing action: {action_dict}  [domain={domain.value}, risk={risk.name}]")

        if verdict == SafetyVerdict.BLOCK:
            print(f"  [BLOCKED] {reason}")
            self._last_action_result = f"BLOCKED: {reason}"
            self.speak(f"I blocked that action for safety: {reason}")
            return False

        if verdict == SafetyVerdict.CONFIRM:
            confirmed = self.listen_for_confirmation(
                f"This action needs your approval: {reason}. Should I proceed?"
            )
            if not confirmed:
                print(f"  [CANCELLED] User declined: {reason}")
                self._last_action_result = f"CANCELLED by user: {reason}"
                return False

        if action == "click":
            return self._do_click(action_dict.get("target_id"))

        elif action == "type":
            self._do_type(action_dict.get("text", ""))

        elif action == "press_key":
            key = action_dict.get("key", "Enter")
            print(f"    Pressing key: {key}")
            if self._browser_mode:
                self.browser.press_key(key)
            else:
                pag.press(key)

        elif action == "scroll":
            self._do_scroll(action_dict.get("direction", "down"))

        elif action == "open_app":
            self._do_open_app(action_dict.get("app_name", ""))

        elif action == "open_browser":
            self._do_open_browser(action_dict.get("query", ""))

        elif action == "navigate":
            url = action_dict.get("url", "")
            if url and self._browser_mode:
                print(f"    Navigating to: {url}")
                self.browser.navigate(url)
                time.sleep(1)

        elif action == "download":
            return self._do_download(action_dict.get("target_id"))

        elif action == "go_back":
            if self._browser_mode:
                print("    Going back")
                self.browser.go_back()

        elif action == "mouse_drag":
            self._do_mouse_drag(action_dict)

        elif action == "mouse_click_xy":
            x, y = action_dict.get("x", 0), action_dict.get("y", 0)
            print(f"    Clicking at ({x}, {y})")
            pag.click(x, y)
            time.sleep(0.3)
            self._last_action_result = f"Clicked at ({x}, {y})"

        elif action == "hotkey":
            keys = action_dict.get("keys", [])
            if keys:
                print(f"    Hotkey: {'+'.join(keys)}")
                pag.hotkey(*keys)
                time.sleep(0.3)
                self._last_action_result = f"Pressed {'+'.join(keys)}"

        elif action == "run_command":
            self._do_run_command(action_dict.get("command", ""))

        elif action == "draw_plan":
            self._do_draw_plan(action_dict.get("subject", ""))

        elif action == "done":
            print("  [OK] Goal marked as done by reasoning model.")
            # Announce what was found
            page_text = self.screen_state.get("page_text", "")
            if page_text and self._browser_mode:
                # Give a brief summary of what's on the page
                summary = page_text[:300].replace("\n", " ").strip()
                self.speak(f"I found what you were looking for. Here's a summary: {summary}")
            return True

        else:
            print(f"  [!] Unknown action '{action}', skipping.")

        return False

    def _do_click(self, target_id):
        if target_id is None:
            print("  [!] click action missing target_id")
            return False
        element = self._find_element(target_id)
        if not element:
            print(f"  [!] Element with id {target_id} not found")
            return False

        if self._browser_mode:
            # Use Playwright — index into _page_elements (id is 1-based)
            idx = target_id - 1
            result = self.browser.click_element(idx, self._page_elements)
            print(f"    {result}")
        else:
            x, y = element["center"]
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))
            print(f"    Clicking \"{element['text'][:50]}\" at ({x}, {y})")
            pag.moveTo(x, y, duration=0.15)
            time.sleep(0.1)
            pag.click()
            time.sleep(0.8)
        return False

    def _do_download(self, target_id):
        """Download a file (confirmation already handled by SafetyValidator)."""
        if target_id is None:
            print("  [!] download action missing target_id")
            return False
        if not self._browser_mode:
            print("  [!] download only works in browser mode")
            return False

        element = self._find_element(target_id)
        if not element:
            print(f"  [!] Element with id {target_id} not found")
            return False

        idx = target_id - 1
        result = self.browser.download_file(idx, self._page_elements)
        print(f"    {result}")
        if result.startswith("Downloaded:"):
            self.speak(f"Download complete. {result}")
            return True
        else:
            self.speak("Download failed. I'll try another option.")
            return False

    def _do_type(self, text: str):
        if not text:
            print("  [!] type action has empty text")
            return
        print(f"    Typing: \"{text}\"")
        if self._browser_mode:
            self.browser.type_text(text)
        else:
            pag.typewrite(text, interval=0.03)

    def _do_scroll(self, direction: str):
        print(f"    Scrolling {direction}")
        if self._browser_mode:
            self.browser.scroll_page(direction)
        else:
            amount = -3 if direction == "down" else 3
            pag.scroll(amount)

    def _do_open_app(self, app_name: str):
        self._last_action_result = self.app_ctrl.open_app(app_name)

    def _do_open_browser(self, query: str):
        if not query:
            print("  [!] open_browser action missing query")
            return
        print(f"    Opening browser and searching: {query}")
        if not self.browser.is_running:
            self.browser.start_browser()
        self.browser.search(query)
        self._browser_mode = True
        time.sleep(1.5)

    def _find_element(self, target_id: int):
        for el in self.screen_state["elements"]:
            if el["id"] == target_id:
                return el
        return None

    # ── General-purpose actions ─────────────────────────────────────────

    def _do_mouse_drag(self, action_dict: dict):
        """Drag the mouse from (x1,y1) to (x2,y2)."""
        x1 = action_dict.get("x1", 0)
        y1 = action_dict.get("y1", 0)
        x2 = action_dict.get("x2", 0)
        y2 = action_dict.get("y2", 0)
        print(f"    Dragging ({x1},{y1}) -> ({x2},{y2})")
        pag.moveTo(x1, y1, duration=0.1)
        pag.mouseDown()
        pag.moveTo(x2, y2, duration=0.15)
        pag.mouseUp()
        time.sleep(0.2)

    def _do_run_command(self, command: str):
        """Run a system/shell command via SystemController."""
        self._last_action_result = self.sys_ctrl.run_command(command)

    def _do_draw_plan(self, subject: str):
        """Generate a drawing plan via LLM and execute it as mouse strokes.
        
        This is a compound action: it asks the LLM for a sequence of
        drawing commands, then executes them all in the current app
        (typically Paint). No separate module needed.
        """
        if not subject:
            print("  [!] draw_plan missing subject")
            return
        print(f"    Generating drawing plan for: {subject}")

        # Detect the active window's canvas area
        try:
            import pygetwindow as gw_mod
            win = gw_mod.getActiveWindow()
            if win:
                wl, wt, ww, wh = win.left, win.top, win.width, win.height
                canvas_left = wl + 2
                canvas_top = wt + 160      # below ribbon/toolbar
                canvas_w = ww - 20
                canvas_h = wh - 160 - 45   # minus status bar
            else:
                canvas_left, canvas_top = 80, 200
                canvas_w, canvas_h = self.width - 100, self.height - 260
        except Exception:
            canvas_left, canvas_top = 80, 200
            canvas_w, canvas_h = self.width - 100, self.height - 260

        LOGICAL_W, LOGICAL_H = 1000, 600

        # Ask LLM for drawing instructions
        prompt = f"""You are a drawing instruction generator. Draw: "{subject}"

Generate a JSON array of drawing commands for mouse movements on a canvas.
The canvas is {LOGICAL_W}x{LOGICAL_H} pixels, (0,0) at top-left, center at ({LOGICAL_W//2},{LOGICAL_H//2}).

Commands:
1. {{"cmd":"color","r":R,"g":G,"b":B}} — set color. Use: black(0,0,0), red(237,28,36), green(34,177,76), blue(63,72,204), yellow(255,242,0), orange(255,127,39), purple(163,73,164), pink(255,174,201), brown(185,122,87)
2. {{"cmd":"line","x1":X1,"y1":Y1,"x2":X2,"y2":Y2}} — straight line
3. {{"cmd":"polyline","points":[[x1,y1],[x2,y2],...]}} — connected line segments (use 10-20 points for curves)
4. {{"cmd":"circle","cx":CX,"cy":CY,"r":RADIUS}} — circle

Rules:
- SIMPLE and RECOGNIZABLE. Max 25 commands.
- Stay within bounds: x in [30,{LOGICAL_W-30}], y in [30,{LOGICAL_H-30}].
- Make drawing CENTERED and LARGE (60%+ of canvas).
- Output ONLY the JSON array."""

        try:
            response = self.reason.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Output only valid JSON arrays. No markdown."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=3000,
                temperature=0.4,
            )
            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            plan = json.loads(raw)
        except Exception as e:
            print(f"    [!] Failed to generate drawing plan: {e}")
            return

        if not isinstance(plan, list) or not plan:
            print("    [!] Empty or invalid drawing plan")
            return

        # Helper: convert logical coords to screen coords
        def to_screen(x, y):
            sx = canvas_left + int(x / LOGICAL_W * canvas_w)
            sy = canvas_top + int(y / LOGICAL_H * canvas_h)
            sx = max(canvas_left + 5, min(sx, canvas_left + canvas_w - 5))
            sy = max(canvas_top + 5, min(sy, canvas_top + canvas_h - 5))
            return sx, sy

        # Click on canvas center first to ensure focus
        cx, cy = to_screen(LOGICAL_W // 2, LOGICAL_H // 2)
        pag.press('escape')  # dismiss any dialog
        time.sleep(0.15)
        pag.click(cx, cy)
        time.sleep(0.3)

        # Color palette mapping (closest match from Paint's 20 presets)
        palette = [
            (0,0,0), (127,127,127), (136,0,21), (237,28,36), (255,127,39),
            (255,242,0), (34,177,76), (0,162,232), (63,72,204), (163,73,164),
            (255,255,255), (195,195,195), (185,122,87), (255,174,201), (255,201,14),
            (239,228,176), (181,230,29), (153,217,234), (112,146,190), (200,191,231),
        ]

        def pick_palette_color(r, g, b):
            best_i = 0
            best_d = float('inf')
            for i, (pr, pg, pb) in enumerate(palette):
                d = (r-pr)**2 + (g-pg)**2 + (b-pb)**2
                if d < best_d:
                    best_d = d
                    best_i = i
            # Click the right swatch in Paint's ribbon
            try:
                active_win = gw_mod.getActiveWindow()
                if active_win:
                    wl2 = active_win.left
                    wt2 = active_win.top
                    ww2 = active_win.width
                    p_start_x = wl2 + int(ww2 * 0.52)
                    sw = max(16, int(ww2 * 0.018))
                    row = best_i // 10
                    col = best_i % 10
                    px = p_start_x + col * sw + sw // 2
                    py = (wt2 + 35) if row == 0 else (wt2 + 55)
                    pag.click(px, py)
                    time.sleep(0.15)
                    # Click back on canvas
                    pag.click(cx, cy)
                    time.sleep(0.1)
            except Exception:
                pass

        # Execute the drawing plan
        saved_pause = pag.PAUSE
        pag.PAUSE = 0.05  # speed up for drawing strokes

        for i, cmd in enumerate(plan):
            action = cmd.get("cmd", "")
            print(f"    [draw {i+1}/{len(plan)}] {action}")
            try:
                if action == "color":
                    pick_palette_color(cmd.get("r",0), cmd.get("g",0), cmd.get("b",0))

                elif action == "line":
                    s1 = to_screen(cmd["x1"], cmd["y1"])
                    s2 = to_screen(cmd["x2"], cmd["y2"])
                    pag.moveTo(*s1, duration=0.05)
                    pag.mouseDown()
                    pag.moveTo(*s2, duration=0.1)
                    pag.mouseUp()

                elif action == "polyline":
                    pts = cmd.get("points", [])
                    if len(pts) >= 2:
                        s = to_screen(pts[0][0], pts[0][1])
                        pag.moveTo(*s, duration=0.05)
                        pag.mouseDown()
                        for pt in pts[1:]:
                            s = to_screen(pt[0], pt[1])
                            pag.moveTo(*s, duration=0.02)
                        pag.mouseUp()

                elif action == "circle":
                    ccx, ccy = cmd.get("cx", LOGICAL_W//2), cmd.get("cy", LOGICAL_H//2)
                    cr = cmd.get("r", 50)
                    segs = cmd.get("segments", 24)
                    pts = []
                    for j in range(segs + 1):
                        angle = 2 * math.pi * j / segs
                        px = int(ccx + cr * math.cos(angle))
                        py = int(ccy + cr * math.sin(angle))
                        pts.append([px, py])
                    s = to_screen(pts[0][0], pts[0][1])
                    pag.moveTo(*s, duration=0.05)
                    pag.mouseDown()
                    for pt in pts[1:]:
                        s = to_screen(pt[0], pt[1])
                        pag.moveTo(*s, duration=0.02)
                    pag.mouseUp()

            except Exception as e:
                print(f"    [!] Draw error: {e}")
            time.sleep(0.08)

        pag.PAUSE = saved_pause  # restore
        print("    Drawing complete.")
        self._last_action_result = f"SUCCESS: drew {subject} with {len(plan)} strokes"

    # ── Main loop ────────────────────────────────────────────────────

    def execute_goal(self, goal: str):
        """Run the full autonomy loop: observe -> reason -> act -> repeat."""
        self.goal = goal
        self.goal_completed = False
        self.action_history.clear()
        self._failed_ids.clear()
        self._last_screen_hash = None
        self._consecutive_unchanged = 0
        self._auto_scroll_count = 0
        step = 0

        # Let the reasoning engine create a plan for this goal
        if hasattr(self.engine, 'set_goal'):
            plan = self.engine.set_goal(goal)
            print(f"\n  Plan: {plan.summary()}\n")

        print(f"\n{'='*60}")
        print(f"  GOAL: {goal}")
        print(f"{'='*60}\n")
        self.speak(f"Starting goal: {goal}")

        # Auto-open browser for goals that clearly need web search
        if self._goal_needs_browser(goal) and not self.browser.is_running:
            print("  [Auto] Goal requires web browsing — opening browser...")
            self._do_open_browser(goal)  # searches the goal directly
            step = 1
            self.action_history.append({
                "step": 1,
                "action": {"action": "open_browser", "query": goal},
                "result": "auto_opened",
                "url": self.screen_state.get("url", ""),
            })

        while not self.goal_completed and step < MAX_STEPS:
            step += 1
            print(f"\n-- Step {step}/{MAX_STEPS} --")

            # 1. Observe
            print("  Capturing screen state...")
            screenshot = self.capture_screen_state()
            screen_hash = self._hash_screenshot(screenshot)
            screen_changed = (screen_hash != self._last_screen_hash)
            self._last_screen_hash = screen_hash
            
            if screen_changed:
                self._consecutive_unchanged = 0
                self._failed_ids.clear()  # new screen = new element IDs
            else:
                self._consecutive_unchanged += 1

            elem_count = len(self.screen_state['elements'])
            extra_info = ""
            if self._browser_mode:
                extra_info = f"  URL: {self.screen_state.get('url', '?')}"
            print(f"  Found {elem_count} elements  "
                  f"(screen {'changed' if screen_changed else 'unchanged'})")
            if extra_info:
                print(extra_info)

            # Mark last-clicked element as failed if screen didn't change
            if not screen_changed and self.action_history:
                last = self.action_history[-1]["action"]
                if last.get("action") in ("click", "download"):
                    failed_id = last.get("target_id")
                    if failed_id is not None:
                        self._failed_ids.add(failed_id)
                        print(f"  [!] Action on [{failed_id}] had no effect, blacklisting")

            # 2. Auto-recover if stuck
            if self._is_stuck() and self._auto_scroll_count < MAX_AUTO_SCROLLS:
                self._auto_scroll_count += 1
                print(f"  [!] Stuck -- same action repeated. Auto-scrolling down. "
                      f"({self._auto_scroll_count}/{MAX_AUTO_SCROLLS})")
                self._do_scroll("down")
                # Record the auto-scroll in history so _is_stuck() won't loop
                self.action_history.append({
                    "step": step,
                    "action": {"action": "scroll", "direction": "down", "auto_recovery": True},
                    "result": "auto_recovery_scroll",
                    "url": self.screen_state.get("url", ""),
                })
                time.sleep(0.5)
                continue

            if self._consecutive_unchanged >= 3 and self._auto_scroll_count < MAX_AUTO_SCROLLS:
                self._auto_scroll_count += 1
                print(f"  [!] Screen unchanged for 3 steps. Scrolling to reveal new content. "
                      f"({self._auto_scroll_count}/{MAX_AUTO_SCROLLS})")
                self._do_scroll("down")
                self.action_history.append({
                    "step": step,
                    "action": {"action": "scroll", "direction": "down", "auto_recovery": True},
                    "result": "auto_recovery_scroll",
                    "url": self.screen_state.get("url", ""),
                })
                self._consecutive_unchanged = 0
                time.sleep(0.5)
                continue

            # Reset auto-scroll counter once we reach a normal reasoning step
            self._auto_scroll_count = 0

            # 3. Reason
            print("  Querying reasoning model...")
            filtered_elements = [
                e for e in self.screen_state["elements"]
                if e["id"] not in self._failed_ids
            ]

            history_context = self._build_history_context(screen_changed, step)

            # Desktop mode → use vision model so the AI can SEE the screen
            if not self._browser_mode and self._current_screenshot_b64:
                action_dict = self.reason.query_model_with_vision(
                    goal=self.goal,
                    elements=filtered_elements,
                    screenshot_b64=self._current_screenshot_b64,
                    user_context=history_context,
                    screen_resolution=(self.width, self.height),
                )
            else:
                # Browser mode → DOM elements are precise, text model is fine
                action_dict = self.reason.query_model(
                    goal=self.goal,
                    elements=filtered_elements,
                    user_context=history_context,
                )

            # 4. Act
            self._last_action_result = ""
            self.goal_completed = self.dispatch_action(action_dict)
            result_note = self._last_action_result
            if not result_note and not screen_changed and action_dict.get("action") == "click":
                result_note = "screen_unchanged_after_click"
            self.action_history.append({
                "step": step,
                "action": action_dict,
                "result": result_note,
                "url": self.screen_state.get("url", ""),
            })

            # Log experience for the learning subsystem
            domain = self.router.route(action_dict, self._browser_mode)
            _, risk, _ = self.safety.validate(action_dict)
            plan_step = None
            if hasattr(self.engine, '_current_plan') and self.engine._current_plan:
                cs = self.engine._current_plan.current_step
                plan_step = cs.to_dict() if cs else None
            self.exp_logger.log(
                goal=self.goal,
                step=step,
                world_state=self.screen_state,
                action=action_dict,
                result=result_note,
                success=self.goal_completed or not result_note.startswith("FAILED"),
                domain=domain.value,
                risk=risk.name,
                plan_step=plan_step,
            )

            # Report step result to reasoning engine for plan tracking
            if hasattr(self.engine, 'report_step_result'):
                success = self.goal_completed or not result_note.startswith("FAILED")
                self.engine.report_step_result(
                    action_dict, success, result_note,
                    world_context=history_context,
                )

            time.sleep(STEP_DELAY)

        # Cleanup
        if self.goal_completed:
            msg = f"Goal completed in {step} steps."
            print(f"\n[OK] {msg}")
            self.speak(msg)
        else:
            msg = f"Could not complete the goal after {MAX_STEPS} steps."
            print(f"\n[!] {msg}")
            self.speak(msg)

        if self.browser.is_running:
            self.browser.close_browser()
            self._browser_mode = False

        # Clean up TTS engine safely
        try:
            self.tts_engine.stop()
        except Exception:
            pass

    def _build_history_context(self, screen_changed: bool, step: int) -> str:
        parts = []
        if self.action_history:
            recent = self.action_history[-5:]
            lines = []
            for h in recent:
                lines.append(f"  Step {h['step']}: {json.dumps(h['action'])}")
                if h.get("result"):
                    lines.append(f"    Result: {h['result']}")
                if h.get("url"):
                    lines.append(f"    URL after: {h['url']}")
            parts.append("Previous actions:\n" + "\n".join(lines))

        if self._failed_ids:
            parts.append(f"Do NOT click/download these IDs (already tried, no effect): {sorted(self._failed_ids)}")

        if not screen_changed and step > 1:
            parts.append("IMPORTANT: The screen did NOT change after the last action. "
                         "Try a DIFFERENT element, scroll, or use the 'download' action for file links.")

        # Tell the model what environment we're in
        if self._browser_mode:
            parts.insert(0, "ENVIRONMENT: Browser is open.")
            url = self.screen_state.get("url", "")
            title = self.screen_state.get("title", "")
            parts.append(f"Current page: {title}  |  URL: {url}")

            # Include page text so the model can judge goal completion
            page_text = self.screen_state.get("page_text", "")
            if page_text:
                parts.append(f"PAGE CONTENT (visible text on page):\n{page_text[:1200]}")
        else:
            parts.insert(0, f"ENVIRONMENT: Desktop (no browser open). "
                         f"OS: {self._os_info}. User: {self._username}. "
                         f"Home: {self._home_dir}. Desktop: {self._desktop_dir}. "
                         f"Use open_browser to search the web if the goal needs internet.")
            # Add window awareness for desktop mode
            win_ctx = self.screen_state.get("window_context", "")
            if win_ctx:
                parts.append(f"WINDOW INFO:\n{win_ctx}")
            parts.append(f"Screen resolution: {self.width}x{self.height}")

        return "\n\n".join(parts)

    @staticmethod
    def _goal_needs_browser(goal: str) -> bool:
        """Heuristic: does this goal require a web browser?"""
        web_keywords = [
            "find", "search", "look up", "lookup", "google", "browse",
            "recipe", "weather", "news", "website", "download", "pdf",
            "how to", "what is", "who is", "where is", "wiki",
            "buy", "price", "shop", "order", "book", "watch",
            "youtube", "video", "article", "blog", "reddit",
            "stackoverflow", "github", "online",
        ]
        g = goal.lower()
        return any(kw in g for kw in web_keywords)

    def _hash_screenshot(self, screenshot) -> str:
        if isinstance(screenshot, Image.Image):
            small = screenshot.resize((64, 64)).convert("L")
        else:
            small = Image.fromarray(np.array(screenshot)).resize((64, 64)).convert("L")
        return hashlib.md5(small.tobytes()).hexdigest()

    def _is_stuck(self) -> bool:
        if len(self.action_history) < MAX_REPEATS:
            return False
        recent = [json.dumps(h["action"], sort_keys=True)
                  for h in self.action_history[-MAX_REPEATS:]]
        return len(set(recent)) == 1

    # ── Legacy helpers ───────────────────────────────────────────────

    def prepare_input(self, ocr_results, screenshot):
        words = [e['text'] for e in ocr_results]
        boxes = self.normalize_boxes([e['bbox'] for e in ocr_results], screenshot.shape)
        encoding = self.processor(
            images=screenshot,
            text=words,
            boxes=boxes,
            return_tensors="pt",
        )
        outputs = self.layout_model(**encoding)
        return outputs


if __name__ == "__main__":
    executor = AutomationExecutor()
    user_goal = input("Enter your goal: ").strip()
    if user_goal:
        executor.execute_goal(user_goal)
    else:
        print("No goal provided.")
        