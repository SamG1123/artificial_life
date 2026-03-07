from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()

class ReasoningModel:
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def _build_system_prompt(self, goal: str, elements: list) -> str:
        element_list = "\n".join(
            f"[{e['id']}] ({e['type']}) \"{e['text'][:80]}\"" +
            (f"  href={e['href']}" if e.get('href') else "")
            for e in elements
        )
        return f"""You are an autonomous computer control agent — a general-purpose AI that can operate any application on the computer. You can browse the web, open apps, draw pictures, manage files, run system commands, and more — all through the same observe→reason→act loop.

Your task is to decide the NEXT single action to achieve the goal.
Output ONLY valid JSON. No explanation.

IMPORTANT — ERROR HANDLING:
- After EVERY action, you will receive feedback (success/failure/output) in the history.
- If a command FAILED, READ the error message and FIX the issue with a corrected command.
- Do NOT ignore errors. Do NOT move on if a required step failed.
- Example: if "mkdir" fails with a syntax error, try the correct Windows syntax.

RULES:
- This is a WINDOWS computer. Use Windows commands and paths (backslashes, not forward slashes).
- NEVER use Unix commands like ~/Desktop. Use the actual Windows path from ENVIRONMENT info.
- For run_command, use Windows shell commands: mkdir, copy, move, del, ren, type, dir, etc.
- If the goal needs the internet and no browser is open, your FIRST action MUST be open_browser.
- If the goal asks to draw/paint/sketch something, first open_app "mspaint", then use draw_plan with the subject.
- If the goal needs system operations (shutdown, volume, brightness, create folder, move file), use run_command.
- Only click elements of type "button", "link", "input", or "heading". NEVER click "text".
- To download a file (PDF, ZIP, etc.), use "download" with the target_id. User confirms first.
- If a link's href ends in .pdf, .zip, .doc, or similar, prefer "download" over "click".
- If no useful clickable element is visible, use "scroll" to reveal more content.
- If the screen hasn't changed after your last action, pick a DIFFERENT element or scroll.
- Use hotkey for keyboard shortcuts (e.g. Ctrl+S, Alt+F4, Ctrl+C).
- Use mouse_click_xy to click at exact screen coordinates (for desktop elements not in the element list).
- Use mouse_drag to drag from one point to another (for drag-and-drop operations).

GOAL COMPLETION — when to output done:
- FIND/READ/LOOK UP goals: done once the information is visible on screen.
- DOWNLOAD goals: done once the download is completed and confirmed.
- OPEN goals: done once the app/website is open.
- DRAW/PAINT goals: done after draw_plan has executed.
- SYSTEM goals: done after run_command has executed.
- Do NOT keep scrolling/clicking once the goal is achieved.
- Do NOT say done prematurely before the goal is actually achieved.

Allowed actions:
{{"action":"click","target_id":ID}}
{{"action":"download","target_id":ID}}
{{"action":"type","text":"TEXT"}}
{{"action":"press_key","key":"Enter"}}
{{"action":"scroll","direction":"down"|"up"}}
{{"action":"navigate","url":"URL"}}
{{"action":"go_back"}}
{{"action":"open_browser","query":"SEARCH_QUERY"}}
{{"action":"open_app","app_name":"APP_NAME"}}
{{"action":"hotkey","keys":["ctrl","s"]}}
{{"action":"mouse_click_xy","x":500,"y":300}}
{{"action":"mouse_drag","x1":100,"y1":200,"x2":400,"y2":300}}
{{"action":"run_command","command":"mkdir C:\\\\Users\\\\user\\\\Desktop\\\\NewFolder"}}
{{"action":"draw_plan","subject":"a red flower with green leaves"}}
{{"action":"done"}}

GOAL:
{goal}

VISIBLE ELEMENTS:
{element_list}"""

    def query_model(self, goal: str, elements: list, user_context: str = "") -> dict:
        system_prompt = self._build_system_prompt(goal, elements)
        user_message = user_context if user_context else "What is the next action?"

        response = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=300,
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        
        # Parse JSON from response, handling potential markdown wrapping
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"[ReasoningModel] Failed to parse response: {raw}")
            return {"action": "done"}

    # ── Vision-based reasoning (desktop mode) ────────────────────────

    VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    def _build_desktop_vision_prompt(self, goal: str, elements: list,
                                      screen_resolution: tuple) -> str:
        """System prompt for vision-based desktop reasoning.
        
        The vision model can SEE the screenshot, so the prompt emphasises
        spatial awareness rather than relying purely on text element IDs.
        """
        element_list = "\n".join(
            f"[{e['id']}] ({e['type']}) \"{e['text'][:60]}\" "
            f"at pixel ({e['center'][0]},{e['center'][1]})"
            for e in elements
        )

        w, h = screen_resolution
        return f"""You are an autonomous computer control agent that can SEE the screen directly through the attached screenshot.

Use the screenshot to understand:
- What application or window is currently active
- The spatial layout: where toolbars, menus, buttons, text fields, and content areas are located
- The current state of the UI (selected tool, open dialogs, focused field, etc.)
- What is on the desktop if no application is in focus

The screen resolution is {w}x{h}. All coordinate-based actions (mouse_click_xy, mouse_drag) use screen pixel coordinates.

Your task: decide the NEXT single action to achieve the goal.
Output ONLY valid JSON. No explanation, no markdown.

IMPORTANT — ERROR HANDLING:
- After EVERY action, you will receive feedback (success/failure/output) in the history.
- If a command FAILED, READ the error message carefully and FIX the issue with a corrected action.
- Do NOT ignore errors. Do NOT move on to the next step if a required step failed.

RULES:
- This is a WINDOWS computer. Use Windows commands and Windows paths (backslashes).
- NEVER use Unix paths like ~/Desktop. Use the actual Windows path from the ENVIRONMENT info.
- For run_command, use Windows shell commands: mkdir, copy, move, del, ren, type, dir, etc.
- LOOK at the screenshot to understand what you see before deciding.
- If the goal needs the internet and no browser is visible, use open_browser.
- If you can see a clickable element in the OCR list below, prefer "click" with its target_id.
- If you see a button, icon, or UI element in the screenshot that is NOT in the OCR list, use "mouse_click_xy" with the pixel coordinates where it appears on screen.
- For drawing tasks: first open_app "mspaint", wait for it to open, then use draw_plan with the subject. Do NOT try to draw manually with mouse_drag.
- For system operations (shutdown, volume, create folder, move file, etc.), use run_command.
- Use hotkey for keyboard shortcuts (e.g. Ctrl+S, Alt+F4, Ctrl+C).
- If you need to type text, first click/focus the input field, then use "type".
- If the screen hasn't changed after your last action, try something DIFFERENT.
- If no useful elements are visible, use scroll to reveal more content.
- When interacting with Save/Open dialogs, look at the screenshot to identify the file name field, folder navigation bar, and Save/Open button. Use the screenshot coordinates to click them precisely.

GOAL COMPLETION — output {{"action":"done"}} when:
- The goal is fully achieved and confirmed on screen.
- Do NOT say done prematurely.
- Do NOT keep acting after the goal is clearly achieved.

Allowed actions:
{{"action":"click","target_id":ID}}
{{"action":"download","target_id":ID}}
{{"action":"type","text":"TEXT"}}
{{"action":"press_key","key":"Enter"}}
{{"action":"scroll","direction":"down"|"up"}}
{{"action":"navigate","url":"URL"}}
{{"action":"go_back"}}
{{"action":"open_browser","query":"SEARCH_QUERY"}}
{{"action":"open_app","app_name":"APP_NAME"}}
{{"action":"hotkey","keys":["ctrl","s"]}}
{{"action":"mouse_click_xy","x":PIXEL_X,"y":PIXEL_Y}}
{{"action":"mouse_drag","x1":X1,"y1":Y1,"x2":X2,"y2":Y2}}
{{"action":"run_command","command":"SHELL_COMMAND"}}
{{"action":"draw_plan","subject":"DESCRIPTION OF WHAT TO DRAW"}}
{{"action":"done"}}

GOAL: {goal}

OCR-detected text on screen (with pixel coordinates):
{element_list if element_list else "(no text detected on screen)"}"""

    def query_model_with_vision(self, goal: str, elements: list,
                                 screenshot_b64: str, user_context: str = "",
                                 screen_resolution: tuple = (1920, 1080)) -> dict:
        """Use a vision-capable model to SEE the screen and decide the next action.
        
        This is used in desktop mode so the AI can actually see what application
        is open, where buttons are, what the UI looks like, etc.
        """
        system_prompt = self._build_desktop_vision_prompt(
            goal, elements, screen_resolution
        )

        user_content = []
        # Text context first
        text_msg = user_context if user_context else (
            "Look at the screenshot carefully. What is the next action to achieve the goal?"
        )
        user_content.append({"type": "text", "text": text_msg})
        # Then the screenshot
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"}
        })

        try:
            response = self.groq_client.chat.completions.create(
                model=self.VISION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=400,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ReasoningModel] Vision model error: {e}")
            print("[ReasoningModel] Falling back to text-only model...")
            return self.query_model(goal, elements, user_context)

        # Parse JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        # Try to extract JSON from response (vision models sometimes add commentary)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            import re
            match = re.search(r'\{[^{}]+\}', raw)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            print(f"[ReasoningModel] Failed to parse vision response: {raw}")
            # Fall back to text model
            return self.query_model(goal, elements, user_context)