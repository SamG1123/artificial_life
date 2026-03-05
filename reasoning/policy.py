"""
PolicyModel — pluggable action-selection strategy.

The *policy* is asked one question: given the current state, which
action should the agent take next?

Two concrete implementations ship out of the box:

  ``LLMPolicy``     — wraps the existing Groq-based ReasoningModel;
                      the LLM reads the screen + history and decides.
  ``MLPolicy``      — stub for a future learned policy (e.g. a small
                      transformer trained on action traces).

Both implement the same ``select_action()`` interface so the
ReasoningEngine can swap them transparently.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# ── Abstract base ────────────────────────────────────────────────

class BasePolicy(ABC):
    """Interface every policy must implement."""

    @abstractmethod
    def select_action(
        self,
        goal: str,
        elements: list[dict],
        history_context: str,
        *,
        screenshot_b64: str | None = None,
        screen_resolution: tuple[int, int] = (1920, 1080),
        plan_context: str = "",
        emotional_context: str = "",
    ) -> dict:
        """Return a single action dict (same schema as executor expects).

        Parameters
        ----------
        goal : str
            The high-level goal being pursued.
        elements : list[dict]
            Visible UI elements (OCR or DOM), each with id/type/text/center.
        history_context : str
            Formatted string of recent actions + results + environment.
        screenshot_b64 : str | None
            Base64-encoded JPEG screenshot (desktop mode).
        screen_resolution : tuple
            (width, height) pixels.
        plan_context : str
            Summary of the current plan (which step we're on, etc.).
        emotional_context : str
            The agent's current emotional / behavioral state.

        Returns
        -------
        dict
            ``{"action": "...", ...}`` — the next action to execute.
        """
        ...


# ── LLM Policy (wraps existing reasoning prompts) ───────────────

class LLMPolicy(BasePolicy):
    """Action selection via LLM reasoning — the default policy.

    Preserves full backward-compatibility with the original
    ``ReasoningModel`` prompts while adding plan + emotion awareness.
    """

    TEXT_MODEL = "llama-3.3-70b-versatile"
    VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # OS context (same as old ReasoningModel)
        self._os_info = f"Windows ({os.name})" if os.name == "nt" else os.name
        self._username = os.environ.get("USERNAME",
                                        os.environ.get("USER", "user"))
        self._home_dir = os.path.expanduser("~").replace("/", os.sep)
        self._desktop_dir = os.path.join(self._home_dir, "Desktop")

    # ── Public interface ─────────────────────────────────────────

    def select_action(
        self,
        goal: str,
        elements: list[dict],
        history_context: str,
        *,
        screenshot_b64: str | None = None,
        screen_resolution: tuple[int, int] = (1920, 1080),
        plan_context: str = "",
        emotional_context: str = "",
    ) -> dict:
        if screenshot_b64:
            return self._query_vision(
                goal, elements, screenshot_b64,
                history_context, screen_resolution,
                plan_context, emotional_context,
            )
        return self._query_text(
            goal, elements, history_context,
            plan_context, emotional_context,
        )

    # ── Text-only query (browser mode) ───────────────────────────

    def _query_text(self, goal: str, elements: list[dict],
                    user_context: str, plan_context: str,
                    emotional_context: str) -> dict:
        system_prompt = self._build_text_prompt(goal, elements,
                                                plan_context, emotional_context)
        user_message = user_context or "What is the next action?"

        try:
            resp = self.groq_client.chat.completions.create(
                model=self.TEXT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=300,
                temperature=0.2,
            )
            return self._parse(resp.choices[0].message.content)
        except Exception as e:
            print(f"[LLMPolicy] Text query failed: {e}")
            return {"action": "done"}

    # ── Vision query (desktop mode) ──────────────────────────────

    def _query_vision(self, goal: str, elements: list[dict],
                      screenshot_b64: str, user_context: str,
                      screen_resolution: tuple[int, int],
                      plan_context: str,
                      emotional_context: str) -> dict:
        system_prompt = self._build_vision_prompt(
            goal, elements, screen_resolution,
            plan_context, emotional_context,
        )

        user_content: list[dict] = [
            {"type": "text",
             "text": user_context or "Look at the screenshot. What is the next action?"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"}},
        ]

        try:
            resp = self.groq_client.chat.completions.create(
                model=self.VISION_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=400,
                temperature=0.2,
            )
            return self._parse(resp.choices[0].message.content)
        except Exception as e:
            print(f"[LLMPolicy] Vision query failed: {e}, falling back to text")
            return self._query_text(goal, elements, user_context,
                                    plan_context, emotional_context)

    # ── Prompt builders ──────────────────────────────────────────

    def _build_text_prompt(self, goal: str, elements: list[dict],
                           plan_context: str,
                           emotional_context: str) -> str:
        element_list = "\n".join(
            f"[{e['id']}] ({e['type']}) \"{e['text'][:80]}\""
            + (f"  href={e['href']}" if e.get("href") else "")
            for e in elements
        )

        sections = [self._core_rules(), f"GOAL:\n{goal}"]
        if plan_context:
            sections.insert(1, f"CURRENT PLAN:\n{plan_context}")
        if emotional_context:
            sections.insert(1, f"AGENT STATE:\n{emotional_context}")
        sections.append(f"VISIBLE ELEMENTS:\n{element_list}")
        return "\n\n".join(sections)

    def _build_vision_prompt(self, goal: str, elements: list[dict],
                             screen_resolution: tuple[int, int],
                             plan_context: str,
                             emotional_context: str) -> str:
        w, h = screen_resolution
        element_list = "\n".join(
            f"[{e['id']}] ({e['type']}) \"{e['text'][:60]}\" "
            f"at pixel ({e['center'][0]},{e['center'][1]})"
            for e in elements
        )

        sections = [
            self._core_rules(vision=True, resolution=f"{w}x{h}"),
            f"GOAL: {goal}",
        ]
        if plan_context:
            sections.insert(1, f"CURRENT PLAN:\n{plan_context}")
        if emotional_context:
            sections.insert(1, f"AGENT STATE:\n{emotional_context}")
        sections.append(
            f"OCR-detected text on screen (with pixel coordinates):\n"
            f"{element_list or '(no text detected)'}"
        )
        return "\n\n".join(sections)

    def _core_rules(self, *, vision: bool = False,
                    resolution: str = "") -> str:
        """The main instruction block shared by both modes."""
        mode = (
            "You are an autonomous computer control agent that can SEE "
            "the screen directly through the attached screenshot."
            if vision else
            "You are an autonomous computer control agent — a general-"
            "purpose AI that can operate any application on the computer."
        )

        res_line = (f"\nThe screen resolution is {resolution}. "
                    "All coordinate-based actions use screen pixel coordinates."
                    if resolution else "")

        return f"""{mode}{res_line}

Your task is to decide the NEXT single action to achieve the goal.
Output ONLY valid JSON. No explanation.

IMPORTANT — ERROR HANDLING:
- After EVERY action, you will receive feedback (success/failure/output).
- If a command FAILED, READ the error message and FIX with a corrected command.
- Do NOT ignore errors. Do NOT move on if a required step failed.

RULES:
- This is a WINDOWS computer. Use Windows commands and paths (backslashes).
- NEVER use Unix paths like ~/Desktop. Use actual Windows paths.
- Home: {self._home_dir}. Desktop: {self._desktop_dir}. User: {self._username}.
- For run_command use Windows shell: mkdir, copy, move, del, ren, dir, etc.
- If the goal needs internet and no browser is open, first open_browser.
- For drawing: open_app "mspaint" then draw_plan with the subject.
- For system ops (shutdown, volume, folders, files), use run_command.
- Only click "button", "link", "input", or "heading" elements. NEVER "text".
- For file downloads (.pdf, .zip, .doc), prefer "download" over "click".
- If nothing useful is visible, use "scroll".
- If screen unchanged after last action, try something DIFFERENT.
- Use hotkey for keyboard shortcuts (Ctrl+S, Alt+F4, …).
- Use mouse_click_xy for UI elements not in the element list.

GOAL COMPLETION — output {{"action":"done"}} when:
- The goal is fully achieved and confirmed on screen.
- Do NOT keep acting after the goal is achieved.
- Do NOT say done prematurely before the goal is actually achieved.

If a PLAN is provided, follow the current step but adapt if the screen
shows something unexpected. Skip steps that are already done.

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
{{"action":"mouse_click_xy","x":X,"y":Y}}
{{"action":"mouse_drag","x1":X1,"y1":Y1,"x2":X2,"y2":Y2}}
{{"action":"run_command","command":"SHELL_COMMAND"}}
{{"action":"draw_plan","subject":"DESCRIPTION"}}
{{"action":"done"}}"""

    # ── Parse LLM output ─────────────────────────────────────────

    @staticmethod
    def _parse(raw: str) -> dict:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{[^{}]+\}", raw)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            print(f"[LLMPolicy] Failed to parse: {raw[:200]}")
            return {"action": "done"}


# ── ML Policy (future stub) ─────────────────────────────────────

class MLPolicy(BasePolicy):
    """Placeholder for a learned policy model.

    When implemented, this will load a small transformer (or similar)
    trained on collected action traces and predict the next action
    from an encoded state vector (screen embedding + history + goal).
    """

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path
        self._model = None  # lazy-loaded

    def select_action(
        self,
        goal: str,
        elements: list[dict],
        history_context: str,
        *,
        screenshot_b64: str | None = None,
        screen_resolution: tuple[int, int] = (1920, 1080),
        plan_context: str = "",
        emotional_context: str = "",
    ) -> dict:
        if self._model is None:
            raise NotImplementedError(
                "MLPolicy is not yet trained. Use LLMPolicy for now."
            )
        # Future: encode state → model.predict → decode action
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load a trained model from disk."""
        self._model_path = path
        # Future: self._model = torch.load(path) or similar
        raise NotImplementedError

    def is_ready(self) -> bool:
        return self._model is not None
