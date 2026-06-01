from __future__ import annotations

import json
import os
import re
from typing import Any

from logging_config import get_logger

log = get_logger("local_fallback")


class LocalFallbackModel:
    """Best-effort local fallback for when hosted LLM calls are unavailable.

    If ``LOCAL_FALLBACK_MODEL`` points to a local Hugging Face causal LM
    directory, this class will use it for JSON generation. If no local model
    is configured or loading fails, it falls back to small deterministic rules.
    """

    _APP_ALIASES = {
        "steam": "steam",
        "steam app": "steam",
        "steam application": "steam",
        "chrome": "chrome",
        "google chrome": "chrome",
        "browser": "chrome",
        "file explorer": "explorer",
        "explorer": "explorer",
        "settings": "settings",
        "notepad": "notepad",
        "discord": "discord",
        "spotify": "spotify",
    }

    def __init__(self, model_path: str | None = None):
        self.model_path = (model_path or os.getenv("LOCAL_FALLBACK_MODEL", "")).strip()
        self._generator = None

    def _load_generator(self):
        if self._generator is not None:
            return self._generator

        if not self.model_path or not os.path.exists(self.model_path):
            return None

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            self._generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            log.info("Loaded local fallback model from %s", self.model_path)
            return self._generator
        except Exception as exc:
            log.warning("Could not load local fallback model '%s': %s", self.model_path, exc)
            self._generator = None
            return None

    def analyze_user_input(
        self,
        user_input: str,
        *,
        context: str = "",
        persona: str = "",
    ) -> dict[str, Any]:
        prompt = self._build_input_prompt(user_input, context, persona)
        generated = self._generate_json(prompt, max_new_tokens=220)
        if generated:
            parsed = self._parse_json(generated)
            if parsed:
                return self._normalize_input_result(parsed, user_input)

        return self._heuristic_input_result(user_input)

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
    ) -> dict[str, Any]:
        prompt = self._build_action_prompt(
            goal,
            elements,
            history_context,
            screen_resolution,
            plan_context,
            emotional_context,
            screenshot_b64=screenshot_b64,
        )
        generated = self._generate_json(prompt, max_new_tokens=180)
        if generated:
            parsed = self._parse_json(generated)
            if parsed:
                return self._normalize_action_result(parsed, goal, elements)

        return self._heuristic_action_result(goal, elements, history_context)

    def _generate_json(self, prompt: str, *, max_new_tokens: int) -> str | None:
        generator = self._load_generator()
        if generator is None:
            return None

        try:
            output = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_full_text=False,
                pad_token_id=getattr(generator.tokenizer, "eos_token_id", None),
            )
            if not output:
                return None
            text = output[0].get("generated_text", "")
            return text.strip() or None
        except Exception as exc:
            log.warning("Local fallback generation failed: %s", exc)
            return None

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, flags=re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    return None
        return None

    def _build_input_prompt(self, user_input: str, context: str, persona: str) -> str:
        return (
            f"{persona}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER INPUT:\n{user_input}\n\n"
            "Return ONLY JSON with fields type, response, goal, schedule_at, recurrence_seconds, user_name, learn, knowledge."
        )

    def _build_action_prompt(
        self,
        goal: str,
        elements: list[dict],
        history_context: str,
        screen_resolution: tuple[int, int],
        plan_context: str,
        emotional_context: str,
        screenshot_b64: str | None = None,
    ) -> str:
        compact = elements[:20]
        element_lines = []
        for element in compact:
            text = (element.get("text") or "")[:60]
            element_lines.append(f"[{element.get('id')}] {element.get('type')} {text}")
        w, h = screen_resolution
        return (
            f"GOAL: {goal}\n"
            f"SCREEN: {w}x{h}\n"
            f"PLAN: {plan_context}\n"
            f"STATE: {emotional_context}\n"
            f"HISTORY: {history_context}\n"
            f"ELEMENTS:\n" + "\n".join(element_lines) + "\n\n"
            "Return ONLY JSON for the next action."
        )

    def _heuristic_input_result(self, user_input: str) -> dict[str, Any]:
        text = user_input.strip()
        lowered = text.lower()

        if any(word in lowered for word in ("hi", "hello", "hey", "good morning", "good evening")):
            return {
                "type": "chat",
                "response": "Hi. What would you like me to do?",
                "goal": "",
                "schedule_at": "",
                "recurrence_seconds": None,
                "user_name": None,
                "learn": [],
                "knowledge": [],
            }

        app_name = self._extract_app_name(lowered)
        if app_name:
            return {
                "type": "action",
                "response": f"Opening {app_name}.",
                "goal": f"open {app_name} application",
                "schedule_at": "",
                "recurrence_seconds": None,
                "user_name": None,
                "learn": [],
                "knowledge": [],
            }

        if any(word in lowered for word in ("search", "look up", "find", "browse", "youtube", "video", "website")):
            return {
                "type": "action",
                "response": "I’ll handle that.",
                "goal": text,
                "schedule_at": "",
                "recurrence_seconds": None,
                "user_name": None,
                "learn": [],
                "knowledge": [],
            }

        return {
            "type": "chat",
            "response": "I’m here. Tell me what to do.",
            "goal": "",
            "schedule_at": "",
            "recurrence_seconds": None,
            "user_name": None,
            "learn": [],
            "knowledge": [],
        }

    def _heuristic_action_result(self, goal: str, elements: list[dict], history_context: str) -> dict[str, Any]:
        lowered = goal.lower()

        app_name = self._extract_app_name(lowered)
        if app_name:
            return {"action": "open_app", "app_name": app_name}

        if any(word in lowered for word in ("search", "look up", "find", "browse", "youtube", "video", "website")):
            return {"action": "open_browser", "query": goal}

        if "latest video" in lowered or "newest video" in lowered:
            return {"action": "open_browser", "query": goal}

        for element in elements:
            href = (element.get("href") or "").lower()
            text = (element.get("text") or "").lower()
            if "youtube.com/watch" in href or "/watch?v=" in href or "youtube.com/shorts" in href:
                return {"action": "click", "target_id": element.get("id")}
            if any(token in text for token in ("open", "view", "download", "play")) and element.get("id"):
                return {"action": "click", "target_id": element.get("id")}

        return {"action": "done"}

    def _extract_app_name(self, lowered_text: str) -> str | None:
        for alias, canonical in self._APP_ALIASES.items():
            if alias in lowered_text:
                return canonical
        return None

    @staticmethod
    def _normalize_input_result(result: dict[str, Any], user_input: str) -> dict[str, Any]:
        result.setdefault("type", "action")
        result.setdefault("response", "I’ll handle that.")
        result.setdefault("goal", user_input)
        result.setdefault("schedule_at", "")
        result.setdefault("recurrence_seconds", None)
        result.setdefault("user_name", None)
        result.setdefault("learn", [])
        result.setdefault("knowledge", [])
        return result

    @staticmethod
    def _normalize_action_result(result: dict[str, Any], goal: str, elements: list[dict]) -> dict[str, Any]:
        if "action" not in result:
            return {"action": "done"}

        if result["action"] == "open_app":
            app_name = result.get("app_name") or result.get("target") or goal
            if not isinstance(app_name, str):
                app_name = goal
            result = {"action": "open_app", "app_name": app_name}

        if result["action"] == "click":
            target_id = result.get("target_id") or result.get("target")
            try:
                result["target_id"] = int(target_id)
            except Exception:
                if elements:
                    result["target_id"] = elements[0].get("id", 0)
        return result