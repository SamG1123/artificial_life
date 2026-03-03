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
            f"[{e['id']}] ({e['type']}) \"{e['text']}\" at {e['center']}"
            for e in elements
        )
        return f"""You are an autonomous computer control agent.

Your task is to decide the NEXT action to achieve the goal.
Choose exactly ONE action.
Do not explain your reasoning.
Output ONLY valid JSON.

Rules:
- Only click elements of type "button" or "link". NEVER click "text" elements — they are not interactive.
- If no useful clickable element is visible, scroll down to reveal more.
- If the screen has not changed after your last action, try a DIFFERENT element or scroll.
- Only output {{"action":"done"}} when the goal has actually been achieved (e.g. a file has been downloaded, a page has loaded, etc).

Allowed actions:
{{"action":"click","target_id":ID}}
{{"action":"type","text":"TEXT"}}
{{"action":"scroll","direction":"down"}}
{{"action":"scroll","direction":"up"}}
{{"action":"open_app","app_name":"APP_NAME"}}
{{"action":"open_browser","query":"SEARCH_QUERY"}}
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