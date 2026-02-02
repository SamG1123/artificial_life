from groq import Groq
import os
from dotenv import load_dotenv
import json

load_dotenv()

class Planner:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.SYSTEM_PROMPT = """
        You are a task planning AI.

        Convert user instructions into JSON steps. For apps normalize names (e.g., "Google Chrome" -> "chrome").

        Allowed actions:
        - open_app
        - navigate
        - search
        - click
        - type
        - download
        - move_file
        - create_folder
        - delete_file
        - shutdown
        Return ONLY valid JSON.

        Example format:
        {
        "steps": [
        {"action": "open_app", "target": "chrome"}
        ]
        }
            """


    def create_plan(self, goal: str):
        completion = self.client.chat.completions.create(
            model = "llama-3.1-8b-instant",
            messages=[
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": goal}
            ],
            temperature=0.2
        )
        output = completion.choices[0].message.content

        return json.loads(output)
    
    def validate_plan(self, plan: dict):
        valid_actions = {
            "open_app",
            "navigate",
            "search",
            "click",
            "type",
            "download",
            "move_file",
            "create_folder",
            "delete_file",
            "shutdown"
        }

        for step in plan.get("steps", []):
            if step["action"] not in valid_actions:
                return False
        return True


if __name__ == "__main__":
    planner = Planner()
    goal = "Open Chrome, search for cute cat pictures, and download three images."
    plan = planner.create_plan(goal)
    print(json.dumps(plan, indent=4))

    if planner.validate_plan(plan):
        print("Plan is valid.")
    else:
        print("Plan is invalid.")