from groq import Groq
from dotenv import load_dotenv
import os

class ReasoningModel:
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def query_model(self, prompt: str):
        response = self.groq_client.chat.completions.create(
            model="gpt-4o-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful reasoning assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message['content']