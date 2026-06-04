import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from agents.schemas import ContextAnalyzerOutput, FoodDirections

from .prompt import SYSTEM_PROMPT


load_dotenv()

client = OpenAI(
    api_key=os.getenv("SOLAR_API_KEY"),
    base_url="https://api.upstage.ai/v1",
)


class Agent2Service:
    def analyze(
        self,
        user_mood_input: str,
        user_situation_input: str,
    ) -> ContextAnalyzerOutput:
        payload = {
            "user_mood_input": user_mood_input,
            "user_situation_input": user_situation_input,
        }

        response = client.chat.completions.create(
            model="solar-mini",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False, indent=2),
                },
            ],
        )

        result = response.choices[0].message.content

        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            print("Invalid JSON response: ")
            print(repr(result))
            raise

        return ContextAnalyzerOutput(
            food_directions=FoodDirections(
                mood=data.get("mood", ""),
                situation=data.get("situation", ""),
                fatigue_level=data.get("fatigue_level", "medium"),
                difficulty=data.get("difficulty", "normal"),
                preferred_taste=data.get("preferred_taste", ""),
                preferred_cooking_method=data.get("preferred_cooking_method", ""),
                cooking_time_limit_minutes=data.get("cooking_time_limit_minutes"),
            )
        )
