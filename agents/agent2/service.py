import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from .prompt import SYSTEM_PROMPT
from .schema import Agent2Request, Agent2Response


load_dotenv()

client = OpenAI(
    api_key=os.getenv("SOLAR_API_KEY"),
    base_url="https://api.upstage.ai/v1",
)


DEFAULT_RESPONSE = Agent2Response(
    mood="",
    situation="",
    fatigue_level="medium",
    difficulty="normal",
    preferred_taste="담백한 맛",
    preferred_cooking_method="팬 조리",
    cooking_time_limit_minutes=20,
)


class Agent2Service:
    def analyze(self, request: Agent2Request) -> Agent2Response:
        if not request.user_mood_input.strip() and not request.user_situation_input.strip():
            return DEFAULT_RESPONSE.model_copy()

        response = client.chat.completions.create(
            model="solar-mini",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": request.model_dump_json(indent=2),
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

        return Agent2Response(
            mood=data.get("mood", ""),
            situation=data.get("situation", ""),
            fatigue_level=data.get("fatigue_level", "medium"),
            difficulty=data.get("difficulty", "normal"),
            preferred_taste=data.get("preferred_taste", ""),
            preferred_cooking_method=data.get("preferred_cooking_method", ""),
            cooking_time_limit_minutes=data.get("cooking_time_limit_minutes"),
        )
