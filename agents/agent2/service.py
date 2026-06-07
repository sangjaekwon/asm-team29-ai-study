import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from agents.schemas import ContextAnalyzerOutput, FoodDirections

from .prompt import SYSTEM_PROMPT


load_dotenv(encoding="utf-8-sig")

SOLAR_BASE_URL = (
    os.getenv("SOLAR_BASE_URL")
    or os.getenv("UPSTAGE_BASE_URL")
    or "https://api.upstage.ai/v1"
)
SOLAR_MODEL = os.getenv("SOLAR_MODEL") or os.getenv("UPSTAGE_MODEL") or "solar-mini"

client = OpenAI(
    api_key=os.getenv("SOLAR_API_KEY") or os.getenv("UPSTAGE_API_KEY"),
    base_url=SOLAR_BASE_URL,
)


def _fallback_output(reason: str = "") -> ContextAnalyzerOutput:
    return ContextAnalyzerOutput(
        food_directions=FoodDirections(
            mood="",
            situation=reason or "Agent2 fallback: default cooking context.",
            fatigue_level="medium",
            difficulty="normal",
            preferred_taste="",
            preferred_cooking_method="",
            cooking_time_limit_minutes=None,
        )
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

        try:
            response = client.chat.completions.create(
                model=SOLAR_MODEL,
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

            result = response.choices[0].message.content or "{}"
            data = json.loads(result)
        except json.JSONDecodeError:
            print("Invalid JSON response: ")
            print(repr(locals().get("result", "")))
            return _fallback_output("Agent2 returned invalid JSON.")
        except Exception as exc:
            print("Agent2 analyze failed: ")
            print(repr(exc))
            return _fallback_output("Agent2 call failed.")

        return ContextAnalyzerOutput(
            food_directions=FoodDirections(
                mood=data.get("mood") or "",
                situation=data.get("situation") or "",
                fatigue_level=data.get("fatigue_level", "medium"),
                difficulty=data.get("difficulty", "normal"),
                preferred_taste=data.get("preferred_taste") or "",
                preferred_cooking_method=data.get("preferred_cooking_method") or "",
                cooking_time_limit_minutes=data.get("cooking_time_limit_minutes"),
            )
        )
