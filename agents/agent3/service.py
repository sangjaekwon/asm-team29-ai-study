import os
import json
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from agents.schemas import IngredientInfo, FoodDirections, CuisineRouterOutput
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
        base_url=SOLAR_BASE_URL
        )


def _fallback_output(reason: str = "") -> CuisineRouterOutput:
    return CuisineRouterOutput(
        recipe_type="korean",
        recipe_type_reason=reason or "Agent3 fallback: defaulted to korean.",
    )


class Agent3Service:
    def classify(
        self,
        ingredient_info: IngredientInfo,
        food_directions: Optional[FoodDirections] = None,
    ) -> CuisineRouterOutput:
        if isinstance(ingredient_info, dict):
            ingredient_info = IngredientInfo.model_validate(ingredient_info)
        if isinstance(food_directions, dict):
            food_directions = FoodDirections.model_validate(food_directions)

        payload = {
            "ingredient_info": ingredient_info.model_dump(),
        }
        if food_directions is not None:
            payload["food_directions"] = food_directions.model_dump()

        try:
            response = client.chat.completions.create(
                    model=SOLAR_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                payload,
                                ensure_ascii=False,
                                indent=2
                            )
                        }
                    ]
                )

            result = response.choices[0].message.content or "{}"
            data = json.loads(result)

        except json.JSONDecodeError:
            print("Invalid JSON response: ")
            print(repr(locals().get("result", "")))
            return _fallback_output("Agent3 returned invalid JSON, defaulted to korean.")
        except Exception as exc:
            print("Agent3 classify failed: ")
            print(repr(exc))
            return _fallback_output("Agent3 call failed, defaulted to korean.")

        return CuisineRouterOutput(
                recipe_type=data.get("recipe_type") or "korean",
                recipe_type_reason=data.get("recipe_type_reason")
                )
