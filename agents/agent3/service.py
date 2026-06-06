import os
import json
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from agents.schemas import IngredientInfo, FoodDirections, CuisineRouterOutput
from .prompt import SYSTEM_PROMPT

load_dotenv()

client = OpenAI(
        api_key=os.getenv("SOLAR_API_KEY"),
        base_url="https://api.upstage.ai/v1"
        )

class Agent3Service:
    def classify(
        self,
        ingredient_info: IngredientInfo,
        food_directions: Optional[FoodDirections] = None,
    ) -> CuisineRouterOutput:
        payload = {
            "ingredient_info": ingredient_info.model_dump(),
        }
        if food_directions is not None:
            payload["food_directions"] = food_directions.model_dump()

        response = client.chat.completions.create(
                model="solar-mini",
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

        result = response.choices[0].message.content

        try:
            data = json.loads(result)

        except json.JSONDecodeError:
            print("Invalid JSON response: ")
            print(repr(result))
            raise

        return CuisineRouterOutput(
                recipe_type=data.get("recipe_type"),
                recipe_type_reason=data.get("recipe_type_reason")
                )
