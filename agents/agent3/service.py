import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from .schema import Agent3Request, Agent3Response
from .prompt import SYSTEM_PROMPT

load_dotenv()

client = OpenAI(
        api_key=os.getenv("SOLAR_API_KEY"),
        base_url="https://api.upstage.ai/v1"
        )

class Agent3Service:
    def classify(self, request: Agent3Request) -> Agent3Response:
        response = client.chat.completions.create(
                model="solar-mini",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": request.model_dump_json(
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
        
        return Agent3Response(
                recipe_type=data.get("recipe_type")
                )

