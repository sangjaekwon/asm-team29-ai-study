import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from agents.agent4.prompt import get_prompt


load_dotenv()

API_KEY = os.getenv("UPSTAGE_API_KEY")
BASE_URL = os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1")
MODEL = os.getenv("UPSTAGE_MODEL", "solar-pro3")


def create_client():
    if not API_KEY:
        raise RuntimeError("UPSTAGE_API_KEY environment variable is required.")

    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )


def parse_output(output_text):
    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        return output_text


def request_agent4(prompt=None):
    client = create_client()
    prompt = prompt or get_prompt()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return parse_output(response.choices[0].message.content)


if __name__ == "__main__":
    result = request_agent4()
    print(result)
