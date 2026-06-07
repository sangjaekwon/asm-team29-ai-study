"""5번 에이전트의 Solar 기반 레시피 생성 서비스."""

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from agents.agent5.messages import (
    COOKING_STEPS,
    COOKING_TIPS,
    FAILURE_MESSAGE,
    SUCCESS_MESSAGE,
)
from agents.schemas import AgentState, GeneratedRecipe


load_dotenv(encoding="utf-8-sig")

SOLAR_BASE_URL = (
    os.getenv("SOLAR_BASE_URL")
    or os.getenv("UPSTAGE_BASE_URL")
    or "https://api.upstage.ai/v1"
)
DEFAULT_SOLAR_MODEL = "solar-pro3"
TEST_API_KEYS = {"", "test", "dummy", "your_upstage_api_key_here"}

SYSTEM_PROMPT = """당신은 AI 요리 도우미의 최종 레시피 생성 에이전트입니다.
이전 에이전트가 고른 음식과 사용할 재료만 바탕으로 실제 사용자가 따라 할 수 있는 한국어 레시피를 작성하세요.

반드시 JSON만 반환하세요. 마크다운, 코드블록, 설명 문장은 쓰지 마세요.
반환 형식:
{
  "recipe_name": "음식 이름",
  "ingredients": ["실제로 사용할 재료와 양념"],
  "cooking_steps": ["구체적인 조리 순서 4~7개"],
  "cooking_time_minutes": 15,
  "difficulty": "easy | normal | hard",
  "servings": 1,
  "cooking_tips": ["실용적인 팁 2~4개"],
  "substitutions": [
    {
      "original": "원래 재료",
      "replacement": "대체 재료 또는 null",
      "reason": "대체/생략 이유"
    }
  ],
  "additional_ingredients": ["추가로 있으면 좋은 재료"]
}

규칙:
- ingredients에는 입력으로 받은 ingredients_to_use와 seasonings_to_use를 우선 포함합니다.
- 사용자가 현재 가진 재료를 벗어난 핵심 재료를 마음대로 추가하지 마세요.
- 부족하거나 선택적인 재료는 additional_ingredients 또는 substitutions에만 적으세요.
- cooking_steps는 실제 조리 행동으로 작성하세요.
- difficulty는 반드시 easy, normal, hard 중 하나입니다.
- recipe_name은 selected_recipe.name과 같은 음식명을 유지하세요.
"""


def _unique_items(items: list[str]) -> list[str]:
    unique: list[str] = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _get_solar_api_key() -> str:
    return (os.getenv("SOLAR_API_KEY") or os.getenv("UPSTAGE_API_KEY") or "").strip()


def _get_solar_model() -> str:
    return (
        os.getenv("SOLAR_MODEL")
        or os.getenv("UPSTAGE_MODEL")
        or DEFAULT_SOLAR_MODEL
    ).strip()


def _should_call_solar() -> bool:
    return _get_solar_api_key().lower() not in TEST_API_KEYS


def _extract_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    return json.loads(cleaned)


def _selected_recipe_name(selected_recipe: Any) -> str:
    if isinstance(selected_recipe, dict):
        return selected_recipe.get("name", "")
    return selected_recipe.name


def _selected_recipe_reason(selected_recipe: Any) -> str:
    if isinstance(selected_recipe, dict):
        return selected_recipe.get("reason", "")
    return selected_recipe.reason


def _food_directions_payload(food_directions: Any) -> dict[str, Any] | None:
    if isinstance(food_directions, dict):
        return food_directions
    if hasattr(food_directions, "model_dump"):
        return food_directions.model_dump()
    return None


def _recipe_context(
    state: AgentState,
    selected_recipe: Any,
    ingredients: list[str],
    difficulty: str,
    cooking_time: int,
) -> dict[str, Any]:
    return {
        "selected_recipe": {
            "name": _selected_recipe_name(selected_recipe),
            "reason": _selected_recipe_reason(selected_recipe),
        },
        "ingredients_to_use": state.get("ingredients_to_use", []),
        "seasonings_to_use": state.get("seasonings_to_use", []),
        "all_ingredients": ingredients,
        "substitutions": state.get("substitutions", []),
        "additional_ingredients": state.get("additional_ingredients", []),
        "food_directions": _food_directions_payload(state.get("food_directions")),
        "route": state.get("route"),
        "route_message": state.get("route_message"),
        "servings": state.get("servings", 1),
        "fallback_difficulty": difficulty,
        "fallback_cooking_time_minutes": cooking_time,
    }


def _build_fallback_recipe(
    state: AgentState,
    selected_recipe: Any,
    ingredients: list[str],
    difficulty: str,
    cooking_time: int,
) -> GeneratedRecipe:
    return GeneratedRecipe(
        recipe_name=_selected_recipe_name(selected_recipe),
        ingredients=ingredients,
        cooking_steps=COOKING_STEPS,
        cooking_time_minutes=cooking_time,
        difficulty=difficulty,
        servings=state.get("servings", 1),
        cooking_tips=COOKING_TIPS,
        substitutions=state.get("substitutions", []),
        additional_ingredients=state.get("additional_ingredients", []),
    )


def _call_solar_recipe_generator(
    state: AgentState,
    selected_recipe: Any,
    ingredients: list[str],
    difficulty: str,
    cooking_time: int,
) -> GeneratedRecipe:
    client = OpenAI(
        api_key=_get_solar_api_key(),
        base_url=SOLAR_BASE_URL,
    )
    payload = _recipe_context(
        state=state,
        selected_recipe=selected_recipe,
        ingredients=ingredients,
        difficulty=difficulty,
        cooking_time=cooking_time,
    )

    response = client.chat.completions.create(
        model=_get_solar_model(),
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
        temperature=0.4,
        stream=False,
    )
    content = response.choices[0].message.content or "{}"
    parsed = _extract_json(content)

    parsed.setdefault("recipe_name", _selected_recipe_name(selected_recipe))
    parsed.setdefault("ingredients", ingredients)
    parsed.setdefault("cooking_time_minutes", cooking_time)
    parsed.setdefault("difficulty", difficulty)
    parsed.setdefault("servings", state.get("servings", 1))
    parsed.setdefault("cooking_tips", [])
    parsed.setdefault("substitutions", state.get("substitutions", []))
    parsed.setdefault("additional_ingredients", state.get("additional_ingredients", []))

    if parsed.get("difficulty") not in {"easy", "normal", "hard"}:
        parsed["difficulty"] = difficulty

    return GeneratedRecipe.model_validate(parsed)


def _resolve_cooking_options(state: AgentState) -> tuple[str, int]:
    food_directions = state.get("food_directions")
    if isinstance(food_directions, dict):
        difficulty = food_directions.get("difficulty", "easy")
        cooking_time_limit = food_directions.get("cooking_time_limit_minutes")
    elif food_directions is not None:
        difficulty = food_directions.difficulty
        cooking_time_limit = food_directions.cooking_time_limit_minutes
    else:
        difficulty = "easy"
        cooking_time_limit = None

    if difficulty not in {"easy", "normal", "hard"}:
        difficulty = "easy"

    cooking_time = min(cooking_time_limit, 15) if cooking_time_limit else 15
    return difficulty, cooking_time


def generate_recipe(state: AgentState) -> dict[str, Any]:
    """공유 State의 라우터 결과로 최종 레시피를 생성한다."""

    selected_recipe = state.get("selected_recipe")
    ingredients = _unique_items(
        state.get("ingredients_to_use", []) + state.get("seasonings_to_use", [])
    )

    # print(f"[agent5] selected_recipe type={type(selected_recipe)}, value={selected_recipe}")
    # print(f"[agent5] ingredients={ingredients}")

    if selected_recipe is None or not ingredients:
        return {
            "generated_recipe": None,
            "generation_status": "failed",
            "generation_message": FAILURE_MESSAGE,
        }

    difficulty, cooking_time = _resolve_cooking_options(state)
    generation_source = "solar" if _should_call_solar() else "rules"
    error = ""

    if _should_call_solar():
        try:
            recipe = _call_solar_recipe_generator(
                state=state,
                selected_recipe=selected_recipe,
                ingredients=ingredients,
                difficulty=difficulty,
                cooking_time=cooking_time,
            )
        except Exception as exc:
            recipe = _build_fallback_recipe(
                state=state,
                selected_recipe=selected_recipe,
                ingredients=ingredients,
                difficulty=difficulty,
                cooking_time=cooking_time,
            )
            generation_source = "rules_fallback"
            error = str(exc)
    else:
        recipe = _build_fallback_recipe(
            state=state,
            selected_recipe=selected_recipe,
            ingredients=ingredients,
            difficulty=difficulty,
            cooking_time=cooking_time,
        )

    return {
        "generated_recipe": recipe,
        "generation_status": "success",
        "generation_message": (
            "Solar API로 레시피를 생성했습니다."
            if generation_source == "solar"
            else SUCCESS_MESSAGE
        ),
        "recipe_generation_source": generation_source,
        "recipe_generation_error": error,
    }
