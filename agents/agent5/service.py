"""5번 에이전트의 규칙 기반 레시피 생성 서비스."""

from typing import Any

from agents.agent5.messages import (
    COOKING_STEPS,
    COOKING_TIPS,
    FAILURE_MESSAGE,
    SUCCESS_MESSAGE,
)
from agents.schemas import AgentState, GeneratedRecipe


def _unique_items(items: list[str]) -> list[str]:
    unique: list[str] = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def generate_recipe(state: AgentState) -> dict[str, Any]:
    """공유 State의 라우터 결과로 최종 레시피를 생성한다."""

    selected_recipe = state.get("selected_recipe")
    ingredients = _unique_items(
        state.get("ingredients_to_use", []) + state.get("seasonings_to_use", [])
    )

    if selected_recipe is None or not ingredients:
        return {
            "generated_recipe": None,
            "generation_status": "failed",
            "generation_message": FAILURE_MESSAGE,
        }

    food_directions = state.get("food_directions")
    difficulty = food_directions.difficulty if food_directions is not None else "easy"
    cooking_time_limit = (
        food_directions.cooking_time_limit_minutes
        if food_directions is not None
        else None
    )
    cooking_time = min(cooking_time_limit, 15) if cooking_time_limit else 15

    recipe = GeneratedRecipe(
        recipe_name=selected_recipe.name,
        ingredients=ingredients,
        cooking_steps=COOKING_STEPS,
        cooking_time_minutes=cooking_time,
        difficulty=difficulty,
        servings=state.get("servings", 1),
        cooking_tips=COOKING_TIPS,
        substitutions=state.get("substitutions", []),
        additional_ingredients=state.get("additional_ingredients", []),
    )

    return {
        "generated_recipe": recipe,
        "generation_status": "success",
        "generation_message": SUCCESS_MESSAGE,
    }
