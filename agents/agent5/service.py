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


load_dotenv()

SOLAR_BASE_URL = "https://api.upstage.ai/v1"
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
  "recommendation_reasons": [
    "Agent4가 selected_recipe.reason으로 전달한 최종 선택 이유"
  ],
  "cooking_tips": ["실용적인 팁 2~4개"],
  "substitutions": [],
  "additional_ingredients": []
}

규칙:
- ingredients에는 입력으로 받은 ingredients_to_use와 seasonings_to_use를 우선 포함합니다.
- Agent4가 전달한 additional_ingredients는 메뉴 성립에 필요한 필수 부족 재료이므로 유지하세요.
- 최종 조리 관점에서 있으면 더 좋은 선택 재료를 additional_ingredients에 1~3개 제안하세요.
- 대체하거나 생략 가능한 부재료를 substitutions에 1~3개 제안하세요.
- selected_recipe의 핵심 재료를 새로 추가하거나 대체해서 메뉴 성립 여부를 바꾸지 마세요.
- cooking_steps는 실제 조리 행동으로 작성하세요.
- difficulty는 반드시 easy, normal, hard 중 하나입니다.
- recipe_name은 selected_recipe.name과 같은 음식명을 유지하세요.
- 추천 이유 판단은 Agent4의 역할입니다. 당신은 추천 이유를 새로 판단하거나 새 근거를 만들지 마세요.
- recommendation_reasons에는 selected_recipe.reason을 그대로 포함하세요. 필요하면 문장 단위로만 나눌 수 있지만 의미를 바꾸거나 추가하지 마세요.
- selected_recipe.reason이 비어 있으면 추천 이유를 지어내지 말고 빈 배열을 반환하세요.
"""


def _unique_items(items: list[str]) -> list[str]:
    unique: list[str] = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _get_solar_api_key() -> str:
    return os.getenv("SOLAR_API_KEY", "").strip()


def _get_solar_model() -> str:
    return os.getenv("SOLAR_MODEL", DEFAULT_SOLAR_MODEL).strip() or DEFAULT_SOLAR_MODEL


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


def _fallback_recommendation_reasons(
    state: AgentState,
    selected_recipe: Any,
    ingredients: list[str],
    difficulty: str,
    cooking_time: int,
) -> list[str]:
    selected_reason = _selected_recipe_reason(selected_recipe)
    if not selected_reason:
        return []

    sentences = [
        sentence.strip().rstrip(".")
        for sentence in selected_reason.split(". ")
        if sentence.strip()
    ]
    reasons = [f"{sentence}." for sentence in sentences] if len(sentences) > 1 else [selected_reason]
    return _dedupe_recommendation_reasons(reasons)


def _normalize_reason_for_compare(reason: str) -> str:
    return " ".join(reason.strip().rstrip(".").split())


def _dedupe_recommendation_reasons(reasons: list[str]) -> list[str]:
    deduped: list[str] = []
    compare_values: list[str] = []

    for reason in reasons:
        cleaned = reason.strip()
        compare_value = _normalize_reason_for_compare(cleaned)
        if not compare_value:
            continue

        is_duplicate = False
        for existing in compare_values:
            if compare_value == existing:
                is_duplicate = True
                break
            if len(compare_value) >= 20 and compare_value in existing:
                is_duplicate = True
                break
            if len(existing) >= 20 and existing in compare_value:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        deduped.append(cleaned)
        compare_values.append(compare_value)

    return deduped


def _coerce_recommendation_reasons(value: Any, fallback: list[str]) -> list[str]:
    if not isinstance(value, list):
        return fallback

    reasons: list[str] = []
    for item in value:
        if isinstance(item, str):
            reason = item.strip()
        elif isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            description = str(item.get("description") or item.get("reason") or "").strip()
            reason = f"{title}: {description}" if title and description else title or description
        else:
            reason = str(item).strip()

        if reason:
            reasons.append(reason)

    return _dedupe_recommendation_reasons(reasons + fallback)[:5] or fallback


def _coerce_substitutions(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []

    substitutions: list[Any] = []
    for item in value:
        payload = item.model_dump() if hasattr(item, "model_dump") else item
        if isinstance(payload, dict) and str(payload.get("original") or "").strip():
            substitutions.append(payload)
    return substitutions


def _coerce_additional_ingredients(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    ingredients: list[str] = []
    for item in value:
        ingredient = str(item or "").strip()
        if ingredient:
            ingredients.append(ingredient)
    return _unique_items(ingredients)


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _selected_core_ingredients(state: AgentState, selected_recipe: Any) -> list[str]:
    selected_name = _selected_recipe_name(selected_recipe)
    for candidate in state.get("candidate_foods", []):
        payload = _plain_mapping(candidate)
        if payload.get("name") != selected_name:
            continue

        core_ingredients = payload.get("core_ingredients") or payload.get(
            "required_ingredients", []
        )
        return _unique_items([str(item) for item in core_ingredients])

    return []


def _merge_substitutions(
    agent4_substitutions: Any,
    agent5_substitutions: Any,
    protected_core_ingredients: list[str],
) -> list[Any]:
    protected = set(protected_core_ingredients)
    merged: list[Any] = []
    seen: set[tuple[str, str | None]] = set()

    for item in _coerce_substitutions(agent4_substitutions):
        original = str(item.get("original") or "").strip()
        replacement = item.get("replacement")
        key = (original, str(replacement).strip() if replacement is not None else None)
        if key not in seen:
            seen.add(key)
            merged.append(item)

    for item in _coerce_substitutions(agent5_substitutions):
        original = str(item.get("original") or "").strip()
        if original in protected:
            continue

        replacement = item.get("replacement")
        key = (original, str(replacement).strip() if replacement is not None else None)
        if key not in seen:
            seen.add(key)
            merged.append(item)

    return merged


def _merge_additional_ingredients(
    agent4_additional: Any,
    agent5_additional: Any,
    protected_core_ingredients: list[str],
) -> list[str]:
    required_additional = _coerce_additional_ingredients(agent4_additional)
    protected = set(protected_core_ingredients)
    merged = required_additional[:]

    for ingredient in _coerce_additional_ingredients(agent5_additional):
        if ingredient in protected and ingredient not in required_additional:
            continue
        if ingredient not in merged:
            merged.append(ingredient)

    return merged


def _default_optional_suggestions(
    selected_recipe: Any,
    ingredients: list[str],
    protected_core_ingredients: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    recipe_name = _selected_recipe_name(selected_recipe)
    available = set(ingredients)
    protected = set(protected_core_ingredients)
    substitutions: list[dict[str, Any]] = []
    additional: list[str] = []

    def add_substitution(original: str, replacement: str | None, reason: str) -> None:
        if original in protected or original not in available:
            return
        substitutions.append(
            {
                "original": original,
                "replacement": replacement,
                "reason": reason,
            }
        )

    def add_additional(ingredient: str) -> None:
        if ingredient not in available and ingredient not in protected:
            additional.append(ingredient)

    if "피자" in recipe_name:
        add_substitution("바질", None, "향을 더하는 선택 재료라 없어도 조리할 수 있습니다.")
        add_substitution("올리브", None, "토핑용 재료라 취향에 따라 생략할 수 있습니다.")
        add_substitution("피망", "파프리카", "식감과 단맛이 비슷해 서로 대체할 수 있습니다.")
        add_additional("올리브오일")
        add_additional("후추")
    elif any(term in recipe_name for term in ["볶음", "불고기", "덮밥"]):
        add_substitution("대파", None, "향을 보조하는 재료라 없어도 조리할 수 있습니다.")
        add_substitution("양파", "대파", "단맛과 향을 보완하는 용도로 대체할 수 있습니다.")
        add_substitution("고추", None, "매운맛 조절용이라 취향에 따라 생략할 수 있습니다.")
        add_additional("후추")
        add_additional("참기름")
    else:
        add_substitution("대파", None, "향을 보조하는 선택 재료라 없어도 조리할 수 있습니다.")
        add_substitution("마늘", None, "풍미를 더하는 재료라 없으면 생략할 수 있습니다.")
        add_additional("후추")

    return substitutions[:3], additional[:3]


def _ensure_optional_suggestions(
    selected_recipe: Any,
    ingredients: list[str],
    substitutions: list[Any],
    additional_ingredients: list[str],
    protected_core_ingredients: list[str],
) -> tuple[list[Any], list[str]]:
    fallback_substitutions, fallback_additional = _default_optional_suggestions(
        selected_recipe=selected_recipe,
        ingredients=ingredients,
        protected_core_ingredients=protected_core_ingredients,
    )

    if not substitutions:
        substitutions = fallback_substitutions
    if not additional_ingredients:
        additional_ingredients = fallback_additional
    if not substitutions and additional_ingredients:
        substitutions = _substitutions_for_optional_additional(additional_ingredients)

    return substitutions, additional_ingredients


def _substitutions_for_optional_additional(
    additional_ingredients: list[str],
) -> list[dict[str, Any]]:
    replacement_by_ingredient = {
        "올리브오일": "식용유",
        "마늘 가루": "다진마늘",
        "파마산치즈": "일반 치즈",
        "후추": None,
        "참기름": None,
    }
    substitutions: list[dict[str, Any]] = []

    for ingredient in additional_ingredients:
        if ingredient not in replacement_by_ingredient:
            continue

        replacement = replacement_by_ingredient[ingredient]
        substitutions.append(
            {
                "original": ingredient,
                "replacement": replacement,
                "reason": "맛을 보완하는 선택 재료라 없으면 대체하거나 생략할 수 있습니다.",
            }
        )

    return substitutions[:3]


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
        "core_ingredients": _selected_core_ingredients(state, selected_recipe),
        "recipe_type": state.get("recipe_type"),
        "recipe_type_reason": state.get("recipe_type_reason", ""),
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
        substitutions=_coerce_substitutions(state.get("substitutions", [])),
        additional_ingredients=_coerce_additional_ingredients(
            state.get("additional_ingredients", [])
        ),
        recommendation_reasons=_fallback_recommendation_reasons(
            state=state,
            selected_recipe=selected_recipe,
            ingredients=ingredients,
            difficulty=difficulty,
            cooking_time=cooking_time,
        ),
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
    fallback_reasons = _fallback_recommendation_reasons(
        state=state,
        selected_recipe=selected_recipe,
        ingredients=ingredients,
        difficulty=difficulty,
        cooking_time=cooking_time,
    )
    parsed["recommendation_reasons"] = fallback_reasons
    parsed.setdefault("cooking_tips", [])
    protected_core_ingredients = _selected_core_ingredients(state, selected_recipe)
    parsed["substitutions"] = _merge_substitutions(
        state.get("substitutions", []),
        parsed.get("substitutions", []),
        protected_core_ingredients,
    )
    parsed["additional_ingredients"] = _merge_additional_ingredients(
        state.get("additional_ingredients", []),
        parsed.get("additional_ingredients", []),
        protected_core_ingredients,
    )
    parsed["substitutions"], parsed["additional_ingredients"] = _ensure_optional_suggestions(
        selected_recipe=selected_recipe,
        ingredients=ingredients,
        substitutions=parsed["substitutions"],
        additional_ingredients=parsed["additional_ingredients"],
        protected_core_ingredients=protected_core_ingredients,
    )

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
