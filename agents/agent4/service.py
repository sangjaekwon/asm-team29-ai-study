"""Rule-based service for agent4, the feasible recipe router."""

import json
import os
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from agents.agent4.prompt import get_prompt as get_agent4_prompt
from agents.schemas import (
    CandidateFood,
    FoodDirections,
    IngredientInfo,
    RecipeCandidateEvaluation,
    RecipeRouterInput,
    RecipeRouterOutput,
    RecipeType,
    SelectedRecipe,
    Substitution,
)

load_dotenv(encoding="utf-8-sig")


_DIFFICULTY_RANK = {"easy": 1, "normal": 2, "hard": 3}

SOLAR_BASE_URL = (
    os.getenv("SOLAR_BASE_URL")
    or os.getenv("UPSTAGE_BASE_URL")
    or "https://api.upstage.ai/v1"
)
SOLAR_MODEL = os.getenv("SOLAR_MODEL") or os.getenv("UPSTAGE_MODEL") or "solar-mini"
TEST_API_KEYS = {"", "test", "dummy", "your_upstage_api_key_here"}

_COOKING_METHOD_ALIASES: dict[str, list[str]] = {
    "빠른 조리": ["간편 조리", "팬 조리", "볶기", "quick", "simple"],
    "간단한 조리": ["간편 조리", "팬 조리", "볶기", "quick", "simple"],
    "간편 조리": ["빠른 조리", "간단한 조리", "팬 조리", "볶기", "quick", "simple"],
    "프라이팬": ["팬 조리", "볶기", "굽기", "pan"],
    "팬": ["팬 조리", "볶기", "굽기", "pan"],
    "pan": ["팬 조리", "볶기", "굽기", "프라이팬"],
}

_RECIPE_TYPE_ALIASES: dict[str, RecipeType] = {
    "korean": "korean",
    "한식": "korean",
    "chinese": "chinese",
    "중식": "chinese",
    "japanese": "japanese",
    "일식": "japanese",
    "western": "western",
    "양식": "western",
}

def _normalize_recipe_type(recipe_type: str | None) -> RecipeType:
    if not recipe_type:
        return "korean"

    return _RECIPE_TYPE_ALIASES.get(recipe_type.strip().lower(), "korean")


def _normalize_items(items: list[str]) -> list[str]:
    normalized: list[str] = []
    for item in items:
        value = item.strip()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _unique_items(items) -> list[str]:
    unique: list[str] = []
    for item in items or []:
        value = " ".join(str(item).strip().split())
        if value and value not in unique:
            unique.append(value)
    return unique


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _state_recipe_type(state: Mapping[str, Any]) -> RecipeType:
    return _normalize_recipe_type(state.get("recipe_type"))


def _state_food_directions(state: Mapping[str, Any]) -> dict[str, Any]:
    return _plain_mapping(state.get("food_directions"))


def _ingredient_groups(state: Mapping[str, Any]) -> dict[str, list[str]]:
    info = _plain_mapping(state.get("ingredient_info"))
    available = _unique_items(
        state.get("available_ingredients") or state.get("user_input_ingredients") or []
    )
    seasonings = _unique_items(info.get("seasonings", []))
    cookable = [item for item in available if item not in seasonings]
    main = _unique_items(info.get("main_ingredients", [])) or cookable[:1]
    sub = _unique_items(info.get("sub_ingredients", []))

    return {
        "available": available,
        "cookable": cookable,
        "main": [item for item in main if item in available],
        "sub": [item for item in sub if item in available],
        "seasonings": [item for item in seasonings if item in available],
        "carbohydrates": [
            item for item in _unique_items(info.get("carbohydrates", [])) if item in available
        ],
        "proteins": [
            item for item in _unique_items(info.get("proteins", [])) if item in available
        ],
        "vegetables": [
            item for item in _unique_items(info.get("vegetables", [])) if item in available
        ],
    }


def _simple_recipe_name(ingredients: list[str]) -> str:
    if not ingredients:
        return "Simple ingredient dish"
    return f"Simple {' '.join(ingredients[:3])}"


def _get_solar_api_key() -> str:
    return (os.getenv("SOLAR_API_KEY") or os.getenv("UPSTAGE_API_KEY") or "").strip()


def _should_call_solar() -> bool:
    return _get_solar_api_key().lower() not in TEST_API_KEYS


def _extract_json(content: str) -> dict[str, Any]:
    cleaned = (content or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()
    return json.loads(cleaned)


def _ai_candidate_payload(
    state: Mapping[str, Any],
    groups: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "available_ingredients": groups["available"],
        "cookable_ingredients": groups["cookable"],
        "main_ingredients": groups["main"],
        "sub_ingredients": groups["sub"],
        "seasonings": groups["seasonings"],
        "recipe_type": _state_recipe_type(state),
        "food_directions": _state_food_directions(state),
        "user_mood_input": state.get("user_mood_input", ""),
        "user_situation_input": state.get("user_situation_input", ""),
        "ingredient_policy": state.get("ingredient_policy", "only_available"),
    }


def _normalize_ai_candidate(
    candidate: dict[str, Any],
    state: Mapping[str, Any],
    groups: dict[str, list[str]],
) -> dict[str, Any] | None:
    available = set(groups["available"])
    recipe_type = _state_recipe_type(state)
    name = str(candidate.get("name") or "").strip()

    raw_required = _unique_items(candidate.get("required_ingredients", []))
    core = _unique_items(candidate.get("core_ingredients", [])) or raw_required
    required = [item for item in raw_required if item in available]
    required.extend(item for item in core if item in available and item not in required)
    optional = [
        item
        for item in _unique_items(candidate.get("optional_ingredients", []))
        if item in available and item not in required
    ]
    seasonings = [
        item
        for item in _unique_items(candidate.get("seasonings", []))
        if item in available
    ]

    if not required:
        required = (
            [item for item in core if item in available]
            or (groups["main"] or groups["cookable"])[:1]
        )
    if not required:
        return None
    if not core:
        core = required[:]

    normalized = {
        "name": name or _simple_recipe_name(required),
        "recipe_type": recipe_type,
        "core_ingredients": core,
        "required_ingredients": required,
        "optional_ingredients": optional,
        "seasonings": seasonings,
        "substitutions": {item: None for item in optional},
        "difficulty": candidate.get("difficulty") or "easy",
        "cooking_time_minutes": candidate.get("cooking_time_minutes") or 15,
        "taste_profile": _unique_items(candidate.get("taste_profile", ["savory"])),
        "cooking_methods": _unique_items(
            candidate.get("cooking_methods", ["simple cooking"])
        ),
        "reason": candidate.get("reason")
        or "현재 보유 재료와 사용자 상황을 바탕으로 바로 활용하기 좋은 후보입니다.",
    }

    try:
        return CandidateFood.model_validate(normalized).model_dump()
    except Exception:
        return None


def _recipe_response_to_candidate(response: dict[str, Any]) -> dict[str, Any] | None:
    recipe_name = response.get("recipe_name") or response.get("name")
    ingredients = response.get("ingredients", [])
    if not recipe_name or not isinstance(ingredients, list):
        return None

    required: list[str] = []
    optional: list[str] = []
    seasonings: list[str] = []

    for ingredient in ingredients:
        if isinstance(ingredient, str):
            optional.append(ingredient)
            continue
        if not isinstance(ingredient, dict):
            continue

        name = str(ingredient.get("name") or "").strip()
        role = str(ingredient.get("role") or "").strip().lower()
        if not name:
            continue
        if role in {"main", "required", "protein"}:
            required.append(name)
        elif role in {"seasoning", "sauce"}:
            seasonings.append(name)
        else:
            optional.append(name)

    return {
        "name": recipe_name,
        "core_ingredients": required,
        "required_ingredients": required,
        "optional_ingredients": optional,
        "seasonings": seasonings,
        "difficulty": "easy",
        "cooking_time_minutes": (
            response.get("cook_time_minutes")
            or response.get("total_time_minutes")
            or response.get("cooking_time_minutes")
            or 15
        ),
        "taste_profile": ["savory"],
        "cooking_methods": ["AI recommendation", "simple cooking"],
        "reason": response.get("description")
        or "현재 보유 재료와 사용자 상황을 바탕으로 생성한 레시피 후보입니다.",
    }


def _raw_ai_candidates(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_foods = parsed.get("candidate_foods")
    if isinstance(candidate_foods, list):
        return [
            candidate for candidate in candidate_foods if isinstance(candidate, dict)
        ]

    recipe_candidate = _recipe_response_to_candidate(parsed)
    return [recipe_candidate] if recipe_candidate else []


def _build_ai_agent4_candidates(
    state: Mapping[str, Any],
    groups: dict[str, list[str]],
) -> list[dict[str, Any]]:
    if not _should_call_solar():
        raise RuntimeError(
            "Solar API key is not configured for Agent4 candidate generation."
        )

    client = OpenAI(
        api_key=_get_solar_api_key(),
        base_url=SOLAR_BASE_URL,
    )
    payload = _ai_candidate_payload(state, groups)
    response = client.chat.completions.create(
        model=SOLAR_MODEL,
        messages=[
            {
                "role": "system",
                "content": get_agent4_prompt(),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            },
        ],
        temperature=0.5,
    )
    parsed = _extract_json(response.choices[0].message.content or "{}")
    raw_candidates = _raw_ai_candidates(parsed)
    selected_payload = parsed.get("selected_recipe") if isinstance(parsed, dict) else None
    selected_name = ""
    selected_reason = ""
    if isinstance(selected_payload, dict):
        selected_name = str(selected_payload.get("name") or "").strip()
        selected_reason = str(selected_payload.get("reason") or "").strip()

    candidates: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for raw_candidate in raw_candidates:
        raw_name = str(raw_candidate.get("name") or raw_candidate.get("recipe_name") or "").strip()
        if selected_reason and raw_name == selected_name:
            raw_candidate = {**raw_candidate, "reason": selected_reason}
        candidate = _normalize_ai_candidate(raw_candidate, state, groups)
        if not candidate or candidate["name"] in seen_names:
            continue
        seen_names.add(candidate["name"])
        candidates.append(candidate)

    return candidates


def _build_agent4_candidates(state: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Build Agent4 candidates with Solar AI and return an error when it fails."""

    groups = _ingredient_groups(state)
    if not groups["cookable"]:
        return [], ""

    try:
        ai_candidates = _build_ai_agent4_candidates(state, groups)
    except Exception as exc:
        return [], str(exc)

    if ai_candidates:
        return ai_candidates, ""

    return [], "Agent4 chat did not return any valid recipe candidates."


def _candidate_generation_failed(error: str) -> dict[str, Any]:
    return {
        "candidate_foods": [],
        "candidate_evaluations": [],
        "route": "no_ingredient",
        "route_message": "candidate_generation_failed",
        "selected_recipe": None,
        "ingredients_to_use": [],
        "seasonings_to_use": [],
        "substitutions": [],
        "additional_ingredients": [],
        "can_pass_to_agent5": False,
        "candidate_generation_error": error,
        "recipe_generation_error": error,
    }


def _is_detailed_reason(reason: str) -> bool:
    stripped = reason.strip()
    return len(stripped) >= 90 and stripped.count(".") + stripped.count("다") >= 3


def _build_selected_recipe_reason(
    candidate: CandidateFood,
    request: RecipeRouterInput,
    evaluation: RecipeCandidateEvaluation,
    ingredients_to_use: list[str],
    seasonings_to_use: list[str],
) -> str:
    if _is_detailed_reason(candidate.reason):
        return candidate.reason

    food_directions = request.food_directions
    reason_parts: list[str] = []

    mood_items = [
        item
        for item in [
            food_directions.mood,
            food_directions.situation,
        ]
        if item
    ]
    mood_context = "', '".join(mood_items)
    if mood_context:
        reason_parts.append(
            f"사용자가 '{mood_context}'라고 했기 때문에 오래 손질하거나 복잡하게 조리하는 음식보다 빠르게 준비할 수 있는 메뉴가 적합합니다"
        )

    materials = ingredients_to_use + seasonings_to_use
    if materials:
        reason_parts.append(
            f"현재 확인된 {', '.join(materials[:5])} 재료를 중심으로 만들 수 있어 새 재료를 많이 추가하지 않아도 됩니다"
        )

    if candidate.reason:
        reason_parts.append(candidate.reason.rstrip("."))

    if evaluation.conflict_reasons:
        reason_parts.append(
            "시간이나 난이도 조건은 완전히 딱 맞지 않더라도 손질을 줄이고 팬 조리 중심으로 진행하면 지금 상황에 맞게 조정할 수 있습니다"
        )
    elif evaluation.route == "simple":
        reason_parts.append(
            "주재료는 갖춰져 있고 부족한 부재료는 생략하거나 대체할 수 있어 조리 부담이 크지 않습니다"
        )
    elif evaluation.route == "can_cook":
        reason_parts.append("주재료와 양념이 갖춰져 있어 현재 재료만으로 바로 조리할 수 있습니다")

    if food_directions.cooking_time_limit_minutes:
        reason_parts.append(
            f"조리 시간이 약 {candidate.cooking_time_minutes}분이라 빨리 먹고 싶은 상황에 맞추기 쉽습니다"
        )
    else:
        reason_parts.append(
            f"난이도가 {candidate.difficulty}라 지금 상황에서 부담 없이 시도할 수 있습니다"
        )

    if request.recipe_type_reason:
        reason_parts.append(f"요리 스타일 선택 근거도 반영했습니다: {request.recipe_type_reason}")

    return ". ".join(_normalize_items(reason_parts[:4])) + "."


def _collect_available_ingredients(request: RecipeRouterInput) -> list[str]:
    info: IngredientInfo = request.ingredient_info
    return _normalize_items(
        [
            *request.available_ingredients,
            *info.main_ingredients,
            *info.sub_ingredients,
            *info.seasonings,
            *info.carbohydrates,
            *info.proteins,
            *info.fats,
            *info.vegetables,
        ]
    )


def _collect_available_seasonings(request: RecipeRouterInput) -> set[str]:
    return set(_normalize_items(request.ingredient_info.seasonings))


def _candidate_pool(request: RecipeRouterInput) -> list[CandidateFood]:
    recipe_type = _normalize_recipe_type(request.recipe_type)
    return [
        candidate
        for candidate in request.candidate_foods
        if candidate.recipe_type == recipe_type
    ]


def _method_matches(preferred_method: str, candidate_methods: list[str]) -> bool:
    preferred = preferred_method.strip()
    if not preferred or not candidate_methods:
        return True

    preferred_terms = [preferred, *_COOKING_METHOD_ALIASES.get(preferred, [])]
    for term in preferred_terms:
        for method in candidate_methods:
            if term in method or method in term:
                return True

    return False


def _find_conflicts(
    candidate: CandidateFood,
    food_directions: FoodDirections,
) -> list[str]:
    conflicts: list[str] = []
    time_limit = food_directions.cooking_time_limit_minutes

    if time_limit is not None and candidate.cooking_time_minutes > time_limit:
        conflicts.append(
            f"cooking_time_exceeds_limit:{candidate.cooking_time_minutes}>{time_limit}"
        )

    wanted_difficulty = food_directions.difficulty
    if _DIFFICULTY_RANK[candidate.difficulty] > _DIFFICULTY_RANK[wanted_difficulty]:
        conflicts.append(
            f"difficulty_exceeds_preference:{candidate.difficulty}>{wanted_difficulty}"
        )

    if food_directions.fatigue_level == "high" and candidate.difficulty != "easy":
        conflicts.append("high_fatigue_requires_easy_recipe")

    preferred_method = food_directions.preferred_cooking_method.strip()
    if preferred_method and candidate.cooking_methods:
        if not _method_matches(preferred_method, candidate.cooking_methods):
            conflicts.append(f"cooking_method_mismatch:{preferred_method}")

    return conflicts


def _evaluate_candidate(
    candidate: CandidateFood,
    available_ingredients: list[str],
    food_directions: FoodDirections,
    allow_additional: bool,
) -> RecipeCandidateEvaluation:
    available = set(available_ingredients)
    substitutions: list[Substitution] = []
    core_ingredients = candidate.core_ingredients or candidate.required_ingredients
    missing_required: list[str] = []

    for ingredient in core_ingredients:
        if ingredient in available:
            continue

        replacement = candidate.substitutions.get(ingredient)
        if replacement and replacement in available:
            substitutions.append(
                Substitution(
                    original=ingredient,
                    replacement=replacement,
                    reason="required_ingredient_replaced",
                )
            )
            continue

        if ingredient not in missing_required:
            missing_required.append(ingredient)

    missing_optional: list[str] = []
    for ingredient in candidate.optional_ingredients:
        if ingredient in available:
            continue

        replacement = candidate.substitutions.get(ingredient)
        if replacement and replacement in available:
            substitutions.append(
                Substitution(
                    original=ingredient,
                    replacement=replacement,
                    reason="optional_ingredient_replaced",
                )
            )
            continue

        missing_optional.append(ingredient)
        if ingredient in candidate.substitutions:
            substitutions.append(
                Substitution(
                    original=ingredient,
                    replacement=None,
                    reason="optional_ingredient_omitted",
                )
            )

    conflicts = _find_conflicts(candidate, food_directions)

    if missing_required:
        route = "no_ingredient"
        can_pass = allow_additional
        score = 45 if allow_additional else 5
    elif substitutions or missing_optional or conflicts:
        route = "simple"
        can_pass = True
        score = 75 if conflicts else 80
    else:
        route = "can_cook"
        can_pass = True
        score = 100

    score -= len(missing_required) * 10
    score -= len(missing_optional) * 2
    score -= len(conflicts) * 5
    score -= max(candidate.cooking_time_minutes - 10, 0) // 5

    return RecipeCandidateEvaluation(
        candidate_name=candidate.name,
        route=route,
        can_pass_to_agent5=can_pass,
        missing_required_ingredients=missing_required,
        missing_optional_ingredients=missing_optional,
        conflict_reasons=conflicts,
        substitutions=substitutions,
        score=score,
    )


def _ingredients_for_recipe(
    candidate: CandidateFood,
    available_ingredients: list[str],
    substitutions: list[Substitution],
    known_seasonings: set[str] | None = None,
) -> list[str]:
    available = set(available_ingredients)
    seasonings = known_seasonings or set()
    replacement_by_original = {
        item.original: item.replacement for item in substitutions if item.replacement
    }
    ingredients: list[str] = []

    for ingredient in (
        candidate.core_ingredients
        + candidate.required_ingredients
        + candidate.optional_ingredients
    ):
        if ingredient in seasonings:
            continue

        if ingredient in available:
            ingredients.append(ingredient)
            continue

        replacement = replacement_by_original.get(ingredient)
        if replacement:
            if replacement in seasonings:
                continue
            ingredients.append(replacement)

    return _normalize_items(ingredients)


def _seasonings_for_recipe(
    candidate: CandidateFood,
    available_ingredients: list[str],
    known_seasonings: set[str] | None = None,
) -> list[str]:
    available = set(available_ingredients)
    seasonings_from_info = known_seasonings or set()
    seasonings = [
        seasoning for seasoning in candidate.seasonings if seasoning in available
    ]
    seasonings.extend(
        ingredient
        for ingredient in (
            candidate.core_ingredients
            + candidate.required_ingredients
            + candidate.optional_ingredients
        )
        if ingredient in seasonings_from_info and ingredient in available
    )

    if not seasonings and candidate.seasonings:
        seasonings = candidate.seasonings[:1]

    return _normalize_items(seasonings)


def _route_message(
    evaluation: RecipeCandidateEvaluation,
    ingredient_policy: str,
) -> str:
    if evaluation.route == "can_cook":
        return "current_ingredients_are_enough"

    if evaluation.route == "simple":
        return "recipe_can_be_made_with_substitution_or_omission"

    if evaluation.route == "no_ingredient" and ingredient_policy == "allow_additional":
        return "required_ingredients_missing_but_user_allows_additional_ingredients"

    if evaluation.route == "no_ingredient":
        return "required_ingredients_missing"

    return "candidate_conflicts_with_user_context"


def _coerce_request(data: RecipeRouterInput | Mapping[str, Any]) -> RecipeRouterInput:
    if isinstance(data, RecipeRouterInput):
        return data

    payload = dict(data)
    payload["recipe_type"] = _normalize_recipe_type(payload.get("recipe_type"))
    return RecipeRouterInput.model_validate(payload)


def route_recipe(data: RecipeRouterInput | Mapping[str, Any]) -> RecipeRouterOutput:
    """Evaluate provided candidates and prepare the selected one for agent5."""

    request = _coerce_request(data)
    candidates = _candidate_pool(request)
    available_ingredients = _collect_available_ingredients(request)
    available_seasonings = _collect_available_seasonings(request)
    allow_additional = request.ingredient_policy == "allow_additional"

    if not request.candidate_foods:
        return RecipeRouterOutput(
            route="no_ingredient",
            route_message="candidate_foods_required",
            can_pass_to_agent5=False,
        )

    if not candidates:
        return RecipeRouterOutput(
            candidate_foods=request.candidate_foods,
            route="conflict",
            route_message="no_candidate_matches_recipe_type",
            can_pass_to_agent5=False,
        )

    if not available_ingredients:
        first_candidate_core = (
            candidates[0].core_ingredients or candidates[0].required_ingredients
        )
        return RecipeRouterOutput(
            candidate_foods=candidates,
            route="no_ingredient",
            route_message="available_ingredients_required",
            additional_ingredients=first_candidate_core,
            can_pass_to_agent5=False,
        )

    evaluations = [
        _evaluate_candidate(
            candidate=candidate,
            available_ingredients=available_ingredients,
            food_directions=request.food_directions,
            allow_additional=allow_additional,
        )
        for candidate in candidates
    ]
    evaluation_by_name = {item.candidate_name: item for item in evaluations}
    passable_candidates = [
        candidate
        for candidate in candidates
        if evaluation_by_name[candidate.name].can_pass_to_agent5
    ]
    selected_candidate = max(
        passable_candidates or candidates,
        key=lambda candidate: evaluation_by_name[candidate.name].score,
    )
    selected_evaluation = evaluation_by_name[selected_candidate.name]

    selected_recipe = None
    ingredients_to_use: list[str] = []
    seasonings_to_use: list[str] = []

    if selected_evaluation.can_pass_to_agent5:
        ingredients_to_use = _ingredients_for_recipe(
            selected_candidate,
            available_ingredients,
            selected_evaluation.substitutions,
            available_seasonings,
        )
        seasonings_to_use = _seasonings_for_recipe(
            selected_candidate,
            available_ingredients,
            available_seasonings,
        )
        selected_recipe = SelectedRecipe(
            name=selected_candidate.name,
            recipe_type=selected_candidate.recipe_type,
            reason=_build_selected_recipe_reason(
                candidate=selected_candidate,
                request=request,
                evaluation=selected_evaluation,
                ingredients_to_use=ingredients_to_use,
                seasonings_to_use=seasonings_to_use,
            ),
        )

    return RecipeRouterOutput(
        candidate_foods=candidates,
        candidate_evaluations=evaluations,
        route=selected_evaluation.route,
        route_message=_route_message(selected_evaluation, request.ingredient_policy),
        selected_recipe=selected_recipe,
        ingredients_to_use=ingredients_to_use,
        seasonings_to_use=seasonings_to_use,
        substitutions=selected_evaluation.substitutions,
        additional_ingredients=selected_evaluation.missing_required_ingredients,
        can_pass_to_agent5=selected_evaluation.can_pass_to_agent5,
    )


def route_recipe_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """LangGraph-friendly wrapper that generates candidates before routing."""

    agent4_state = dict(state)
    for field_name in ("ingredient_info", "food_directions"):
        field_value = agent4_state.get(field_name)
        if hasattr(field_value, "model_dump"):
            agent4_state[field_name] = field_value.model_dump()

    if not agent4_state.get("candidate_foods"):
        candidate_foods, candidate_error = _build_agent4_candidates(agent4_state)
        if candidate_error:
            return _candidate_generation_failed(candidate_error)
        agent4_state["candidate_foods"] = candidate_foods

    return route_recipe(agent4_state).model_dump()
