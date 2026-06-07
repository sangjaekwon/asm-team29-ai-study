"""Rule-based service for agent4, the feasible recipe router."""

from collections.abc import Mapping
from typing import Any

from agents.agent4.schema import (
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


_DIFFICULTY_RANK = {"easy": 1, "normal": 2, "hard": 3}

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
    missing_required: list[str] = []

    for ingredient in candidate.required_ingredients:
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
    elif conflicts:
        route = "conflict"
        can_pass = False
        score = 0
    elif substitutions or missing_optional:
        route = "simple"
        can_pass = True
        score = 80
    else:
        route = "can_cook"
        can_pass = True
        score = 100

    score -= len(missing_required) * 10
    score -= len(missing_optional) * 2
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

    for ingredient in candidate.required_ingredients + candidate.optional_ingredients:
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
        for ingredient in candidate.required_ingredients + candidate.optional_ingredients
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
        return RecipeRouterOutput(
            candidate_foods=candidates,
            route="no_ingredient",
            route_message="available_ingredients_required",
            additional_ingredients=candidates[0].required_ingredients,
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
        selected_recipe = SelectedRecipe(
            name=selected_candidate.name,
            recipe_type=selected_candidate.recipe_type,
            reason=selected_candidate.reason,
        )
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
    """LangGraph-friendly wrapper that returns only updated state fields."""

    return route_recipe(state).model_dump()
