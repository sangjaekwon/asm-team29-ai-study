"""1번 에이전트의 재료 파악 서비스.

MVP 단계에서는 사용자가 직접 입력한 재료 목록을 기준으로 표준화와
분류를 수행한다. 이미지 기반 Vision 모델 연동은 후속 작업에서 확장한다.
"""

from typing import Any

from agents.schemas import (
    AgentState,
    DetectedIngredient,
    IngredientAnalyzerOutput,
    IngredientCategory,
    IngredientInfo,
    IngredientSource,
    NutritionType,
)


STANDARD_INGREDIENT_NAMES = {
    "달걀": "계란",
    "계란후라이": "계란",
    "흰쌀밥": "밥",
    "쌀밥": "밥",
    "대파": "파",
    "쪽파": "파",
    "돼지 고기": "돼지고기",
    "소 고기": "소고기",
    "닭 고기": "닭고기",
}

SEASONINGS = {
    "간장",
    "고추장",
    "된장",
    "소금",
    "설탕",
    "후추",
    "식초",
    "고춧가루",
    "다진마늘",
    "마늘가루",
    "참기름",
    "굴소스",
    "두반장",
    "미림",
    "사케",
    "와사비",
    "케첩",
    "마요네즈",
    "올리고당",
    "물엿",
    "MSG",
}

CARBOHYDRATES = {
    "밥",
    "쌀",
    "면",
    "라면",
    "파스타",
    "국수",
    "우동",
    "빵",
    "식빵",
    "감자",
    "고구마",
    "떡",
    "밀가루",
}

PROTEINS = {
    "계란",
    "달걀",
    "돼지고기",
    "소고기",
    "닭고기",
    "닭가슴살",
    "연어",
    "참치",
    "두부",
    "햄",
    "베이컨",
    "새우",
    "오징어",
    "고등어",
    "참치캔",
}

FATS = {
    "버터",
    "치즈",
    "생크림",
    "올리브오일",
    "식용유",
    "카놀라유",
    "들기름",
    "참기름",
    "아보카도",
}

VEGETABLES = {
    "양파",
    "파",
    "마늘",
    "당근",
    "배추",
    "양배추",
    "상추",
    "깻잎",
    "오이",
    "애호박",
    "버섯",
    "표고버섯",
    "느타리버섯",
    "토마토",
    "브로콜리",
    "청경채",
    "죽순",
    "목이버섯",
    "김치",
    "콩나물",
    "숙주",
    "대파",
}


def _normalize_name(name: str) -> str:
    normalized = " ".join(name.strip().split())
    return STANDARD_INGREDIENT_NAMES.get(normalized, normalized)


def _unique_ingredients(ingredients: list[str]) -> list[str]:
    unique: list[str] = []
    for ingredient in ingredients:
        normalized = _normalize_name(ingredient)
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _classify_nutrition(name: str) -> NutritionType:
    if name in SEASONINGS:
        return "seasoning"
    if name in PROTEINS:
        return "protein"
    if name in CARBOHYDRATES:
        return "carbohydrate"
    if name in FATS:
        return "fat"
    if name in VEGETABLES:
        return "vegetable"
    return "vegetable"


def _classify_category(nutrition_type: NutritionType) -> IngredientCategory:
    if nutrition_type == "seasoning":
        return "seasoning"
    if nutrition_type in {"carbohydrate", "protein"}:
        return "main"
    return "sub"


def _build_detected_ingredient(
    name: str,
    source: IngredientSource = "manual",
) -> DetectedIngredient:
    nutrition_type = _classify_nutrition(name)
    return DetectedIngredient(
        name=name,
        category=_classify_category(nutrition_type),
        nutrition_type=nutrition_type,
        confidence=1.0,
        needs_confirmation=False,
        source=source,
    )


def _build_ingredient_info(detected_ingredients: list[DetectedIngredient]) -> IngredientInfo:
    info = IngredientInfo()

    for ingredient in detected_ingredients:
        if ingredient.category == "main":
            info.main_ingredients.append(ingredient.name)
        elif ingredient.category == "sub":
            info.sub_ingredients.append(ingredient.name)
        else:
            info.seasonings.append(ingredient.name)

        if ingredient.nutrition_type == "carbohydrate":
            info.carbohydrates.append(ingredient.name)
        elif ingredient.nutrition_type == "protein":
            info.proteins.append(ingredient.name)
        elif ingredient.nutrition_type == "fat":
            info.fats.append(ingredient.name)
        elif ingredient.nutrition_type == "vegetable":
            info.vegetables.append(ingredient.name)

    return info


def _build_raw_vision_result(state: AgentState) -> dict[str, Any]:
    image_path = state.get("image_path", "")
    image_id = state.get("image_id", "")
    if not image_path and not image_id:
        return {}

    return {
        "image_path": image_path,
        "image_id": image_id,
        "status": "not_connected",
        "message": "MVP에서는 이미지 분석 대신 사용자 입력 재료를 사용합니다.",
    }


def analyze_ingredients(state: AgentState) -> dict[str, Any]:
    """사용자 입력 재료를 표준화하고 다음 에이전트용 State를 생성한다."""

    ingredients = _unique_ingredients(state.get("user_input_ingredients", []))

    if not ingredients:
        return IngredientAnalyzerOutput(
            detected_ingredients=[],
            uncertain_ingredients=[],
            available_ingredients=[],
            ingredient_info=IngredientInfo(),
            vision_status="no_ingredient_detected",
            vision_message="사용 가능한 재료가 입력되지 않았습니다.",
            raw_vision_result=_build_raw_vision_result(state),
        ).model_dump()

    detected_ingredients = [
        _build_detected_ingredient(ingredient) for ingredient in ingredients
    ]

    return IngredientAnalyzerOutput(
        detected_ingredients=detected_ingredients,
        uncertain_ingredients=[],
        available_ingredients=ingredients,
        ingredient_info=_build_ingredient_info(detected_ingredients),
        vision_status="success",
        vision_message="사용자가 입력한 재료를 표준화하고 분류했습니다.",
        raw_vision_result=_build_raw_vision_result(state),
    ).model_dump()
