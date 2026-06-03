"""1번 에이전트의 재료 파악 서비스.

사용자가 직접 입력한 재료 목록을 Solar API로 표준화하고 분류한다.
API 키가 없거나 LLM 응답을 사용할 수 없는 경우 규칙 기반 fallback을
사용한다. 이미지 기반 Vision 모델 연동은 후속 작업에서 확장한다.
"""

import json
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from agents.schemas import (
    AgentState,
    DetectedIngredient,
    IngredientAnalyzerOutput,
    IngredientCategory,
    IngredientInfo,
    IngredientSource,
    NutritionType,
)


load_dotenv()

SOLAR_BASE_URL = "https://api.upstage.ai/v1"
DEFAULT_SOLAR_MODEL = "solar-pro3"
TEST_API_KEYS = {"", "test", "dummy", "your_upstage_api_key_here"}
VALID_CATEGORIES = {"main", "sub", "seasoning"}
VALID_NUTRITION_TYPES = {
    "carbohydrate",
    "protein",
    "fat",
    "vegetable",
    "seasoning",
}

SYSTEM_PROMPT = """당신은 AI 요리 도우미의 재료 파악 에이전트입니다.
사용자가 입력한 한국어/영어 식재료 표현을 한국어 표준 식재료 이름으로 정리하고 분류하세요.

반드시 JSON만 반환하세요. 마크다운 코드블록이나 설명 문장은 쓰지 마세요.
반환 형식:
{
  "ingredients": [
    {
      "name": "한국어로 표준화된 식재료 이름",
      "category": "main | sub | seasoning",
      "nutrition_type": "carbohydrate | protein | fat | vegetable | seasoning",
      "confidence": 0.0부터 1.0 사이 숫자,
      "needs_confirmation": true 또는 false
    }
  ],
  "message": "사용자에게 보여줄 짧은 안내 문장"
}

분류 기준:
- category가 seasoning이면 nutrition_type도 seasoning으로 설정합니다.
- egg, onion처럼 영어 재료명이 들어오면 계란, 양파처럼 한국어 표준명으로 변환합니다.
- 밥, 면, 빵, 감자류는 carbohydrate입니다.
- 계란, 육류, 생선, 두부, 해산물은 protein입니다.
- 버터, 치즈, 오일류는 fat입니다.
- 채소, 버섯, 김치, 허브류는 vegetable입니다.
- 애매한 재료는 confidence를 낮추고 needs_confirmation을 true로 설정합니다.
"""

def _get_solar_api_key() -> str:
    return os.getenv("SOLAR_API_KEY", "").strip()


def _get_solar_model() -> str:
    return os.getenv("SOLAR_MODEL", DEFAULT_SOLAR_MODEL).strip() or DEFAULT_SOLAR_MODEL


def _should_call_solar() -> bool:
    return _get_solar_api_key().lower() not in TEST_API_KEYS


def _clean_name(name: str) -> str:
    return " ".join(name.strip().split())


def _unique_ingredients(ingredients: list[str]) -> list[str]:
    unique: list[str] = []
    for ingredient in ingredients:
        cleaned = _clean_name(ingredient)
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return unique


def _build_detected_ingredient(
    name: str,
    source: IngredientSource = "manual",
) -> DetectedIngredient:
    return DetectedIngredient(
        name=name,
        category="sub",
        nutrition_type="vegetable",
        confidence=1.0,
        needs_confirmation=False,
        source=source,
    )


def _coerce_nutrition_type(item: dict[str, Any]) -> NutritionType:
    nutrition_type = str(item.get("nutrition_type", "")).strip().lower()
    category = str(item.get("category", "")).strip().lower()

    if nutrition_type in VALID_NUTRITION_TYPES:
        return nutrition_type  # type: ignore[return-value]
    if category in VALID_NUTRITION_TYPES:
        return category  # type: ignore[return-value]
    return "vegetable"


def _coerce_category(item: dict[str, Any], nutrition_type: NutritionType) -> IngredientCategory:
    category = str(item.get("category", "")).strip().lower()

    if category in VALID_CATEGORIES:
        return category  # type: ignore[return-value]
    if nutrition_type == "seasoning":
        return "seasoning"
    if nutrition_type in {"carbohydrate", "protein"}:
        return "main"
    return "sub"


def _deduplicate_detected_ingredients(
    detected_ingredients: list[DetectedIngredient],
) -> list[DetectedIngredient]:
    unique: list[DetectedIngredient] = []
    seen: set[str] = set()

    for ingredient in detected_ingredients:
        cleaned_name = _clean_name(ingredient.name)
        if not cleaned_name or cleaned_name in seen:
            continue

        seen.add(cleaned_name)
        unique.append(ingredient.model_copy(update={"name": cleaned_name}))

    return unique


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


def _build_raw_result(
    state: AgentState,
    analysis_source: str,
    llm_result: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "image_path": state.get("image_path", ""),
        "image_id": state.get("image_id", ""),
        "analysis_source": analysis_source,
        "vision_status": "not_connected",
        "vision_message": "이미지 분석은 아직 연결되지 않았습니다.",
    }

    if llm_result is not None:
        result["llm_result"] = llm_result
    if error:
        result["error"] = error

    return result


def _extract_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    return json.loads(cleaned)


def _build_solar_messages(ingredients: list[str]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                {"user_input_ingredients": ingredients},
                ensure_ascii=False,
            ),
        },
    ]


def _call_solar_ingredient_analyzer(
    ingredients: list[str],
) -> tuple[list[DetectedIngredient], dict[str, Any]]:
    client = OpenAI(
        api_key=_get_solar_api_key(),
        base_url=SOLAR_BASE_URL,
    )

    response = client.chat.completions.create(
        model=_get_solar_model(),
        messages=_build_solar_messages(ingredients),
        temperature=0,
        stream=False,
    )
    content = response.choices[0].message.content or "{}"
    parsed = _extract_json(content)

    detected_ingredients: list[DetectedIngredient] = []
    for item in parsed.get("ingredients", []):
        if not item.get("name"):
            continue

        nutrition_type = _coerce_nutrition_type(item)
        detected_ingredients.append(
            DetectedIngredient(
                name=item["name"],
                category=_coerce_category(item, nutrition_type),
                nutrition_type=nutrition_type,
                confidence=float(item.get("confidence", 1.0)),
                needs_confirmation=bool(item.get("needs_confirmation", False)),
                source="manual",
            )
        )

    return _deduplicate_detected_ingredients(detected_ingredients), parsed


def _analyze_with_rules(state: AgentState, error: str = "") -> dict[str, Any]:
    ingredients = _unique_ingredients(state.get("user_input_ingredients", []))

    if not ingredients:
        return IngredientAnalyzerOutput(
            detected_ingredients=[],
            uncertain_ingredients=[],
            available_ingredients=[],
            ingredient_info=IngredientInfo(),
            vision_status="no_ingredient_detected",
            vision_message="사용 가능한 재료가 입력되지 않았습니다.",
            raw_vision_result=_build_raw_result(
                state,
                analysis_source="rules",
                error=error,
            ),
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
        vision_message="사용자 입력 재료를 규칙 기반으로 표준화하고 분류했습니다.",
        raw_vision_result=_build_raw_result(
            state,
            analysis_source="rules",
            error=error,
        ),
    ).model_dump()


def _build_llm_output(
    state: AgentState,
    detected_ingredients: list[DetectedIngredient],
    llm_result: dict[str, Any],
) -> dict[str, Any]:
    if not detected_ingredients:
        return IngredientAnalyzerOutput(
            detected_ingredients=[],
            uncertain_ingredients=[],
            available_ingredients=[],
            ingredient_info=IngredientInfo(),
            vision_status="no_ingredient_detected",
            vision_message="Solar API가 사용 가능한 재료를 찾지 못했습니다.",
            raw_vision_result=_build_raw_result(
                state,
                analysis_source="solar",
                llm_result=llm_result,
            ),
        ).model_dump()

    uncertain_ingredients = [
        ingredient.name
        for ingredient in detected_ingredients
        if ingredient.needs_confirmation
    ]

    vision_status = "need_user_confirmation" if uncertain_ingredients else "success"
    vision_message = llm_result.get("message") or "Solar API로 재료를 표준화하고 분류했습니다."

    return IngredientAnalyzerOutput(
        detected_ingredients=detected_ingredients,
        uncertain_ingredients=uncertain_ingredients,
        available_ingredients=[ingredient.name for ingredient in detected_ingredients],
        ingredient_info=_build_ingredient_info(detected_ingredients),
        vision_status=vision_status,
        vision_message=vision_message,
        raw_vision_result=_build_raw_result(
            state,
            analysis_source="solar",
            llm_result=llm_result,
        ),
    ).model_dump()


def analyze_ingredients(state: AgentState) -> dict[str, Any]:
    """사용자 입력 재료를 표준화하고 다음 에이전트용 State를 생성한다.

    Solar API 키가 설정되어 있으면 LLM을 우선 사용하고, 실패하면 규칙 기반
    fallback을 사용한다.
    """

    ingredients = _unique_ingredients(state.get("user_input_ingredients", []))
    if not ingredients:
        return _analyze_with_rules(state)

    if not _should_call_solar():
        return _analyze_with_rules(state)

    try:
        detected_ingredients, llm_result = _call_solar_ingredient_analyzer(ingredients)
        return _build_llm_output(state, detected_ingredients, llm_result)
    except (KeyError, TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        return _analyze_with_rules(state, error=f"Solar 응답 파싱 실패: {exc}")
    except Exception as exc:
        return _analyze_with_rules(state, error=f"Solar API 호출 실패: {exc}")
