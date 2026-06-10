from agents.agent5.service import (
    _dedupe_recommendation_reasons,
    _ensure_optional_suggestions,
    _merge_additional_ingredients,
    _merge_substitutions,
    generate_recipe,
)
from agents.schemas import AgentState, SelectedRecipe


def test_generate_recipe_from_selected_recipe_and_ingredients():
    state: AgentState = {
        "selected_recipe": SelectedRecipe(
            name="김치볶음밥",
            recipe_type="korean",
            reason="가지고 있는 밥을 활용할 수 있습니다.",
        ),
        "ingredients_to_use": ["김치", "밥"],
        "seasonings_to_use": ["간장"],
        "servings": 2,
    }

    result = generate_recipe(state)

    assert result["generation_status"] == "success"
    assert result["generated_recipe"] is not None
    assert result["generated_recipe"].recipe_name == "김치볶음밥"
    assert result["generated_recipe"].ingredients == ["김치", "밥", "간장"]
    assert result["generated_recipe"].servings == 2
    assert result["generated_recipe"].cooking_steps


def test_generate_recipe_fails_without_recipe_or_ingredients():
    result = generate_recipe({})

    assert result["generation_status"] == "failed"
    assert result["generated_recipe"] is None
    assert "레시피" in result["generation_message"]


def test_agent5_optional_additional_ingredients_do_not_add_new_core_ingredients():
    result = _merge_additional_ingredients(
        agent4_additional=["김치"],
        agent5_additional=["대파", "김치", "후추"],
        protected_core_ingredients=["김치", "돼지고기"],
    )

    assert result == ["김치", "대파", "후추"]


def test_agent5_substitutions_do_not_replace_core_ingredients():
    result = _merge_substitutions(
        agent4_substitutions=[
            {"original": "대파", "replacement": None, "reason": "optional"}
        ],
        agent5_substitutions=[
            {"original": "김치", "replacement": "양배추", "reason": "core replacement"},
            {"original": "고추", "replacement": "파프리카", "reason": "optional"},
        ],
        protected_core_ingredients=["김치", "돼지고기"],
    )

    assert result == [
        {"original": "대파", "replacement": None, "reason": "optional"},
        {"original": "고추", "replacement": "파프리카", "reason": "optional"},
    ]


def test_agent5_adds_default_optional_suggestions_when_llm_returns_empty_lists():
    substitutions, additional = _ensure_optional_suggestions(
        selected_recipe={"name": "마르게리타 피자"},
        ingredients=["피자도우", "토마토소스", "모짜렐라치즈", "바질"],
        substitutions=[],
        additional_ingredients=[],
        protected_core_ingredients=["피자도우", "토마토소스", "모짜렐라치즈", "바질"],
    )

    assert substitutions
    assert additional
    assert all(item["original"] != "피자도우" for item in substitutions)
    assert "피자도우" not in additional


def test_agent5_adds_missing_substitutions_even_when_additional_exists():
    substitutions, additional = _ensure_optional_suggestions(
        selected_recipe={"name": "마르게리타 피자"},
        ingredients=["피자도우", "토마토소스", "모짜렐라치즈", "바질"],
        substitutions=[],
        additional_ingredients=["올리브오일"],
        protected_core_ingredients=["피자도우", "토마토소스", "모짜렐라치즈", "바질"],
    )

    assert substitutions
    assert substitutions[0]["original"] == "올리브오일"
    assert substitutions[0]["replacement"] == "식용유"
    assert additional == ["올리브오일"]


def test_recommendation_reasons_remove_repeated_sentence_and_paragraph():
    reasons = _dedupe_recommendation_reasons(
        [
            "피자도우와 토마토소스가 있어 바로 조리할 수 있습니다.",
            "피자도우와 토마토소스가 있어 바로 조리할 수 있습니다. 조리 시간이 짧아 지금 상황에 맞습니다.",
            "조리 시간이 짧아 지금 상황에 맞습니다.",
        ]
    )

    assert reasons == [
        "피자도우와 토마토소스가 있어 바로 조리할 수 있습니다.",
        "조리 시간이 짧아 지금 상황에 맞습니다.",
    ]
