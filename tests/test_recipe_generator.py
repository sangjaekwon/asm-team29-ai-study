from agents.agent5.service import generate_recipe
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
