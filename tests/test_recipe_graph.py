from agents.graph import build_recipe_graph, run_recipe_graph
from agents.schemas import AgentState


def test_recipe_graph_runs_stub_agents_and_generates_recipe():
    state: AgentState = {
        "user_input_ingredients": ["계란", "밥"],
        "user_mood_input": "피곤해",
        "user_situation_input": "빠른 저녁",
        "servings": 2,
    }

    result = run_recipe_graph(state)

    assert result["vision_status"] == "success"
    assert result["recipe_type"] == "korean"
    assert result["route"] == "can_cook"
    assert result["generation_status"] == "success"
    assert result["generated_recipe"] is not None
    assert result["generated_recipe"].recipe_name == "간단한 계란 밥"
    assert result["generated_recipe"].servings == 2
    assert "계란" in result["generated_recipe"].ingredients
    assert "밥" in result["generated_recipe"].ingredients


def test_recipe_graph_compiles():
    graph = build_recipe_graph()

    assert graph is not None
