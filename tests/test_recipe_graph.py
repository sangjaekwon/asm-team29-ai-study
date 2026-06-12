from agents.graph import build_recipe_graph, route_after_analysis, run_recipe_graph
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

def test_route_after_analysis_continues_when_no_confirmation_needed():
    assert route_after_analysis({"vision_status": "success"}) == "continue"
    assert route_after_analysis({"vision_status": "no_ingredient_detected"}) == "continue"
    assert route_after_analysis({}) == "continue"


def test_route_after_analysis_stops_when_confirmation_needed():
    assert (
        route_after_analysis({"vision_status": "need_user_confirmation"})
        == "wait_for_user_confirmation"
    )


def test_recipe_graph_stops_before_cuisine_router_when_confirmation_needed(monkeypatch):
    import agents.graph as graph_module

    def fake_ingredient_analyzer(state):
        return {"vision_status": "need_user_confirmation", "uncertain_ingredients": ["계란"]}

    monkeypatch.setattr(graph_module, "analyze_ingredients", fake_ingredient_analyzer)

    result = run_recipe_graph(
        {
            "user_mood_input": "피곤해",
            "user_situation_input": "빠른 저녁",
            "servings": 1,
        }
    )

    assert result["vision_status"] == "need_user_confirmation"
    assert "recipe_type" not in result
    assert "route" not in result
    assert "generation_status" not in result
    # context_analyzer still runs and produces food_directions for the
    # follow-up trace, even though cuisine_router never executes.
    assert "food_directions" in result


def test_recipe_graph_uses_agent3_node():
    result = run_recipe_graph(
        {
            "user_input_ingredients": ["계란", "밥"],
            "user_mood_input": "배고픔",
            "user_situation_input": "집에서 간단히 먹고 싶음",
            "servings": 1,
        }
    )

    assert "recipe_type" in result
    assert result["recipe_type"] in [
        "korean",
        "chinese",
        "japanese",
        "western"
    ]
    assert "recipe_type_reason" in result
    assert isinstance(result["recipe_type_reason"], str)
    assert len(result["recipe_type_reason"]) > 0
