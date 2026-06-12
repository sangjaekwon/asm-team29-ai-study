from agents.agent2.node import analyze_context
from agents.agent2.service import Agent2Service
from agents.schemas import AgentState, ContextAnalyzerOutput, FoodDirections


def test_agent2_analyze_tired():
    service = Agent2Service()

    result = service.analyze(
        user_mood_input="피곤하다",
        user_situation_input="퇴근 직후",
    )

    assert isinstance(result, ContextAnalyzerOutput)
    fd = result.food_directions
    assert isinstance(fd, FoodDirections)
    assert fd.fatigue_level == "high"
    assert fd.difficulty == "easy"
    assert fd.cooking_time_limit_minutes is not None
    assert fd.cooking_time_limit_minutes <= 15

    print(result.model_dump())


def test_agent2_analyze_relaxed():
    service = Agent2Service()

    result = service.analyze(
        user_mood_input="기분 좋다",
        user_situation_input="여유 있는 주말",
    )

    assert isinstance(result, ContextAnalyzerOutput)
    fd = result.food_directions
    assert fd.difficulty in ("normal", "hard")


def test_agent2_node_skips_when_inputs_empty():
    state: AgentState = {"user_mood_input": "", "user_situation_input": ""}

    update = analyze_context(state)

    assert update == {}, "빈 입력일 땐 state 갱신 없이 비어 있어야 한다"


def test_agent2_node_skips_when_inputs_whitespace_only():
    state: AgentState = {"user_mood_input": "   ", "user_situation_input": "\n"}

    update = analyze_context(state)

    assert update == {}


def test_agent2_node_skips_when_food_directions_already_present():
    """이전 트레이스에서 받은 food_directions가 있으면 재호출하지 않는다."""

    state: AgentState = {
        "user_mood_input": "피곤하다",
        "user_situation_input": "퇴근 직후",
        "food_directions": FoodDirections(fatigue_level="high", difficulty="easy").model_dump(),
    }

    update = analyze_context(state)

    assert update == {}
