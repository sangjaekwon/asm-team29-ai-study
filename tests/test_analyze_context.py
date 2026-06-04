from agents.agent2.service import Agent2Service
from agents.schemas import ContextAnalyzerOutput, FoodDirections


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


def test_agent2_analyze_empty_input_uses_default():
    service = Agent2Service()

    result = service.analyze(user_mood_input="", user_situation_input="")

    assert isinstance(result, ContextAnalyzerOutput)
    fd = result.food_directions
    assert fd.fatigue_level == "medium"
    assert fd.difficulty == "normal"
