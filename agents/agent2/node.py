from agents.schemas import AgentState, ContextAnalyzerOutput

from .service import Agent2Service


service = Agent2Service()


def analyze_context(state: AgentState) -> ContextAnalyzerOutput | dict:
    """기분/상황 분석 노드.

    사용자 입력(mood/situation)이 모두 비어 있으면 LLM을 호출하지 않고
    state를 갱신하지 않는다. 후속 노드(agent3 등)는 `food_directions`가
    state에 없을 수 있다는 가정하에 동작해야 한다.
    """

    user_mood_input = state.get("user_mood_input", "")
    user_situation_input = state.get("user_situation_input", "")

    if not user_mood_input.strip() and not user_situation_input.strip():
        return {}

    return service.analyze(
        user_mood_input=user_mood_input,
        user_situation_input=user_situation_input,
    )
