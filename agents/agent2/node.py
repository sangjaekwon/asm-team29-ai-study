from agents.schemas import AgentState, ContextAnalyzerOutput

from .service import Agent2Service


service = Agent2Service()


def analyze_context(state: AgentState) -> ContextAnalyzerOutput:
    return service.analyze(
        user_mood_input=state.get("user_mood_input", ""),
        user_situation_input=state.get("user_situation_input", ""),
    )
