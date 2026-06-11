"""Recipe recommendation multi-agent LangGraph workflow."""

from langgraph.graph import END, START, StateGraph

from agents.agent2.node import analyze_context
from agents.agent1.service import analyze_ingredients
from agents.agent3.node import route_cuisine
from agents.agent4.service import route_recipe_node
from agents.agent5.service import generate_recipe
from agents.schemas import AgentState


def route_after_ingredient_analyzer(state: AgentState) -> str:
    """Stop the graph when ingredient confirmation is required."""

    if state.get("vision_status") == "need_user_confirmation":
        return "wait_for_user_confirmation"

    return "continue"


def build_recipe_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("ingredient_analyzer", analyze_ingredients)
    workflow.add_node("context_analyzer", analyze_context)
    workflow.add_node("cuisine_router", route_cuisine)
    workflow.add_node("recipe_router", route_recipe_node)
    workflow.add_node("recipe_generator", generate_recipe)

    workflow.add_edge(START, "ingredient_analyzer")
    workflow.add_edge(START, "context_analyzer")
    workflow.add_conditional_edges(
        "ingredient_analyzer",
        route_after_ingredient_analyzer,
        {
            "continue": "cuisine_router",
            "wait_for_user_confirmation": END,
        },
    )
    workflow.add_edge("context_analyzer", "cuisine_router")
    workflow.add_edge("cuisine_router", "recipe_router")
    workflow.add_edge("recipe_router", "recipe_generator")
    workflow.add_edge("recipe_generator", END)

    return workflow.compile()


def run_recipe_graph(state: AgentState) -> AgentState:
    return build_recipe_graph().invoke(state)
