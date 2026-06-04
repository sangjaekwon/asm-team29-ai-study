"""레시피 추천 멀티 에이전트 LangGraph 워크플로."""

from langgraph.graph import END, StateGraph

from agents.agent2.node import analyze_context
from agents.agent3.node import route_cuisine
from agents.agent5.service import generate_recipe
from agents.schemas import (
    AgentState,
    CandidateFood,
    DetectedIngredient,
    IngredientAnalyzerOutput,
    IngredientInfo,
    RecipeRouterOutput,
    SelectedRecipe,
)


def analyze_ingredients(state: AgentState) -> IngredientAnalyzerOutput:
    """1번 에이전트 구현 전까지 사용하는 재료 분석 stub."""

    ingredients = state.get("user_input_ingredients", [])
    detected = [
        DetectedIngredient(
            name=ingredient,
            category="main",
            nutrition_type="protein" if ingredient == "계란" else "carbohydrate",
            source="manual",
        )
        for ingredient in ingredients
    ]

    return IngredientAnalyzerOutput(
        detected_ingredients=detected,
        available_ingredients=ingredients,
        ingredient_info=IngredientInfo(main_ingredients=ingredients),
        vision_status="success",
        vision_message="사용자가 입력한 재료를 사용했습니다.",
    )


def route_recipe(state: AgentState) -> RecipeRouterOutput:
    """4번 에이전트 구현 전까지 사용하는 가능 레시피 라우터 stub."""

    ingredients = state.get("available_ingredients", [])
    recipe_type = state.get("recipe_type") or "korean"

    if not ingredients:
        return RecipeRouterOutput(
            route="missing_ingredient",
            route_message="사용 가능한 재료가 입력되지 않았습니다.",
            selected_recipe=None,
        )

    selected_recipe = SelectedRecipe(
        name=f"간단한 {' '.join(ingredients)}",
        recipe_type=recipe_type,
        reason="현재 사용할 수 있는 재료를 바탕으로 구성했습니다.",
    )

    return RecipeRouterOutput(
        candidate_foods=[
            CandidateFood(
                name=selected_recipe.name,
                recipe_type=recipe_type,
                required_ingredients=ingredients,
                reason="현재 보유한 재료를 사용합니다.",
            )
        ],
        route="can_cook",
        route_message="사용 가능한 재료로 간단한 레시피를 만들 수 있습니다.",
        selected_recipe=selected_recipe,
        ingredients_to_use=ingredients,
        seasonings_to_use=[],
    )


def build_recipe_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("ingredient_analyzer", analyze_ingredients)
    workflow.add_node("context_analyzer", analyze_context)
    workflow.add_node("cuisine_router", route_cuisine)
    workflow.add_node("recipe_router", route_recipe)
    workflow.add_node("recipe_generator", generate_recipe)

    workflow.set_entry_point("ingredient_analyzer")
    workflow.add_edge("ingredient_analyzer", "context_analyzer")
    workflow.add_edge("context_analyzer", "cuisine_router")
    workflow.add_edge("cuisine_router", "recipe_router")
    workflow.add_edge("recipe_router", "recipe_generator")
    workflow.add_edge("recipe_generator", END)

    return workflow.compile()


def run_recipe_graph(state: AgentState) -> AgentState:
    return build_recipe_graph().invoke(state)
