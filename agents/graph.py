"""레시피 추천 멀티 에이전트 LangGraph 워크플로."""

from langgraph.graph import END, StateGraph

from agents.agent2.node import analyze_context
from agents.agent1.service import analyze_ingredients
from agents.agent3.node import route_cuisine
from agents.agent4.service import route_recipe_node
from agents.agent5.service import generate_recipe
from agents.schemas import AgentState
"""1번 에이전트 구현을 시작함에 따라, 이전 stub는 삭제할게요~"""

# def route_recipe(state: AgentState) -> RecipeRouterOutput:
#     """4번 에이전트 구현 전까지 사용하는 가능 레시피 라우터 stub."""
#
#     ingredients = state.get("available_ingredients", [])
#     recipe_type = state.get("recipe_type") or "korean"
#
#     if not ingredients:
#         return RecipeRouterOutput(
#             route="missing_ingredient",
#             route_message="사용 가능한 재료가 입력되지 않았습니다.",
#             selected_recipe=None,
#         )
#
#     selected_recipe = SelectedRecipe(
#         name=f"간단한 {' '.join(ingredients)}",
#         recipe_type=recipe_type,
#         reason="현재 사용할 수 있는 재료를 바탕으로 구성했습니다.",
#     )
#
#     return RecipeRouterOutput(
#         candidate_foods=[
#             CandidateFood(
#                 name=selected_recipe.name,
#                 recipe_type=recipe_type,
#                 required_ingredients=ingredients,
#                 reason="현재 보유한 재료를 사용합니다.",
#             )
#         ],
#         route="can_cook",
#         route_message="사용 가능한 재료로 간단한 레시피를 만들 수 있습니다.",
#         selected_recipe=selected_recipe,
#         ingredients_to_use=ingredients,
#         seasonings_to_use=[],
#     )
""" 머지과정중 잘못 덮어씌인부분"""


def _default_candidate_foods(state: AgentState) -> list[dict]:
    """Agent4 demo fallback candidates when no upstream candidate generator exists."""

    recipe_type = state.get("recipe_type") or "korean"
    ingredient_info = state.get("ingredient_info")
    available_ingredients = state.get("available_ingredients", [])

    if isinstance(ingredient_info, dict):
        main_ingredients = ingredient_info.get("main_ingredients", [])
        sub_ingredients = ingredient_info.get("sub_ingredients", [])
        seasonings = ingredient_info.get("seasonings", [])
    elif ingredient_info is not None:
        main_ingredients = ingredient_info.main_ingredients
        sub_ingredients = ingredient_info.sub_ingredients
        seasonings = ingredient_info.seasonings
    else:
        main_ingredients = []
        sub_ingredients = []
        seasonings = []
    primary = (main_ingredients or available_ingredients or ["재료"])[0]
    dynamic_optional = [
        item
        for item in available_ingredients
        if item != primary and item not in seasonings
    ][:4]
    dynamic_candidate_name = (
        f"간단한 {primary} 한 접시" if primary != "재료" else "간단한 재료 한 접시"
    )
    dynamic_candidate = {
        "name": dynamic_candidate_name,
        "recipe_type": recipe_type,
        "required_ingredients": [primary],
        "optional_ingredients": dynamic_optional or sub_ingredients[:4],
        "seasonings": seasonings[:4],
        "substitutions": {
            item: None for item in (dynamic_optional or sub_ingredients[:2])
        },
        "difficulty": "easy",
        "cooking_time_minutes": 10,
        "taste_profile": ["savory"],
        "cooking_methods": [
            "빠른 조리",
            "간편 조리",
            "팬 조리",
            "볶기",
            "굽기",
            "끓이기",
            "삶기",
        ],
        "reason": "현재 인식된 재료를 반드시 활용하기 위한 데모 후보입니다.",
    }

    candidates = [
        dynamic_candidate,
        {
            "name": "계란볶음밥",
            "recipe_type": "korean",
            "required_ingredients": ["밥", "계란"],
            "optional_ingredients": ["대파", "양파"],
            "seasonings": ["간장", "소금", "참기름"],
            "substitutions": {"대파": None, "양파": None},
            "difficulty": "easy",
            "cooking_time_minutes": 10,
            "taste_profile": ["savory"],
            "cooking_methods": ["빠른 조리", "팬 조리", "볶기", "굽기", "끓이기", "삶기", "간편 조리"],
            "reason": "밥과 계란이 있으면 빠르게 만들 수 있는 기본 한식 후보입니다.",
        },
        {
            "name": "제육볶음",
            "recipe_type": "korean",
            "required_ingredients": ["돼지고기"],
            "optional_ingredients": ["양파", "대파", "고추", "채소"],
            "seasonings": ["고추장", "간장", "설탕", "다진마늘"],
            "substitutions": {"채소": None, "고추": None},
            "difficulty": "easy",
            "cooking_time_minutes": 15,
            "taste_profile": ["spicy", "savory"],
            "cooking_methods": ["빠른 조리", "팬 조리", "볶기", "굽기", "끓이기", "삶기", "간편 조리"],
            "reason": "돼지고기와 매콤한 양념을 활용하기 좋은 후보입니다.",
        },
        {
            "name": "두부조림",
            "recipe_type": "korean",
            "required_ingredients": ["두부"],
            "optional_ingredients": ["대파", "양파", "고추"],
            "seasonings": ["간장", "고춧가루", "설탕"],
            "substitutions": {"고추": None},
            "difficulty": "easy",
            "cooking_time_minutes": 15,
            "taste_profile": ["savory"],
            "cooking_methods": ["빠른 조리", "팬 조리", "조림", "굽기", "끓이기", "삶기", "간편 조리"],
            "reason": "단백질 재료와 기본 양념으로 만들 수 있는 후보입니다.",
        },
    ]

    return candidates


def route_recipe(state: AgentState) -> dict:
    """Agent4 구현체를 LangGraph state에 연결한다."""

    agent4_state = dict(state)
    for field_name in ("ingredient_info", "food_directions"):
        field_value = agent4_state.get(field_name)
        if hasattr(field_value, "model_dump"):
            agent4_state[field_name] = field_value.model_dump()

    if not agent4_state.get("candidate_foods"):
        agent4_state["candidate_foods"] = _default_candidate_foods(state)
    return route_recipe_node(agent4_state)


def route_after_ingredient_analyzer(state: AgentState) -> str:
    """재료 확인이 필요한 경우 다음 agent 실행을 멈춘다."""

    if state.get("vision_status") == "need_user_confirmation":
        return "wait_for_user_confirmation"

    return "continue"


def build_recipe_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("ingredient_analyzer", analyze_ingredients)
    workflow.add_node("context_analyzer", analyze_context)
    workflow.add_node("cuisine_router", route_cuisine)
    workflow.add_node("recipe_router", route_recipe)
    workflow.add_node("recipe_generator", generate_recipe)

    workflow.set_entry_point("ingredient_analyzer")
    workflow.add_conditional_edges(
        "ingredient_analyzer",
        route_after_ingredient_analyzer,
        {
            "continue": "context_analyzer",
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
