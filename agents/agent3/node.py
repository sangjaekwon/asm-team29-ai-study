from agents.schemas import AgentState, CuisineRouterOutput, FoodDirections, IngredientInfo

from .service import Agent3Service

service = Agent3Service()

def route_cuisine(state: AgentState) -> CuisineRouterOutput:
    ingredient_info = state["ingredient_info"]
    if isinstance(ingredient_info, dict):
        ingredient_info = IngredientInfo.model_validate(ingredient_info)

    food_directions = state.get("food_directions")
    if isinstance(food_directions, dict):
        food_directions = FoodDirections.model_validate(food_directions)

    return service.classify(
            ingredient_info=ingredient_info,
            food_directions=food_directions,
            )
