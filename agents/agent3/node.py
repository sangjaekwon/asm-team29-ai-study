from agents.schemas import AgentState, CuisineRouterOutput

from .service import Agent3Service

service = Agent3Service()

def route_cuisine(state: AgentState) -> CuisineRouterOutput:
    return service.classify(
            ingredient_info=state["ingredient_info"],
            food_directions=state["food_directions"]
            )
