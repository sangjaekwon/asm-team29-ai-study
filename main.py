from fastapi import FastAPI
from pydantic import BaseModel, Field

from agents.graph import run_recipe_graph
from agents.schemas import AgentState

app = FastAPI(
        title="Recommend API",
        version="1.0.0"
        )

@app.get("/")
def root():
    return {"message": "안녕하세요"}


class RecommendRequest(BaseModel):
    image_path: str = ""
    image_id: str = ""
    user_input_ingredients: list[str] = Field(default_factory=list)
    confidence_threshold: float = 0.7
    user_mood_input: str = ""
    user_situation_input: str = ""
    servings: int = 1


@app.post("/recommend")
def recommend(request: RecommendRequest):
    state: AgentState = request.model_dump()
    result = run_recipe_graph(state)

    return {
        "generated_recipe": result.get("generated_recipe"),
        "generation_status": result.get("generation_status"),
        "generation_message": result.get("generation_message"),
        "route": result.get("route"),
        "route_message": result.get("route_message"),
        "recipe_type": result.get("recipe_type"),
    }
