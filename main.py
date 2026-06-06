from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.graph import run_recipe_graph
from agents.schemas import (
    AgentState,
    DetectedIngredient,
    IngredientConfirmationInput,
)

from api.agent3 import router as agent3_router

RUNTIME_OUTPUTS_DIR = Path("runtime_outputs")
RUNTIME_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
        title="Recommend API",
        version="1.0.0"
        )

app.mount(
        "/outputs",
        StaticFiles(directory=str(RUNTIME_OUTPUTS_DIR)),
        name="outputs"
        )

app.include_router(
        agent3_router,
        prefix="/agent3",
        tags=["Agent3"]
        )

@app.get("/")
def root():
    return {"message": "안녕하세요"}


class RecommendRequest(BaseModel):
    image_path: str = ""
    image_id: str = ""
    annotation_output_path: str = ""
    user_input_ingredients: list[str] = Field(default_factory=list)
    detected_ingredients: list[DetectedIngredient] = Field(default_factory=list)
    ingredient_confirmation: IngredientConfirmationInput = Field(
        default_factory=IngredientConfirmationInput
    )
    confirmed_ingredients: list[str] = Field(default_factory=list)
    confidence_threshold: float = 0.7
    user_mood_input: str = ""
    user_situation_input: str = ""
    servings: int = 1


def _runtime_output_url(path: str) -> str:
    if not path:
        return ""

    output_path = Path(path)
    try:
        relative_path = output_path.relative_to(RUNTIME_OUTPUTS_DIR)
    except ValueError:
        return ""

    return f"/outputs/{relative_path.as_posix()}"


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
        "detected_ingredients": result.get("detected_ingredients", []),
        "uncertain_ingredients": result.get("uncertain_ingredients", []),
        "available_ingredients": result.get("available_ingredients", []),
        "ingredient_info": result.get("ingredient_info"),
        "confirmation_options": result.get("confirmation_options", []),
        "annotated_image_path": result.get("annotated_image_path", ""),
        "annotated_image_url": _runtime_output_url(
            result.get("annotated_image_path", "")
        ),
        "vision_status": result.get("vision_status"),
        "vision_message": result.get("vision_message"),
        "raw_vision_result": result.get("raw_vision_result", {}),
    }
