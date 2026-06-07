import base64
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv(encoding="utf-8-sig")

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "asm-team29-recipe-demo")

from agents.graph import run_recipe_graph
from agents.schemas import (
    AgentState,
    DetectedIngredient,
    IngredientConfirmationInput,
)
from agents.workflow_trace import latest_trace, record_backend_step, start_trace

RUNTIME_OUTPUTS_DIR = Path("runtime_outputs")
UPLOAD_OUTPUTS_DIR = RUNTIME_OUTPUTS_DIR / "uploads"
FRONTEND_DIR = Path("frontend")
RUNTIME_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
        title="Recommend API",
        version="1.0.0"
        )

app.mount(
        "/outputs",
        StaticFiles(directory=str(RUNTIME_OUTPUTS_DIR)),
        name="outputs"
        )

if FRONTEND_DIR.exists():
    app.mount(
            "/frontend",
            StaticFiles(directory=str(FRONTEND_DIR)),
            name="frontend"
            )

@app.get("/")
def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    return {"message": "안녕하세요"}


@app.get("/debug/workflow/latest")
def debug_workflow_latest():
    return latest_trace()


class RecommendRequest(BaseModel):
    image_path: str = ""
    image_id: str = ""
    image_base64: str = ""
    image_filename: str = ""
    annotation_output_path: str = ""
    user_input_ingredients: list[str] = Field(default_factory=list)
    detected_ingredients: list[DetectedIngredient] = Field(default_factory=list)
    ingredient_confirmation: IngredientConfirmationInput = Field(
        default_factory=IngredientConfirmationInput
    )
    confirmed_ingredients: list[str] = Field(default_factory=list)
    confidence_threshold: float = 0.7
    ingredient_policy: str = "only_available"
    candidate_foods: list[dict] = Field(default_factory=list)
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


def _save_base64_image(request: RecommendRequest) -> str:
    if not request.image_base64:
        return ""

    encoded_image = request.image_base64
    if "," in encoded_image:
        encoded_image = encoded_image.split(",", 1)[1]

    image_bytes = base64.b64decode(encoded_image, validate=True)
    suffix = Path(request.image_filename or "upload.png").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".png"

    image_path = UPLOAD_OUTPUTS_DIR / f"{uuid.uuid4().hex}{suffix}"
    image_path.write_bytes(image_bytes)
    return str(image_path)


def _display_image_url(result: AgentState, state: AgentState) -> str:
    annotated_url = _runtime_output_url(result.get("annotated_image_path", ""))
    if annotated_url:
        return annotated_url

    return _runtime_output_url(state.get("image_path", ""))


@app.post("/recommend")
def recommend(request: RecommendRequest):
    trace_id = uuid.uuid4().hex
    start_trace(trace_id)
    record_backend_step("POST /recommend", "running")
    record_backend_step("FastAPI Request", "running")

    state: AgentState = request.model_dump(
        exclude={
            "image_base64",
            "image_filename",
        }
    )
    record_backend_step("FastAPI Request", "completed")

    record_backend_step("Save Upload Image", "running")
    uploaded_image_path = _save_base64_image(request)
    record_backend_step("Save Upload Image", "completed")

    record_backend_step("Build AgentState", "running")
    if uploaded_image_path:
        state["image_path"] = uploaded_image_path
        state["image_id"] = state.get("image_id") or Path(uploaded_image_path).stem
    record_backend_step("Build AgentState", "completed")

    record_backend_step("POST /recommend", "completed")
    record_backend_step("run_recipe_graph()", "running")
    result = run_recipe_graph(state)
    record_backend_step("run_recipe_graph()", "completed")

    record_backend_step("Format API Response", "running")
    response_payload = {
        "workflow_trace_id": trace_id,
        "generated_recipe": result.get("generated_recipe"),
        "generation_status": result.get("generation_status"),
        "generation_message": result.get("generation_message"),
        "recipe_generation_source": result.get("recipe_generation_source"),
        "recipe_generation_error": result.get("recipe_generation_error"),
        "route": result.get("route"),
        "route_message": result.get("route_message"),
        "recipe_type": result.get("recipe_type"),
        "recipe_type_reason": result.get("recipe_type_reason"),
        "candidate_foods": result.get("candidate_foods", []),
        "candidate_evaluations": result.get("candidate_evaluations", []),
        "selected_recipe": result.get("selected_recipe"),
        "ingredients_to_use": result.get("ingredients_to_use", []),
        "seasonings_to_use": result.get("seasonings_to_use", []),
        "substitutions": result.get("substitutions", []),
        "additional_ingredients": result.get("additional_ingredients", []),
        "detected_ingredients": result.get("detected_ingredients", []),
        "uncertain_ingredients": result.get("uncertain_ingredients", []),
        "available_ingredients": result.get("available_ingredients", []),
        "ingredient_info": result.get("ingredient_info"),
        "confirmation_options": result.get("confirmation_options", []),
        "annotated_image_path": result.get("annotated_image_path", ""),
        "annotated_image_url": _display_image_url(result, state),
        "vision_status": result.get("vision_status"),
        "vision_message": result.get("vision_message"),
        "raw_vision_result": result.get("raw_vision_result", {}),
    }
    record_backend_step("Format API Response", "completed")
    record_backend_step("API Response", "completed")
    return response_payload
