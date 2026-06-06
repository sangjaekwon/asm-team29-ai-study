"""레시피 추천 멀티 에이전트 공통 스키마.

현재 프로젝트 초기 단계에서는 이 파일만 작성한다.
각 에이전트 구현체는 이후 작성하며, State에 값을 저장할 때는
반드시 이 파일에 정의된 필드명을 사용한다.
"""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


# ============================================================
# 공통 Literal 타입
# ============================================================

IngredientCategory = Literal["main", "sub", "seasoning"]
NutritionType = Literal[
    "carbohydrate",
    "protein",
    "fat",
    "vegetable",
    "seasoning",
]
IngredientSource = Literal["vision", "manual", "user_confirmed"]
VisionStatus = Literal[
    "success",
    "need_user_confirmation",
    "no_ingredient_detected",
    "vision_error",
]
FatigueLevel = Literal["low", "medium", "high"]
Difficulty = Literal["easy", "normal", "hard"]
RecipeType = Literal["korean", "chinese", "japanese", "western"]
RouteType = Literal[
    "can_cook",
    "simple_recipe",
    "missing_ingredient",
    "constraint_conflict",
]
GenerationStatus = Literal["success", "failed"]


# ============================================================
# 1. 재료 파악 에이전트 스키마
# ============================================================

class DetectedIngredient(BaseModel):
    """사진 분석 또는 사용자 입력으로 확인된 재료 한 개."""

    name: str = Field(description="표준화된 식재료 이름")
    category: IngredientCategory = Field(description="요리 관점의 재료 분류")
    nutrition_type: NutritionType = Field(description="영양 성격 분류")
    boundary_box: list[int] = Field(
        default_factory=list,
        description="이미지 내 바운딩 박스 [x1, y1, x2, y2]",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="재료 인식 신뢰도",
    )
    needs_confirmation: bool = Field(
        default=False,
        description="사용자 확인 필요 여부",
    )
    source: IngredientSource = Field(
        default="vision",
        description="재료 정보 출처",
    )


class IngredientInfo(BaseModel):
    """다음 에이전트가 공통으로 사용하는 구조화된 재료 정보."""

    main_ingredients: list[str] = Field(default_factory=list)
    sub_ingredients: list[str] = Field(default_factory=list)
    seasonings: list[str] = Field(default_factory=list)
    carbohydrates: list[str] = Field(default_factory=list)
    proteins: list[str] = Field(default_factory=list)
    fats: list[str] = Field(default_factory=list)
    vegetables: list[str] = Field(default_factory=list)


class IngredientConfirmationOption(BaseModel):
    """사용자 확인이 필요한 재료에 대해 프론트가 보여줄 선택지."""

    name: str = Field(description="현재 Agent1이 판단한 재료명")
    boundary_box: list[int] = Field(
        default_factory=list,
        description="이미지 내 바운딩 박스 [x1, y1, x2, y2]",
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = Field(default="", description="사용자 확인이 필요한 이유")
    candidates: list[str] = Field(default_factory=list)
    allow_manual_input: bool = True


class IngredientConfirmationInput(BaseModel):
    """사용자가 재료 인식 결과를 확정/수정하기 위해 보내는 입력."""

    accepted_ingredients: list[str] = Field(default_factory=list)
    rejected_ingredients: list[str] = Field(default_factory=list)
    replacements: dict[str, str] = Field(default_factory=dict)
    additional_ingredients: list[str] = Field(default_factory=list)
    additional_ingredients_text: str = ""


class IngredientAnalyzerOutput(BaseModel):
    """재료 파악 에이전트가 State에 기록하는 출력."""

    detected_ingredients: list[DetectedIngredient] = Field(default_factory=list)
    uncertain_ingredients: list[str] = Field(default_factory=list)
    available_ingredients: list[str] = Field(default_factory=list)
    ingredient_info: IngredientInfo = Field(default_factory=IngredientInfo)
    confirmation_options: list[IngredientConfirmationOption] = Field(default_factory=list)
    annotated_image_path: str = ""
    vision_status: VisionStatus = "success"
    vision_message: str = ""
    raw_vision_result: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# 2. 기분·상황 추천 에이전트 스키마
# ============================================================

class FoodDirections(BaseModel):
    """사용자의 기분과 상황을 반영한 음식 추천 방향."""

    mood: str = ""
    situation: str = ""
    fatigue_level: FatigueLevel = "medium"
    difficulty: Difficulty = "normal"
    preferred_taste: str = ""
    preferred_cooking_method: str = ""
    cooking_time_limit_minutes: int | None = None


class ContextAnalyzerOutput(BaseModel):
    """기분·상황 추천 에이전트가 State에 기록하는 출력."""

    food_directions: FoodDirections = Field(default_factory=FoodDirections)


# ============================================================
# 3. 요리 스타일 분기 에이전트 스키마
# ============================================================

class CuisineRouterOutput(BaseModel):
    """요리 스타일 분기 에이전트가 State에 기록하는 출력."""

    recipe_type: RecipeType | None = None
    recipe_type_reason: str = ""


# ============================================================
# 4. 가능 레시피 라우터 에이전트 스키마
# ============================================================

class CandidateFood(BaseModel):
    """가능 레시피 라우터가 비교할 후보 음식."""

    name: str
    recipe_type: RecipeType
    required_ingredients: list[str] = Field(default_factory=list)
    optional_ingredients: list[str] = Field(default_factory=list)
    reason: str = ""


class SelectedRecipe(BaseModel):
    """가능 레시피 라우터가 최종 선택한 음식."""

    name: str
    recipe_type: RecipeType
    reason: str = ""


class Substitution(BaseModel):
    """대체하거나 생략할 수 있는 재료 정보."""

    original: str
    replacement: str
    reason: str = ""


class RecipeRouterOutput(BaseModel):
    """가능 레시피 라우터가 State에 기록하는 출력."""

    candidate_foods: list[CandidateFood] = Field(default_factory=list)
    route: RouteType = "can_cook"
    route_message: str = ""
    selected_recipe: SelectedRecipe | None = None
    ingredients_to_use: list[str] = Field(default_factory=list)
    seasonings_to_use: list[str] = Field(default_factory=list)
    substitutions: list[Substitution] = Field(default_factory=list)
    additional_ingredients: list[str] = Field(default_factory=list)


# ============================================================
# 5. 레시피 생성 에이전트 스키마
# ============================================================

class GeneratedRecipe(BaseModel):
    """사용자에게 제공할 최종 레시피."""

    recipe_name: str
    ingredients: list[str] = Field(default_factory=list)
    cooking_steps: list[str] = Field(default_factory=list)
    cooking_time_minutes: int
    difficulty: Difficulty = "normal"
    servings: int = 1
    cooking_tips: list[str] = Field(default_factory=list)
    substitutions: list[Substitution] = Field(default_factory=list)
    additional_ingredients: list[str] = Field(default_factory=list)


class RecipeGeneratorOutput(BaseModel):
    """레시피 생성 에이전트가 State에 기록하는 출력."""

    generated_recipe: GeneratedRecipe | None = None
    generation_status: GenerationStatus = "success"
    generation_message: str = ""


# ============================================================
# LangGraph 전체 공유 State
# ============================================================

class AgentState(TypedDict, total=False):
    """다섯 개 에이전트가 공유하는 State.

    각 노드는 자신이 담당하는 필드만 반환한다.
    서로 연결되는 출력과 입력은 반드시 아래 필드명을 그대로 사용한다.
    """

    # 사용자 최초 입력
    image_path: str
    image_id: str
    annotation_output_path: str
    user_input_ingredients: list[str]
    ingredient_confirmation: IngredientConfirmationInput | dict[str, Any]
    confirmed_ingredients: list[str]
    confidence_threshold: float
    user_mood_input: str
    user_situation_input: str
    servings: int

    # 1. 재료 파악 에이전트 출력
    detected_ingredients: list[DetectedIngredient]
    uncertain_ingredients: list[str]
    available_ingredients: list[str]
    ingredient_info: IngredientInfo
    confirmation_options: list[IngredientConfirmationOption]
    annotated_image_path: str
    vision_status: VisionStatus
    vision_message: str
    raw_vision_result: dict[str, Any]

    # 2. 기분·상황 추천 에이전트 출력
    food_directions: FoodDirections

    # 3. 요리 스타일 분기 에이전트 출력
    recipe_type: RecipeType | None
    recipe_type_reason: str

    # 4. 가능 레시피 라우터 에이전트 출력
    candidate_foods: list[CandidateFood]
    route: RouteType
    route_message: str
    selected_recipe: SelectedRecipe | None
    ingredients_to_use: list[str]
    seasonings_to_use: list[str]
    substitutions: list[Substitution]
    additional_ingredients: list[str]

    # 5. 레시피 생성 에이전트 출력
    generated_recipe: GeneratedRecipe | None
    generation_status: GenerationStatus
    generation_message: str
