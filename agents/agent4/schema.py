from typing import Literal

from pydantic import BaseModel, Field


RecipeType = Literal["korean", "chinese", "japanese", "western"]
RecipeRoute = Literal["can_cook", "no_ingredient", "conflict", "simple"]
IngredientPolicy = Literal["only_available", "allow_additional"]
Difficulty = Literal["easy", "normal", "hard"]


class IngredientInfo(BaseModel):
    """Structured ingredient information produced by agent1."""

    main_ingredients: list[str] = Field(default_factory=list)
    sub_ingredients: list[str] = Field(default_factory=list)
    seasonings: list[str] = Field(default_factory=list)
    carbohydrates: list[str] = Field(default_factory=list)
    proteins: list[str] = Field(default_factory=list)
    fats: list[str] = Field(default_factory=list)
    vegetables: list[str] = Field(default_factory=list)


class FoodDirections(BaseModel):
    """Mood and situation based constraints produced by agent2."""

    mood: str = ""
    situation: str = ""
    fatigue_level: Literal["low", "medium", "high"] = "medium"
    difficulty: Difficulty = "normal"
    preferred_taste: str = ""
    preferred_cooking_method: str = ""
    cooking_time_limit_minutes: int | None = None


class CandidateFood(BaseModel):
    """A recipe candidate agent4 can evaluate before selecting one."""

    name: str = Field(description="Candidate food name")
    recipe_type: RecipeType = Field(description="Cuisine style selected by agent3")
    required_ingredients: list[str] = Field(
        default_factory=list,
        description="Core ingredients required to keep this food's identity",
    )
    optional_ingredients: list[str] = Field(
        default_factory=list,
        description="Ingredients that improve the recipe but can be omitted",
    )
    seasonings: list[str] = Field(
        default_factory=list,
        description="Seasonings normally used for this food",
    )
    substitutions: dict[str, str | None] = Field(
        default_factory=dict,
        description="Allowed replacements. None means the ingredient may be omitted.",
    )
    difficulty: Difficulty = "easy"
    cooking_time_minutes: int = Field(default=15, ge=1)
    taste_profile: list[str] = Field(default_factory=list)
    cooking_methods: list[str] = Field(default_factory=list)
    reason: str = ""


class Substitution(BaseModel):
    """A replacement or omission chosen by agent4."""

    original: str
    replacement: str | None = Field(
        default=None,
        description="Replacement ingredient. None means omit it.",
    )
    reason: str = ""


class SelectedRecipe(BaseModel):
    """The final candidate selected by agent4 for agent5."""

    name: str
    recipe_type: RecipeType
    reason: str = ""


class RecipeCandidateEvaluation(BaseModel):
    """Evaluation result for one candidate food."""

    candidate_name: str
    route: RecipeRoute
    can_pass_to_agent5: bool = False
    missing_required_ingredients: list[str] = Field(default_factory=list)
    missing_optional_ingredients: list[str] = Field(default_factory=list)
    conflict_reasons: list[str] = Field(default_factory=list)
    substitutions: list[Substitution] = Field(default_factory=list)
    score: int = Field(
        default=0,
        description="Higher score means this candidate is preferred.",
    )


class RecipeRouterInput(BaseModel):
    """Inputs required by agent4 from previous agents and the user."""

    available_ingredients: list[str] = Field(default_factory=list)
    ingredient_info: IngredientInfo = Field(default_factory=IngredientInfo)
    food_directions: FoodDirections = Field(default_factory=FoodDirections)
    recipe_type: RecipeType | None = None
    recipe_type_reason: str = ""
    ingredient_policy: IngredientPolicy = "only_available"
    candidate_foods: list[CandidateFood] = Field(
        default_factory=list,
        description="Recipe candidates created by a previous node or external source.",
    )


class RecipeRouterOutput(BaseModel):
    """Outputs agent4 writes to shared state for agent5 and API responses."""

    candidate_foods: list[CandidateFood] = Field(default_factory=list)
    candidate_evaluations: list[RecipeCandidateEvaluation] = Field(default_factory=list)
    route: RecipeRoute = "no_ingredient"
    route_message: str = ""
    selected_recipe: SelectedRecipe | None = None
    ingredients_to_use: list[str] = Field(default_factory=list)
    seasonings_to_use: list[str] = Field(default_factory=list)
    substitutions: list[Substitution] = Field(default_factory=list)
    additional_ingredients: list[str] = Field(default_factory=list)
    can_pass_to_agent5: bool = False


class RecipeRouterState(RecipeRouterInput, RecipeRouterOutput):
    """Standalone agent4 state for direct tests or API usage."""

    pass
