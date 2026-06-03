from pydantic import BaseModel
from typing import List, Optional

class Ingredients(BaseModel):
    main_ingredients: List[str]
    sub_ingredients: List[str]
    seasonings: List[str]
    carbohydrates: List[str]
    proteins: List[str]
    fats: List[str]
    vegetables: List[str]

class FoodDirections(BaseModel):
    mood: str
    fatigue_level: str
    difficulty: str
    preferred_taste: str
    preferred_cooking_method: str

class Agent3Request(BaseModel):
    ingredients: Ingredients
    food_directions: FoodDirections

class Agent3Response(BaseModel):
    recipe_type: Optional[str]
    recipe_type_reason: str

