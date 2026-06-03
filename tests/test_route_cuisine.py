from agents.agent3.service import Agent3Service
from agents.schemas import IngredientInfo, FoodDirections, CuisineRouterOutput

def test_agent3_classify():
    service = Agent3Service()

    ingredient_info = IngredientInfo(
            main_ingredients=["계란", "밥"],
            sub_ingredients=["대파"],
            seasonings=["간장"]
            )

    food_directions = FoodDirections(
            mood="든든한 식사를 하고 싶음",
            situation="",
            fatigue_level="medium",
            difficulty="easy",
            preferred_taste="짭짤한 맛",
            preferred_cooking_method="볶음",
            cooking_time_limit_minutes=20
            )

    result = service.classify(
            ingredient_info=ingredient_info,
            food_directions=food_directions
            )

    assert isinstance(result, CuisineRouterOutput)
    assert result.recipe_type in [
            "korean",
            "chinese",
            "japanese",
            "western"
            ]
    assert isinstance(result.recipe_type_reason, str)
    assert len(result.recipe_type_reason) > 0

    print(result.model_dump())
