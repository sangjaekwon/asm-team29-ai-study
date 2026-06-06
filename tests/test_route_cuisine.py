from agents.agent3.service import Agent3Service
from agents.agent3.node import route_cuisine
from agents.schemas import AgentState, IngredientInfo, FoodDirections, CuisineRouterOutput

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
    print(result.model_dump())

    assert isinstance(result, CuisineRouterOutput)
    assert result.recipe_type in [
            "korean",
            "chinese",
            "japanese",
            "western"
            ]
    assert isinstance(result.recipe_type_reason, str)
    assert len(result.recipe_type_reason) > 0

def test_route_cuisine_node():
    state: AgentState = {
        "ingredient_info": IngredientInfo(
            main_ingredients=["계란", "밥"]
        ),
        "food_directions": FoodDirections(
            mood="배고픔",
            difficulty="easy"
        )
    }

    result = route_cuisine(state)
    print(result.model_dump())

    assert result.recipe_type in [
        "korean",
        "chinese",
        "japanese",
        "western"
    ]

    assert len(result.recipe_type_reason) > 0


def test_route_cuisine_without_food_directions():
    """agent2가 빈 입력으로 실행되지 않은 경우 — food_directions가 state에 없음.

    agent3는 ingredient_info만으로 recipe_type을 판단해야 한다.
    """

    state: AgentState = {
        "ingredient_info": IngredientInfo(
            main_ingredients=["스파게티 면", "토마토"],
            seasonings=["올리브오일", "바질"],
        ),
    }

    result = route_cuisine(state)
    print(result.model_dump())

    assert isinstance(result, CuisineRouterOutput)
    assert result.recipe_type in [
        "korean",
        "chinese",
        "japanese",
        "western",
    ]
    assert len(result.recipe_type_reason) > 0


def test_agent3_classify_food_directions_none():
    """service 레이어 직접 검증: food_directions=None 호출도 정상 동작."""

    service = Agent3Service()
    ingredient_info = IngredientInfo(
        main_ingredients=["김치", "두부"],
        seasonings=["고춧가루", "참기름"],
    )

    result = service.classify(
        ingredient_info=ingredient_info,
        food_directions=None,
    )
    print(result.model_dump())

    assert isinstance(result, CuisineRouterOutput)
    assert result.recipe_type in [
        "korean",
        "chinese",
        "japanese",
        "western",
    ]