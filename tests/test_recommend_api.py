import os

import pytest
from fastapi.testclient import TestClient

import agents.agent1.service as agent1_service
from main import app


def test_recommend_endpoint_returns_generated_recipe():
    client = TestClient(app)

    response = client.post(
        "/recommend",
        json={
            "user_input_ingredients": ["계란", "밥"],
            "user_mood_input": "피곤해",
            "user_situation_input": "빠른 저녁",
            "servings": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["generation_status"] == "success"
    assert body["generated_recipe"]["recipe_name"] == "간단한 계란 밥"
    assert body["generated_recipe"]["servings"] == 2
    assert body["route"] == "can_cook"
    assert body["recipe_type"] == "korean"


@pytest.mark.integration
def test_recommend_endpoint_confirms_agent1_ingredients_with_real_solar():
    api_key = os.getenv("SOLAR_API_KEY", "").strip().lower()
    should_run = os.getenv("RUN_SOLAR_INTEGRATION", "") == "1"

    if not should_run or api_key in agent1_service.TEST_API_KEYS:
        pytest.skip("Set RUN_SOLAR_INTEGRATION=1 and a real SOLAR_API_KEY to run.")

    client = TestClient(app)
    response = client.post(
        "/recommend",
        json={
            "detected_ingredients": [
                {
                    "name": "돼지고기",
                    "category": "main",
                    "nutrition_type": "protein",
                    "boundary_box": [10, 10, 200, 200],
                    "confidence": 0.48,
                    "needs_confirmation": False,
                    "source": "vision",
                },
                {
                    "name": "채소",
                    "category": "sub",
                    "nutrition_type": "vegetable",
                    "boundary_box": [210, 10, 300, 200],
                    "confidence": 0.59,
                    "needs_confirmation": True,
                    "source": "vision",
                },
                {
                    "name": "대파",
                    "category": "sub",
                    "nutrition_type": "vegetable",
                    "boundary_box": [310, 10, 400, 200],
                    "confidence": 0.29,
                    "needs_confirmation": True,
                    "source": "vision",
                },
            ],
            "ingredient_confirmation": {
                "rejected_ingredients": ["채소"],
                "replacements": {"대파": "쪽파"},
                "additional_ingredients_text": "간마늘, 설탕",
            },
            "user_mood_input": "피곤해",
            "user_situation_input": "빠른 저녁",
            "servings": 2,
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["vision_status"] == "success"
    assert body["confirmation_options"] == []
    assert body["uncertain_ingredients"] == []
    assert "채소" not in body["available_ingredients"]
    assert "돼지고기" in body["available_ingredients"]
    assert "쪽파" in body["available_ingredients"]
    assert any(
        ingredient in body["available_ingredients"]
        for ingredient in ["다진마늘", "마늘"]
    )
    assert "설탕" in body["available_ingredients"]
    assert body["ingredient_info"]["main_ingredients"]
    assert body["generation_status"] == "success"
    assert body["route"] == "can_cook"
