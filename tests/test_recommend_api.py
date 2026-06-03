from fastapi.testclient import TestClient

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
