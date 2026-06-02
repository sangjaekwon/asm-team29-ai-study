from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_korean_recipe_type():
    payload = {
            "ingredients": {
                "main_ingredients": ["돼지고기"],
                "sub_ingredients": ["양파", "대파"],
                "seasonings": ["고추장", "간장"],
                "carbohydrates": [],
                "proteins": [],
                "fats": [],
                "vegetables": []
                },
            "food_directions": {
                "mood": "stress",
                "fatigue_level": "high",
                "difficulty": "easy",
                "preferred_taste": "spicy",
                "preferred_cooking_method": "stir_fry"
                }
            }

    response = client.post("/agent3/", json=payload)
    
    print(response.status_code)
    print(response.json())
    assert response.status_code == 200

    data = response.json()

    assert "recipe_type" in data
    assert "recipe_type_reason" in data
