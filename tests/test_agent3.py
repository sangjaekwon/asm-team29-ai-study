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
    
    assert response.status_code == 200

    data = response.json()
    print(data)

    assert "recipe_type" in data
    assert data["recipe_type"] is not None
    assert "recipe_type_reason" in data

def test_japanese_recipe_type():
    pass

def test_chinese_recipe_type():
    pass

def test_western_recipe_type():
    pass

def test_no_ingredients():
    payload = {
            "ingredients": {
                "main_ingredients": [],
                "sub_ingredients": [],
                "seasonings": [],
                "carbohydrates": [],
                "proteins": [],
                "fats": [],
                "vegetables": []
                },
            "food_directions": {
                "mood": "많이 배고픔",
                "fatigue_level": "높음",
                "difficulty": "낮음",
                "preferred_taste": "단 맛",
                "preferred_cooking_method": "조림"
                }
            }
    response = client.post("/agent3/", json=payload)
    
    assert response.status_code == 200

    data = response.json()
    print(data)

    assert "recipe_type" not in data

def test_no_food_directions():
    payload = {
            "ingredients": {
                "main_ingredients": ["돼지고기", "소고기"],
                "sub_ingredients": ["표고버섯"],
                "seasonings": ["고추장", "간장", "MSG", "설탕"],
                "carbohydrates": ["쌀"],
                "proteins": ["계란"],
                "fats": ["치즈"],
                "vegetables": ["양파", "당근", "대파", "배추"]
                },
            "food_directions": {
                "mood": "",
                "fatigue_level": "",
                "difficulty": "",
                "preferred_taste": "",
                "preferred_cooking_method": ""
                }
            }
    response = client.post("/agent3/", json=payload)

    assert response.status_code == 200

    data = response.json()
    print(data)

    assert "recipe_type" not in data

def test_no_input():
    payload = {
            "ingredients": {
                "main_ingredients": [],
                "sub_ingredients": [],
                "seasonings": [],
                "carbohydrates": [],
                "proteins": [],
                "fats": [],
                "vegetables": []
                },
            "food_directions": {
                "mood": "",
                "fatigue_level": "",
                "difficulty": "",
                "preferred_taste": "",
                "preferred_cooking_method": ""
                }
            }
    response = client.post("/agent3/", json=payload)
    
    assert response.status_code == 200

    data = response.json()
    print(data)

    assert "recipe_type" in data
    assert data["recipe_type"] is None
    assert "recipe_type_reason" in data
