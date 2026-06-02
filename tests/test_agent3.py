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
    assert data["recipe_type"] == "한식"
    assert "recipe_type_reason" in data
    assert data["recipe_type_reason"] != "Error"

def test_japanese_recipe_type():
    payload = {
        "ingredients": {
            "main_ingredients": ["연어", "참치"],
            "sub_ingredients": ["생강", "와사비"],
            "seasonings": ["간장", "미림", "사케"],
            "carbohydrates": ["초밥용 쌀"],
            "proteins": [],
            "fats": [],
            "vegetables": ["오이", "아보카도"]
        },
        "food_directions": {
            "mood": "calm",
            "fatigue_level": "low",
            "difficulty": "medium",
            "preferred_taste": "umami",
            "preferred_cooking_method": "raw"
        }
    }

    response = client.post("/agent3/", json=payload)

    assert response.status_code == 200

    data = response.json()
    print(data)

    assert "recipe_type" in data
    assert data["recipe_type"] == "일식"
    assert "recipe_type_reason" in data
    assert data["recipe_type_reason"] != "Error"

def test_chinese_recipe_type():
    payload = {
        "ingredients": {
            "main_ingredients": ["돼지고기"],
            "sub_ingredients": ["죽순", "목이버섯"],
            "seasonings": ["굴소스", "두반장", "팔각", "화자오"],
            "carbohydrates": ["면"],
            "proteins": [],
            "fats": ["참기름"],
            "vegetables": ["청경채", "대파", "마늘"]
        },
        "food_directions": {
            "mood": "happy",
            "fatigue_level": "medium",
            "difficulty": "medium",
            "preferred_taste": "savory",
            "preferred_cooking_method": "stir_fry"
        }
    }

    response = client.post("/agent3/", json=payload)

    assert response.status_code == 200

    data = response.json()
    print(data)

    assert "recipe_type" in data
    assert data["recipe_type"] == "중식"
    assert "recipe_type_reason" in data
    assert data["recipe_type_reason"] != "Error"

def test_western_recipe_type():
    payload = {
        "ingredients": {
            "main_ingredients": ["소고기"],
            "sub_ingredients": ["로즈마리", "타임", "마늘"],
            "seasonings": ["소금", "후추", "올리브오일"],
            "carbohydrates": ["파스타"],
            "proteins": [],
            "fats": ["버터", "생크림"],
            "vegetables": ["양파", "토마토", "브로콜리"]
        },
        "food_directions": {
            "mood": "romantic",
            "fatigue_level": "low",
            "difficulty": "hard",
            "preferred_taste": "rich",
            "preferred_cooking_method": "bake"
        }
    }

    response = client.post("/agent3/", json=payload)

    assert response.status_code == 200

    data = response.json()
    print(data)

    assert "recipe_type" in data
    assert data["recipe_type"] == "양식"
    assert "recipe_type_reason" in data
    assert data["recipe_type_reason"] != "Error"

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

    assert "recipe_type" in data
    assert data["recipe_type"] is None
    assert "recipe_type_reason" in data
    assert data["recipe_type_reason"] == "Error"

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

    assert "recipe_type" in data
    assert data["recipe_type"] is not None
    assert "recipe_type_reason" in data
    assert data["recipe_type_reason"] != "Error"

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
    assert data["recipe_type_reason"] == "Error"
