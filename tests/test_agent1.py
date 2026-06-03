from agents.agent1.service import analyze_ingredients
from agents.schemas import AgentState


def test_analyze_ingredients_normalizes_and_deduplicates_manual_input():
    state: AgentState = {
        "user_input_ingredients": [" 달걀 ", "계란", "쌀밥", "양파", "간장"],
    }

    result = analyze_ingredients(state)

    assert result["vision_status"] == "success"
    assert result["available_ingredients"] == ["계란", "밥", "양파", "간장"]
    assert result["uncertain_ingredients"] == []


def test_analyze_ingredients_builds_detected_ingredients_and_info():
    state: AgentState = {
        "user_input_ingredients": ["계란", "밥", "양파", "간장", "버터"],
    }

    result = analyze_ingredients(state)

    detected = result["detected_ingredients"]
    ingredient_info = result["ingredient_info"]

    assert detected[0]["name"] == "계란"
    assert detected[0]["category"] == "main"
    assert detected[0]["nutrition_type"] == "protein"
    assert detected[0]["source"] == "manual"
    assert detected[0]["confidence"] == 1.0

    assert ingredient_info["main_ingredients"] == ["계란", "밥"]
    assert ingredient_info["sub_ingredients"] == ["양파", "버터"]
    assert ingredient_info["seasonings"] == ["간장"]
    assert ingredient_info["proteins"] == ["계란"]
    assert ingredient_info["carbohydrates"] == ["밥"]
    assert ingredient_info["vegetables"] == ["양파"]
    assert ingredient_info["fats"] == ["버터"]


def test_analyze_ingredients_handles_empty_input():
    result = analyze_ingredients({"user_input_ingredients": []})

    assert result["vision_status"] == "no_ingredient_detected"
    assert result["available_ingredients"] == []
    assert result["detected_ingredients"] == []
    assert result["ingredient_info"]["main_ingredients"] == []


def test_analyze_ingredients_keeps_image_metadata_until_vision_is_connected():
    state: AgentState = {
        "image_path": "/tmp/fridge.jpg",
        "image_id": "image-1",
        "user_input_ingredients": ["두부"],
    }

    result = analyze_ingredients(state)

    assert result["raw_vision_result"]["image_path"] == "/tmp/fridge.jpg"
    assert result["raw_vision_result"]["image_id"] == "image-1"
    assert result["raw_vision_result"]["status"] == "not_connected"
