import os

import pytest

import agents.agent1.service as agent1_service
from agents.agent1.service import analyze_ingredients
from agents.schemas import AgentState, DetectedIngredient


def test_analyze_ingredients_cleans_and_deduplicates_manual_input_without_api_key():
    state: AgentState = {
        "user_input_ingredients": [" 달걀 ", "달걀", "쌀밥", "양파"],
    }

    result = analyze_ingredients(state)

    assert result["vision_status"] == "success"
    assert result["available_ingredients"] == ["달걀", "쌀밥", "양파"]
    assert result["uncertain_ingredients"] == []


def test_analyze_ingredients_builds_detected_ingredients_and_info():
    state: AgentState = {"user_input_ingredients": ["계란", "밥"]}

    result = analyze_ingredients(state)

    detected = result["detected_ingredients"]
    ingredient_info = result["ingredient_info"]

    assert detected[0]["name"] == "계란"
    assert detected[0]["category"] == "sub"
    assert detected[0]["nutrition_type"] == "vegetable"
    assert detected[0]["source"] == "manual"
    assert detected[0]["confidence"] == 1.0

    assert ingredient_info["main_ingredients"] == []
    assert ingredient_info["sub_ingredients"] == ["계란", "밥"]
    assert ingredient_info["seasonings"] == []
    assert ingredient_info["proteins"] == []
    assert ingredient_info["carbohydrates"] == []
    assert ingredient_info["vegetables"] == ["계란", "밥"]
    assert ingredient_info["fats"] == []


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
    assert result["raw_vision_result"]["vision_status"] == "not_connected"


def test_analyze_ingredients_uses_solar_when_api_key_is_available(monkeypatch):
    state: AgentState = {
        "user_input_ingredients": ["egg", "leftover rice", "unclear green vegetable", "soy sauce"],
    }

    def fake_call_solar_ingredient_analyzer(ingredients):
        assert ingredients == [
            "egg",
            "leftover rice",
            "unclear green vegetable",
            "soy sauce",
        ]
        return (
            [
                DetectedIngredient(
                    name="계란",
                    category="main",
                    nutrition_type="protein",
                    confidence=0.98,
                    needs_confirmation=False,
                    source="manual",
                ),
                DetectedIngredient(
                    name="밥",
                    category="main",
                    nutrition_type="carbohydrate",
                    confidence=0.92,
                    needs_confirmation=False,
                    source="manual",
                ),
                DetectedIngredient(
                    name="초록 채소",
                    category="sub",
                    nutrition_type="vegetable",
                    confidence=0.52,
                    needs_confirmation=True,
                    source="manual",
                ),
                DetectedIngredient(
                    name="간장",
                    category="seasoning",
                    nutrition_type="seasoning",
                    confidence=0.9,
                    needs_confirmation=False,
                    source="manual",
                ),
            ],
            {
                "ingredients": [],
                "message": "영어 재료명을 한국어 표준명으로 정리했습니다.",
            },
        )

    monkeypatch.setattr(agent1_service, "_should_call_solar", lambda: True)
    monkeypatch.setattr(
        agent1_service,
        "_call_solar_ingredient_analyzer",
        fake_call_solar_ingredient_analyzer,
    )

    result = analyze_ingredients(state)

    assert result["vision_status"] == "need_user_confirmation"
    assert result["available_ingredients"] == ["계란", "밥", "초록 채소", "간장"]
    assert result["uncertain_ingredients"] == ["초록 채소"]
    assert result["ingredient_info"]["main_ingredients"] == ["계란", "밥"]
    assert result["ingredient_info"]["sub_ingredients"] == ["초록 채소"]
    assert result["ingredient_info"]["seasonings"] == ["간장"]
    assert result["ingredient_info"]["proteins"] == ["계란"]
    assert result["ingredient_info"]["carbohydrates"] == ["밥"]
    assert result["ingredient_info"]["vegetables"] == ["초록 채소"]
    assert result["raw_vision_result"]["analysis_source"] == "solar"


def test_analyze_ingredients_falls_back_to_rules_when_solar_fails(monkeypatch):
    state: AgentState = {
        "user_input_ingredients": ["달걀", "쌀밥"],
    }

    def fake_call_solar_ingredient_analyzer(ingredients):
        raise ValueError("invalid json")

    monkeypatch.setattr(agent1_service, "_should_call_solar", lambda: True)
    monkeypatch.setattr(
        agent1_service,
        "_call_solar_ingredient_analyzer",
        fake_call_solar_ingredient_analyzer,
    )

    result = analyze_ingredients(state)

    assert result["vision_status"] == "success"
    assert result["available_ingredients"] == ["달걀", "쌀밥"]
    assert result["raw_vision_result"]["analysis_source"] == "rules"
    assert "Solar 응답 파싱 실패" in result["raw_vision_result"]["error"]


def test_solar_response_maps_english_detector_labels_to_korean_ingredients(
    monkeypatch,
):
    response_content = """
    {
      "ingredients": [
        {
          "name": "계란",
          "category": "main",
          "nutrition_type": "protein",
          "confidence": 0.94,
          "needs_confirmation": false
        },
        {
          "name": "양파",
          "category": "sub",
          "nutrition_type": "vegetable",
          "confidence": 0.88,
          "needs_confirmation": false
        }
      ],
      "message": "영어 재료명을 한국어 표준명으로 정리했습니다."
    }
    """

    class FakeMessage:
        content = response_content

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "solar-pro3"
            assert kwargs["messages"][1]["content"] == (
                '{"user_input_ingredients": ["egg", "onion"]}'
            )
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            assert api_key == "real-key"
            assert base_url == agent1_service.SOLAR_BASE_URL
            self.chat = FakeChat()

    monkeypatch.setenv("SOLAR_API_KEY", "real-key")
    monkeypatch.setattr(agent1_service, "OpenAI", FakeOpenAI)

    detected_ingredients, llm_result = agent1_service._call_solar_ingredient_analyzer(
        ["egg", "onion"]
    )

    assert [ingredient.name for ingredient in detected_ingredients] == ["계란", "양파"]
    assert detected_ingredients[0].category == "main"
    assert detected_ingredients[0].nutrition_type == "protein"
    assert detected_ingredients[0].confidence == 0.94
    assert detected_ingredients[1].category == "sub"
    assert detected_ingredients[1].nutrition_type == "vegetable"
    assert llm_result["message"] == "영어 재료명을 한국어 표준명으로 정리했습니다."


def test_solar_response_coerces_nutrition_value_in_category_field(monkeypatch):
    response_content = """
    {
      "ingredients": [
        {
          "name": "계란",
          "category": "protein",
          "nutrition_type": "protein",
          "confidence": 0.94,
          "needs_confirmation": false
        }
      ],
      "message": "재료를 정리했습니다."
    }
    """

    class FakeMessage:
        content = response_content

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            self.chat = FakeChat()

    monkeypatch.setenv("SOLAR_API_KEY", "real-key")
    monkeypatch.setattr(agent1_service, "OpenAI", FakeOpenAI)

    detected_ingredients, _ = agent1_service._call_solar_ingredient_analyzer(["egg"])

    assert detected_ingredients[0].name == "계란"
    assert detected_ingredients[0].category == "main"
    assert detected_ingredients[0].nutrition_type == "protein"


@pytest.mark.integration
def test_real_solar_api_maps_english_labels_to_korean_ingredients():
    api_key = os.getenv("SOLAR_API_KEY", "").strip().lower()
    should_run = os.getenv("RUN_SOLAR_INTEGRATION", "") == "1"

    if not should_run or api_key in agent1_service.TEST_API_KEYS:
        pytest.skip("Set RUN_SOLAR_INTEGRATION=1 and a real SOLAR_API_KEY to run.")

    result = analyze_ingredients({"user_input_ingredients": ["egg", "onion"]})

    assert result["raw_vision_result"]["analysis_source"] == "solar"
    assert result["vision_status"] == "success"
    assert result["available_ingredients"] == ["계란", "양파"]
    assert result["ingredient_info"]["main_ingredients"] == ["계란"]
    assert result["ingredient_info"]["sub_ingredients"] == ["양파"]
    assert result["ingredient_info"]["seasonings"] == []
    assert result["ingredient_info"]["proteins"] == ["계란"]
    assert result["ingredient_info"]["vegetables"] == ["양파"]
