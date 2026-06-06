import json
import os
from pathlib import Path

import pytest

import agents.agent1.service as agent1_service
from agents.agent1.ingredient_aliases import standardize_ingredient_name
from agents.agent1.service import analyze_ingredients
from agents.agent1.service import ImageDetection
from agents.schemas import AgentState, DetectedIngredient


def test_analyze_ingredients_cleans_and_deduplicates_manual_input_without_api_key():
    state: AgentState = {
        "user_input_ingredients": [" 달걀 ", "달걀", "쌀밥", "양파"],
    }

    result = analyze_ingredients(state)

    assert result["vision_status"] == "success"
    assert result["available_ingredients"] == ["계란", "밥", "양파"]
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


def test_analyze_ingredients_returns_vision_error_when_image_detection_fails():
    state: AgentState = {
        "image_path": "/tmp/not-found.jpg",
        "image_id": "image-404",
        "user_input_ingredients": [],
    }

    result = analyze_ingredients(state)

    assert result["vision_status"] == "vision_error"
    assert result["available_ingredients"] == []
    assert "이미지 재료 탐지 실패" in result["vision_message"]
    assert result["raw_vision_result"]["analysis_source"] == "detector"
    assert result["raw_vision_result"]["vision_status"] == "vision_error"


def test_suppress_duplicate_detections_keeps_highest_confidence_box():
    detections = [
        ImageDetection(
            label="tomato",
            boundary_box=[10, 10, 100, 100],
            confidence=0.8,
        ),
        ImageDetection(
            label="tomato",
            boundary_box=[20, 20, 110, 110],
            confidence=0.7,
        ),
        ImageDetection(
            label="beef",
            boundary_box=[20, 20, 110, 110],
            confidence=0.65,
        ),
    ]

    result = agent1_service._suppress_duplicate_detections(detections)

    assert [detection.label for detection in result] == ["tomato", "beef"]
    assert result[0].confidence == 0.8


def test_standard_ingredient_alias_table_maps_common_detector_labels():
    assert standardize_ingredient_name("oyster mushroom") == "느타리버섯"
    assert standardize_ingredient_name("king oyster mushroom") == "새송이버섯"
    assert standardize_ingredient_name("enoki mushroom") == "팽이버섯"
    assert standardize_ingredient_name("bok choy") == "청경채"
    assert standardize_ingredient_name("green onion") == "대파"
    assert standardize_ingredient_name("굴버섯") == "느타리버섯"
    assert standardize_ingredient_name("계란후라이") == "계란"


def test_detection_metadata_matches_unique_labels_after_solar_dedup(monkeypatch):
    state: AgentState = {
        "image_path": "/tmp/fridge.jpg",
        "confidence_threshold": 0.7,
        "user_input_ingredients": [],
    }

    def fake_detect_ingredients_from_image(image_path):
        return [
            ImageDetection(
                label="egg",
                boundary_box=[10, 20, 110, 120],
                confidence=0.91,
            ),
            ImageDetection(
                label="egg",
                boundary_box=[15, 25, 115, 125],
                confidence=0.72,
            ),
            ImageDetection(
                label="apple",
                boundary_box=[130, 20, 220, 120],
                confidence=0.81,
            ),
        ]

    def fake_call_solar_ingredient_analyzer(ingredients):
        assert ingredients == ["계란", "사과"]
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
                    name="사과",
                    category="sub",
                    nutrition_type="vegetable",
                    confidence=0.88,
                    needs_confirmation=False,
                    source="manual",
                ),
            ],
            {"ingredients": [], "message": "이미지 재료를 정리했습니다."},
        )

    monkeypatch.setattr(
        agent1_service,
        "detect_ingredients_from_image",
        fake_detect_ingredients_from_image,
    )
    monkeypatch.setattr(agent1_service, "_should_call_solar", lambda: True)
    monkeypatch.setattr(
        agent1_service,
        "_call_solar_ingredient_analyzer",
        fake_call_solar_ingredient_analyzer,
    )

    result = analyze_ingredients(state)

    assert result["available_ingredients"] == ["계란", "사과"]
    assert result["detected_ingredients"][0]["boundary_box"] == [10, 20, 110, 120]
    assert result["detected_ingredients"][1]["boundary_box"] == [130, 20, 220, 120]
    assert [item["label"] for item in result["raw_vision_result"]["detections"]] == ["계란", "사과"]


def test_analyze_ingredients_maps_detector_result_to_solar_output(monkeypatch):
    state: AgentState = {
        "image_path": "/tmp/fridge.jpg",
        "image_id": "image-1",
        "confidence_threshold": 0.7,
        "user_input_ingredients": ["soy sauce"],
    }

    def fake_detect_ingredients_from_image(image_path):
        assert image_path == "/tmp/fridge.jpg"
        return [
            ImageDetection(
                label="egg",
                boundary_box=[10, 20, 110, 120],
                confidence=0.93,
            ),
            ImageDetection(
                label="apple",
                boundary_box=[130, 20, 220, 120],
                confidence=0.41,
            ),
        ]

    def fake_call_solar_ingredient_analyzer(ingredients):
        assert ingredients == ["계란", "사과", "간장"]
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
                    name="사과",
                    category="sub",
                    nutrition_type="vegetable",
                    confidence=0.88,
                    needs_confirmation=False,
                    source="manual",
                ),
                DetectedIngredient(
                    name="간장",
                    category="seasoning",
                    nutrition_type="seasoning",
                    confidence=0.95,
                    needs_confirmation=False,
                    source="manual",
                ),
            ],
            {"ingredients": [], "message": "이미지 재료를 정리했습니다."},
        )

    monkeypatch.setattr(
        agent1_service,
        "detect_ingredients_from_image",
        fake_detect_ingredients_from_image,
    )
    monkeypatch.setattr(agent1_service, "_should_call_solar", lambda: True)
    monkeypatch.setattr(
        agent1_service,
        "_call_solar_ingredient_analyzer",
        fake_call_solar_ingredient_analyzer,
    )

    result = analyze_ingredients(state)

    assert result["vision_status"] == "need_user_confirmation"
    assert result["available_ingredients"] == ["계란", "사과", "간장"]
    assert result["uncertain_ingredients"] == ["사과"]
    assert result["ingredient_info"]["main_ingredients"] == ["계란"]
    assert result["ingredient_info"]["sub_ingredients"] == ["사과"]
    assert result["ingredient_info"]["seasonings"] == ["간장"]

    detected = result["detected_ingredients"]
    assert detected[0]["source"] == "vision"
    assert detected[0]["boundary_box"] == [10, 20, 110, 120]
    assert detected[0]["confidence"] == 0.93
    assert detected[0]["needs_confirmation"] is False
    assert detected[1]["source"] == "vision"
    assert detected[1]["boundary_box"] == [130, 20, 220, 120]
    assert detected[1]["confidence"] == 0.41
    assert detected[1]["needs_confirmation"] is True
    assert detected[2]["source"] == "manual"

    raw = result["raw_vision_result"]
    assert raw["analysis_source"] == "solar"
    assert raw["vision_status"] == "success"
    assert raw["detections"][0]["label"] == "계란"


def test_analyze_ingredients_uses_solar_when_api_key_is_available(monkeypatch):
    state: AgentState = {
        "user_input_ingredients": ["egg", "leftover rice", "unclear green vegetable", "soy sauce"],
    }

    def fake_call_solar_ingredient_analyzer(ingredients):
        assert ingredients == [
            "계란",
            "밥",
            "채소",
            "간장",
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
                    name="채소",
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
    assert result["available_ingredients"] == ["계란", "밥", "채소", "간장"]
    assert result["uncertain_ingredients"] == ["채소"]
    assert result["ingredient_info"]["main_ingredients"] == ["계란", "밥"]
    assert result["ingredient_info"]["sub_ingredients"] == ["채소"]
    assert result["ingredient_info"]["seasonings"] == ["간장"]
    assert result["ingredient_info"]["proteins"] == ["계란"]
    assert result["ingredient_info"]["carbohydrates"] == ["밥"]
    assert result["ingredient_info"]["vegetables"] == ["채소"]
    assert result["raw_vision_result"]["analysis_source"] == "solar"


def test_standardizes_common_detector_labels_before_calling_solar(monkeypatch):
    state: AgentState = {
        "image_path": "/tmp/fridge.jpg",
        "confidence_threshold": 0.4,
        "user_input_ingredients": [],
    }

    def fake_detect_ingredients_from_image(image_path):
        return [
            ImageDetection(
                label="oyster mushroom",
                boundary_box=[10, 20, 110, 120],
                confidence=0.72,
            ),
            ImageDetection(
                label="enoki mushroom",
                boundary_box=[130, 20, 220, 120],
                confidence=0.68,
            ),
            ImageDetection(
                label="green onion",
                boundary_box=[230, 20, 320, 120],
                confidence=0.61,
            ),
            ImageDetection(
                label="vegetable",
                boundary_box=[330, 20, 420, 120],
                confidence=0.8,
            ),
        ]

    def fake_call_solar_ingredient_analyzer(ingredients):
        assert ingredients == ["채소", "느타리버섯", "팽이버섯", "대파"]
        return (
            [
                DetectedIngredient(
                    name="채소",
                    category="sub",
                    nutrition_type="vegetable",
                    confidence=0.9,
                    needs_confirmation=False,
                    source="manual",
                ),
                DetectedIngredient(
                    name="굴버섯",
                    category="sub",
                    nutrition_type="vegetable",
                    confidence=0.9,
                    needs_confirmation=False,
                    source="manual",
                ),
                DetectedIngredient(
                    name="팽이버섯",
                    category="sub",
                    nutrition_type="vegetable",
                    confidence=0.9,
                    needs_confirmation=False,
                    source="manual",
                ),
                DetectedIngredient(
                    name="파",
                    category="sub",
                    nutrition_type="vegetable",
                    confidence=0.9,
                    needs_confirmation=False,
                    source="manual",
                ),
            ],
            {"ingredients": [], "message": "표준명으로 정리했습니다."},
        )

    monkeypatch.setattr(
        agent1_service,
        "detect_ingredients_from_image",
        fake_detect_ingredients_from_image,
    )
    monkeypatch.setattr(agent1_service, "_should_call_solar", lambda: True)
    monkeypatch.setattr(
        agent1_service,
        "_call_solar_ingredient_analyzer",
        fake_call_solar_ingredient_analyzer,
    )

    result = analyze_ingredients(state)

    assert result["available_ingredients"] == ["채소", "느타리버섯", "팽이버섯", "대파"]
    assert result["ingredient_info"]["sub_ingredients"] == [
        "채소",
        "느타리버섯",
        "팽이버섯",
        "대파",
    ]
    assert result["uncertain_ingredients"] == ["채소"]


def test_deduplicates_detector_results_by_standard_ingredient_name(monkeypatch):
    state: AgentState = {
        "image_path": "/tmp/fridge.jpg",
        "confidence_threshold": 0.4,
        "user_input_ingredients": [],
    }

    def fake_detect_ingredients_from_image(image_path):
        return [
            ImageDetection(
                label="green onion",
                boundary_box=[10, 20, 110, 120],
                confidence=0.71,
            ),
            ImageDetection(
                label="scallion",
                boundary_box=[12, 22, 112, 122],
                confidence=0.88,
            ),
        ]

    def fake_call_solar_ingredient_analyzer(ingredients):
        assert ingredients == ["대파"]
        return (
            [
                DetectedIngredient(
                    name="파",
                    category="sub",
                    nutrition_type="vegetable",
                    confidence=0.9,
                    needs_confirmation=False,
                    source="manual",
                ),
            ],
            {"ingredients": [], "message": "표준명으로 정리했습니다."},
        )

    monkeypatch.setattr(
        agent1_service,
        "detect_ingredients_from_image",
        fake_detect_ingredients_from_image,
    )
    monkeypatch.setattr(agent1_service, "_should_call_solar", lambda: True)
    monkeypatch.setattr(
        agent1_service,
        "_call_solar_ingredient_analyzer",
        fake_call_solar_ingredient_analyzer,
    )

    result = analyze_ingredients(state)

    assert result["available_ingredients"] == ["대파"]
    assert result["detected_ingredients"][0]["boundary_box"] == [12, 22, 112, 122]
    assert result["raw_vision_result"]["detections"][0]["label"] == "대파"
    assert result["raw_vision_result"]["detections"][0]["original_label"] == "scallion"
    assert result["raw_vision_result"]["detections"][0]["boundary_box"] == [12, 22, 112, 122]
    assert result["raw_vision_result"]["detections"][0]["confidence"] == 0.88


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
    assert result["available_ingredients"] == ["계란", "밥"]
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


@pytest.mark.integration
def test_real_image_detection_flow_builds_agent3_ingredient_info():
    api_key = os.getenv("SOLAR_API_KEY", "").strip().lower()
    should_run = os.getenv("RUN_IMAGE_DETECTION_INTEGRATION", "") == "1"
    image_path = "요리재료사진.jpg"

    if not should_run or api_key in agent1_service.TEST_API_KEYS:
        pytest.skip("Set RUN_IMAGE_DETECTION_INTEGRATION=1 and a real SOLAR_API_KEY.")
    if not os.path.exists(image_path):
        pytest.skip("요리재료사진.jpg is required for this integration test.")

    result = analyze_ingredients(
        {
            "image_path": image_path,
            "image_id": "local-food-poc",
            "confidence_threshold": 0.4,
            "user_input_ingredients": [],
        }
    )

    assert result["raw_vision_result"]["analysis_source"] == "solar"
    assert result["raw_vision_result"]["detections"]
    assert result["detected_ingredients"]
    assert result["available_ingredients"]
    assert (
        result["ingredient_info"]["main_ingredients"]
        or result["ingredient_info"]["sub_ingredients"]
        or result["ingredient_info"]["seasonings"]
    )

    first_detected = result["detected_ingredients"][0]
    assert first_detected["source"] == "vision"
    assert len(first_detected["boundary_box"]) == 4
    assert 0.0 <= first_detected["confidence"] <= 1.0


def _load_annotation_font(size: int):
    from PIL import ImageFont

    font_candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)

    return ImageFont.load_default()


def _draw_agent1_e2e_result(
    image_path: Path,
    result: dict,
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_annotation_font(size=22)

    for ingredient in result["detected_ingredients"]:
        box = ingredient.get("boundary_box") or []
        if len(box) != 4:
            continue

        x1, y1, x2, y2 = box
        color = "red" if ingredient.get("needs_confirmation") else "lime"
        detector_label = ""
        for detection in result["raw_vision_result"].get("detections", []):
            if detection.get("boundary_box") == box:
                detector_label = detection.get("original_label") or detection.get("label", "")
                break

        caption = (
            f"{ingredient['name']} | detector:{detector_label} | "
            f"{ingredient['confidence']:.2f}"
        )
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)

        text_box = draw.textbbox((x1, max(0, y1 - 30)), caption, font=font)
        draw.rectangle(text_box, fill="black")
        draw.text((x1, max(0, y1 - 30)), caption, fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


@pytest.mark.integration
def test_real_jeyuk_image_detection_and_solar_output_for_agent3():
    api_key = os.getenv("SOLAR_API_KEY", "").strip().lower()
    should_run = os.getenv("RUN_IMAGE_DETECTION_INTEGRATION", "") == "1"
    image_path = Path("tests/fixtures/agent1/images/jeyuk.png")
    output_dir = Path("test_outputs/agent1")
    result_json_path = output_dir / "jeyuk_agent1_e2e_result.json"
    annotated_image_path = output_dir / "jeyuk_agent1_e2e_result.png"

    if not should_run or api_key in agent1_service.TEST_API_KEYS:
        pytest.skip("Set RUN_IMAGE_DETECTION_INTEGRATION=1 and a real SOLAR_API_KEY.")
    if not image_path.exists():
        pytest.skip("tests/fixtures/agent1/images/jeyuk.png is required.")

    result = analyze_ingredients(
        {
            "image_path": str(image_path),
            "image_id": "jeyuk-local-e2e",
            "confidence_threshold": 0.4,
            "user_input_ingredients": [],
        }
    )

    detector_output = result["raw_vision_result"].get("detections", [])
    agent3_input = result["ingredient_info"]
    report = {
        "image_path": str(image_path),
        "detector_output": detector_output,
        "solar_output": result["raw_vision_result"].get("llm_result", {}),
        "solar_detected_ingredients": result["detected_ingredients"],
        "solar_all_ingredients": result["available_ingredients"],
        "agent3_input": agent3_input,
        "vision_status": result["vision_status"],
        "vision_message": result["vision_message"],
        "annotated_image_path": str(annotated_image_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result_json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _draw_agent1_e2e_result(image_path, result, annotated_image_path)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    assert result["raw_vision_result"]["analysis_source"] == "solar"
    assert detector_output
    assert result["detected_ingredients"]
    assert result["available_ingredients"]
    assert any(len(item["boundary_box"]) == 4 for item in result["detected_ingredients"])
    assert (
        agent3_input["main_ingredients"]
        or agent3_input["sub_ingredients"]
        or agent3_input["seasonings"]
    )
    assert result_json_path.exists()
    assert annotated_image_path.exists()
