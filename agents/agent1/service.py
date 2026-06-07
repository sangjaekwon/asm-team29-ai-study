"""1번 에이전트의 재료 파악 서비스.

사용자가 직접 입력한 재료 목록을 Solar API로 표준화하고 분류한다.
API 키가 없거나 LLM 응답을 사용할 수 없는 경우 규칙 기반 fallback을
사용한다. 이미지 기반 Vision 모델 연동은 후속 작업에서 확장한다.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from agents.agent1.ingredient_aliases import (
    DEFAULT_DETECTOR_LABELS,
    GENERIC_DETECTION_LABELS,
    build_standard_name_prompt_rules,
    standardize_ingredient_name,
    suggest_confirmation_candidates,
)
from agents.schemas import (
    AgentState,
    DetectedIngredient,
    IngredientConfirmationInput,
    IngredientConfirmationOption,
    IngredientAnalyzerOutput,
    IngredientCategory,
    IngredientInfo,
    IngredientSource,
    NutritionType,
)


load_dotenv(encoding="utf-8-sig")

SOLAR_BASE_URL = (
    os.getenv("SOLAR_BASE_URL")
    or os.getenv("UPSTAGE_BASE_URL")
    or "https://api.upstage.ai/v1"
)
DEFAULT_SOLAR_MODEL = "solar-pro3"
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_DETECTOR_BACKEND = "owlv2"
DEFAULT_DETECTOR_MODEL = "google/owlv2-base-patch16-ensemble"
DEFAULT_DETECTOR_THRESHOLD = 0.25
DEFAULT_ANNOTATION_OUTPUT_DIR = "runtime_outputs/agent1"
TEST_API_KEYS = {"", "test", "dummy", "your_upstage_api_key_here"}
VALID_CATEGORIES = {"main", "sub", "seasoning"}
VALID_NUTRITION_TYPES = {
    "carbohydrate",
    "protein",
    "fat",
    "vegetable",
    "seasoning",
}

SYSTEM_PROMPT_TEMPLATE = """당신은 AI 요리 도우미의 재료 파악 에이전트입니다.
사용자가 입력한 한국어/영어 식재료 표현을 한국어 표준 식재료 이름으로 정리하고 분류하세요.

반드시 JSON만 반환하세요. 마크다운 코드블록이나 설명 문장은 쓰지 마세요.
반환 형식:
{
  "ingredients": [
    {
      "name": "한국어로 표준화된 식재료 이름",
      "category": "main | sub | seasoning",
      "nutrition_type": "carbohydrate | protein | fat | vegetable | seasoning",
      "confidence": 0.0부터 1.0 사이 숫자,
      "needs_confirmation": true 또는 false
    }
  ],
  "message": "사용자에게 보여줄 짧은 안내 문장"
}

분류 기준:
- category가 seasoning이면 nutrition_type도 seasoning으로 설정합니다.
- egg, onion처럼 영어 재료명이 들어오면 계란, 양파처럼 한국어 표준명으로 변환합니다.
- 밥, 면, 빵, 감자류는 carbohydrate입니다.
- 계란, 육류, 생선, 두부, 해산물은 protein입니다.
- 버터, 치즈, 오일류는 fat입니다.
- 채소, 버섯, 김치, 허브류는 vegetable입니다.
- 애매한 재료는 confidence를 낮추고 needs_confirmation을 true로 설정합니다.

표준명 매핑 규칙:
- 아래 표준명 매핑 규칙을 우선 적용합니다.
- oyster mushroom은 굴버섯/새송이버섯이 아니라 반드시 느타리버섯입니다.
- king oyster mushroom은 새송이버섯입니다.
- enoki mushroom은 팽이버섯입니다.
- bok choy는 청경채입니다.
- green onion, scallion, green leek은 대파입니다.
{standard_name_prompt_rules}
"""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.replace(
    "{standard_name_prompt_rules}",
    build_standard_name_prompt_rules(),
)


class ImageDetection(BaseModel):
    """이미지 detector가 반환하는 재료 후보."""

    label: str
    original_label: str = ""
    boundary_box: list[int] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


def _get_solar_api_key() -> str:
    return (os.getenv("SOLAR_API_KEY") or os.getenv("UPSTAGE_API_KEY") or "").strip()


def _get_solar_model() -> str:
    return (
        os.getenv("SOLAR_MODEL")
        or os.getenv("UPSTAGE_MODEL")
        or DEFAULT_SOLAR_MODEL
    ).strip()


def _get_detector_backend() -> str:
    return os.getenv("AGENT1_DETECTOR_BACKEND", DEFAULT_DETECTOR_BACKEND).strip().lower()


def _get_detector_model() -> str:
    return os.getenv("AGENT1_DETECTOR_MODEL", DEFAULT_DETECTOR_MODEL).strip()


def _get_detector_threshold() -> float:
    threshold = os.getenv("AGENT1_DETECTOR_THRESHOLD", str(DEFAULT_DETECTOR_THRESHOLD))
    return min(max(float(threshold), 0.0), 1.0)


def _get_detector_labels() -> list[str]:
    labels = os.getenv("AGENT1_DETECTOR_LABELS", "").strip()
    if not labels:
        return DEFAULT_DETECTOR_LABELS

    return [label.strip() for label in labels.split(",") if label.strip()]


def _get_annotation_output_dir() -> str:
    output_dir = os.getenv(
        "AGENT1_ANNOTATION_OUTPUT_DIR",
        DEFAULT_ANNOTATION_OUTPUT_DIR,
    ).strip()
    return output_dir or DEFAULT_ANNOTATION_OUTPUT_DIR


def _should_call_solar() -> bool:
    return _get_solar_api_key().lower() not in TEST_API_KEYS


def _clean_name(name: str) -> str:
    return " ".join(name.strip().split())


def _standardize_alias(name: str) -> str:
    return standardize_ingredient_name(name)


def _unique_ingredients(ingredients: list[str]) -> list[str]:
    unique: list[str] = []
    for ingredient in ingredients:
        cleaned = _clean_name(ingredient)
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    return unique


def _parse_ingredient_text(text: str) -> list[str]:
    if not text.strip():
        return []

    if re.search(r"[,;\n]", text):
        tokens = re.split(r"[,;\n]+", text)
    else:
        tokens = text.split()

    return [_clean_name(token) for token in tokens if _clean_name(token)]


def _get_confidence_threshold(state: AgentState) -> float:
    threshold = state.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    return min(max(float(threshold), 0.0), 1.0)


def _build_detected_ingredient(
    name: str,
    source: IngredientSource = "manual",
) -> DetectedIngredient:
    return DetectedIngredient(
        name=name,
        category="sub",
        nutrition_type="vegetable",
        confidence=1.0,
        needs_confirmation=False,
        source=source,
    )


def _coerce_nutrition_type(item: dict[str, Any]) -> NutritionType:
    nutrition_type = str(item.get("nutrition_type", "")).strip().lower()
    category = str(item.get("category", "")).strip().lower()

    if nutrition_type in VALID_NUTRITION_TYPES:
        return nutrition_type  # type: ignore[return-value]
    if category in VALID_NUTRITION_TYPES:
        return category  # type: ignore[return-value]
    return "vegetable"


def _coerce_category(item: dict[str, Any], nutrition_type: NutritionType) -> IngredientCategory:
    category = str(item.get("category", "")).strip().lower()

    if category in VALID_CATEGORIES:
        return category  # type: ignore[return-value]
    if nutrition_type == "seasoning":
        return "seasoning"
    if nutrition_type in {"carbohydrate", "protein"}:
        return "main"
    return "sub"


def _deduplicate_detected_ingredients(
    detected_ingredients: list[DetectedIngredient],
) -> list[DetectedIngredient]:
    unique: list[DetectedIngredient] = []
    seen: set[str] = set()

    for ingredient in detected_ingredients:
        cleaned_name = _standardize_alias(ingredient.name)
        if not cleaned_name or cleaned_name in seen:
            continue

        seen.add(cleaned_name)
        unique.append(ingredient.model_copy(update={"name": cleaned_name}))

    return unique


def _coerce_detected_ingredients(items: Any) -> list[DetectedIngredient]:
    detected: list[DetectedIngredient] = []

    for item in items or []:
        try:
            detected.append(DetectedIngredient.model_validate(item))
        except ValidationError:
            continue

    return detected


def _build_ingredient_info(detected_ingredients: list[DetectedIngredient]) -> IngredientInfo:
    info = IngredientInfo()

    for ingredient in detected_ingredients:
        if ingredient.category == "main":
            info.main_ingredients.append(ingredient.name)
        elif ingredient.category == "sub":
            info.sub_ingredients.append(ingredient.name)
        else:
            info.seasonings.append(ingredient.name)

        if ingredient.nutrition_type == "carbohydrate":
            info.carbohydrates.append(ingredient.name)
        elif ingredient.nutrition_type == "protein":
            info.proteins.append(ingredient.name)
        elif ingredient.nutrition_type == "fat":
            info.fats.append(ingredient.name)
        elif ingredient.nutrition_type == "vegetable":
            info.vegetables.append(ingredient.name)

    return info


def _build_raw_result(
    state: AgentState,
    analysis_source: str,
    llm_result: dict[str, Any] | None = None,
    detections: list[ImageDetection] | None = None,
    annotated_image_path: str = "",
    vision_status: str = "not_connected",
    error: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "image_path": state.get("image_path", ""),
        "image_id": state.get("image_id", ""),
        "analysis_source": analysis_source,
        "annotated_image_path": annotated_image_path,
        "vision_status": vision_status,
        "vision_message": (
            "이미지 분석을 수행했습니다."
            if vision_status != "not_connected"
            else "이미지 분석은 아직 연결되지 않았습니다."
        ),
    }

    if llm_result is not None:
        result["llm_result"] = llm_result
    if detections is not None:
        result["detections"] = [detection.model_dump() for detection in detections]
    if error:
        result["error"] = error

    return result


def _get_annotation_output_path(state: AgentState) -> str:
    output_path = state.get("annotation_output_path", "")
    if output_path:
        return output_path

    image_path = state.get("image_path", "")
    if not image_path:
        return ""

    image_stem = Path(image_path).stem or "agent1"
    return str(Path(_get_annotation_output_dir()) / f"{image_stem}_agent1_annotated.png")


def _load_annotation_font(size: int):
    from PIL import ImageFont

    font_candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for font_path in font_candidates:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)

    return ImageFont.load_default()


def _matched_detector_label(
    ingredient: DetectedIngredient,
    detections: list[ImageDetection],
) -> str:
    for detection in detections:
        if detection.boundary_box == ingredient.boundary_box:
            return detection.original_label or detection.label

    return ""


def _annotation_caption(
    ingredient: DetectedIngredient,
    detections: list[ImageDetection],
) -> str:
    confidence = min(max(ingredient.confidence, 0.0), 1.0)
    confidence_percent = int(round(confidence * 100))
    return f"{ingredient.name} {confidence_percent}%"


def _rects_overlap(first: list[int], second: list[int]) -> bool:
    return not (
        first[2] <= second[0]
        or first[0] >= second[2]
        or first[3] <= second[1]
        or first[1] >= second[3]
    )


def _place_annotation_label(
    box: list[int],
    text_width: int,
    text_height: int,
    image_width: int,
    image_height: int,
    padding: int,
    occupied_rects: list[list[int]],
) -> list[int]:
    x1, y1, x2, y2 = box
    label_width = text_width + (padding * 2)
    label_height = text_height + (padding * 2)
    candidate_points = [
        (x1, y1 - label_height - padding),
        (x1, y2 + padding),
        (x2 - label_width, y1 - label_height - padding),
        (x2 - label_width, y2 + padding),
        (x1, y1 + padding),
        (x2 + padding, y1),
    ]

    for point_x, point_y in candidate_points:
        label_x = max(0, min(point_x, image_width - label_width))
        label_y = max(0, min(point_y, image_height - label_height))
        rect = [
            int(label_x),
            int(label_y),
            int(label_x + label_width),
            int(label_y + label_height),
        ]
        if not any(_rects_overlap(rect, occupied) for occupied in occupied_rects):
            return rect

    step_x = max(padding * 2, label_width // 3)
    step_y = max(padding * 2, label_height // 2)
    x_positions = {
        0,
        max(0, image_width - label_width),
        max(0, min(x1, image_width - label_width)),
        max(0, min(x2 - label_width, image_width - label_width)),
        max(0, min((x1 + x2 - label_width) // 2, image_width - label_width)),
    }
    x_positions.update(range(0, max(1, image_width - label_width + 1), step_x))

    for label_y in range(0, max(1, image_height - label_height + 1), step_y):
        for label_x in sorted(x_positions):
            rect = [
                int(label_x),
                int(label_y),
                int(label_x + label_width),
                int(label_y + label_height),
            ]
            if not any(_rects_overlap(rect, occupied) for occupied in occupied_rects):
                return rect

    label_x = max(0, min(x1, image_width - label_width))
    label_y = max(0, min(y1 - label_height - padding, image_height - label_height))
    return [
        int(label_x),
        int(label_y),
        int(label_x + label_width),
        int(label_y + label_height),
    ]


def _create_annotated_image(
    state: AgentState,
    detected_ingredients: list[DetectedIngredient],
    detections: list[ImageDetection],
) -> str:
    image_path = state.get("image_path", "")
    output_path = _get_annotation_output_path(state)

    if not image_path or not output_path or not os.path.exists(image_path):
        return ""
    if not any(ingredient.boundary_box for ingredient in detected_ingredients):
        return ""

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return ""

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font_size = max(20, min(34, image.width // 34))
    font = _load_annotation_font(size=font_size)
    stroke_width = max(3, image.width // 360)
    text_stroke_width = max(1, font_size // 18)
    padding = max(6, font_size // 5)
    occupied_rects: list[list[int]] = []

    for ingredient in detected_ingredients:
        box = ingredient.boundary_box
        if len(box) != 4:
            continue

        x1, y1, x2, y2 = box
        color = "#d92d20" if ingredient.needs_confirmation else "#12b76a"
        label_background = "#9b1c1c" if ingredient.needs_confirmation else "#176c43"
        caption = _annotation_caption(ingredient, detections)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke_width)

        text_box = draw.multiline_textbbox((0, 0), caption, font=font, spacing=2)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        background = _place_annotation_label(
            box,
            text_width,
            text_height,
            image.width,
            image.height,
            padding,
            occupied_rects,
        )
        draw.rectangle(background, fill=label_background)
        draw.multiline_text(
            (background[0] + padding, background[1] + padding),
            caption,
            fill="white",
            font=font,
            spacing=2,
            stroke_width=text_stroke_width,
            stroke_fill="black",
        )
        occupied_rects.append(background)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _confirmation_reason(
    ingredient: DetectedIngredient,
    detection: ImageDetection | None,
    threshold: float,
) -> str:
    reasons: list[str] = []
    label_keys = {ingredient.name.strip().lower()}

    if detection is not None:
        label_keys.add(detection.label.strip().lower())
        label_keys.add(detection.original_label.strip().lower())
        if detection.confidence < threshold:
            reasons.append("detector confidence가 기준값보다 낮습니다.")
        if label_keys & GENERIC_DETECTION_LABELS:
            reasons.append("detector가 너무 일반적인 라벨로 판단했습니다.")

    if ingredient.needs_confirmation:
        reasons.append("LLM 또는 이전 단계에서 사용자 확인이 필요하다고 판단했습니다.")

    return " ".join(reasons) or "사용자 확인이 필요합니다."


def _build_confirmation_options(
    detected_ingredients: list[DetectedIngredient],
    detections: list[ImageDetection],
    threshold: float,
) -> list[IngredientConfirmationOption]:
    options: list[IngredientConfirmationOption] = []

    for index, ingredient in enumerate(detected_ingredients):
        if not ingredient.needs_confirmation:
            continue

        detection = detections[index] if index < len(detections) else None
        options.append(
            IngredientConfirmationOption(
                name=ingredient.name,
                boundary_box=ingredient.boundary_box,
                confidence=ingredient.confidence,
                reason=_confirmation_reason(ingredient, detection, threshold),
                candidates=suggest_confirmation_candidates(
                    ingredient.name,
                    detection.original_label if detection is not None else "",
                ),
                allow_manual_input=True,
            )
        )

    return options


def detect_ingredients_from_image(image_path: str) -> list[ImageDetection]:
    """이미지에서 식재료 후보를 탐지한다.

    기본 detector는 OWLv2 zero-shot detector이다. `AGENT1_DETECTOR_BACKEND=yolo`
    와 YOLO 모델 경로를 지정하면 YOLO 계열 detector도 사용할 수 있다.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    backend = _get_detector_backend()
    if backend == "owlv2":
        return _detect_with_owlv2(image_path)
    if backend == "yolo":
        return _detect_with_yolo(image_path)

    raise RuntimeError(f"지원하지 않는 detector backend입니다: {backend}")


def _detect_with_yolo(image_path: str) -> list[ImageDetection]:
    model_path = _get_detector_model()
    if not model_path:
        raise RuntimeError("YOLO 모델 경로가 설정되지 않았습니다.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics 패키지가 설치되어 있지 않습니다.") from exc

    model = YOLO(model_path)
    results = model(image_path)
    detections: list[ImageDetection] = []

    if not results:
        return detections

    result = results[0]
    names = getattr(result, "names", {})
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls[0].item())
        label = str(names.get(class_id, class_id))
        confidence = float(box.conf[0].item())
        coordinates = [int(value) for value in box.xyxy[0].tolist()]
        detections.append(
            ImageDetection(
                label=label,
                original_label=label,
                boundary_box=coordinates,
                confidence=confidence,
            )
        )

    return detections


def _detect_with_owlv2(image_path: str) -> list[ImageDetection]:
    try:
        import torch
        from PIL import Image
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
    except ImportError as exc:
        raise RuntimeError(
            "OWLv2 detector를 사용하려면 torch, pillow, transformers가 필요합니다."
        ) from exc

    model_id = _get_detector_model()
    labels = _get_detector_labels()
    threshold = _get_detector_threshold()

    image = Image.open(image_path).convert("RGB")
    processor = Owlv2Processor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id)

    inputs = processor(text=[labels], images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)])
    result = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold,
        text_labels=[labels],
    )[0]

    detections = [
        ImageDetection(
            label=str(label),
            original_label=str(label),
            boundary_box=[int(value) for value in box.tolist()],
            confidence=float(score),
        )
        for score, label, box in zip(
            result["scores"],
            result["text_labels"],
            result["boxes"],
        )
    ]

    return _suppress_duplicate_detections(detections)


def _box_iou(first: list[int], second: list[int]) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second

    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)
    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    union_area = first_area + second_area - intersection_area

    return intersection_area / union_area if union_area else 0.0


def _suppress_duplicate_detections(
    detections: list[ImageDetection],
    iou_threshold: float = 0.25,
    max_detections: int = 30,
) -> list[ImageDetection]:
    selected: list[ImageDetection] = []

    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        is_duplicate = any(
            detection.label == selected_detection.label
            and _box_iou(detection.boundary_box, selected_detection.boundary_box)
            > iou_threshold
            for selected_detection in selected
        )
        if is_duplicate:
            continue

        selected.append(detection)
        if len(selected) >= max_detections:
            break

    return selected


def _deduplicate_detections_by_label(
    detections: list[ImageDetection],
) -> list[ImageDetection]:
    unique: list[ImageDetection] = []
    seen: set[str] = set()

    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        label = _standardize_alias(detection.label)
        if not label or label in seen:
            continue

        seen.add(label)
        unique.append(
            detection.model_copy(
                update={
                    "label": label,
                    "original_label": detection.original_label or detection.label,
                }
            )
        )

    return unique


def _detect_image_ingredients(state: AgentState) -> list[ImageDetection]:
    image_path = state.get("image_path", "")
    if not image_path:
        return []

    return _deduplicate_detections_by_label(detect_ingredients_from_image(image_path))


def _ingredient_inputs_from_state(
    state: AgentState,
    detections: list[ImageDetection],
) -> list[str]:
    detection_labels = [_standardize_alias(detection.label) for detection in detections]
    manual_inputs = [
        _standardize_alias(ingredient)
        for ingredient in state.get("user_input_ingredients", [])
    ]
    return _unique_ingredients(detection_labels + manual_inputs)


def _apply_detection_metadata(
    detected_ingredients: list[DetectedIngredient],
    detections: list[ImageDetection],
    threshold: float,
) -> list[DetectedIngredient]:
    updated: list[DetectedIngredient] = []

    for index, ingredient in enumerate(detected_ingredients):
        if index >= len(detections):
            updated.append(ingredient.model_copy(update={"source": "manual"}))
            continue

        detection = detections[index]
        label_keys = {
            detection.label.strip().lower(),
            detection.original_label.strip().lower(),
        }
        is_generic_label = bool(label_keys & GENERIC_DETECTION_LABELS)
        needs_confirmation = (
            ingredient.needs_confirmation
            or detection.confidence < threshold
            or is_generic_label
        )
        updated.append(
            ingredient.model_copy(
                update={
                    "boundary_box": detection.boundary_box,
                    "confidence": detection.confidence,
                    "needs_confirmation": needs_confirmation,
                    "source": "vision",
                }
            )
        )

    return updated


def _extract_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    return json.loads(cleaned)


def _build_solar_messages(ingredients: list[str]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                {"user_input_ingredients": ingredients},
                ensure_ascii=False,
            ),
        },
    ]


def _call_solar_ingredient_analyzer(
    ingredients: list[str],
) -> tuple[list[DetectedIngredient], dict[str, Any]]:
    client = OpenAI(
        api_key=_get_solar_api_key(),
        base_url=SOLAR_BASE_URL,
    )

    response = client.chat.completions.create(
        model=_get_solar_model(),
        messages=_build_solar_messages(ingredients),
        temperature=0,
        stream=False,
    )
    content = response.choices[0].message.content or "{}"
    parsed = _extract_json(content)

    detected_ingredients: list[DetectedIngredient] = []
    for item in parsed.get("ingredients", []):
        if not item.get("name"):
            continue

        nutrition_type = _coerce_nutrition_type(item)
        detected_ingredients.append(
            DetectedIngredient(
                name=item["name"],
                category=_coerce_category(item, nutrition_type),
                nutrition_type=nutrition_type,
                confidence=float(item.get("confidence", 1.0)),
                needs_confirmation=bool(item.get("needs_confirmation", False)),
                source="manual",
            )
        )

    return _deduplicate_detected_ingredients(detected_ingredients), parsed


def _coerce_confirmation_input(state: AgentState) -> IngredientConfirmationInput | None:
    raw_confirmation = state.get("ingredient_confirmation")

    if raw_confirmation:
        try:
            parsed = IngredientConfirmationInput.model_validate(raw_confirmation)
            has_data = (
                parsed.accepted_ingredients
                or parsed.rejected_ingredients
                or parsed.replacements
                or parsed.additional_ingredients
                or parsed.additional_ingredients_text.strip()
            )
            if has_data:
                return parsed
        except ValidationError:
            return None

    confirmed_ingredients = state.get("confirmed_ingredients", [])
    if confirmed_ingredients:
        return IngredientConfirmationInput(accepted_ingredients=confirmed_ingredients)

    return None


def _has_confirmation_input(state: AgentState) -> bool:
    confirmation = _coerce_confirmation_input(state)
    if confirmation is None:
        return False

    return bool(
        confirmation.accepted_ingredients
        or confirmation.rejected_ingredients
        or confirmation.replacements
        or confirmation.additional_ingredients
        or confirmation.additional_ingredients_text.strip()
    )


def _ingredient_metadata_by_name(
    detected_ingredients: list[DetectedIngredient],
) -> dict[str, DetectedIngredient]:
    metadata: dict[str, DetectedIngredient] = {}

    for ingredient in detected_ingredients:
        metadata[_standardize_alias(ingredient.name)] = ingredient

    return metadata


def _confirmed_names_from_state(state: AgentState) -> list[str]:
    confirmation = _coerce_confirmation_input(state)
    if confirmation is None:
        return []

    if state.get("confirmed_ingredients"):
        base_names = [
            _standardize_alias(name)
            for name in state.get("confirmed_ingredients", [])
        ]
    else:
        previous_ingredients = _coerce_detected_ingredients(
            state.get("detected_ingredients", [])
        )
        rejected = {
            _standardize_alias(name)
            for name in confirmation.rejected_ingredients
        }
        replacements = {
            _standardize_alias(source): target
            for source, target in confirmation.replacements.items()
            if _clean_name(source) and _clean_name(target)
        }
        base_names = []

        for ingredient in previous_ingredients:
            original_name = _standardize_alias(ingredient.name)
            if original_name in rejected:
                continue

            base_names.append(_standardize_alias(replacements.get(original_name, original_name)))

        for accepted in confirmation.accepted_ingredients:
            base_names.append(_standardize_alias(accepted))

    additions = [
        _standardize_alias(name)
        for name in confirmation.additional_ingredients
    ]
    additions.extend(
        _standardize_alias(name)
        for name in _parse_ingredient_text(confirmation.additional_ingredients_text)
    )

    return _unique_ingredients(base_names + additions)


def _build_user_confirmed_output(
    state: AgentState,
) -> dict[str, Any]:
    confirmed_names = _confirmed_names_from_state(state)
    previous_ingredients = _coerce_detected_ingredients(
        state.get("detected_ingredients", [])
    )
    previous_metadata = _ingredient_metadata_by_name(previous_ingredients)
    confirmation = _coerce_confirmation_input(state)
    if confirmation is not None:
        for source, target in confirmation.replacements.items():
            source_name = _standardize_alias(source)
            target_name = _standardize_alias(target)
            if source_name in previous_metadata and target_name not in previous_metadata:
                previous_metadata[target_name] = previous_metadata[source_name]

    if not confirmed_names:
        return IngredientAnalyzerOutput(
            detected_ingredients=[],
            uncertain_ingredients=[],
            available_ingredients=[],
            ingredient_info=IngredientInfo(),
            confirmation_options=[],
            vision_status="no_ingredient_detected",
            vision_message="사용자가 확정한 재료가 없습니다.",
            raw_vision_result=_build_raw_result(
                state,
                analysis_source="user_confirmation",
                vision_status="success",
            ),
        ).model_dump()

    try:
        if _should_call_solar():
            detected_ingredients, llm_result = _call_solar_ingredient_analyzer(
                confirmed_names
            )
        else:
            detected_ingredients = [
                _build_detected_ingredient(name, source="user_confirmed")
                for name in confirmed_names
            ]
            llm_result = {
                "ingredients": confirmed_names,
                "message": "사용자 확인 재료를 규칙 기반으로 확정했습니다.",
            }
    except (KeyError, TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        detected_ingredients = [
            _build_detected_ingredient(name, source="user_confirmed")
            for name in confirmed_names
        ]
        llm_result = {
            "ingredients": confirmed_names,
            "message": "Solar 응답 파싱 실패로 규칙 기반 확정 결과를 사용했습니다.",
            "error": str(exc),
        }
    except Exception as exc:
        detected_ingredients = [
            _build_detected_ingredient(name, source="user_confirmed")
            for name in confirmed_names
        ]
        llm_result = {
            "ingredients": confirmed_names,
            "message": "Solar 호출 실패로 규칙 기반 확정 결과를 사용했습니다.",
            "error": str(exc),
        }

    if not detected_ingredients:
        detected_ingredients = [
            _build_detected_ingredient(name, source="user_confirmed")
            for name in confirmed_names
        ]
        llm_result = {
            "ingredients": confirmed_names,
            "message": "Solar returned no confirmed ingredients, used rule-based confirmation.",
        }

    confirmed_detected: list[DetectedIngredient] = []
    for ingredient in _deduplicate_detected_ingredients(detected_ingredients):
        metadata = previous_metadata.get(_standardize_alias(ingredient.name))
        update: dict[str, Any] = {
            "source": "user_confirmed",
            "confidence": 1.0,
            "needs_confirmation": False,
        }
        if metadata is not None:
            update["boundary_box"] = metadata.boundary_box

        confirmed_detected.append(ingredient.model_copy(update=update))

    annotated_image_path = _create_annotated_image(state, confirmed_detected, [])

    return IngredientAnalyzerOutput(
        detected_ingredients=confirmed_detected,
        uncertain_ingredients=[],
        available_ingredients=[ingredient.name for ingredient in confirmed_detected],
        ingredient_info=_build_ingredient_info(confirmed_detected),
        confirmation_options=[],
        annotated_image_path=annotated_image_path,
        vision_status="success",
        vision_message="사용자가 확인한 재료를 최종 확정했습니다.",
        raw_vision_result=_build_raw_result(
            state,
            analysis_source="user_confirmation",
            llm_result=llm_result,
            annotated_image_path=annotated_image_path,
            vision_status="success",
        )
        | {
            "ingredient_confirmation": _coerce_confirmation_input(state).model_dump()
            if _coerce_confirmation_input(state)
            else {},
            "previous_detected_ingredients": [
                ingredient.model_dump() for ingredient in previous_ingredients
            ],
        },
    ).model_dump()


def _analyze_with_rules(
    state: AgentState,
    detections: list[ImageDetection] | None = None,
    error: str = "",
) -> dict[str, Any]:
    detections = detections or []
    ingredients = _ingredient_inputs_from_state(state, detections)

    if not ingredients:
        return IngredientAnalyzerOutput(
            detected_ingredients=[],
            uncertain_ingredients=[],
            available_ingredients=[],
            ingredient_info=IngredientInfo(),
            confirmation_options=[],
            vision_status="no_ingredient_detected",
            vision_message="사용 가능한 재료가 입력되지 않았습니다.",
            raw_vision_result=_build_raw_result(
                state,
                analysis_source="detector" if state.get("image_path") else "rules",
                detections=detections,
                vision_status="success" if state.get("image_path") else "not_connected",
                error=error,
            ),
        ).model_dump()

    detected_ingredients = [
        _build_detected_ingredient(ingredient) for ingredient in ingredients
    ]
    detected_ingredients = _apply_detection_metadata(
        detected_ingredients,
        detections,
        _get_confidence_threshold(state),
    )
    uncertain_ingredients = [
        ingredient.name
        for ingredient in detected_ingredients
        if ingredient.needs_confirmation
    ]
    vision_status = "need_user_confirmation" if uncertain_ingredients else "success"
    confirmation_options = _build_confirmation_options(
        detected_ingredients,
        detections,
        _get_confidence_threshold(state),
    )
    annotated_image_path = _create_annotated_image(
        state,
        detected_ingredients,
        detections,
    )

    return IngredientAnalyzerOutput(
        detected_ingredients=detected_ingredients,
        uncertain_ingredients=uncertain_ingredients,
        available_ingredients=ingredients,
        ingredient_info=_build_ingredient_info(detected_ingredients),
        confirmation_options=confirmation_options,
        annotated_image_path=annotated_image_path,
        vision_status=vision_status,
        vision_message=(
            "인식 결과 확인이 필요합니다. 누락된 재료가 있으면 직접 추가하세요."
            if confirmation_options
            else "사용자 입력 재료를 규칙 기반으로 표준화하고 분류했습니다."
        ),
        raw_vision_result=_build_raw_result(
            state,
            analysis_source="detector" if detections else "rules",
            detections=detections,
            annotated_image_path=annotated_image_path,
            vision_status="success" if detections else "not_connected",
            error=error,
        ),
    ).model_dump()


def _build_llm_output(
    state: AgentState,
    detected_ingredients: list[DetectedIngredient],
    llm_result: dict[str, Any],
    detections: list[ImageDetection] | None = None,
) -> dict[str, Any]:
    detections = detections or []
    detected_ingredients = _deduplicate_detected_ingredients(detected_ingredients)
    detected_ingredients = _apply_detection_metadata(
        detected_ingredients,
        detections,
        _get_confidence_threshold(state),
    )

    if not detected_ingredients:
        return IngredientAnalyzerOutput(
            detected_ingredients=[],
            uncertain_ingredients=[],
            available_ingredients=[],
            ingredient_info=IngredientInfo(),
            confirmation_options=[],
            vision_status="no_ingredient_detected",
            vision_message="Solar API가 사용 가능한 재료를 찾지 못했습니다.",
            raw_vision_result=_build_raw_result(
                state,
                analysis_source="solar",
                llm_result=llm_result,
                detections=detections,
                vision_status="success" if detections else "not_connected",
            ),
        ).model_dump()

    uncertain_ingredients = [
        ingredient.name
        for ingredient in detected_ingredients
        if ingredient.needs_confirmation
    ]

    vision_status = "need_user_confirmation" if uncertain_ingredients else "success"
    vision_message = llm_result.get("message") or "Solar API로 재료를 표준화하고 분류했습니다."
    confirmation_options = _build_confirmation_options(
        detected_ingredients,
        detections,
        _get_confidence_threshold(state),
    )
    annotated_image_path = _create_annotated_image(
        state,
        detected_ingredients,
        detections,
    )

    return IngredientAnalyzerOutput(
        detected_ingredients=detected_ingredients,
        uncertain_ingredients=uncertain_ingredients,
        available_ingredients=[ingredient.name for ingredient in detected_ingredients],
        ingredient_info=_build_ingredient_info(detected_ingredients),
        confirmation_options=confirmation_options,
        annotated_image_path=annotated_image_path,
        vision_status=vision_status,
        vision_message=(
            f"{vision_message} 누락된 재료가 있으면 직접 추가하세요."
            if confirmation_options
            else vision_message
        ),
        raw_vision_result=_build_raw_result(
            state,
            analysis_source="solar",
            llm_result=llm_result,
            detections=detections,
            annotated_image_path=annotated_image_path,
            vision_status="success" if detections else "not_connected",
        ),
    ).model_dump()


def analyze_ingredients(state: AgentState) -> dict[str, Any]:
    """사용자 입력 재료를 표준화하고 다음 에이전트용 State를 생성한다.

    Solar API 키가 설정되어 있으면 LLM을 우선 사용하고, 실패하면 규칙 기반
    fallback을 사용한다.
    """

    has_confirmation = _has_confirmation_input(state)
    # print(f"[agent1] has_confirmation={has_confirmation}")
    # print(f"[agent1] confirmed_ingredients={state.get('confirmed_ingredients', [])}")
    # print(f"[agent1] user_input_ingredients={state.get('user_input_ingredients', [])}")
    # print(f"[agent1] ingredient_confirmation={state.get('ingredient_confirmation')}")

    if has_confirmation:
        return _build_user_confirmed_output(state)

    detections: list[ImageDetection] = []
    detector_error = ""

    if state.get("image_path"):
        try:
            detections = _detect_image_ingredients(state)
        except Exception as exc:
            detector_error = f"이미지 재료 탐지 실패: {exc}"

    ingredients = _ingredient_inputs_from_state(state, detections)
    # print(f"[agent1] ingredients from state={ingredients}")
    # print(f"[agent1] _should_call_solar={_should_call_solar()}")
    if not ingredients:
        if detector_error:
            return IngredientAnalyzerOutput(
                detected_ingredients=[],
                uncertain_ingredients=[],
                available_ingredients=[],
                ingredient_info=IngredientInfo(),
                confirmation_options=[],
                vision_status="vision_error",
                vision_message=detector_error,
                raw_vision_result=_build_raw_result(
                    state,
                    analysis_source="detector",
                    detections=detections,
                    vision_status="vision_error",
                    error=detector_error,
                ),
            ).model_dump()

        return _analyze_with_rules(state)

    if not _should_call_solar():
        return _analyze_with_rules(state, detections=detections, error=detector_error)

    try:
        detected_ingredients, llm_result = _call_solar_ingredient_analyzer(ingredients)
        return _build_llm_output(state, detected_ingredients, llm_result, detections)
    except (KeyError, TypeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        return _analyze_with_rules(
            state,
            detections=detections,
            error=f"Solar 응답 파싱 실패: {exc}",
        )
    except Exception as exc:
        return _analyze_with_rules(
            state,
            detections=detections,
            error=f"Solar API 호출 실패: {exc}",
        )
