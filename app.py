"""Streamlit 프론트엔드 — 냉장고 재료 기반 레시피 추천 서비스."""

import base64
import json
import tempfile
from pathlib import Path

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="오늘 뭐 먹지?",
    page_icon="🍳",
    layout="centered",
)

# ── 세션 초기화 ──────────────────────────────────────────────
if "ingredients" not in st.session_state:
    st.session_state.ingredients = []          # 확정된 재료 목록
if "page" not in st.session_state:
    st.session_state.page = "home"             # home | recipe
if "recipe_result" not in st.session_state:
    st.session_state.recipe_result = None
if "pending_confirmation" not in st.session_state:
    st.session_state.pending_confirmation = None  # 미확정 재료 목록


# ── 헬퍼 ─────────────────────────────────────────────────────
def call_recommend(
    image_path: str,
    user_input_ingredients: list[str],
    mood: str,
    situation: str,
    confirmed_ingredients: list[str] | None = None,
) -> dict:
    payload = {
        "image_path": image_path,
        "user_input_ingredients": user_input_ingredients,
        "confirmed_ingredients": confirmed_ingredients or [],
        "user_mood_input": mood,
        "user_situation_input": situation,
        "confidence_threshold": 0.7,
        "servings": 1,
    }
    resp = requests.post(f"{API_BASE}/recommend", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def save_upload_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.getbuffer())
        return f.name


def ingredient_badge(name: str) -> str:
    return f"**{name}**"


# ── 홈 페이지 ────────────────────────────────────────────────
def page_home():
    st.title("🍳 오늘 뭐 먹지?")
    st.caption("냉장고 재료와 오늘 기분으로 딱 맞는 레시피를 찾아드려요.")

    # ── 재료 섹션 ──────────────────────────────
    st.subheader("🥦 남은 재료")

    if st.session_state.ingredients:
        cols = st.columns(4)
        remove_target = None
        for i, ing in enumerate(st.session_state.ingredients):
            with cols[i % 4]:
                if st.button(f"✕ {ing}", key=f"rm_{i}", use_container_width=True):
                    remove_target = ing
        if remove_target:
            st.session_state.ingredients.remove(remove_target)
            st.rerun()
    else:
        st.info("아직 재료가 없어요. 사진을 찍거나 직접 추가해 보세요.")

    st.divider()

    # ── 재료 추가 방법 탭 ──────────────────────
    tab_photo, tab_text = st.tabs(["📷 사진으로 추가", "✏️ 직접 입력"])

    with tab_photo:
        uploaded = st.file_uploader(
            "냉장고 사진을 올려주세요",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.image(uploaded, use_container_width=True)
            if st.button("재료 인식하기", type="primary", use_container_width=True):
                with st.spinner("사진에서 재료를 인식하는 중..."):
                    image_path = save_upload_to_temp(uploaded)
                    try:
                        result = call_recommend(
                            image_path=image_path,
                            user_input_ingredients=[],
                            mood="",
                            situation="",
                        )
                        vision_status = result.get("vision_status", "success")
                        detected = result.get("available_ingredients", [])
                        uncertain = result.get("uncertain_ingredients", [])
                        message = result.get("vision_message", "")

                        if vision_status == "vision_error":
                            st.error(f"인식 오류: {message}")
                        elif vision_status == "no_ingredient_detected":
                            st.warning("재료를 찾지 못했어요. 다른 사진을 시도하거나 직접 입력해 주세요.")
                        elif vision_status == "need_user_confirmation" and uncertain:
                            st.session_state.pending_confirmation = {
                                "detected": detected,
                                "uncertain": uncertain,
                                "message": message,
                            }
                            st.rerun()
                        else:
                            new_items = [
                                i for i in detected
                                if i not in st.session_state.ingredients
                            ]
                            st.session_state.ingredients.extend(new_items)
                            if message:
                                st.success(message)
                            st.rerun()
                    except Exception as e:
                        st.error(f"서버 오류: {e}")

    # ── 미확정 재료 확인 UI ──────────────────────
    if st.session_state.pending_confirmation:
        pend = st.session_state.pending_confirmation
        st.divider()
        st.warning(pend.get("message", "아래 재료를 확인해 주세요."))

        accepted = list(pend.get("detected", []))
        for unc in pend.get("uncertain", []):
            checked = st.checkbox(f"{unc} (불확실)", key=f"unc_{unc}", value=True)
            if not checked and unc in accepted:
                accepted.remove(unc)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("확인 완료", type="primary", use_container_width=True):
                new_items = [i for i in accepted if i not in st.session_state.ingredients]
                st.session_state.ingredients.extend(new_items)
                st.session_state.pending_confirmation = None
                st.rerun()
        with col2:
            if st.button("취소", use_container_width=True):
                st.session_state.pending_confirmation = None
                st.rerun()

    with tab_text:
        manual = st.text_input(
            "재료를 쉼표로 구분해서 입력하세요",
            placeholder="예) 계란, 양파, 간장, 두부",
        )
        if st.button("추가", use_container_width=True):
            if manual.strip():
                items = [i.strip() for i in manual.split(",") if i.strip()]
                new_items = [i for i in items if i not in st.session_state.ingredients]
                st.session_state.ingredients.extend(new_items)
                st.rerun()
            else:
                st.warning("재료를 입력해 주세요.")

    st.divider()

    # ── 음식 추천하기 섹션 ─────────────────────
    st.subheader("😊 오늘 기분이 어때요?")

    mood_input = st.text_area(
        "기분이나 상황을 자유롭게 적어주세요",
        placeholder="예) 너무 피곤해서 간단한 걸 먹고 싶어요 / 기분 전환이 필요해요 / 든든하게 먹고 싶어요",
        height=100,
        label_visibility="collapsed",
    )

    situation_input = st.text_input(
        "현재 상황 (선택)",
        placeholder="예) 퇴근 직후, 늦은 밤, 시간이 30분밖에 없어요",
    )

    recommend_disabled = len(st.session_state.ingredients) == 0
    if recommend_disabled:
        st.caption("재료를 먼저 추가해야 추천받을 수 있어요.")

    if st.button(
        "🍽️ 음식 추천받기",
        type="primary",
        use_container_width=True,
        disabled=recommend_disabled,
    ):
        with st.spinner("AI가 기분을 분석하고 알맞은 음식을 찾는 중..."):
            try:
                result = call_recommend(
                    image_path="",
                    user_input_ingredients=st.session_state.ingredients,
                    mood=mood_input.strip(),
                    situation=situation_input.strip(),
                    confirmed_ingredients=st.session_state.ingredients,
                )
                st.session_state.recipe_result = result
                st.session_state.page = "recipe"
                st.rerun()
            except requests.exceptions.ConnectionError:
                st.error("백엔드 서버에 연결할 수 없어요. `uvicorn main:app` 을 먼저 실행해 주세요.")
            except Exception as e:
                st.error(f"오류가 발생했어요: {e}")


# ── 레시피 페이지 ────────────────────────────────────────────
def page_recipe():
    result = st.session_state.recipe_result or {}

    if st.button("← 돌아가기"):
        st.session_state.page = "home"
        st.rerun()

    gen_status = result.get("generation_status", "failed")
    recipe = result.get("generated_recipe")
    route = result.get("route", "")
    route_message = result.get("route_message", "")
    recipe_type_raw = result.get("recipe_type") or "korean"

    cuisine_label = {
        "korean": "한식",
        "chinese": "중식",
        "japanese": "일식",
        "western": "양식",
    }.get(recipe_type_raw, recipe_type_raw)

    # ── 실패 처리 ───────────────────────────────
    if gen_status == "failed" or recipe is None:
        st.error("레시피를 생성하지 못했어요.")

        if route == "missing_ingredient":
            additional = result.get("additional_ingredients", [])
            st.warning("핵심 재료가 부족해요.")
            if additional:
                st.write("**추가로 필요한 재료:**")
                for ing in additional:
                    st.write(f"- {ing}")
        elif route == "constraint_conflict":
            st.info("조건과 재료가 맞지 않아요. 기분이나 재료를 바꿔서 다시 시도해 보세요.")
        else:
            st.info(route_message or "다른 재료로 다시 시도해 보세요.")
        return

    # ── 레시피 출력 ─────────────────────────────
    name = recipe.get("recipe_name", "추천 레시피")
    st.title(f"🍽️ {name}")

    meta_cols = st.columns(3)
    with meta_cols[0]:
        st.metric("요리 스타일", cuisine_label)
    with meta_cols[1]:
        difficulty_label = {"easy": "쉬움", "normal": "보통", "hard": "어려움"}.get(
            recipe.get("difficulty", "normal"), "보통"
        )
        st.metric("난이도", difficulty_label)
    with meta_cols[2]:
        st.metric("조리 시간", f"{recipe.get('cooking_time_minutes', '?')}분")

    st.divider()

    # ── 추천 이유 ───────────────────────────────
    raw_recommendation_reasons = [
        reason
        for reason in [
            *recipe.get("recommendation_reasons", []),
            (result.get("selected_recipe") or {}).get("reason", ""),
        ]
        if str(reason).strip()
    ]
    recommendation_reasons = []
    compare_values = []
    for reason in raw_recommendation_reasons:
        cleaned = str(reason).strip()
        compare_value = " ".join(cleaned.rstrip(".").split())
        is_duplicate = any(
            compare_value == existing
            or (len(compare_value) >= 20 and compare_value in existing)
            or (len(existing) >= 20 and existing in compare_value)
            for existing in compare_values
        )
        if not is_duplicate:
            recommendation_reasons.append(cleaned)
            compare_values.append(compare_value)

    if not recommendation_reasons:
        recommendation_reasons.append("Agent4가 이 레시피를 최종 후보로 선택했지만, 선택 이유가 응답에 포함되지 않았습니다.")

    st.subheader("추천한 이유")
    for reason in recommendation_reasons:
        st.write(f"- {reason}")
    st.divider()

    # ── 재료 ────────────────────────────────────
    st.subheader("🛒 재료")
    ingredients = recipe.get("ingredients", [])
    if ingredients:
        col1, col2 = st.columns(2)
        for i, ing in enumerate(ingredients):
            with (col1 if i % 2 == 0 else col2):
                st.write(f"- {ing}")
    else:
        st.write("재료 정보가 없어요.")

    # ── 부족한 재료 ──────────────────────────────
    additional = recipe.get("additional_ingredients", [])
    if additional:
        st.info(f"**추가로 준비하면 좋은 재료:** {', '.join(additional)}")

    # ── 대체 재료 ──────────────────────────────
    substitutions = [
        sub for sub in recipe.get("substitutions", []) if str(sub.get("original", "")).strip()
    ]
    if substitutions:
        with st.expander("재료 대체/생략 정보"):
            for sub in substitutions:
                original = sub.get("original", "")
                replacement = sub.get("replacement")
                reason = sub.get("reason", "")
                if replacement:
                    st.write(f"- **{original}** → {replacement}  ({reason})")
                else:
                    st.write(f"- **{original}** 생략 가능 ({reason})")

    st.divider()

    # ── 조리 순서 ───────────────────────────────
    st.subheader("👨‍🍳 조리 순서")
    steps = recipe.get("cooking_steps", [])
    if steps:
        for i, step in enumerate(steps, 1):
            st.markdown(f"**{i}.** {step}")
    else:
        st.write("조리 순서 정보가 없어요.")

    # ── 요리 팁 ─────────────────────────────────
    tips = recipe.get("cooking_tips", [])
    if tips:
        st.divider()
        st.subheader("💡 요리 팁")
        for tip in tips:
            st.write(f"- {tip}")

    st.divider()

    # ── 다시 추천받기 ────────────────────────────
    if st.button("🔄 다시 추천받기", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()


# ── 라우팅 ───────────────────────────────────────────────────
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "recipe":
    page_recipe()
