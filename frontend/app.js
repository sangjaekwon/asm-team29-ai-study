const state = {
  imageBase64: "",
  imageFilename: "",
  previewUrl: "",
  lastResult: null,
  progressTimer: null,
  progressValue: 0,
  progressStartedAt: 0,
};

const elements = {
  form: document.querySelector("#demoForm"),
  inputScreen: document.querySelector("#inputScreen"),
  scanScreen: document.querySelector("#scanScreen"),
  confirmScreen: document.querySelector("#confirmScreen"),
  recipeScreen: document.querySelector("#recipeScreen"),
  imageInput: document.querySelector("#imageInput"),
  uploadBox: document.querySelector("#uploadBox"),
  uploadTitle: document.querySelector("#uploadTitle"),
  uploadHint: document.querySelector("#uploadHint"),
  uploadPreviewImage: document.querySelector("#uploadPreviewImage"),
  manualIngredients: document.querySelector("#manualIngredients"),
  moodInput: document.querySelector("#moodInput"),
  situationInput: document.querySelector("#situationInput"),
  servingsInput: document.querySelector("#servingsInput"),
  confidenceInput: document.querySelector("#confidenceInput"),
  allowAdditionalToggle: document.querySelector("#allowAdditionalToggle"),
  runButton: document.querySelector("#runButton"),
  resetButton: document.querySelector("#resetButton"),
  serverStatus: document.querySelector("#serverStatus"),
  scanMessage: document.querySelector("#scanMessage"),
  previewImage: document.querySelector("#previewImage"),
  imageStage: document.querySelector("#analysisImageStage"),
  confirmedImage: document.querySelector("#confirmedImage"),
  messageBox: document.querySelector("#messageBox"),
  progressPanel: document.querySelector("#progressPanel"),
  progressLabel: document.querySelector("#progressLabel"),
  progressPercent: document.querySelector("#progressPercent"),
  progressBar: document.querySelector("#progressBar"),
  progressHint: document.querySelector("#progressHint"),
  sureList: document.querySelector("#sureList"),
  uncertainList: document.querySelector("#uncertainList"),
  confirmationSection: document.querySelector("#confirmationSection"),
  confirmationList: document.querySelector("#confirmationList"),
  additionalIngredients: document.querySelector("#additionalIngredients"),
  confirmButton: document.querySelector("#confirmButton"),
  recipeSummary: document.querySelector("#recipeSummary"),
  recipeCard: document.querySelector("#recipeCard"),
  langsmithStatus: document.querySelector("#langsmithStatus"),
  langsmithHint: document.querySelector("#langsmithHint"),
  langsmithLink: document.querySelector("#langsmithLink"),
};

const LANGSMITH_HOME = "https://smith.langchain.com";

function splitIngredients(value) {
  return value
    .split(/[,;\n]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function confidencePercent(value) {
  return `${Math.round((value ?? 1) * 100)}%`;
}

function showScreen(name) {
  const screens = {
    input: elements.inputScreen,
    scan: elements.scanScreen,
    confirm: elements.confirmScreen,
    recipe: elements.recipeScreen,
  };

  Object.values(screens).forEach((screen) => screen.classList.remove("active"));
  screens[name].classList.add("active");

  document.querySelectorAll(".flow-step").forEach((step) => {
    step.classList.remove("active", "done");
  });

  if (name === "input") {
    document.querySelector("[data-step='input']").classList.add("active");
  }
  if (name === "scan" || name === "confirm") {
    document.querySelector("[data-step='input']").classList.add("done");
    document.querySelector("[data-step='confirm']").classList.add("active");
  }
  if (name === "recipe") {
    document.querySelector("[data-step='input']").classList.add("done");
    document.querySelector("[data-step='confirm']").classList.add("done");
    document.querySelector("[data-step='recipe']").classList.add("active");
  }

  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setBusy(isBusy) {
  elements.runButton.disabled = isBusy;
  elements.confirmButton.disabled = isBusy;
  elements.serverStatus.textContent = isBusy ? "실행 중" : "대기 중";
  elements.serverStatus.className = `server-pill${isBusy ? " running" : ""}`;
}

function progressText(value, stage) {
  if (stage === "final") {
    if (value < 35) {
      return ["수정한 재료 반영", "방금 확인한 재료를 다시 정리하고 있습니다."];
    }
    if (value < 58) {
      return ["상황 분석", "기분과 상황에 맞는 요리 방향을 잡고 있습니다."];
    }
    if (value < 78) {
      return ["요리 후보 선택", "지금 재료로 만들기 좋은 메뉴를 고르고 있습니다."];
    }
    return ["레시피 작성", "친절하게 따라 할 수 있는 조리 순서를 만들고 있습니다."];
  }

  if (value < 44) {
    return ["사진 분석", "사진 속 재료와 위치를 찾고 있습니다."];
  }
  if (value < 68) {
    return ["재료 이름 정리", "찾은 재료를 한국어 식재료명으로 바꾸고 있습니다."];
  }
  return ["확인 화면 준비", "애매한 재료와 빠진 재료를 확인할 수 있게 정리하고 있습니다."];
}

function updateProgress(value, label, hint) {
  const progress = Math.max(0, Math.min(100, Math.round(value)));
  state.progressValue = progress;
  elements.progressLabel.textContent = label;
  elements.progressPercent.textContent = `${progress}%`;
  elements.progressBar.style.width = `${progress}%`;
  elements.progressHint.textContent = hint;
}

function startProgress(stage) {
  window.clearInterval(state.progressTimer);
  state.progressValue = 5;
  state.progressStartedAt = Date.now();
  elements.progressPanel.classList.remove("hidden", "error");

  const [initialLabel, initialHint] = progressText(state.progressValue, stage);
  updateProgress(state.progressValue, initialLabel, initialHint);

  state.progressTimer = window.setInterval(() => {
    const elapsedSeconds = Math.floor((Date.now() - state.progressStartedAt) / 1000);
    const ceiling = stage === "final" ? 98 : 96;
    const remaining = ceiling - state.progressValue;
    const delta = remaining > 8 ? Math.max(1, Math.ceil(remaining * 0.08)) : 1;
    const nextValue = Math.min(ceiling, state.progressValue + delta);
    let [label, hint] = progressText(nextValue, stage);
    if (stage === "final" && nextValue >= 92) {
      label = "레시피 응답 대기";
      hint = `Agent4/Agent5가 레시피를 정리하는 중입니다. ${elapsedSeconds}초째 처리 중입니다.`;
    }
    updateProgress(nextValue, label, hint);
  }, 700);
}

function completeProgress(stage) {
  window.clearInterval(state.progressTimer);
  state.progressTimer = null;
  state.progressStartedAt = 0;
  const label = stage === "final" ? "레시피 완성" : "인식 완료";
  const hint =
    stage === "final"
      ? "레시피를 보기 좋게 정리했습니다."
      : "인식한 재료를 확인 화면에 정리했습니다.";
  updateProgress(100, label, hint);
}

function failProgress(message) {
  window.clearInterval(state.progressTimer);
  state.progressTimer = null;
  state.progressStartedAt = 0;
  elements.progressPanel.classList.remove("hidden");
  elements.progressPanel.classList.add("error");
  updateProgress(state.progressValue || 100, "오류", message || "요청 처리 중 오류가 발생했습니다.");
}

function imageUrlFromResult(result) {
  return result?.annotated_image_url || state.previewUrl;
}

function cacheSafeImageUrl(imageUrl) {
  if (!imageUrl || imageUrl.startsWith("data:")) {
    return imageUrl;
  }

  return `${imageUrl}${imageUrl.includes("?") ? "&" : "?"}t=${Date.now()}`;
}

function setLangSmithTarget(url) {
  const targetUrl = url || LANGSMITH_HOME;
  elements.langsmithLink.href = targetUrl;
}

function renderLangSmithStatus(langsmith = {}) {
  const status = langsmith.status || "token_required";
  const project = langsmith.project || "asm-team29-recipe-demo";
  const traceUrl = langsmith.latest_run_url || LANGSMITH_HOME;

  if (status === "connected") {
    elements.langsmithStatus.textContent = "LangSmith 연결됨";
    elements.langsmithStatus.className = "server-pill running";
    elements.langsmithHint.textContent = `project: ${project} / LangSmith에서 waterfall trace를 확인하세요.`;
    elements.langsmithLink.textContent = "Tracing 바로 열기";
    setLangSmithTarget(traceUrl);
    return;
  }

  elements.langsmithStatus.textContent =
    status === "token_error" ? "LangSmith 토큰 오류" : "LangSmith 토큰 필요";
  elements.langsmithStatus.className = "server-pill";
  elements.langsmithHint.textContent =
    status === "token_error"
      ? "LangSmith API key를 다시 발급해서 .env에 넣어주세요."
      : ".env에 LANGSMITH_API_KEY를 넣으면 LangSmith trace가 연결됩니다.";
  elements.langsmithLink.textContent = "LangSmith 열기";
  setLangSmithTarget(LANGSMITH_HOME);
}

async function refreshLangSmithStatus() {
  try {
    const response = await fetch("/debug/workflow/latest", {
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const trace = await response.json();
    renderLangSmithStatus(trace.langsmith || {});
  } catch (error) {
    elements.langsmithStatus.textContent = "LangSmith 연결 실패";
    elements.langsmithStatus.className = "server-pill";
    elements.langsmithHint.textContent = error.message;
    setLangSmithTarget(LANGSMITH_HOME);
  }
}

function renderAnalysisImage(result) {
  const imageUrl = imageUrlFromResult(result);
  if (!imageUrl) {
    elements.previewImage.removeAttribute("src");
    elements.imageStage.classList.remove("has-image");
    return;
  }

  elements.previewImage.src = cacheSafeImageUrl(imageUrl);
  elements.imageStage.classList.add("has-image");
}

function renderConfirmedImage(result) {
  const imageUrl = imageUrlFromResult(result);
  if (!imageUrl) {
    elements.confirmedImage.removeAttribute("src");
    elements.confirmedImage.parentElement.classList.remove("has-image");
    return;
  }

  elements.confirmedImage.src = cacheSafeImageUrl(imageUrl);
  elements.confirmedImage.parentElement.classList.add("has-image");
}

function ingredientChip(ingredient) {
  const chip = document.createElement("span");
  chip.className = `chip${ingredient.needs_confirmation ? " warn" : ""}`;
  chip.textContent = `${ingredient.name} ${confidencePercent(ingredient.confidence)}`;
  return chip;
}

function renderIngredientGroups(result) {
  const detected = result?.detected_ingredients || [];
  const sure = detected.filter((ingredient) => !ingredient.needs_confirmation);
  const uncertain = detected.filter((ingredient) => ingredient.needs_confirmation);

  elements.sureList.innerHTML = "";
  elements.uncertainList.innerHTML = "";

  if (!sure.length) {
    const empty = document.createElement("p");
    empty.className = "empty-text";
    empty.textContent = "확실하게 찾은 재료가 아직 없습니다.";
    elements.sureList.appendChild(empty);
  } else {
    sure.forEach((ingredient) => elements.sureList.appendChild(ingredientChip(ingredient)));
  }

  if (!uncertain.length) {
    const empty = document.createElement("p");
    empty.className = "empty-text";
    empty.textContent = "확인이 필요한 재료는 없습니다. 빠진 재료만 추가하면 됩니다.";
    elements.uncertainList.appendChild(empty);
  } else {
    uncertain.forEach((ingredient) => elements.uncertainList.appendChild(ingredientChip(ingredient)));
  }
}

function renderConfirmation(result) {
  const options = result.confirmation_options || [];
  elements.confirmationList.innerHTML = "";

  if (!options.length) {
    const item = document.createElement("article");
    item.className = "confirmation-empty";
    item.innerHTML = `
      <strong>크게 애매한 재료는 없어 보여요.</strong>
      <p>그래도 사진에서 빠진 재료가 있으면 아래에 직접 적어주세요.</p>
    `;
    elements.confirmationList.appendChild(item);
    return;
  }

  options.forEach((option, index) => {
    const item = document.createElement("article");
    item.className = "confirmation-item";
    item.dataset.name = option.name;

    const candidates = [option.name, ...(option.candidates || [])]
      .filter(Boolean)
      .filter((candidate, candidateIndex, array) => array.indexOf(candidate) === candidateIndex);

    const choices = candidates
      .map((candidate, candidateIndex) => {
        const checked = candidateIndex === 0 ? "checked" : "";
        return `
          <label class="candidate-choice">
            <input
              data-role="candidate"
              name="candidate-${index}"
              type="radio"
              value="${escapeHtml(candidate)}"
              ${checked}
            />
            <span>${escapeHtml(candidate)}</span>
          </label>
        `;
      })
      .join("");

    item.innerHTML = `
      <div class="confirmation-title">
        <div>
          <strong>${escapeHtml(option.name)}</strong>
          <p>${escapeHtml(option.reason || "이 재료는 한 번 확인이 필요합니다.")}</p>
        </div>
        <span class="confidence-badge">${confidencePercent(option.confidence)}</span>
      </div>
      <div class="candidate-grid">${choices}</div>
      <label class="field compact-field">
        <span>아니면 직접 고쳐쓰기</span>
        <input data-role="manual" type="text" placeholder="예: 청경채" />
      </label>
      <label class="reject-row">
        <input data-role="reject" type="checkbox" />
        이 재료는 빼주세요
      </label>
    `;

    elements.confirmationList.appendChild(item);
  });
}

function renderConfirmScreen(result) {
  state.lastResult = result;
  elements.messageBox.textContent =
    result.vision_message || "사진에서 찾은 재료를 확인해주세요.";
  renderConfirmedImage(result);
  renderIngredientGroups(result);
  renderConfirmation(result);
  showScreen("confirm");
}

function difficultyLabel(value) {
  const labels = {
    easy: "쉬워요",
    medium: "보통이에요",
    hard: "손이 좀 가요",
  };
  return labels[value] || value || "난이도 보통";
}

function friendlyRouteMessage(result) {
  const messages = {
    candidate_conflicts_with_user_context:
      "지금 기분이나 시간 조건과 후보 요리가 잘 맞지 않았어요. 조건을 조금 완화하거나 재료를 다시 확인해보면 좋아요.",
    no_candidate_matches_recipe_type:
      "선택된 요리 스타일에 맞는 후보를 찾지 못했어요. 재료를 조금 더 추가해보세요.",
    required_ingredients_missing:
      "핵심 재료가 부족해서 바로 만들기는 어려워요.",
    available_ingredients_required:
      "사용할 재료가 아직 확인되지 않았어요.",
    candidate_foods_required:
      "비교할 요리 후보가 아직 준비되지 않았어요.",
    candidate_generation_failed:
      "AI candidate generation failed. Check API settings and server logs.",
  };

  return (
    messages[result.route_message] ||
    result.candidate_generation_error ||
    result.recipe_generation_error ||
    result.generation_message ||
    result.route_message ||
    "재료를 조금 더 확인해주세요."
  );
}

function buildRecommendationReasons(recipe, result) {
  const reasons = [
    ...(recipe.recommendation_reasons || []),
    result.selected_recipe?.reason || "",
  ]
    .map((reason) => String(reason).trim())
    .filter(Boolean);

  if (!reasons.length) {
    reasons.push("Agent4가 이 레시피를 최종 후보로 선택했지만, 선택 이유가 응답에 포함되지 않았습니다.");
  }

  return reasons.reduce((deduped, reason) => {
    const current = reason.replace(/\.+$/, "").replace(/\s+/g, " ").trim();
    const duplicated = deduped.some((existing) => {
      const previous = existing.replace(/\.+$/, "").replace(/\s+/g, " ").trim();
      return (
        current === previous ||
        (current.length >= 20 && previous.includes(current)) ||
        (previous.length >= 20 && current.includes(previous))
      );
    });

    if (!duplicated) {
      deduped.push(reason);
    }
    return deduped;
  }, []);
}

function renderRecipe(result) {
  const recipe = result.generated_recipe;
  if (!recipe) {
    elements.recipeCard.innerHTML = `
      <div class="friendly-card warning-card">
        <h3>아직 레시피를 만들기 어려워요</h3>
        <p>${escapeHtml(friendlyRouteMessage(result))}</p>
      </div>
    `;
    showScreen("recipe");
    return;
  }

  const mood = elements.moodInput.value.trim();
  const situation = elements.situationInput.value.trim();
  const substitutions = (recipe.substitutions || result.substitutions || []).filter(
    (item) => item && String(item.original || "").trim(),
  );
  const additional = (recipe.additional_ingredients || result.additional_ingredients || []).filter(
    (item) => String(item || "").trim(),
  );
  const tips = recipe.cooking_tips || [];
  const recommendationReasons = buildRecommendationReasons(recipe, result);

  elements.recipeSummary.textContent = `${mood || "오늘"} 기분과 "${situation || "든든하게 먹고 싶어요"}" 상황에 맞춰 ${recipe.recipe_name}을 추천합니다.`;
  elements.recipeCard.innerHTML = `
    <article class="recipe-card">
      <div class="recipe-hero">
        <p>좋아요. 지금 재료라면 이 메뉴가 제일 잘 맞아요.</p>
        <h3>${escapeHtml(recipe.recipe_name)}</h3>
      </div>

      <div class="recipe-meta">
        <span class="chip strong">${recipe.cooking_time_minutes || 15}분 정도</span>
        <span class="chip strong">${difficultyLabel(recipe.difficulty)}</span>
        <span class="chip strong">${recipe.servings || 1}인분</span>
      </div>

      <section class="recipe-block">
        <h3>추천한 이유</h3>
        <ul class="plain-list">
          ${recommendationReasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}
        </ul>
      </section>

      <section class="recipe-block">
        <h3>오늘 쓸 재료</h3>
        <div class="chip-list">
          ${(recipe.ingredients || []).map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join("")}
        </div>
      </section>

      <section class="recipe-block">
        <h3>이 순서대로 해보세요</h3>
        <ol class="step-list">
          ${(recipe.cooking_steps || [])
            .map((step) => `<li><span>${escapeHtml(step)}</span></li>`)
            .join("")}
        </ol>
      </section>

      ${
        tips.length
          ? `<section class="recipe-block"><h3>맛있게 되는 한 끗</h3><ul class="plain-list">${tips
              .map((tip) => `<li>${escapeHtml(tip)}</li>`)
              .join("")}</ul></section>`
          : ""
      }

      ${
        substitutions.length || additional.length
          ? `<section class="recipe-block soft-block">
              <h3>재료가 조금 달라도 괜찮아요</h3>
              ${
                substitutions.length
                  ? `<ul class="plain-list">${substitutions
                      .map(
                        (item) =>
                          `<li>${escapeHtml(item.original)}은 ${
                            item.replacement
                              ? `${escapeHtml(item.replacement)}로 바꿔도 됩니다`
                              : "없으면 빼도 됩니다"
                          }.</li>`,
                      )
                      .join("")}</ul>`
                  : ""
              }
              ${
                additional.length
                  ? `<p class="need-text">있으면 더 좋은 재료: ${additional
                      .map((item) => escapeHtml(item))
                      .join(", ")}</p>`
                  : ""
              }
            </section>`
          : ""
      }
    </article>
  `;

  showScreen("recipe");
}

function renderResult(result, stage) {
  state.lastResult = result;
  renderAnalysisImage(result);

  if (stage === "scan") {
    renderConfirmScreen(result);
    return;
  }

  renderRecipe(result);
}

async function postRecommend(payload, stage) {
  setBusy(true);
  if (stage === "scan") {
    elements.scanMessage.textContent = "사진을 분석하고 있습니다. 완료되면 확인 화면으로 넘어갑니다.";
    renderAnalysisImage(null);
    showScreen("scan");
  } else {
    showScreen("scan");
    elements.scanMessage.textContent = "확인한 재료로 레시피를 만들고 있습니다.";
  }
  startProgress(stage);

  try {
    const response = await fetch("/recommend", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }

    const result = await response.json();
    completeProgress(stage);
    refreshLangSmithStatus();
    window.setTimeout(() => renderResult(result, stage), 350);
  } catch (error) {
    elements.scanMessage.textContent = error.message;
    failProgress(error.message);
  } finally {
    setBusy(false);
  }
}

function ingredientPolicyValue() {
  return elements.allowAdditionalToggle.checked ? "allow_additional" : "only_available";
}

function basePayload() {
  return {
    image_base64: state.imageBase64,
    image_filename: state.imageFilename,
    user_input_ingredients: splitIngredients(elements.manualIngredients.value),
    user_mood_input: elements.moodInput.value.trim(),
    user_situation_input: elements.situationInput.value.trim(),
    servings: Number(elements.servingsInput.value || 1),
    confidence_threshold: Number(elements.confidenceInput.value || 0.4),
    ingredient_policy: ingredientPolicyValue(),
  };
}

elements.imageInput.addEventListener("change", () => {
  const file = elements.imageInput.files?.[0];
  if (!file) {
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    state.imageBase64 = String(reader.result || "");
    state.imageFilename = file.name;
    state.previewUrl = state.imageBase64;
    elements.uploadPreviewImage.src = state.previewUrl;
    elements.uploadBox.classList.add("uploaded");
    elements.uploadTitle.textContent = "업로드 완료";
    elements.uploadHint.textContent = file.name;
    renderAnalysisImage(null);
  };
  reader.readAsDataURL(file);
});

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!state.imageBase64) {
    elements.uploadBox.classList.add("needs-attention");
    elements.uploadTitle.textContent = "사진을 먼저 올려주세요";
    elements.uploadHint.textContent = "데모에서는 사진 분석 흐름을 먼저 보여줍니다";
    return;
  }
  elements.uploadBox.classList.remove("needs-attention");
  postRecommend(basePayload(), "scan");
});

elements.confirmButton.addEventListener("click", () => {
  const lastResult = state.lastResult;
  if (!lastResult) {
    return;
  }

  const rejected = [];
  const replacements = {};
  const accepted = [];
  const items = elements.confirmationList.querySelectorAll(".confirmation-item");

  items.forEach((item) => {
    const name = item.dataset.name || "";
    const selected = item.querySelector("[data-role='candidate']:checked")?.value || name;
    const manual = item.querySelector("[data-role='manual']").value.trim();
    const reject = item.querySelector("[data-role='reject']").checked;
    const replacement = manual || selected;

    if (reject) {
      rejected.push(name);
      return;
    }
    if (replacement) {
      accepted.push(replacement);
    }
    if (replacement && replacement !== name) {
      replacements[name] = replacement;
    }
  });

  if (!accepted.length) {
    (lastResult.detected_ingredients || []).forEach((ingredient) => {
      if (ingredient?.name && !rejected.includes(ingredient.name)) {
        accepted.push(ingredient.name);
      }
    });
  }

  postRecommend(
    {
      detected_ingredients: lastResult.detected_ingredients || [],
      ingredient_confirmation: {
        accepted_ingredients: accepted,
        rejected_ingredients: rejected,
        replacements,
        additional_ingredients_text: elements.additionalIngredients.value.trim(),
      },
      user_mood_input: elements.moodInput.value.trim(),
      user_situation_input: elements.situationInput.value.trim(),
      servings: Number(elements.servingsInput.value || 1),
      confidence_threshold: Number(elements.confidenceInput.value || 0.4),
      ingredient_policy: ingredientPolicyValue(),
    },
    "final",
  );
});

elements.resetButton.addEventListener("click", () => {
  state.imageBase64 = "";
  state.imageFilename = "";
  state.previewUrl = "";
  state.lastResult = null;
  state.progressValue = 0;
  window.clearInterval(state.progressTimer);
  state.progressTimer = null;
  state.progressStartedAt = 0;

  elements.form.reset();
  elements.manualIngredients.value = "";
  elements.servingsInput.value = "1";
  elements.confidenceInput.value = "0.4";
  elements.allowAdditionalToggle.checked = false;
  elements.uploadPreviewImage.removeAttribute("src");
  elements.uploadBox.classList.remove("uploaded", "needs-attention");
  elements.uploadTitle.textContent = "재료 사진 올리기";
  elements.uploadHint.textContent = "사진을 선택하면 바로 미리보기가 표시됩니다";
  elements.previewImage.removeAttribute("src");
  elements.imageStage.classList.remove("has-image");
  elements.confirmedImage.removeAttribute("src");
  elements.confirmedImage.parentElement.classList.remove("has-image");
  elements.sureList.innerHTML = "";
  elements.uncertainList.innerHTML = "";
  elements.confirmationList.innerHTML = "";
  elements.additionalIngredients.value = "";
  elements.recipeCard.innerHTML = "";
  elements.recipeSummary.textContent = "확인한 재료와 지금 상황을 바탕으로 레시피를 정리했습니다.";
  elements.scanMessage.textContent = "잠시만 기다려주세요. 인식 결과는 바로 아래 사진에 표시됩니다.";
  elements.progressPanel.classList.add("hidden");
  elements.progressPanel.classList.remove("error");
  updateProgress(0, "대기 중", "실행 버튼을 누르면 분석이 시작됩니다.");
  setBusy(false);
  showScreen("input");
});

refreshLangSmithStatus();
window.setInterval(refreshLangSmithStatus, 8000);
