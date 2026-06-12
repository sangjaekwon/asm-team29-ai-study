# ASM Team29 — 냉장고 재료 기반 레시피 추천 서비스

사용자가 가진 재료(사진 또는 직접 입력)와 현재 기분·상황을 바탕으로,
LangGraph 멀티 에이전트 파이프라인이 실제로 만들 수 있는 레시피를
추천해 주는 서비스입니다.

Python: 3.14.4

## System Architecture

```
Browser (frontend/)
   │  fetch("/recommend"), fetch("/debug/workflow/latest")
   ▼
FastAPI Backend (main.py)
   │  AgentState 구성 → run_recipe_graph(state)
   ▼
LangGraph Multi-Agent Pipeline (agents/graph.py)
   │
   ├─ Agent1 ingredient_analyzer   (재료 인식: OWLv2 로컬 모델 + Solar)
   │     └─ vision_status == "need_user_confirmation" 이면 END (사용자 확인 대기)
   ├─ Agent2 context_analyzer      (기분/상황 → food_directions)
   ├─ Agent3 cuisine_router        (recipe_type 결정)
   ├─ Agent4 recipe_router         (후보 생성 + 룰 기반 라우팅)
   └─ Agent5 recipe_generator      (최종 레시피 생성)
   ▼
External AI: Solar API (Upstage LLM)
```

- **Orchestration**: LangGraph (`agents/graph.py`)
- **Observability**: LangSmith (선택, `.env`에 `LANGSMITH_API_KEY` 설정 시 활성화)

## Multi-Agent Workflow

| Agent | 모듈 | 역할 |
| --- | --- | --- |
| Agent1 | `agents/agent1` | 이미지/텍스트에서 재료를 인식 (OWLv2 기본, 대체로 YOLO 지원), 확인 필요 재료가 있으면 그래프를 멈추고 사용자 확인을 대기 |
| Agent2 | `agents/agent2` | 사용자의 기분/상황 입력을 분석해 `food_directions`(난이도, 조리 시간, 선호 등) 생성 |
| Agent3 | `agents/agent3` | 재료·상황 정보를 바탕으로 요리 스타일(`korean`/`chinese`/`japanese`/`western`) 결정 |
| Agent4 | `agents/agent4` | 레시피 후보를 생성하고, 보유 재료/조건과 비교해 `can_cook`/`simple`/`no_ingredient`/`conflict` 중 하나로 라우팅 |
| Agent5 | `agents/agent5` | Agent4가 선택한 레시피를 바탕으로 조리 순서·계량·팁이 담긴 최종 레시피 생성 |

공유 상태(State) 스키마는 `agents/schemas.py`의 `AgentState`(TypedDict)에 정의되어
있으며, 각 노드는 자신이 담당하는 필드만 갱신합니다.

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic
- **Multi-Agent Orchestration**: LangGraph
- **LLM**: Solar (Upstage) via OpenAI-compatible API
- **Vision**: OWLv2 (`transformers`, zero-shot object detection)
- **Frontend**: `frontend/` 정적 HTML/CSS/JS (FastAPI가 `/frontend`로 서빙)
- **Testing**: pytest

## Project Structure

```
agents/
  agent1/   재료 인식 (vision + Solar)
  agent2/   기분·상황 분석
  agent3/   요리 스타일 분기
  agent4/   레시피 후보 생성 및 라우팅
  agent5/   최종 레시피 생성
  graph.py     LangGraph 워크플로우 정의
  schemas.py   공유 State / Pydantic 모델
  workflow_trace.py  디버그용 실행 트레이스 기록
frontend/   정적 프론트엔드 (index.html, app.js, styles.css)
main.py     FastAPI 엔트리포인트 (/recommend, /debug/workflow/latest)
tests/      pytest 테스트
```

## Getting Started

### 1. 가상환경 및 의존성 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env.example`을 참고해 `.env` 파일을 작성합니다.

```bash
cp .env.example .env
```

주요 변수:

| 변수 | 설명 |
| --- | --- |
| `SOLAR_API_KEY` | Upstage Solar API 키 (필수) |
| `SOLAR_BASE_URL` | Solar API base URL |
| `SOLAR_MODEL` | 사용할 Solar 모델명 |
| `AGENT1_DETECTOR_BACKEND` | 재료 인식 detector (`owlv2` 기본, `yolo` 대체) |
| `AGENT1_DETECTOR_MODEL` | OWLv2 모델 ID |
| `AGENT1_DETECTOR_THRESHOLD` | 인식 confidence 임계값 |
| `AGENT1_ANNOTATION_OUTPUT_DIR` | 인식 결과 이미지 저장 경로 |
| `LANGSMITH_API_KEY` | (선택) LangSmith 트레이싱용 API 키, 비워두면 비활성화 |

### 3. 서버 실행

```bash
uvicorn main:app --reload
```

서버가 뜨면 다음 주소에서 확인할 수 있습니다.

- `http://localhost:8000/` — 프론트엔드 (`frontend/index.html`)
- `http://localhost:8000/docs` — FastAPI Swagger UI

## API

### `POST /recommend`

보유 재료(이미지/텍스트), 기분/상황, 인분 수, 재료 정책 등을 받아
멀티 에이전트 파이프라인을 실행하고 추천 레시피(또는 사용자 확인 요청)를
반환합니다. 요청/응답 필드는 `main.py`의 `RecommendRequest`와
`/recommend` 핸들러를 참고하세요.

### `GET /debug/workflow/latest`

가장 최근 `/recommend` 실행의 단계별 트레이스를 반환합니다
(`agents/workflow_trace.py`).

## Testing

```bash
pytest
```
