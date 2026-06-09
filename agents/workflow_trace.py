"""Lightweight in-memory workflow trace for the demo page."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any


_TRACE_LOCK = Lock()
_LATEST_TRACE: dict[str, Any] = {
    "trace_id": "",
    "events": [],
    "backend": {},
    "langgraph": {},
    "langsmith": {},
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _langsmith_info() -> dict[str, Any]:
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or ""
    project = os.getenv("LANGSMITH_PROJECT") or "asm-team29-recipe-demo"
    return {
        "enabled": bool(api_key),
        "project": project,
        "status": "connected" if api_key else "token_required",
        "latest_run_url": os.getenv("LANGSMITH_WEB_URL") or "https://smith.langchain.com",
        "error": "",
    }


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _recipe_name(value: Any) -> str:
    plain = _plain(value)
    if isinstance(plain, dict):
        return plain.get("recipe_name") or plain.get("name") or ""
    return ""


def summarize_agent_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return only compact fields that are useful for the demo UI."""

    ingredient_info = _plain(state.get("ingredient_info"))
    food_directions = _plain(state.get("food_directions"))
    selected_recipe = _plain(state.get("selected_recipe"))

    summary: dict[str, Any] = {
        "available_ingredients": _plain(state.get("available_ingredients", [])),
        "vision_status": state.get("vision_status") or "",
        "ingredient_info": "분류 완료" if ingredient_info else "",
        "food_directions": "",
        "recipe_type": state.get("recipe_type") or "",
        "selected_recipe": _recipe_name(selected_recipe),
        "route": state.get("route") or "",
        "ingredients_to_use": _plain(state.get("ingredients_to_use", [])),
        "seasonings_to_use": _plain(state.get("seasonings_to_use", [])),
        "generated_recipe": _recipe_name(state.get("generated_recipe")) or (
            "생성 완료" if state.get("generation_status") == "success" else ""
        ),
    }

    if isinstance(food_directions, dict) and food_directions:
        parts = [
            food_directions.get("difficulty", ""),
            (
                f"{food_directions.get('cooking_time_limit_minutes')}분 이내"
                if food_directions.get("cooking_time_limit_minutes")
                else ""
            ),
            food_directions.get("preferred_taste", ""),
            food_directions.get("preferred_cooking_method", ""),
        ]
        summary["food_directions"] = ", ".join(part for part in parts if part)

    return summary


def start_trace(trace_id: str) -> None:
    with _TRACE_LOCK:
        _LATEST_TRACE.clear()
        _LATEST_TRACE.update(
            {
                "trace_id": trace_id,
                "events": [],
                "backend": {
                    "POST /recommend": "waiting",
                    "FastAPI Request": "waiting",
                    "Save Upload Image": "waiting",
                    "Build AgentState": "waiting",
                    "run_recipe_graph()": "waiting",
                    "Format API Response": "waiting",
                    "API Response": "waiting",
                },
                "langgraph": {
                    "current_node": "",
                    "nodes": {
                        "ingredient_analyzer": "waiting",
                        "context_analyzer": "waiting",
                        "cuisine_router": "waiting",
                        "recipe_router": "waiting",
                        "recipe_generator": "waiting",
                    },
                    "agent_state": {},
                },
                "langsmith": _langsmith_info(),
            }
        )


def record_backend_step(step: str, status: str) -> None:
    with _TRACE_LOCK:
        _LATEST_TRACE.setdefault("backend", {})[step] = status
        _LATEST_TRACE.setdefault("events", []).append(
            {
                "type": "backend",
                "step": step,
                "status": status,
                "timestamp": _now_iso(),
            }
        )


def record_langgraph_node(node_name: str, status: str, state: dict[str, Any]) -> None:
    with _TRACE_LOCK:
        langgraph = _LATEST_TRACE.setdefault("langgraph", {})
        langgraph["current_node"] = node_name
        langgraph.setdefault("nodes", {})[node_name] = status
        langgraph["agent_state"] = summarize_agent_state(state)
        _LATEST_TRACE.setdefault("events", []).append(
            {
                "type": "langgraph",
                "node": node_name,
                "status": status,
                "timestamp": _now_iso(),
            }
        )


def latest_trace() -> dict[str, Any]:
    with _TRACE_LOCK:
        _LATEST_TRACE["langsmith"] = _langsmith_info()
        return _plain(_LATEST_TRACE)
