from typing import Literal, Optional

from pydantic import BaseModel, Field


FatigueLevel = Literal["low", "medium", "high"]
Difficulty = Literal["easy", "normal", "hard"]


class Agent2Request(BaseModel):
    user_mood_input: str = Field(default="", description="사용자가 입력한 현재 기분/컨디션")
    user_situation_input: str = Field(default="", description="사용자가 입력한 현재 상황")


class Agent2Response(BaseModel):
    mood: str = Field(default="", description="표준화된 기분")
    situation: str = Field(default="", description="표준화된 현재 상황")
    fatigue_level: FatigueLevel = Field(default="medium", description="피로도")
    difficulty: Difficulty = Field(default="normal", description="권장 조리 난이도")
    preferred_taste: str = Field(default="", description="추천할 맛")
    preferred_cooking_method: str = Field(default="", description="추천 조리 방식")
    cooking_time_limit_minutes: Optional[int] = Field(default=None, description="권장 최대 조리 시간(분)")
