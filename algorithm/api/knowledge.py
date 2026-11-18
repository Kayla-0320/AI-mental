"""知识图谱只读研究接口：默认关闭，返回条目仍为待审状态。"""
from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(prefix="/knowledge", tags=["知识图谱（研究）"])


class EvidenceInput(BaseModel):
    """只接受有明确目标与场景的检索请求，不允许把概念当个人结论。"""

    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    goal: Literal["listen", "understand", "act", "pause"] = "understand"
    setting: Literal["general_concept", "parent_child"] = "general_concept"
    research: bool = False
    include_unreviewed: bool = False
    budget: int = Field(default=4000, ge=100, le=8000)


def _enabled() -> bool:
    """默认关闭；只有显式设置环境变量才开放检索。"""
    return os.environ.get("SUPPORT_KNOWLEDGE_ENABLED", "0").strip() == "1"


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    """规模、待审数量与默认关闭状态；不返回任何概念正文。"""
    from perception.support_knowledge import capabilities as describe

    try:
        info = describe()
    except (OSError, ValueError) as exc:
        raise HTTPException(503, "知识图谱工件不可用") from exc
    return {**info, "enabled": _enabled()}


@router.post("/evidence")
def evidence(body: EvidenceInput) -> dict[str, Any]:
    """返回出处概念块；待审条目须同时显式 research 与 include_unreviewed。"""
    if not _enabled():
        raise HTTPException(503, "知识图谱研究接口默认关闭")
    from perception.support_knowledge import load

    try:
        return load().evidence(body.query, body.goal, setting=body.setting, budget=body.budget,
                               research=body.research, include_unreviewed=body.include_unreviewed)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
