"""正式业务的创新能力入口；与研究令牌和研究路由解耦。"""
from __future__ import annotations

from dataclasses import asdict
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from perception.knowledge_moe.observation import infer_observation

router = APIRouter(prefix="/innovation", tags=["业务创新能力"])


class ObservationRequest(BaseModel):
    """仅接收有出处的本人文字；Node层负责鉴权和所有权校验。"""
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4000)
    source_message_id: str = Field(min_length=1, max_length=160)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_at: str = Field(min_length=1, max_length=64)


@router.post("/observation")
def observation(body: ObservationRequest) -> dict[str, Any]:
    """执行已部署文本编码器和校准头，保留出处校验与缺失弃权。"""
    if os.environ.get("INNOVATION_ENABLED", "1") == "0":
        raise HTTPException(503, "创新参考已关闭")
    try:
        return asdict(infer_observation(body.text, body.source_message_id, body.source_sha256, body.observed_at))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    """报告部署文件状态；是否可实际推理由单次返回available决定。"""
    from perception.innovation import ARTIFACTS
    from pathlib import Path
    folder = Path(os.environ.get("INNOVATION_FUSION_ARTIFACT", str(ARTIFACTS / "innovation_fusion")))
    return {"enabled": os.environ.get("INNOVATION_ENABLED", "1") != "0",
            "fusion_artifact_present": (folder / "manifest.json").is_file(),
            "quality_head_present": (ARTIFACTS / "quality_head" / "manifest.json").is_file(),
            "quality_guard_mode": os.environ.get("INNOVATION_QUALITY_GUARD", "shadow"),
            "tasks": ["sentiment_polarity_3", "raw_visual_quality", "personal_support"],
            "clinical_effectiveness_validated": False, "raw_media_persisted": False}
