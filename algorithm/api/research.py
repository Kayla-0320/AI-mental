"""隔离的第二轮研究接口；令牌未配置时拒绝运行研究模型。"""
from __future__ import annotations

import hmac
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from assessment.forecasting import ForecastPredictor
from shared import EmotionObservation, MultimodalFeatureBatch

router = APIRouter(prefix="/research", tags=["第二轮独立研究"])


class SourceTextInput(BaseModel):
    """本人消息出处契约，不能由其他用户ID替代实际文本证据。"""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4000)
    source_message_id: str = Field(min_length=1, max_length=160)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_at: str = Field(min_length=1, max_length=64)
    language: Literal["zh"] = "zh"


class SourceObservationInput(BaseModel):
    """第三轮业务桥接，只接本人可核验的有界文本。"""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["emotion_observation.v1"]
    request_id: str = Field(min_length=1, max_length=100)
    observation: SourceTextInput


class StrategyInput(BaseModel):
    """描述性策略研究请求，未经验证的语言显式拒绝推断。"""

    model_config = ConfigDict(extra="forbid")
    messages: list[dict[str, str]] = Field(min_length=1, max_length=64)
    language: Literal["zh", "en"] = "en"


class ObservationInput(BaseModel):
    """与自评量表匹配的输入，不能混入分类概率。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    step: int = Field(strict=True, gt=0)
    positive: float = Field(strict=True, ge=0, le=100)
    negative: float = Field(strict=True, ge=0, le=100)
    source: str = "self_report"


class ForecastInput(BaseModel):
    """上限有界且只有过去观测的预测请求。"""

    model_config = ConfigDict(extra="forbid")
    history: list[ObservationInput] = Field(min_length=8, max_length=512)
    horizon: int = Field(default=1, strict=True, ge=1, le=3)


class FusionInput(BaseModel):
    """有界的离线冻结特征，携带完整任务和编码器身份。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sample_ids: list[str] = Field(min_length=1, max_length=64)
    modalities: dict[str, list[list[float]]]
    observed: dict[str, list[bool]]
    quality: dict[str, list[float]]
    task: str = Field(min_length=1, max_length=128)
    class_names: list[str] = Field(min_length=2, max_length=32)
    encoder_fingerprints: dict[str, str]
    raw_text: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def validate_size_and_values(self) -> FusionInput:
        """拒绝无界向量、不对齐样本和非有限/越界质量。"""
        import math

        count = len(self.sample_ids)
        if self.raw_text and (len(self.raw_text) != count or any(len(text) > 4000 for text in self.raw_text)):
            raise ValueError("知识原文必须与样本对齐且每条不超过4000字符")
        names = set(self.modalities)
        if not 1 <= len(names) <= 4 or any(len(key) > 64 for key in names):
            raise ValueError("需要1至4个有界模态")
        if not names == set(self.observed) == set(self.quality) == set(self.encoder_fingerprints):
            raise ValueError("模态、掩码、质量和编码器名称不一致")
        if any(not sample_id or len(sample_id) > 256 for sample_id in self.sample_ids):
            raise ValueError("非法来源样本ID")
        if any(len(name) > 128 for name in self.class_names) or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("类别名称必须有界且唯一")
        for name, rows in self.modalities.items():
            if len(rows) != count or any(not 1 <= len(row) <= 4096 for row in rows):
                raise ValueError("每个模态必须含相同样本数且特征宽度为1至4096")
            if any(not math.isfinite(value) for row in rows for value in row):
                raise ValueError("API 特征必须有限；离线质量模块应先审计无效值")
            if len(self.observed[name]) != count or len(self.quality[name]) != count:
                raise ValueError("观测掩码及质量未与样本对齐")
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in self.quality[name]):
                raise ValueError("质量必须是0至1的有限数")
            if not self.encoder_fingerprints[name] or len(self.encoder_fingerprints[name]) > 256:
                raise ValueError("编码器指纹无效")
        return self


def require_research_token(authorization: str | None = Header(default=None)) -> None:
    """独立研究令牌通过环境配置，缺失时不开放推理。"""
    expected = os.environ.get("ROUND2_RESEARCH_TOKEN", "")
    if not expected:
        raise HTTPException(503, "研究接口未启用")
    supplied = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
    if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(401, "研究接口令牌无效")


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    """声明研究输出的任务边界，不将文件存在等同于效果验证。"""
    return {
        "version": "round3.v1", "forecast_task": "positive_negative_self_report_0_100",
        "forecast_horizons": [1, 3], "time_unit": "ordinal_sampling_step",
        "fusion_task": "independent_source_sentiment_labels_not_patient_five_emotions",
        "source_observation_task": "sentiment_polarity_3",
        "strategy_task": "descriptive_next_support_strategy",
        "strategy_supported_languages": ["en"],
        "research_inference_enabled": bool(os.environ.get("ROUND2_RESEARCH_TOKEN")),
        "clinical_effectiveness_validated": False,
        "automatic_safety_rule_changes": False,
    }


@router.post("/forecast", dependencies=[Depends(require_research_token)])
def forecast(body: ForecastInput) -> dict[str, Any]:
    """返回短期自评预测；不写用户档案、不驱动升级或干预决策。"""
    artifact = Path(os.environ.get("ROUND2_FORECAST_ARTIFACT", "")) if os.environ.get(
        "ROUND2_FORECAST_ARTIFACT") else Path(__file__).resolve().parents[1] / "artifacts/round2_forecast/forecast.json"
    try:
        predictor = ForecastPredictor(artifact)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(503, "研究产物缺失或校验未通过") from exc
    try:
        result = predictor.predict([EmotionObservation(**point.model_dump()) for point in body.history], body.horizon)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return asdict(result)


@router.post("/fusion", dependencies=[Depends(require_research_token)])
def fusion(body: FusionInput) -> dict[str, Any]:
    """执行有指纹约束的研究融合头，不改线上五类情绪和安全工作点。"""
    from perception.knowledge_moe.runtime import FusionRuntime

    default = Path(__file__).resolve().parents[1] / "artifacts/round2_fusion"
    directory = Path(os.environ.get("ROUND2_FUSION_ARTIFACT", str(default)))
    try:
        runtime = FusionRuntime(directory)
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as exc:
        raise HTTPException(503, "融合研究产物缺失或校验未通过") from exc
    try:
        result = runtime.predict(MultimodalFeatureBatch(**body.model_dump()))
    except (ValueError, TypeError, RuntimeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return asdict(result)


@router.post("/observation", dependencies=[Depends(require_research_token)])
def source_observation(body: SourceObservationInput) -> dict[str, Any]:
    """实际编码器和研究头前向，缺证据时返回明确不可用。"""
    from perception.knowledge_moe.observation import infer_observation

    point = body.observation
    try:
        result = infer_observation(point.text, point.source_message_id, point.source_sha256, point.observed_at)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return asdict(result)


@router.post("/strategy", dependencies=[Depends(require_research_token)])
def support_strategy(body: StrategyInput) -> dict[str, Any]:
    """返回数据集中下一支持策略的描述分布，不直接驱动工具。"""
    from intervention.strategy_learning import SupportStrategyModel

    path = Path(os.environ.get("ROUND3_STRATEGY_ARTIFACT", str(Path(__file__).resolve().parents[1] / "artifacts/round3_strategy")))
    try:
        model = SupportStrategyModel(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(503, "支持策略研究产物不可用") from exc
    try:
        return asdict(model.predict(body.messages, body.language))
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
