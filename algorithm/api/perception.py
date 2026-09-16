"""
感知层路由 —— 多模态情绪分析 API 端点
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from perception.perception_service import get_perception_service
from shared.dataclasses import EmotionResult

router = APIRouter(prefix="/perception", tags=["感知层"])


# ----------------------------------------------------------
# 请求/响应模型
# ----------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """感知分析请求体"""
    text: str = Field(..., description="待分析的文本内容")
    wav_path: Optional[str] = Field(None, description="WAV 音频文件路径（可选）")
    facial_features: Optional[dict] = Field(None, description="面部特征字典（可选，含 AU 强度等）")
    behavior_features: Optional[dict] = Field(None, description="行为特征字典（可选，含活跃时段等）")


class AnalyzeResponse(BaseModel):
    """感知分析响应体"""
    text_emotion_probs: list[float] = Field(..., description="5维文本情绪概率分布")
    audio_risk_prob: Optional[float] = Field(None, description="语音风险概率")
    confidence: float = Field(..., description="综合置信度")
    timestamp: float = Field(..., description="Unix 时间戳")
    evidence: list[str] = Field(default_factory=list, description="证据描述列表")


def _emotion_result_to_response(result: EmotionResult) -> AnalyzeResponse:
    """将 EmotionResult 转换为 API 响应模型"""
    return AnalyzeResponse(
        text_emotion_probs=result.text_emotion_probs,
        audio_risk_prob=result.audio_risk_prob,
        confidence=result.confidence,
        timestamp=result.timestamp,
        evidence=result.evidence,
    )


# ----------------------------------------------------------
# 路由端点
# ----------------------------------------------------------

@router.post("/analyze", response_model=AnalyzeResponse, summary="多模态情感分析")
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """统一情感分析接口（支持文本/语音/面部/行为四模态融合）

    - 仅传 text → 文本单模态分析（使用真实 RoBERTa 模型）
    - 同时传 text + wav_path → 文本+语音融合
    - 同时传 text + facial_features → 文本+面部融合
    - 同时传 text + behavior_features → 文本+行为融合
    - 全部传入 → 四模态融合
    - 任何模态缺失时自动降级
    """
    service = get_perception_service()
    result = service.analyze_multimodal(
        text=request.text,
        wav_path=request.wav_path,
        facial_features=request.facial_features,
        behavior_features=request.behavior_features,
    )
    return _emotion_result_to_response(result)
