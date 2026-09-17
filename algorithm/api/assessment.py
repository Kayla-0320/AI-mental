"""
评估层路由 —— 心理状态评估 API 端点

提供多模态→量表分值自动映射、风险预测和趋势分析接口。
"""
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from shared.dataclasses import EmotionResult
from assessment.scale import (
    map_all, map_to_phq9, map_to_gad7, map_to_pss10,
    interpret_score, ScaleName,
    PHQ9_CUTOFFS, GAD7_CUTOFFS, PSS10_CUTOFFS,
)

router = APIRouter(prefix="/assessment", tags=["评估层"])


# ============================================================
# 请求/响应模型
# ============================================================

class EmotionInput(BaseModel):
    """情绪概率输入"""
    text_emotion_probs: list[float] = Field(
        ..., description="5维情绪概率 [快乐, 悲伤, 焦虑, 愤怒, 中性]",
        min_length=5, max_length=5,
    )
    audio_risk_prob: Optional[float] = Field(None, description="语音风险概率")
    confidence: float = Field(0.8, ge=0, le=1, description="感知置信度")


class ScaleEstimateRequest(BaseModel):
    """量表映射请求"""
    emotion: EmotionInput
    user_id: Optional[str] = Field(None, description="用户ID（用于表型特征补充）")


class ScaleItemResponse(BaseModel):
    item_number: int
    item_text: str
    score: int
    confidence: float


class ScaleResultResponse(BaseModel):
    scale_name: str
    total_score: float
    max_score: int
    severity: str
    confidence_interval: list[float]
    items: list[ScaleItemResponse]
    evidence: list[str]
    interpretation: Optional[str] = None
    percentile: Optional[float] = None


class ScaleEstimateResponse(BaseModel):
    """量表映射响应"""
    timestamp: float
    emotion_probs: list[float]
    confidence: float
    scales: dict[str, ScaleResultResponse]
    risk_level: str
    summary: str


class RiskTrendPoint(BaseModel):
    timestamp: float
    risk_score: float
    risk_level: str
    dominant_emotion: str
    evidence: list[str]


class RiskTrendResponse(BaseModel):
    """风险趋势响应"""
    user_id: str
    trend: list[RiskTrendPoint]
    current_risk: str
    trend_direction: str  # "improving" / "stable" / "worsening"


# ============================================================
# 量表映射接口
# ============================================================

@router.post("/scale-estimate", response_model=ScaleEstimateResponse,
             summary="多模态→量表分值自动映射")
async def estimate_scales(request: ScaleEstimateRequest):
    """将多模态感知输出自动映射为标准量表分值

    支持 PHQ-9（抑郁）、GAD-7（焦虑）、PSS-10（压力）三个量表。
    输出包含：总分、严重程度分级、置信区间、各条目得分、证据链。
    """
    emotion_input = request.emotion

    # 构建 EmotionResult
    emotion_result = EmotionResult(
        text_emotion_probs=emotion_input.text_emotion_probs,
        audio_risk_prob=emotion_input.audio_risk_prob,
        confidence=emotion_input.confidence,
        timestamp=time.time(),
        evidence=["前端多模态融合输出"],
    )

    # 执行量表映射
    scale_results = map_all(emotion_result)

    # 构建响应
    scales_response = {}
    emotion_labels = ['快乐', '悲伤', '焦虑', '愤怒', '中性']
    dominant_idx = emotion_input.text_emotion_probs.index(max(emotion_input.text_emotion_probs))

    for scale_name, result in scale_results.items():
        interpretation = interpret_score(
            ScaleName(scale_name), result.total_score
        )
        scales_response[scale_name] = ScaleResultResponse(
            scale_name=result.scale_name.value,
            total_score=result.total_score,
            max_score=result.max_score,
            severity=result.severity,
            confidence_interval=[result.confidence_interval[0], result.confidence_interval[1]],
            items=[
                ScaleItemResponse(
                    item_number=item.item_number,
                    item_text=item.item_text,
                    score=item.score,
                    confidence=item.confidence,
                )
                for item in result.items
            ],
            evidence=result.evidence,
            interpretation=interpretation.recommendation,
            percentile=interpretation.percentile,
        )

    # 综合风险等级判断
    phq9_score = scale_results[ScaleName.PHQ9.value].total_score
    gad7_score = scale_results[ScaleName.GAD7.value].total_score

    if phq9_score >= 20 or gad7_score >= 15:
        risk_level = "crisis"
    elif phq9_score >= 15 or gad7_score >= 10:
        risk_level = "high"
    elif phq9_score >= 10 or gad7_score >= 5:
        risk_level = "medium"
    else:
        risk_level = "low"

    # 生成摘要
    dominant_emotion = emotion_labels[dominant_idx]
    summary_parts = [
        f"主导情绪为{dominant_emotion}（概率{max(emotion_input.text_emotion_probs):.0%}）",
        f"PHQ-9 预估 {phq9_score} 分（{scales_response[ScaleName.PHQ9.value].severity}）",
        f"GAD-7 预估 {gad7_score} 分（{scales_response[ScaleName.GAD7.value].severity}）",
    ]
    pss10_score = scale_results[ScaleName.PSS10.value].total_score
    summary_parts.append(
        f"PSS-10 预估 {pss10_score} 分（{scales_response[ScaleName.PSS10.value].severity}）"
    )

    return ScaleEstimateResponse(
        timestamp=time.time(),
        emotion_probs=emotion_input.text_emotion_probs,
        confidence=emotion_input.confidence,
        scales=scales_response,
        risk_level=risk_level,
        summary="；".join(summary_parts) + "。",
    )


@router.post("/risk-trend", response_model=RiskTrendResponse,
             summary="风险趋势分析")
async def risk_trend(user_id: str, days: int = 7):
    """分析用户近期风险变化趋势

    返回最近 N 天的风险评分变化序列，用于咨询师咨询前查看患者风险趋势。
    """
    import numpy as np

    # 模拟历史风险数据（实际应从数据库获取）
    rng = np.random.RandomState(hash(user_id + "trend") % (2**31))
    emotion_labels = ['快乐', '悲伤', '焦虑', '愤怒', '中性']

    trend = []
    for i in range(days):
        # 生成模拟情绪概率
        probs = rng.dirichlet([2, 1.5, 1.8, 1.2, 3]).tolist()
        dominant_idx = probs.index(max(probs))
        dominant_emotion = emotion_labels[dominant_idx]

        # 简化风险评分（基于负面情绪比例）
        neg_ratio = probs[1] + probs[2] + probs[3]  # 悲伤+焦虑+愤怒
        risk_score = min(1.0, neg_ratio * 1.5)

        if risk_score > 0.7:
            level = "high"
        elif risk_score > 0.4:
            level = "medium"
        else:
            level = "low"

        trend.append(RiskTrendPoint(
            timestamp=time.time() - (days - i) * 86400,
            risk_score=round(risk_score, 3),
            risk_level=level,
            dominant_emotion=dominant_emotion,
            evidence=[f"当日主导情绪: {dominant_emotion}", f"负面情绪占比: {neg_ratio:.0%}"],
        ))

    # 判断趋势方向
    if len(trend) >= 3:
        recent = [t.risk_score for t in trend[-3:]]
        earlier = [t.risk_score for t in trend[:3]]
        avg_recent = sum(recent) / len(recent)
        avg_earlier = sum(earlier) / len(earlier)
        diff = avg_recent - avg_earlier
        if diff > 0.1:
            direction = "worsening"
        elif diff < -0.1:
            direction = "improving"
        else:
            direction = "stable"
    else:
        direction = "stable"

    return RiskTrendResponse(
        user_id=user_id,
        trend=trend,
        current_risk=trend[-1].risk_level if trend else "low",
        trend_direction=direction,
    )
