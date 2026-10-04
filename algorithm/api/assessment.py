"""
评估层路由 —— 心理状态评估 API 端点

提供多模态→量表分值自动映射、风险预测和趋势分析接口。
包含个人基线同步与偏离计算接口。
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
from assessment.personal_baseline import (
    sync_baseline,
    compute_deviation,
    get_baseline,
    PersonalBaseline,
    BaselineDeviation,
)
from assessment.comorbidity import analyze_comorbidity as _analyze_comorbidity
from perception.perception_service import get_perception_service

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


@router.api_route("/risk-trend", methods=["GET", "POST"], response_model=RiskTrendResponse,
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


# ============================================================
# 个人基线接口
# ============================================================

class BaselineSyncRequest(BaseModel):
    """基线同步请求

    支持两种模式：
    1. 完整同步：发送 {"metrics": {...}, "emotion_baseline": [...], "sample_count": N}
    2. 增量更新：发送实时观测值 {"heartRate": 72, "breathingRate": 16, ...}
    """
    user_id: str = Field(..., description="用户唯一标识")
    baseline_data: dict = Field(..., description="基线数据（完整结构或增量观测）")


class BaselineDeviationRequest(BaseModel):
    """基线偏离计算请求"""
    user_id: str = Field(..., description="用户唯一标识")
    current_features: dict = Field(
        ...,
        description="当前各模态观测值，如 {\"heartRate\": 72, \"emotionProbs\": [0.1, 0.3, 0.4, 0.1, 0.1]}",
    )


class BaselineMetricResponse(BaseModel):
    """单模态基线指标响应"""
    mean: float
    std: float
    n: int


class PersonalBaselineResponse(BaseModel):
    """个人基线响应"""
    user_id: str
    metrics: dict[str, BaselineMetricResponse]
    emotion_baseline: list[float]
    created_at: float
    last_updated: float
    sample_count: int
    calibrated: bool


class BaselineDeviationResponse(BaseModel):
    """基线偏离响应"""
    user_id: str
    computed_at: float
    modality_z_scores: dict[str, float]
    emotion_z_scores: list[float]
    significant_deviations: list[str]
    confidence: float
    calibrated: bool


@router.post("/baseline/sync", response_model=PersonalBaselineResponse,
             summary="同步个人基线数据")
async def sync_baseline_endpoint(request: BaselineSyncRequest):
    """同步前端个人基线数据到后端

    支持两种模式：
    1. 完整同步：前端发送完整 PersonalBaseline 结构
    2. 增量更新：前端发送实时观测值，后端用 EMA 更新

    与前端 usePersonalBaseline.ts 的 EMA 算法保持一致（alpha=0.05）。
    """
    baseline = sync_baseline(request.user_id, request.baseline_data)
    config = __import__("assessment.personal_baseline", fromlist=["load_baseline_config"]).load_baseline_config()
    min_samples = config.get("calibration", {}).get("min_samples", 20)

    return PersonalBaselineResponse(
        user_id=baseline.user_id,
        metrics={
            k: BaselineMetricResponse(mean=v.mean, std=v.std, n=v.n)
            for k, v in baseline.metrics.items()
        },
        emotion_baseline=baseline.emotion_baseline,
        created_at=baseline.created_at,
        last_updated=baseline.last_updated,
        sample_count=baseline.sample_count,
        calibrated=baseline.sample_count >= min_samples,
    )


@router.post("/baseline/deviation", response_model=BaselineDeviationResponse,
             summary="计算基线偏离度")
async def compute_deviation_endpoint(request: BaselineDeviationRequest):
    """计算当前特征相对个人基线的 Z-score 偏离

    |z| > 2 表示显著偏离个人常态。
    返回各模态 Z-score、显著偏离维度列表、置信度。
    """
    deviation = compute_deviation(request.user_id, request.current_features)

    return BaselineDeviationResponse(
        user_id=deviation.user_id,
        computed_at=deviation.computed_at,
        modality_z_scores=deviation.modality_z_scores,
        emotion_z_scores=deviation.emotion_z_scores,
        significant_deviations=deviation.significant_deviations,
        confidence=deviation.confidence,
        calibrated=deviation.calibrated,
    )


@router.get("/baseline/{user_id}", response_model=PersonalBaselineResponse,
            summary="获取个人基线")
async def get_baseline_endpoint(user_id: str):
    """获取指定用户的个人基线数据

    返回完整的基线指标（6 个生理/行为模态 + 5 维情绪）。
    若无基线数据返回 404。
    """
    baseline = get_baseline(user_id)
    if baseline is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 无基线数据")

    from assessment.personal_baseline import load_baseline_config
    config = load_baseline_config()
    min_samples = config.get("calibration", {}).get("min_samples", 20)

    return PersonalBaselineResponse(
        user_id=baseline.user_id,
        metrics={
            k: BaselineMetricResponse(mean=v.mean, std=v.std, n=v.n)
            for k, v in baseline.metrics.items()
        },
        emotion_baseline=baseline.emotion_baseline,
        created_at=baseline.created_at,
        last_updated=baseline.last_updated,
        sample_count=baseline.sample_count,
        calibrated=baseline.sample_count >= min_samples,
    )


# ============================================================
# 共病模式分析接口
# ============================================================

class ComorbidityRequest(BaseModel):
    """共病分析请求"""
    user_id: str = Field(..., description="用户 ID")
    depression_prob: float = Field(..., ge=0.0, le=1.0, description="抑郁风险概率")
    anxiety_prob: float = Field(..., ge=0.0, le=1.0, description="焦虑风险概率")
    sleep_prob: float = Field(..., ge=0.0, le=1.0, description="睡眠风险概率")


class ComorbidityResponse(BaseModel):
    """共病分析响应"""
    user_id: str
    comorbidity_type: str
    comorbidity_name: str
    depression_prob: float
    anxiety_prob: float
    sleep_prob: float
    risk_combinations: list[str]
    severity_score: float
    recommendation: str
    timestamp: float


@router.post("/comorbidity/analyze", response_model=ComorbidityResponse,
             summary="共病模式分析")
async def comorbidity_analyze(request: ComorbidityRequest) -> ComorbidityResponse:
    """分析用户的共病模式

    基于多任务模型输出的三个风险概率，判定共病类型：
    - 抑郁 + 焦虑（最常见）
    - 抑郁 + 睡眠障碍
    - 焦虑 + 睡眠障碍
    - 三重共病
    """
    from assessment.comorbidity import COMORBIDITY_NAMES

    result = _analyze_comorbidity(
        user_id=request.user_id,
        depression_prob=request.depression_prob,
        anxiety_prob=request.anxiety_prob,
        sleep_prob=request.sleep_prob,
    )

    return ComorbidityResponse(
        user_id=result.user_id,
        comorbidity_type=result.comorbidity_type.value,
        comorbidity_name=COMORBIDITY_NAMES[result.comorbidity_type],
        depression_prob=result.depression_prob,
        anxiety_prob=result.anxiety_prob,
        sleep_prob=result.sleep_prob,
        risk_combinations=result.risk_combinations,
        severity_score=result.severity_score,
        recommendation=result.recommendation,
        timestamp=result.timestamp,
    )


# ============================================================
# 便捷文本风险评估接口（供 Node.js 桥接层调用）
# ============================================================

class TextPredictRequest(BaseModel):
    """文本风险评估请求（供桥接层调用）"""
    user_id: str = Field(..., description="用户 ID")
    text: str = Field(..., description="待评估文本")


class TextPredictResponse(BaseModel):
    """文本风险评估响应"""
    phq9_estimated: list[float]
    gad7_estimated: list[float]
    risk_level: str
    confidence: float
    evidence: list[dict]
    emotion_probs: list[float]


@router.post("/predict", response_model=TextPredictResponse,
             summary="文本风险快速评估")
async def text_predict(request: TextPredictRequest) -> TextPredictResponse:
    """文本风险快速评估 —— 感知 + 量表映射一步完成

    供 Node.js 桥接层直接调用，输入文本即出风险评估。
    内部流程：文本感知 → 情绪概率 → 量表映射 → 风险等级
    """
    # 步骤 1：文本情感感知
    perception = get_perception_service()
    emotion_result = perception.analyze_text(request.text)

    # 步骤 2：量表映射
    scale_results = map_all(emotion_result)

    # 提取 PHQ-9 和 GAD-7 估值
    phq9 = scale_results.get("PHQ-9")
    gad7 = scale_results.get("GAD-7")

    phq9_estimated = [
        phq9.total_score if phq9 else 0,
        phq9.max_score if phq9 else 27,
    ]
    gad7_estimated = [
        gad7.total_score if gad7 else 0,
        gad7.max_score if gad7 else 21,
    ]

    # 推断风险等级
    risk_level = "low"
    for scale_name, result in scale_results.items():
        severity = result.severity if hasattr(result, 'severity') else "mild"
        if severity in ("moderate", "severe"):
            risk_level = "high"
        elif severity == "mild" and risk_level != "high":
            risk_level = "medium"

    # 构建证据列表
    evidence = []
    for scale_name, result in scale_results.items():
        evidence.append({
            "source": f"assessment.{scale_name}",
            "description": f"{scale_name}: {result.severity} (score={result.total_score})",
            "weight": emotion_result.confidence,
        })

    return TextPredictResponse(
        phq9_estimated=phq9_estimated,
        gad7_estimated=gad7_estimated,
        risk_level=risk_level,
        confidence=emotion_result.confidence,
        evidence=evidence,
        emotion_probs=emotion_result.text_emotion_probs,
    )
