"""
感知层路由 —— 多模态情绪分析 API 端点
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from perception.perception_service import get_perception_service, PerceptionService
from shared.dataclasses import EmotionResult
from perception.text_emotion.sprop_gnn import debias_predict, debias_predict_batch, get_debiaser
from perception.age_config import AgeGroup, get_age_profile, get_all_age_groups

router = APIRouter(prefix="/perception", tags=["感知层"])

# 按年龄组缓存 PerceptionService 实例，避免每次请求重复加载模型/初始化融合引擎
_service_cache: dict[str, PerceptionService] = {}


def _get_service_for_age(age_group) -> PerceptionService:
    """获取指定年龄组的 PerceptionService 缓存实例"""
    key = age_group.value if age_group is not None else "default"
    if key not in _service_cache:
        _service_cache[key] = PerceptionService(age_group=age_group)
    return _service_cache[key]


# ----------------------------------------------------------
# 请求/响应模型
# ----------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """感知分析请求体"""
    text: str = Field(..., description="待分析的文本内容")
    wav_path: Optional[str] = Field(None, description="WAV 音频文件路径（可选）")
    facial_features: Optional[dict] = Field(None, description="面部特征字典（可选，含 AU 强度等）")
    behavior_features: Optional[dict] = Field(None, description="行为特征字典（可选，含活跃时段等）")
    age_group: Optional[str] = Field(
        None,
        description="年龄分组: early_adolescent(12-15岁初中) / mid_adolescent(15-18岁高中) / young_adult(18-22岁大学)",
    )
    age: Optional[int] = Field(None, description="用户实际年龄（可选，自动推断年龄分组）")
    user_id: Optional[str] = Field(None, description="用户 ID（可选，用于更新心理数字孪生）")


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
    """统一情感分析接口（支持文本/语音/面部/行为四模态融合 + 年龄差异化校准）

    - 仅传 text → 文本单模态分析（使用真实 RoBERTa 模型）
    - 同时传 text + wav_path → 文本+语音融合
    - 同时传 text + facial_features → 文本+面部融合
    - 同时传 text + behavior_features → 文本+行为融合
    - 全部传入 → 四模态融合
    - 任何模态缺失时自动降级
    - 传入 age_group 或 age 启用年龄差异化校准
    """
    # 解析年龄画像
    age_profile = None
    if request.age_group is not None:
        try:
            ag = AgeGroup(request.age_group)
            age_profile = get_age_profile(ag)
        except ValueError:
            pass
    elif request.age is not None:
        from perception.age_config import get_age_profile_by_age
        age_profile = get_age_profile_by_age(request.age)

    # 使用带年龄画像的缓存 PerceptionService 实例（避免每次请求重复初始化）
    service = _get_service_for_age(age_profile.age_group if age_profile else None)
    result = service.analyze_multimodal(
        text=request.text,
        wav_path=request.wav_path,
        facial_features=request.facial_features,
        behavior_features=request.behavior_features,
        age_profile=age_profile,
        user_id=request.user_id,
    )
    return _emotion_result_to_response(result)


# ----------------------------------------------------------
# SProp GNN 语义盲法去偏 API
# ----------------------------------------------------------

class DebiasRequest(BaseModel):
    """去偏分析请求体"""
    text: str = Field(..., description="待分析文本")
    protected_attr: Optional[int] = Field(None, description="受保护属性 (0/1, -1=未知)")


class DebiasResponse(BaseModel):
    """去偏分析响应体"""
    text: str
    original_predictions: list[float]
    debiased_predictions: list[float]
    fairness_score: float
    bias_reduction: float
    graph_stats: dict
    latency_ms: float
    dominant_emotion: str
    dominant_score: float


class DebiasBatchRequest(BaseModel):
    """批量去偏请求体"""
    texts: list[str] = Field(..., description="待分析文本列表")


@router.post("/debias", response_model=DebiasResponse, summary="SProp GNN 语义盲法去偏分析")
async def debias(request: DebiasRequest) -> DebiasResponse:
    """SProp GNN 语义盲法去偏情绪预测

    通过图神经网络构建语义图，在保留情绪信息的同时去除受保护属性偏见。
    - 语义图构建：词级/句级节点 + 语义/位置/层级边
    - GCN 消息传递：3 层图卷积
    - 对抗去偏：梯度反转层最小化受保护属性预测
    - SProp 正则化：确保群体间预测比例一致
    """
    result = debias_predict(request.text, request.protected_attr or -1)
    EMOTION_LABELS = ['焦虑', '抑郁', '愤怒', '中性', '积极']
    import numpy as np
    debiased_arr = np.array(result.debiased_predictions)
    dominant_idx = int(np.argmax(debiased_arr))

    return DebiasResponse(
        text=result.text,
        original_predictions=result.original_predictions,
        debiased_predictions=result.debiased_predictions,
        fairness_score=result.fairness_score,
        bias_reduction=result.bias_reduction,
        graph_stats=result.graph_stats,
        latency_ms=result.latency_ms,
        dominant_emotion=EMOTION_LABELS[dominant_idx],
        dominant_score=round(float(debiased_arr[dominant_idx]), 4),
    )


@router.post("/debias/batch", summary="批量 SProp GNN 去偏分析")
async def debias_batch(request: DebiasBatchRequest):
    """批量去偏情绪预测"""
    results = debias_predict_batch(request.texts)
    EMOTION_LABELS = ['焦虑', '抑郁', '愤怒', '中性', '积极']
    import numpy as np

    items = []
    for r in results:
        debiased_arr = np.array(r.debiased_predictions)
        dominant_idx = int(np.argmax(debiased_arr))
        items.append({
            'text': r.text,
            'dominant_emotion': EMOTION_LABELS[dominant_idx],
            'dominant_score': round(float(debiased_arr[dominant_idx]), 4),
            'fairness_score': r.fairness_score,
            'bias_reduction': r.bias_reduction,
            'latency_ms': r.latency_ms,
        })

    return {'results': items, 'count': len(items)}


@router.get("/debias/stats", summary="SProp GNN 统计信息")
async def debias_stats():
    """获取去偏器统计信息"""
    return get_debiaser().get_stats()


# ----------------------------------------------------------
# 年龄分组 API
# ----------------------------------------------------------

@router.get("/age-groups", summary="获取年龄分组列表")
async def list_age_groups():
    """获取所有年龄分组的摘要信息

    返回各年龄段的分组名称、年龄范围、描述，
    供前端展示和选择使用。
    """
    return {"age_groups": get_all_age_groups()}
