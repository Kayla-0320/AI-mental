"""
推理接口 —— predict(features) -> RiskAssessment

将多任务模型的输出转换为共享数据类 RiskAssessment，
供下游干预层和升级层消费。

使用方式：
    from assessment.multi_task.predictor import predict

    result = predict(features_array)
    # result: RiskAssessment
"""
from __future__ import annotations

import functools
from typing import Optional

import numpy as np

from assessment.multi_task.config import (
    MODEL_SAVE_PATH,
    TASK_NAMES,
)
from assessment.multi_task.model import MultiTaskModel
from shared.dataclasses import EvidenceItem, RiskAssessment, RiskLevel


# ============================================================
# 风险等级阈值
# 来源：参考 PHQ-9 临床分级标准（Kroenke et al., 2001）
# ============================================================

# 多任务综合风险评分 → 风险等级映射
# 综合评分 = max(抑郁概率, 焦虑概率, 睡眠概率)
# - 0.0 ~ 0.3: low    正常情绪波动
# - 0.3 ~ 0.6: medium 需要关注，建议轻度干预
# - 0.6 ~ 0.8: high   建议专业评估（必须触发升级流程）
# - 0.8 ~ 1.0: crisis 立即触发紧急协议

def _score_to_risk_level(score: float) -> RiskLevel:
    """综合风险评分 → 风险等级"""
    if score >= 0.8:
        return RiskLevel.CRISIS
    elif score >= 0.6:
        return RiskLevel.HIGH
    elif score >= 0.3:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


# ============================================================
# PHQ-9 / GAD-7 预估
# ============================================================

def _estimate_phq9(depression_prob: float) -> tuple[float, float]:
    """抑郁概率 → PHQ-9 预估分数区间

    基于线性映射：prob * 27（PHQ-9 满分）
    返回 (下界, 上界) 区间，±2 分不确定性
    """
    center = depression_prob * 27
    lower = max(0, center - 2)
    upper = min(27, center + 2)
    return (round(lower, 1), round(upper, 1))


def _estimate_gad7(anxiety_prob: float) -> tuple[float, float]:
    """焦虑概率 → GAD-7 预估分数区间

    基于线性映射：prob * 21（GAD-7 满分）
    """
    center = anxiety_prob * 21
    lower = max(0, center - 2)
    upper = min(21, center + 2)
    return (round(lower, 1), round(upper, 1))


# ============================================================
# 模型加载缓存
# ============================================================

@functools.lru_cache(maxsize=1)
def _load_model() -> MultiTaskModel:
    """加载训练好的多任务模型（仅加载一次）"""
    return MultiTaskModel.load(MODEL_SAVE_PATH)


# ============================================================
# 预测接口
# ============================================================

def predict(
    features: np.ndarray,
    model: Optional[MultiTaskModel] = None,
) -> RiskAssessment:
    """多任务风险预测

    输入感知层特征，输出统一的 RiskAssessment 数据结构，
    包含抑郁/焦虑/睡眠三任务的联合评估结果。

    Args:
        features: 感知层输出特征向量，形状 (feature_dim,) 或 (n_samples, feature_dim)
        model: 模型实例（可选，默认使用缓存的全局单例）

    Returns:
        RiskAssessment: 包含 PHQ-9/GAD-7 预估、风险等级、证据列表
    """
    if model is None:
        model = _load_model()

    # 确保 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)

    # 多任务推理
    proba_dict = model.predict_proba(features)

    # 取第一个样本（单条预测）
    dep_prob = float(proba_dict["depression"][0])
    anx_prob = float(proba_dict["anxiety"][0])
    slp_prob = float(proba_dict["sleep"][0])

    # 综合风险评分 = 三任务最大概率
    risk_score = max(dep_prob, anx_prob, slp_prob)
    risk_level = _score_to_risk_level(risk_score)

    # 置信度 = 1 - 三任务概率的方差（一致性越高越可信）
    probs = np.array([dep_prob, anx_prob, slp_prob])
    confidence = float(1.0 - np.std(probs))
    confidence = max(0.0, min(1.0, confidence))

    # 构建证据列表
    evidence = []
    if dep_prob > 0.3:
        evidence.append(EvidenceItem(
            source="assessment.multi_task.depression",
            description=f"抑郁风险概率: {dep_prob:.2f}",
            weight=dep_prob,
        ))
    if anx_prob > 0.3:
        evidence.append(EvidenceItem(
            source="assessment.multi_task.anxiety",
            description=f"焦虑风险概率: {anx_prob:.2f}",
            weight=anx_prob,
        ))
    if slp_prob > 0.3:
        evidence.append(EvidenceItem(
            source="assessment.multi_task.sleep",
            description=f"睡眠风险概率: {slp_prob:.2f}",
            weight=slp_prob,
        ))
    if not evidence:
        evidence.append(EvidenceItem(
            source="assessment.multi_task",
            description="各维度风险概率均低于阈值",
            weight=risk_score,
        ))

    return RiskAssessment(
        phq9_estimated=_estimate_phq9(dep_prob),
        gad7_estimated=_estimate_gad7(anx_prob),
        risk_level=risk_level,
        confidence=round(confidence, 4),
        evidence=evidence,
    )


def predict_batch(
    features: np.ndarray,
    model: Optional[MultiTaskModel] = None,
) -> list[RiskAssessment]:
    """批量预测

    Args:
        features: 特征矩阵 (n_samples, feature_dim)
        model: 模型实例

    Returns:
        RiskAssessment 列表
    """
    if model is None:
        model = _load_model()

    if features.ndim == 1:
        features = features.reshape(1, -1)

    results = []
    for i in range(len(features)):
        results.append(predict(features[i], model))
    return results
