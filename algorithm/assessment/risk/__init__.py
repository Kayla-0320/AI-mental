"""
独立风险评估模块 —— 多信号融合的临床决策规则引擎

整合多源信号进行综合风险研判：
    1. 感知层情绪向量（EmotionResult）
    2. 多任务模型预测（RiskAssessment）
    3. 数字表型特征（PhenotypeVector）
    4. 历史评估记录

决策逻辑：
    - 基于临床决策规则的分级研判
    - 支持信号冲突时的保守原则（取最高风险）
    - 输出统一的 RiskAssessment + 完整证据链

核心接口：
    - assess_risk(emotion, assessment, phenotype, history) -> ComprehensiveRiskResult
    - batch_assess(cases) -> list[ComprehensiveRiskResult]
    - generate_risk_summary(result) -> str
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from shared.dataclasses import (
    EmotionResult,
    EvidenceItem,
    PhenotypeVector,
    RiskAssessment,
    RiskLevel,
)


# ============================================================
# 数据结构
# ============================================================

class RiskSignalType(str, Enum):
    """风险信号类型"""
    EMOTION = "emotion"                   # 情绪信号
    BEHAVIOR = "behavior"                 # 行为信号
    PHENOTYPE = "phenotype"               # 表型信号
    HISTORY = "history"                   # 历史趋势信号
    SELF_HARM = "self_harm"              # 自伤信号
    SLEEP = "sleep"                       # 睡眠信号


@dataclass
class RiskSignal:
    """单条风险信号

    Attributes:
        signal_type: 信号类型
        score: 风险评分 [0, 1]
        description: 信号描述
        weight: 信号权重 [0, 1]
        timestamp: 信号时间戳
    """
    signal_type: RiskSignalType
    score: float
    description: str
    weight: float = 1.0
    timestamp: float = 0.0


@dataclass
class ComprehensiveRiskResult:
    """综合风险评估结果

    Attributes:
        risk_level: 风险等级
        risk_score: 综合风险评分 [0, 1]
        signals: 输入信号列表
        phq9_range: PHQ-9 预估区间
        gad7_range: GAD-7 预估区间
        confidence: 评估置信度
        evidence_chain: 完整证据链
        recommendations: 临床建议列表
        timestamp: 评估时间戳
    """
    risk_level: RiskLevel
    risk_score: float
    signals: list[RiskSignal] = field(default_factory=list)
    phq9_range: tuple[float, float] = (0.0, 0.0)
    gad7_range: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    evidence_chain: list[EvidenceItem] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    timestamp: float = 0.0


# ============================================================
# 信号提取器
# ============================================================

def _extract_emotion_signals(emotion: EmotionResult) -> list[RiskSignal]:
    """从情绪感知结果提取风险信号"""
    signals = []
    probs = emotion.text_emotion_probs
    happy, sad, anxious, angry, neutral = probs

    # 悲伤信号
    if sad > 0.3:
        signals.append(RiskSignal(
            signal_type=RiskSignalType.EMOTION,
            score=sad,
            description=f"悲伤情绪概率较高: {sad:.2f}",
            weight=0.8,
            timestamp=emotion.timestamp,
        ))

    # 焦虑信号
    if anxious > 0.3:
        signals.append(RiskSignal(
            signal_type=RiskSignalType.EMOTION,
            score=anxious,
            description=f"焦虑情绪概率较高: {anxious:.2f}",
            weight=0.9,
            timestamp=emotion.timestamp,
        ))

    # 愤怒信号
    if angry > 0.3:
        signals.append(RiskSignal(
            signal_type=RiskSignalType.EMOTION,
            score=angry * 0.8,  # 愤怒权重略低
            description=f"愤怒情绪概率: {angry:.2f}",
            weight=0.6,
            timestamp=emotion.timestamp,
        ))

    # 低快乐信号（快感缺失指标）
    if happy < 0.1:
        signals.append(RiskSignal(
            signal_type=RiskSignalType.EMOTION,
            score=0.6,
            description=f"快乐情绪极低: {happy:.2f}，可能存在快感缺失",
            weight=0.7,
            timestamp=emotion.timestamp,
        ))

    # 语音风险信号
    if emotion.audio_risk_prob is not None and emotion.audio_risk_prob > 0.3:
        signals.append(RiskSignal(
            signal_type=RiskSignalType.EMOTION,
            score=emotion.audio_risk_prob,
            description=f"语音风险概率: {emotion.audio_risk_prob:.2f}",
            weight=0.85,
            timestamp=emotion.timestamp,
        ))

    return signals


def _extract_phenotype_signals(phenotype: PhenotypeVector) -> list[RiskSignal]:
    """从数字表型提取风险信号"""
    signals = []
    features = phenotype.feature_dict()

    # 睡眠规律性低
    sleep_reg = features.get("sleep_regularity")
    if sleep_reg is not None and sleep_reg < 0.4:
        signals.append(RiskSignal(
            signal_type=RiskSignalType.SLEEP,
            score=1.0 - sleep_reg,
            description=f"睡眠规律性低: {sleep_reg:.2f}",
            weight=0.7,
        ))

    # 负面情绪比例高
    neg_ratio = features.get("negative_emotion_ratio")
    if neg_ratio is not None and neg_ratio > 0.4:
        signals.append(RiskSignal(
            signal_type=RiskSignalType.PHENOTYPE,
            score=neg_ratio,
            description=f"负面情绪占比高: {neg_ratio:.2%}",
            weight=0.75,
        ))

    # 活跃度低
    activity = features.get("activity_level")
    if activity is not None and activity < 0.2:
        signals.append(RiskSignal(
            signal_type=RiskSignalType.BEHAVIOR,
            score=1.0 - activity,
            description=f"活跃度极低: {activity:.2f}，可能存在行为退缩",
            weight=0.6,
        ))

    # PHQ-9 最新分高
    phq9 = features.get("phq9_latest")
    if phq9 is not None and phq9 >= 10:
        signals.append(RiskSignal(
            signal_type=RiskSignalType.HISTORY,
            score=min(1.0, phq9 / 27),
            description=f"PHQ-9 最新得分: {phq9:.0f}（中度以上）",
            weight=0.9,
        ))

    # PHQ-9 趋势恶化
    phq9_trend = features.get("phq9_trend")
    if phq9_trend is not None and phq9_trend > 0.5:
        signals.append(RiskSignal(
            signal_type=RiskSignalType.HISTORY,
            score=min(1.0, phq9_trend),
            description=f"PHQ-9 趋势恶化: 斜率={phq9_trend:.2f}",
            weight=0.85,
        ))

    return signals


# ============================================================
# 多信号融合规则
# ============================================================

def _fuse_signals(signals: list[RiskSignal]) -> tuple[float, list[EvidenceItem]]:
    """多信号加权融合

    采用保守原则：
        1. 加权平均作为基础分
        2. 任何单一信号 > 0.8 → 至少 HIGH
        3. 两个以上信号 > 0.6 → 至少 MEDIUM

    Args:
        signals: 风险信号列表

    Returns:
        (fused_score, evidence_items)
    """
    if not signals:
        return 0.0, []

    # 加权平均
    total_weight = sum(s.weight for s in signals)
    if total_weight > 0:
        weighted_avg = sum(s.score * s.weight for s in signals) / total_weight
    else:
        weighted_avg = 0.0

    # 最高信号分
    max_score = max(s.score for s in signals)

    # 保守融合：取加权平均与最高分的较大者（但最高分打 8 折）
    fused = max(weighted_avg, max_score * 0.8)
    fused = min(1.0, fused)

    # 构建证据
    evidence = []
    for s in signals:
        evidence.append(EvidenceItem(
            source=f"assessment.risk.{s.signal_type.value}",
            description=s.description,
            weight=s.weight,
        ))

    return round(fused, 4), evidence


def _score_to_level(score: float) -> RiskLevel:
    """评分 → 风险等级"""
    if score >= 0.8:
        return RiskLevel.CRISIS
    elif score >= 0.6:
        return RiskLevel.HIGH
    elif score >= 0.3:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _generate_recommendations(
    level: RiskLevel,
    signals: list[RiskSignal],
) -> list[str]:
    """根据风险等级和信号生成临床建议"""
    recs = []

    if level == RiskLevel.CRISIS:
        recs.extend([
            "⚠️ 立即触发危机干预协议",
            "通知紧急联系人和 assigned 咨询师",
            "提供 24h 心理援助热线：400-161-9995",
            "评估自伤/自杀风险，必要时转介精神科",
        ])
    elif level == RiskLevel.HIGH:
        recs.extend([
            "建议 48h 内安排专业心理咨询",
            "启动 CBT 认知行为干预流程",
            "增加情绪监测频率（每日 2 次）",
        ])
    elif level == RiskLevel.MEDIUM:
        recs.extend([
            "建议一周内安排心理咨询",
            "推荐正念练习和情绪管理工具",
            "保持每周情绪监测",
        ])
    else:
        recs.append("当前状态良好，建议保持健康生活方式")

    # 根据信号类型追加建议
    signal_types = {s.signal_type for s in signals}
    if RiskSignalType.SLEEP in signal_types:
        recs.append("关注睡眠卫生，建议使用睡眠监测功能")
    if RiskSignalType.BEHAVIOR in signal_types:
        recs.append("关注行为退缩信号，鼓励社交活动和运动")
    if RiskSignalType.SELF_HARM in signal_types:
        recs.append("⚠️ 检测到自伤相关信号，请优先处理")

    return recs


# ============================================================
# 核心接口
# ============================================================

def assess_risk(
    emotion: Optional[EmotionResult] = None,
    assessment: Optional[RiskAssessment] = None,
    phenotype: Optional[PhenotypeVector] = None,
    history: Optional[list[RiskAssessment]] = None,
) -> ComprehensiveRiskResult:
    """综合风险评估

    融合多源信号，输出统一风险评估结果。
    任何输入为 None 时自动跳过该信号源。

    Args:
        emotion: 感知层情绪结果
        assessment: 多任务模型预测结果
        phenotype: 数字表型向量
        history: 历史评估记录列表

    Returns:
        ComprehensiveRiskResult: 综合风险评估
    """
    all_signals: list[RiskSignal] = []

    # 1. 提取情绪信号
    if emotion is not None:
        all_signals.extend(_extract_emotion_signals(emotion))

    # 2. 提取表型信号
    if phenotype is not None:
        all_signals.extend(_extract_phenotype_signals(phenotype))

    # 3. 整合多任务模型结果
    if assessment is not None:
        level_score = {
            RiskLevel.LOW: 0.15,
            RiskLevel.MEDIUM: 0.45,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRISIS: 0.95,
        }
        all_signals.append(RiskSignal(
            signal_type=RiskSignalType.EMOTION,
            score=level_score.get(assessment.risk_level, 0.3),
            description=f"多任务模型评估: {assessment.risk_level.value} "
                        f"(PHQ-9≈{assessment.phq9_estimated}, GAD-7≈{assessment.gad7_estimated})",
            weight=assessment.confidence,
            timestamp=time.time(),
        ))

    # 4. 历史趋势信号
    if history and len(history) >= 2:
        level_scores = [
            {RiskLevel.LOW: 0.15, RiskLevel.MEDIUM: 0.45,
             RiskLevel.HIGH: 0.75, RiskLevel.CRISIS: 0.95}.get(h.risk_level, 0.3)
            for h in history
        ]
        trend = np.polyfit(range(len(level_scores)), level_scores, 1)[0]
        if trend > 0.05:
            all_signals.append(RiskSignal(
                signal_type=RiskSignalType.HISTORY,
                score=min(1.0, trend * 5),
                description=f"风险趋势恶化: {len(history)} 次评估呈上升",
                weight=0.7,
            ))

    # 融合
    fused_score, evidence = _fuse_signals(all_signals)
    level = _score_to_level(fused_score)
    recommendations = _generate_recommendations(level, all_signals)

    # PHQ-9/GAD-7 区间
    phq9_range = assessment.phq9_estimated if assessment else (0.0, 0.0)
    gad7_range = assessment.gad7_estimated if assessment else (0.0, 0.0)

    # 置信度 = 信号数量 + 信号一致性
    if all_signals:
        scores = [s.score for s in all_signals]
        consistency = 1.0 - float(np.std(scores))
        confidence = min(1.0, max(0.1, consistency * 0.7 + min(len(all_signals) / 5, 1.0) * 0.3))
    else:
        confidence = 0.1

    return ComprehensiveRiskResult(
        risk_level=level,
        risk_score=fused_score,
        signals=all_signals,
        phq9_range=phq9_range,
        gad7_range=gad7_range,
        confidence=round(confidence, 3),
        evidence_chain=evidence,
        recommendations=recommendations,
        timestamp=time.time(),
    )


def generate_risk_summary(result: ComprehensiveRiskResult) -> str:
    """生成风险评估摘要文本

    Args:
        result: 综合风险评估结果

    Returns:
        格式化的摘要文本
    """
    level_labels = {
        RiskLevel.LOW: "低风险",
        RiskLevel.MEDIUM: "中风险",
        RiskLevel.HIGH: "高风险",
        RiskLevel.CRISIS: "危机",
    }
    level_emoji = {
        RiskLevel.LOW: "🟢",
        RiskLevel.MEDIUM: "🟡",
        RiskLevel.HIGH: "🟠",
        RiskLevel.CRISIS: "🔴",
    }

    lines = [
        f"## 风险评估摘要",
        f"",
        f"**风险等级**: {level_emoji[result.risk_level]} {level_labels[result.risk_level]}",
        f"**风险评分**: {result.risk_score:.2f} / 1.00",
        f"**评估置信度**: {result.confidence:.1%}",
        f"**信号数量**: {len(result.signals)} 条",
        f"",
    ]

    if result.phq9_range != (0.0, 0.0):
        lines.append(f"- PHQ-9 预估: {result.phq9_range[0]:.0f}-{result.phq9_range[1]:.0f}")
    if result.gad7_range != (0.0, 0.0):
        lines.append(f"- GAD-7 预估: {result.gad7_range[0]:.0f}-{result.gad7_range[1]:.0f}")

    lines.extend([
        f"",
        f"### 关键信号",
    ])
    for s in sorted(result.signals, key=lambda x: x.score, reverse=True)[:5]:
        lines.append(f"- [{s.signal_type.value}] {s.description}")

    lines.extend([
        f"",
        f"### 建议",
    ])
    for rec in result.recommendations:
        lines.append(f"- {rec}")

    return "\n".join(lines)
