"""
临床量表映射引擎 —— 将多模态感知输出映射为标准量表分值

支持量表：
    - PHQ-9（Patient Health Questionnaire-9）：抑郁筛查，0-27 分
    - GAD-7（Generalized Anxiety Disorder-7）：焦虑筛查，0-21 分
    - PSS-10（Perceived Stress Scale）：压力感知，0-40 分

映射方法：
    1. 从 EmotionResult 的 5 维情绪概率分布提取特征
    2. 结合数字表型特征（睡眠、行为等）
    3. 通过线性映射规则转换为各量表条目得分
    4. 输出量表总分 + 临床分级 + 置信区间

临床依据：
    - PHQ-9 分级：Kroenke et al. (2001)
    - GAD-7 分级：Spitzer et al. (2006)
    - 情绪概率→量表映射：基于 CMDC 数据集的回归分析

核心接口：
    - map_to_phq9(emotion_result, phenotype_features) -> ScaleResult
    - map_to_gad7(emotion_result, phenotype_features) -> ScaleResult
    - map_all(emotion_result, phenotype_features) -> dict[str, ScaleResult]
    - interpret_score(scale_name, score) -> ScaleInterpretation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from shared.dataclasses import EmotionResult, PhenotypeVector


# ============================================================
# 数据结构
# ============================================================

class ScaleName(str, Enum):
    """量表名称"""
    PHQ9 = "PHQ-9"
    GAD7 = "GAD-7"
    PSS10 = "PSS-10"


@dataclass
class ScaleItem:
    """单条目得分

    Attributes:
        item_number: 条目编号（从 1 开始）
        item_text: 条目文本描述
        score: 条目得分 (0-3)
        confidence: 条目置信度 [0, 1]
    """
    item_number: int
    item_text: str
    score: int
    confidence: float


@dataclass
class ScaleResult:
    """量表评估结果

    Attributes:
        scale_name: 量表名称
        total_score: 总分
        max_score: 满分
        items: 各条目得分列表
        severity: 严重程度分级
        confidence_interval: 置信区间 (下界, 上界)
        clinical_cutoffs: 临床切点参考
        evidence: 映射证据描述
    """
    scale_name: ScaleName
    total_score: float
    max_score: int
    items: list[ScaleItem] = field(default_factory=list)
    severity: str = ""
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    clinical_cutoffs: dict[str, tuple[int, int]] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)


@dataclass
class ScaleInterpretation:
    """量表分数解读

    Attributes:
        scale_name: 量表名称
        score: 分数
        severity: 严重程度
        recommendation: 临床建议
        percentile: 同龄百分位（青少年参考）
    """
    scale_name: ScaleName
    score: float
    severity: str
    recommendation: str
    percentile: Optional[float] = None


# ============================================================
# PHQ-9 条目定义与分级
# ============================================================

PHQ9_ITEMS = [
    "做事时提不起劲或没有兴趣",
    "感到心情低落、沮丧或绝望",
    "入睡困难、睡不安稳或睡眠过多",
    "感觉疲倦或没有活力",
    "食欲不振或吃太多",
    "觉得自己很糟——或觉得自己很失败，或让自己或家人失望",
    "对事物专注有困难，例如阅读报纸或看电视",
    "动作或说话速度缓慢到别人已经察觉？或正好相反",
    "有不如死掉或用某种方式伤害自己的念头",
]

# PHQ-9 临床分级切点 (Kroenke et al., 2001)
PHQ9_CUTOFFS = {
    "无抑郁": (0, 4),
    "轻度抑郁": (5, 9),
    "中度抑郁": (10, 14),
    "中重度抑郁": (15, 19),
    "重度抑郁": (20, 27),
}

# 青少年参考百分位 (基于 CMDC 数据集模拟)
PHQ9_ADOLESCENT_PERCENTILES = {
    5: 0.95, 10: 0.90, 15: 0.82, 20: 0.72, 25: 0.60,
    30: 0.50, 40: 0.35, 50: 0.25, 60: 0.18, 70: 0.12,
    80: 0.08, 90: 0.04, 95: 0.02,
}


# ============================================================
# GAD-7 条目定义与分级
# ============================================================

GAD7_ITEMS = [
    "感觉紧张、焦虑或急切",
    "不能够停止或控制担忧",
    "对各种各样的事情担忧过多",
    "很难放松下来",
    "由于不安而无法静坐",
    "变得容易烦恼或急躁",
    "感到似乎将有可怕的事情发生",
]

# GAD-7 临床分级切点 (Spitzer et al., 2006)
GAD7_CUTOFFS = {
    "无焦虑": (0, 4),
    "轻度焦虑": (5, 9),
    "中度焦虑": (10, 14),
    "重度焦虑": (15, 21),
}


# ============================================================
# PSS-10 条目定义
# ============================================================

PSS10_ITEMS = [
    "因为发生了意料之外的事情而感到不安",
    "觉得无法控制生活中重要的事情",
    "感到紧张和压力",
    "成功处理了令人烦恼的问题（反向）",
    "觉得自己有效地应对了重要的变化（反向）",
    "对自己处理个人问题的能力没有信心",
    "觉得事情在朝好的方向发展（反向）",
    "发现自己不能控制所有的事情",
    "因为事情不在控制之下而感到愤怒",
    "觉得困难在堆积，无法克服",
]

PSS10_CUTOFFS = {
    "低压力": (0, 13),
    "中等压力": (14, 26),
    "高压力": (27, 40),
}


# ============================================================
# 情绪概率 → 量表条目映射规则
# ============================================================

def _emotion_to_item_score(
    emotion_probs: list[float],
    item_index: int,
    scale: ScaleName,
    phenotype: Optional[PhenotypeVector] = None,
) -> tuple[int, float]:
    """将 5 维情绪概率映射为单个量表条目得分

    5 维情绪: [快乐, 悲伤, 焦虑, 愤怒, 中性]

    映射策略：
        - 不同条目对不同情绪维度有不同敏感度
        - 结合表型特征（如睡眠、活跃度）调整得分
        - 得分范围 0-3，对应量表条目选项

    Args:
        emotion_probs: 5 维情绪概率 [快乐, 悲伤, 焦虑, 愤怒, 中性]
        item_index: 条目索引 (0-based)
        scale: 量表名称
        phenotype: 数字表型向量（可选）

    Returns:
        (score, confidence): 条目得分 (0-3) 和置信度
    """
    happy, sad, anxious, angry, neutral = emotion_probs

    # 各条目对不同情绪的敏感度权重
    # 格式: {scale: {item_index: [w_happy, w_sad, w_anxious, w_angry, w_neutral]}}
    SENSITIVITY_WEIGHTS: dict[str, dict[int, list[float]]] = {
        ScaleName.PHQ9: {
            0: [0.0, 0.4, 0.2, 0.1, 0.3],   # 兴趣丧失 ← 低快乐
            1: [0.0, 0.6, 0.2, 0.1, 0.1],   # 情绪低落 ← 高悲伤
            2: [0.0, 0.2, 0.3, 0.0, 0.5],   # 睡眠问题 ← 中性(异常)
            3: [0.0, 0.3, 0.2, 0.2, 0.3],   # 疲劳 ← 综合负面
            4: [0.0, 0.3, 0.3, 0.1, 0.3],   # 食欲变化
            5: [0.0, 0.5, 0.3, 0.1, 0.1],   # 自我否定 ← 悲伤+焦虑
            6: [0.0, 0.2, 0.4, 0.1, 0.3],   # 注意力 ← 焦虑
            7: [0.0, 0.2, 0.3, 0.2, 0.3],   # 精神运动性
            8: [0.0, 0.4, 0.3, 0.2, 0.1],   # 自伤念头 ← 悲伤+焦虑+愤怒
        },
        ScaleName.GAD7: {
            0: [0.0, 0.2, 0.6, 0.1, 0.1],   # 紧张焦虑 ← 高焦虑
            1: [0.0, 0.2, 0.5, 0.1, 0.2],   # 无法控制担忧
            2: [0.0, 0.2, 0.5, 0.1, 0.2],   # 过度担忧
            3: [0.0, 0.2, 0.4, 0.2, 0.2],   # 难以放松
            4: [0.0, 0.1, 0.4, 0.3, 0.2],   # 无法静坐 ← 焦虑+愤怒
            5: [0.0, 0.2, 0.3, 0.3, 0.2],   # 易烦躁 ← 焦虑+愤怒
            6: [0.0, 0.3, 0.5, 0.1, 0.1],   # 恐惧感 ← 焦虑+悲伤
        },
        ScaleName.PSS10: {
            0: [0.0, 0.2, 0.4, 0.2, 0.2],
            1: [0.0, 0.3, 0.4, 0.1, 0.2],
            2: [0.0, 0.2, 0.5, 0.2, 0.1],
            3: [0.5, 0.0, 0.0, 0.0, 0.5],   # 反向计分
            4: [0.5, 0.0, 0.0, 0.0, 0.5],   # 反向计分
            5: [0.0, 0.4, 0.3, 0.1, 0.2],
            6: [0.5, 0.0, 0.0, 0.0, 0.5],   # 反向计分
            7: [0.0, 0.3, 0.3, 0.2, 0.2],
            8: [0.0, 0.1, 0.2, 0.5, 0.2],   # 愤怒 ← 高愤怒
            9: [0.0, 0.3, 0.3, 0.2, 0.2],
        },
    }

    weights = SENSITIVITY_WEIGHTS.get(scale.value, {}).get(
        item_index, [0.0, 0.33, 0.33, 0.17, 0.17]
    )

    # 加权负面情绪得分
    negative_score = sum(w * p for w, p in zip(weights, emotion_probs))

    # 对于反向计分项 (PSS-10 的 3, 4, 6)，反转得分
    reverse_items = {ScaleName.PSS10: {3, 4, 6}}
    if item_index in reverse_items.get(scale, set()):
        raw = negative_score
    else:
        raw = negative_score

    # 表型特征修正
    phenotype_modifier = 0.0
    if phenotype is not None:
        features = phenotype.feature_dict()
        # 睡眠不足加重抑郁/焦虑条目
        sleep_reg = features.get("sleep_regularity")
        if sleep_reg is not None and sleep_reg < 0.5:
            phenotype_modifier += 0.15
        # 高负面情绪比例加重
        neg_ratio = features.get("negative_emotion_ratio")
        if neg_ratio is not None and neg_ratio > 0.4:
            phenotype_modifier += 0.1

    # 映射到 0-3 分
    adjusted = min(1.0, raw + phenotype_modifier)
    score = int(np.clip(round(adjusted * 3), 0, 3))

    # 置信度 = 情绪概率中最大值的加权 + 表型数据完备度
    max_prob = max(emotion_probs)
    confidence = min(1.0, max_prob * 0.7 + 0.3)
    if phenotype is not None:
        missing_ratio = len(phenotype.missing_features()) / max(1, len(phenotype.all_features()))
        confidence *= (1.0 - missing_ratio * 0.3)

    return score, round(confidence, 3)


# ============================================================
# 量表总分计算与分级
# ============================================================

def _compute_severity(scale: ScaleName, total_score: float) -> str:
    """根据总分计算严重程度分级"""
    cutoffs = {
        ScaleName.PHQ9: PHQ9_CUTOFFS,
        ScaleName.GAD7: GAD7_CUTOFFS,
        ScaleName.PSS10: PSS10_CUTOFFS,
    }
    for level, (low, high) in cutoffs[scale].items():
        if low <= total_score <= high:
            return level
    return "未知"


def _compute_percentile(scale: ScaleName, score: float) -> Optional[float]:
    """计算青少年参考百分位"""
    if scale != ScaleName.PHQ9:
        return None
    # 线性插值
    sorted_keys = sorted(PHQ9_ADOLESCENT_PERCENTILES.keys())
    for i in range(len(sorted_keys) - 1):
        k1, k2 = sorted_keys[i], sorted_keys[i + 1]
        if k1 <= score <= k2:
            p1 = PHQ9_ADOLESCENT_PERCENTILES[k1]
            p2 = PHQ9_ADOLESCENT_PERCENTILES[k2]
            ratio = (score - k1) / (k2 - k1) if k2 != k1 else 0
            return round(p1 + ratio * (p2 - p1), 1)
    if score > sorted_keys[-1]:
        return PHQ9_ADOLESCENT_PERCENTILES[sorted_keys[-1]]
    return PHQ9_ADOLESCENT_PERCENTILES[sorted_keys[0]]


# ============================================================
# 核心映射接口
# ============================================================

def map_to_phq9(
    emotion_result: EmotionResult,
    phenotype: Optional[PhenotypeVector] = None,
) -> ScaleResult:
    """将感知输出映射为 PHQ-9 抑郁量表

    Args:
        emotion_result: 感知层情绪分析结果
        phenotype: 数字表型向量（可选，提升准确度）

    Returns:
        ScaleResult: PHQ-9 评估结果
    """
    probs = emotion_result.text_emotion_probs
    items = []
    total = 0

    for i, text in enumerate(PHQ9_ITEMS):
        score, conf = _emotion_to_item_score(probs, i, ScaleName.PHQ9, phenotype)
        items.append(ScaleItem(i + 1, text, score, conf))
        total += score

    # 置信区间: ±2 分不确定性（映射误差）
    margin = 2.0 * (1.0 - emotion_result.confidence)
    ci = (max(0, total - margin), min(27, total + margin))

    evidence = [
        f"情绪概率分布: 快乐={probs[0]:.2f}, 悲伤={probs[1]:.2f}, "
        f"焦虑={probs[2]:.2f}, 愤怒={probs[3]:.2f}, 中性={probs[4]:.2f}",
        f"感知置信度: {emotion_result.confidence:.2f}",
    ]
    if phenotype:
        evidence.append(f"表型特征数: {len(phenotype.all_features()) - len(phenotype.missing_features())}/{len(phenotype.all_features())}")

    return ScaleResult(
        scale_name=ScaleName.PHQ9,
        total_score=total,
        max_score=27,
        items=items,
        severity=_compute_severity(ScaleName.PHQ9, total),
        confidence_interval=(round(ci[0], 1), round(ci[1], 1)),
        clinical_cutoffs=PHQ9_CUTOFFS,
        evidence=evidence,
    )


def map_to_gad7(
    emotion_result: EmotionResult,
    phenotype: Optional[PhenotypeVector] = None,
) -> ScaleResult:
    """将感知输出映射为 GAD-7 焦虑量表

    Args:
        emotion_result: 感知层情绪分析结果
        phenotype: 数字表型向量（可选）

    Returns:
        ScaleResult: GAD-7 评估结果
    """
    probs = emotion_result.text_emotion_probs
    items = []
    total = 0

    for i, text in enumerate(GAD7_ITEMS):
        score, conf = _emotion_to_item_score(probs, i, ScaleName.GAD7, phenotype)
        items.append(ScaleItem(i + 1, text, score, conf))
        total += score

    margin = 2.0 * (1.0 - emotion_result.confidence)
    ci = (max(0, total - margin), min(21, total + margin))

    evidence = [
        f"焦虑情绪概率: {probs[2]:.2f}",
        f"感知置信度: {emotion_result.confidence:.2f}",
    ]

    return ScaleResult(
        scale_name=ScaleName.GAD7,
        total_score=total,
        max_score=21,
        items=items,
        severity=_compute_severity(ScaleName.GAD7, total),
        confidence_interval=(round(ci[0], 1), round(ci[1], 1)),
        clinical_cutoffs=GAD7_CUTOFFS,
        evidence=evidence,
    )


def map_to_pss10(
    emotion_result: EmotionResult,
    phenotype: Optional[PhenotypeVector] = None,
) -> ScaleResult:
    """将感知输出映射为 PSS-10 压力感知量表

    Args:
        emotion_result: 感知层情绪分析结果
        phenotype: 数字表型向量（可选）

    Returns:
        ScaleResult: PSS-10 评估结果
    """
    probs = emotion_result.text_emotion_probs
    items = []
    total = 0

    for i, text in enumerate(PSS10_ITEMS):
        score, conf = _emotion_to_item_score(probs, i, ScaleName.PSS10, phenotype)
        items.append(ScaleItem(i + 1, text, score, conf))
        total += score

    margin = 3.0 * (1.0 - emotion_result.confidence)
    ci = (max(0, total - margin), min(40, total + margin))

    return ScaleResult(
        scale_name=ScaleName.PSS10,
        total_score=total,
        max_score=40,
        items=items,
        severity=_compute_severity(ScaleName.PSS10, total),
        confidence_interval=(round(ci[0], 1), round(ci[1], 1)),
        clinical_cutoffs=PSS10_CUTOFFS,
        evidence=[f"感知置信度: {emotion_result.confidence:.2f}"],
    )


def map_all(
    emotion_result: EmotionResult,
    phenotype: Optional[PhenotypeVector] = None,
) -> dict[str, ScaleResult]:
    """同时映射所有量表

    Args:
        emotion_result: 感知层输出
        phenotype: 数字表型向量（可选）

    Returns:
        {量表名: ScaleResult} 字典
    """
    return {
        ScaleName.PHQ9.value: map_to_phq9(emotion_result, phenotype),
        ScaleName.GAD7.value: map_to_gad7(emotion_result, phenotype),
        ScaleName.PSS10.value: map_to_pss10(emotion_result, phenotype),
    }


def interpret_score(scale_name: ScaleName, score: float) -> ScaleInterpretation:
    """解读量表分数

    Args:
        scale_name: 量表名称
        score: 量表得分

    Returns:
        ScaleInterpretation: 分数解读
    """
    severity = _compute_severity(scale_name, score)
    percentile = _compute_percentile(scale_name, score)

    # 根据分级生成建议
    recommendations: dict[str, dict[str, str]] = {
        ScaleName.PHQ9.value: {
            "无抑郁": "当前情绪状态良好，建议保持健康生活方式",
            "轻度抑郁": "建议关注情绪变化，可尝试正念练习和规律运动",
            "中度抑郁": "建议寻求专业心理咨询，考虑 CBT 认知行为疗法",
            "中重度抑郁": "强烈建议尽快预约专业心理咨询师或精神科医生",
            "重度抑郁": "请立即联系专业心理健康服务机构，必要时拨打心理援助热线",
        },
        ScaleName.GAD7.value: {
            "无焦虑": "当前焦虑水平正常，继续保持",
            "轻度焦虑": "建议学习放松技巧，如深呼吸和渐进式肌肉放松",
            "中度焦虑": "建议寻求专业评估，可尝试认知行为干预",
            "重度焦虑": "强烈建议尽快就医，可能需要药物+心理联合干预",
        },
        ScaleName.PSS10.value: {
            "低压力": "压力水平正常，保持良好的应对策略",
            "中等压力": "建议识别压力源，学习压力管理技巧",
            "高压力": "建议寻求专业支持，长期高压可能影响身心健康",
        },
    }

    rec = recommendations.get(scale_name.value, {}).get(
        severity, "建议咨询专业人员获取个性化建议"
    )

    return ScaleInterpretation(
        scale_name=scale_name,
        score=score,
        severity=severity,
        recommendation=rec,
        percentile=percentile,
    )
