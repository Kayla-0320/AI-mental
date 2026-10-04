"""
共病模式分析模块

分析多任务风险评估模型输出的联合风险模式，识别常见共病组合：
    - 抑郁 + 焦虑（最常见）
    - 抑郁 + 睡眠障碍
    - 焦虑 + 睡眠障碍
    - 抑郁 + 焦虑 + 睡眠障碍（三重共病）

核心接口：
    - analyze_comorbidity(risk_probs) -> ComorbidityResult
    - batch_analyze(risk_probs_list) -> list[ComorbidityResult]
    - generate_heatmap_data(results) -> dict (供前端可视化)
    - get_comorbidity_prevalence(results) -> dict (各组合流行率)

临床依据：
    - Kessler et al. (2005) —— 抑郁与焦虑共病率高达 50-60%
    - Ohayon (2002) —— 失眠与抑郁/焦虑高度共病
    - 青少年群体共病率高于成年人（Merikangas et al., 2010）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ============================================================
# 共病类型定义
# ============================================================

class ComorbidityType(str, Enum):
    """共病组合类型"""
    NONE = "none"                           # 无共病
    DEPRESSION_ONLY = "depression_only"     # 仅抑郁风险
    ANXIETY_ONLY = "anxiety_only"           # 仅焦虑风险
    SLEEP_ONLY = "sleep_only"               # 仅睡眠风险
    DEP_ANX = "dep_anx"                     # 抑郁 + 焦虑
    DEP_SLEEP = "dep_sleep"                 # 抑郁 + 睡眠
    ANX_SLEEP = "anx_sleep"                 # 焦虑 + 睡眠
    TRIPLE = "triple"                       # 三重共病（抑郁+焦虑+睡眠）


# ============================================================
# 风险阈值
# ============================================================

# 风险概率阈值（与 assessment/multi_task/predictor.py 一致）
RISK_THRESHOLD_LOW = 0.3       # 低风险上限
RISK_THRESHOLD_MEDIUM = 0.6    # 中风险上限
RISK_THRESHOLD_HIGH = 0.8      # 高风险/危机下限


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ComorbidityResult:
    """共病分析结果

    Attributes:
        user_id: 用户 ID
        comorbidity_type: 共病类型
        depression_prob: 抑郁风险概率
        anxiety_prob: 焦虑风险概率
        sleep_prob: 睡眠风险概率
        risk_combinations: 风险组合描述列表
        severity_score: 综合严重程度评分 [0.0, 3.0]
        recommendation: 干预建议
        timestamp: 分析时间戳
    """
    user_id: str
    comorbidity_type: ComorbidityType
    depression_prob: float
    anxiety_prob: float
    sleep_prob: float
    risk_combinations: list[str] = field(default_factory=list)
    severity_score: float = 0.0
    recommendation: str = ""
    timestamp: float = 0.0


# ============================================================
# 共病判定规则
# ============================================================

# 共病类型的可读名称
COMORBIDITY_NAMES: dict[ComorbidityType, str] = {
    ComorbidityType.NONE: "无显著风险",
    ComorbidityType.DEPRESSION_ONLY: "抑郁风险",
    ComorbidityType.ANXIETY_ONLY: "焦虑风险",
    ComorbidityType.SLEEP_ONLY: "睡眠问题风险",
    ComorbidityType.DEP_ANX: "抑郁 + 焦虑共病",
    ComorbidityType.DEP_SLEEP: "抑郁 + 睡眠共病",
    ComorbidityType.ANX_SLEEP: "焦虑 + 睡眠共病",
    ComorbidityType.TRIPLE: "抑郁 + 焦虑 + 睡眠三重共病",
}

# 干预建议映射
COMORBIDITY_RECOMMENDATIONS: dict[ComorbidityType, str] = {
    ComorbidityType.NONE: (
        "当前各项风险指标均在正常范围。建议保持健康的生活方式和规律的作息。"
    ),
    ComorbidityType.DEPRESSION_ONLY: (
        "检测到抑郁风险升高。建议关注情绪变化，尝试正念练习，"
        "如持续两周以上建议寻求专业评估。"
    ),
    ComorbidityType.ANXIETY_ONLY: (
        "检测到焦虑风险升高。建议尝试深呼吸放松练习，减少信息过载，"
        "如影响日常生活建议寻求专业帮助。"
    ),
    ComorbidityType.SLEEP_ONLY: (
        "检测到睡眠问题风险。建议保持规律作息，睡前减少屏幕使用，"
        "如持续影响日间功能建议咨询专业人士。"
    ),
    ComorbidityType.DEP_ANX: (
        "检测到抑郁与焦虑共病风险。这两种问题经常同时出现，相互影响。\n"
        "建议：\n"
        "  1. 优先处理当前最困扰的症状\n"
        "  2. 尝试 CBT 认知重构练习\n"
        "  3. 建议寻求专业心理咨询师评估"
    ),
    ComorbidityType.DEP_SLEEP: (
        "检测到抑郁与睡眠问题共病风险。睡眠问题可能加重抑郁情绪，反之亦然。\n"
        "建议：\n"
        "  1. 建立规律的睡眠时间表\n"
        "  2. 睡前进行放松练习\n"
        "  3. 建议寻求专业评估"
    ),
    ComorbidityType.ANX_SLEEP: (
        "检测到焦虑与睡眠问题共病风险。焦虑可能导致入睡困难，"
        "睡眠不足又可能加重焦虑。\n"
        "建议：\n"
        "  1. 睡前进行正念呼吸练习\n"
        "  2. 避免睡前思考压力事件\n"
        "  3. 建议寻求专业帮助"
    ),
    ComorbidityType.TRIPLE: (
        "⚠️ 检测到抑郁、焦虑和睡眠问题的三重共病风险。\n"
        "这种情况需要特别关注，建议：\n"
        "  1. 尽快联系专业心理咨询师或精神科医生\n"
        "  2. 建立规律的作息和运动习惯\n"
        "  3. 减少独自承受，与信任的人分享感受\n"
        "  4. 如有自伤/自杀想法，请立即拨打危机热线"
    ),
}


# ============================================================
# 核心分析函数
# ============================================================

def analyze_comorbidity(
    user_id: str,
    depression_prob: float,
    anxiety_prob: float,
    sleep_prob: float,
    threshold: float = RISK_THRESHOLD_LOW,
) -> ComorbidityResult:
    """分析单个用户的共病模式

    基于多任务模型输出的三个风险概率，判定共病类型。

    Args:
        user_id: 用户唯一标识
        depression_prob: 抑郁风险概率 [0.0, 1.0]
        anxiety_prob: 焦虑风险概率 [0.0, 1.0]
        sleep_prob: 睡眠风险概率 [0.0, 1.0]
        threshold: 风险判定阈值，默认 0.3

    Returns:
        ComorbidityResult: 共病分析结果
    """
    import time

    # 判定各维度是否超过阈值
    dep_risk = depression_prob >= threshold
    anx_risk = anxiety_prob >= threshold
    sleep_risk = sleep_prob >= threshold

    # 确定共病类型
    risk_count = sum([dep_risk, anx_risk, sleep_risk])

    if risk_count == 0:
        comorbidity_type = ComorbidityType.NONE
    elif risk_count == 1:
        if dep_risk:
            comorbidity_type = ComorbidityType.DEPRESSION_ONLY
        elif anx_risk:
            comorbidity_type = ComorbidityType.ANXIETY_ONLY
        else:
            comorbidity_type = ComorbidityType.SLEEP_ONLY
    elif risk_count == 2:
        if dep_risk and anx_risk:
            comorbidity_type = ComorbidityType.DEP_ANX
        elif dep_risk and sleep_risk:
            comorbidity_type = ComorbidityType.DEP_SLEEP
        else:
            comorbidity_type = ComorbidityType.ANX_SLEEP
    else:
        comorbidity_type = ComorbidityType.TRIPLE

    # 计算严重程度评分（0-3 分，每个风险维度 0-1 分）
    severity_score = round(
        depression_prob + anxiety_prob + sleep_prob, 3
    )

    # 构建风险组合描述
    risk_combinations = []
    if dep_risk:
        risk_combinations.append(f"抑郁风险: {depression_prob:.3f}")
    if anx_risk:
        risk_combinations.append(f"焦虑风险: {anxiety_prob:.3f}")
    if sleep_risk:
        risk_combinations.append(f"睡眠风险: {sleep_prob:.3f}")

    return ComorbidityResult(
        user_id=user_id,
        comorbidity_type=comorbidity_type,
        depression_prob=depression_prob,
        anxiety_prob=anxiety_prob,
        sleep_prob=sleep_prob,
        risk_combinations=risk_combinations,
        severity_score=severity_score,
        recommendation=COMORBIDITY_RECOMMENDATIONS[comorbidity_type],
        timestamp=time.time(),
    )


def batch_analyze(
    user_risk_data: list[dict],
    threshold: float = RISK_THRESHOLD_LOW,
) -> list[ComorbidityResult]:
    """批量共病分析

    Args:
        user_risk_data: 用户风险数据列表
            每项格式：{"user_id": str, "depression": float, "anxiety": float, "sleep": float}
        threshold: 风险判定阈值

    Returns:
        共病分析结果列表
    """
    return [
        analyze_comorbidity(
            user_id=d["user_id"],
            depression_prob=d["depression"],
            anxiety_prob=d["anxiety"],
            sleep_prob=d["sleep"],
            threshold=threshold,
        )
        for d in user_risk_data
    ]


# ============================================================
# 统计分析
# ============================================================

def get_comorbidity_prevalence(
    results: list[ComorbidityResult],
) -> dict[str, dict]:
    """计算各共病类型的流行率

    Args:
        results: 共病分析结果列表

    Returns:
        {
            "comorbidity_type": {"count": int, "prevalence": float, "avg_severity": float},
            ...
        }
    """
    total = len(results)
    if total == 0:
        return {}

    # 按类型分组
    groups: dict[ComorbidityType, list[ComorbidityResult]] = {}
    for r in results:
        groups.setdefault(r.comorbidity_type, []).append(r)

    prevalence = {}
    for ctype, group in groups.items():
        count = len(group)
        avg_severity = np.mean([r.severity_score for r in group])
        prevalence[ctype.value] = {
            "name": COMORBIDITY_NAMES[ctype],
            "count": count,
            "prevalence": round(count / total, 4),
            "avg_severity": round(float(avg_severity), 3),
        }

    return prevalence


def generate_heatmap_data(
    results: list[ComorbidityResult],
) -> dict:
    """生成共病热力图数据（供前端可视化）

    输出一个 3x3 矩阵，显示两两共病的联合概率分布。

    Args:
        results: 共病分析结果列表

    Returns:
        {
            "labels": ["抑郁", "焦虑", "睡眠"],
            "matrix": [[...], [...], [...]],  # 3x3 相关矩阵
            "comorbidity_counts": {type: count},
        }
    """
    if not results:
        return {"labels": ["抑郁", "焦虑", "睡眠"], "matrix": [], "comorbidity_counts": {}}

    # 提取概率矩阵
    probs = np.array([
        [r.depression_prob, r.anxiety_prob, r.sleep_prob]
        for r in results
    ])

    # 计算 Pearson 相关系数矩阵
    if probs.shape[0] > 1:
        corr_matrix = np.corrcoef(probs.T)
    else:
        corr_matrix = np.eye(3)

    # 统计共病类型计数
    comorbidity_counts = {}
    for r in results:
        key = r.comorbidity_type.value
        comorbidity_counts[key] = comorbidity_counts.get(key, 0) + 1

    return {
        "labels": ["抑郁", "焦虑", "睡眠"],
        "matrix": corr_matrix.round(3).tolist(),
        "comorbidity_counts": comorbidity_counts,
        "total_users": len(results),
    }


def get_high_risk_users(
    results: list[ComorbidityResult],
    min_severity: float = 1.5,
    comorbidity_types: Optional[list[ComorbidityType]] = None,
) -> list[ComorbidityResult]:
    """筛选高风险用户

    Args:
        results: 共病分析结果列表
        min_severity: 最低严重程度阈值
        comorbidity_types: 限定共病类型（None 表示不限）

    Returns:
        符合条件的高风险用户结果列表，按严重程度降序排列
    """
    filtered = [r for r in results if r.severity_score >= min_severity]

    if comorbidity_types is not None:
        filtered = [r for r in filtered if r.comorbidity_type in comorbidity_types]

    # 按严重程度降序
    filtered.sort(key=lambda r: r.severity_score, reverse=True)
    return filtered
