"""
昼夜节律分析模块 —— 基于时间戳的活动模式分析

临床依据：
- Walker & Harvey (2019): 青少年睡眠模式与情绪调节
- DSM-5: 睡眠紊乱是抑郁核心诊断标准 (Criterion A3)
- Owens et al. (2014): 青少年年龄特异性睡眠需求

分析维度：
- 深夜活动风险：凌晨时段的活跃度 → 睡眠紊乱指标
- 作息规律性：活动时间的标准差 → 昼夜节律紊乱
- 活跃度时段分布：白天 vs 夜间活动比 → 社会节律偏离
- 活动窗口：最活跃时段偏移 → 社会时差 (social jet lag)

年龄差异化：
- 初中：理想就寝 21:30，22:00 后活动即有风险
- 高中：理想就寝 22:30，23:00 后活动有风险
- 大学：理想就寝 23:30，0:00 后活动有风险
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from perception.age_config import CircadianAgeParams, AgeProfile


@dataclass
class CircadianFeatures:
    """昼夜节律特征（前端/外部传入）"""

    # 最近 N 次活动的时间戳列表 (Unix timestamp)
    activity_timestamps: list[float] = field(default_factory=list)

    # 最近一次活动的 hour (0-23)
    last_activity_hour: float = 12.0

    # 过去 7 天每天的活动小时分布 {day: [hours]}
    daily_activity_hours: dict[int, list[float]] = field(default_factory=dict)

    # 当前时间戳 (可选，默认 time.time())
    current_timestamp: Optional[float] = None


def analyze_circadian_risk(
    features: CircadianFeatures,
    age_profile: Optional[AgeProfile] = None,
) -> dict:
    """分析昼夜节律风险

    Args:
        features: 昼夜节律特征
        age_profile: 年龄画像（可选，用于年龄差异化校准）

    Returns:
        {
            "risk_score": float (0-1),
            "late_night_risk": float (0-1),
            "regularity_score": float (0-1, 越高越规律),
            "social_jetlag_hours": float,
            "evidence": list[str],
        }
    """
    # 年龄校准参数
    if age_profile is not None:
        cp = age_profile.circadian
    else:
        cp = CircadianAgeParams()

    evidence: list[str] = []
    hour = features.last_activity_hour

    # === 1. 深夜活动风险 ===
    late_risk_hour = cp.late_activity_risk_hour
    ideal_bedtime = cp.ideal_bedtime

    if hour >= late_risk_hour or (hour < 5 and hour < cp.ideal_waketime):
        # 深夜活跃
        if hour >= late_risk_hour:
            hours_past = hour - late_risk_hour
        else:
            hours_past = (24 - late_risk_hour) + hour
        late_night_risk = min(1.0, 0.3 + hours_past * 0.1)
        evidence.append(f"深夜活跃({hour:.0f}:00，超过风险阈值{late_risk_hour}:00)")
    elif hour < cp.ideal_waketime:
        # 凌晨过早活动
        late_night_risk = 0.4
        evidence.append(f"凌晨过早活动({hour:.0f}:00)")
    else:
        late_night_risk = 0.0

    # === 2. 作息规律性 ===
    regularity_score = 0.5  # 默认中等
    social_jetlag = 0.0

    if features.daily_activity_hours and len(features.daily_activity_hours) >= 3:
        # 计算每天活动中心的均值和标准差
        daily_centers = []
        for day, hours in features.daily_activity_hours.items():
            if hours:
                center = sum(hours) / len(hours)
                daily_centers.append(center)

        if len(daily_centers) >= 3:
            mean_center = sum(daily_centers) / len(daily_centers)
            variance = sum((c - mean_center) ** 2 for c in daily_centers) / len(daily_centers)
            std_center = math.sqrt(variance)

            # 规律性：标准差越小越规律
            regularity_score = max(0, min(1, 1 - std_center / 6.0))

            # 社会时差：工作日 vs 周末活动中心差异
            # 简化：用标准差近似
            social_jetlag = std_center

            if regularity_score < 0.4:
                evidence.append(f"作息极不规律(变异度={std_center:.1f}h)")
            elif regularity_score < 0.6:
                evidence.append(f"作息偏不规律(变异度={std_center:.1f}h)")

    # === 3. 综合风险 ===
    penalty = cp.irregularity_penalty_weight
    risk_score = (
        late_night_risk * 0.6
        + (1 - regularity_score) * penalty
        + max(0, (1 - regularity_score) * 0.1)
    )
    risk_score = min(1.0, max(0.0, risk_score))

    if not evidence:
        evidence.append("昼夜节律正常")

    return {
        "risk_score": round(risk_score, 4),
        "late_night_risk": round(late_night_risk, 4),
        "regularity_score": round(regularity_score, 4),
        "social_jetlag_hours": round(social_jetlag, 2),
        "evidence": evidence,
    }
