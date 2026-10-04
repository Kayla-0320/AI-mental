"""
行为激活水平分析模块 —— 数字表型行为监测

临床依据：
- Martell et al. (2001): 行为激活理论 — 抑郁的核心是正性强化减少
- Masi et al. (2011): 青少年社交行为与心理健康
- DSM-5: 兴趣/愉悦感丧失是抑郁核心症状 (Criterion A2)

分析维度：
- 社交互动量：消息发送频率、互动对象多样性
- 功能模块探索率：用户访问不同功能模块的广度
- 活动时段分布：活动是否过于集中或过于稀疏
- 行为变化趋势：近 7 天行为活跃度变化方向

年龄差异化：
- 初中：社交需求高，每日最低互动 15 次
- 高中：社交需求中等，每日最低互动 10 次
- 大学：社交需求较低但质量更重要，每日最低互动 8 次
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from perception.age_config import BehavioralActivationAgeParams, AgeProfile


@dataclass
class BehavioralActivationFeatures:
    """行为激活特征"""

    # 今日互动次数 (消息、点击等)
    daily_interaction_count: int = 0

    # 互动对象多样性 (不同联系人/聊天数)
    interaction_diversity: int = 0

    # 功能模块探索率 (0-1)：访问过的模块数 / 总模块数
    exploration_rate: float = 0.0

    # 活动时段分布 (24 个桶，每个桶的活动比例)
    hourly_activity: list[float] = field(default_factory=lambda: [0.0] * 24)

    # 近 7 天每日互动数 [day1, day2, ..., day7]
    weekly_interaction_trend: list[int] = field(default_factory=list)

    # 社交退缩指标 (0-1)：近期互动减少程度
    social_withdrawal_score: float = 0.0


def analyze_behavioral_activation_risk(
    features: BehavioralActivationFeatures,
    age_profile: Optional[AgeProfile] = None,
) -> dict:
    """分析行为激活水平风险

    Args:
        features: 行为激活特征
        age_profile: 年龄画像

    Returns:
        {
            "risk_score": float (0-1),
            "withdrawal_risk": float,
            "exploration_risk": float,
            "activity_concentration": float,
            "trend_direction": str ("improving" | "stable" | "declining"),
            "evidence": list[str],
        }
    """
    if age_profile is not None:
        ba = age_profile.behavioral_activation
    else:
        ba = BehavioralActivationAgeParams()

    evidence: list[str] = []

    # === 1. 社交互动量 → 退缩风险 ===
    withdrawal_risk = 0.0
    if features.daily_interaction_count < ba.min_daily_interactions:
        ratio = features.daily_interaction_count / max(ba.min_daily_interactions, 1)
        withdrawal_risk = min(1.0, (1 - ratio) * 0.7)
        evidence.append(
            f"互动量偏低({features.daily_interaction_count}次，"
            f"低于阈值{ba.min_daily_interactions}次)"
        )

    # 社交退缩得分直接贡献
    if features.social_withdrawal_score > ba.social_withdrawal_threshold:
        withdrawal_risk = min(1.0, withdrawal_risk + features.social_withdrawal_score * 0.3)
        evidence.append("社交退缩信号")

    # === 2. 功能探索率 ===
    exploration_risk = 0.0
    if features.exploration_rate < ba.exploration_rate_baseline * 0.5:
        exploration_risk = min(1.0,
            (ba.exploration_rate_baseline - features.exploration_rate) / ba.exploration_rate_baseline)
        evidence.append(f"功能探索率低({features.exploration_rate:.0%})")

    # === 3. 活动时段集中度 ===
    concentration = 0.0
    if features.hourly_activity and sum(features.hourly_activity) > 0:
        total = sum(features.hourly_activity)
        # 计算活动集中程度 (HHI - Herfindahl 指数变体)
        concentration = sum((h / total) ** 2 for h in features.hourly_activity if h > 0)
        if concentration > ba.activity_concentration_threshold:
            evidence.append(f"活动过于集中(HHI={concentration:.2f})")

    # === 4. 行为变化趋势 ===
    trend_direction = "stable"
    if len(features.weekly_interaction_trend) >= 4:
        trend = features.weekly_interaction_trend
        first_half = sum(trend[:len(trend)//2]) / max(len(trend)//2, 1)
        second_half = sum(trend[len(trend)//2:]) / max(len(trend) - len(trend)//2, 1)

        if first_half > 0:
            change_ratio = (second_half - first_half) / first_half
            if change_ratio < -0.3:
                trend_direction = "declining"
                withdrawal_risk = min(1.0, withdrawal_risk + 0.2)
                evidence.append("行为活跃度持续下降")
            elif change_ratio > 0.3:
                trend_direction = "improving"

    # === 5. 综合风险 ===
    risk_score = (
        withdrawal_risk * ba.withdrawal_risk_weight
        + exploration_risk * 0.2
        + max(0, concentration - 0.5) * 0.1
    )
    risk_score = min(1.0, max(0.0, risk_score))

    if not evidence:
        evidence.append("行为激活水平正常")

    return {
        "risk_score": round(risk_score, 4),
        "withdrawal_risk": round(withdrawal_risk, 4),
        "exploration_risk": round(exploration_risk, 4),
        "activity_concentration": round(concentration, 4),
        "trend_direction": trend_direction,
        "evidence": evidence,
    }
