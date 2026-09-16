"""
行为模式分析模块 —— 基于用户行为特征的情绪推断

分析维度：
- 使用时段模式（深夜活跃、作息紊乱）
- 交互频率变化（突然增加/减少使用）
- 会话特征（打字速度变化、消息长度变化）
- 功能使用偏好（回避社交、过度使用某些功能）
- 行为节律性（日常规律性/紊乱程度）

使用方式：
    from perception.behavior.behavior_pattern import (
        BehaviorFeatures,
        extract_behavior_features,
        behavior_to_emotion_risk,
    )
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np


# ============================================================
# 数据结构
# ============================================================

@dataclass
class BehaviorEvent:
    """用户行为事件"""
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""  # "chat", "assessment", "healing", "sleep", "community", "browse"
    duration_seconds: float = 0.0
    message_count: int = 0
    typing_speed: float = 0.0  # 字/分钟
    message_length_avg: float = 0.0  # 平均消息长度
    sentiment_score: float = 0.0  # 消息情感得分 -1~1


@dataclass
class BehaviorFeatures:
    """行为特征汇总"""
    # 时段特征
    late_night_ratio: float = 0.0  # 深夜(0-5点)使用占比
    early_morning_ratio: float = 0.0  # 凌晨(5-7点)使用占比
    peak_hour: int = 0  # 最活跃时段
    sleep_time_std: float = 0.0  # 入睡时间标准差（规律性）

    # 交互频率
    daily_sessions: list[int] = field(default_factory=list)  # 每日会话数
    session_frequency_change: float = 0.0  # 会话频率变化率
    avg_session_duration: float = 0.0  # 平均会话时长
    session_duration_trend: float = 0.0  # 会话时长趋势（正=增加）

    # 交互特征
    typing_speed_mean: float = 0.0
    typing_speed_trend: float = 0.0  # 打字速度变化趋势
    message_length_mean: float = 0.0
    message_length_trend: float = 0.0
    response_latency_mean: float = 0.0  # 平均响应延迟

    # 功能偏好
    feature_usage: dict[str, float] = field(default_factory=dict)  # 各功能使用占比
    social_engagement: float = 0.0  # 社交参与度
    avoidance_score: float = 0.0  # 回避分数（减少社交/增加独处活动）

    # 节律性
    circadian_regularity: float = 0.0  # 昼夜节律规律性 0-1
    weekly_pattern_stability: float = 0.0  # 周模式稳定性 0-1
    behavior_change_score: float = 0.0  # 行为变化分数 0-1

    # 元数据
    observation_days: int = 0
    total_events: int = 0
    data_quality: float = 0.0


# ============================================================
# 特征提取
# ============================================================

def extract_behavior_features(
    events: list[BehaviorEvent],
    observation_days: int = 14,
) -> BehaviorFeatures:
    """从行为事件序列提取行为特征

    Args:
        events: 行为事件列表（按时间排序）
        observation_days: 观察天数

    Returns:
        BehaviorFeatures
    """
    if not events:
        return BehaviorFeatures(observation_days=observation_days)

    # 按时段统计
    hour_counts: dict[int, int] = Counter()
    late_night_count = 0
    early_morning_count = 0
    total_count = len(events)

    # 按日统计
    daily_events: dict[str, list[BehaviorEvent]] = defaultdict(list)

    # 功能使用统计
    feature_counts: Counter = Counter()

    # 交互特征收集
    typing_speeds: list[float] = []
    message_lengths: list[float] = []
    session_durations: list[float] = []

    for event in events:
        hour = event.timestamp.hour
        hour_counts[hour] += 1

        if 0 <= hour < 5:
            late_night_count += 1
        elif 5 <= hour < 7:
            early_morning_count += 1

        date_key = event.timestamp.strftime("%Y-%m-%d")
        daily_events[date_key].append(event)

        if event.event_type:
            feature_counts[event.event_type] += 1

        if event.typing_speed > 0:
            typing_speeds.append(event.typing_speed)
        if event.message_length_avg > 0:
            message_lengths.append(event.message_length_avg)
        if event.duration_seconds > 0:
            session_durations.append(event.duration_seconds)

    # 时段特征
    late_night_ratio = late_night_count / max(total_count, 1)
    early_morning_ratio = early_morning_count / max(total_count, 1)
    peak_hour = max(hour_counts, key=hour_counts.get) if hour_counts else 12

    # 每日会话数
    daily_session_counts = [len(evts) for evts in daily_events.values()]

    # 会话频率变化率（后半段 vs 前半段）
    if len(daily_session_counts) >= 4:
        mid = len(daily_session_counts) // 2
        first_half = np.mean(daily_session_counts[:mid])
        second_half = np.mean(daily_session_counts[mid:])
        session_freq_change = (second_half - first_half) / max(first_half, 1)
    else:
        session_freq_change = 0.0

    # 会话时长趋势
    if len(session_durations) >= 4:
        mid = len(session_durations) // 2
        first_dur = np.mean(session_durations[:mid])
        second_dur = np.mean(session_durations[mid:])
        dur_trend = (second_dur - first_dur) / max(first_dur, 1)
    else:
        dur_trend = 0.0

    # 打字速度趋势
    if len(typing_speeds) >= 4:
        mid = len(typing_speeds) // 2
        first_ts = np.mean(typing_speeds[:mid])
        second_ts = np.mean(typing_speeds[mid:])
        ts_trend = (second_ts - first_ts) / max(first_ts, 1)
    else:
        ts_trend = 0.0

    # 消息长度趋势
    if len(message_lengths) >= 4:
        mid = len(message_lengths) // 2
        first_ml = np.mean(message_lengths[:mid])
        second_ml = np.mean(message_lengths[mid:])
        ml_trend = (second_ml - first_ml) / max(first_ml, 1)
    else:
        ml_trend = 0.0

    # 功能使用占比
    total_feature_usage = sum(feature_counts.values())
    feature_usage = {
        k: v / max(total_feature_usage, 1)
        for k, v in feature_counts.items()
    }

    # 社交参与度
    social_events = feature_counts.get("community", 0) + feature_counts.get("chat", 0)
    social_engagement = social_events / max(total_count, 1)

    # 回避分数
    avoidance_indicators = []
    if social_engagement < 0.2:
        avoidance_indicators.append(0.5)
    if late_night_ratio > 0.3:
        avoidance_indicators.append(0.3)
    if session_freq_change < -0.3:
        avoidance_indicators.append(0.4)
    avoidance_score = np.mean(avoidance_indicators) if avoidance_indicators else 0.0

    # 昼夜节律规律性
    if hour_counts:
        # 基于活动分布的熵
        probs = np.array(list(hour_counts.values()), dtype=float)
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(24)  # 均匀分布熵
        circadian_regularity = 1.0 - entropy / max(max_entropy, 1e-10)
    else:
        circadian_regularity = 0.0

    # 周模式稳定性
    weekly_patterns: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for event in events:
        week_day = event.timestamp.weekday()
        hour = event.timestamp.hour
        weekly_patterns[week_day][hour] += 1

    if len(weekly_patterns) >= 5:
        # 计算工作日 vs 周末的模式差异
        workday_pattern = np.zeros(24)
        weekend_pattern = np.zeros(24)
        for day, hours in weekly_patterns.items():
            for h, c in hours.items():
                if day < 5:
                    workday_pattern[h] += c
                else:
                    weekend_pattern[h] += c
        workday_norm = workday_pattern / max(workday_pattern.sum(), 1)
        weekend_norm = weekend_pattern / max(weekend_pattern.sum(), 1)
        # 余弦相似度
        cos_sim = np.dot(workday_norm, weekend_norm) / (
            np.linalg.norm(workday_norm) * np.linalg.norm(weekend_norm) + 1e-10
        )
        weekly_stability = float(np.clip(cos_sim, 0, 1))
    else:
        weekly_stability = 0.5

    # 行为变化综合分数
    change_indicators = [
        abs(session_freq_change),
        abs(dur_trend),
        abs(ts_trend),
        abs(ml_trend),
        avoidance_score,
    ]
    behavior_change_score = float(np.clip(np.mean(change_indicators), 0, 1))

    return BehaviorFeatures(
        late_night_ratio=late_night_ratio,
        early_morning_ratio=early_morning_ratio,
        peak_hour=peak_hour,
        sleep_time_std=0.0,  # 需要更精确的就寝数据
        daily_sessions=daily_session_counts,
        session_frequency_change=session_freq_change,
        avg_session_duration=np.mean(session_durations) if session_durations else 0.0,
        session_duration_trend=dur_trend,
        typing_speed_mean=np.mean(typing_speeds) if typing_speeds else 0.0,
        typing_speed_trend=ts_trend,
        message_length_mean=np.mean(message_lengths) if message_lengths else 0.0,
        message_length_trend=ml_trend,
        response_latency_mean=0.0,
        feature_usage=feature_usage,
        social_engagement=social_engagement,
        avoidance_score=avoidance_score,
        circadian_regularity=circadian_regularity,
        weekly_pattern_stability=weekly_stability,
        behavior_change_score=behavior_change_score,
        observation_days=observation_days,
        total_events=total_count,
        data_quality=min(total_count / (observation_days * 5), 1.0),
    )


# ============================================================
# 行为 → 情绪风险映射
# ============================================================

def behavior_to_emotion_risk(features: BehaviorFeatures) -> dict[str, Any]:
    """将行为特征映射为情绪风险

    基于行为心理学研究的经验规则：
    - 深夜活跃 + 作息紊乱 → 抑郁/焦虑风险
    - 社交回避 + 使用减少 → 抑郁风险
    - 打字速度减慢 + 消息变短 → 抑郁风险
    - 使用频率突增 + 深夜使用 → 焦虑风险

    Args:
        features: 行为特征

    Returns:
        dict with emotion_probs, risk_score, risk_label, confidence, metadata
    """
    emotion_probs: dict[str, float] = {
        "neutral": 0.5,
        "depression": 0.0,
        "anxiety": 0.0,
        "stress": 0.0,
        "sleep_disorder": 0.0,
    }

    # 抑郁指标
    depression_signals = []
    if features.social_engagement < 0.2:
        depression_signals.append(0.4)
    if features.avoidance_score > 0.3:
        depression_signals.append(0.3)
    if features.typing_speed_trend < -0.2:
        depression_signals.append(0.3)
    if features.message_length_trend < -0.2:
        depression_signals.append(0.2)
    if features.session_frequency_change < -0.3:
        depression_signals.append(0.3)
    if features.circadian_regularity < 0.3:
        depression_signals.append(0.2)

    depression_risk = np.mean(depression_signals) if depression_signals else 0.0

    # 焦虑指标
    anxiety_signals = []
    if features.late_night_ratio > 0.3:
        anxiety_signals.append(0.3)
    if features.session_duration_trend > 0.3:
        anxiety_signals.append(0.3)
    if features.typing_speed_trend > 0.2:
        anxiety_signals.append(0.2)
    if features.behavior_change_score > 0.4:
        anxiety_signals.append(0.3)
    if features.circadian_regularity < 0.4:
        anxiety_signals.append(0.2)

    anxiety_risk = np.mean(anxiety_signals) if anxiety_signals else 0.0

    # 压力指标
    stress_signals = []
    if features.typing_speed_mean > 80:  # 打字很快
        stress_signals.append(0.2)
    if features.session_duration_trend > 0.2:
        stress_signals.append(0.2)
    if features.behavior_change_score > 0.3:
        stress_signals.append(0.3)

    stress_risk = np.mean(stress_signals) if stress_signals else 0.0

    # 睡眠障碍指标
    sleep_signals = []
    if features.late_night_ratio > 0.25:
        sleep_signals.append(0.4)
    if features.early_morning_ratio > 0.15:
        sleep_signals.append(0.3)
    if features.circadian_regularity < 0.4:
        sleep_signals.append(0.3)
    if features.weekly_pattern_stability < 0.5:
        sleep_signals.append(0.2)

    sleep_risk = np.mean(sleep_signals) if sleep_signals else 0.0

    # 更新概率
    emotion_probs["depression"] = float(np.clip(depression_risk, 0, 1))
    emotion_probs["anxiety"] = float(np.clip(anxiety_risk, 0, 1))
    emotion_probs["stress"] = float(np.clip(stress_risk, 0, 1))
    emotion_probs["sleep_disorder"] = float(np.clip(sleep_risk, 0, 1))

    # 归一化
    total = sum(emotion_probs.values())
    emotion_probs = {k: v / total for k, v in emotion_probs.items()}

    # 综合风险
    risk_score = float(np.clip(
        depression_risk * 0.35 + anxiety_risk * 0.25 + stress_risk * 0.15 + sleep_risk * 0.25,
        0, 1,
    ))

    if risk_score > 0.6:
        risk_label = "high_risk"
    elif risk_score > 0.3:
        risk_label = "moderate_risk"
    else:
        risk_label = "low_risk"

    return {
        "emotion_probs": emotion_probs,
        "risk_score": risk_score,
        "risk_label": risk_label,
        "confidence": features.data_quality,
        "metadata": {
            "source": "behavior_pattern",
            "observation_days": features.observation_days,
            "total_events": features.total_events,
            "social_engagement": features.social_engagement,
            "circadian_regularity": features.circadian_regularity,
            "late_night_ratio": features.late_night_ratio,
            "behavior_change_score": features.behavior_change_score,
        },
    }


def behavior_to_summary(features: BehaviorFeatures) -> dict[str, Any]:
    """生成行为特征摘要

    Returns:
        包含行为模式、风险等级、建议的字典
    """
    result = behavior_to_emotion_risk(features)

    # 生成建议
    recommendations = []
    if features.late_night_ratio > 0.25:
        recommendations.append("建议调整作息，减少深夜使用电子设备")
    if features.social_engagement < 0.2:
        recommendations.append("建议增加社交互动，参与社区活动")
    if features.session_frequency_change < -0.3:
        recommendations.append("活动量明显减少，建议保持规律的运动和社交")
    if features.circadian_regularity < 0.4:
        recommendations.append("作息规律性较差，建议建立固定的日常节奏")
    if features.typing_speed_trend < -0.2:
        recommendations.append("打字速度明显减慢，可能反映精力下降")
    if not recommendations:
        recommendations.append("行为模式正常，继续保持良好的生活习惯")

    return {
        "risk_score": result["risk_score"],
        "risk_label": result["risk_label"],
        "emotion_probs": result["emotion_probs"],
        "behavior_summary": {
            "peak_hour": features.peak_hour,
            "late_night_ratio": features.late_night_ratio,
            "social_engagement": features.social_engagement,
            "circadian_regularity": features.circadian_regularity,
            "behavior_change": features.behavior_change_score,
            "typing_speed": features.typing_speed_mean,
            "avg_session_duration": features.avg_session_duration,
        },
        "recommendations": recommendations,
        "confidence": result["confidence"],
    }
