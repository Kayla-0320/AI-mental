"""
多模态数字表型特征提取管道

整合五类特征：
    1. 睡眠特征（sleep）：时长、规律性、入睡时间
    2. 对话情感特征（emotion）：情感分布、情感波动、负面情感占比
    3. 行为特征（behavior）：活跃度、社交频率、使用模式
    4. 测评历史特征（assessment）：量表趋势、风险变化
    5. 生理特征（physiological）：心率变异性、活动量

核心接口：
    - extract_digital_phenotype(user_id, time_window_days=14) -> PhenotypeVector
    - detect_temporal_changes(current, previous) -> PhenotypeVector (标记显著变化)
    - compute_baseline_deviation(user_id) -> dict (Z-score)

数据缺失时对应特征标记为 None，不报错。
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from shared.dataclasses import PhenotypeFeature, PhenotypeVector


# ============================================================
# 配置
# ============================================================

# 默认时间窗口（天）
DEFAULT_TIME_WINDOW_DAYS = 14

# 显著变化阈值（Cohen's d 效应量）
# 来源：Cohen (1988) —— d=0.5 为中等效应
SIGNIFICANT_CHANGE_THRESHOLD = 0.5

# 同龄基线参数（均值, 标准差）
# 来源：模拟数据，实际应从大规模样本中统计
# 格式：{feature_name: (mean, std, age_group)}
AGE_GROUP_BASELINES: dict[str, dict] = {
    "12-17": {  # 青少年
        "sleep_duration_mean": (8.0, 1.0),
        "sleep_regularity": (0.75, 0.15),
        "emotion_valence_mean": (0.55, 0.20),
        "emotion_volatility": (0.25, 0.10),
        "negative_emotion_ratio": (0.25, 0.12),
        "activity_level": (0.65, 0.20),
        "social_interaction_count": (5.0, 2.5),
        "phq9_latest": (5.0, 4.0),
        "gad7_latest": (4.5, 3.5),
        "heart_rate_variability": (55.0, 15.0),
        "daily_steps": (7000.0, 3000.0),
    },
    "18-25": {  # 青年
        "sleep_duration_mean": (7.2, 1.2),
        "sleep_regularity": (0.70, 0.18),
        "emotion_valence_mean": (0.52, 0.22),
        "emotion_volatility": (0.28, 0.12),
        "negative_emotion_ratio": (0.28, 0.15),
        "activity_level": (0.58, 0.22),
        "social_interaction_count": (6.0, 3.0),
        "phq9_latest": (5.5, 4.5),
        "gad7_latest": (5.0, 4.0),
        "heart_rate_variability": (50.0, 14.0),
        "daily_steps": (6500.0, 3200.0),
    },
    "26-35": {  # 成年
        "sleep_duration_mean": (6.8, 1.3),
        "sleep_regularity": (0.65, 0.20),
        "emotion_valence_mean": (0.50, 0.23),
        "emotion_volatility": (0.30, 0.13),
        "negative_emotion_ratio": (0.30, 0.16),
        "activity_level": (0.52, 0.24),
        "social_interaction_count": (5.5, 2.8),
        "phq9_latest": (5.2, 4.2),
        "gad7_latest": (4.8, 3.8),
        "heart_rate_variability": (45.0, 13.0),
        "daily_steps": (6000.0, 3000.0),
    },
}


# ============================================================
# 数据获取层（模拟数据源，实际接入数据库/API）
# ============================================================

def _fetch_sleep_data(user_id: str, time_window_days: int) -> Optional[dict]:
    """获取睡眠数据

    实际实现中应从可穿戴设备 API 或睡眠日志中获取。
    当前使用模拟数据。

    Returns:
        {"durations": [...], "bedtimes": [...], "wake_times": [...]}
        或 None（数据缺失）
    """
    rng = np.random.RandomState(hash(user_id) % (2**31))
    n_days = time_window_days

    return {
        "durations": rng.normal(7.0, 1.2, n_days).clip(3, 12).tolist(),
        "bedtimes": rng.normal(23.5, 1.0, n_days).clip(21, 26).tolist(),
        "wake_times": rng.normal(7.0, 0.8, n_days).clip(5, 11).tolist(),
    }


def _fetch_emotion_data(user_id: str, time_window_days: int) -> Optional[dict]:
    """获取对话情感数据

    实际实现中应从对话日志 + 感知层输出中获取。

    Returns:
        {"valence_scores": [...], "emotion_labels": [...]}
        或 None
    """
    rng = np.random.RandomState(hash(user_id + "emotion") % (2**31))
    n_entries = time_window_days * 3  # 平均每天 3 条对话

    return {
        "valence_scores": rng.beta(2, 2, n_entries).tolist(),
        "emotion_labels": rng.choice(
            ["positive", "neutral", "anxiety", "sadness", "anger"],
            size=n_entries,
            p=[0.3, 0.35, 0.15, 0.12, 0.08],
        ).tolist(),
    }


def _fetch_behavior_data(user_id: str, time_window_days: int) -> Optional[dict]:
    """获取行为数据

    Returns:
        {"daily_sessions": [...], "social_interactions": [...], "active_hours": [...]}
        或 None
    """
    rng = np.random.RandomState(hash(user_id + "behavior") % (2**31))
    n_days = time_window_days

    return {
        "daily_sessions": rng.poisson(4, n_days).tolist(),
        "social_interactions": rng.poisson(5, n_days).tolist(),
        "active_hours": rng.normal(6.0, 2.0, n_days).clip(0.5, 14).tolist(),
    }


def _fetch_assessment_data(user_id: str) -> Optional[dict]:
    """获取测评历史数据

    Returns:
        {"phq9_scores": [...], "gad7_scores": [...], "timestamps": [...]}
        或 None
    """
    rng = np.random.RandomState(hash(user_id + "assess") % (2**31))
    n_assessments = rng.randint(2, 8)

    return {
        "phq9_scores": rng.poisson(5, n_assessments).clip(0, 27).tolist(),
        "gad7_scores": rng.poisson(4, n_assessments).clip(0, 21).tolist(),
        "timestamps": list(range(n_assessments)),
    }


def _fetch_physiological_data(user_id: str, time_window_days: int) -> Optional[dict]:
    """获取生理数据

    Returns:
        {"heart_rate_variability": [...], "daily_steps": [...], "resting_hr": [...]}
        或 None
    """
    rng = np.random.RandomState(hash(user_id + "physio") % (2**31))
    n_days = time_window_days

    return {
        "heart_rate_variability": rng.normal(50, 12, n_days).clip(20, 100).tolist(),
        "daily_steps": rng.normal(6500, 2800, n_days).clip(500, 20000).tolist(),
        "resting_hr": rng.normal(72, 8, n_days).clip(55, 100).tolist(),
    }


def _get_user_age_group(user_id: str) -> str:
    """获取用户年龄段

    实际实现中应从用户档案获取。
    """
    # 模拟：基于 user_id 哈希分配
    groups = ["12-17", "18-25", "26-35"]
    return groups[hash(user_id) % len(groups)]


# ============================================================
# 特征提取函数
# ============================================================

def _extract_sleep_features(data: Optional[dict]) -> list[PhenotypeFeature]:
    """从睡眠数据提取特征"""
    if data is None:
        return [
            PhenotypeFeature("sleep_duration_mean", None, 0.0, "missing"),
            PhenotypeFeature("sleep_regularity", None, 0.0, "missing"),
            PhenotypeFeature("sleep_bedtime_mean", None, 0.0, "missing"),
        ]

    durations = data["durations"]
    bedtimes = data["bedtimes"]

    # 睡眠时长均值
    duration_mean = float(np.mean(durations))
    duration_confidence = min(1.0, len(durations) / 14.0)  # 14天满置信度

    # 睡眠规律性（时长标准差的倒数，归一化到 [0,1]）
    duration_std = float(np.std(durations))
    regularity = max(0.0, 1.0 - duration_std / 3.0)  # std=3h → regularity=0

    # 平均入睡时间
    bedtime_mean = float(np.mean(bedtimes))

    return [
        PhenotypeFeature("sleep_duration_mean", round(duration_mean, 2), round(duration_confidence, 2), "wearable", "hours"),
        PhenotypeFeature("sleep_regularity", round(regularity, 2), round(duration_confidence, 2), "wearable", "score"),
        PhenotypeFeature("sleep_bedtime_mean", round(bedtime_mean, 2), round(duration_confidence, 2), "wearable", "hour_of_day"),
    ]


def _extract_emotion_features(data: Optional[dict]) -> list[PhenotypeFeature]:
    """从对话情感数据提取特征"""
    if data is None:
        return [
            PhenotypeFeature("emotion_valence_mean", None, 0.0, "missing"),
            PhenotypeFeature("emotion_volatility", None, 0.0, "missing"),
            PhenotypeFeature("negative_emotion_ratio", None, 0.0, "missing"),
        ]

    valence = data["valence_scores"]
    labels = data["emotion_labels"]

    # 情感效价均值
    valence_mean = float(np.mean(valence))
    confidence = min(1.0, len(valence) / 30.0)

    # 情感波动性（标准差）
    volatility = float(np.std(valence))

    # 负面情感占比
    negative_labels = {"anxiety", "sadness", "anger"}
    negative_count = sum(1 for l in labels if l in negative_labels)
    negative_ratio = negative_count / len(labels) if labels else 0.0

    return [
        PhenotypeFeature("emotion_valence_mean", round(valence_mean, 3), round(confidence, 2), "chat_log", "valence"),
        PhenotypeFeature("emotion_volatility", round(volatility, 3), round(confidence, 2), "chat_log", "std"),
        PhenotypeFeature("negative_emotion_ratio", round(negative_ratio, 3), round(confidence, 2), "chat_log", "ratio"),
    ]


def _extract_behavior_features(data: Optional[dict]) -> list[PhenotypeFeature]:
    """从行为数据提取特征"""
    if data is None:
        return [
            PhenotypeFeature("activity_level", None, 0.0, "missing"),
            PhenotypeFeature("social_interaction_count", None, 0.0, "missing"),
            PhenotypeFeature("session_frequency", None, 0.0, "missing"),
        ]

    sessions = data["daily_sessions"]
    social = data["social_interactions"]
    active = data["active_hours"]

    confidence = min(1.0, len(sessions) / 14.0)

    # 活跃度（日均活跃小时数归一化）
    activity_level = float(np.mean(active)) / 14.0  # 14h → 1.0

    # 社交互动次数（日均）
    social_mean = float(np.mean(social))

    # 使用频率（日均会话数）
    session_freq = float(np.mean(sessions))

    return [
        PhenotypeFeature("activity_level", round(activity_level, 3), round(confidence, 2), "app_usage", "normalized"),
        PhenotypeFeature("social_interaction_count", round(social_mean, 2), round(confidence, 2), "app_usage", "count/day"),
        PhenotypeFeature("session_frequency", round(session_freq, 2), round(confidence, 2), "app_usage", "sessions/day"),
    ]


def _extract_assessment_features(data: Optional[dict]) -> list[PhenotypeFeature]:
    """从测评历史提取特征"""
    if data is None:
        return [
            PhenotypeFeature("phq9_latest", None, 0.0, "missing"),
            PhenotypeFeature("gad7_latest", None, 0.0, "missing"),
            PhenotypeFeature("phq9_trend", None, 0.0, "missing"),
        ]

    phq9 = data["phq9_scores"]
    gad7 = data["gad7_scores"]

    if not phq9:
        return [
            PhenotypeFeature("phq9_latest", None, 0.0, "missing"),
            PhenotypeFeature("gad7_latest", None, 0.0, "missing"),
            PhenotypeFeature("phq9_trend", None, 0.0, "missing"),
        ]

    # 最新 PHQ-9
    phq9_latest = float(phq9[-1])
    # 最新 GAD-7
    gad7_latest = float(gad7[-1]) if gad7 else 0.0

    # PHQ-9 趋势（线性回归斜率）
    if len(phq9) >= 2:
        x = np.arange(len(phq9))
        slope = float(np.polyfit(x, phq9, 1)[0])
    else:
        slope = 0.0

    confidence = min(1.0, len(phq9) / 5.0)  # 5次测评满置信度

    return [
        PhenotypeFeature("phq9_latest", round(phq9_latest, 1), round(confidence, 2), "assessment", "score"),
        PhenotypeFeature("gad7_latest", round(gad7_latest, 1), round(confidence, 2), "assessment", "score"),
        PhenotypeFeature("phq9_trend", round(slope, 3), round(confidence, 2), "assessment", "slope"),
    ]


def _extract_physiological_features(data: Optional[dict]) -> list[PhenotypeFeature]:
    """从生理数据提取特征"""
    if data is None:
        return [
            PhenotypeFeature("heart_rate_variability", None, 0.0, "missing"),
            PhenotypeFeature("daily_steps", None, 0.0, "missing"),
            PhenotypeFeature("resting_heart_rate", None, 0.0, "missing"),
        ]

    hrv = data["heart_rate_variability"]
    steps = data["daily_steps"]
    resting_hr = data["resting_hr"]

    confidence = min(1.0, len(hrv) / 14.0)

    return [
        PhenotypeFeature("heart_rate_variability", round(float(np.mean(hrv)), 1), round(confidence, 2), "wearable", "ms"),
        PhenotypeFeature("daily_steps", round(float(np.mean(steps)), 0), round(confidence, 2), "wearable", "steps/day"),
        PhenotypeFeature("resting_heart_rate", round(float(np.mean(resting_hr)), 1), round(confidence, 2), "wearable", "bpm"),
    ]


# ============================================================
# 核心接口
# ============================================================

def extract_digital_phenotype(
    user_id: str,
    time_window_days: int = DEFAULT_TIME_WINDOW_DAYS,
) -> PhenotypeVector:
    """提取多模态数字表型向量

    整合五类特征：睡眠、对话情感、行为、测评历史、生理。
    数据缺失时对应特征标记为 None，不报错。

    Args:
        user_id: 用户唯一标识
        time_window_days: 时间窗口（天），默认 14

    Returns:
        PhenotypeVector: 五类特征整合的表型向量
    """
    # 获取各类数据
    sleep_data = _fetch_sleep_data(user_id, time_window_days)
    emotion_data = _fetch_emotion_data(user_id, time_window_days)
    behavior_data = _fetch_behavior_data(user_id, time_window_days)
    assessment_data = _fetch_assessment_data(user_id)
    physio_data = _fetch_physiological_data(user_id, time_window_days)

    # 提取特征
    sleep_features = _extract_sleep_features(sleep_data)
    emotion_features = _extract_emotion_features(emotion_data)
    behavior_features = _extract_behavior_features(behavior_data)
    assessment_features = _extract_assessment_features(assessment_data)
    physio_features = _extract_physiological_features(physio_data)

    return PhenotypeVector(
        user_id=user_id,
        time_window_days=time_window_days,
        sleep_features=sleep_features,
        emotion_features=emotion_features,
        behavior_features=behavior_features,
        assessment_features=assessment_features,
        physiological_features=physio_features,
        timestamp=time.time(),
    )


# ============================================================
# 时序变化检测
# ============================================================

def detect_temporal_changes(
    current: PhenotypeVector,
    previous: PhenotypeVector,
    threshold: float = SIGNIFICANT_CHANGE_THRESHOLD,
) -> PhenotypeVector:
    """检测时序显著变化

    对比当前窗口与上一个窗口，标记显著变化。
    使用 Cohen's d 效应量衡量变化幅度。

    Args:
        current: 当前窗口的表型向量
        previous: 上一个窗口的表型向量
        threshold: 显著变化阈值（Cohen's d），默认 0.5

    Returns:
        PhenotypeVector: 标记了 significant_change 的当前向量
    """
    current_dict = current.feature_dict()
    previous_dict = previous.feature_dict()

    for feature in current.all_features():
        if feature.value is None:
            feature.significant_change = False
            continue

        prev_value = previous_dict.get(feature.name)
        if prev_value is None:
            feature.significant_change = False
            continue

        # 计算 Cohen's d（简化版：使用单值差 / 全局标准差估计）
        # 实际中应有历史数据计算标准差
        diff = abs(feature.value - prev_value)
        # 使用特征值范围的粗略估计作为标准差
        estimated_std = _estimate_feature_std(feature.name)
        if estimated_std > 0:
            cohens_d = diff / estimated_std
        else:
            cohens_d = 0.0

        feature.significant_change = cohens_d >= threshold

    return current


def _estimate_feature_std(feature_name: str) -> float:
    """估计特征的标准差（用于 Cohen's d 计算）

    实际中应从历史数据中计算，此处使用经验值。
    """
    std_estimates = {
        "sleep_duration_mean": 1.0,
        "sleep_regularity": 0.15,
        "sleep_bedtime_mean": 1.0,
        "emotion_valence_mean": 0.20,
        "emotion_volatility": 0.10,
        "negative_emotion_ratio": 0.12,
        "activity_level": 0.20,
        "social_interaction_count": 2.5,
        "session_frequency": 1.5,
        "phq9_latest": 4.0,
        "gad7_latest": 3.5,
        "phq9_trend": 0.5,
        "heart_rate_variability": 12.0,
        "daily_steps": 2500.0,
        "resting_heart_rate": 8.0,
    }
    return std_estimates.get(feature_name, 1.0)


# ============================================================
# 基线偏离度计算（Z-score）
# ============================================================

def compute_baseline_deviation(
    user_id: str,
    phenotype: Optional[PhenotypeVector] = None,
    time_window_days: int = DEFAULT_TIME_WINDOW_DAYS,
) -> dict[str, dict]:
    """计算相对于同龄基线的 Z-score

    Z-score = (用户值 - 同龄均值) / 同龄标准差

    |Z| > 2 表示显著偏离同龄群体。

    Args:
        user_id: 用户唯一标识
        phenotype: 已计算的表型向量（可选，不传则自动计算）
        time_window_days: 时间窗口

    Returns:
        {feature_name: {"z_score": float, "age_group": str, "percentile_approx": float}}
    """
    if phenotype is None:
        phenotype = extract_digital_phenotype(user_id, time_window_days)

    age_group = _get_user_age_group(user_id)
    baselines = AGE_GROUP_BASELINES.get(age_group, AGE_GROUP_BASELINES["18-25"])

    result = {}
    for feature in phenotype.all_features():
        if feature.value is None:
            result[feature.name] = {
                "z_score": None,
                "age_group": age_group,
                "percentile_approx": None,
            }
            continue

        baseline = baselines.get(feature.name)
        if baseline is None:
            result[feature.name] = {
                "z_score": None,
                "age_group": age_group,
                "percentile_approx": None,
            }
            continue

        mean, std = baseline
        if std > 0:
            z_score = (feature.value - mean) / std
        else:
            z_score = 0.0

        # 近似百分位（基于正态分布 CDF）
        from math import erf, sqrt
        percentile = 0.5 * (1 + erf(z_score / sqrt(2))) * 100

        result[feature.name] = {
            "z_score": round(z_score, 3),
            "age_group": age_group,
            "percentile_approx": round(percentile, 1),
        }

    return result
