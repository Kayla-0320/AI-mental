"""
数字表型 API 接口

POST /api/v1/profile/phenotype —— 获取用户数字表型向量
GET  /api/v1/profile/phenotype/{user_id}/deviation —— 获取基线偏离度
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from assessment.phenotype import (
    DEFAULT_TIME_WINDOW_DAYS,
    compute_baseline_deviation,
    detect_temporal_changes,
    extract_digital_phenotype,
)

router = APIRouter(prefix="/profile", tags=["phenotype"])


@router.post("/phenotype")
def get_phenotype(
    user_id: str,
    time_window_days: int = Query(default=DEFAULT_TIME_WINDOW_DAYS, ge=1, le=90),
    compare_previous: bool = Query(default=False, description="是否对比上一窗口标记显著变化"),
) -> dict:
    """获取用户数字表型向量

    Args:
        user_id: 用户 ID
        time_window_days: 时间窗口（天）
        compare_previous: 是否对比上一窗口

    Returns:
        表型向量 + 可选的时序变化标记
    """
    phenotype = extract_digital_phenotype(user_id, time_window_days)

    # 可选：对比上一窗口
    if compare_previous:
        previous = extract_digital_phenotype(
            user_id, time_window_days
        )  # 实际应获取上一窗口数据
        phenotype = detect_temporal_changes(phenotype, previous)

    # 序列化为 dict
    return {
        "user_id": phenotype.user_id,
        "time_window_days": phenotype.time_window_days,
        "timestamp": phenotype.timestamp,
        "features": {
            "sleep": [
                {"name": f.name, "value": f.value, "confidence": f.confidence,
                 "source": f.source, "unit": f.unit, "significant_change": f.significant_change}
                for f in phenotype.sleep_features
            ],
            "emotion": [
                {"name": f.name, "value": f.value, "confidence": f.confidence,
                 "source": f.source, "unit": f.unit, "significant_change": f.significant_change}
                for f in phenotype.emotion_features
            ],
            "behavior": [
                {"name": f.name, "value": f.value, "confidence": f.confidence,
                 "source": f.source, "unit": f.unit, "significant_change": f.significant_change}
                for f in phenotype.behavior_features
            ],
            "assessment": [
                {"name": f.name, "value": f.value, "confidence": f.confidence,
                 "source": f.source, "unit": f.unit, "significant_change": f.significant_change}
                for f in phenotype.assessment_features
            ],
            "physiological": [
                {"name": f.name, "value": f.value, "confidence": f.confidence,
                 "source": f.source, "unit": f.unit, "significant_change": f.significant_change}
                for f in phenotype.physiological_features
            ],
        },
        "missing_features": phenotype.missing_features(),
        "total_features": len(phenotype.all_features()),
    }


@router.get("/phenotype/{user_id}/deviation")
def get_baseline_deviation(
    user_id: str,
    time_window_days: int = Query(default=DEFAULT_TIME_WINDOW_DAYS, ge=1, le=90),
) -> dict:
    """获取基线偏离度（Z-score）

    Args:
        user_id: 用户 ID
        time_window_days: 时间窗口

    Returns:
        各特征的 Z-score 和百分位
    """
    deviation = compute_baseline_deviation(user_id, time_window_days=time_window_days)
    return {
        "user_id": user_id,
        "deviation": deviation,
    }
