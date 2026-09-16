"""
纵向趋势分析模块 —— 心理状态时序变化检测与预测

分析用户心理指标随时间的变化趋势：
    1. 线性回归：整体趋势方向与斜率
    2. 变化点检测：识别突变时刻（CUSUM 算法）
    3. 趋势分类：改善/稳定/恶化
    4. 短期预测：基于历史趋势预测下一窗口

核心接口：
    - analyze_trend(values, timestamps) -> TrendResult
    - detect_change_points(values) -> list[ChangePoint]
    - classify_trend(result) -> TrendClassification
    - predict_next(values, timestamps) -> float
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ============================================================
# 数据结构
# ============================================================

class TrendDirection(str, Enum):
    """趋势方向"""
    IMPROVING = "improving"         # 改善
    STABLE = "stable"               # 稳定
    WORSENING = "worsening"         # 恶化
    VOLATILE = "volatile"           # 波动
    INSUFFICIENT_DATA = "insufficient"  # 数据不足


class ChangePointSeverity(str, Enum):
    """变化点严重程度"""
    MINOR = "minor"                 # 轻微波化
    MODERATE = "moderate"           # 中等变化
    SIGNIFICANT = "significant"     # 显著变化
    CRITICAL = "critical"           # 临界变化


@dataclass
class ChangePoint:
    """变化点

    Attributes:
        index: 变化点在序列中的索引
        timestamp: 变化时间戳
        before_mean: 变化前均值
        after_mean: 变化后均值
        effect_size: Cohen's d 效应量
        severity: 严重程度
    """
    index: int
    timestamp: float
    before_mean: float
    after_mean: float
    effect_size: float
    severity: ChangePointSeverity


@dataclass
class TrendResult:
    """趋势分析结果

    Attributes:
        direction: 趋势方向
        slope: 线性回归斜率（单位/天）
        r_squared: 拟合优度 R²
        mean_value: 均值
        std_value: 标准差
        change_points: 变化点列表
        data_points: 数据点数
        analysis_window_days: 分析时间窗口（天）
        is_significant: 趋势是否统计显著
    """
    direction: TrendDirection
    slope: float = 0.0
    r_squared: float = 0.0
    mean_value: float = 0.0
    std_value: float = 0.0
    change_points: list[ChangePoint] = field(default_factory=list)
    data_points: int = 0
    analysis_window_days: float = 0.0
    is_significant: bool = False


@dataclass
class TrendClassification:
    """趋势分类与临床解读

    Attributes:
        direction: 趋势方向
        description: 中文描述
        clinical_significance: 临床意义
        recommendation: 建议
        urgency: 紧急程度 0-1
    """
    direction: TrendDirection
    description: str
    clinical_significance: str
    recommendation: str
    urgency: float = 0.0


# ============================================================
# 线性回归趋势分析
# ============================================================

def _linear_trend(
    values: list[float],
    timestamps: Optional[list[float]] = None,
) -> tuple[float, float]:
    """线性回归计算趋势

    Args:
        values: 观测值序列
        timestamps: 时间戳序列（可选，默认等间距）

    Returns:
        (slope, r_squared): 斜率和拟合优度
    """
    n = len(values)
    if n < 2:
        return 0.0, 0.0

    y = np.array(values, dtype=float)

    if timestamps is not None and len(timestamps) == n:
        # 将时间戳转换为天数
        t = np.array(timestamps, dtype=float)
        t = (t - t[0]) / 86400.0  # 秒 → 天
    else:
        t = np.arange(n, dtype=float)

    # 线性回归
    coeffs = np.polyfit(t, y, 1)
    slope = float(coeffs[0])

    # R²
    y_pred = np.polyval(coeffs, t)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    r_squared = max(0.0, r_squared)

    return round(slope, 4), round(r_squared, 4)


# ============================================================
# 变化点检测 (CUSUM)
# ============================================================

def _detect_change_points_cusum(
    values: list[float],
    threshold: float = 2.0,
    min_segment: int = 3,
    timestamps: Optional[list[float]] = None,
) -> list[ChangePoint]:
    """CUSUM 变化点检测

    累积和 (CUSUM) 算法检测均值漂移。

    Args:
        values: 观测值序列
        threshold: 检测阈值（标准差倍数）
        min_segment: 最小段长度
        timestamps: 时间戳列表

    Returns:
        变化点列表
    """
    n = len(values)
    if n < min_segment * 2:
        return []

    arr = np.array(values, dtype=float)
    global_mean = np.mean(arr)
    global_std = np.std(arr)

    if global_std < 1e-6:
        return []

    change_points = []

    # 滑动窗口检测
    for i in range(min_segment, n - min_segment):
        before = arr[:i]
        after = arr[i:]

        before_mean = np.mean(before)
        after_mean = np.mean(after)

        # Cohen's d
        pooled_std = np.sqrt(
            (np.std(before) ** 2 + np.std(after) ** 2) / 2
        )
        if pooled_std < 1e-6:
            continue

        d = (after_mean - before_mean) / pooled_std

        if abs(d) >= threshold:
            # 确定严重程度
            abs_d = abs(d)
            if abs_d >= 2.0:
                severity = ChangePointSeverity.CRITICAL
            elif abs_d >= 1.0:
                severity = ChangePointSeverity.SIGNIFICANT
            elif abs_d >= 0.5:
                severity = ChangePointSeverity.MODERATE
            else:
                severity = ChangePointSeverity.MINOR

            ts = timestamps[i] if timestamps and i < len(timestamps) else time.time()

            change_points.append(ChangePoint(
                index=i,
                timestamp=ts,
                before_mean=round(float(before_mean), 2),
                after_mean=round(float(after_mean), 2),
                effect_size=round(float(d), 3),
                severity=severity,
            ))

    # 去重：只保留效应量最大的变化点（在 ±3 范围内）
    if not change_points:
        return []

    filtered = [change_points[0]]
    for cp in change_points[1:]:
        if abs(cp.index - filtered[-1].index) > min_segment:
            filtered.append(cp)
        elif abs(cp.effect_size) > abs(filtered[-1].effect_size):
            filtered[-1] = cp

    return filtered


# ============================================================
# 趋势分类
# ============================================================

def _classify_direction(
    slope: float,
    r_squared: float,
    values: list[float],
) -> TrendDirection:
    """分类趋势方向"""
    n = len(values)
    if n < 3:
        return TrendDirection.INSUFFICIENT_DATA

    arr = np.array(values, dtype=float)
    cv = np.std(arr) / (np.mean(arr) + 1e-6)  # 变异系数

    # R² 太低说明趋势不稳定
    if r_squared < 0.2:
        if cv > 0.5:
            return TrendDirection.VOLATILE
        return TrendDirection.STABLE

    # 斜率方向（考虑指标含义：高分=高风险）
    # 斜率 > 0 → 恶化，斜率 < 0 → 改善
    mean_val = np.mean(arr)
    normalized_slope = slope / (abs(mean_val) + 1e-6)

    if normalized_slope > 0.02:
        return TrendDirection.WORSENING
    elif normalized_slope < -0.02:
        return TrendDirection.IMPROVING
    else:
        return TrendDirection.STABLE


def classify_trend(result: TrendResult) -> TrendClassification:
    """将趋势结果转化为临床解读

    Args:
        result: 趋势分析结果

    Returns:
        TrendClassification: 临床解读
    """
    direction = result.direction

    classifications = {
        TrendDirection.IMPROVING: TrendClassification(
            direction=TrendDirection.IMPROVING,
            description="心理状态呈改善趋势，各项指标逐步向好",
            clinical_significance="干预措施可能正在起效，建议继续当前方案",
            recommendation="保持当前干预方案，可适当降低监测频率",
            urgency=0.1,
        ),
        TrendDirection.STABLE: TrendClassification(
            direction=TrendDirection.STABLE,
            description="心理状态整体稳定，无明显波动",
            clinical_significance="当前状态平稳，维持常规关注即可",
            recommendation="保持定期评估，关注潜在变化信号",
            urgency=0.2,
        ),
        TrendDirection.WORSENING: TrendClassification(
            direction=TrendDirection.WORSENING,
            description="心理状态呈恶化趋势，指标持续上升",
            clinical_significance="需要引起重视，可能需要调整干预方案",
            recommendation="建议增加评估频率，考虑调整干预策略或转介",
            urgency=0.7,
        ),
        TrendDirection.VOLATILE: TrendClassification(
            direction=TrendDirection.VOLATILE,
            description="心理状态波动较大，缺乏稳定趋势",
            clinical_significance="情绪不稳定可能反映环境压力或干预效果不一致",
            recommendation="建议深入评估波动原因，排查外部压力源",
            urgency=0.5,
        ),
        TrendDirection.INSUFFICIENT_DATA: TrendClassification(
            direction=TrendDirection.INSUFFICIENT_DATA,
            description="数据不足，无法判断趋势",
            clinical_significance="需要更多数据点才能进行可靠分析",
            recommendation="继续保持定期评估，积累足够数据后再做判断",
            urgency=0.3,
        ),
    }

    return classifications.get(direction, classifications[TrendDirection.STABLE])


# ============================================================
# 短期预测
# ============================================================

def predict_next(
    values: list[float],
    timestamps: Optional[list[float]] = None,
) -> Optional[float]:
    """基于历史趋势预测下一个值

    使用线性外推 + 指数加权移动平均的混合预测。

    Args:
        values: 历史观测值
        timestamps: 时间戳（可选）

    Returns:
        预测值，数据不足时返回 None
    """
    n = len(values)
    if n < 2:
        return values[0] if n == 1 else None

    arr = np.array(values, dtype=float)

    # 线性外推
    if timestamps and len(timestamps) == n:
        t = np.array(timestamps, dtype=float)
        t_next = t[-1] + (t[-1] - t[-2])  # 假设等间距
    else:
        t = np.arange(n, dtype=float)
        t_next = n

    coeffs = np.polyfit(t, arr, 1)
    linear_pred = float(np.polyval(coeffs, t_next))

    # 指数加权移动平均 (EWMA)
    alpha = 0.3
    ewma = arr[0]
    for v in arr[1:]:
        ewma = alpha * v + (1 - alpha) * ewma

    # 混合：R² 高时更信任线性，低时更信任 EWMA
    y_pred = np.polyval(coeffs, t)
    ss_res = np.sum((arr - y_pred) ** 2)
    ss_tot = np.sum((arr - np.mean(arr)) ** 2)
    r_squared = max(0.0, 1.0 - ss_res / (ss_tot + 1e-6))

    blended = r_squared * linear_pred + (1 - r_squared) * ewma

    return round(float(blended), 2)


# ============================================================
# 核心接口
# ============================================================

def analyze_trend(
    values: list[float],
    timestamps: Optional[list[float]] = None,
    cusum_threshold: float = 2.0,
) -> TrendResult:
    """完整趋势分析

    整合线性回归 + 变化点检测 + 趋势分类。

    Args:
        values: 观测值序列（按时间排序）
        timestamps: 时间戳序列（可选）
        cusum_threshold: CUSUM 检测阈值

    Returns:
        TrendResult: 完整趋势分析结果
    """
    n = len(values)
    arr = np.array(values, dtype=float)

    # 线性趋势
    slope, r_squared = _linear_trend(values, timestamps)

    # 变化点检测
    change_points = _detect_change_points_cusum(
        values, threshold=cusum_threshold, timestamps=timestamps
    )

    # 趋势分类
    direction = _classify_direction(slope, r_squared, values)

    # 时间窗口
    if timestamps and len(timestamps) >= 2:
        window_days = (timestamps[-1] - timestamps[0]) / 86400.0
    else:
        window_days = float(n)

    # 统计显著性：R² > 0.4 且数据点 ≥ 5
    is_significant = r_squared > 0.4 and n >= 5

    return TrendResult(
        direction=direction,
        slope=slope,
        r_squared=r_squared,
        mean_value=round(float(np.mean(arr)), 2),
        std_value=round(float(np.std(arr)), 2),
        change_points=change_points,
        data_points=n,
        analysis_window_days=round(window_days, 1),
        is_significant=is_significant,
    )


def batch_analyze(
    metrics: dict[str, list[float]],
    timestamps: Optional[list[float]] = None,
) -> dict[str, TrendResult]:
    """批量趋势分析

    Args:
        metrics: {指标名: 值序列} 字典
        timestamps: 共用时间戳

    Returns:
        {指标名: TrendResult} 字典
    """
    return {
        name: analyze_trend(values, timestamps)
        for name, values in metrics.items()
        if len(values) >= 2
    }
