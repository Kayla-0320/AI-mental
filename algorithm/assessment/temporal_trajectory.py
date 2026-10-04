"""
CUSUM 变点检测算法 —— 情绪时序轨迹分析

算法创新：
    传统方法：最近 N 轮平均 - 前 N 轮平均（简单差分，噪声敏感）
    我们的方法：CUSUM (Cumulative Sum) 变点检测
    
    CUSUM 能检测时间序列中的微小但持续的漂移，比简单差分更鲁棒。

数学模型：
    S_t = max(0, S_{t-1} + (x_t - μ_0 - δ))
    
    其中：
    - x_t: 第 t 轮的负性情绪强度
    - μ_0: 基线均值（历史平均）
    - δ: 允许漂移（slack parameter，默认 0.1）
    - S_t: 累积和统计量
    
    变点判定：
    - S_t > h → 检测到向上变点（恶化）
    - S_t < -h → 检测到向下变点（好转）
    
    阈值 h 基于历史标准差：h = k * σ

设计依据：
    - Page (1954) CUSUM 算法
    - 适用于检测心理状态的渐进变化
    - 比滑动窗口平均更敏感、更鲁棒
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# CUSUM 变点检测器
# ============================================================

class CUSUMDetector:
    """CUSUM 变点检测器
    
    用于检测情绪时间序列中的变点（恶化/好转）。
    
    使用方式：
        detector = CUSUMDetector(baseline_mean=0.3, baseline_std=0.1)
        for emotion_score in emotion_sequence:
            result = detector.update(emotion_score)
            if result["change_point"]:
                print(f"检测到变点：{result['direction']}")
    """
    
    def __init__(
        self,
        baseline_mean: float = 0.3,
        baseline_std: float = 0.1,
        slack: float = 0.1,
        threshold_multiplier: float = 4.0,
    ) -> None:
        """初始化 CUSUM 检测器
        
        Args:
            baseline_mean: 基线均值
            baseline_std: 基线标准差
            slack: 允许漂移（灵敏度调节）
            threshold_multiplier: 阈值倍数（h = multiplier * std）
        """
        self.baseline_mean = baseline_mean
        self.baseline_std = max(baseline_std, 0.01)  # 避免除零
        self.slack = slack
        self.threshold = threshold_multiplier * self.baseline_std
        
        # CUSUM 统计量
        self.s_pos = 0.0  # 向上累积和（检测恶化）
        self.s_neg = 0.0  # 向下累积和（检测好转）
        
        # 历史记录
        self.history: list[float] = []
        self.change_points: list[dict] = []
    
    def update(self, value: float) -> dict:
        """更新 CUSUM 统计量并检测变点
        
        Args:
            value: 当前观测值（负性情绪强度）
        
        Returns:
            dict: {
                "change_point": bool,
                "direction": str,  # "worsening"/"improving"/"none"
                "s_pos": float,
                "s_neg": float,
                "threshold": float,
            }
        """
        self.history.append(value)
        
        # 更新 CUSUM 统计量
        self.s_pos = max(0, self.s_pos + (value - self.baseline_mean - self.slack))
        self.s_neg = max(0, self.s_neg - (value - self.baseline_mean + self.slack))
        
        # 检测变点
        change_point = False
        direction = "none"
        
        if self.s_pos > self.threshold:
            change_point = True
            direction = "worsening"
            self.change_points.append({
                "index": len(self.history) - 1,
                "value": value,
                "direction": "worsening",
                "s_pos": self.s_pos,
            })
            # 重置统计量
            self.s_pos = 0.0
        
        if self.s_neg > self.threshold:
            change_point = True
            direction = "improving"
            self.change_points.append({
                "index": len(self.history) - 1,
                "value": value,
                "direction": "improving",
                "s_neg": self.s_neg,
            })
            # 重置统计量
            self.s_neg = 0.0
        
        return {
            "change_point": change_point,
            "direction": direction,
            "s_pos": round(self.s_pos, 4),
            "s_neg": round(self.s_neg, 4),
            "threshold": round(self.threshold, 4),
            "value": value,
        }
    
    def get_trajectory(self) -> str:
        """获取当前轨迹趋势
        
        Returns:
            str: "worsening"/"improving"/"stable"/"unknown"
        """
        if len(self.history) < 3:
            return "unknown"
        
        # 基于最近变点判断
        if self.change_points:
            last_cp = self.change_points[-1]
            if last_cp["index"] >= len(self.history) - 3:
                return last_cp["direction"]
        
        # 基于 CUSUM 统计量判断
        if self.s_pos > self.threshold * 0.7:
            return "worsening"
        elif self.s_neg > self.threshold * 0.7:
            return "improving"
        else:
            return "stable"
    
    def reset(self) -> None:
        """重置检测器"""
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.history = []
        self.change_points = []


# ============================================================
# 情绪轨迹分析
# ============================================================

def analyze_emotion_trajectory(
    emotion_sequence: list[list[float]],
    baseline_mean: Optional[float] = None,
    baseline_std: Optional[float] = None,
) -> dict:
    """分析情绪轨迹
    
    Args:
        emotion_sequence: 情绪概率序列 [[快乐，悲伤，焦虑，愤怒，中性], ...]
        baseline_mean: 基线均值（可选，自动计算）
        baseline_std: 基线标准差（可选，自动计算）
    
    Returns:
        dict: {
            "trajectory": str,  # worsening/improving/stable/unknown
            "change_points": list[dict],
            "current_trend": str,
            "negative_emotion_scores": list[float],
        }
    """
    if len(emotion_sequence) < 3:
        return {
            "trajectory": "unknown",
            "change_points": [],
            "current_trend": "unknown",
            "negative_emotion_scores": [],
        }
    
    # 提取负性情绪强度（悲伤 + 焦虑 + 愤怒）
    negative_scores = []
    for probs in emotion_sequence:
        if len(probs) >= 5:
            neg = sum(probs[1:4])  # 悲伤 + 焦虑 + 愤怒
            negative_scores.append(neg)
        else:
            negative_scores.append(0.0)
    
    # 计算基线（如果未提供）
    if baseline_mean is None:
        baseline_mean = sum(negative_scores[:3]) / min(3, len(negative_scores))
    if baseline_std is None:
        if len(negative_scores) >= 3:
            mean = sum(negative_scores) / len(negative_scores)
            variance = sum((x - mean) ** 2 for x in negative_scores) / len(negative_scores)
            baseline_std = math.sqrt(variance)
        else:
            baseline_std = 0.1
    
    # 运行 CUSUM 检测
    detector = CUSUMDetector(
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        slack=0.05,
        threshold_multiplier=3.0,
    )
    
    for score in negative_scores:
        detector.update(score)
    
    trajectory = detector.get_trajectory()
    
    return {
        "trajectory": trajectory,
        "change_points": detector.change_points,
        "current_trend": trajectory,
        "negative_emotion_scores": [round(s, 4) for s in negative_scores],
        "baseline_mean": round(baseline_mean, 4),
        "baseline_std": round(baseline_std, 4),
        "cusum_s_pos": round(detector.s_pos, 4),
        "cusum_s_neg": round(detector.s_neg, 4),
    }


# ============================================================
# 轨迹可视化数据
# ============================================================

def get_trajectory_visualization_data(
    emotion_sequence: list[list[float]],
) -> dict:
    """获取轨迹可视化数据
    
    Args:
        emotion_sequence: 情绪概率序列
    
    Returns:
        dict: 用于前端可视化的数据
    """
    result = analyze_emotion_trajectory(emotion_sequence)
    
    return {
        "negative_scores": result["negative_emotion_scores"],
        "trajectory": result["trajectory"],
        "change_point_indices": [cp["index"] for cp in result["change_points"]],
        "baseline": result["baseline_mean"],
        "threshold": result["baseline_mean"] + 3 * result["baseline_std"],
    }
