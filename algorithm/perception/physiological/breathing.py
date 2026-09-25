"""
呼吸模式分析模块 —— 通过摄像头/麦克风提取呼吸信号

临床依据：
- Ley (1987): 呼吸模式与焦虑 — 过度换气综合征
- Mezzasalma & Abelson (2001): 青少年焦虑与过度换气
- Yorke et al. (2017): 呼吸模式作为焦虑 biomarker

分析维度：
- 呼吸频率：过快→焦虑，过慢→抑郁
- 呼吸规律性：不规律→焦虑/恐慌
- 叹气频率：频繁叹气→焦虑/抑郁
- 呼吸深度估计：浅快→恐慌，深慢→可能正常

年龄差异化：
- 初中：正常基线 18次/分，焦虑阈值 24次/分
- 高中：正常基线 16次/分，焦虑阈值 22次/分
- 大学：正常基线 15次/分，焦虑阈值 20次/分
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from perception.age_config import BreathingAgeParams, AgeProfile


@dataclass
class BreathingFeatures:
    """呼吸模式特征"""

    # 呼吸频率 (次/分钟)
    breathing_rate: float = 0.0

    # 呼吸规律性 (变异系数 CV, 0-1)
    breathing_regularity_cv: float = 0.0

    # 叹气次数 (最近 5 分钟)
    sigh_count: int = 0

    # 呼吸深度估计 (0-1, 0=极浅, 1=正常深度)
    breathing_depth: float = 0.5

    # 信号质量 (0-1)
    signal_quality: float = 0.0


def analyze_breathing_risk(
    features: BreathingFeatures,
    age_profile: Optional[AgeProfile] = None,
) -> dict:
    """分析呼吸模式风险

    Args:
        features: 呼吸特征
        age_profile: 年龄画像

    Returns:
        {
            "risk_score": float (0-1),
            "anxiety_breathing_risk": float,
            "depression_breathing_risk": float,
            "regularity_risk": float,
            "evidence": list[str],
        }
    """
    if age_profile is not None:
        bp = age_profile.breathing
    else:
        bp = BreathingAgeParams()

    evidence: list[str] = []
    rate = features.breathing_rate
    quality = features.signal_quality

    if quality < 0.3 or rate <= 0:
        return {
            "risk_score": 0.0,
            "anxiety_breathing_risk": 0.0,
            "depression_breathing_risk": 0.0,
            "regularity_risk": 0.0,
            "evidence": ["呼吸信号质量不足"],
        }

    # === 1. 呼吸频率 → 焦虑/抑郁 ===
    anxiety_risk = 0.0
    depression_risk = 0.0

    if rate > bp.anxiety_rate_threshold:
        # 呼吸过快 → 过度换气 → 焦虑
        anxiety_risk = min(1.0, (rate - bp.anxiety_rate_threshold) / 6.0)
        evidence.append(f"呼吸偏快({rate:.1f}次/分，焦虑指标)")
    elif rate < bp.depression_rate_threshold:
        # 呼吸过慢 → 抑郁
        depression_risk = min(1.0, (bp.depression_rate_threshold - rate) / 4.0)
        evidence.append(f"呼吸偏慢({rate:.1f}次/分，抑郁指标)")

    # === 2. 呼吸规律性 ===
    regularity_risk = 0.0
    if features.breathing_regularity_cv > bp.regularity_normal_cv:
        regularity_risk = min(1.0,
            (features.breathing_regularity_cv - bp.regularity_normal_cv) / 0.3)
        evidence.append(f"呼吸不规律(CV={features.breathing_regularity_cv:.2f})")

    # === 3. 叹气频率 ===
    sigh_risk = 0.0
    if features.sigh_count > bp.sigh_risk_threshold:
        sigh_risk = min(1.0, (features.sigh_count - bp.sigh_risk_threshold) / 5.0)
        evidence.append(f"频繁叹气({features.sigh_count}次/5min)")
        # 叹气同时贡献焦虑和抑郁
        anxiety_risk = min(1.0, anxiety_risk + sigh_risk * 0.3)
        depression_risk = min(1.0, depression_risk + sigh_risk * 0.3)

    # === 4. 呼吸深度 ===
    if features.breathing_depth < 0.2 and rate > bp.normal_rate_baseline:
        # 浅快呼吸 → 恐慌指标
        anxiety_risk = min(1.0, anxiety_risk + 0.2)
        evidence.append("浅快呼吸模式")

    # === 5. 综合风险 ===
    risk_score = (
        anxiety_risk * 0.4
        + depression_risk * 0.3
        + regularity_risk * 0.2
        + sigh_risk * 0.1
    )
    risk_score = min(1.0, max(0.0, risk_score))

    if not evidence:
        evidence.append(f"呼吸模式正常({rate:.1f}次/分)")

    return {
        "risk_score": round(risk_score, 4),
        "anxiety_breathing_risk": round(anxiety_risk, 4),
        "depression_breathing_risk": round(depression_risk, 4),
        "regularity_risk": round(regularity_risk, 4),
        "evidence": evidence,
    }
