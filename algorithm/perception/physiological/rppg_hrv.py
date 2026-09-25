"""
rPPG 心率变异性分析模块 —— 通过摄像头提取心率信号

临床依据：
- Thayer & Sternberg (2006): HRV 与情绪调节的关系
- Marcovitch et al. (2010): 青少年 HRV 发展常模
- Kim et al. (2018): 摄像头 HRV 估计的准确性验证

技术原理：
- rPPG (Remote Photoplethysmography): 通过分析面部皮肤颜色微变化
  提取心率信号，无需接触式传感器
- 绿色通道 (G-channel) 对血液容积变化最敏感
- HRV (Heart Rate Variability) 反映自主神经系统活动

年龄差异化：
- 初中：静息心率基线 85bpm，HRV 基线 55ms
- 高中：静息心率基线 75bpm，HRV 基线 45ms
- 大学：静息心率基线 70bpm，HRV 基线 40ms
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from perception.age_config import HRVAgeParams, AgeProfile


@dataclass
class HRVFeatures:
    """心率变异性特征（前端 rPPG 提取）"""

    # 估计的心率 (bpm)
    heart_rate: float = 0.0

    # HRV RMSSD (ms) — 副交感神经活动指标
    hrv_rmssd: float = 0.0

    # 心率时间序列 (bpm 列表，最近 N 个采样点)
    heart_rate_history: list[float] = field(default_factory=list)

    # 信号质量 (0-1)：rPPG 信号的信噪比
    signal_quality: float = 0.0


def analyze_hrv_risk(
    features: HRVFeatures,
    age_profile: Optional[AgeProfile] = None,
) -> dict:
    """分析心率变异性风险

    Args:
        features: HRV 特征
        age_profile: 年龄画像

    Returns:
        {
            "risk_score": float (0-1),
            "heart_rate": float,
            "hrv_rmssd": float,
            "hr_anxiety_risk": float,
            "hrv_risk": float,
            "signal_quality": float,
            "evidence": list[str],
        }
    """
    if age_profile is not None:
        hp = age_profile.hrv
    else:
        hp = HRVAgeParams()

    evidence: list[str] = []
    hr = features.heart_rate
    rmssd = features.hrv_rmssd
    quality = features.signal_quality

    # 信号质量过低时不输出可靠结果
    if quality < 0.3 or hr <= 0:
        return {
            "risk_score": 0.0,
            "heart_rate": hr,
            "hrv_rmssd": rmssd,
            "hr_anxiety_risk": 0.0,
            "hrv_risk": 0.0,
            "signal_quality": quality,
            "evidence": ["rPPG信号质量不足，暂不分析"],
        }

    # === 1. 心率焦虑风险 ===
    hr_anxiety_risk = 0.0
    if hr > hp.anxiety_hr_threshold:
        hr_anxiety_risk = min(1.0, (hr - hp.anxiety_hr_threshold) / 20.0)
        evidence.append(f"心率偏高({hr:.0f}bpm，超过焦虑阈值{hp.anxiety_hr_threshold:.0f}bpm)")
    elif hr < hp.resting_hr_baseline * 0.7:
        # 过低心率 → 可能抑郁/退缩
        hr_anxiety_risk = 0.3
        evidence.append(f"心率偏低({hr:.0f}bpm，可能抑郁指标)")

    # === 2. HRV 风险 ===
    hrv_risk = 0.0
    if rmssd > 0 and rmssd < hp.hrv_low_risk_threshold:
        # 低 HRV → 自主神经失调 → 情绪调节困难
        hrv_risk = min(1.0, (hp.hrv_low_risk_threshold - rmssd) / hp.hrv_low_risk_threshold)
        evidence.append(f"HRV偏低({rmssd:.1f}ms，自主神经调节能力下降)")
    elif rmssd >= hp.hrv_rmssd_baseline * 0.8:
        # 正常 HRV
        hrv_risk = 0.0

    # === 3. 心率变异性分析 ===
    if len(features.heart_rate_history) >= 5:
        hr_values = features.heart_rate_history
        hr_mean = sum(hr_values) / len(hr_values)
        hr_std = (sum((v - hr_mean) ** 2 for v in hr_values) / len(hr_values)) ** 0.5

        # 心率变异性过大 → 情绪不稳定
        if hr_std > 15:
            hrv_risk = min(1.0, hrv_risk + 0.2)
            evidence.append(f"心率波动大(σ={hr_std:.1f}bpm，情绪不稳定)")

    # === 4. 综合风险 ===
    risk_score = (
        hr_anxiety_risk * 0.5
        + hrv_risk * hp.hr_variability_risk_weight
        + (1 - quality) * 0.1  # 信号质量惩罚
    )
    risk_score = min(1.0, max(0.0, risk_score))

    if not evidence:
        evidence.append(f"心率正常({hr:.0f}bpm)，HRV正常({rmssd:.1f}ms)")

    return {
        "risk_score": round(risk_score, 4),
        "heart_rate": round(hr, 1),
        "hrv_rmssd": round(rmssd, 1),
        "hr_anxiety_risk": round(hr_anxiety_risk, 4),
        "hrv_risk": round(hrv_risk, 4),
        "signal_quality": round(quality, 2),
        "evidence": evidence,
    }
