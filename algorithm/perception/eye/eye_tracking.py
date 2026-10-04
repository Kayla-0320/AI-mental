"""
眼动模式分析模块 —— 通过摄像头提取注视与眨眼特征

临床依据：
- Armstrong & Olatunji (2012): 焦虑青少年的负面注意偏向
- Peckham et al. (2010): 抑郁群体的注视模式 meta 分析
- Ehlers & Margraf (2002): 眨眼频率与焦虑/解离

分析维度：
- 眨眼频率：过低→注意力过度集中/解离，过高→焦虑/干眼
- 注视方向：持续向下注视→回避/抑郁
- 注意力分散度：注视频繁转移→焦虑/ADHD
- 瞳孔变化估计（受光照限制，简化模型）

年龄差异化：
- 初中：眨眼基线 18次/分（发育期偏高）
- 高中：眨眼基线 16次/分
- 大学：眨眼基线 15次/分（接近成人）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from perception.age_config import EyeMovementAgeParams, AgeProfile


@dataclass
class EyeMovementFeatures:
    """眼动模式特征"""

    # 眨眼频率 (次/分钟)
    blink_rate: float = 0.0

    # 向下注视比例 (0-1)：向下注视时间 / 总注视时间
    downward_gaze_ratio: float = 0.0

    # 注意力分散度 (0-1)：注视点变化频率
    attention_scatter: float = 0.0

    # 平均注视方向角度 (度, 0=正前方, 正=向下)
    mean_gaze_angle: float = 0.0

    # 信号质量 (0-1)
    signal_quality: float = 0.0


def analyze_eye_movement_risk(
    features: EyeMovementFeatures,
    age_profile: Optional[AgeProfile] = None,
) -> dict:
    """分析眼动模式风险

    Args:
        features: 眼动特征
        age_profile: 年龄画像

    Returns:
        {
            "risk_score": float (0-1),
            "blink_risk": float,
            "gaze_risk": float,
            "attention_risk": float,
            "evidence": list[str],
        }
    """
    if age_profile is not None:
        ep = age_profile.eye
    else:
        ep = EyeMovementAgeParams()

    evidence: list[str] = []
    quality = features.signal_quality

    if quality < 0.3:
        return {
            "risk_score": 0.0,
            "blink_risk": 0.0,
            "gaze_risk": 0.0,
            "attention_risk": 0.0,
            "evidence": ["眼动信号质量不足"],
        }

    # === 1. 眨眼频率风险 ===
    blink_risk = 0.0
    if features.blink_rate > 0:
        if features.blink_rate < ep.low_blink_threshold:
            # 眨眼过少 → 注意力过度集中 / 解离
            blink_risk = min(1.0,
                (ep.low_blink_threshold - features.blink_rate) / ep.low_blink_threshold)
            evidence.append(f"眨眼过少({features.blink_rate:.0f}次/分，可能解离)")
        elif features.blink_rate > ep.high_blink_threshold:
            # 眨眼过多 → 焦虑
            blink_risk = min(1.0,
                (features.blink_rate - ep.high_blink_threshold) / 15.0)
            evidence.append(f"眨眼偏多({features.blink_rate:.0f}次/分，焦虑指标)")

    # === 2. 注视方向风险 ===
    gaze_risk = 0.0
    if features.downward_gaze_ratio > ep.downward_gaze_risk_threshold:
        gaze_risk = min(1.0,
            (features.downward_gaze_ratio - ep.downward_gaze_risk_threshold) / 0.3)
        evidence.append(f"持续向下注视({features.downward_gaze_ratio:.0%}，回避/抑郁指标)")

    # === 3. 注意力分散 ===
    attention_risk = 0.0
    if features.attention_scatter > ep.attention_scatter_threshold:
        attention_risk = min(1.0,
            (features.attention_scatter - ep.attention_scatter_threshold) / 0.3)
        evidence.append(f"注意力过度分散(散度={features.attention_scatter:.2f})")

    # === 4. 综合风险 ===
    risk_score = (
        blink_risk * 0.35
        + gaze_risk * 0.35
        + attention_risk * 0.3
    )
    risk_score = min(1.0, max(0.0, risk_score))

    if not evidence:
        evidence.append(f"眼动模式正常(眨眼{features.blink_rate:.0f}次/分)")

    return {
        "risk_score": round(risk_score, 4),
        "blink_risk": round(blink_risk, 4),
        "gaze_risk": round(gaze_risk, 4),
        "attention_risk": round(attention_risk, 4),
        "evidence": evidence,
    }
