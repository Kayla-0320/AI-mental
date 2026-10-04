"""
语音深层语义分析模块 —— 对语音识别文本的深度语言分析

临床依据：
- Rude et al. (2004): 第一人称代词使用与抑郁
- Stirman & Pennebaker (2001): 语言标记与自杀风险
- Tackman et al. (2019): 自我指涉语言与抑郁的元分析
- Eich et al. (2018): 绝对化表达与情绪障碍

分析维度：
- 第一人称单数密度 ("我" 的过度使用 → 自我关注过度 → 抑郁)
- 第一人称复数密度 ("我们" 的少用 → 社交疏离)
- 绝对化表达比例 ("总是/从不/一定" → 认知僵化)
- 自我指涉总密度 (所有自我相关词汇的比例)
- 消极情感词比例 (消极词 / 总情感词)

年龄差异化：
- 初中：第一人称阈值 0.15，表达直接
- 高中：第一人称阈值 0.15，开始隐晦
- 大学：第一人称阈值 0.12，表达更成熟（正常自我指涉更高）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from perception.age_config import VoiceSemanticsAgeParams, AgeProfile


@dataclass
class VoiceSemanticsFeatures:
    """语音深层语义特征"""

    # 语音识别文本
    speech_text: str = ""

    # 历史语音文本列表
    speech_text_history: list[str] = field(default_factory=list)


# 中文分词辅助模式
_FIRST_PERSON_SINGULAR = re.compile(r'我(?!们)')
_FIRST_PERSON_PLURAL = re.compile(r'我们')
_SELF_REFERENTIAL = re.compile(r'我|自己|本人|我的|俺|咱')
_ABSOLUTIST = re.compile(r'总是|从不|永远|绝对|一定|所有|全部|完全|根本|每次|必须|应该')
_NEGATIVE_AFFECT = re.compile(
    r'难过|伤心|焦虑|害怕|恐惧|愤怒|痛苦|绝望|孤独|寂寞|空虚|无聊|'
    r'烦|累|疲|丧|衰|暗|灰|冷|痛|哭|死|恨|厌'
)
_POSITIVE_AFFECT = re.compile(
    r'开心|快乐|高兴|幸福|棒|好|喜欢|爱|感恩|温暖|阳光|'
    r'笑|甜|美|亮|暖'
)


def analyze_voice_semantics_risk(
    features: VoiceSemanticsFeatures,
    age_profile: Optional[AgeProfile] = None,
) -> dict:
    """分析语音深层语义风险

    Args:
        features: 语音语义特征
        age_profile: 年龄画像

    Returns:
        {
            "risk_score": float (0-1),
            "first_person_singular_ratio": float,
            "first_person_plural_ratio": float,
            "absolutist_ratio": float,
            "self_referential_density": float,
            "negative_affect_ratio": float,
            "evidence": list[str],
        }
    """
    if age_profile is not None:
        vs = age_profile.voice_semantics
    else:
        vs = VoiceSemanticsAgeParams()

    text = features.speech_text
    evidence: list[str] = []

    if not text or len(text.strip()) < 5:
        return {
            "risk_score": 0.0,
            "first_person_singular_ratio": 0.0,
            "first_person_plural_ratio": 0.0,
            "absolutist_ratio": 0.0,
            "self_referential_density": 0.0,
            "negative_affect_ratio": 0.0,
            "evidence": ["语音文本过短，无法分析语义"],
        }

    text_len = max(len(text), 1)

    # === 1. 第一人称单数密度 ===
    fps_count = len(_FIRST_PERSON_SINGULAR.findall(text))
    fps_ratio = fps_count / text_len

    fps_risk = 0.0
    if fps_ratio > vs.first_person_singular_threshold:
        fps_risk = min(1.0,
            (fps_ratio - vs.first_person_singular_threshold) / 0.1)
        evidence.append(f"第一人称单数过度使用(密度={fps_ratio:.3f})")

    # === 2. 第一人称复数密度 ===
    fpp_count = len(_FIRST_PERSON_PLURAL.findall(text))
    fpp_ratio = fpp_count / text_len

    fpp_risk = 0.0
    if fpp_ratio < vs.first_person_plural_threshold and text_len > 20:
        fpp_risk = 0.2  # 少用"我们" → 社交疏离信号
        evidence.append(f"第一人称复数极少(密度={fpp_ratio:.4f}，社交疏离指标)")

    # === 3. 绝对化表达 ===
    abs_count = len(_ABSOLUTIST.findall(text))
    abs_ratio = abs_count / text_len

    abs_risk = 0.0
    if abs_ratio > vs.absolutist_expression_threshold:
        abs_risk = min(1.0,
            (abs_ratio - vs.absolutist_expression_threshold) / 0.03)
        evidence.append(f"绝对化表达偏多(密度={abs_ratio:.3f})")

    # === 4. 自我指涉总密度 ===
    self_count = len(_SELF_REFERENTIAL.findall(text))
    self_density = self_count / text_len

    self_risk = 0.0
    if self_density > vs.self_referential_density_threshold:
        self_risk = min(1.0,
            (self_density - vs.self_referential_density_threshold) / 0.15)
        evidence.append(f"自我指涉密度高({self_density:.3f})")

    # === 5. 消极情感词比例 ===
    neg_count = len(_NEGATIVE_AFFECT.findall(text))
    pos_count = len(_POSITIVE_AFFECT.findall(text))
    total_affect = neg_count + pos_count
    neg_ratio = neg_count / max(total_affect, 1) if total_affect > 0 else 0.0

    neg_risk = 0.0
    if neg_ratio > vs.negative_affect_ratio_threshold and total_affect > 0:
        neg_risk = min(1.0,
            (neg_ratio - vs.negative_affect_ratio_threshold) / 0.3)
        evidence.append(f"消极情感词占比高({neg_ratio:.0%})")

    # === 6. 综合风险 ===
    risk_score = (
        fps_risk * 0.25
        + fpp_risk * 0.1
        + abs_risk * 0.15
        + self_risk * 0.25
        + neg_risk * 0.25
    )
    risk_score = min(1.0, max(0.0, risk_score))

    if not evidence:
        evidence.append("语音语义正常")

    return {
        "risk_score": round(risk_score, 4),
        "first_person_singular_ratio": round(fps_ratio, 4),
        "first_person_plural_ratio": round(fpp_ratio, 4),
        "absolutist_ratio": round(abs_ratio, 4),
        "self_referential_density": round(self_density, 4),
        "negative_affect_ratio": round(neg_ratio, 4),
        "evidence": evidence,
    }
