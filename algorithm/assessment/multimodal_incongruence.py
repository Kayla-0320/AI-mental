"""
多模态矛盾分数 (MIS) —— 基于余弦距离的跨模态矛盾检测算法

算法创新：
    传统方法：加权平均融合多模态（矛盾被平均掉）
    我们的方法：MIS = 1 - cosine_similarity(text_emotion, voice_emotion)
    
    当 MIS > 自适应阈值时，判定为矛盾信号。
    自适应阈值基于用户历史 MIS 分布动态调整。

数学模型：
    MIS(t) = 1 - (text_vec · voice_vec) / (||text_vec|| * ||voice_vec||)
    
    自适应阈值：
    threshold(t) = μ_MIS + k * σ_MIS
    
    其中：
    - μ_MIS: 用户历史 MIS 均值
    - σ_MIS: 用户历史 MIS 标准差
    - k: 灵敏度系数（默认 1.5）

矛盾类型判定：
    - masking（掩饰）: text_calm AND voice_aroused AND MIS > threshold
    - performative（表演性）: text_distressed AND voice_calm AND MIS > threshold
    - congruent（一致）: MIS ≤ threshold

设计依据：
    - Ekman (2003) 跨通道情绪不一致理论
    - 余弦距离衡量情绪向量方向差异，不受幅度影响
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 矛盾类型
# ============================================================

class IncongruenceType:
    CONGRUENT = "congruent"       # 一致
    MASKING = "masking"           # 掩饰（言语平静但声音紧张）
    PERFORMATIVE = "performative" # 表演性（言语痛苦但声音平淡）


# ============================================================
# 余弦相似度计算
# ============================================================

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦相似度
    
    Args:
        vec_a: 向量 A
        vec_b: 向量 B
    
    Returns:
        float: 余弦相似度 (-1 到 1)
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("向量长度不一致")
    
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """计算两个向量的余弦距离
    
    Args:
        vec_a: 向量 A
        vec_b: 向量 B
    
    Returns:
        float: 余弦距离 (0 到 2)
    """
    return 1.0 - cosine_similarity(vec_a, vec_b)


# ============================================================
# 声学特征 → 情绪向量转换
# ============================================================

def acoustic_to_emotion_vector(
    energy_mean: float,
    pitch_range: float,
    speech_rate: float,
    pause_ratio: float,
) -> list[float]:
    """将声学特征转换为情绪向量 [快乐，悲伤，焦虑，愤怒，中性]
    
    基于声学特征与情绪的映射关系（Scherer, 2003）：
    - 高能量 + 高基频范围 → 快乐/愤怒
    - 低能量 + 低基频 → 悲伤
    - 高语速 + 高能量 → 焦虑
    - 高停顿比例 → 中性/悲伤
    
    Args:
        energy_mean: 能量均值 (0-1)
        pitch_range: 基频范围 (0-1)
        speech_rate: 语速 (0-1)
        pause_ratio: 停顿比例 (0-1)
    
    Returns:
        list[float]: 情绪向量 [快乐，悲伤，焦虑，愤怒，中性]
    """
    # 快乐：高能量 + 高基频范围 + 中等语速
    happy = 0.3 * energy_mean + 0.3 * pitch_range + 0.2 * (1 - abs(speech_rate - 0.5))
    
    # 悲伤：低能量 + 低基频 + 高停顿
    sad = 0.3 * (1 - energy_mean) + 0.3 * (1 - pitch_range) + 0.4 * pause_ratio
    
    # 焦虑：高语速 + 高能量 + 中等基频
    anxious = 0.4 * speech_rate + 0.3 * energy_mean + 0.3 * (1 - abs(pitch_range - 0.5))
    
    # 愤怒：高能量 + 高基频 + 低停顿
    angry = 0.4 * energy_mean + 0.3 * pitch_range + 0.3 * (1 - pause_ratio)
    
    # 中性：中等所有特征
    neutral = (
        0.2 * (1 - abs(energy_mean - 0.4)) +
        0.2 * (1 - abs(pitch_range - 0.3)) +
        0.2 * (1 - abs(speech_rate - 0.4)) +
        0.4 * (1 - abs(pause_ratio - 0.3))
    )
    
    # 归一化
    total = happy + sad + anxious + angry + neutral
    if total < 1e-10:
        return [0.2, 0.2, 0.2, 0.2, 0.2]
    
    return [
        max(0.0, happy / total),
        max(0.0, sad / total),
        max(0.0, anxious / total),
        max(0.0, angry / total),
        max(0.0, neutral / total),
    ]


# ============================================================
# MIS 计算
# ============================================================

def compute_mis(
    text_emotion: list[float],
    voice_emotion: list[float],
) -> float:
    """计算多模态矛盾分数 (MIS)
    
    Args:
        text_emotion: 文本情绪向量 [快乐，悲伤，焦虑，愤怒，中性]
        voice_emotion: 声学情绪向量 [快乐，悲伤，焦虑，愤怒，中性]
    
    Returns:
        float: MIS 得分 (0 到 2，0=完全一致，2=完全相反)
    """
    return cosine_distance(text_emotion, voice_emotion)


# ============================================================
# 自适应阈值
# ============================================================

def adaptive_threshold(
    mis_history: list[float],
    k: float = 1.5,
    min_samples: int = 5,
) -> float:
    """计算自适应矛盾阈值
    
    基于用户历史 MIS 分布：threshold = μ + k * σ
    
    Args:
        mis_history: 历史 MIS 值列表
        k: 灵敏度系数（越大越不敏感）
        min_samples: 最小样本数（不足时使用默认阈值）
    
    Returns:
        float: 自适应阈值
    """
    if len(mis_history) < min_samples:
        return 0.3  # 默认阈值
    
    mean_mis = sum(mis_history) / len(mis_history)
    variance = sum((m - mean_mis) ** 2 for m in mis_history) / len(mis_history)
    std_mis = math.sqrt(variance)
    
    return mean_mis + k * std_mis


# ============================================================
# 矛盾类型判定
# ============================================================

# 通道方向性判据的边界（2026-10-10 标定）。
#
# 标定口径：先把客户端的原始声学量纲（dBFS / Hz / 原始语速刻度）经
# `tamer_framework.normalize_*` 归一化，再在真实取值范围内扫描声学唤醒
# （happy + anxious + angry），实测（`_tamer_arousal_probe.py`）：
#
#   很平静（轻声、低音、慢、多停顿） arousal ≈ 0.35
#   平静偏中性                     arousal ≈ 0.46
#   中性说话                       arousal ≈ 0.54
#   略紧张                         arousal ≈ 0.63
#   紧张/激动（大声、高音、快）       arousal ≈ 0.71
#   非常激动                       arousal ≈ 0.77
#   （理论下界 ≈ 0.14，上界 ≈ 0.65~0.77 视输入范围）
#
# 因此 0.62 以上算"声音紧张"，0.46 以下算"声音平静"，中间留一条**死区**：
# 声学证据落在死区时不下"掩饰/表演"的结论（宁可漏报不误报）。
# 文本侧 0.45 / 0.55 把"言语平静 / 言语痛苦"切开（实测平静文本 neg=0.30、
# 痛苦文本 neg=0.93，留有余量）。
_TEXT_CALM_MAX = 0.45       # 文本负性 ≤ 此值视为"言语平静"
_TEXT_DISTRESS_MIN = 0.55   # 文本负性 ≥ 此值视为"言语痛苦"
_VOICE_AROUSAL_HIGH = 0.62  # 声学唤醒 ≥ 此值视为"声音紧张"
_VOICE_AROUSAL_LOW = 0.46   # 声学唤醒 ≤ 此值视为"声音平静"


def _channel_confidence(*margins: float) -> float:
    """把"离判定边界还有多远"折成 0.55~0.95 的置信度。

    Args:
        *margins: 每个已满足条件的归一化余量（0 = 刚好压线，1 = 远离边界）。

    Returns:
        float: 置信度。
    """
    if not margins:
        return 0.55
    m = min(max(0.0, min(1.0, x)) for x in margins)
    return round(min(0.95, 0.55 + 0.4 * m), 4)


def classify_incongruence(
    text_emotion: list[float],
    voice_emotion: list[float],
    mis_history: list[float],
    k: float = 1.5,
) -> dict:
    """判定跨模态矛盾类型。

    ⚠️ 判据说明（2026-10-10 修正，请连同下面的实测一起读）
    ---------------------------------------------------------
    原实现把 ``MIS > threshold`` 当作**唯一开关**。实测证明这条开关在本项目的
    特征空间里失效，而且方向是反的：

    | 样本 | MIS |
    | --- | --- |
    | 一致：平静文本 + 平静声音 | 0.33 |
    | 一致：痛苦文本 + 紧张声音 | **0.59** |
    | 掩饰：平静文本 + 紧张声音 | 0.27 |
    | 表演：痛苦文本 + 平静声音 | **0.14** |

    真正一致的"痛苦 + 紧张"反而比教科书式的掩饰样本 MIS 更高。
    原因是两条通道的向量来源不同：文本情绪是分类器输出的概率向量，声学情绪是
    :func:`acoustic_to_emotion_vector` 这个粗粒度启发式映射的输出，其"形状"主要
    由映射本身决定，而非由跨通道一致性决定；再加上两个概率向量的余弦相似度天然
    偏高（随机分布对实测中位数仅 0.20、p90 仅 0.42），默认阈值 0.3 几乎不可达。

    因此改用**通道方向性对比**作为主判据 —— 这正是本模块 docstring 与 Ekman
    跨通道不一致理论所描述的量，也是"掩饰/表演"的临床定义：

    - ``masking``：言语平静（低负性）∧ 声音紧张（高唤醒）
    - ``performative``：言语痛苦（高负性）∧ 声音平静（低唤醒）

    ``MIS`` 仍按原公式计算并返回，供**展示与可视化**（前端 MIS 数值、矛盾提示卡）
    以及 confidence 调制使用，但不再单独决定"是否矛盾"。

    Args:
        text_emotion: 文本情绪概率 ``[快乐, 悲伤, 焦虑, 愤怒, 中性]``。
        voice_emotion: 声学情绪向量，同序 5 维。
        mis_history: 历史 MIS 值（保留形参以兼容原接口并计算自适应阈值）。
        k: 自适应阈值灵敏度系数（仅影响返回的 ``threshold`` 字段）。

    Returns:
        dict: ``{type, mis, threshold, confidence}``；``type`` ∈
        ``congruent / masking / performative / other_incongruence``。
    """
    # MIS 照原公式算，仅作幅度展示（判据见上方说明，不作为开关）
    mis = compute_mis(text_emotion, voice_emotion)
    threshold = adaptive_threshold(mis_history, k)

    # 通道方向性特征
    text_negative = sum(text_emotion[1:4])          # 悲伤 + 焦虑 + 愤怒
    text_calm = text_negative <= _TEXT_CALM_MAX
    text_distressed = text_negative >= _TEXT_DISTRESS_MIN
    voice_arousal = voice_emotion[0] + voice_emotion[2] + voice_emotion[3]  # 快乐 + 焦虑 + 愤怒
    voice_aroused = voice_arousal >= _VOICE_AROUSAL_HIGH
    voice_flat = voice_arousal <= _VOICE_AROUSAL_LOW

    # 掩饰型：言语平静 ∧ 声音紧张
    if text_calm and voice_aroused:
        return {
            "type": IncongruenceType.MASKING,
            "mis": round(mis, 4),
            "threshold": round(threshold, 4),
            "confidence": _channel_confidence(
                (_TEXT_CALM_MAX - text_negative) / _TEXT_CALM_MAX,
                (voice_arousal - _VOICE_AROUSAL_HIGH) / (1.0 - _VOICE_AROUSAL_HIGH),
            ),
        }

    # 表演型：言语痛苦 ∧ 声音平静
    if text_distressed and voice_flat:
        return {
            "type": IncongruenceType.PERFORMATIVE,
            "mis": round(mis, 4),
            "threshold": round(threshold, 4),
            "confidence": _channel_confidence(
                (text_negative - _TEXT_DISTRESS_MIN) / (1.0 - _TEXT_DISTRESS_MIN),
                (_VOICE_AROUSAL_LOW - voice_arousal) / _VOICE_AROUSAL_LOW,
            ),
        }

    # 通道方向一致（平静配平静 / 痛苦配紧张）
    if (text_calm and voice_flat) or (text_distressed and voice_aroused):
        return {
            "type": IncongruenceType.CONGRUENT,
            "mis": round(mis, 4),
            "threshold": round(threshold, 4),
            "confidence": 0.8,
        }

    # 中间地带（例如文本中性 + 声音中性）：既不算一致，也不算这两类矛盾
    return {
        "type": "other_incongruence",
        "mis": round(mis, 4),
        "threshold": round(threshold, 4),
        "confidence": 0.5,
    }

