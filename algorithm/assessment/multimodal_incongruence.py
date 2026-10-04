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

def classify_incongruence(
    text_emotion: list[float],
    voice_emotion: list[float],
    mis_history: list[float],
    k: float = 1.5,
) -> dict:
    """判定矛盾类型
    
    Args:
        text_emotion: 文本情绪向量
        voice_emotion: 声学情绪向量
        mis_history: 历史 MIS 值
        k: 灵敏度系数
    
    Returns:
        dict: {
            "type": str,  # congruent/masking/performative
            "mis": float,
            "threshold": float,
            "confidence": float,
        }
    """
    # 计算 MIS
    mis = compute_mis(text_emotion, voice_emotion)
    
    # 计算自适应阈值
    threshold = adaptive_threshold(mis_history, k)
    
    # 判定是否矛盾
    if mis <= threshold:
        return {
            "type": IncongruenceType.CONGRUENT,
            "mis": round(mis, 4),
            "threshold": round(threshold, 4),
            "confidence": 0.8,
        }
    
    # 矛盾类型判定
    text_negative = sum(text_emotion[1:4])  # 悲伤 + 焦虑 + 愤怒
    voice_arousal = voice_emotion[0] + voice_emotion[2] + voice_emotion[3]  # 快乐 + 焦虑 + 愤怒
    
    if text_negative < 0.3 and voice_arousal > 0.5:
        # 言语平静但声音高唤醒 → 掩饰
        confidence = min(0.95, 0.5 + (mis - threshold) * 2)
        return {
            "type": IncongruenceType.MASKING,
            "mis": round(mis, 4),
            "threshold": round(threshold, 4),
            "confidence": round(confidence, 4),
        }
    elif text_negative > 0.5 and voice_arousal < 0.3:
        # 言语痛苦但声音低唤醒 → 表演性
        confidence = min(0.95, 0.5 + (mis - threshold) * 2)
        return {
            "type": IncongruenceType.PERFORMATIVE,
            "mis": round(mis, 4),
            "threshold": round(threshold, 4),
            "confidence": round(confidence, 4),
        }
    else:
        # 其他矛盾类型
        return {
            "type": "other_incongruence",
            "mis": round(mis, 4),
            "threshold": round(threshold, 4),
            "confidence": 0.5,
        }
