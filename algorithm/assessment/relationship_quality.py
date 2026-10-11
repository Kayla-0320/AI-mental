"""
关系质量指数 (RQI) —— 多信号融合的关系建模算法

算法创新：
    传统方法：session_count → 关系阶段（简单计数器）
    我们的方法：RQI = f(互动频率, 自我暴露深度, 话题多样性, 信任信号, 情感共鸣)
    
    RQI 使用加权多信号融合 + 时间衰减，比单纯计数更能反映真实关系质量。

数学模型：
    RQI(t) = α * RQI(t-1) + (1-α) * Σ(w_i * signal_i(t))
    
    其中：
    - signal_1: 互动频率（归一化）
    - signal_2: 自我暴露深度（消息长度 + 情感词汇密度）
    - signal_3: 话题多样性（唯一话题数 / 总话题数）
    - signal_4: 信任信号计数（归一化）
    - signal_5: 情感共鸣（情绪匹配度）
    - w_i: 信号权重（可学习）
    - α: 时间衰减系数（0.7）

关系阶段划分（基于 RQI 分位数）：
    - 陌生期：RQI < 0.3
    - 熟悉期：0.3 ≤ RQI < 0.6
    - 信任期：RQI ≥ 0.6

设计依据：
    - Horvath & Greenberg (1989) 工作联盟量表 (WAI)
    - 治疗关系的多维性：不能只用时间衡量
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# RQI 信号权重（可配置）
# ============================================================

_RQI_WEIGHTS = {
    "interaction_frequency": 0.20,  # 互动频率
    "self_disclosure": 0.25,        # 自我暴露深度
    "topic_diversity": 0.15,        # 话题多样性
    "trust_signals": 0.25,          # 信任信号
    "emotional_resonance": 0.15,    # 情感共鸣
}

_RQI_DECAY = 0.7  # 时间衰减系数（EMA）


# ============================================================
# RQI 计算
# ============================================================

def compute_rqI(
    session_count: int,
    avg_message_length: float,
    emotional_vocabulary_density: float,
    unique_topics: int,
    total_topics: int,
    trust_signal_count: int,
    emotion_match_score: float,
    max_sessions: int = 50,
    max_trust_signals: int = 10,
) -> float:
    """计算关系质量指数 (RQI)
    
    Args:
        session_count: 对话次数
        avg_message_length: 平均消息长度（字符数）
        emotional_vocabulary_density: 情感词汇密度 (0-1)
        unique_topics: 唯一话题数
        total_topics: 总话题提及次数
        trust_signal_count: 信任信号计数
        emotion_match_score: 情感共鸣得分 (0-1)
        max_sessions: 最大会话数（归一化用）
        max_trust_signals: 最大信任信号数（归一化用）
    
    Returns:
        float: RQI 得分 (0-1)
    """
    # 信号 1：互动频率（对数归一化）
    freq_signal = math.log(1 + session_count) / math.log(1 + max_sessions)
    freq_signal = min(1.0, freq_signal)
    
    # 信号 2：自我暴露深度（消息长度 + 情感词汇）
    # 长消息 + 高情感词汇密度 = 深度自我暴露
    length_signal = min(1.0, avg_message_length / 200)  # 200 字符为满分
    disclosure_signal = 0.6 * length_signal + 0.4 * emotional_vocabulary_density
    
    # 信号 3：话题多样性
    if total_topics > 0:
        diversity_signal = unique_topics / total_topics
    else:
        diversity_signal = 0.0
    
    # 信号 4：信任信号
    trust_signal = min(1.0, trust_signal_count / max_trust_signals)
    
    # 信号 5：情感共鸣
    resonance_signal = emotion_match_score
    
    # 加权融合
    rqi = (
        _RQI_WEIGHTS["interaction_frequency"] * freq_signal +
        _RQI_WEIGHTS["self_disclosure"] * disclosure_signal +
        _RQI_WEIGHTS["topic_diversity"] * diversity_signal +
        _RQI_WEIGHTS["trust_signals"] * trust_signal +
        _RQI_WEIGHTS["emotional_resonance"] * resonance_signal
    )
    
    return round(min(1.0, max(0.0, rqi)), 4)


def rqi_to_relationship_depth(rqi: float) -> str:
    """将 RQI 转换为关系阶段。

    ⚠️ 返回**中文阶段名**（与模块 docstring、注入 prompt 的默认值、
    前端 ``RelationshipStageIndicator`` 的分档标签三处一致）。
    原实现返回的是英文 ``stranger/acquainted/trusted``，与另外三处对不上：
    prompt 里的默认值是"陌生期"、前端 ``getStage`` 也按中文渲染，
    导致后端算出的阶段在 UI 与 prompt 里都落不进任何一档。

    Args:
        rqi: 关系质量指数 (0-1)

    Returns:
        str: 关系阶段，取值 ``陌生期 / 熟悉期 / 信任期``
    """
    if rqi < 0.3:
        return "陌生期"
    elif rqi < 0.6:
        return "熟悉期"
    else:
        return "信任期"


# ============================================================
# 会话质量评估
# ============================================================

def assess_session_quality(
    user_message: str,
    ai_response: str,
    emotion_probs: list[float],
    trust_keywords: list[str],
) -> dict:
    """评估单次会话质量，提取 RQI 信号
    
    Args:
        user_message: 用户消息
        ai_response: AI 回复
        emotion_probs: 情绪概率分布 [快乐，悲伤，焦虑，愤怒，中性]
        trust_keywords: 信任关键词列表
    
    Returns:
        dict: 会话质量指标
    """
    # 消息长度
    msg_length = len(user_message)
    
    # 情感词汇密度（简化：负性情绪概率和）
    emotional_density = sum(emotion_probs[1:4]) if len(emotion_probs) >= 5 else 0.0
    
    # 信任信号检测
    trust_signal_count = sum(
        1 for kw in trust_keywords
        if kw in user_message
    )
    
    # 话题检测（简化：按句子分割）
    sentences = user_message.split("。")
    topic_count = len([s for s in sentences if len(s.strip()) > 5])
    
    return {
        "message_length": msg_length,
        "emotional_density": emotional_density,
        "trust_signal_count": trust_signal_count,
        "topic_count": topic_count,
        "response_ratio": len(ai_response) / max(1, msg_length),
    }


# ============================================================
# 信任关键词库
# ============================================================

TRUST_KEYWORDS = [
    # 自我暴露
    "秘密", "从来没告诉过别人", "只有你知道", "别告诉",
    # 情感依赖
    "只有你能懂", "你明白", "你懂", "只有你",
    # 未来规划
    "下次", "以后", "到时候", "等我",
    # 亲密称呼
    "朋友", "兄弟", "姐妹",
]
