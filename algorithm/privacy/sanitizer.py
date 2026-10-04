"""
数据脱敏模块 —— 端侧特征上传前的隐私保护

核心功能：
    1. summarize(perception_output) -> FeatureSummary
       将感知层输出转换为脱敏特征摘要，确保不含原始文本或语音。

    2. validate_no_raw_data(summary) -> bool
       验证 FeatureSummary 不包含原始数据泄露。

    3. 差分隐私噪声注入（可选，epsilon 参数控制）

法规依据：
    - 《人工智能拟人化互动服务管理暂行办法》数据最小化原则：
      "收集个人信息应当限于实现目的的最小范围，不得过度收集。"
    - 《个人信息保护法》第六条：
      "处理个人信息应当具有明确、合理的目的，并应当与处理目的直接相关，
       采取对个人权益影响最小的方式。"

设计原则：
    - 只上传统计特征（概率分布、置信度等），不上传原始文本/语音
    - session_id 为随机 UUID，不关联用户真实身份
    - 差分隐私噪声确保即使数据被截获也无法反推个体信息
"""
from __future__ import annotations

import hashlib
import re
import secrets
import time
import uuid
from typing import Optional, Union

import numpy as np

from shared.dataclasses import EmotionResult, FeatureSummary


# ============================================================
# 正则模式：检测原始数据泄露
# ============================================================

# 中文句子模式（连续 5+ 中文字符 = 可能是原始文本）
_RAW_CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]{5,}")

# 英文句子模式（连续 10+ 字母 = 可能是原始文本）
_RAW_ENGLISH_PATTERN = re.compile(r"[a-zA-Z]{10,}")

# 邮箱/手机号模式
_RAW_PII_PATTERN = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.-]+"  # 邮箱
    r"|1[3-9]\d{9}"               # 中国手机号
    r"|\d{17}[\dXx]"              # 身份证号
)

# 音频波形数据模式（连续浮点数序列可能是原始波形）
_RAW_AUDIO_PATTERN = re.compile(r"(\d+\.\d+,\s*){20,}")


# ============================================================
# 核心接口：脱敏摘要
# ============================================================

def summarize(
    perception_output: Union[EmotionResult, dict],
    epsilon: Optional[float] = None,
    seed: Optional[int] = None,
) -> FeatureSummary:
    """将感知层输出转换为脱敏特征摘要

    依据《人工智能拟人化互动服务管理暂行办法》数据最小化原则：
    只提取统计特征用于云端分析，不上传原始文本或语音数据。

    Args:
        perception_output: 感知层输出（EmotionResult 或 dict）
        epsilon: 差分隐私参数（可选）
            - None: 不添加噪声
            - 1.0: 强隐私保护（噪声大）
            - 5.0: 弱隐私保护（噪声小）
            - 10.0+: 几乎无噪声
        seed: 随机种子（用于复现，默认使用 secrets 生成真随机）

    Returns:
        FeatureSummary: 脱敏后的特征摘要

    示例：
        >>> result = EmotionResult(
        ...     text_emotion_probs=[0.1, 0.2, 0.3, 0.1, 0.3],
        ...     audio_risk_prob=0.4,
        ...     confidence=0.85,
        ...     timestamp=time.time(),
        ...     evidence=["用户表达了焦虑情绪"],
        ... )
        >>> summary = summarize(result, epsilon=5.0)
        >>> summary.text_emotion_probs  # 可能添加了差分隐私噪声
        [0.12, 0.18, 0.31, 0.09, 0.30]
    """
    # 解析输入
    if isinstance(perception_output, dict):
        text_probs = perception_output.get("text_emotion_probs", [])
        audio_prob = perception_output.get("audio_risk_prob")
        confidence = perception_output.get("confidence", 0.0)
    elif isinstance(perception_output, EmotionResult):
        text_probs = perception_output.text_emotion_probs
        audio_prob = perception_output.audio_risk_prob
        confidence = perception_output.confidence
    else:
        raise TypeError(
            f"perception_output 必须是 EmotionResult 或 dict，"
            f"实际类型: {type(perception_output)}"
        )

    # 确保 text_probs 是 list[float]
    text_probs = [float(p) for p in text_probs] if text_probs else []

    # 差分隐私噪声注入
    # 使用拉普拉斯机制：noise ~ Laplace(0, sensitivity/epsilon)
    # 对于概率向量，sensitivity = 1/n（n 为向量维度）
    if epsilon is not None and epsilon > 0 and text_probs:
        rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        sensitivity = 1.0 / len(text_probs)
        scale = sensitivity / epsilon
        noise = rng.laplace(0, scale, len(text_probs))
        text_probs = [float(np.clip(p + n, 0.0, 1.0)) for p, n in zip(text_probs, noise)]

        # 音频概率也添加噪声
        if audio_prob is not None:
            audio_noise = rng.laplace(0, 1.0 / epsilon)
            audio_prob = float(np.clip(audio_prob + audio_noise, 0.0, 1.0))

    # session_id：每次会话随机生成 UUID（不关联用户真实身份）
    # 依据：《个人信息保护法》—— 不得过度收集个人信息
    session_id = str(uuid.UUID(bytes=secrets.token_bytes(16), version=4))

    return FeatureSummary(
        text_emotion_probs=text_probs,
        audio_risk_prob=audio_prob,
        confidence=float(confidence),
        session_id=session_id,
        timestamp=time.time(),
    )


# ============================================================
# 验证接口：检查数据泄露
# ============================================================

def validate_no_raw_data(summary: FeatureSummary) -> bool:
    """验证 FeatureSummary 不包含原始数据泄露

    检查项：
        1. text_emotion_probs 是否为纯浮点数列表（非原始文本）
        2. 所有字段是否不含中文句子、英文句子、PII 信息
        3. session_id 是否为合法 UUID 格式（非用户 ID）

    依据：《人工智能拟人化互动服务管理暂行办法》数据最小化原则 ——
    上传至云端的数据不得包含可识别个人身份的原始信息。

    Args:
        summary: 待验证的特征摘要

    Returns:
        True: 验证通过，无原始数据泄露
        False: 验证失败，检测到原始数据

    Raises:
        ValueError: 详细报告检测到的泄露类型
    """
    violations = []

    # 检查 1: text_emotion_probs 必须是纯浮点数列表
    if not isinstance(summary.text_emotion_probs, list):
        violations.append(f"text_emotion_probs 类型错误: {type(summary.text_emotion_probs)}")
    else:
        for i, v in enumerate(summary.text_emotion_probs):
            if not isinstance(v, (int, float)):
                violations.append(f"text_emotion_probs[{i}] 不是数值: {type(v)} = {v!r}")
            elif not (0.0 <= v <= 1.0):
                violations.append(f"text_emotion_probs[{i}] 超出 [0,1] 范围: {v}")

    # 检查 2: 所有字段序列化后不得包含原始文本模式
    all_values_str = _serialize_to_string(summary)

    if _RAW_CHINESE_PATTERN.search(all_values_str):
        violations.append("检测到中文字符串（可能是原始文本泄露）")

    if _RAW_ENGLISH_PATTERN.search(all_values_str):
        violations.append("检测到长英文字符串（可能是原始文本泄露）")

    if _RAW_PII_PATTERN.search(all_values_str):
        violations.append("检测到 PII 信息（邮箱/手机号/身份证号）")

    if _RAW_AUDIO_PATTERN.search(all_values_str):
        violations.append("检测到音频波形数据（可能是原始音频泄露）")

    # 检查 3: session_id 必须是 UUID 格式
    try:
        uuid.UUID(summary.session_id)
    except ValueError:
        violations.append(f"session_id 不是合法 UUID: {summary.session_id}")

    # 检查 4: confidence 必须在 [0, 1] 范围
    if not (0.0 <= summary.confidence <= 1.0):
        violations.append(f"confidence 超出 [0,1] 范围: {summary.confidence}")

    # 检查 5: audio_risk_prob 如果存在，必须在 [0, 1] 范围
    if summary.audio_risk_prob is not None:
        if not (0.0 <= summary.audio_risk_prob <= 1.0):
            violations.append(f"audio_risk_prob 超出 [0,1] 范围: {summary.audio_risk_prob}")

    if violations:
        raise ValueError(
            "FeatureSummary 数据泄露检测失败:\n" +
            "\n".join(f"  - {v}" for v in violations)
        )

    return True


def _serialize_to_string(summary: FeatureSummary) -> str:
    """将 FeatureSummary 序列化为字符串（用于正则检测）"""
    parts = []

    # 数值字段
    for p in summary.text_emotion_probs:
        parts.append(str(p))

    if summary.audio_risk_prob is not None:
        parts.append(str(summary.audio_risk_prob))

    parts.append(str(summary.confidence))
    parts.append(summary.session_id)
    parts.append(str(summary.timestamp))

    return " ".join(parts)


# ============================================================
# 批量脱敏
# ============================================================

def summarize_batch(
    perception_outputs: list[Union[EmotionResult, dict]],
    epsilon: Optional[float] = None,
) -> list[FeatureSummary]:
    """批量脱敏

    Args:
        perception_outputs: 感知层输出列表
        epsilon: 差分隐私参数

    Returns:
        FeatureSummary 列表
    """
    return [summarize(p, epsilon=epsilon) for p in perception_outputs]


# ============================================================
# 差分隐私工具函数
# ============================================================

def compute_epsilon_for_budget(
    total_budget: float,
    n_queries: int,
    composition: str = "basic",
) -> float:
    """根据隐私预算计算每次查询的 epsilon

    依据差分隐私组合定理：
    - 基本组合：total_epsilon = n * epsilon → epsilon = total / n
    - 高级组合：total_epsilon ≈ sqrt(2n * log(1/delta)) * epsilon + n * epsilon * (e^epsilon - 1)

    Args:
        total_budget: 总隐私预算
        n_queries: 查询次数
        composition: 组合方式（"basic" 或 "advanced"）

    Returns:
        每次查询的 epsilon
    """
    if composition == "basic":
        return total_budget / n_queries
    elif composition == "advanced":
        # 简化版高级组合（delta = 1e-5）
        delta = 1e-5
        # epsilon ≈ total_budget / (sqrt(2n * log(1/delta)) + n * (e^epsilon - 1))
        # 近似求解
        epsilon = total_budget / (np.sqrt(2 * n_queries * np.log(1 / delta)) + n_queries * 0.1)
        return float(epsilon)
    else:
        raise ValueError(f"未知的组合方式: {composition}")
