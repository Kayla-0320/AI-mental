"""
多模态 Late Fusion 模块

实现三种融合策略：
1. 加权平均（Weighted Average）
2. Stacking（LogisticRegression meta-learner）
3. 基于置信度的动态加权（Confidence-based Dynamic Weighting）

依赖：
- text_emotion 模块输出：5 维概率分布（焦虑/抑郁/愤怒/中性/积极）
- 语音模块输出：audio_risk_prob（float）或 None

所有模块间通信使用 shared/dataclasses.py 中的 EmotionResult。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

from shared.dataclasses import EmotionResult

# ============================================================
# 配置加载
# ============================================================

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "fusion.yaml"


def load_fusion_config(config_path: Optional[str] = None) -> dict:
    """加载融合配置 YAML

    Args:
        config_path: 配置文件路径，默认使用 config/fusion.yaml

    Returns:
        配置字典
    """
    path = Path(config_path) if config_path else _CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 风险等级判定
# ============================================================

# 风险等级阈值（来源：参考 PHQ-9 临床分级标准 Kroenke et al., 2001，
# 结合青少年心理健康筛查实践调整）
# low:    0.0 ~ 0.3  正常情绪波动
# medium: 0.3 ~ 0.6  需要关注，建议轻度干预
# high:   0.6 ~ 0.8  建议专业评估
# crisis: 0.8 ~ 1.0  立即触发升级流程
def _risk_level_from_score(score: float, thresholds: dict) -> str:
    """根据风险评分判定风险等级

    Args:
        score: 风险评分 [0.0, 1.0]
        thresholds: 阈值配置 {"low": 0.3, "medium": 0.6, "high": 0.8}

    Returns:
        风险等级字符串：low / medium / high / crisis
    """
    if score >= thresholds.get("high", 0.8):
        return "crisis"
    elif score >= thresholds.get("medium", 0.6):
        return "high"
    elif score >= thresholds.get("low", 0.3):
        return "medium"
    else:
        return "low"


# ============================================================
# 策略 1：加权平均
# ============================================================

def fuse_weighted_average(
    text_probs: np.ndarray,
    audio_probs: Optional[np.ndarray],
    config: Optional[dict] = None,
) -> np.ndarray:
    """加权平均融合

    将文本和语音的概率分布按配置权重加权求和。
    若 audio_probs 为 None，降级为文本单模态。

    Args:
        text_probs: 文本情绪概率分布，形状 (n_samples, 5) 或 (5,)
        audio_probs: 语音情绪概率分布，形状同 text_probs；None 时降级
        config: 融合配置，默认从 fusion.yaml 加载

    Returns:
        融合后的概率分布，形状同输入
    """
    if config is None:
        config = load_fusion_config()

    wa_config = config["weighted_average"]
    text_weight = wa_config["text_weight"]
    audio_weight = wa_config["audio_weight"]

    # 模态缺失降级
    if audio_probs is None:
        return text_probs.copy()

    fused = text_weight * text_probs + audio_weight * audio_probs
    # 归一化确保概率和为 1
    fused = fused / fused.sum(axis=-1, keepdims=True)
    return fused


# ============================================================
# 策略 2：Stacking（LogisticRegression meta-learner）
# ============================================================

class StackingFusion:
    """Stacking 融合器 —— 使用 LogisticRegression 作为 meta-learner

    将文本和语音的概率分布拼接为特征向量，
    用 LogisticRegression 学习最优融合权重。

    使用方式：
        1. fit(text_probs, audio_probs, labels) 训练 meta-learner
        2. predict(text_probs, audio_probs) 推理
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        if config is None:
            config = load_fusion_config()

        stacking_config = config["stacking"]
        self._model = LogisticRegression(
            max_iter=stacking_config.get("max_iter", 1000),
            C=stacking_config.get("C", 1.0),
            solver=stacking_config.get("solver", "lbfgs"),
            random_state=42,
        )
        self._is_fitted = False

    def fit(
        self,
        text_probs: np.ndarray,
        audio_probs: np.ndarray,
        labels: np.ndarray,
    ) -> "StackingFusion":
        """训练 meta-learner

        Args:
            text_probs: 文本概率分布 (n_samples, n_classes)
            audio_probs: 语音概率分布 (n_samples, n_classes)
            labels: 真实标签 (n_samples,)

        Returns:
            self
        """
        # 拼接双模态特征
        features = np.concatenate([text_probs, audio_probs], axis=1)
        self._model.fit(features, labels)
        self._is_fitted = True
        return self

    def predict(
        self,
        text_probs: np.ndarray,
        audio_probs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """使用 meta-learner 预测

        若 audio_probs 为 None 或未训练，降级为文本单模态。

        Args:
            text_probs: 文本概率分布 (n_samples, n_classes) 或 (n_classes,)
            audio_probs: 语音概率分布，形状同 text_probs

        Returns:
            预测概率分布 (n_samples, n_classes)
        """
        # 降级处理
        if audio_probs is None or not self._is_fitted:
            return text_probs.copy()

        features = np.concatenate([text_probs, audio_probs], axis=1)
        return self._model.predict_proba(features)

    @property
    def is_fitted(self) -> bool:
        """是否已训练"""
        return self._is_fitted


# ============================================================
# 策略 3：基于置信度的动态加权
# ============================================================

def fuse_dynamic_weighting(
    text_probs: np.ndarray,
    audio_probs: Optional[np.ndarray],
    text_confidence: float,
    audio_confidence: Optional[float] = None,
    config: Optional[dict] = None,
) -> np.ndarray:
    """基于置信度的动态加权融合

    根据各模态的置信度动态分配权重：
    - 置信度高的模态获得更大权重
    - 置信度低于阈值的模态权重归零
    - 使用 softmax 温度控制权重分配锐度

    Args:
        text_probs: 文本概率分布 (n_samples, n_classes) 或 (n_classes,)
        audio_probs: 语音概率分布；None 时降级
        text_confidence: 文本模态置信度 [0.0, 1.0]
        audio_confidence: 语音模态置信度 [0.0, 1.0]；None 时降级
        config: 融合配置

    Returns:
        融合后的概率分布
    """
    if config is None:
        config = load_fusion_config()

    dw_config = config["dynamic_weighting"]
    temperature = dw_config.get("temperature", 1.0)
    min_confidence = dw_config.get("min_confidence", 0.3)

    # 模态缺失降级
    if audio_probs is None or audio_confidence is None:
        return text_probs.copy()

    # 检查置信度是否低于阈值
    if text_confidence < min_confidence and audio_confidence < min_confidence:
        # 两个模态都不可靠，等权融合
        text_w, audio_w = 0.5, 0.5
    elif text_confidence < min_confidence:
        # 文本不可靠，仅用语音
        return audio_probs.copy()
    elif audio_confidence < min_confidence:
        # 语音不可靠，仅用文本
        return text_probs.copy()
    else:
        # softmax 动态分配权重
        raw = np.array([text_confidence, audio_confidence]) / temperature
        exp_raw = np.exp(raw - raw.max())  # 数值稳定
        softmax = exp_raw / exp_raw.sum()
        text_w, audio_w = softmax[0], softmax[1]

    fused = text_w * text_probs + audio_w * audio_probs
    fused = fused / fused.sum(axis=-1, keepdims=True)
    return fused


# ============================================================
# 统一融合接口（供 PerceptionService 调用）
# ============================================================

def fuse_multimodal(
    text_result: EmotionResult,
    audio_result: Optional[EmotionResult] = None,
    strategy: str = "weighted_average",
    stacking_model: Optional[StackingFusion] = None,
    config: Optional[dict] = None,
) -> EmotionResult:
    """统一多模态融合接口

    将文本和语音的 EmotionResult 融合为统一的 EmotionResult 输出，
    供下游模块（评估层、干预层）消费。

    Args:
        text_result: 文本模态的 EmotionResult
        audio_result: 语音模态的 EmotionResult（可选）
        strategy: 融合策略 "weighted_average" / "stacking" / "dynamic"
        stacking_model: Stacking 模型实例（仅 strategy="stacking" 时需要）
        config: 融合配置

    Returns:
        融合后的 EmotionResult
    """
    import time

    if config is None:
        config = load_fusion_config()

    text_probs = np.array(text_result.text_emotion_probs)
    audio_probs = None
    audio_confidence = None

    if audio_result is not None:
        audio_probs = np.array(audio_result.text_emotion_probs)
        audio_confidence = audio_result.confidence

    # 执行融合
    if strategy == "weighted_average":
        fused_probs = fuse_weighted_average(text_probs, audio_probs, config)
    elif strategy == "stacking":
        if stacking_model is not None and stacking_model.is_fitted:
            fused_probs = stacking_model.predict(text_probs, audio_probs)
        else:
            fused_probs = text_probs.copy()
    elif strategy == "dynamic":
        fused_probs = fuse_dynamic_weighting(
            text_probs, audio_probs,
            text_result.confidence, audio_confidence,
            config,
        )
    else:
        raise ValueError(f"未知融合策略: {strategy}")

    # 确保输出为 1D
    if fused_probs.ndim > 1:
        fused_probs = fused_probs.squeeze(0)

    # 计算融合后置信度
    fused_confidence = float(fused_probs.max())

    # 置信度衰减（单模态降级时）
    if audio_result is None:
        decay = config.get("fallback", {}).get("confidence_decay", 0.85)
        fused_confidence *= decay

    # 构建证据列表
    evidence = list(text_result.evidence)
    if audio_result is not None:
        evidence.extend(audio_result.evidence)
        evidence.append(f"融合策略: {strategy}")
    else:
        evidence.append(f"模态降级: 仅文本通道 (策略={strategy})")

    return EmotionResult(
        text_emotion_probs=fused_probs.tolist(),
        audio_risk_prob=audio_result.audio_risk_prob if audio_result else None,
        confidence=round(fused_confidence, 4),
        timestamp=time.time(),
        evidence=evidence,
    )
