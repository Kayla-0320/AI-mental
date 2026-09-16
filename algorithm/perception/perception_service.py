"""
感知层统一服务接口

所有下游模块（评估、干预、画像）通过此接口获取情感数据。
多模态融合调用 fusion/fusion.py 中的 fuse_multimodal() 统一接口，
支持 weighted_average / stacking / dynamic 三种融合策略。

v2: 接入真实文本模型 + 面部/行为特征融合
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Optional

from shared.dataclasses import EmotionResult
from perception.fusion.fusion import fuse_multimodal, StackingFusion

logger = logging.getLogger(__name__)


# ============================================================
# 模型标签映射
# text_emotion 模型输出: [anxiety(0), depression(1), anger(2), neutral(3), positive(4)]
# EmotionResult 期望:     [快乐(0),    悲伤(1),      焦虑(2),  愤怒(3),   中性(4)]
# ============================================================

def _remap_probs(model_probs: list[float]) -> list[float]:
    """将模型输出概率映射为 EmotionResult 格式

    Args:
        model_probs: 模型输出 [anxiety, depression, anger, neutral, positive]

    Returns:
        EmotionResult 格式 [快乐, 悲伤, 焦虑, 愤怒, 中性]
    """
    anxiety, depression, anger, neutral, positive = model_probs
    return [positive, depression, anxiety, anger, neutral]


# ============================================================
# 文本模型加载（真实 RoBERTa 模型）
# ============================================================

_text_model_available = False

@functools.lru_cache(maxsize=1)
def _try_load_text_model() -> bool:
    """尝试加载文本情感模型，返回是否成功"""
    global _text_model_available
    try:
        from perception.text_emotion.predict import predict as _test_predict
        # 测试一下模型是否能正常推理
        _test_predict("测试")
        _text_model_available = True
        logger.info("文本情感模型加载成功")
    except Exception as e:
        logger.warning(f"文本情感模型加载失败，使用降级模式: {e}")
        _text_model_available = False
    return _text_model_available


def _real_text_predict(text: str) -> list[float]:
    """调用真实文本情感模型

    Args:
        text: 待分析文本

    Returns:
        EmotionResult 格式的 5 维概率分布 [快乐, 悲伤, 焦虑, 愤怒, 中性]
    """
    from perception.text_emotion.predict import predict
    result = predict(text)
    model_probs = result["probs"]  # [anxiety, depression, anger, neutral, positive]
    return _remap_probs(model_probs)


# ============================================================
# 面部/行为特征提取（可选模块）
# ============================================================

def _extract_facial_risk(facial_features: Optional[dict] = None) -> Optional[float]:
    """从面部特征提取情绪风险分

    Args:
        facial_features: 面部特征字典（来自 facial_expression.py）

    Returns:
        风险概率 [0, 1]，或 None（无数据时）
    """
    if not facial_features:
        return None
    try:
        from perception.face.facial_expression import FacialFeatures, features_to_emotion_risk
        # 将 dict 转换为 FacialFeatures dataclass
        if isinstance(facial_features, dict):
            ff = FacialFeatures(**{k: v for k, v in facial_features.items() if hasattr(FacialFeatures, k)})
        else:
            ff = facial_features
        result = features_to_emotion_risk(ff)
        return result.get("risk_score", None)
    except Exception as e:
        logger.warning(f"面部特征提取失败: {e}")
        return None


def _extract_behavior_risk(behavior_features: Optional[dict] = None) -> Optional[float]:
    """从行为特征提取情绪风险分

    Args:
        behavior_features: 行为特征字典（来自 behavior_pattern.py）

    Returns:
        风险概率 [0, 1]，或 None（无数据时）
    """
    if not behavior_features:
        return None
    try:
        from perception.behavior.behavior_pattern import BehaviorFeatures, behavior_to_emotion_risk
        # 将 dict 转换为 BehaviorFeatures dataclass
        if isinstance(behavior_features, dict):
            bf = BehaviorFeatures(**{k: v for k, v in behavior_features.items() if hasattr(BehaviorFeatures, k)})
        else:
            bf = behavior_features
        result = behavior_to_emotion_risk(bf)
        return result.get("risk_score", None)
    except Exception as e:
        logger.warning(f"行为特征提取失败: {e}")
        return None


def _acoustic_to_emotion_probs(acoustic) -> list[float]:
    """将声学特征映射为5维情绪概率分布

    基于 Scherer (2003) 和 Cummins et al. (2015) 的经验规则：
    - 低基频 + 低能量 → 悲伤
    - 高基频 + 高语速 → 焦虑
    - 高能量 + 高 F0 范围 → 愤怒
    - 平稳基频 + 中等能量 → 中性
    - 中等 F0 + 变化丰富 → 快乐

    Args:
        acoustic: AcousticFeatures 实例

    Returns:
        [快乐, 悲伤, 焦虑, 愤怒, 中性] 概率分布
    """
    probs = [0.0] * 5  # [快乐, 悲伤, 焦虑, 愤怒, 中性]

    # 基频指标
    f0_norm = min(1.0, max(0.0, (acoustic.f0_mean - 100) / 200))  # 归一化到 [0,1]

    # 低基频 + 低能量 → 悲伤
    if acoustic.f0_mean < 150 and acoustic.energy_mean < 50:
        probs[1] += 0.4

    # 高基频 + 高语速 → 焦虑
    if acoustic.f0_mean > 250 or acoustic.speech_rate > 5.0:
        probs[2] += 0.35

    # 高能量 + 大 F0 范围 → 愤怒
    if acoustic.energy_mean > 65 and acoustic.f0_range > 100:
        probs[3] += 0.35

    # 高停顿 → 悲伤/焦虑
    if acoustic.pause_ratio > 0.3:
        probs[1] += 0.15
        probs[2] += 0.1

    # 语速慢 → 悲伤
    if acoustic.speech_rate < 2.5:
        probs[1] += 0.2

    # 能量波动大 → 愤怒/焦虑
    if acoustic.energy_std > 10:
        probs[2] += 0.1
        probs[3] += 0.1

    # F0 范围大 + 中等语速 → 快乐
    if acoustic.f0_range > 80 and 3.0 < acoustic.speech_rate < 5.0:
        probs[0] += 0.3

    # 如果所有指标都正常 → 中性
    if max(probs) < 0.1:
        probs[4] = 0.6
    else:
        probs[4] = max(0.05, 1.0 - sum(probs))

    # 归一化
    s = sum(probs)
    if s > 0:
        probs = [p / s for p in probs]
    else:
        probs = [0.2, 0.2, 0.2, 0.2, 0.2]

    return probs


# ============================================================
# 融合策略配置
# ============================================================

DEFAULT_FUSION_STRATEGY = "weighted_average"
TEXT_WEIGHT = 0.6
AUDIO_WEIGHT = 0.4


# ============================================================
# PerceptionService v2
# ============================================================

class PerceptionService:
    """感知层统一服务 v2

    支持文本、语音、面部、行为四种模态，
    通过融合引擎输出统一的 EmotionResult。
    """

    def __init__(
        self,
        fusion_strategy: str = DEFAULT_FUSION_STRATEGY,
        stacking_model: Optional[StackingFusion] = None,
    ) -> None:
        self._fusion_strategy = fusion_strategy
        self._stacking_model = stacking_model
        # 尝试加载真实文本模型
        _try_load_text_model()

    # ----------------------------------------------------------
    # 文本情感分析
    # ----------------------------------------------------------

    def analyze_text(self, text: str) -> EmotionResult:
        """文本情感分析

        优先使用真实 RoBERTa 模型，加载失败时降级为规则引擎。

        Args:
            text: 待分析的文本内容

        Returns:
            EmotionResult: 5维情绪概率分布 + 证据
        """
        if _text_model_available:
            try:
                probs = _real_text_predict(text)
                # 找到主导情绪
                emotion_labels = ['快乐', '悲伤', '焦虑', '愤怒', '中性']
                dominant_idx = max(range(5), key=lambda i: probs[i])
                dominant = emotion_labels[dominant_idx]
                confidence = probs[dominant_idx]

                return EmotionResult(
                    text_emotion_probs=probs,
                    audio_risk_prob=None,
                    confidence=round(confidence, 4),
                    timestamp=time.time(),
                    evidence=[f"文本情感分析（RoBERTa）：主导情绪={dominant}，置信度={confidence:.2%}"],
                )
            except Exception as e:
                logger.warning(f"真实模型推理失败，降级: {e}")

        # 降级：基于关键词的规则引擎
        return self._rule_based_text_analysis(text)

    def _rule_based_text_analysis(self, text: str) -> EmotionResult:
        """基于关键词的降级文本分析"""
        crisis_keywords = ['自杀', '自残', '不想活', '去死', '跳楼', '割腕']
        anxiety_keywords = ['焦虑', '紧张', '担心', '害怕', '恐惧', '不安', '心跳', '失眠']
        depression_keywords = ['低落', '难过', '悲伤', '没意思', '没用', '疲惫', '哭泣', '灰色']
        anger_keywords = ['生气', '愤怒', '气死', '过分', '受够了', '不公平']

        text_lower = text.lower()
        crisis_count = sum(1 for w in crisis_keywords if w in text_lower)
        anxiety_count = sum(1 for w in anxiety_keywords if w in text_lower)
        depression_count = sum(1 for w in depression_keywords if w in text_lower)
        anger_count = sum(1 for w in anger_keywords if w in text_lower)

        total = crisis_count + anxiety_count + depression_count + anger_count + 1
        # [快乐, 悲伤, 焦虑, 愤怒, 中性]
        probs = [
            0.05,
            depression_count / total * 0.7 + 0.05,
            anxiety_count / total * 0.7 + crisis_count / total * 0.3 + 0.05,
            anger_count / total * 0.7 + 0.05,
            max(0.05, 1.0 - (anxiety_count + depression_count + anger_count + crisis_count) / total * 0.7),
        ]
        # 归一化
        s = sum(probs)
        probs = [p / s for p in probs]

        emotion_labels = ['快乐', '悲伤', '焦虑', '愤怒', '中性']
        dominant_idx = max(range(5), key=lambda i: probs[i])
        dominant = emotion_labels[dominant_idx]

        evidence_parts = []
        if crisis_count > 0:
            evidence_parts.append(f"检测到危机关键词({crisis_count}个)")
        if anxiety_count > 0:
            evidence_parts.append(f"焦虑相关词({anxiety_count}个)")
        if depression_count > 0:
            evidence_parts.append(f"抑郁相关词({depression_count}个)")
        if anger_count > 0:
            evidence_parts.append(f"愤怒相关词({anger_count}个)")
        if not evidence_parts:
            evidence_parts.append("未检测到明显情绪关键词")

        return EmotionResult(
            text_emotion_probs=probs,
            audio_risk_prob=None,
            confidence=round(probs[dominant_idx], 4),
            timestamp=time.time(),
            evidence=[f"文本情感分析（规则引擎降级）：主导情绪={dominant}，" + "；".join(evidence_parts)],
        )

    # ----------------------------------------------------------
    # 语音情感分析
    # ----------------------------------------------------------

    def analyze_audio(self, wav_path: str) -> EmotionResult:
        """语音情感分析 —— 使用真实 librosa 声学特征提取

        提取 F0/能量/语速/停顿等声学标志物，
        通过经验规则映射为情绪风险概率。

        Args:
            wav_path: WAV 音频文件路径

        Returns:
            EmotionResult: 语音风险概率 + 声学证据

        Raises:
            FileNotFoundError: 音频文件不存在
        """
        import os
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"音频文件不存在: {wav_path}")

        try:
            from perception.voice.acoustic import extract_acoustic_features, features_to_emotion_risk
            # 提取真实声学特征
            acoustic = extract_acoustic_features(wav_path)

            if acoustic.valid:
                # 将声学特征映射为风险概率
                risk_prob = features_to_emotion_risk(acoustic)

                # 根据声学特征生成证据
                evidence_parts = []
                if acoustic.f0_mean < 150:
                    evidence_parts.append(f"低基频({acoustic.f0_mean:.0f}Hz，抑郁指标)")
                elif acoustic.f0_mean > 250:
                    evidence_parts.append(f"高基频({acoustic.f0_mean:.0f}Hz，焦虑指标)")
                if acoustic.speech_rate > 5.0:
                    evidence_parts.append(f"快语速({acoustic.speech_rate:.1f}音节/s，焦虑指标)")
                elif acoustic.speech_rate < 2.5:
                    evidence_parts.append(f"慢语速({acoustic.speech_rate:.1f}音节/s，抑郁指标)")
                if acoustic.pause_ratio > 0.3:
                    evidence_parts.append(f"高停顿比({acoustic.pause_ratio:.0%}，犹豫指标)")
                if acoustic.energy_std > 10:
                    evidence_parts.append(f"高能量波动({acoustic.energy_std:.1f}dB，不稳定指标)")

                if not evidence_parts:
                    evidence_parts.append(f"声学特征正常(F0={acoustic.f0_mean:.0f}Hz, 语速={acoustic.speech_rate:.1f})")

                # 根据声学特征推断情绪倾向
                emotion_probs = _acoustic_to_emotion_probs(acoustic)

                return EmotionResult(
                    text_emotion_probs=emotion_probs,
                    audio_risk_prob=round(risk_prob, 4),
                    confidence=round(min(0.95, 0.5 + acoustic.duration * 0.05), 4),
                    timestamp=time.time(),
                    evidence=[f"语音声学分析（librosa）：风险={risk_prob:.2f}，" + "；".join(evidence_parts)],
                )
            else:
                # 特征提取失败（音频太短或无语音）
                return EmotionResult(
                    text_emotion_probs=[0.20, 0.20, 0.20, 0.20, 0.20],
                    audio_risk_prob=0.0,
                    confidence=0.3,
                    timestamp=time.time(),
                    evidence=["语音分析：音频过短或无有效语音段"],
                )
        except ImportError:
            logger.warning("librosa 未安装，语音分析降级")
            return EmotionResult(
                text_emotion_probs=[0.20, 0.20, 0.20, 0.20, 0.20],
                audio_risk_prob=0.0,
                confidence=0.3,
                timestamp=time.time(),
                evidence=["语音分析降级：librosa 未安装"],
            )
        except Exception as e:
            logger.warning(f"语音分析异常: {e}")
            return EmotionResult(
                text_emotion_probs=[0.20, 0.20, 0.20, 0.20, 0.20],
                audio_risk_prob=0.0,
                confidence=0.3,
                timestamp=time.time(),
                evidence=[f"语音分析异常: {e}"],
            )

    # ----------------------------------------------------------
    # 多模态融合（支持4种模态）
    # ----------------------------------------------------------

    def analyze_multimodal(
        self,
        text: str,
        wav_path: Optional[str] = None,
        facial_features: Optional[dict] = None,
        behavior_features: Optional[dict] = None,
    ) -> EmotionResult:
        """多模态情感分析（文本 + 语音 + 面部 + 行为融合）

        Args:
            text: 待分析的文本内容
            wav_path: WAV 音频文件路径（可选）
            facial_features: 面部特征字典（可选）
            behavior_features: 行为特征字典（可选）

        Returns:
            EmotionResult: 融合后的情绪分析结果
        """
        # 1. 文本分析（始终执行）
        text_result = self.analyze_text(text)

        # 2. 语音分析（可选）
        audio_result: Optional[EmotionResult] = None
        if wav_path is not None:
            try:
                audio_result = self.analyze_audio(wav_path)
            except Exception:
                audio_result = None

        # 3. 面部/行为风险分（可选，作为辅助证据）
        facial_risk = _extract_facial_risk(facial_features)
        behavior_risk = _extract_behavior_risk(behavior_features)

        # 4. 融合文本+语音
        if audio_result is not None:
            fused = fuse_multimodal(
                text_result=text_result,
                audio_result=audio_result,
                strategy=self._fusion_strategy,
                stacking_model=self._stacking_model,
            )
        else:
            fused = fuse_multimodal(
                text_result=text_result,
                audio_result=None,
                strategy=self._fusion_strategy,
                stacking_model=self._stacking_model,
            )

        # 5. 如果有面部/行为数据，追加到证据中
        extra_evidence = list(fused.evidence)
        if facial_risk is not None:
            extra_evidence.append(f"面部微表情风险分: {facial_risk:.2f}")
        if behavior_risk is not None:
            extra_evidence.append(f"行为模式风险分: {behavior_risk:.2f}")

        # 6. 如果面部/行为风险分显著偏高，提升综合风险
        adjusted_probs = list(fused.text_emotion_probs)
        if facial_risk is not None and facial_risk > 0.6:
            # 面部高风险 → 提升焦虑维度
            adjusted_probs[2] = min(0.95, adjusted_probs[2] + facial_risk * 0.1)
        if behavior_risk is not None and behavior_risk > 0.6:
            # 行为高风险 → 提升悲伤+焦虑维度
            adjusted_probs[1] = min(0.95, adjusted_probs[1] + behavior_risk * 0.05)
            adjusted_probs[2] = min(0.95, adjusted_probs[2] + behavior_risk * 0.05)

        # 重新归一化
        s = sum(adjusted_probs)
        adjusted_probs = [p / s for p in adjusted_probs]

        return EmotionResult(
            text_emotion_probs=adjusted_probs,
            audio_risk_prob=fused.audio_risk_prob,
            confidence=fused.confidence,
            timestamp=fused.timestamp,
            evidence=extra_evidence,
        )


# ============================================================
# 全局单例
# ============================================================

_perception_service: Optional[PerceptionService] = None


def get_perception_service(
    fusion_strategy: str = DEFAULT_FUSION_STRATEGY,
) -> PerceptionService:
    """获取 PerceptionService 全局单例"""
    global _perception_service
    if _perception_service is None:
        _perception_service = PerceptionService(fusion_strategy=fusion_strategy)
    return _perception_service
