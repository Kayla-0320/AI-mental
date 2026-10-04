"""
危机信号检测器 —— 实时监测多模态信号中的危机特征

与 self_harm 模块的区别：
    - self_harm: 专注于文本中的自伤/自杀关键词
    - detector: 综合多模态信号的异常突变检测

检测维度：
    1. 情绪突变：短时间内情绪向量剧烈变化
    2. 行为突变：活跃度/社交频率骤降
    3. 语音异常：语音风险概率骤升
    4. 综合预警：多维度信号同时异常

核心接口：
    - detect_crisis(emotion_history, current_emotion) -> CrisisDetectionResult
    - update_detector(detector, emotion_result) -> CrisisDetectionResult
    - CrisisSignalDetector: 有状态的检测器类
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from shared.dataclasses import EmotionResult, CrisisAlert


# ============================================================
# 数据结构
# ============================================================

class CrisisSignalType(str, Enum):
    """危机信号类型"""
    EMOTION_SPIKE = "emotion_spike"           # 情绪突变
    BEHAVIOR_DROP = "behavior_drop"           # 行为骤降
    AUDIO_SPIKE = "audio_spike"               # 语音风险突变
    MULTI_SIGNAL = "multi_signal"             # 多信号同时异常
    SUSTAINED_NEGATIVE = "sustained_negative" # 持续负面状态


@dataclass
class CrisisSignal:
    """单条危机信号

    Attributes:
        signal_type: 信号类型
        severity: 严重程度 [0, 1]
        description: 信号描述
        timestamp: 信号时间戳
    """
    signal_type: CrisisSignalType
    severity: float
    description: str
    timestamp: float = 0.0


@dataclass
class CrisisDetectionResult:
    """危机检测结果

    Attributes:
        crisis_detected: 是否检测到危机
        crisis_score: 危机评分 [0, 1]
        signals: 检测到的危机信号列表
        should_alert: 是否应触发 CrisisAlert
        alert: 生成的 CrisisAlert（如有）
        recommendation: 建议动作
    """
    crisis_detected: bool
    crisis_score: float
    signals: list[CrisisSignal] = field(default_factory=list)
    should_alert: bool = False
    alert: Optional[CrisisAlert] = None
    recommendation: str = ""


# ============================================================
# 有状态危机检测器
# ============================================================

class CrisisSignalDetector:
    """有状态的危机信号检测器

    维护情绪历史窗口，检测突变和持续异常。

    Attributes:
        window_size: 历史窗口大小
        emotion_history: 情绪概率历史
        baseline: 基线情绪（滑动窗口均值）
        alert_cooldown: 警报冷却时间（秒）
        last_alert_time: 上次警报时间
    """

    def __init__(
        self,
        window_size: int = 20,
        alert_cooldown: float = 300.0,
    ):
        """初始化检测器

        Args:
            window_size: 历史窗口大小（消息数）
            alert_cooldown: 警报冷却时间（秒），防止重复报警
        """
        self.window_size = window_size
        self.emotion_history: deque[list[float]] = deque(maxlen=window_size)
        self.alert_cooldown = alert_cooldown
        self.last_alert_time = 0.0
        self._total_messages = 0

    def reset(self):
        """重置检测器状态"""
        self.emotion_history.clear()
        self.last_alert_time = 0.0
        self._total_messages = 0

    def update(self, emotion_result: EmotionResult, user_id: str = "") -> CrisisDetectionResult:
        """更新检测器并检测危机

        Args:
            emotion_result: 新的情绪感知结果
            user_id: 用户 ID（用于生成 CrisisAlert）

        Returns:
            CrisisDetectionResult: 检测结果
        """
        self._total_messages += 1
        probs = emotion_result.text_emotion_probs
        signals: list[CrisisSignal] = []
        now = time.time()

        # 记录历史
        self.emotion_history.append(list(probs))

        # 1. 情绪突变检测
        if len(self.emotion_history) >= 3:
            prev_probs = list(self.emotion_history)[-2]
            delta = np.abs(np.array(probs) - np.array(prev_probs))
            max_delta = float(np.max(delta))

            if max_delta > 0.4:
                # 计算整体漂移
                drift = float(np.linalg.norm(delta))
                severity = min(1.0, drift / 2.0)
                signals.append(CrisisSignal(
                    signal_type=CrisisSignalType.EMOTION_SPIKE,
                    severity=severity,
                    description=f"情绪向量突变: 最大变化={max_delta:.2f}, 漂移={drift:.2f}",
                    timestamp=now,
                ))

        # 2. 持续负面状态检测
        if len(self.emotion_history) >= 5:
            recent = list(self.emotion_history)[-5:]
            # 快乐+中性 持续低 = 持续负面
            positive_probs = [p[0] + p[4] for p in recent]  # 快乐 + 中性
            avg_positive = np.mean(positive_probs)
            if avg_positive < 0.2:
                signals.append(CrisisSignal(
                    signal_type=CrisisSignalType.SUSTAINED_NEGATIVE,
                    severity=min(1.0, 1.0 - avg_positive),
                    description=f"近5次交互持续负面: 正面概率均值={avg_positive:.2f}",
                    timestamp=now,
                ))

        # 3. 语音风险突变
        if emotion_result.audio_risk_prob is not None:
            if emotion_result.audio_risk_prob > 0.7:
                signals.append(CrisisSignal(
                    signal_type=CrisisSignalType.AUDIO_SPIKE,
                    severity=emotion_result.audio_risk_prob,
                    description=f"语音风险概率极高: {emotion_result.audio_risk_prob:.2f}",
                    timestamp=now,
                ))

        # 4. 多信号同时异常
        if len(signals) >= 2:
            multi_severity = min(1.0, sum(s.severity for s in signals) / len(signals) * 1.5)
            signals.append(CrisisSignal(
                signal_type=CrisisSignalType.MULTI_SIGNAL,
                severity=multi_severity,
                description=f"多信号同时异常: {len(signals)} 个信号触发",
                timestamp=now,
            ))

        # 综合危机评分
        if signals:
            crisis_score = min(1.0, max(s.severity for s in signals) * 0.7 +
                               sum(s.severity for s in signals) / len(signals) * 0.3)
        else:
            crisis_score = 0.0

        crisis_detected = crisis_score >= 0.6

        # 是否触发警报（考虑冷却时间）
        should_alert = False
        alert = None
        if crisis_detected and (now - self.last_alert_time) > self.alert_cooldown:
            should_alert = True
            self.last_alert_time = now

            # 生成 CrisisAlert
            evidence = [s.description for s in signals]
            alert = CrisisAlert(
                user_id=user_id or "unknown",
                risk_score=crisis_score,
                trigger_evidence=evidence,
                timestamp=now,
                recommended_action=_get_recommendation(crisis_score),
            )

        # 建议
        recommendation = _get_recommendation(crisis_score)

        return CrisisDetectionResult(
            crisis_detected=crisis_detected,
            crisis_score=round(crisis_score, 3),
            signals=signals,
            should_alert=should_alert,
            alert=alert,
            recommendation=recommendation,
        )


def _get_recommendation(crisis_score: float) -> str:
    """根据危机评分生成建议"""
    if crisis_score >= 0.8:
        return "立即触发危机干预协议，通知咨询师和紧急联系人，提供24h热线"
    elif crisis_score >= 0.6:
        return "高度关注，建议立即发起主动关怀对话，评估安全状态"
    elif crisis_score >= 0.4:
        return "中度关注，增加情绪监测频率，考虑主动推送应对资源"
    elif crisis_score >= 0.2:
        return "轻度关注，保持常规监测"
    else:
        return "状态正常，无需特殊处理"


# ============================================================
# 无状态便捷接口
# ============================================================

def detect_crisis(
    emotion_history: list[EmotionResult],
    current_emotion: EmotionResult,
    user_id: str = "",
) -> CrisisDetectionResult:
    """无状态危机检测（一次性使用）

    Args:
        emotion_history: 历史情绪结果列表
        current_emotion: 当前情绪结果
        user_id: 用户 ID

    Returns:
        CrisisDetectionResult
    """
    detector = CrisisSignalDetector(window_size=len(emotion_history) + 1)

    # 回放历史
    for em in emotion_history:
        detector.emotion_history.append(em.text_emotion_probs)

    # 检测当前
    return detector.update(current_emotion, user_id)


def update_detector(
    detector: CrisisSignalDetector,
    emotion_result: EmotionResult,
    user_id: str = "",
) -> CrisisDetectionResult:
    """更新有状态检测器

    Args:
        detector: 检测器实例
        emotion_result: 新的情绪结果
        user_id: 用户 ID

    Returns:
        CrisisDetectionResult
    """
    return detector.update(emotion_result, user_id)
