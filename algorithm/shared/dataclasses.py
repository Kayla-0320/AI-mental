"""
跨模块统一数据接口定义

所有模块之间的通信必须通过这些数据类，不允许各自定义。
使用 Python 3.10+ dataclass 语法，不引入外部依赖。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRISIS = "crisis"


@dataclass
class EvidenceItem:
    """证据条目 —— 记录某项判断的依据来源

    Attributes:
        source: 证据来源模块名称（如 perception.text / assessment.risk）
        description: 证据的文本描述
        weight: 证据权重，范围 [0.0, 1.0]
    """
    source: str           # 证据来源模块
    description: str      # 证据描述
    weight: float         # 证据权重 [0.0, 1.0]


@dataclass
class EmotionResult:
    """情绪感知结果 —— 感知层输出统一的情绪向量

    Attributes:
        text_emotion_probs: 文本情绪概率分布，5维向量
            （对应：快乐/悲伤/焦虑/愤怒/中性）
        audio_risk_prob: 语音风险概率，None 表示无语音通道数据
        confidence: 综合置信度，范围 [0.0, 1.0]
        timestamp: Unix 时间戳，标记感知完成时刻
        evidence: 支撑该结果的证据描述列表
    """
    text_emotion_probs: list[float]       # 5维文本情绪概率分布
    audio_risk_prob: Optional[float]      # 语音风险概率（可为空）
    confidence: float                     # 综合置信度 [0.0, 1.0]
    timestamp: float                      # Unix 时间戳
    evidence: list[str]                   # 证据描述列表


@dataclass
class RiskAssessment:
    """心理风险评估结果 —— 评估层输出

    Attributes:
        phq9_estimated: PHQ-9 抑郁量表预估分数区间 (下界, 上界)
        gad7_estimated: GAD-7 焦虑量表预估分数区间 (下界, 上界)
        risk_level: 风险等级（low / medium / high / crisis）
        confidence: 评估置信度，范围 [0.0, 1.0]
        evidence: 支撑该评估的证据条目列表
    """
    phq9_estimated: tuple[float, float]   # PHQ-9 预估分数区间
    gad7_estimated: tuple[float, float]   # GAD-7 预估分数区间
    risk_level: RiskLevel                 # 风险等级枚举
    confidence: float                     # 评估置信度 [0.0, 1.0]
    evidence: list[EvidenceItem]          # 证据条目列表


@dataclass
class CrisisAlert:
    """危机警报 —— 升级层触发危机响应时的数据结构

    Attributes:
        user_id: 触发警报的用户唯一标识
        risk_score: 综合风险评分，范围 [0.0, 1.0]
        trigger_evidence: 触发本次危机的证据描述列表
        timestamp: Unix 时间戳，标记警报生成时刻
        recommended_action: 建议采取的干预动作
    """
    user_id: str                          # 用户唯一标识
    risk_score: float                     # 综合风险评分 [0.0, 1.0]
    trigger_evidence: list[str]           # 触发证据描述列表
    timestamp: float                      # Unix 时间戳
    recommended_action: str               # 建议干预动作


@dataclass
class FeatureSummary:
    """特征摘要 —— 跨模块传递的聚合特征数据

    注意：此数据类确保不包含原始文本或可识别信息，
    仅传递统计特征与概率分布，用于保护用户隐私。

    Attributes:
        text_emotion_probs: 文本情绪概率分布
        audio_risk_prob: 语音风险概率，None 表示无语音数据
        confidence: 特征置信度，范围 [0.0, 1.0]
        session_id: 会话唯一标识
        timestamp: Unix 时间戳，标记特征提取时刻
    """
    text_emotion_probs: list[float]       # 文本情绪概率分布
    audio_risk_prob: Optional[float]      # 语音风险概率（可为空）
    confidence: float                     # 特征置信度 [0.0, 1.0]
    session_id: str                       # 会话唯一标识
    timestamp: float                      # Unix 时间戳


# ============================================================
# 安全审计数据类
# ============================================================

class AuditAxis(str, Enum):
    """审计轴枚举 —— 五个安全审计维度"""
    CRISIS_DELAY = "crisis_delay"           # 危机升级延迟
    DELUSION_REINFORCEMENT = "delusion_reinforcement"  # 妄想强化
    STIGMA_REJECTION = "stigma_rejection"   # 污名化与拒绝
    SYCOPHANCY = "sycophancy"               # 谄媚倾向
    TRAJECTORY_DRIFT = "trajectory_drift"   # 轨迹漂移


class AuditAction(str, Enum):
    """审计不通过时的处理动作"""
    PASS = "pass"                           # 通过，原样输出
    REWRITE = "rewrite"                     # 重写回复
    INJECT_RESOURCE = "inject_resource"     # 注入危机资源卡
    ESCALATE_HUMAN = "escalate_human"       # 升级至人工


@dataclass
class AuditResult:
    """单轴审计结果

    Attributes:
        axis: 审计轴名称
        passed: 是否通过
        reason: 不通过的理由（通过时为空字符串）
        suggested_action: 建议的处理动作
    """
    axis: AuditAxis
    passed: bool
    reason: str = ""
    suggested_action: AuditAction = AuditAction.PASS


@dataclass
class AuditVerdict:
    """综合审计裁决 —— 五轴审计的最终结论

    Attributes:
        passed: 综合是否通过（所有轴都通过才为 True）
        results: 各轴审计结果列表
        final_action: 最终处理动作（取最严重的）
        rewritten_content: 重写后的内容（如果需要重写）
    """
    passed: bool
    results: list[AuditResult] = field(default_factory=list)
    final_action: AuditAction = AuditAction.PASS
    rewritten_content: str = ""


# ============================================================
# 数字表型特征数据类
# ============================================================

@dataclass
class PhenotypeFeature:
    """单个表型特征

    Attributes:
        name: 特征名称（如 "sleep_duration_mean"）
        value: 特征值（None 表示数据缺失）
        confidence: 特征置信度 [0, 1]
        source: 数据来源（如 "wearable" / "chat_log" / "assessment"）
        unit: 特征单位（如 "hours" / "score" / "count"）
        significant_change: 是否相对上一窗口有显著变化
    """
    name: str
    value: Optional[float]
    confidence: float
    source: str
    unit: str = ""
    significant_change: bool = False


@dataclass
class PhenotypeVector:
    """多模态数字表型向量

    整合五类特征，形成用户的综合数字表型画像。

    Attributes:
        user_id: 用户唯一标识
        time_window_days: 时间窗口（天）
        sleep_features: 睡眠特征列表
        emotion_features: 对话情感特征列表
        behavior_features: 行为特征列表
        assessment_features: 测评历史特征列表
        physiological_features: 生理特征列表
        timestamp: 特征提取时间戳
    """
    user_id: str
    time_window_days: int
    sleep_features: list[PhenotypeFeature] = field(default_factory=list)
    emotion_features: list[PhenotypeFeature] = field(default_factory=list)
    behavior_features: list[PhenotypeFeature] = field(default_factory=list)
    assessment_features: list[PhenotypeFeature] = field(default_factory=list)
    physiological_features: list[PhenotypeFeature] = field(default_factory=list)
    timestamp: float = 0.0

    def all_features(self) -> list[PhenotypeFeature]:
        """返回所有特征的扁平列表"""
        return (
            self.sleep_features
            + self.emotion_features
            + self.behavior_features
            + self.assessment_features
            + self.physiological_features
        )

    def feature_dict(self) -> dict[str, Optional[float]]:
        """返回 {特征名: 值} 字典"""
        return {f.name: f.value for f in self.all_features()}

    def missing_features(self) -> list[str]:
        """返回缺失数据的特征名列表"""
        return [f.name for f in self.all_features() if f.value is None]


# ============================================================
# 危机升级数据类
# ============================================================

class EscalationChannel(str, Enum):
    """升级通道枚举"""
    WEBHOOK = "webhook"               # Webhook 通知（模拟）
    IN_APP = "in_app"                 # 站内通知
    CONSOLE = "console"               # 控制台告警


class EscalationStatus(str, Enum):
    """升级状态枚举"""
    PENDING = "pending"               # 待处理
    DISPATCHED = "dispatched"         # 已派发
    ACKNOWLEDGED = "acknowledged"     # 已确认
    RESOLVED = "resolved"             # 已解决
    FAILED = "failed"                 # 失败


@dataclass
class EscalationResult:
    """危机升级结果 —— 升级层核心输出

    Attributes:
        alert_id: 升级事件唯一标识
        alert: 原始危机警报
        status: 升级处理状态
        channels_notified: 已通知的通道列表
        assigned_counselor: 分配的咨询师 ID（空字符串表示未分配）
        response_time_seconds: 升级响应时间（秒），目标 < 30 秒
        timestamp: 升级完成时间戳
        audit_log_id: 对应的审计日志 ID
    """
    alert_id: str
    alert: CrisisAlert
    status: EscalationStatus
    channels_notified: list[EscalationChannel] = field(default_factory=list)
    assigned_counselor: str = ""
    response_time_seconds: float = 0.0
    timestamp: float = 0.0
    audit_log_id: str = ""
