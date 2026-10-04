"""
自伤/自杀风险检测模块 —— 多层级信号识别与危机触发

检测策略：
    1. 关键词匹配：自伤/自杀相关词汇（中文青少年语境适配）
    2. 语义模式：间接表达、告别式语言、绝望叙事
    3. 行为信号：突然的社交退缩、异常作息变化
    4. 情绪信号：极端负面情绪突变

分级响应：
    - Level 0: 无风险信号
    - Level 1: 潜在风险（间接表达，需持续关注）
    - Level 2: 中度风险（明确消极表达，建议主动干预）
    - Level 3: 高风险（直接自伤/自杀意念，立即触发危机协议）

⚠️ 声明：本模块为辅助筛查工具，不能替代专业临床评估。
    宁可误报（高召回率），不可漏报（低漏判率）。

核心接口：
    - detect_self_harm_risk(text, emotion, behavior) -> SelfHarmRiskResult
    - batch_detect(texts) -> list[SelfHarmRiskResult]
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from shared.dataclasses import EmotionResult


# ============================================================
# 风险等级
# ============================================================

class SelfHarmRiskLevel(int, Enum):
    """自伤风险等级"""
    NONE = 0          # 无风险
    POTENTIAL = 1     # 潜在风险
    MODERATE = 2      # 中度风险
    HIGH = 3          # 高风险


# ============================================================
# 关键词词典（青少年语境适配）
# ============================================================

# Level 3: 直接自伤/自杀表达（最高优先级）
DIRECT_KEYWORDS = [
    # 直接自杀表达
    "不想活", "想死", "去死", "活着没意思", "活着没意义",
    "不如死了", "死了算了", "想自杀", "自杀",
    "结束生命", "了结", "解脱",
    "跳楼", "割腕", "上吊",
    "从.*死", "不想.*活",
    # 青少年口语化表达
    "人间不值得", "世界没有我会更好",
    "我消失了也没人在意", "没人会想我",
    "如果我死了", "要是我不在了",
]

# Level 2: 明确消极/绝望表达
EXPLICIT_NEGATIVE_KEYWORDS = [
    "绝望", "没有希望", "永远不会好",
    "痛苦", "受不了", "撑不下去",
    "没有用", "什么都做不好", "我是个废物",
    "没有人关心我", "没人在乎", "被抛弃",
    "好累", "心好痛", "活得好辛苦",
    "不想醒来", "想消失", "想逃避",
    "伤害自己", "自残",
]

# Level 1: 间接/隐晦表达
INDIRECT_KEYWORDS = [
    "一个人", "孤独", "没人理解",
    "好暗", "看不到光", "黑暗",
    "累了", "疲倦", "没有力气",
    "无所谓", "都无所谓", "随便",
    "再见", "告别", "最后的",
    "如果我不在了", "忘记我",
    "睡不着", "失眠",
    "不想说话", "不想出门",
]

# 告别式语言模式
FAREWELL_PATTERNS = [
    r"把.*送.*", r"把.*给.*",  # "把我的猫给朋友"
    r"最后.*想.*", r"以后.*不.*",
    r"帮我.*照顾", r"替我.*",
    r"对不起.*",  # 道歉式告别
]


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SelfHarmRiskResult:
    """自伤风险检测结果

    Attributes:
        risk_level: 风险等级 (0-3)
        risk_score: 风险评分 [0, 1]
        matched_keywords: 匹配到的关键词
        matched_patterns: 匹配到的语义模式
        evidence: 检测证据列表
        should_escalate: 是否应触发危机升级
        recommended_action: 建议动作
        timestamp: 检测时间戳
    """
    risk_level: SelfHarmRiskLevel
    risk_score: float
    matched_keywords: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    should_escalate: bool = False
    recommended_action: str = ""
    timestamp: float = 0.0


# ============================================================
# 检测引擎
# ============================================================

def _match_keywords(text: str) -> tuple[list[str], SelfHarmRiskLevel]:
    """关键词匹配

    Returns:
        (matched_keywords, highest_level)
    """
    matched = []
    highest_level = SelfHarmRiskLevel.NONE

    # Level 3 直接表达
    for kw in DIRECT_KEYWORDS:
        if ".*" in kw:
            if re.search(kw, text):
                matched.append(kw)
                highest_level = max(highest_level, SelfHarmRiskLevel.HIGH)
        elif kw in text:
            matched.append(kw)
            highest_level = max(highest_level, SelfHarmRiskLevel.HIGH)

    # Level 2 明确消极
    for kw in EXPLICIT_NEGATIVE_KEYWORDS:
        if kw in text:
            matched.append(kw)
            highest_level = max(highest_level, SelfHarmRiskLevel.MODERATE)

    # Level 1 间接表达
    for kw in INDIRECT_KEYWORDS:
        if kw in text:
            matched.append(kw)
            highest_level = max(highest_level, SelfHarmRiskLevel.POTENTIAL)

    return matched, highest_level


def _match_patterns(text: str) -> tuple[list[str], SelfHarmRiskLevel]:
    """语义模式匹配"""
    matched = []
    highest_level = SelfHarmRiskLevel.NONE

    for pattern in FAREWELL_PATTERNS:
        if re.search(pattern, text):
            matched.append(pattern)
            highest_level = max(highest_level, SelfHarmRiskLevel.MODERATE)

    return matched, highest_level


def _assess_emotion_risk(emotion: Optional[EmotionResult]) -> tuple[float, list[str]]:
    """从情绪结果评估自伤风险"""
    if emotion is None:
        return 0.0, []

    probs = emotion.text_emotion_probs
    # [快乐, 悲伤, 焦虑, 愤怒, 中性]
    sad, anxious, angry = probs[1], probs[2], probs[3]
    happy = probs[0]

    risk_score = 0.0
    evidence = []

    # 极高悲伤 + 极低快乐 = 绝望指标
    if sad > 0.5 and happy < 0.05:
        risk_score += 0.3
        evidence.append(f"极端负面情绪: 悲伤={sad:.2f}, 快乐={happy:.2f}")

    # 极高焦虑
    if anxious > 0.6:
        risk_score += 0.2
        evidence.append(f"高焦虑概率: {anxious:.2f}")

    # 情绪突变（低置信度可能表示异常）
    if emotion.confidence < 0.3:
        risk_score += 0.1
        evidence.append(f"感知置信度极低: {emotion.confidence:.2f}")

    return min(1.0, risk_score), evidence


# ============================================================
# 核心接口
# ============================================================

def detect_self_harm_risk(
    text: str = "",
    emotion: Optional[EmotionResult] = None,
    behavior_change_score: Optional[float] = None,
) -> SelfHarmRiskResult:
    """综合自伤风险检测

    Args:
        text: 用户输入文本
        emotion: 情绪感知结果（可选）
        behavior_change_score: 行为变化评分 [0, 1]（可选）

    Returns:
        SelfHarmRiskResult: 检测结果
    """
    all_keywords = []
    all_patterns = []
    all_evidence = []
    keyword_level = SelfHarmRiskLevel.NONE
    risk_score = 0.0

    # 1. 关键词检测
    if text:
        kw_matched, kw_level = _match_keywords(text)
        all_keywords.extend(kw_matched)
        keyword_level = max(keyword_level, kw_level)

        # 模式检测
        pat_matched, pat_level = _match_patterns(text)
        all_patterns.extend(pat_matched)
        keyword_level = max(keyword_level, pat_level)

        # 关键词风险评分
        if keyword_level == SelfHarmRiskLevel.HIGH:
            risk_score = max(risk_score, 0.85)
            all_evidence.append(f"检测到直接自伤/自杀关键词: {', '.join(all_keywords[:5])}")
        elif keyword_level == SelfHarmRiskLevel.MODERATE:
            risk_score = max(risk_score, 0.55)
            all_evidence.append(f"检测到明确消极表达: {', '.join(all_keywords[:5])}")
        elif keyword_level == SelfHarmRiskLevel.POTENTIAL:
            risk_score = max(risk_score, 0.25)
            if all_keywords:
                all_evidence.append(f"检测到间接风险表达: {', '.join(all_keywords[:5])}")

        if all_patterns:
            all_evidence.append(f"匹配告别式语言模式: {len(all_patterns)} 个")

    # 2. 情绪信号
    emotion_risk, emotion_evidence = _assess_emotion_risk(emotion)
    risk_score = max(risk_score, emotion_risk)
    all_evidence.extend(emotion_evidence)

    # 3. 行为变化信号
    if behavior_change_score is not None and behavior_change_score > 0.7:
        risk_score = max(risk_score, 0.3)
        all_evidence.append(f"行为突变评分: {behavior_change_score:.2f}")

    # 综合评分
    risk_score = min(1.0, risk_score)

    # 确定风险等级
    if risk_score >= 0.75 or keyword_level >= SelfHarmRiskLevel.HIGH:
        final_level = SelfHarmRiskLevel.HIGH
    elif risk_score >= 0.45 or keyword_level >= SelfHarmRiskLevel.MODERATE:
        final_level = SelfHarmRiskLevel.MODERATE
    elif risk_score >= 0.2 or keyword_level >= SelfHarmRiskLevel.POTENTIAL:
        final_level = SelfHarmRiskLevel.POTENTIAL
    else:
        final_level = SelfHarmRiskLevel.NONE

    # 是否触发升级
    should_escalate = final_level >= SelfHarmRiskLevel.HIGH

    # 建议动作
    actions = {
        SelfHarmRiskLevel.NONE: "无需特殊处理",
        SelfHarmRiskLevel.POTENTIAL: "持续关注，增加情绪监测频率",
        SelfHarmRiskLevel.MODERATE: "主动发起关怀对话，评估安全状态",
        SelfHarmRiskLevel.HIGH: "立即触发危机升级协议，通知咨询师和紧急联系人",
    }

    return SelfHarmRiskResult(
        risk_level=final_level,
        risk_score=round(risk_score, 3),
        matched_keywords=all_keywords,
        matched_patterns=all_patterns,
        evidence=all_evidence,
        should_escalate=should_escalate,
        recommended_action=actions[final_level],
        timestamp=time.time(),
    )


def batch_detect(
    texts: list[str],
    emotion: Optional[EmotionResult] = None,
) -> list[SelfHarmRiskResult]:
    """批量检测

    Args:
        texts: 文本列表
        emotion: 情绪结果（可选，共用）

    Returns:
        检测结果列表
    """
    return [detect_self_harm_risk(text=t, emotion=emotion) for t in texts]
