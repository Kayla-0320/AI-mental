"""
危机严重程度分级器 —— 参考 Suici-Urgency 6 级标注体系

⚠️ 声明：本模块为辅助筛查工具，不能替代专业临床评估。
    宁可误报（高召回率），不可漏报（低漏判率）。

参考数据：
    Suici-Urgency (finexsf/Suici-Urgency) —— 真实中文心理危机分级数据
    许可证：CC BY-NC 4.0（仅限学术研究）
    6 级标注：Level 1 (无风险) → Level 6 (即将实施)

分级定义：
    Level 1: 无危机信号（日常表达、一般困扰）
    Level 2: 低风险（轻度消极情绪、间接抱怨）
    Level 3: 中风险（明确绝望感、无价值感、间接自伤意念）
    Level 4: 高风险（直接自杀意念、具体计划讨论）
    Level 5: 极高风险（有准备行为、告别暗示、后事安排）
    Level 6: 紧迫风险（正在实施或即将实施）

临床依据：
    - C-SSRS (Columbia Suicide Severity Rating Scale)
    - NICE Guidelines (2011) 自杀风险评估框架
    - Joiner (2005) 人际关系自杀理论

核心接口：
    - classify_severity(text) -> CrisisSeverityResult
    - severity_to_risk_level(severity) -> RiskLevel（映射到项目统一风险等级）
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from shared.dataclasses import RiskLevel


# ============================================================
# 6 级危机严重程度
# ============================================================

class CrisisSeverity(IntEnum):
    """危机严重程度等级（参考 Suici-Urgency 6 级标注）"""
    NONE = 1          # 无危机信号
    LOW = 2           # 低风险
    MODERATE = 3      # 中风险
    HIGH = 4          # 高风险
    SEVERE = 5        # 极高风险
    IMMINENT = 6      # 紧迫风险


# 各等级映射到项目统一 RiskLevel
_SEVERITY_TO_RISK = {
    CrisisSeverity.NONE: RiskLevel.LOW,
    CrisisSeverity.LOW: RiskLevel.LOW,
    CrisisSeverity.MODERATE: RiskLevel.MEDIUM,
    CrisisSeverity.HIGH: RiskLevel.HIGH,
    CrisisSeverity.SEVERE: RiskLevel.CRISIS,
    CrisisSeverity.IMMINENT: RiskLevel.CRISIS,
}


def severity_to_risk_level(severity: CrisisSeverity) -> RiskLevel:
    """将 6 级严重程度映射到项目统一风险等级"""
    return _SEVERITY_TO_RISK.get(severity, RiskLevel.MEDIUM)


# ============================================================
# 分级结果
# ============================================================

@dataclass
class CrisisSeverityResult:
    """危机分级结果

    Attributes:
        severity: 严重程度等级 (1-6)
        confidence: 分级置信度 [0, 1]
        matched_signals: 命中的信号列表
        risk_level: 映射后的统一风险等级
        recommendation: 建议动作
        timestamp: 评估时间戳
    """
    severity: CrisisSeverity = CrisisSeverity.NONE
    confidence: float = 0.0
    matched_signals: list[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    recommendation: str = ""
    timestamp: float = 0.0


# ============================================================
# 各级信号词典
# ============================================================

# Level 6: 紧迫风险（正在实施或即将实施）
_IMMINENT_SIGNALS = [
    "正在割", "正在吃", "已经吃了", "已经割了",
    "站在楼顶", "站在天台", "站在桥上",
    "绳子已经", "药已经吃了", "安眠药都",
    "遗书已经写好", "最后的", "告别",
    "再见了", "永别了", "来世",
    "这是我最后", "写完这封信",
]

# Level 5: 极高风险（有准备行为、告别暗示、后事安排）
_SEVERE_SIGNALS = [
    "遗书", "写遗书", "写了信",
    "把东西都送人", "把东西都给别人", "交代后事",
    "把我的", "留给你们", "留给你们",
    "最后的告别", "告别信",
    "买好了", "准备好了", "工具都准备好了",
    "查了方法", "搜了方法", "怎么死",
    "对不起大家", "请原谅我", "我不值得",
    "我配不上", "没有我会更好", "你们会过得更好",
    "没有我的日子", "我走了以后",
    "攒够了药", "买了药", "收集了",
]

# Level 4: 高风险（直接自杀意念、具体计划讨论）
_HIGH_SIGNALS = [
    "想自杀", "想死", "不想活了", "想结束生命",
    "想跳楼", "想割腕", "想上吊", "想服毒",
    "怎么自杀", "自杀的方法", "死的方法",
    "活不下去", "死了算了", "不想醒来",
    "结束一切", "结束这一切",
    "伤害自己", "吃药自杀",
    "烧自己", "刺自己", "掐自己",
    "跳下去", "从楼上跳",
]

# Level 3: 中风险（明确绝望感、无价值感、间接自伤意念）
_MODERATE_SIGNALS = [
    "活着没意思", "活着没什么意义", "没有活下去的理由",
    "活着是负担", "活着好累", "不想存在",
    "消失就好了", "消失好了", "如果我不在了",
    "没人会在意", "世界没有我会更好",
    "我是废物", "没人要我", "我是个负担",
    "了结", "解脱", "一了百了", "了断",
    "撑不住了", "撑不下去",
    "不想努力了", "放弃一切",
    "不想面对", "看不到希望", "看不到出路",
    "一切都没意义", "一切都没有意义",
    "好想逃", "无尽的黑暗", "深渊",
    "人间不值得", "生而为人我很抱歉",
    "活着像行尸走肉", "灵魂已经死了",
    "心已经空了", "什么都感觉不到了",
]

# Level 2: 低风险（轻度消极情绪、间接抱怨）
_LOW_SIGNALS = [
    "好累", "好烦", "不想上学", "不想动",
    "好难受", "不开心", "心情很差",
    "没有意义", "活着干嘛", "有什么用",
    "反正没人在乎", "多我一个不多",
    "我就是个笑话", "谁都不会心疼",
    "好痛苦", "受不了了", "太累了",
    "不想说话", "不想见任何人",
    "失眠好几天", "睡不着",
    "觉得自己很没用", "我什么都做不好",
]


# 所有信号按严重程度排序（用于匹配优先级）
_ALL_LEVELS = [
    (CrisisSeverity.IMMINENT, _IMMINENT_SIGNALS),
    (CrisisSeverity.SEVERE, _SEVERE_SIGNALS),
    (CrisisSeverity.HIGH, _HIGH_SIGNALS),
    (CrisisSeverity.MODERATE, _MODERATE_SIGNALS),
    (CrisisSeverity.LOW, _LOW_SIGNALS),
]


# ============================================================
# 推荐动作映射
# ============================================================

_RECOMMENDATIONS = {
    CrisisSeverity.NONE: "状态正常，无需特殊处理",
    CrisisSeverity.LOW: "轻度关注，保持常规监测，适时提供倾听支持",
    CrisisSeverity.MODERATE: "中度关注，建议主动发起关怀对话，评估安全状态",
    CrisisSeverity.HIGH: "高度关注，立即发起安全确认，提供危机热线资源",
    CrisisSeverity.SEVERE: "紧急关注，立即触发危机干预协议，通知紧急联系人",
    CrisisSeverity.IMMINENT: "紧迫！立即拨打 120 或 110，同时通知所有紧急联系人",
}


# ============================================================
# 核心分级函数
# ============================================================

def classify_severity(text: str) -> CrisisSeverityResult:
    """危机严重程度分级

    按严重程度从高到低逐级匹配，返回最高命中等级。
    同一等级内多个信号命中可提高置信度。

    ⚠️ 本函数为基于规则的辅助筛查，不替代专业临床评估。

    Args:
        text: 用户输入文本

    Returns:
        CrisisSeverityResult: 分级结果
    """
    if not text or not text.strip():
        return CrisisSeverityResult(
            severity=CrisisSeverity.NONE,
            confidence=1.0,
            recommendation=_RECOMMENDATIONS[CrisisSeverity.NONE],
            timestamp=time.time(),
        )

    text_lower = text.lower()

    # 从高到低逐级匹配
    for severity, signals in _ALL_LEVELS:
        hits = [s for s in signals if s in text_lower]
        if hits:
            # 置信度：命中 1 个 = 0.6，2 个 = 0.75，3+ = 0.85
            n_hits = len(hits)
            if n_hits == 1:
                confidence = 0.6
            elif n_hits == 2:
                confidence = 0.75
            else:
                confidence = min(0.95, 0.65 + 0.1 * n_hits)

            return CrisisSeverityResult(
                severity=severity,
                confidence=round(confidence, 3),
                matched_signals=hits,
                risk_level=severity_to_risk_level(severity),
                recommendation=_RECOMMENDATIONS[severity],
                timestamp=time.time(),
            )

    # 无命中 → 无危机信号
    return CrisisSeverityResult(
        severity=CrisisSeverity.NONE,
        confidence=1.0,
        risk_level=RiskLevel.LOW,
        recommendation=_RECOMMENDATIONS[CrisisSeverity.NONE],
        timestamp=time.time(),
    )


def batch_classify(texts: list[str]) -> list[CrisisSeverityResult]:
    """批量危机分级"""
    return [classify_severity(t) for t in texts]


# ============================================================
# 统计信息
# ============================================================

def get_signal_stats() -> dict:
    """获取各级信号词条统计（自检用）"""
    stats = {}
    total = 0
    for severity, signals in _ALL_LEVELS:
        n = len(signals)
        stats[severity.name] = n
        total += n
    stats["total"] = total
    return stats
