"""
个性化策略生成模块

核心功能：
    将对话状态机输出的结构化 Action 对象转换为具体的回复文本。
    Action → 策略模板 → 青少年语言适配 → 安全红线检查 → 最终回复

设计原则：
    1. LLM 不直接生成话术，而是输出结构化 Action
    2. 独立的生成层将 Action 转为具体回复
    3. 确保干预策略可控、可审计
    4. 青少年友好的语言风格

Action 类型：
    - OPEN_QUESTION: 开放式提问
    - EMOTION_REFLECTION: 情感反射
    - CBT_GUIDE: CBT 认知重构引导
    - MINDFULNESS_GUIDE: 正念引导
    - SAFETY_CHECK: 安全检查
    - RESOURCE_PROVIDE: 资源提供
    - SUMMARY: 总结回顾
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# 数据结构
# ============================================================

class ActionType(str, Enum):
    """干预动作类型"""
    OPEN_QUESTION = "open_question"
    EMOTION_REFLECTION = "emotion_reflection"
    CBT_GUIDE = "cbt_guide"
    MINDFULNESS_GUIDE = "mindfulness_guide"
    SAFETY_CHECK = "safety_check"
    RESOURCE_PROVIDE = "resource_provide"
    SUMMARY = "summary"


@dataclass
class Action:
    """结构化干预动作"""
    action_type: ActionType
    emotion: str = ""                    # 当前检测到的情绪
    risk_level: str = "low"              # 当前风险等级
    dialog_state: str = "EXPLORE"        # 当前对话状态
    topic: str = ""                      # 当前话题
    params: dict = field(default_factory=dict)  # 额外参数
    forbidden_patterns: list[str] = field(default_factory=list)  # 安全红线


@dataclass
class StrategyResult:
    """策略生成结果"""
    reply: str                           # 生成的回复文本
    action_type: str                     # 动作类型
    confidence: float                    # 生成置信度
    safety_checked: bool                 # 是否通过安全检查
    word_count: int                      # 回复字数
    latency_ms: float                    # 生成延迟
    audit_trail: dict = field(default_factory=dict)  # 审计追踪


# ============================================================
# 策略模板库（青少年友好语言）
# ============================================================

STRATEGY_TEMPLATES: dict[str, list[dict]] = {
    ActionType.OPEN_QUESTION: [
        {
            "pattern": "explore_feeling",
            "templates": [
                "你刚才提到{topic}的时候，心里是什么感觉呀？",
                "能多跟我说说{topic}这件事吗？我很好奇你的感受。",
                "当你想到{topic}的时候，脑海里第一个浮现的画面是什么？",
                "如果给现在的感受打个分，0分是很糟糕、10分是很棒，你会打几分？",
            ],
        },
        {
            "pattern": "explore_context",
            "templates": [
                "这件事是从什么时候开始让你有这种感觉的？",
                "当时发生了什么让你有这种想法呢？",
                "身边有其他人注意到你的变化吗？",
            ],
        },
    ],

    ActionType.EMOTION_REFLECTION: [
        {
            "pattern": "reflect_emotion",
            "templates": [
                "听起来你现在感到{emotion}，这种感觉一定不好受。",
                "我能感受到你的{emotion}。谢谢你愿意跟我分享这些。",
                "你描述的这些感受，让我觉得你正在经历一段很{emotion_adj}的时光。",
                "如果我用「{emotion}」来形容你现在的状态，你觉得准确吗？",
            ],
        },
    ],

    ActionType.CBT_GUIDE: [
        {
            "pattern": "identify_thought",
            "templates": [
                "当你感到{emotion}的时候，脑海里有没有一个特别强烈的想法？比如……",
                "我们来试试一个小练习：当你{emotion}的时候，脑子里「自动弹出」的想法是什么？",
            ],
        },
        {
            "pattern": "examine_evidence",
            "templates": [
                "我们来当一回「小侦探」：有什么证据支持这个想法？又有什么证据不太支持？",
                "如果你的好朋友遇到同样的事，你会怎么安慰TA？",
            ],
        },
        {
            "pattern": "alternative_thought",
            "templates": [
                "有没有另一种方式来理解这件事？哪怕只有一点点不同？",
                "如果一个月后再回头看，你觉得这件事可能没那么严重吗？",
            ],
        },
    ],

    ActionType.MINDFULNESS_GUIDE: [
        {
            "pattern": "breathing",
            "templates": [
                "我们来做一个简单的呼吸练习吧。吸气……4秒……慢慢呼出……6秒……试试看？",
                "现在把注意力放在你的呼吸上。不需要改变什么，只是感受空气进出身体的感觉。",
            ],
        },
        {
            "pattern": "grounding",
            "templates": [
                "我们来做一个「5-4-3-2-1」练习：说出你能看到的5样东西、能摸到的4样东西……",
                "感受一下你现在坐着/站着的感觉，脚踩在地面上是什么感觉？",
            ],
        },
    ],

    ActionType.SAFETY_CHECK: [
        {
            "pattern": "direct_check",
            "templates": [
                "我想直接问你一个问题，可以吗？你有没有想过伤害自己？",
                "你现在的感受让我有些担心。你有没有出现过不想活了的念头？",
            ],
        },
        {
            "pattern": "safety_plan",
            "templates": [
                "如果你现在有伤害自己的想法，请立刻告诉身边的大人，或者拨打24小时心理援助热线：400-161-9995。",
                "你现在安全吗？如果感觉控制不住自己，我们可以一起联系专业的咨询师，好吗？",
            ],
        },
    ],

    ActionType.RESOURCE_PROVIDE: [
        {
            "pattern": "hotline",
            "templates": [
                "如果你觉得很难受，可以随时拨打这个电话：400-161-9995（24小时心理援助热线），那里有专业的人可以帮助你。",
                "推荐你一个APP叫「壹心理」，里面有很多适合青少年的心理自助资源。",
            ],
        },
        {
            "pattern": "technique",
            "templates": [
                "下次感到焦虑的时候，可以试试「蝴蝶拍」：双手交叉放在胸前，左右交替轻拍。",
                "你可以试试「情绪日记」：每天花3分钟写下今天的情绪和想法，坚持一周看看变化。",
            ],
        },
    ],

    ActionType.SUMMARY: [
        {
            "pattern": "session_summary",
            "templates": [
                "今天聊了很多，你提到了{topic}，也分享了你{emotion}的感受。你做得很棒，愿意面对这些不容易的事情。",
                "回顾一下我们今天的对话：你发现了自己在{topic}上的一些想法和感受。记住，改变是一步一步来的。",
            ],
        },
    ],
}

# 情绪形容词映射
EMOTION_ADJ: dict[str, str] = {
    "焦虑": "紧张不安", "悲伤": "难过", "愤怒": "气愤",
    "恐惧": "害怕", "快乐": "开心", "中性": "平淡",
}

# 安全红线关键词（生成回复中不能出现的内容）
FORBIDDEN_PATTERNS = [
    "你应该", "你必须", "你不应该", "你想太多了",
    "这没什么", "别人比你更惨", "坚强一点",
    "你不应该有这样的感觉", "这不是什么大事",
    "你太敏感了", "别矫情",
]


# ============================================================
# 策略生成器
# ============================================================

class StrategyGenerator:
    """个性化策略生成器

    将结构化 Action 转换为具体的回复文本：
    1. 根据 Action 类型选择模板
    2. 填充情绪/话题等参数
    3. 青少年语言风格适配
    4. 安全红线二次检查
    5. 回复长度控制（≤ 200字）
    """

    MAX_REPLY_LENGTH = 200

    def __init__(self):
        self._total_generated = 0
        self._total_safety_rejects = 0

    def generate(self, action: Action) -> StrategyResult:
        """根据 Action 生成回复

        Args:
            action: 结构化干预动作

        Returns:
            StrategyResult: 生成的回复及审计信息
        """
        start = time.perf_counter()
        self._total_generated += 1

        # 1. 获取对应模板
        templates = STRATEGY_TEMPLATES.get(action.action_type, [])
        if not templates:
            return StrategyResult(
                reply="我听到你说的了。能再多告诉我一些吗？",
                action_type=action.action_type.value,
                confidence=0.3,
                safety_checked=True,
                word_count=16,
                latency_ms=(time.perf_counter() - start) * 1000,
                audit_trail={"fallback": True, "reason": "no_templates"},
            )

        # 2. 选择合适的模板组
        pattern = action.params.get("pattern", "")
        selected_group = None
        for group in templates:
            if group["pattern"] == pattern:
                selected_group = group
                break
        if not selected_group:
            selected_group = random.choice(templates)

        # 3. 随机选择一个模板并填充参数
        template = random.choice(selected_group["templates"])
        reply = template.format(
            emotion=action.emotion or "不舒服",
            emotion_adj=EMOTION_ADJ.get(action.emotion, "不容易"),
            topic=action.topic or "这件事",
            **action.params,
        )

        # 4. 长度控制
        if len(reply) > self.MAX_REPLY_LENGTH:
            reply = reply[:self.MAX_REPLY_LENGTH - 3] + "……"

        # 5. 安全红线检查
        safety_passed = True
        matched_forbidden = []
        for fp in FORBIDDEN_PATTERNS:
            if fp in reply:
                safety_passed = False
                matched_forbidden.append(fp)

        if not safety_passed:
            self._total_safety_rejects += 1
            # 替换为安全兜底回复
            reply = self._safe_fallback(action)

        # 6. 检查 Action 自带的 forbidden_patterns
        for fp in action.forbidden_patterns:
            if fp in reply:
                reply = self._safe_fallback(action)
                break

        latency = (time.perf_counter() - start) * 1000

        return StrategyResult(
            reply=reply,
            action_type=action.action_type.value,
            confidence=0.85 if safety_passed else 0.5,
            safety_checked=True,
            word_count=len(reply),
            latency_ms=round(latency, 2),
            audit_trail={
                "template_pattern": selected_group["pattern"],
                "safety_passed": safety_passed,
                "forbidden_matched": matched_forbidden,
                "risk_level": action.risk_level,
                "dialog_state": action.dialog_state,
            },
        )

    def _safe_fallback(self, action: Action) -> str:
        """安全兜底回复"""
        fallbacks = [
            "我听到你说的了。你现在安全吗？如果感觉不好，可以随时告诉我。",
            "谢谢你跟我分享这些。你的感受很重要，我想好好听你说。",
            "我能感受到你现在不太舒服。你愿意多跟我说说吗？",
        ]
        if action.risk_level in ("high", "crisis"):
            return "你现在的感受让我很在意。如果你有任何伤害自己的想法，请立刻拨打24小时心理援助热线：400-161-9995。你也可以告诉我，我会帮你联系专业的咨询师。"
        return random.choice(fallbacks)

    def get_stats(self) -> dict:
        """获取生成统计"""
        return {
            "total_generated": self._total_generated,
            "safety_rejects": self._total_safety_rejects,
            "safety_reject_rate": (
                self._total_safety_rejects / self._total_generated
                if self._total_generated > 0 else 0.0
            ),
        }


# ============================================================
# 便捷函数
# ============================================================

_generator: Optional[StrategyGenerator] = None


def get_generator() -> StrategyGenerator:
    """获取全局策略生成器实例"""
    global _generator
    if _generator is None:
        _generator = StrategyGenerator()
    return _generator


def generate_reply(action: Action) -> StrategyResult:
    """便捷函数：根据 Action 生成回复"""
    return get_generator().generate(action)


# ============================================================
# 演示
# ============================================================

def main():
    """演示策略生成"""
    print("=" * 50)
    print("结构化策略生成演示")
    print("=" * 50)

    gen = StrategyGenerator()

    test_actions = [
        Action(ActionType.OPEN_QUESTION, emotion="焦虑", topic="考试压力", dialog_state="EXPLORE",
               params={"pattern": "explore_feeling"}),
        Action(ActionType.EMOTION_REFLECTION, emotion="悲伤", topic="和朋友吵架", dialog_state="EXPLORE",
               params={"pattern": "reflect_emotion"}),
        Action(ActionType.CBT_GUIDE, emotion="焦虑", topic="担心未来", dialog_state="INTERVENE",
               params={"pattern": "examine_evidence"}),
        Action(ActionType.MINDFULNESS_GUIDE, emotion="焦虑", dialog_state="INTERVENE",
               params={"pattern": "breathing"}),
        Action(ActionType.SAFETY_CHECK, emotion="悲伤", risk_level="high", dialog_state="CRISIS",
               params={"pattern": "direct_check"}),
        Action(ActionType.RESOURCE_PROVIDE, emotion="焦虑", dialog_state="INTERVENE",
               params={"pattern": "hotline"}),
        Action(ActionType.SUMMARY, emotion="快乐", topic="自我发现", dialog_state="CLOSE",
               params={"pattern": "session_summary"}),
    ]

    for i, action in enumerate(test_actions, 1):
        result = gen.generate(action)
        print(f"\n[{i}] Action: {action.action_type.value}")
        print(f"    情绪: {action.emotion} | 风险: {action.risk_level} | 状态: {action.dialog_state}")
        print(f"    回复: {result.reply}")
        print(f"    安全检查: {'✅' if result.safety_checked else '❌'} | 字数: {result.word_count} | 延迟: {result.latency_ms:.1f}ms")

    print(f"\n统计: {gen.get_stats()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
