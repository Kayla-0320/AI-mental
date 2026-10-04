"""
苏格拉底式对话模块

核心思想：
    通过引导式提问帮助用户自我反思，而非直接给出答案或建议。
    基于苏格拉底问答法（Socratic Method），通过系统性提问引导用户：
    1. 澄清自己的想法
    2. 探究想法背后的证据
    3. 从多角度审视问题
    4. 发现思维中的盲点
    5. 自主得出更合理的结论

理论依据：
    - Socratic Method (Plato, 399 BC)
    - Socratic Questioning in CBT (Beck, 1979)
    - Motivational Interviewing (Miller & Rollnick, 2013)

设计原则：
    1. 不直接给答案，而是引导用户自己发现
    2. 青少年友好：温暖、好奇、非说教
    3. 每步一个问题，不给用户压力
    4. 支持随时退出，不强制完成

与 CBT 认知重构的区别：
    - CBT 是结构化的 5 步流程，侧重识别和修正认知扭曲
    - 苏格拉底式对话更灵活，侧重通过提问激发自我反思
    - 两者可互补使用：先用苏格拉底式探索，再用 CBT 结构化干预
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

class SocraticStep(str, Enum):
    """苏格拉底式对话步骤"""
    CLARIFY = "clarify"              # 澄清问题
    EXPLORE_EVIDENCE = "evidence"    # 探究证据
    MULTIPLE_VIEWS = "views"         # 多角度思考
    FIND_BLIND_SPOT = "blind_spot"   # 发现盲点
    DRAW_CONCLUSION = "conclusion"   # 归纳总结
    COMPLETED = "completed"          # 完成
    EXITED = "exited"                # 退出


@dataclass
class SocraticPrompt:
    """单步引导内容"""
    step: SocraticStep
    question: str                    # 引导问题
    hint: str = ""                   # 提示（可选）
    follow_up: list[str] = field(default_factory=list)  # 追问选项
    empathy: str = ""                # 共情回应


@dataclass
class SocraticSession:
    """苏格拉底式对话会话"""
    session_id: str
    topic: str                       # 用户想探讨的主题
    current_step: SocraticStep = SocraticStep.CLARIFY
    step_history: list[SocraticPrompt] = field(default_factory=list)
    user_responses: list[str] = field(default_factory=list)
    emotion_context: str = ""        # 当前情绪背景
    risk_level: str = "low"          # 风险等级
    started_at: float = field(default_factory=time.time)
    turns: int = 0
    is_active: bool = True
    insight_summary: str = ""        # 最终洞察总结


# ============================================================
# 引导话术库（青少年友好语言）
# ============================================================

SOCRATIC_TEMPLATES: dict[str, list[dict]] = {
    SocraticStep.CLARIFY: [
        {
            "question": "你说{topic}让你很困扰。能帮我更具体地描述一下，最让你难受的是哪个部分吗？",
            "hint": "有时候把大问题拆成小问题，会看得更清楚。",
            "empathy": "谢谢你愿意跟我聊这个，这不容易。",
        },
        {
            "question": "如果给{topic}的困扰程度打个分，0分是完全不在意、10分是超级在意，你会打几分？",
            "hint": "不用想太久，凭直觉就好。",
            "empathy": "不管打几分，你的感受都是真实的、重要的。",
        },
        {
            "question": "当你想到{topic}的时候，脑海里第一个蹦出来的想法是什么？",
            "hint": "就是那个最快、最自动冒出来的念头。",
            "empathy": "嗯，我听到了。那个想法一定让你很不好受。",
        },
    ],

    SocraticStep.EXPLORE_EVIDENCE: [
        {
            "question": "你刚才说「{last_response}」。我们来当一回小侦探：有什么证据支持这个想法？又有什么证据不太支持？",
            "hint": "就像法庭上，我们要看看两边的证据。",
            "empathy": "你分析得很认真，这很棒。",
        },
        {
            "question": "如果你的好朋友跟你说了一模一样的话——「{last_response}」——你会怎么回应TA？",
            "hint": "有时候我们对别人比对自己更客观。",
            "empathy": "你对朋友的关心说明你内心很温暖。",
        },
        {
            "question": "回想一下，有没有过{topic}相关的事情其实没那么糟的时候？当时发生了什么？",
            "hint": "例外往往藏着我们忽略的力量。",
            "empathy": "你找到了一个不一样的时刻，这很有价值。",
        },
    ],

    SocraticStep.MULTIPLE_VIEWS: [
        {
            "question": "如果让你最信任的人（比如好朋友/家人/老师）来看这件事，TA可能会怎么说？",
            "hint": "不同的人看到的东西不一样，这很正常。",
            "empathy": "你选择的人一定对你很重要。",
        },
        {
            "question": "假设一年后回头看今天这件事，你觉得那时的你会怎么看？",
            "hint": "时间有时候是最好的咨询师。",
            "empathy": "你能想到未来的自己，说明你在思考更长远的东西。",
        },
        {
            "question": "如果这件事发生在一个你佩服的人身上（比如某个明星/角色），你会觉得TA会怎么处理？",
            "hint": "榜样有时候能给我们灵感。",
            "empathy": "你选择的人身上一定有你欣赏的品质。",
        },
    ],

    SocraticStep.FIND_BLIND_SPOT: [
        {
            "question": "你觉得在这件事里，有没有什么是你之前没注意到、但现在回想起来觉得重要的？",
            "hint": "盲点不可怕，发现它就是进步。",
            "empathy": "你能看到这一点，说明你在成长。",
        },
        {
            "question": "如果「最坏的情况」真的发生了，你觉得你能怎么应对？",
            "hint": "面对恐惧的最好方式就是看清它。",
            "empathy": "你的应对能力比你想象的要强。",
        },
        {
            "question": "有没有一种可能：这件事不完全是坏事，它也在教你一些什么？",
            "hint": "每段经历都有它想教会我们的东西。",
            "empathy": "你愿意从困难中寻找意义，这很了不起。",
        },
    ],

    SocraticStep.DRAW_CONCLUSION: [
        {
            "question": "回顾我们刚才聊的，你觉得最大的收获是什么？哪怕只是一点点。",
            "hint": "每一个小收获都值得被看见。",
            "empathy": "你今天的思考很有深度。",
        },
        {
            "question": "如果用一句话总结你今天的新发现，你会怎么说？",
            "hint": "把想法说出来，会让它更清晰。",
            "empathy": "这句话很有力量。记住它。",
        },
        {
            "question": "接下来你打算怎么做？有没有一个小小的、今天就能做的行动？",
            "hint": "改变从最小的一步开始。",
            "empathy": "你愿意行动，这本身就是勇气。",
        },
    ],
}

# 步骤顺序
STEP_ORDER = [
    SocraticStep.CLARIFY,
    SocraticStep.EXPLORE_EVIDENCE,
    SocraticStep.MULTIPLE_VIEWS,
    SocraticStep.FIND_BLIND_SPOT,
    SocraticStep.DRAW_CONCLUSION,
]


# ============================================================
# 苏格拉底式对话引导器
# ============================================================

class SocraticGuide:
    """苏格拉底式对话引导器

    通过 6 步引导流程帮助用户自我反思：
    1. 澄清问题 → 2. 探究证据 → 3. 多角度思考 → 4. 发现盲点 → 5. 归纳总结

    每步提供 1 个引导问题 + 提示 + 共情回应，
    根据用户回答动态调整下一步的提问。
    """

    def __init__(self):
        self._sessions: dict[str, SocraticSession] = {}
        self._total_sessions = 0

    def start_session(
        self,
        session_id: str,
        topic: str,
        emotion_context: str = "",
        risk_level: str = "low",
    ) -> SocraticPrompt:
        """开始苏格拉底式对话会话

        Args:
            session_id: 会话唯一标识
            topic: 用户想探讨的主题
            emotion_context: 当前情绪背景
            risk_level: 风险等级

        Returns:
            SocraticPrompt: 第一步的引导内容
        """
        self._total_sessions += 1

        session = SocraticSession(
            session_id=session_id,
            topic=topic,
            emotion_context=emotion_context,
            risk_level=risk_level,
        )
        self._sessions[session_id] = session

        # 生成第一步引导
        prompt = self._generate_prompt(session)
        session.step_history.append(prompt)

        return prompt

    def next_turn(self, session_id: str, user_response: str) -> Optional[SocraticPrompt]:
        """处理用户回复，生成下一步引导

        Args:
            session_id: 会话 ID
            user_response: 用户的回复

        Returns:
            SocraticPrompt 或 None（会话结束/退出）
        """
        session = self._sessions.get(session_id)
        if not session or not session.is_active:
            return None

        # 检查退出信号
        if self._is_exit_signal(user_response):
            session.is_active = False
            session.current_step = SocraticStep.EXITED
            return SocraticPrompt(
                step=SocraticStep.EXITED,
                question="好的，我们今天就聊到这里。你已经很勇敢了，愿意面对这些不容易的事情。随时欢迎你回来。",
                empathy="你做得很棒。记住，改变是一步一步来的。",
            )

        # 记录用户回复
        session.user_responses.append(user_response)
        session.turns += 1

        # 推进到下一步
        current_idx = STEP_ORDER.index(session.current_step) if session.current_step in STEP_ORDER else -1
        if current_idx < len(STEP_ORDER) - 1:
            session.current_step = STEP_ORDER[current_idx + 1]
        else:
            session.current_step = SocraticStep.COMPLETED
            session.is_active = False
            session.insight_summary = self._generate_insight_summary(session)

        # 生成下一步引导
        prompt = self._generate_prompt(session)
        session.step_history.append(prompt)

        return prompt

    def exit_session(self, session_id: str) -> SocraticPrompt:
        """提前结束会话"""
        session = self._sessions.get(session_id)
        if not session:
            return SocraticPrompt(
                step=SocraticStep.EXITED,
                question="会话已结束。",
            )

        session.is_active = False
        session.current_step = SocraticStep.EXITED

        if session.turns > 0:
            return SocraticPrompt(
                step=SocraticStep.EXITED,
                question="好的，我们今天就聊到这里。谢谢你的分享，你的感受很重要。",
                empathy=f"我们今天聊了 {session.turns} 轮，你做得很棒。",
            )
        else:
            return SocraticPrompt(
                step=SocraticStep.EXITED,
                question="好的，没关系。等你准备好了，随时可以来聊。",
                empathy="我一直在这里。",
            )

    def get_session(self, session_id: str) -> Optional[SocraticSession]:
        """获取会话状态"""
        return self._sessions.get(session_id)

    def get_stats(self) -> dict:
        """获取统计信息"""
        total = self._total_sessions
        completed = sum(
            1 for s in self._sessions.values()
            if s.current_step == SocraticStep.COMPLETED
        )
        exited = sum(
            1 for s in self._sessions.values()
            if s.current_step == SocraticStep.EXITED
        )
        avg_turns = (
            sum(s.turns for s in self._sessions.values()) / total
            if total > 0 else 0
        )
        return {
            "total_sessions": total,
            "completed": completed,
            "exited_early": exited,
            "completion_rate": round(completed / total, 4) if total > 0 else 0,
            "avg_turns": round(avg_turns, 2),
        }

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _generate_prompt(self, session: SocraticSession) -> SocraticPrompt:
        """根据当前步骤生成引导内容"""
        step = session.current_step
        templates = SOCRATIC_TEMPLATES.get(step, [])

        if not templates:
            return SocraticPrompt(
                step=step,
                question="感谢你的分享。你觉得还有什么想聊的吗？",
                empathy="你的每一个想法都很重要。",
            )

        template = random.choice(templates)

        # 填充模板参数
        last_response = session.user_responses[-1] if session.user_responses else ""
        question = template["question"].format(
            topic=session.topic,
            last_response=last_response[:100],  # 截断过长回复
        )
        hint = template.get("hint", "").format(topic=session.topic)
        empathy = template.get("empathy", "")

        return SocraticPrompt(
            step=step,
            question=question,
            hint=hint,
            empathy=empathy,
            follow_up=template.get("follow_up", []),
        )

    def _is_exit_signal(self, text: str) -> bool:
        """检测退出信号"""
        exit_keywords = [
            "不想聊了", "算了", "不说了", "结束吧", "就这样",
            "不想说了", "拜拜", "再见", "够了", "别问了",
            "不想聊", "不聊了", "退出",
        ]
        text_lower = text.strip().lower()
        return any(kw in text_lower for kw in exit_keywords)

    def _generate_insight_summary(self, session: SocraticSession) -> str:
        """生成洞察总结"""
        responses = session.user_responses
        if len(responses) < 2:
            return "今天的对话虽然简短，但你愿意开始思考，这本身就是进步。"

        # 简单总结：提取用户回复中的关键词
        summary_parts = [
            f"今天我们围绕「{session.topic}」进行了 {session.turns} 轮对话。",
        ]

        if session.turns >= 3:
            summary_parts.append("你认真思考了自己的想法，探索了不同的角度，这很了不起。")
        elif session.turns >= 1:
            summary_parts.append("你愿意面对这个问题，这本身就需要勇气。")

        summary_parts.append("记住，改变不需要一步到位，每天进步一点点就好。")

        return " ".join(summary_parts)


# ============================================================
# 便捷函数
# ============================================================

_guide: Optional[SocraticGuide] = None


def get_guide() -> SocraticGuide:
    """获取全局引导器实例"""
    global _guide
    if _guide is None:
        _guide = SocraticGuide()
    return _guide


def start_socratic_session(
    session_id: str,
    topic: str,
    emotion_context: str = "",
    risk_level: str = "low",
) -> SocraticPrompt:
    """便捷函数：开始苏格拉底式对话"""
    return get_guide().start_session(session_id, topic, emotion_context, risk_level)


def next_socratic_turn(session_id: str, user_response: str) -> Optional[SocraticPrompt]:
    """便捷函数：处理用户回复"""
    return get_guide().next_turn(session_id, user_response)


# ============================================================
# 演示
# ============================================================

def main():
    """演示苏格拉底式对话"""
    print("=" * 60)
    print("苏格拉底式对话演示")
    print("=" * 60)

    guide = SocraticGuide()

    # 模拟会话
    session_id = "demo_session_001"
    topic = "考试成绩不理想"

    print(f"\n📝 主题：{topic}")
    print("-" * 40)

    # 开始会话
    prompt = guide.start_session(session_id, topic, emotion_context="焦虑")
    print(f"\n[步骤1: 澄清问题]")
    print(f"💬 {prompt.question}")
    if prompt.hint:
        print(f"💡 提示：{prompt.hint}")
    if prompt.empathy:
        print(f"❤️ {prompt.empathy}")

    # 模拟用户回复
    simulated_responses = [
        "最难受的是觉得自己很笨，明明努力了还是考不好",
        "支持这个想法的证据是每次都考不好，不支持的是有时候也考得还行",
        "如果好朋友这样，我会说一次考试不代表什么",
        "也许我太关注结果了，没有看到自己学到的东西",
        "我决定了，下次考试前换个复习方法，不要死记硬背",
    ]

    for i, response in enumerate(simulated_responses, 1):
        print(f"\n👤 用户：{response}")
        result = guide.next_turn(session_id, response)
        if result:
            step_name = {
                SocraticStep.EXPLORE_EVIDENCE: "探究证据",
                SocraticStep.MULTIPLE_VIEWS: "多角度思考",
                SocraticStep.FIND_BLIND_SPOT: "发现盲点",
                SocraticStep.DRAW_CONCLUSION: "归纳总结",
                SocraticStep.COMPLETED: "完成",
            }.get(result.step, str(result.step))
            print(f"\n[步骤: {step_name}]")
            print(f"💬 {result.question}")
            if result.empathy:
                print(f"❤️ {result.empathy}")

    # 统计
    session = guide.get_session(session_id)
    if session and session.insight_summary:
        print(f"\n📋 洞察总结：{session.insight_summary}")

    stats = guide.get_stats()
    print(f"\n📊 统计：{stats}")
    print("=" * 60)


if __name__ == "__main__":
    main()
