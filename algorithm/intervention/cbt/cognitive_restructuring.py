"""
CBT 认知重构 5 步引导脚本

实现认知行为疗法 (CBT) 的核心技术 —— 认知重构 (Cognitive Restructuring)。
通过 5 个结构化步骤引导用户识别和挑战负面自动思维。

5 步流程：
    1. IDENTIFY_THOUGHT    - 识别自动思维（"你脑海中闪过了什么想法？"）
    2. IDENTIFY_DISTORTION - 识别认知扭曲类型
    3. EXAMINE_EVIDENCE    - 检验证据（支持 vs 反对）
    4. ALTERNATIVE_THOUGHT - 寻找替代解释
    5. EVALUATE_AND_PLAN   - 评估情绪变化 + 制定行动计划

设计原则：
    - 使用青少年能理解的语言，避免临床术语
    - 每步输出结构化引导话术（不含具体 LLM 生成内容）
    - 用户拒绝配合时有退出机制
    - 与对话状态机 (DialogEngine) 集成

临床依据：
    - Beck (1979) Cognitive Therapy and the Emotional Disorders
    - Beck et al. (2011) Cognitive Behavior Therapy: Basics and Beyond
    - 青少年适配参考：Stallard (2002) Think Good, Feel Good
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# CBT 步骤定义
# ============================================================

class CBTStep(str, Enum):
    """CBT 认知重构 5 步"""
    IDENTIFY_THOUGHT = "identify_thought"       # 步骤 1：识别自动思维
    IDENTIFY_DISTORTION = "identify_distortion"  # 步骤 2：识别认知扭曲
    EXAMINE_EVIDENCE = "examine_evidence"        # 步骤 3：检验证据
    ALTERNATIVE_THOUGHT = "alternative_thought"  # 步骤 4：寻找替代解释
    EVALUATE_AND_PLAN = "evaluate_and_plan"      # 步骤 5：评估 + 行动计划
    COMPLETED = "completed"                      # 完成
    EXITED = "exited"                            # 用户退出


# ============================================================
# 认知扭曲类型
# ============================================================

class CognitiveDistortion(str, Enum):
    """常见认知扭曲类型（青少年适配命名）

    来源：Burns (1980) Feeling Good，经青少年语言适配。
    """
    ALL_OR_NOTHING = "all_or_nothing"           # 非黑即白思维
    CATASTROPHIZING = "catastrophizing"          # 灾难化
    MIND_READING = "mind_reading"                # 读心术（"他们肯定觉得我很差"）
    FORTUNE_TELLING = "fortune_telling"          # 算命式预测（"我肯定会失败"）
    SHOULD_STATEMENTS = "should_statements"      # "应该"思维
    PERSONALIZATION = "personalization"          # 个人化（"都是我的错"）
    EMOTIONAL_REASONING = "emotional_reasoning"  # 情绪推理（"我感觉很糟，所以一定很糟"）
    LABELING = "labeling"                        # 贴标签（"我是个失败者"）
    MENTAL_FILTER = "mental_filter"              # 心理过滤（只关注负面）
    UNKNOWN = "unknown"                          # 无法归类


# ============================================================
# 数据结构
# ============================================================

@dataclass
class CBTGuidance:
    """单步引导话术

    Attributes:
        step: 当前步骤
        prompt: 引导提示语（面向用户）
        examples: 示例列表（帮助理解）
        follow_up_questions: 追问列表
        is_terminal: 是否为最后一步
    """
    step: CBTStep
    prompt: str
    examples: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    is_terminal: bool = False


@dataclass
class CBTSession:
    """CBT 认知重构会话状态

    记录用户在 5 步流程中的进度和输入。

    Attributes:
        user_id: 用户 ID
        current_step: 当前步骤
        automatic_thought: 用户报告的自动思维（步骤 1）
        identified_distortion: 识别出的认知扭曲类型（步骤 2）
        supporting_evidence: 支持自动思维的证据（步骤 3）
        contradicting_evidence: 反对自动思维的证据（步骤 3）
        alternative_thought: 替代思维（步骤 4）
        initial_emotion_intensity: 初始情绪强度 0-10（步骤 1/5）
        final_emotion_intensity: 最终情绪强度 0-10（步骤 5）
        action_plan: 行动计划（步骤 5）
        exited: 用户是否中途退出
        exit_reason: 退出原因
    """
    user_id: str
    current_step: CBTStep = CBTStep.IDENTIFY_THOUGHT
    automatic_thought: str = ""
    identified_distortion: Optional[CognitiveDistortion] = None
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    alternative_thought: str = ""
    initial_emotion_intensity: Optional[int] = None
    final_emotion_intensity: Optional[int] = None
    action_plan: str = ""
    exited: bool = False
    exit_reason: str = ""


# ============================================================
# 认知扭曲识别提示
# ============================================================

DISTORTION_DESCRIPTIONS: dict[CognitiveDistortion, dict] = {
    CognitiveDistortion.ALL_OR_NOTHING: {
        "name": "非黑即白",
        "description": "把事情看成只有两种极端，没有中间地带",
        "example": "「这次没考好，我彻底完了」",
        "challenge": "有没有介于「完美」和「彻底完了」之间的可能？",
    },
    CognitiveDistortion.CATASTROPHIZING: {
        "name": "灾难化",
        "description": "把一件小事想象成最坏的结果",
        "example": "「老师批评了我一句，我肯定要被开除了」",
        "challenge": "最坏的情况发生的概率有多大？有没有更可能的结果？",
    },
    CognitiveDistortion.MIND_READING: {
        "name": "读心术",
        "description": "觉得自己知道别人在想什么（通常是负面的）",
        "example": "「他们没回我消息，肯定是不想理我了」",
        "challenge": "有没有其他原因解释他们的行为？你确定你知道他们的想法吗？",
    },
    CognitiveDistortion.FORTUNE_TELLING: {
        "name": "算命式预测",
        "description": "预言未来一定是负面的",
        "example": "「我明天演讲肯定会搞砸」",
        "challenge": "你之前有没有成功过？有什么证据表明这次一定会失败？",
    },
    CognitiveDistortion.SHOULD_STATEMENTS: {
        "name": "「应该」思维",
        "description": "用「应该」「必须」来要求自己或他人",
        "example": "「我不应该感到难过」「我必须每次都做好」",
        "challenge": "「应该」的规则是谁定的？如果没做到，真的有那么严重吗？",
    },
    CognitiveDistortion.PERSONALIZATION: {
        "name": "个人化",
        "description": "把不好的事情都归咎于自己",
        "example": "「小组项目失败了，都是我的错」",
        "challenge": "这件事中有哪些因素是你无法控制的？其他人有没有责任？",
    },
    CognitiveDistortion.EMOTIONAL_REASONING: {
        "name": "情绪推理",
        "description": "因为感觉某件事是真的，就认为它一定是真的",
        "example": "「我感觉自己很笨，所以我肯定很笨」",
        "challenge": "感觉和事实是一样的吗？有没有感觉不对但事实相反的例子？",
    },
    CognitiveDistortion.LABELING: {
        "name": "贴标签",
        "description": "用一个负面标签定义自己或他人",
        "example": "「我是个失败者」「他是个坏人」",
        "challenge": "一个行为能定义一个人吗？你有没有做过相反的事情？",
    },
    CognitiveDistortion.MENTAL_FILTER: {
        "name": "心理过滤",
        "description": "只关注负面信息，忽略正面信息",
        "example": "「虽然大部分同学都支持我，但那个人反对了我，所以我很差」",
        "challenge": "如果把所有信息都放在一起看，画面是什么样的？",
    },
    CognitiveDistortion.UNKNOWN: {
        "name": "其他",
        "description": "暂时无法归类的思维模式",
        "example": "",
        "challenge": "让我们一起仔细看看这个想法，找找有没有不太合理的地方。",
    },
}


# ============================================================
# CBT 5 步引导生成器
# ============================================================

class CBTGuide:
    """CBT 认知重构 5 步引导器

    使用方式：
        guide = CBTGuide()
        session = guide.start(user_id="user_123")

        # 每步获取引导话术
        guidance = guide.get_guidance(session)
        # 用户回复后推进
        guide.record_thought(session, "我觉得没人喜欢我")
        session = guide.next_step(session)
        guidance = guide.get_guidance(session)
        # ... 重复直到完成
    """

    def __init__(self) -> None:
        self._guidance_cache: dict[CBTStep, CBTGuidance] = {}

    # ----------------------------------------------------------
    # 会话管理
    # ----------------------------------------------------------

    def start(self, user_id: str) -> CBTSession:
        """启动 CBT 认知重构会话

        Args:
            user_id: 用户唯一标识

        Returns:
            初始化的 CBT 会话
        """
        return CBTSession(user_id=user_id)

    def next_step(self, session: CBTSession) -> CBTSession:
        """推进到下一步

        Args:
            session: 当前会话状态

        Returns:
            更新后的会话（步骤 +1）
        """
        if session.exited:
            return session

        step_order = [
            CBTStep.IDENTIFY_THOUGHT,
            CBTStep.IDENTIFY_DISTORTION,
            CBTStep.EXAMINE_EVIDENCE,
            CBTStep.ALTERNATIVE_THOUGHT,
            CBTStep.EVALUATE_AND_PLAN,
        ]

        current_idx = step_order.index(session.current_step)
        if current_idx < len(step_order) - 1:
            session.current_step = step_order[current_idx + 1]
        else:
            session.current_step = CBTStep.COMPLETED

        return session

    def exit_session(
        self,
        session: CBTSession,
        reason: str = "用户选择退出",
    ) -> CBTSession:
        """安全退出 CBT 会话

        用户在任何步骤都可以选择退出，不会强制继续。
        退出时给予正向反馈，不施加压力。

        Args:
            session: 当前会话
            reason: 退出原因

        Returns:
            标记为退出的会话
        """
        session.exited = True
        session.exit_reason = reason
        session.current_step = CBTStep.EXITED
        return session

    # ----------------------------------------------------------
    # 数据记录
    # ----------------------------------------------------------

    def record_thought(self, session: CBTSession, thought: str) -> CBTSession:
        """记录用户的自动思维（步骤 1）"""
        session.automatic_thought = thought
        return session

    def record_distortion(
        self,
        session: CBTSession,
        distortion: CognitiveDistortion,
    ) -> CBTSession:
        """记录识别出的认知扭曲（步骤 2）"""
        session.identified_distortion = distortion
        return session

    def record_evidence(
        self,
        session: CBTSession,
        supporting: list[str],
        contradicting: list[str],
    ) -> CBTSession:
        """记录证据（步骤 3）"""
        session.supporting_evidence = supporting
        session.contradicting_evidence = contradicting
        return session

    def record_alternative(
        self,
        session: CBTSession,
        alternative: str,
    ) -> CBTSession:
        """记录替代思维（步骤 4）"""
        session.alternative_thought = alternative
        return session

    def record_emotion_intensity(
        self,
        session: CBTSession,
        initial: Optional[int] = None,
        final: Optional[int] = None,
    ) -> CBTSession:
        """记录情绪强度（步骤 1 和步骤 5）"""
        if initial is not None:
            session.initial_emotion_intensity = max(0, min(10, initial))
        if final is not None:
            session.final_emotion_intensity = max(0, min(10, final))
        return session

    def record_action_plan(
        self,
        session: CBTSession,
        plan: str,
    ) -> CBTSession:
        """记录行动计划（步骤 5）"""
        session.action_plan = plan
        return session

    # ----------------------------------------------------------
    # 引导话术生成
    # ----------------------------------------------------------

    def get_guidance(self, session: CBTSession) -> CBTGuidance:
        """获取当前步骤的引导话术

        Args:
            session: 当前 CBT 会话状态

        Returns:
            CBTGuidance: 包含提示语、示例和追问的引导对象
        """
        generators = {
            CBTStep.IDENTIFY_THOUGHT: self._guidance_identify_thought,
            CBTStep.IDENTIFY_DISTORTION: self._guidance_identify_distortion,
            CBTStep.EXAMINE_EVIDENCE: self._guidance_examine_evidence,
            CBTStep.ALTERNATIVE_THOUGHT: self._guidance_alternative_thought,
            CBTStep.EVALUATE_AND_PLAN: self._guidance_evaluate_and_plan,
            CBTStep.COMPLETED: self._guidance_completed,
            CBTStep.EXITED: self._guidance_exited,
        }

        generator = generators.get(session.current_step)
        if generator:
            return generator(session)

        # 兜底
        return CBTGuidance(
            step=session.current_step,
            prompt="让我们继续。",
        )

    # ----------------------------------------------------------
    # 步骤 1：识别自动思维
    # ----------------------------------------------------------

    def _guidance_identify_thought(self, session: CBTSession) -> CBTGuidance:
        """步骤 1 引导：识别自动思维

        目标：帮助用户捕捉脑海中自动闪过的负面想法。
        青少年适配：用"脑海中的小声音"比喻自动思维。
        """
        return CBTGuidance(
            step=CBTStep.IDENTIFY_THOUGHT,
            prompt=(
                "有时候，当我们心情不好的时候，脑海里会闪过一些想法，"
                "就像一个小声音在说话。这些想法出现得很快，我们可能都没注意到。\n\n"
                "现在，我想请你回想一下刚才让你心情不好的那一刻 —— "
                "你脑海中闪过了什么想法？那个「小声音」说了什么？"
            ),
            examples=[
                "「没人喜欢我」",
                "「我肯定做不好」",
                "「都是我的错」",
                "「我永远都不会变好」",
            ],
            follow_up_questions=[
                "这个想法出现的时候，你的情绪有多强烈？（0-10 分，0 是完全平静，10 是最强烈）",
                "这个想法是第一次出现，还是经常会出现？",
            ],
        )

    # ----------------------------------------------------------
    # 步骤 2：识别认知扭曲
    # ----------------------------------------------------------

    def _guidance_identify_distortion(self, session: CBTSession) -> CBTGuidance:
        """步骤 2 引导：识别认知扭曲类型

        目标：帮助用户认识到自己的思维可能存在偏差。
        青少年适配：用"思维陷阱"比喻认知扭曲，列出常见类型。
        """
        # 构建扭曲类型描述
        distortion_list = []
        for dist_type in [
            CognitiveDistortion.ALL_OR_NOTHING,
            CognitiveDistortion.CATASTROPHIZING,
            CognitiveDistortion.MIND_READING,
            CognitiveDistortion.FORTUNE_TELLING,
            CognitiveDistortion.PERSONALIZATION,
            CognitiveDistortion.LABELING,
        ]:
            info = DISTORTION_DESCRIPTIONS[dist_type]
            distortion_list.append(f"  - {info['name']}：{info['description']}（比如：{info['example']}）")

        return CBTGuidance(
            step=CBTStep.IDENTIFY_DISTORTION,
            prompt=(
                f"你刚才提到的想法是：「{session.automatic_thought}」\n\n"
                "心理学发现，当我们心情不好的时候，大脑容易掉进一些「思维陷阱」—— "
                "就是那些看起来很有道理、但其实不太准确的想法模式。\n\n"
                "来看看下面这些常见的思维陷阱，你觉得你的想法最像哪一种？\n\n"
                + "\n".join(distortion_list)
                + "\n\n如果都不太像也没关系，选「其他」就好。"
            ),
            examples=[
                "「我觉得我的想法像『非黑即白』—— 我把事情想得太绝对了」",
                "「好像是『读心术』—— 我其实在猜测别人的想法」",
            ],
            follow_up_questions=[
                "你觉得这个思维陷阱是怎么影响你的情绪的？",
                "如果换一个角度，这个想法还成立吗？",
            ],
        )

    # ----------------------------------------------------------
    # 步骤 3：检验证据
    # ----------------------------------------------------------

    def _guidance_examine_evidence(self, session: CBTSession) -> CBTGuidance:
        """步骤 3 引导：检验证据

        目标：像侦探一样收集支持和反对自动思维的证据。
        青少年适配：用"当小侦探"的比喻。
        """
        distortion_info = DISTORTION_DESCRIPTIONS.get(
            session.identified_distortion or CognitiveDistortion.UNKNOWN,
            DISTORTION_DESCRIPTIONS[CognitiveDistortion.UNKNOWN],
        )

        return CBTGuidance(
            step=CBTStep.EXAMINE_EVIDENCE,
            prompt=(
                f"你的想法：「{session.automatic_thought}」\n"
                f"思维陷阱类型：{distortion_info['name']}\n\n"
                "现在我们来当一回小侦探！\n\n"
                "请你想一想：\n\n"
                "**支持这个想法的证据有哪些？**\n"
                "（就是那些让你觉得这个想法可能是对的事情）\n\n"
                "**反对这个想法的证据有哪些？**\n"
                "（就是那些让你觉得这个想法可能不太对的事情）\n\n"
                "提示：试着像法官一样，两边都听听。"
            ),
            examples=[
                "支持：「上次考试确实没考好」",
                "反对：「但上上次考得还不错」「老师说我进步了」",
                "反对：「一次考试不能说明全部」",
            ],
            follow_up_questions=[
                "两边的证据哪边更多？",
                "有没有你之前没注意到的证据？",
            ],
        )

    # ----------------------------------------------------------
    # 步骤 4：寻找替代解释
    # ----------------------------------------------------------

    def _guidance_alternative_thought(self, session: CBTSession) -> CBTGuidance:
        """步骤 4 引导：寻找替代思维

        目标：基于证据，构建一个更平衡、更准确的想法。
        青少年适配：强调不是"正能量"，而是"更接近真相"。
        """
        distortion_info = DISTORTION_DESCRIPTIONS.get(
            session.identified_distortion or CognitiveDistortion.UNKNOWN,
            DISTORTION_DESCRIPTIONS[CognitiveDistortion.UNKNOWN],
        )

        return CBTGuidance(
            step=CBTStep.ALTERNATIVE_THOUGHT,
            prompt=(
                f"原来的想法：「{session.automatic_thought}」\n"
                f"思维陷阱：{distortion_info['name']}\n\n"
                "现在我们来看看刚才收集的证据。\n\n"
                "基于这些证据，有没有一个更平衡、更接近真相的想法？\n\n"
                "注意：不是要你强行想一个「正能量」的想法，"
                "而是找一个更准确、更公平的想法。\n\n"
                f" 提示：{distortion_info['challenge']}"
            ),
            examples=[
                "原来：「没人喜欢我」→ 替代：「有些人可能暂时不了解我，但也有关心我的人」",
                "原来：「我肯定做不好」→ 替代：「我不确定结果，但我可以尽力试试」",
                "原来：「都是我的错」→ 替代：「这件事有多个原因，我有责任但不是全部责任」",
            ],
            follow_up_questions=[
                "这个新的想法让你感觉怎么样？",
                "你觉得这个新想法比原来的想法更准确吗？",
            ],
        )

    # ----------------------------------------------------------
    # 步骤 5：评估情绪变化 + 制定行动计划
    # ----------------------------------------------------------

    def _guidance_evaluate_and_plan(self, session: CBTSession) -> CBTGuidance:
        """步骤 5 引导：评估情绪变化 + 制定行动计划

        目标：对比认知重构前后的情绪强度，制定具体的行动计划。
        """
        # 计算情绪变化
        emotion_change = ""
        if session.initial_emotion_intensity is not None:
            if session.final_emotion_intensity is not None:
                change = session.initial_emotion_intensity - session.final_emotion_intensity
                if change > 0:
                    emotion_change = (
                        f"\n\n📊 情绪变化：从 {session.initial_emotion_intensity}/10 "
                        f"降到了 {session.final_emotion_intensity}/10（降低了 {change} 分）\n"
                        "这说明认知重构对你有效果！"
                    )
                elif change == 0:
                    emotion_change = (
                        f"\n\n📊 情绪变化：{session.initial_emotion_intensity}/10 → "
                        f"{session.final_emotion_intensity}/10（没有变化）\n"
                        "没关系，有时候改变需要时间，多练习几次会有效果的。"
                    )
                else:
                    emotion_change = (
                        f"\n\n📊 情绪变化：从 {session.initial_emotion_intensity}/10 "
                        f"变成了 {session.final_emotion_intensity}/10（升高了 {abs(change)} 分）\n"
                        "这可能是因为面对负面想法本身就会让人不舒服。"
                        "这是正常的，重要的是你勇敢地面对了它。"
                    )
            else:
                emotion_change = (
                    f"\n\n📊 初始情绪强度：{session.initial_emotion_intensity}/10\n"
                    "如果方便的话，告诉我你现在的情绪强度是多少？（0-10 分）"
                )

        return CBTGuidance(
            step=CBTStep.EVALUATE_AND_PLAN,
            prompt=(
                f"原来的想法：「{session.automatic_thought}」\n"
                f"新的想法：「{session.alternative_thought}」"
                f"{emotion_change}\n\n"
                "最后一步：基于我们今天的讨论，你想在接下来的几天里做一件什么小事？\n\n"
                "不用很大，越具体越好。比如：\n"
                "- 「明天主动跟一个同学打招呼」\n"
                "- 「今天晚上写 3 件今天发生的好事」\n"
                "- 「下次有负面想法时，先停下来问问自己是不是掉进了思维陷阱」"
            ),
            examples=[
                "「这周每天写一件让我开心的小事」",
                "「下次想放弃的时候，先试试再说」",
                "「跟信任的人聊聊我的感受」",
            ],
            follow_up_questions=[
                "你打算什么时候开始这个行动？",
                "如果遇到困难，你会怎么提醒自己？",
            ],
            is_terminal=True,
        )

    # ----------------------------------------------------------
    # 完成 & 退出
    # ----------------------------------------------------------

    def _guidance_completed(self, session: CBTSession) -> CBTGuidance:
        """完成引导"""
        return CBTGuidance(
            step=CBTStep.COMPLETED,
            prompt=(
                "太棒了！你完成了整个认知重构练习 \n\n"
                f"回顾一下：\n"
                f"  原来的想法：「{session.automatic_thought}」\n"
                f"  新的想法：「{session.alternative_thought}」\n"
                f"  行动计划：{session.action_plan or '（未设置）'}\n\n"
                "记住：思维就像肌肉，越练习越强。\n"
                "下次再遇到负面想法时，可以试试这个 5 步法。\n\n"
                "如果你觉得需要更多帮助，随时可以联系专业咨询师。💙"
            ),
            is_terminal=True,
        )

    def _guidance_exited(self, session: CBTSession) -> CBTGuidance:
        """退出引导 —— 正向反馈，不施加压力"""
        return CBTGuidance(
            step=CBTStep.EXITED,
            prompt=(
                "没关系，不想继续完全可以。\n\n"
                "面对自己的想法需要勇气，你今天愿意尝试就已经很棒了。\n\n"
                "这个练习随时可以回来继续，没有压力。\n"
                "如果你想聊聊别的，或者需要其他帮助，我都在这里。💙"
            ),
            is_terminal=True,
        )

    # ----------------------------------------------------------
    # 快速执行完整流程（用于测试/演示）
    # ----------------------------------------------------------

    def run_full_session(
        self,
        user_id: str,
        automatic_thought: str,
        distortion: CognitiveDistortion = CognitiveDistortion.CATASTROPHIZING,
        supporting: Optional[list[str]] = None,
        contradicting: Optional[list[str]] = None,
        alternative: str = "",
        initial_emotion: int = 7,
        final_emotion: int = 4,
        action_plan: str = "",
    ) -> CBTSession:
        """运行完整的 CBT 认知重构会话（用于测试或演示）

        Args:
            user_id: 用户 ID
            automatic_thought: 自动思维
            distortion: 认知扭曲类型
            supporting: 支持证据
            contradicting: 反对证据
            alternative: 替代思维
            initial_emotion: 初始情绪强度
            final_emotion: 最终情绪强度
            action_plan: 行动计划

        Returns:
            完成的 CBT 会话
        """
        session = self.start(user_id)

        # 步骤 1
        self.record_thought(session, automatic_thought)
        self.record_emotion_intensity(session, initial=initial_emotion)
        self.next_step(session)

        # 步骤 2
        self.record_distortion(session, distortion)
        self.next_step(session)

        # 步骤 3
        self.record_evidence(
            session,
            supporting or [],
            contradicting or [],
        )
        self.next_step(session)

        # 步骤 4
        self.record_alternative(session, alternative)
        self.next_step(session)

        # 步骤 5
        self.record_emotion_intensity(session, final=final_emotion)
        self.record_action_plan(session, action_plan)
        self.next_step(session)

        return session
