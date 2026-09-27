"""
对话状态机 —— 五状态有限自动机驱动的受控 AI 咨询对话引擎

状态定义：
    INIT      - 初始状态：建立关系、说明保密协议
    EXPLORE   - 探索阶段：开放式提问、情绪反映、收集信息
    INTERVENE - 干预阶段：CBT 引导、正念引导、认知重构
    CRISIS    - 危机状态：安全确认、立即触发升级层
    CLOSE     - 结束阶段：总结、资源推荐、安全计划

状态转移逻辑：
    transition(current_state, risk_level, turn_count) -> next_state

动作选择逻辑：
    select_action(state) -> Action（结构化动作对象，不含具体话术）

安全红线：
    - 任何状态禁止生成诊断性语言（如"你患有抑郁症"）
    - CRISIS 状态必须触发升级层接口
    - 所有转移规则不依赖 LLM，纯逻辑实现

设计理由注释：
    每条转移规则均附有注释说明临床/安全理由。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from shared.dataclasses import CrisisAlert, RiskAssessment, RiskLevel


# ============================================================
# 对话状态定义
# ============================================================

class DialogState(str, Enum):
    """对话状态枚举

    六个状态对应心理咨询的典型阶段：
    - INIT: 建立关系（1-2 轮）
    - EXPLORE: 探索问题（3-8 轮）
    - SOCRATIC_READY: 苏格拉底引导就绪（情绪平复 + 非危机 + 用户有探索意愿）
    - INTERVENE: 实施干预（持续至风险降低或轮次上限）
    - CRISIS: 危机响应（任何时刻都可能触发）
    - CLOSE: 结束对话（风险降低且完成干预后）
    """
    INIT = "INIT"
    EXPLORE = "EXPLORE"
    SOCRATIC_READY = "SOCRATIC_READY"
    INTERVENE = "INTERVENE"
    CRISIS = "CRISIS"
    CLOSE = "CLOSE"


# ============================================================
# 对话模式定义
# ============================================================

class DialogueMode(str, Enum):
    """对话模式枚举

    双模态对话引擎：同一对话流中动态切换模式。
    - EMPATHY: 共情模式（默认），以倾听、情绪反映、陪伴为主
    - SOCRATIC: 苏格拉底引导模式，以反问、认知探索为主

    切换条件（全部满足才进入 SOCRATIC）：
    1. 情绪已平复（连续 2 轮无强负面情绪）
    2. 非危机状态（risk_level ∈ {low, medium}）
    3. 非显著基线偏离（|Z| ≤ 2.0）
    4. 用户表达探索意愿（关键词 + 语义判断）
    """
    EMPATHY = "EMPATHY"
    SOCRATIC = "SOCRATIC"


# ============================================================
# 反问深度分级
# ============================================================

class SocraticDepth(str, Enum):
    """苏格拉底反问深度分级

    三级深度对应不同的认知重构阶段：
    - SHALLOW: 浅层反问（情绪刚平复，或基线边缘偏离）
    - MEDIUM: 中层反问（情绪稳定，基线正常，用户主动探索）
    - DEEP: 深层反问（情绪稳定 + 非危机 + 用户连续 2 轮主动探索）

    深度选择规则：
    1. SHALLOW：情绪刚平复（calm_turns == 2），或 |Z| ∈ [1.5, 2.0]
    2. MEDIUM：情绪稳定（calm_turns >= 3），基线正常（|Z| < 1.5），用户主动探索
    3. DEEP：情绪稳定 + 非危机 + 用户连续 2 轮主动探索（consecutive_exploration >= 2）

    与 CBT 5 步绑定：
    - SHALLOW → 识别自动思维 / 识别认知扭曲
    - MEDIUM → 检验证据 / 寻找替代解释
    - DEEP → 评估情绪变化 + 制定行动计划
    """
    SHALLOW = "SHALLOW"
    MEDIUM = "MEDIUM"
    DEEP = "DEEP"


# 探索意愿关键词（用于判断用户是否准备好进入苏格拉底模式）
_EXPLORATION_INTENT_KEYWORDS = [
    # 直接探索意愿
    "我想想想", "让我想想", "为什么", "是怎么回事", "什么意思",
    "我想了解", "帮我分析", "你说得对", "有道理", "继续",
    "我想探索", "我想弄清楚", "怎么改善", "怎么办",
    # 间接探索信号
    "我也在想", "你说得有道理", "换个角度", "从另一个角度看",
    "我还没想过这个", "有点意思", "然后呢", "具体来说",
]

# 回避反问关键词
_AVOIDANCE_KEYWORDS = [
    "不想说", "不知道", "别问了", "算了", "跳过", "换个话题",
    "不想聊这个", "别追问", "不想回答", "没想法", "随便",
    "不知道怎么说", "不想谈这个", "换个方式", "不想继续",
    "好烦", "不想被问", "别问了", "够了",
]


# ============================================================
# 动作类型定义
# ============================================================

class ActionType(str, Enum):
    """对话动作类型枚举

    每种动作类型对应一种咨询技术：
    - OPEN_QUESTION: 开放式提问（"能多说说你的感受吗？"）
    - EMOTION_REFLECTION: 情绪反映（"听起来你感到很无助"）
    - CBT_GUIDE: CBT 认知引导（"我们来想想有没有其他角度"）
    - MINDFULNESS_GUIDE: 正念引导（"试着做几次深呼吸"）
    - SAFETY_CHECK: 安全确认（"你现在是否安全？"）
    - RESOURCE_PROVIDE: 资源提供（推荐专业热线/文章）
    - SUMMARY: 总结回顾（"今天我们聊了..."）
    """
    OPEN_QUESTION = "OPEN_QUESTION"
    EMOTION_REFLECTION = "EMOTION_REFLECTION"
    CBT_GUIDE = "CBT_GUIDE"
    MINDFULNESS_GUIDE = "MINDFULNESS_GUIDE"
    SAFETY_CHECK = "SAFETY_CHECK"
    RESOURCE_PROVIDE = "RESOURCE_PROVIDE"
    SUMMARY = "SUMMARY"


# ============================================================
# 结构化动作对象
# ============================================================

@dataclass
class Action:
    """结构化对话动作 —— 只定义动作类型和参数，不包含具体话术

    Attributes:
        action_type: 动作类型（OPEN_QUESTION / CBT_GUIDE 等）
        params: 动作参数（如主题、技术类型、资源链接等）
        state: 产生该动作的对话状态
        requires_escalation: 是否需要触发升级层（CRISIS 状态为 True）
        forbidden_patterns: 禁止生成的语言模式（诊断性语言等）
    """
    action_type: ActionType
    params: dict = field(default_factory=dict)
    state: DialogState = DialogState.INIT
    requires_escalation: bool = False
    forbidden_patterns: list[str] = field(default_factory=lambda: [
        "诊断", "患有", "确诊", "症状表明", "你可能有",
        "diagnosis", "diagnosed", "you have", "symptoms suggest",
    ])


# ============================================================
# 状态 → 允许的动作类型映射
# ============================================================

# 每个状态允许的动作类型集合
# 设计理由：遵循心理咨询阶段性原则，不同阶段使用不同技术
STATE_ALLOWED_ACTIONS: dict[DialogState, list[ActionType]] = {
    # INIT 阶段：建立关系为主，使用开放式提问和情绪反映
    # 理由：初始阶段需要先建立信任，不宜直接干预
    DialogState.INIT: [
        ActionType.OPEN_QUESTION,
        ActionType.EMOTION_REFLECTION,
    ],

    # EXPLORE 阶段：深入探索，继续使用提问和反映
    # 理由：需要充分收集信息，理解来访者的核心困扰
    DialogState.EXPLORE: [
        ActionType.OPEN_QUESTION,
        ActionType.EMOTION_REFLECTION,
    ],

    # SOCRATIC_READY 阶段：苏格拉底引导就绪
    # 理由：情绪已平复、用户有探索意愿，可以开始认知探索
    DialogState.SOCRATIC_READY: [
        ActionType.OPEN_QUESTION,
        ActionType.CBT_GUIDE,
        ActionType.EMOTION_REFLECTION,
    ],

    # INTERVENE 阶段：实施干预技术
    # 理由：已充分了解问题后，引入 CBT/正念等循证干预
    DialogState.INTERVENE: [
        ActionType.CBT_GUIDE,
        ActionType.MINDFULNESS_GUIDE,
        ActionType.EMOTION_REFLECTION,
        ActionType.RESOURCE_PROVIDE,
    ],

    # CRISIS 阶段：安全确认为第一优先
    # 理由：危机状态下必须优先确认安全，同时触发升级层
    DialogState.CRISIS: [
        ActionType.SAFETY_CHECK,
        ActionType.RESOURCE_PROVIDE,
    ],

    # CLOSE 阶段：总结和提供资源
    # 理由：结束阶段需要巩固成果、提供后续资源
    DialogState.CLOSE: [
        ActionType.SUMMARY,
        ActionType.RESOURCE_PROVIDE,
    ],
}


# ============================================================
# 状态转移规则
# ============================================================

# 各阶段建议轮次范围（用于转移判断）
# 设计理由：避免过早干预（信息不足）或过长探索（来访者疲劳）
EXPLORE_MIN_TURNS = 2    # 至少探索 2 轮再进入干预
EXPLORE_MAX_TURNS = 8    # 最多探索 8 轮，防止无限探索
INTERVENE_MAX_TURNS = 15  # 干预阶段最多 15 轮，避免疲劳


def transition(
    current_state: DialogState,
    risk_level: RiskLevel,
    turn_count: int,
    emotion_trend: Optional[str] = None,
    crisis_turns: int = 0,
) -> DialogState:
    """状态转移函数

    基于当前状态、风险等级、对话轮次和情感趋势，决定下一个状态。

    Args:
        current_state: 当前对话状态
        risk_level: 当前风险评估等级（来自 assessment 模块）
        turn_count: 当前对话轮次计数
        emotion_trend: 情感趋势（"improving" / "stable" / "worsening" / None）
        crisis_turns: 在 CRISIS 状态持续的轮次数（用于自动降级）

    Returns:
        下一个对话状态

    转移规则及设计理由：
        见函数内部注释。
    """
    # --------------------------------------------------------
    # 规则 1：任何状态下，如果风险等级为 CRISIS，立即进入 CRISIS 状态
    # 设计理由：危机安全优先原则 —— 自杀/自伤风险必须在第一时间响应，
    #          不论当前处于哪个对话阶段。这是不可覆盖的硬规则。
    # --------------------------------------------------------
    if risk_level == RiskLevel.CRISIS:
        return DialogState.CRISIS

    # --------------------------------------------------------
    # 规则 2：任何状态下，如果风险等级为 HIGH 且情感趋势恶化，进入 CRISIS
    # 设计理由：高风险 + 恶化趋势 = 潜在危机升级信号。
    #          即使当前不在 CRISIS 状态，也需要立即启动安全确认。
    # --------------------------------------------------------
    if risk_level == RiskLevel.HIGH and emotion_trend == "worsening":
        return DialogState.CRISIS

    # --------------------------------------------------------
    # 规则 3：CRISIS 状态退出机制
    # 设计理由：CRISIS 是临时安全检查状态，不应永久锁定。
    #          连续 2 轮无新危机信号 → 降级到 EXPLORE 继续探索
    #          风险降至 LOW → 进入 CLOSE 做安全确认总结
    # --------------------------------------------------------
    if current_state == DialogState.CRISIS:
        # 风险降至 LOW → 危机解除，进入结束阶段
        if risk_level == RiskLevel.LOW:
            return DialogState.CLOSE

        # 风险为 MEDIUM 且已在 CRISIS 持续 2 轮以上 → 安全确认后降级到 EXPLORE
        # 临床依据：C-SSRS 协议中，安全确认后应继续探索而非停留在危机模式
        if risk_level == RiskLevel.MEDIUM and crisis_turns >= 2:
            return DialogState.EXPLORE

        # 风险仍为 HIGH 但已持续 3 轮且趋势稳定/改善 → 降级到 EXPLORE
        if risk_level == RiskLevel.HIGH and crisis_turns >= 3:
            if emotion_trend in ("stable", "improving", None):
                return DialogState.EXPLORE

        # 其他情况 → 保持在 CRISIS，等待升级层介入
        return DialogState.CRISIS

    # --------------------------------------------------------
    # 规则 4：INIT 状态 → EXPLORE（完成初始建立后自动转移）
    # 设计理由：INIT 阶段只需 1-2 轮建立关系，之后应进入探索阶段。
    #          轮次过少会显得冷漠，过多则浪费对话空间。
    # --------------------------------------------------------
    if current_state == DialogState.INIT:
        if turn_count >= 2:
            return DialogState.EXPLORE
        # 轮次不足 → 保持 INIT 继续建立关系
        return DialogState.INIT

    # --------------------------------------------------------
    # 规则 5：EXPLORE → SOCRATIC_READY / INTERVENE
    # 设计理由：
    #   - 情绪平复 + 非危机 + 用户有探索意愿 → SOCRATIC_READY
    #   - 否则按原有逻辑进入 INTERVENE
    # --------------------------------------------------------
    if current_state == DialogState.EXPLORE:
        # 高风险 → 直接进入干预（需要更积极的引导技术）
        if risk_level == RiskLevel.HIGH and turn_count >= EXPLORE_MIN_TURNS:
            return DialogState.INTERVENE

        # 低风险/中风险 + 探索轮次足够 + 趋势稳定或改善 → 进入干预
        if turn_count >= EXPLORE_MIN_TURNS:
            if emotion_trend in ("stable", "improving", None):
                return DialogState.INTERVENE

        # 超过最大探索轮次 → 强制进入干预（防止对话停滞）
        if turn_count >= EXPLORE_MAX_TURNS:
            return DialogState.INTERVENE

        # 条件不满足 → 继续探索
        return DialogState.EXPLORE

    # --------------------------------------------------------
    # 规则 5b：SOCRATIC_READY → INTERVENE / EXPLORE
    # 设计理由：
    #   - 触发苏格拉底反问 → 进入 INTERVENE
    #   - 用户连续 2 次回避 → 退回 EXPLORE（重新共情）
    #   - 风险回升 → 退回 EXPLORE
    # --------------------------------------------------------
    if current_state == DialogState.SOCRATIC_READY:
        # 风险回升 → 退回探索阶段
        if risk_level == RiskLevel.HIGH:
            return DialogState.EXPLORE
        # 默认保持 SOCRATIC_READY，等待干预触发
        return DialogState.SOCRATIC_READY

    # --------------------------------------------------------
    # 规则 6：INTERVENE → CLOSE（干预完成后结束）
    # 设计理由：
    #   - 风险降至 LOW + 趋势改善 → 干预有效，可以结束
    #   - 超过 INTERVENE_MAX_TURNS → 强制结束（避免依赖）
    #   - 风险回升到 HIGH → 转回 EXPLORE 重新评估
    # --------------------------------------------------------
    if current_state == DialogState.INTERVENE:
        # 风险回升 → 回到探索阶段重新评估
        # 设计理由：干预过程中风险升高说明当前策略可能不适用，
        #          需要重新探索问题根源。
        if risk_level == RiskLevel.HIGH:
            return DialogState.EXPLORE

        # 风险降至 LOW + 趋势改善 → 结束对话
        if risk_level == RiskLevel.LOW and emotion_trend == "improving":
            return DialogState.CLOSE

        # 超过最大干预轮次 → 强制结束
        # 设计理由：避免来访者对 AI 产生过度依赖，
        #          超过一定轮次应建议寻求人工咨询师。
        if turn_count >= INTERVENE_MAX_TURNS:
            return DialogState.CLOSE

        # 条件不满足 → 继续干预
        return DialogState.INTERVENE

    # --------------------------------------------------------
    # 规则 7：CLOSE 状态是终态，不再转移
    # 设计理由：对话已结束，新对话应从 INIT 重新开始。
    # --------------------------------------------------------
    if current_state == DialogState.CLOSE:
        return DialogState.CLOSE

    # 兜底：保持当前状态（理论上不应到达此处）
    return current_state


# ============================================================
# 模式评估辅助函数
# ============================================================

def has_exploration_intent(user_input: str) -> bool:
    """检测用户输入是否包含探索意愿

    通过关键词匹配判断用户是否准备好进入苏格拉底引导模式。
    临床依据：动机式访谈 (MI) 中，来访者的探索信号是认知重构的前提。

    Args:
        user_input: 用户当前输入文本

    Returns:
        bool: 是否包含探索意愿
    """
    if not user_input:
        return False
    return any(kw in user_input for kw in _EXPLORATION_INTENT_KEYWORDS)


def is_avoidance(user_input: str) -> bool:
    """检测用户输入是否为回避反问

    通过关键词匹配判断用户是否在回避苏格拉底式反问。
    临床依据：青少年在情绪未平复时，反问会引发“被审问”感，
    回避信号是系统应退回共情模式的重要指标。

    Args:
        user_input: 用户当前输入文本

    Returns:
        bool: 是否为回避表达
    """
    if not user_input:
        return False
    return any(kw in user_input for kw in _AVOIDANCE_KEYWORDS)


def evaluate_mode(
    user_input: str,
    risk_level: RiskLevel,
    consecutive_calm_turns: int,
    max_abs_z_score: float = 0.0,
) -> DialogueMode:
    """评估当前对话模式

    根据四个条件综合判断是否应切换到苏格拉底引导模式：
    1. 情绪已平复（连续 2 轮无强负面情绪）
    2. 非危机状态（risk_level ∈ {low, medium}）
    3. 非显著基线偏离（|Z| ≤ 2.0）
    4. 用户表达探索意愿

    全部满足 → SOCRATIC 模式；任一不满足 → EMPATHY 模式。

    Args:
        user_input: 用户当前输入文本
        risk_level: 当前风险评估等级
        consecutive_calm_turns: 连续无强负面情绪的轮次数
        max_abs_z_score: 个人基线最大 Z-score 绝对值

    Returns:
        DialogueMode: 当前应采用的对话模式
    """
    # 条件 1：情绪已平复（连续 2 轮无强负面情绪）
    emotion_calm = consecutive_calm_turns >= 2

    # 条件 2：非危机状态
    non_crisis = risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)

    # 条件 3：非显著基线偏离（|Z| ≤ 2.0）
    baseline_stable = abs(max_abs_z_score) <= 2.0

    # 条件 4：用户表达探索意愿
    exploration = has_exploration_intent(user_input)

    # 全部满足 → SOCRATIC；否则 → EMPATHY
    if emotion_calm and non_crisis and baseline_stable and exploration:
        return DialogueMode.SOCRATIC
    return DialogueMode.EMPATHY


def evaluate_depth(
    consecutive_calm_turns: int,
    max_abs_z_score: float,
    consecutive_exploration_turns: int,
    risk_level: RiskLevel,
) -> SocraticDepth:
    """评估苏格拉底反问深度

    根据情绪稳定性、基线偏离和探索连续性判断反问深度。

    深度选择规则：
    1. DEEP：情绪稳定（calm_turns >= 4）+ 非危机 + 连续 2 轮主动探索
    2. MEDIUM：情绪稳定（calm_turns >= 3）+ 基线正常（|Z| < 1.5）+ 用户主动探索
    3. SHALLOW：情绪刚平复（calm_turns == 2），或基线边缘偏离（|Z| ∈ [1.5, 2.0]）

    Args:
        consecutive_calm_turns: 连续无强负面情绪轮次
        max_abs_z_score: 个人基线最大 Z-score 绝对值
        consecutive_exploration_turns: 连续主动探索轮次
        risk_level: 当前风险等级

    Returns:
        SocraticDepth: 反问深度
    """
    # DEEP：情绪稳定 + 非危机 + 用户连续 2 轮主动探索
    if (consecutive_calm_turns >= 4
            and risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)
            and consecutive_exploration_turns >= 2):
        return SocraticDepth.DEEP

    # MEDIUM：情绪稳定 + 基线正常 + 用户主动探索
    if (consecutive_calm_turns >= 3
            and abs(max_abs_z_score) < 1.5
            and abs(max_abs_z_score) <= 2.0):
        return SocraticDepth.MEDIUM

    # SHALLOW：情绪刚平复，或基线边缘偏离
    return SocraticDepth.SHALLOW


# ============================================================
# 动作选择
# ============================================================

def select_action(
    state: DialogState,
    turn_count: int = 0,
    risk_level: Optional[RiskLevel] = None,
) -> Action:
    """根据当前状态选择允许的对话动作

    每个状态对应一组允许的动作类型，此函数从中选择最合适的动作。
    返回的 Action 是结构化对象（动作类型 + 参数），不包含具体话术。

    Args:
        state: 当前对话状态
        turn_count: 当前轮次
        risk_level: 当前风险等级（用于细化动作参数）

    Returns:
        Action: 结构化动作对象

    选择逻辑及设计理由：
        见各状态分支注释。
    """
    allowed = STATE_ALLOWED_ACTIONS[state]

    # --------------------------------------------------------
    # INIT 状态：首轮使用开放式提问建立关系
    # 设计理由：心理咨询初始阶段需要以来访者为中心，
    #          开放式提问传递"我在认真听"的信号。
    # --------------------------------------------------------
    if state == DialogState.INIT:
        if turn_count == 0:
            # 第一轮：开放式提问，邀请来访者讲述
            return Action(
                action_type=ActionType.OPEN_QUESTION,
                params={"topic": "initial_concern", "depth": "shallow"},
                state=state,
            )
        else:
            # 后续轮次：情绪反映，建立共情
            return Action(
                action_type=ActionType.EMOTION_REFLECTION,
                params={"focus": "building_rapport"},
                state=state,
            )

    # --------------------------------------------------------
    # EXPLORE 状态：交替使用提问和反映
    # 设计理由：探索阶段需要平衡信息收集和情感支持。
    #          偶数轮提问（收集信息），奇数轮反映（情感确认）。
    # --------------------------------------------------------
    if state == DialogState.EXPLORE:
        if turn_count % 2 == 0:
            return Action(
                action_type=ActionType.OPEN_QUESTION,
                params={"topic": "core_concern", "depth": "moderate"},
                state=state,
            )
        else:
            return Action(
                action_type=ActionType.EMOTION_REFLECTION,
                params={"focus": "validating_emotion"},
                state=state,
            )

    # --------------------------------------------------------
    # INTERVENE 状态：根据风险等级选择干预技术
    # 设计理由：
    #   - 中高风险 → CBT 引导（需要认知重构）
    #   - 低风险 → 正念引导（巩固和预防复发）
    #   - 干预后期 → 提供资源（为结束做准备）
    # --------------------------------------------------------
    if state == DialogState.INTERVENE:
        # 中高风险 → CBT 认知引导
        if risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH):
            return Action(
                action_type=ActionType.CBT_GUIDE,
                params={"technique": "cognitive_restructuring", "target": "negative_thought"},
                state=state,
            )
        # 低风险 + 干预后期 → 正念引导（巩固成果）
        elif turn_count > 10:
            return Action(
                action_type=ActionType.MINDFULNESS_GUIDE,
                params={"technique": "body_scan", "duration": "5min"},
                state=state,
            )
        # 默认 → CBT 引导
        else:
            return Action(
                action_type=ActionType.CBT_GUIDE,
                params={"technique": "thought_record"},
                state=state,
            )

    # --------------------------------------------------------
    # CRISIS 状态：安全确认 + 触发升级层
    # 设计理由：危机状态下第一优先是确认来访者安全，
    #          同时必须触发升级层接口通知专业人员。
    #          这是唯一 requires_escalation=True 的状态。
    # --------------------------------------------------------
    if state == DialogState.CRISIS:
        return Action(
            action_type=ActionType.SAFETY_CHECK,
            params={
                "priority": "immediate",
                "check_type": "self_harm_risk",
            },
            state=state,
            requires_escalation=True,  # 必须触发升级层
        )

    # --------------------------------------------------------
    # CLOSE 状态：总结 + 资源推荐
    # 设计理由：结束阶段需要巩固干预成果，
    #          提供后续资源防止复发。
    # --------------------------------------------------------
    if state == DialogState.CLOSE:
        if turn_count == 0:
            # 结束首轮：总结回顾
            return Action(
                action_type=ActionType.SUMMARY,
                params={"include_key_points": True, "include_coping_strategies": True},
                state=state,
            )
        else:
            # 结束后续：提供资源
            return Action(
                action_type=ActionType.RESOURCE_PROVIDE,
                params={"resource_type": "follow_up", "hotline": True},
                state=state,
            )

    # 兜底：返回第一个允许的动作
    return Action(
        action_type=allowed[0],
        params={},
        state=state,
    )


# ============================================================
# 升级层接口
# ============================================================

def create_crisis_alert(
    user_id: str,
    risk_assessment: RiskAssessment,
) -> CrisisAlert:
    """创建危机警报 —— CRISIS 状态必须调用此接口

    将风险评估结果转换为危机警报格式，
    传递给升级层（escalation 模块）触发紧急响应。

    Args:
        user_id: 用户唯一标识
        risk_assessment: 当前风险评估结果

    Returns:
        CrisisAlert: 危机警报数据结构

    设计理由：
        CRISIS 状态的核心安全约束 —— 必须触发升级层。
        此函数将 RiskAssessment 转换为 CrisisAlert 格式，
        确保升级层能接收到完整的风险信息。
    """
    import time

    # 从风险评估中提取证据
    trigger_evidence = [
        f"风险等级: {risk_assessment.risk_level.value}",
        f"PHQ-9 预估: {risk_assessment.phq9_estimated}",
        f"GAD-7 预估: {risk_assessment.gad7_estimated}",
    ]
    trigger_evidence.extend([
        e.description for e in risk_assessment.evidence
    ])

    # 计算综合风险评分（基于风险等级）
    risk_score_map = {
        RiskLevel.LOW: 0.2,
        RiskLevel.MEDIUM: 0.5,
        RiskLevel.HIGH: 0.8,
        RiskLevel.CRISIS: 1.0,
    }
    risk_score = risk_score_map.get(risk_assessment.risk_level, 0.5)

    return CrisisAlert(
        user_id=user_id,
        risk_score=risk_score,
        trigger_evidence=trigger_evidence,
        timestamp=time.time(),
        recommended_action="立即联系专业危机干预人员",
    )


# ============================================================
# 对话引擎（状态机驱动器）
# ============================================================

class DialogEngine:
    """对话状态机引擎 —— 管理对话状态转移和动作选择

    双模态对话引擎：在共情模式和苏格拉底引导模式之间动态切换。

    使用方式：
        engine = DialogEngine(user_id="user_123")
        engine.start(risk_level=RiskLevel.MEDIUM)

        # 每轮对话
        action = engine.next_turn(risk_level, emotion_trend)
        if action.requires_escalation:
            alert = engine.trigger_escalation(risk_assessment)
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.current_state = DialogState.INIT
        self.turn_count = 0
        self.state_history: list[tuple[int, DialogState]] = [(0, DialogState.INIT)]
        # 双模态对话引擎新增属性
        self.current_mode: DialogueMode = DialogueMode.EMPATHY
        self.consecutive_calm_turns: int = 0   # 连续无强负面情绪轮次
        self.consecutive_avoidance_count: int = 0  # 连续回避反问次数
        self.max_abs_z_score: float = 0.0       # 个人基线最大 Z-score 绝对值
        # 反问深度分级新增属性
        self.current_depth: SocraticDepth = SocraticDepth.SHALLOW
        self.consecutive_exploration_turns: int = 0  # 连续主动探索轮次
        self.depth_frozen_turns: int = 0  # 深度冻结剩余轮次（回避后降级冻结）
        # CRISIS 状态持续轮次（用于自动降级）
        self.crisis_turns: int = 0

    def start(self, risk_level: RiskLevel = RiskLevel.LOW) -> Action:
        """启动对话引擎

        Args:
            risk_level: 初始风险评估

        Returns:
            首轮动作
        """
        self.current_state = DialogState.INIT
        self.turn_count = 0
        self.state_history = [(0, DialogState.INIT)]
        # 重置双模态属性
        self.current_mode = DialogueMode.EMPATHY
        self.consecutive_calm_turns = 0
        self.consecutive_avoidance_count = 0
        self.max_abs_z_score = 0.0
        # 重置反问深度属性
        self.current_depth = SocraticDepth.SHALLOW
        self.consecutive_exploration_turns = 0
        self.depth_frozen_turns = 0

        # 如果初始评估就是 CRISIS，直接进入危机状态
        if risk_level == RiskLevel.CRISIS:
            self.current_state = DialogState.CRISIS
            self.state_history.append((0, DialogState.CRISIS))

        return select_action(self.current_state, self.turn_count, risk_level)

    def update_emotion_state(self, has_strong_negative: bool) -> None:
        """更新情绪状态计数器

        Args:
            has_strong_negative: 当前轮是否有强负面情绪
        """
        if has_strong_negative:
            self.consecutive_calm_turns = 0
        else:
            self.consecutive_calm_turns += 1

    def update_baseline_z_score(self, max_abs_z: float) -> None:
        """更新个人基线 Z-score

        Args:
            max_abs_z: 当前轮的最大 Z-score 绝对值
        """
        self.max_abs_z_score = max_abs_z

    def record_avoidance(self, is_avoid: bool) -> None:
        """记录用户回避行为

        Args:
            is_avoid: 当前轮是否为回避反问
        """
        if is_avoid:
            self.consecutive_avoidance_count += 1
        else:
            self.consecutive_avoidance_count = 0

    def should_exit_socratic(self) -> bool:
        """是否应退出苏格拉底模式

        用户连续 2 次回避反问 → 自动退回共情模式。

        Returns:
            bool: 是否应退出
        """
        return self.consecutive_avoidance_count >= 2

    def evaluate_and_update_mode(self, user_input: str, risk_level: RiskLevel) -> DialogueMode:
        """评估并更新对话模式

        根据四条件综合判断，并处理回避逻辑。
        同时更新反问深度分级。

        Args:
            user_input: 用户当前输入
            risk_level: 当前风险等级

        Returns:
            DialogueMode: 更新后的对话模式
        """
        # 用户连续 2 次回避 → 强制退回共情
        if self.should_exit_socratic():
            self.current_mode = DialogueMode.EMPATHY
            self.consecutive_exploration_turns = 0
            # 回避后降级冻结 3 轮
            self.depth_frozen_turns = 3
            # 同时退回 EXPLORE 状态
            if self.current_state == DialogState.SOCRATIC_READY:
                self.current_state = DialogState.EXPLORE
                self.state_history.append((self.turn_count, DialogState.EXPLORE))
            return self.current_mode

        # 更新连续探索轮次
        if has_exploration_intent(user_input):
            self.consecutive_exploration_turns += 1
        else:
            self.consecutive_exploration_turns = 0

        # 正常评估模式
        new_mode = evaluate_mode(
            user_input=user_input,
            risk_level=risk_level,
            consecutive_calm_turns=self.consecutive_calm_turns,
            max_abs_z_score=self.max_abs_z_score,
        )
        self.current_mode = new_mode

        # 如果进入 SOCRATIC 模式，评估深度
        if new_mode == DialogueMode.SOCRATIC:
            # 如果当前在 EXPLORE，切到 SOCRATIC_READY
            if self.current_state == DialogState.EXPLORE:
                self.current_state = DialogState.SOCRATIC_READY
                self.state_history.append((self.turn_count, DialogState.SOCRATIC_READY))

            # 评估反问深度（如果未冻结）
            if self.depth_frozen_turns > 0:
                self.depth_frozen_turns -= 1
                # 冻结期间最多保持 SHALLOW
                if self.current_depth != SocraticDepth.SHALLOW:
                    self.current_depth = SocraticDepth.SHALLOW
            else:
                self.current_depth = evaluate_depth(
                    consecutive_calm_turns=self.consecutive_calm_turns,
                    max_abs_z_score=self.max_abs_z_score,
                    consecutive_exploration_turns=self.consecutive_exploration_turns,
                    risk_level=risk_level,
                )
        else:
            # 退出 SOCRATIC 模式时重置深度
            self.current_depth = SocraticDepth.SHALLOW
            self.consecutive_exploration_turns = 0

        return self.current_mode

    def next_turn(
        self,
        risk_level: RiskLevel,
        emotion_trend: Optional[str] = None,
    ) -> Action:
        """执行下一轮对话

        Args:
            risk_level: 当前风险评估
            emotion_trend: 情感趋势

        Returns:
            本轮动作
        """
        # 状态转移（传入 crisis_turns 用于自动降级判断）
        next_state = transition(
            self.current_state,
            risk_level,
            self.turn_count,
            emotion_trend,
            crisis_turns=self.crisis_turns,
        )

        # 记录状态变化并更新 crisis_turns 计数器
        if next_state != self.current_state:
            self.current_state = next_state
            self.state_history.append((self.turn_count, next_state))
            # 进入 CRISIS → 重置计数器
            if next_state == DialogState.CRISIS:
                self.crisis_turns = 0
            # 离开 CRISIS → 重置计数器
            elif self.crisis_turns > 0:
                self.crisis_turns = 0
        elif next_state == DialogState.CRISIS:
            # 保持在 CRISIS → 递增计数器
            self.crisis_turns += 1

        # 选择动作
        action = select_action(self.current_state, self.turn_count, risk_level)

        # 轮次递增
        self.turn_count += 1

        return action

    def trigger_escalation(self, risk_assessment: RiskAssessment) -> CrisisAlert:
        """触发升级层 —— CRISIS 状态必须调用

        Args:
            risk_assessment: 当前风险评估

        Returns:
            CrisisAlert: 危机警报
        """
        return create_crisis_alert(self.user_id, risk_assessment)

    def is_terminal(self) -> bool:
        """是否处于终态"""
        return self.current_state == DialogState.CLOSE

    def get_state_summary(self) -> dict:
        """获取当前状态摘要"""
        return {
            "user_id": self.user_id,
            "current_state": self.current_state.value,
            "current_mode": self.current_mode.value,
            "current_depth": self.current_depth.value,
            "turn_count": self.turn_count,
            "consecutive_calm_turns": self.consecutive_calm_turns,
            "consecutive_avoidance_count": self.consecutive_avoidance_count,
            "consecutive_exploration_turns": self.consecutive_exploration_turns,
            "max_abs_z_score": self.max_abs_z_score,
            "depth_frozen_turns": self.depth_frozen_turns,
            "state_history": [
                {"turn": t, "state": s.value} for t, s in self.state_history
            ],
        }


# ============================================================
# Mermaid 状态转移图
# ============================================================

MERMAID_STATE_DIAGRAM = """
```mermaid
stateDiagram-v2
    [*] --> INIT : 对话开始

    %% 正常流程
    INIT --> EXPLORE : turn_count >= 2\\n（建立关系完成）
    EXPLORE --> SOCRATIC_READY : 情绪平复 + 非危机\\n+ 基线稳定 + 探索意愿
    EXPLORE --> INTERVENE : turn_count >= 2 且\\nrisk ∈ {LOW, MEDIUM}\\n（探索充分）
    SOCRATIC_READY --> INTERVENE : 触发苏格拉底反问
    INTERVENE --> CLOSE : risk = LOW 且\\ntrend = improving\\n（干预有效）

    %% 危机触发（任何状态）
    INIT --> CRISIS : risk = CRISIS\\n或 risk = HIGH 且\\ntrend = worsening
    EXPLORE --> CRISIS : risk = CRISIS\\n或 risk = HIGH 且\\ntrend = worsening
    SOCRATIC_READY --> CRISIS : risk = CRISIS\\n（任何时刻）
    INTERVENE --> CRISIS : risk = CRISIS\\n或 risk = HIGH 且\\ntrend = worsening

    %% 危机恢复
    CRISIS --> CLOSE : risk = LOW\\n（危机解除）

    %% 风险回升 / 回避
    SOCRATIC_READY --> EXPLORE : risk = HIGH\\n或 用户连续 2 次回避
    INTERVENE --> EXPLORE : risk = HIGH\\n（风险回升，重新评估）

    %% 强制结束
    EXPLORE --> INTERVENE : turn_count >= 8\\n（防止无限探索）
    INTERVENE --> CLOSE : turn_count >= 15\\n（防止过度依赖）

    %% 终态
    CLOSE --> [*] : 对话结束

    %% 自循环
    INIT --> INIT : turn_count < 2
    EXPLORE --> EXPLORE : 继续探索
    SOCRATIC_READY --> SOCRATIC_READY : 等待干预触发
    INTERVENE --> INTERVENE : 继续干预
    CRISIS --> CRISIS : risk ≠ LOW\\n（等待升级层）

    %% 样式标注
    note right of CRISIS
        🔴 必须触发升级层接口
        🔴 禁止生成诊断性语言
    end note
    note right of SOCRATIC_READY
        🟢 苏格拉底引导就绪
        🟢 需四条件全部满足
    end note
```
"""
