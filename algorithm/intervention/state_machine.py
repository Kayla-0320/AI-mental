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

    五个状态对应心理咨询的典型阶段：
    - INIT: 建立关系（1-2 轮）
    - EXPLORE: 探索问题（3-8 轮）
    - INTERVENE: 实施干预（持续至风险降低或轮次上限）
    - CRISIS: 危机响应（任何时刻都可能触发）
    - CLOSE: 结束对话（风险降低且完成干预后）
    """
    INIT = "INIT"
    EXPLORE = "EXPLORE"
    INTERVENE = "INTERVENE"
    CRISIS = "CRISIS"
    CLOSE = "CLOSE"


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
) -> DialogState:
    """状态转移函数

    基于当前状态、风险等级、对话轮次和情感趋势，决定下一个状态。

    Args:
        current_state: 当前对话状态
        risk_level: 当前风险评估等级（来自 assessment 模块）
        turn_count: 当前对话轮次计数
        emotion_trend: 情感趋势（"improving" / "stable" / "worsening" / None）

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
    # 规则 3：CRISIS 状态只在风险降至 LOW 后允许转移到 CLOSE
    # 设计理由：危机解除后不能直接回到干预阶段，需要经过总结确认。
    #          如果风险仍为 MEDIUM/HIGH，保持在 CRISIS 等待升级层处理。
    # --------------------------------------------------------
    if current_state == DialogState.CRISIS:
        if risk_level == RiskLevel.LOW:
            # 危机解除 → 进入结束阶段做安全确认总结
            return DialogState.CLOSE
        # 风险未降至 LOW → 保持在 CRISIS，等待升级层介入
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
    # 规则 5：EXPLORE → INTERVENE（探索充分后进入干预）
    # 设计理由：
    #   - 至少探索 EXPLORE_MIN_TURNS 轮（确保信息充分）
    #   - 风险为 LOW/MEDIUM 且轮次足够 → 进入干预
    #   - 风险为 HIGH → 进入干预（需要更积极的引导）
    #   - 超过 EXPLORE_MAX_TURNS → 强制进入干预（防止无限探索）
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

        # 如果初始评估就是 CRISIS，直接进入危机状态
        if risk_level == RiskLevel.CRISIS:
            self.current_state = DialogState.CRISIS
            self.state_history.append((0, DialogState.CRISIS))

        return select_action(self.current_state, self.turn_count, risk_level)

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
        # 状态转移
        next_state = transition(
            self.current_state,
            risk_level,
            self.turn_count,
            emotion_trend,
        )

        # 记录状态变化
        if next_state != self.current_state:
            self.current_state = next_state
            self.state_history.append((self.turn_count, next_state))

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
            "turn_count": self.turn_count,
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
    EXPLORE --> INTERVENE : turn_count >= 2 且\\nrisk ∈ {LOW, MEDIUM}\\n（探索充分）
    INTERVENE --> CLOSE : risk = LOW 且\\ntrend = improving\\n（干预有效）

    %% 危机触发（任何状态）
    INIT --> CRISIS : risk = CRISIS\\n或 risk = HIGH 且\\ntrend = worsening
    EXPLORE --> CRISIS : risk = CRISIS\\n或 risk = HIGH 且\\ntrend = worsening
    INTERVENE --> CRISIS : risk = CRISIS\\n或 risk = HIGH 且\\ntrend = worsening

    %% 危机恢复
    CRISIS --> CLOSE : risk = LOW\\n（危机解除）

    %% 风险回升
    INTERVENE --> EXPLORE : risk = HIGH\\n（风险回升，重新评估）

    %% 强制结束
    EXPLORE --> INTERVENE : turn_count >= 8\\n（防止无限探索）
    INTERVENE --> CLOSE : turn_count >= 15\\n（防止过度依赖）

    %% 终态
    CLOSE --> [*] : 对话结束

    %% 自循环
    INIT --> INIT : turn_count < 2
    EXPLORE --> EXPLORE : 继续探索
    INTERVENE --> INTERVENE : 继续干预
    CRISIS --> CRISIS : risk ≠ LOW\\n（等待升级层）

    %% 样式标注
    note right of CRISIS
        🔴 必须触发升级层接口
        🔴 禁止生成诊断性语言
    end note
```
"""
