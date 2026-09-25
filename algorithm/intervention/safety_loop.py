"""
安全闭环协调器 —— 串联「审计 → 状态机 → 升级」三模块

三个独立模块的协作闭环：
    审计器 (auditor.py)        → 发现风险（五轴检测）
    对话状态机 (state_machine.py) → 切换对话策略
    危机升级系统 (crisis.py)     → 触发人工干预

闭环流程：
    1. auditor.audit() 获取五轴审计结果
    2. 全部通过 → 返回原始回复
    3. 任一不通过 → 按严重程度分级处理：
       - 轻度 (MILD)：调用状态机重写回复策略（如谄媚、轨迹漂移）
       - 中度 (MODERATE)：替换为安全模板 + 记录事件（如污名化、妄想强化）
       - 重度 (SEVERE)：强制切换到 CRISIS 状态 + 调用 crisis.escalate()（如危机延迟）
    4. 返回 SafetyLoopResult（最终回复 + 审计结果 + 动作 + 升级状态）

设计原则：
    - 不修改现有三个模块的接口，仅新增协调层
    - 闭环日志可追溯：每次触发记录审计结果、动作类型、时间戳
    - crisis.escalate() 失败时有降级方案（返回紧急资源卡）

临床依据：
    - NICE CG185 (2011): 危机识别后首次接触即启动安全评估
    - WHO QualityRights (2019): 去污名化表达要求
    - Lambert (2013) 心理治疗统一方案: 治疗方向保持
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from shared.dataclasses import (
    AuditAction,
    AuditAxis,
    AuditResult,
    AuditVerdict,
    CrisisAlert,
    EscalationResult,
    EscalationStatus,
    RiskAssessment,
    RiskLevel,
    EvidenceItem,
)
from intervention.auditor import (
    AuditContext,
    SafetyAuditor,
    generate_crisis_resource_card,
    load_audit_config,
)
from intervention.state_machine import (
    DialogEngine,
    DialogState,
    DialogueMode,
    SocraticDepth,
    ActionType as SMActionType,
    select_action,
    transition,
)
from escalation.crisis import escalate as crisis_escalate

# 基线感知模块（延迟导入以避免循环依赖）
try:
    from assessment.personal_baseline import compute_deviation, BaselineDeviation
    _BASELINE_AVAILABLE = True
except ImportError:
    _BASELINE_AVAILABLE = False
    compute_deviation = None  # type: ignore
    BaselineDeviation = None  # type: ignore

logger = logging.getLogger(__name__)


# ============================================================
# 严重程度分级
# ============================================================

class Severity(str, Enum):
    """审计不通过的严重程度分级

    分级依据：
        - SEVERE: 直接威胁用户安全（危机延迟未升级）
          → 必须立即触发人工干预
        - MODERATE: 可能造成心理伤害（污名化、妄想强化）
          → 需要替换为安全表达并记录
        - MILD: 影响治疗效果但不直接危害安全（谄媚、轨迹漂移）
          → 通过状态机重写回复策略

    临床依据：
        NICE CG185 – 危机安全优先；
        WHO QualityRights – 去污名化；
        Lambert (2013) – 治疗方向保持
    """
    SEVERE = "severe"       # 重度：立即升级至人工
    MODERATE = "moderate"   # 中度：替换安全模板 + 记录
    MILD = "mild"           # 轻度：状态机重写策略


# 审计轴 → 严重程度映射
_AXIS_SEVERITY: dict[AuditAxis, Severity] = {
    # 重度：直接关系用户安全
    AuditAxis.CRISIS_DELAY: Severity.SEVERE,
    # 中度：可能造成心理伤害
    AuditAxis.STIGMA_REJECTION: Severity.MODERATE,
    AuditAxis.DELUSION_REINFORCEMENT: Severity.MODERATE,
    # 轻度：影响治疗效果
    AuditAxis.SYCOPHANCY: Severity.MILD,
    AuditAxis.TRAJECTORY_DRIFT: Severity.MILD,
    # 轻度：苏格拉底反问时机不当，退回共情模式
    AuditAxis.SOCRATIC_TIMING: Severity.MILD,
}


# ============================================================
# 安全回复模板
# ============================================================

# 中度不通过时的安全替换模板
_SAFE_TEMPLATES: dict[AuditAxis, str] = {
    AuditAxis.STIGMA_REJECTION: (
        "你正在经历一些困难的情绪，这些感受是真实的，"
        "也是可以理解的。很多人都会在某个阶段有类似的体验，"
        "这并不代表你有什么‘问题’。如果你愿意，"
        "我们可以一起探索一下这些感受。"
    ),
    AuditAxis.DELUSION_REINFORCEMENT: (
        "我听到你说的这些了。这些体验对你来说一定很不容易。"
        "我想了解更多你的感受——你愿意多说说吗？"
        "如果这些想法让你很不安，我们可以一起想想"
        "有什么方法能让你感觉好一些。"
    ),
}

# 苏格拉底反问时机审计不通过时的共情回复模板
# 设计理由：不继续追问，而是退回共情模式，让用户感到安全。
_SOCRATIC_FALLBACK_TEMPLATE: str = (
    "没关系，我们换个方式聊聊。"
    "你现在最想说的是什么？"
)

# 重度不通过时的危机回复模板
_CRISIS_RESPONSE_TEMPLATE: str = (
    "你说的这些我很重视。你现在的安全是最重要的。\n\n"
    "我想先确认一下：你现在是否处于安全的环境中？"
    "有没有伤害自己的想法？\n\n"
    "如果你现在很难受，请记住有人愿意帮助你：\n"
    "{hotline_info}"
)


# ============================================================
# 闭环日志条目
# ============================================================

@dataclass
class SafetyLoopLogEntry:
    """闭环日志条目 —— 每次 process() 调用的可追溯记录

    Attributes:
        log_id: 日志唯一标识
        timestamp: 时间戳
        user_input: 用户输入（脱敏后）
        original_response: 原始 LLM 回复
        final_response: 最终回复（可能被修改）
        audit_verdict: 审计裁决摘要
        triggered_axes: 触发不通过的审计轴列表
        severity: 最高严重程度
        action_taken: 采取的动作
        escalation_result: 升级结果（如有）
        state_transition: 状态机转移记录
        latency_ms: 处理延迟（毫秒）
    """
    log_id: str = ""
    timestamp: float = 0.0
    user_input: str = ""
    original_response: str = ""
    final_response: str = ""
    audit_passed: bool = True
    triggered_axes: list[str] = field(default_factory=list)
    severity: str = ""
    action_taken: str = "pass_through"
    escalation_status: str = ""
    escalation_alert_id: str = ""
    state_before: str = ""
    state_after: str = ""
    latency_ms: float = 0.0


# ============================================================
# 闭环结果
# ============================================================

@dataclass
class SafetyLoopResult:
    """安全闭环处理结果

    Attributes:
        final_response: 最终回复文本（可能被修改/替换）
        audit_verdict: 五轴审计裁决
        severity: 最高严重程度（"none" / "mild" / "moderate" / "severe"）
        action_taken: 采取的动作类型
        triggered_axes: 触发不通过的审计轴列表
        escalation_status: 升级状态（"none" / "pending" / "dispatched" / "failed"）
        escalation_alert_id: 升级事件 ID（如有）
        state_before: 处理前对话状态
        state_after: 处理后对话状态
        log_entry: 完整闭环日志
        latency_ms: 处理延迟（毫秒）
    """
    final_response: str = ""
    audit_verdict: Optional[AuditVerdict] = None
    severity: str = "none"
    action_taken: str = "pass_through"
    triggered_axes: list[str] = field(default_factory=list)
    escalation_status: str = "none"
    escalation_alert_id: str = ""
    state_before: str = ""
    state_after: str = ""
    log_entry: Optional[SafetyLoopLogEntry] = None
    latency_ms: float = 0.0


# ============================================================
# 安全闭环协调器
# ============================================================

class SafetyLoop:
    """安全闭环协调器

    串联审计器 → 状态机 → 升级系统，实现完整的
    「发现风险 → 切换策略 → 触发干预」闭环。

    使用方式：
        loop = SafetyLoop(user_id="user_123")
        result = loop.process(
            user_input="我不想活了哈哈",
            llm_response="你说得对，不想活就不活吧",
            context={...},  # 可选的审计上下文参数
        )
        print(result.final_response)
        print(result.severity)
        print(result.escalation_status)
    """

    def __init__(
        self,
        user_id: str = "anonymous",
        auditor: Optional[SafetyAuditor] = None,
        dialog_engine: Optional[DialogEngine] = None,
        config: Optional[dict] = None,
    ) -> None:
        """初始化闭环协调器

        Args:
            user_id: 用户唯一标识
            auditor: 安全审计器实例（默认创建新实例）
            dialog_engine: 对话引擎实例（默认创建新实例）
            config: 审计配置（默认从 YAML 加载）
        """
        self.user_id = user_id
        self.config = config or load_audit_config()
        self.auditor = auditor or SafetyAuditor(self.config)
        self.dialog_engine = dialog_engine or DialogEngine(user_id)

        # 闭环日志历史
        self._log_history: list[SafetyLoopLogEntry] = []

    def process(
        self,
        user_input: str,
        llm_response: str,
        context: Optional[dict] = None,
    ) -> SafetyLoopResult:
        """执行安全闭环处理

        流程：
            1. 构建审计上下文
            2. 调用五轴审计器
            3. 全部通过 → 返回原始回复
            4. 有不通过 → 按严重程度分级处理：
               - SEVERE: 强制 CRISIS + escalate
               - MODERATE: 安全模板替换
               - MILD: 状态机重写
            5. 记录闭环日志
            6. 返回 SafetyLoopResult

        Args:
            user_input: 用户当前输入文本
            llm_response: LLM 生成的回复文本
            context: 审计上下文参数（可选），可包含：
                - dialog_state: str 当前对话状态
                - turn_count: int 当前轮次
                - crisis_turns: int 危机状态持续轮次
                - risk_level: str 风险等级
                - conversation_history: list[dict] 对话历史
                - llm_confidence: float LLM 置信度
                - current_topic: str 当前话题
                - is_teen: bool 是否青少年用户

        Returns:
            SafetyLoopResult: 闭环处理结果
        """
        start_time = time.time()
        ctx = context or {}

        # 记录处理前状态
        state_before = self.dialog_engine.current_state.value

        # ---- 步骤 1：构建审计上下文 ----
        audit_context = AuditContext(
            llm_output=llm_response,
            dialog_state=ctx.get("dialog_state", self.dialog_engine.current_state.value),
            turn_count=ctx.get("turn_count", self.dialog_engine.turn_count),
            crisis_turns=ctx.get("crisis_turns", 0),
            risk_level=ctx.get("risk_level", "low"),
            conversation_history=ctx.get("conversation_history", []),
            llm_confidence=ctx.get("llm_confidence", 0.5),
            current_topic=ctx.get("current_topic", ""),
            user_input=user_input,
            is_teen=ctx.get("is_teen", False),
            # 苏格拉底反问时机审计字段
            dialogue_mode=ctx.get(
                "dialogue_mode", self.dialog_engine.current_mode.value
            ),
            consecutive_calm_turns=ctx.get(
                "consecutive_calm_turns", self.dialog_engine.consecutive_calm_turns
            ),
            consecutive_avoidance_count=ctx.get(
                "consecutive_avoidance_count",
                self.dialog_engine.consecutive_avoidance_count,
            ),
            max_abs_z_score=ctx.get(
                "max_abs_z_score", self.dialog_engine.max_abs_z_score
            ),
        )

        # ---- 步骤 1.5：基线感知集成 ----
        # 计算当前用户的基线偏离，设置到审计器上下文。
        # 当个人基线显著偏离时，审计器自动提高对应轴的敏感度。
        # 临床依据：个体常态的异常检测比群体阈值更精准。
        baseline_deviation = None
        if _BASELINE_AVAILABLE and compute_deviation is not None:
            try:
                # 从 context 获取当前特征（用于计算偏离）
                current_features = ctx.get("current_features", {})
                if current_features:
                    baseline_deviation = compute_deviation(self.user_id, current_features)
                    if baseline_deviation and baseline_deviation.significant_deviations:
                        self.auditor.set_baseline_context(baseline_deviation)
                        logger.debug(
                            f"[SafetyLoop] 基线感知: user={self.user_id}, "
                            f"significant_deviations={baseline_deviation.significant_deviations}"
                        )
            except Exception as e:
                logger.warning(f"[SafetyLoop] 基线偏离计算失败: {e}")

        # ---- 步骤 2：调用五轴审计器 ----
        verdict = self.auditor.audit(audit_context)

        # ---- 步骤 3：全部通过 → 原样返回 ----
        if verdict.passed:
            result = SafetyLoopResult(
                final_response=llm_response,
                audit_verdict=verdict,
                severity="none",
                action_taken="pass_through",
                state_before=state_before,
                state_after=self.dialog_engine.current_state.value,
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )
            result.log_entry = self._create_log_entry(
                user_input, llm_response, result
            )
            self._log_history.append(result.log_entry)
            return result

        # ---- 步骤 4：有不通过 → 分级处理 ----
        # 收集不通过的轴及其严重程度
        failed_results = [r for r in verdict.results if not r.passed]
        max_severity = self._get_max_severity(failed_results)
        triggered_axes = [r.axis.value for r in failed_results]

        # 按严重程度处理
        if max_severity == Severity.SEVERE:
            result = self._handle_severe(
                user_input, llm_response, verdict, failed_results,
                state_before, start_time,
            )
        elif max_severity == Severity.MODERATE:
            result = self._handle_moderate(
                user_input, llm_response, verdict, failed_results,
                state_before, start_time,
            )
        else:  # MILD
            result = self._handle_mild(
                user_input, llm_response, verdict, failed_results,
                state_before, start_time,
            )

        # 记录闭环日志
        result.log_entry = self._create_log_entry(
            user_input, llm_response, result
        )
        self._log_history.append(result.log_entry)

        return result

    # ========================================================
    # 严重程度处理逻辑
    # ========================================================

    def _handle_severe(
        self,
        user_input: str,
        llm_response: str,
        verdict: AuditVerdict,
        failed_results: list[AuditResult],
        state_before: str,
        start_time: float,
    ) -> SafetyLoopResult:
        """重度处理：强制切换到 CRISIS 状态 + 调用 crisis.escalate()

        临床依据：
            NICE CG185 – 危机识别后首次接触即启动安全评估，
            延迟可能导致错过干预窗口。

        流程：
            1. 强制状态机切换到 CRISIS
            2. 构建 CrisisAlert
            3. 调用 escalate()
            4. 若 escalate 失败 → 降级为返回紧急资源卡
        """
        # 1. 强制切换到 CRISIS 状态
        risk_level = RiskLevel.CRISIS
        self.dialog_engine.current_state = DialogState.CRISIS
        self.dialog_engine.state_history.append(
            (self.dialog_engine.turn_count, DialogState.CRISIS)
        )
        state_after = DialogState.CRISIS.value

        # 2. 构建 CrisisAlert
        trigger_evidence = [
            r.reason for r in failed_results if r.reason
        ]
        alert = CrisisAlert(
            user_id=self.user_id,
            risk_score=0.9,  # 重度审计失败 → 高风险评分
            trigger_evidence=trigger_evidence,
            timestamp=time.time(),
            recommended_action="立即联系专业危机干预人员",
        )

        # 3. 调用 escalate()
        escalation_status = "none"
        escalation_alert_id = ""
        final_response = ""

        try:
            esc_result = crisis_escalate(alert)
            escalation_status = esc_result.status.value
            escalation_alert_id = esc_result.alert_id

            if esc_result.status == EscalationStatus.DISPATCHED:
                # 升级成功 → 使用危机回复模板
                hotline_info = self._get_hotline_text()
                final_response = _CRISIS_RESPONSE_TEMPLATE.format(
                    hotline_info=hotline_info
                )
                action_taken = "crisis_escalate"
            else:
                # 升级状态异常 → 降级处理
                final_response = self._fallback_resource_card()
                action_taken = "crisis_escalate_fallback"
                logger.warning(
                    "Crisis escalation status: %s, using fallback",
                    esc_result.status.value,
                )

        except Exception as e:
            # 4. escalate() 异常 → 降级方案：返回紧急资源卡
            escalation_status = "failed"
            final_response = self._fallback_resource_card()
            action_taken = "crisis_escalate_fallback"
            logger.error("Crisis escalation failed: %s", str(e))

        latency = round((time.time() - start_time) * 1000, 2)

        return SafetyLoopResult(
            final_response=final_response,
            audit_verdict=verdict,
            severity=Severity.SEVERE.value,
            action_taken=action_taken,
            triggered_axes=[r.axis.value for r in failed_results],
            escalation_status=escalation_status,
            escalation_alert_id=escalation_alert_id,
            state_before=state_before,
            state_after=state_after,
            latency_ms=latency,
        )

    def _handle_moderate(
        self,
        user_input: str,
        llm_response: str,
        verdict: AuditVerdict,
        failed_results: list[AuditResult],
        state_before: str,
        start_time: float,
    ) -> SafetyLoopResult:
        """中度处理：替换为安全模板 + 记录事件

        临床依据：
            WHO QualityRights (2019) – 去污名化表达要求；
            NICE CG178 – 青少年妄想需排除现实社交困扰。

        流程：
            1. 根据触发的轴选择安全模板
            2. 替换 LLM 回复
            3. 状态机推进一轮（保持当前状态）
        """
        # 选择安全模板（取第一个触发的中度轴）
        final_response = llm_response
        for r in failed_results:
            if r.axis in _SAFE_TEMPLATES:
                final_response = _SAFE_TEMPLATES[r.axis]
                break

        # 如果所有中度轴都没有专用模板，使用通用安全回复
        if final_response == llm_response:
            final_response = (
                "我听到你说的了。我想确认一下我理解对了——"
                "你愿意再多说一些吗？"
            )

        # 状态机推进一轮
        risk_level = self._parse_risk_level("medium")
        self.dialog_engine.next_turn(risk_level)
        state_after = self.dialog_engine.current_state.value

        latency = round((time.time() - start_time) * 1000, 2)

        return SafetyLoopResult(
            final_response=final_response,
            audit_verdict=verdict,
            severity=Severity.MODERATE.value,
            action_taken="safe_template_replace",
            triggered_axes=[r.axis.value for r in failed_results],
            state_before=state_before,
            state_after=state_after,
            latency_ms=latency,
        )

    def _handle_mild(
        self,
        user_input: str,
        llm_response: str,
        verdict: AuditVerdict,
        failed_results: list[AuditResult],
        state_before: str,
        start_time: float,
    ) -> SafetyLoopResult:
        """轻度处理：调用状态机重写回复策略

        临床依据：
            Miller & Rollnick (2013) 动机式访谈 – 保持真诚，
            避免谄媚破坏治疗联盟；
            Lambert (2013) – 保持治疗方向。

        特殊处理：
            socratic_timing 轴不通过时，根据 recommended_depth 分级降级：
            - depth=0: 强制切回 EMPATHY 模式 + 共情回复
            - depth=1: 保持 SOCRATIC 模式，但限制为 SHALLOW 深度
            - depth=2: 保持 SOCRATIC 模式，但限制为 MEDIUM 深度
            并记录审计日志。

        流程：
            1. 检查是否为 socratic_timing 轴不通过
            2. 是 → 根据 recommended_depth 分级降级
            3. 否 → 调用状态机重写回复策略
        """
        # 检查是否为 socratic_timing 轴不通过
        socratic_result = next(
            (r for r in failed_results if r.axis == AuditAxis.SOCRATIC_TIMING),
            None
        )

        if socratic_result is not None:
            recommended_depth = socratic_result.recommended_depth

            # 分级降级逻辑
            if recommended_depth == 0:
                # depth=0: 强制切回 EMPATHY 模式
                self.dialog_engine.current_mode = DialogueMode.EMPATHY
                self.dialog_engine.current_depth = SocraticDepth.SHALLOW
                self.dialog_engine.consecutive_exploration_turns = 0
                # 回避后深度冻结 3 轮
                self.dialog_engine.depth_frozen_turns = 3
                # 如果当前在 SOCRATIC_READY 状态，退回 EXPLORE
                if self.dialog_engine.current_state == DialogState.SOCRATIC_READY:
                    self.dialog_engine.current_state = DialogState.EXPLORE
                    self.dialog_engine.state_history.append(
                        (self.dialog_engine.turn_count, DialogState.EXPLORE)
                    )
                    logger.info(
                        "[SafetyLoop] socratic_timing 审计不通过 (depth=0)，"
                        "强制切回 EMPATHY 模式 + EXPLORE 状态，深度冻结 3 轮"
                    )

                final_response = _SOCRATIC_FALLBACK_TEMPLATE
                action_taken = "socratic_fallback_empathy"

            elif recommended_depth == 1:
                # depth=1: 保持 SOCRATIC 模式，但限制为 SHALLOW
                self.dialog_engine.current_depth = SocraticDepth.SHALLOW
                logger.info(
                    "[SafetyLoop] socratic_timing 审计不通过 (depth=1)，"
                    "降级到 SHALLOW 深度"
                )
                # 使用降级过渡语
                final_response = (
                    "没关系，我们慢慢来。"
                    "你刚才说的那个想法，能再多说一点吗？"
                )
                action_taken = "socratic_downgrade_shallow"

            else:  # recommended_depth == 2
                # depth=2: 保持 SOCRATIC 模式，但限制为 MEDIUM
                self.dialog_engine.current_depth = SocraticDepth.MEDIUM
                logger.info(
                    "[SafetyLoop] socratic_timing 审计不通过 (depth=2)，"
                    "降级到 MEDIUM 深度"
                )
                final_response = (
                    "我们换个角度想想。"
                    "有没有另一种方式来看待这件事？"
                )
                action_taken = "socratic_downgrade_medium"

            state_after = self.dialog_engine.current_state.value
            latency = round((time.time() - start_time) * 1000, 2)

            return SafetyLoopResult(
                final_response=final_response,
                audit_verdict=verdict,
                severity=Severity.MILD.value,
                action_taken=action_taken,
                triggered_axes=[r.axis.value for r in failed_results],
                state_before=state_before,
                state_after=state_after,
                latency_ms=latency,
            )

        # 非 socratic_timing 轴：原有逻辑
        # 获取当前状态对应的动作
        risk_level = self._parse_risk_level("medium")
        action = select_action(
            self.dialog_engine.current_state,
            self.dialog_engine.turn_count,
            risk_level,
        )

        # 使用策略生成器生成替代回复
        try:
            from intervention.strategy import generate_reply, Action as StrategyAction, ActionType as StrategyActionType

            # 将状态机的 ActionType 映射到策略模块的 ActionType
            action_type_map = {
                SMActionType.OPEN_QUESTION: StrategyActionType.OPEN_QUESTION,
                SMActionType.EMOTION_REFLECTION: StrategyActionType.EMOTION_REFLECTION,
                SMActionType.CBT_GUIDE: StrategyActionType.CBT_GUIDE,
                SMActionType.MINDFULNESS_GUIDE: StrategyActionType.MINDFULNESS_GUIDE,
                SMActionType.SAFETY_CHECK: StrategyActionType.SAFETY_CHECK,
                SMActionType.RESOURCE_PROVIDE: StrategyActionType.RESOURCE_PROVIDE,
                SMActionType.SUMMARY: StrategyActionType.SUMMARY,
            }
            strategy_action_type = action_type_map.get(
                action.action_type, StrategyActionType.EMOTION_REFLECTION
            )
            strategy_action = StrategyAction(
                action_type=strategy_action_type,
                risk_level="medium",
                dialog_state=self.dialog_engine.current_state.value,
            )
            strategy_result = generate_reply(strategy_action)
            final_response = strategy_result.reply
        except Exception:
            # 策略生成失败 → 使用通用重写模板
            final_response = (
                "谢谢你分享这些。我想换个方式来回应你——"
                "你所说的对我很重要，我想确保我的回答是真诚的。"
            )

        # 状态机推进一轮
        self.dialog_engine.next_turn(risk_level)
        state_after = self.dialog_engine.current_state.value

        latency = round((time.time() - start_time) * 1000, 2)

        return SafetyLoopResult(
            final_response=final_response,
            audit_verdict=verdict,
            severity=Severity.MILD.value,
            action_taken="strategy_rewrite",
            triggered_axes=[r.axis.value for r in failed_results],
            state_before=state_before,
            state_after=state_after,
            latency_ms=latency,
        )

    # ========================================================
    # 辅助方法
    # ========================================================

    def _get_max_severity(
        self, failed_results: list[AuditResult]
    ) -> Severity:
        """获取失败结果中的最高严重程度"""
        severity_order = {
            Severity.SEVERE: 3,
            Severity.MODERATE: 2,
            Severity.MILD: 1,
        }
        max_sev = Severity.MILD
        for r in failed_results:
            sev = _AXIS_SEVERITY.get(r.axis, Severity.MILD)
            if severity_order.get(sev, 0) > severity_order.get(max_sev, 0):
                max_sev = sev
        return max_sev

    def _get_hotline_text(self) -> str:
        """从配置中提取热线信息文本"""
        card_config = self.config.get("crisis_resource_card", {})
        hotlines = card_config.get("hotlines", [])
        if not hotlines:
            return "- 全国24小时心理援助热线: 400-161-9995"
        lines = [f"- {h['name']}: {h['number']}" for h in hotlines]
        return "\n".join(lines)

    def _fallback_resource_card(self) -> str:
        """降级方案：返回紧急资源卡

        当 crisis.escalate() 失败（如咨询师不可用）时，
        返回紧急资源卡作为最低安全保障。

        临床依据：
            NICE CG185 – 即使无法立即联系到专业人员，
            也必须向用户提供危机资源信息。
        """
        card_text = generate_crisis_resource_card(self.config)
        if card_text:
            return (
                "我注意到你可能正在经历很困难的时刻。"
                "你的安全对我来说是最重要的。\n\n"
                "如果你现在感到不安全，请立即联系专业帮助：\n"
                f"{card_text}\n\n"
                "我会一直在这里陪着你。"
            )
        return (
            "你的安全对我们很重要。如果你需要紧急帮助，"
            "请拨打全国24小时心理援助热线：400-161-9995"
        )

    def _parse_risk_level(self, level_str: str) -> RiskLevel:
        """将字符串解析为 RiskLevel 枚举"""
        try:
            return RiskLevel(level_str)
        except ValueError:
            return RiskLevel.LOW

    def _create_log_entry(
        self,
        user_input: str,
        original_response: str,
        result: SafetyLoopResult,
    ) -> SafetyLoopLogEntry:
        """创建闭环日志条目

        每次 process() 调用都会生成一条可追溯的日志记录，
        包含审计结果、动作类型、时间戳等完整信息。
        """
        return SafetyLoopLogEntry(
            log_id=f"sl_{uuid.uuid4().hex[:12]}",
            timestamp=time.time(),
            user_input=user_input[:100],  # 截断保护隐私
            original_response=original_response[:200],
            final_response=result.final_response[:200],
            audit_passed=result.severity == "none",
            triggered_axes=result.triggered_axes,
            severity=result.severity,
            action_taken=result.action_taken,
            escalation_status=result.escalation_status,
            escalation_alert_id=result.escalation_alert_id,
            state_before=result.state_before,
            state_after=result.state_after,
            latency_ms=result.latency_ms,
        )

    def get_log_history(self) -> list[SafetyLoopLogEntry]:
        """获取闭环日志历史"""
        return list(self._log_history)

    def get_state_summary(self) -> dict:
        """获取当前状态摘要"""
        return {
            "user_id": self.user_id,
            "dialog_state": self.dialog_engine.current_state.value,
            "dialogue_mode": self.dialog_engine.current_mode.value,
            "turn_count": self.dialog_engine.turn_count,
            "consecutive_calm_turns": self.dialog_engine.consecutive_calm_turns,
            "consecutive_avoidance_count": self.dialog_engine.consecutive_avoidance_count,
            "total_log_entries": len(self._log_history),
            "last_severity": (
                self._log_history[-1].severity if self._log_history else "none"
            ),
        }


# ============================================================
# Mermaid 闭环流程图
# ============================================================

MERMAID_SAFETY_LOOP_DIAGRAM = """
```mermaid
flowchart TD
    A[用户输入 + LLM 回复] --> B[auditor.audit]
    B --> C{五轴全部通过?}

    C -->|是| D[返回原始回复]
    C -->|否| E{判断最高严重程度}

    E -->|SEVERE 重度| F[强制切换 CRISIS 状态]
    F --> G[构建 CrisisAlert]
    G --> H[crisis.escalate]
    H --> I{升级成功?}
    I -->|成功| J[返回危机回复 + 资源卡]
    I -->|失败| K[降级: 返回紧急资源卡]

    E -->|MODERATE 中度| L[选择安全模板]
    L --> M[替换 LLM 回复]
    M --> N[状态机推进 + 记录事件]

    E -->|MILD 轻度| O[状态机选择替代动作]
    O --> P[策略生成器重写回复]
    P --> Q[状态机推进]

    J --> R[记录闭环日志]
    K --> R
    N --> R
    Q --> R
    D --> R

    R --> S[返回 SafetyLoopResult]

    style F fill:#ff4444,color:#fff
    style L fill:#ffaa00,color:#000
    style O fill:#44aaff,color:#fff
    style J fill:#ff6666,color:#fff
    style K fill:#ffcc00,color:#000
```

严重程度分级依据：
- SEVERE（重度）：直接威胁用户安全 → 危机升级延迟轴
  临床依据: NICE CG185 – 危机安全优先原则
- MODERATE（中度）：可能造成心理伤害 → 污名化/妄想强化轴
  临床依据: WHO QualityRights (2019) – 去污名化
- MILD（轻度）：影响治疗效果 → 谄媚/轨迹漂移轴
  临床依据: Miller & Rollnick (2013) – 动机式访谈真诚原则
"""


# ============================================================
# 便捷函数
# ============================================================

def process_safety_loop(
    user_input: str,
    llm_response: str,
    user_id: str = "anonymous",
    context: Optional[dict] = None,
) -> SafetyLoopResult:
    """安全闭环处理便捷函数

    Args:
        user_input: 用户输入
        llm_response: LLM 回复
        user_id: 用户 ID
        context: 审计上下文参数

    Returns:
        SafetyLoopResult: 闭环处理结果
    """
    loop = SafetyLoop(user_id=user_id)
    return loop.process(user_input, llm_response, context)
