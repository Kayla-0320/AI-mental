"""
干预层路由 —— AI 心理干预 API 端点

包含：
    1. 安全审计事件查询（咨询师端实时审计面板）
    2. 结构化策略生成（Action → 回复文本）
    3. CBT 认知重构会话管理
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import random
import requests
from dataclasses import dataclass
from typing import Any, Iterator, Optional
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from intervention.strategy import generate_reply, get_generator, Action, ActionType
from intervention.auditor import SafetyAuditor, AuditContext, load_audit_config
from intervention.safety_loop import SafetyLoop, SafetyLoopResult
from intervention.cbt.cbt_record import (
    get_cbt_record,
    get_user_cbt_records,
    CBTRestructuringRecord,
)
from intervention.style_detector import StyleDetector, UserStyle, get_reply_style_params
from perception.perception_service import get_perception_service
from perception.cognitive.distortion_classifier import (
    format_distortion_hint,
    predict_distortion_safe,
)
from assessment.scale import map_all
from assessment.user_memory import get_memory_store
from assessment.memory_extractor import MemoryExtractor, build_memory_context
from intervention.scene_retrieval import build_scene_context
from shared.dataclasses import ChatStreamEventType

router = APIRouter(prefix="/intervention", tags=["干预层"])

logger = logging.getLogger(__name__)


# ============================================================
# 内存审计事件存储（无真实数据库时的替代方案）
# ============================================================

class InMemoryAuditStore:
    """内存审计事件存储

    生产环境应替换为数据库持久化存储。
    当前使用内存列表 + 自动生成模拟事件来演示功能。
    """

    def __init__(self):
        self._events: list[dict] = []
        self._seed_events()

    def _seed_events(self):
        """生成初始模拟审计事件"""
        base_time = int(time.time() * 1000)
        self._events = [
            {
                "id": 1, "type": "safety_intercept",
                "time": time.strftime("%H:%M:%S", time.localtime(base_time / 1000 - 3600)),
                "severity": "warning",
                "description": "AI 回复被拦截：检测到潜在谄媚倾向（轴4：同意率 > 90%）",
                "action": "已自动替换为中性回复",
                "axis": "sycophancy",
            },
            {
                "id": 2, "type": "crisis_signal",
                "time": time.strftime("%H:%M:%S", time.localtime(base_time / 1000 - 1800)),
                "severity": "info",
                "description": "文本分析检测到轻微消极情绪词汇（C-SSRS 标记）",
                "action": "已记录，未触发升级",
                "axis": "crisis_detection",
            },
            {
                "id": 3, "type": "audit_pass",
                "time": time.strftime("%H:%M:%S", time.localtime(base_time / 1000 - 600)),
                "severity": "success",
                "description": "AI 回复通过五轴安全审计（危机延迟✓ 妄想强化✓ 污名化✓ 谄媚✓ 漂移✓）",
                "action": "已正常发送",
                "axis": "all_pass",
            },
            {
                "id": 4, "type": "safety_intercept",
                "time": time.strftime("%H:%M:%S", time.localtime(base_time / 1000 - 300)),
                "severity": "warning",
                "description": "AI 回复被拦截：检测到潜在诊断性标签（轴3：污名化用语）",
                "action": "已移除诊断性标签后重新发送",
                "axis": "stigma",
            },
            {
                "id": 5, "type": "crisis_signal",
                "time": time.strftime("%H:%M:%S", time.localtime(base_time / 1000 - 120)),
                "severity": "info",
                "description": "情绪突变检测：悲伤概率从 0.2 跳升至 0.7",
                "action": "已标记，持续监控中",
                "axis": "emotion_shift",
            },
        ]

    def get_events(self, limit: int = 50) -> list[dict]:
        """获取审计事件（最新在前）"""
        return list(reversed(self._events[-limit:]))

    def add_event(self, event: dict):
        """添加审计事件"""
        event["id"] = len(self._events) + 1
        event["time"] = time.strftime("%H:%M:%S", time.localtime())
        self._events.append(event)

    def get_stats(self) -> dict:
        """获取审计统计"""
        total = len(self._events)
        intercepts = sum(1 for e in self._events if e["type"] == "safety_intercept")
        crisis_signals = sum(1 for e in self._events if e["type"] == "crisis_signal")
        passes = sum(1 for e in self._events if e["type"] == "audit_pass")
        return {
            "total": total,
            "intercepts": intercepts,
            "crisis_signals": crisis_signals,
            "passes": passes,
            "intercept_rate": round(intercepts / total, 4) if total > 0 else 0,
        }


# 全局审计存储实例
_audit_store = InMemoryAuditStore()


# ============================================================
# 请求/响应模型
# ============================================================

class StrategyRequest(BaseModel):
    """策略生成请求"""
    action_type: str = Field(..., description="动作类型")
    emotion: str = Field("", description="当前情绪")
    risk_level: str = Field("low", description="风险等级")
    dialog_state: str = Field("EXPLORE", description="对话状态")
    topic: str = Field("", description="当前话题")
    params: dict = Field(default_factory=dict, description="额外参数")


class StrategyResponse(BaseModel):
    """策略生成响应"""
    reply: str
    action_type: str
    confidence: float
    safety_checked: bool
    word_count: int
    latency_ms: float


class AuditEventResponse(BaseModel):
    """审计事件响应"""
    id: int
    type: str
    time: str
    severity: str
    description: str
    action: str


# ============================================================
# 路由端点
# ============================================================

@router.get("/audit-events", summary="获取安全审计事件列表")
async def get_audit_events(limit: int = 50):
    """获取安全审计事件列表（咨询师端审计面板数据源）

    返回最近的安全审计事件，包括：
    - safety_intercept: AI 回复被拦截
    - crisis_signal: 危机信号检测
    - audit_pass: 审计通过
    """
    events = _audit_store.get_events(limit)
    stats = _audit_store.get_stats()
    return {"events": events, "stats": stats}


@router.post("/audit-event", summary="添加审计事件")
async def add_audit_event(event: dict):
    """添加一条审计事件（由安全审计中间件调用）"""
    _audit_store.add_event(event)
    return {"status": "ok", "id": len(_audit_store._events)}


@router.post("/strategy", response_model=StrategyResponse, summary="结构化策略生成")
async def generate_strategy(request: StrategyRequest) -> StrategyResponse:
    """将结构化 Action 转换为具体回复文本

    流程：Action → 模板选择 → 参数填充 → 青少年语言适配 → 安全红线检查 → 回复
    """
    try:
        action_type = ActionType(request.action_type)
    except ValueError:
        return StrategyResponse(
            reply="我听到你说的了。能再多告诉我一些吗？",
            action_type=request.action_type,
            confidence=0.3,
            safety_checked=True,
            word_count=16,
            latency_ms=0,
        )

    action = Action(
        action_type=action_type,
        emotion=request.emotion,
        risk_level=request.risk_level,
        dialog_state=request.dialog_state,
        topic=request.topic,
        params=request.params,
    )

    result = generate_reply(action)

    return StrategyResponse(
        reply=result.reply,
        action_type=result.action_type,
        confidence=result.confidence,
        safety_checked=result.safety_checked,
        word_count=result.word_count,
        latency_ms=result.latency_ms,
    )


@router.get("/strategy/stats", summary="策略生成统计")
async def strategy_stats():
    """获取策略生成器运行统计"""
    return get_generator().get_stats()


# ============================================================
# 安全闭环协调器
# ============================================================

# 全局 SafetyLoop 实例池（按 user_id 缓存）
_safety_loops: dict[str, SafetyLoop] = {}


class SafetyLoopRequest(BaseModel):
    """安全闭环处理请求"""
    user_input: str = Field(..., description="用户输入文本")
    llm_response: str = Field(..., description="LLM 生成的回复文本")
    user_id: str = Field("anonymous", description="用户 ID")
    dialog_state: str = Field("INIT", description="当前对话状态")
    turn_count: int = Field(0, description="当前轮次")
    crisis_turns: int = Field(0, description="危机状态持续轮次")
    risk_level: str = Field("low", description="风险等级")
    conversation_history: list[dict] = Field(
        default_factory=list, description="对话历史"
    )
    llm_confidence: float = Field(0.5, description="LLM 置信度")
    current_topic: str = Field("", description="当前话题")
    is_teen: bool = Field(False, description="是否青少年用户")


class SafetyLoopResponse(BaseModel):
    """安全闭环处理响应"""
    final_response: str
    severity: str
    action_taken: str
    audit_passed: bool
    triggered_axes: list[str]
    escalation_status: str
    escalation_alert_id: str
    state_before: str
    state_after: str
    latency_ms: float
    audit_details: list[dict]


@router.post(
    "/safety-loop",
    response_model=SafetyLoopResponse,
    summary="安全闭环处理",
)
async def safety_loop_process(request: SafetyLoopRequest) -> SafetyLoopResponse:
    """安全闭环处理 —— 串联「审计 → 状态机 → 升级」三模块

    流程：
        1. 五轴安全审计
        2. 全部通过 → 返回原始回复
        3. 有不通过 → 按严重程度分级处理：
           - SEVERE: 强制 CRISIS + 升级至人工
           - MODERATE: 安全模板替换
           - MILD: 状态机重写策略
    """
    # 获取或创建 SafetyLoop 实例
    if request.user_id not in _safety_loops:
        _safety_loops[request.user_id] = SafetyLoop(user_id=request.user_id)
    loop = _safety_loops[request.user_id]

    # 构建上下文
    context = {
        "dialog_state": request.dialog_state,
        "turn_count": request.turn_count,
        "crisis_turns": request.crisis_turns,
        "risk_level": request.risk_level,
        "conversation_history": request.conversation_history,
        "llm_confidence": request.llm_confidence,
        "current_topic": request.current_topic,
        "is_teen": request.is_teen,
    }

    # 执行闭环处理
    result = loop.process(
        user_input=request.user_input,
        llm_response=request.llm_response,
        context=context,
    )

    # 构建审计详情
    audit_details = []
    if result.audit_verdict:
        for r in result.audit_verdict.results:
            audit_details.append({
                "axis": r.axis.value,
                "passed": r.passed,
                "reason": r.reason,
                "suggested_action": r.suggested_action.value,
            })

    return SafetyLoopResponse(
        final_response=result.final_response,
        severity=result.severity,
        action_taken=result.action_taken,
        audit_passed=result.severity == "none",
        triggered_axes=result.triggered_axes,
        escalation_status=result.escalation_status,
        escalation_alert_id=result.escalation_alert_id,
        state_before=result.state_before,
        state_after=result.state_after,
        latency_ms=result.latency_ms,
        audit_details=audit_details,
    )


@router.get("/safety-loop/log", summary="获取闭环日志")
async def safety_loop_log(user_id: str = "anonymous"):
    """获取指定用户的安全闭环处理日志"""
    if user_id in _safety_loops:
        loop = _safety_loops[user_id]
        history = loop.get_log_history()
        return {
            "user_id": user_id,
            "total_entries": len(history),
            "entries": [
                {
                    "log_id": e.log_id,
                    "timestamp": e.timestamp,
                    "severity": e.severity,
                    "action_taken": e.action_taken,
                    "triggered_axes": e.triggered_axes,
                    "escalation_status": e.escalation_status,
                    "state_before": e.state_before,
                    "state_after": e.state_after,
                    "latency_ms": e.latency_ms,
                }
                for e in history
            ],
        }
    return {"user_id": user_id, "total_entries": 0, "entries": []}


@router.get("/safety-loop/state", summary="获取闭环状态")
async def safety_loop_state(user_id: str = "anonymous"):
    """获取指定用户的安全闭环当前状态"""
    if user_id in _safety_loops:
        return _safety_loops[user_id].get_state_summary()
    return {
        "user_id": user_id,
        "dialog_state": "INIT",
        "turn_count": 0,
        "total_log_entries": 0,
        "last_severity": "none",
    }


# ============================================================
# 智能对话闭环端点（供 Node.js 桥接层调用）
# ============================================================

class SmartChatRequest(BaseModel):
    """智能对话请求（对应 algorithm-bridge.ts SmartChatRequest）"""
    user_id: str = Field(..., description="用户 ID")
    message: str = Field(..., description="用户输入文本")
    conversation_history: list[dict] = Field(
        default_factory=list, description="对话历史 [{role, content}]"
    )
    session_id: Optional[str] = Field(None, description="会话 ID")


class SmartChatResponse(BaseModel):
    """智能对话响应（对应 algorithm-bridge.ts SmartChatResponse）"""
    reply: str
    dialog_state: str
    dialogue_mode: str = "EMPATHY"  # 对话模式：EMPATHY（共情）/ SOCRATIC（引导）
    action_type: str
    risk_level: str
    audit_passed: bool
    requires_escalation: bool
    emotion_probs: list[float]
    evidence: list[str]
    fallback: bool = False
    # --- C2D2 两根轴的可见化 ---
    # 供咨询师端与审计链路使用；**不是给患者的诊断结论**，见
    # perception/cognitive/distortion_classifier.py 的诚实边界说明。
    scene_retrieval: Optional[dict] = None
    cognitive_distortion: Optional[dict] = None


# 简易共情回复生成器（不依赖外部 LLM 服务）
#
# ⚠️ 降级路径也必须遵守同一件事：**开场不许复述情绪**。
# 这一池原先清一色是「我能感受到你的 X / 我理解你的感受」+ 提问，
# 与主提示词里被禁掉的那个句式是同一个 —— 降级时用户会突然又开始收到套路回复。
# 改后统一为「落到具体内容 / 说句实话」开场。
_EMPATHY_TEMPLATES = {
    "low": [
        "我听到你说的了。愿意再多讲一点吗？",
        "谢谢你愿意跟我说这些。最近还有别的事让你挂心吗？",
        "嗯，我在听。这件事最让你放不下的是哪一块？",
    ],
    "medium": [
        "听起来这段时间你扛了不少东西。是什么让你最不好受？",
        "这些感受都是真的，你不用一个人扛着。想从哪儿说起都行。",
        "你正在经历的事确实不容易。后来怎么样了？",
    ],
    "high": [
        "你说的这些我都听到了，你的感受很重要。你现在安全吗？",
        "谢谢你告诉我这些。你安全吗？有没有想过伤害自己？",
        "你现在经历的这些一定很困难。我想让你知道，有专业的人员可以帮助你。",
    ],
    "crisis": {
        # C-SSRS 阶梯式响应：根据危机持续轮次调整响应策略
        # Turn 1: 接纳 + 探索时间线（不急于确认安全，先让用户感到被听见）
        1: [
            "谢谢你愿意告诉我这些，说出来需要很大的勇气。这种感觉是从什么时候开始的？",
            "我能感受到你现在很痛苦。能跟我说说，最近发生了什么让你有这种感觉吗？",
            "听到你这么说我很担心。这种感觉持续多久了？",
        ],
        # Turn 2: 评估严重程度（直接询问自伤/自杀想法）
        2: [
            "我想直接问你，你有没有想过伤害自己或者不想活了？",
            "你提到'活着没意思'，你有没有具体想过要怎么做？",
            "这种想法出现的频率高吗？是偶尔闪过，还是一直存在？",
        ],
        # Turn 3+: 提供资源 + 安全计划
        3: [
            "你的安全非常重要。24小时心理援助热线：400-161-9995，他们可以提供专业帮助。",
            "请现在就联系一个你信任的人（家人、朋友或老师），告诉他们你的感受。",
            "如果你现在感觉无法控制自己，请立即拨打120或去最近的医院急诊。",
        ],
    },
}


# 风格化回复模板（根据用户表达风格调整回复节奏）
_STYLE_REPLY_TEMPLATES = {
    UserStyle.INTROVERTED: [
        "我在这里陪着你。",
        "慢慢来，不着急。",
        "嗯，我理解。",
        "想说的时候随时说。",
        "我在听。",
    ],
    UserStyle.EXTROVERTED: [
        "听起来你经历了很多！能再多说说吗？",
        "我感觉你有很多想法，一起聊聊吧！你觉得呢？",
        "你说的很有意思！后来发生了什么？",
        "谢谢你分享这些！你当时是什么感受？",
        "我听到你说的了。如果换一种方式，你会怎么做？",
    ],
    UserStyle.AGITATED: [
        "我在这里。",
        "你的感受很重要。",
        "我陪着你。",
        "深呼吸，慢慢来。",
        "你不需要独自承受。",
    ],
    UserStyle.CALM: [
        "我理解你的感受。能再多说说吗？",
        "谢谢你分享这些。你觉得是什么原因让你有这种感觉？",
        "我听到你说的了。如果换个角度看，会怎样？",
        "你愿意聊聊更多吗？",
    ],
}

# 通用陪伴话术（**全部不含问句**）。
#
# 用途：ask_question=False（用户拒绝深挖 / 情绪激动 / 高风险）时的降级回复。
# 不能复用 _STYLE_REPLY_TEMPLATES[CALM] —— 那一池本身就含问句
# （如"如果换个角度看，会怎样？"），会在"本轮不要提问"时自相矛盾。
_COMPANIONSHIP_TEMPLATES = [
    "我在这里，你不用一个人扛着。",
    "嗯，我在听着呢。",
    "你的感受很重要，不用急着把它赶走。",
    "慢慢来，不着急。",
    "我陪着你。",
    "你不需要独自承受这些。",
]


# 文本情绪概率的标签顺序，与 perception.TextEmotionResult 一致：
# [快乐, 悲伤, 焦虑, 愤怒, 中性]
_NEGATIVE_EMOTION_INDICES = (1, 2, 3)


def _negative_emotion_intensity(emotion_probs: list[float]) -> float:
    """负性情绪强度（0-1）。

    取负面三类（悲伤/焦虑/愤怒）概率之和。刻意**不用** ``max(probs)``：
    后者是分类器置信度，说「你好」时中性概率可达 0.80，会被
    ``StyleDetector`` 当成"高情绪强度"，把寒暄误判为 AGITATED。

    Args:
        emotion_probs: ``[快乐, 悲伤, 焦虑, 愤怒, 中性]`` 概率。

    Returns:
        float: 负面情绪概率之和，截断到 [0, 1]；输入异常时为 0.0。
    """
    try:
        total = sum(
            float(emotion_probs[i])
            for i in _NEGATIVE_EMOTION_INDICES
            if i < len(emotion_probs)
        )
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, total))


def _is_avoidance_or_refusal(text: str) -> bool:
    """用户是否明确表示不想继续谈（对应提示词"往下走"一节的留白要求）。

    与 ``state_machine._AVOIDANCE_KEYWORDS`` 同源，但刻意多覆盖
    "这点小事""没事"这类**主动压下情绪**的说法 —— 它们同样意味着
    "现在别追"。此处只用于本轮是否追问，不参与任何风险判定。

    Args:
        text: 用户当前发言。

    Returns:
        bool: True 表示应当停止追问、改为尊重节奏。
    """
    if not text:
        return False
    return any(kw in text for kw in _REFUSAL_KEYWORDS)


# 拒绝深挖 / 主动压下情绪的表达（"往下走"一节里"不再追"的判定依据）
_REFUSAL_KEYWORDS = (
    "不想说", "不想聊", "不想谈", "别问了", "不要问", "别追问",
    "换个话题", "换一个话题", "跳过", "算了", "没想法", "不知道怎么说",
    "不想回答", "够了", "别管我", "没事", "没什么", "这点小事",
    "不想被问", "不想继续", "不想讲", "不聊了", "别烦我",
)

# 情绪激动（而非单纯难过）的表达。用于区分
#   "我好难受"（高负性、零感叹号 → 应温柔追问一句）
#   "烦死了！！！"（高负性 + 激动表达 → 先安抚、不追问）
_AGITATION_KEYWORDS = (
    "烦死", "受不了", "崩溃", "疯了", "讨厌", "气死",
    "滚", "闭嘴", "凭什么", "不公平",
)


def _ask_question_this_turn(
    style: Optional[UserStyle],
    risk_level: str,
    user_message: str = "",
    negative_intensity: float = 0.0,
) -> bool:
    """本轮是否可以提出**一个**开放式追问。

    规则来源：提示词的"底线"（尊重节奏）与"往下走"一节（至少隔一轮才追问）。
    注意这里只判断**本轮能不能问**；"连着两轮都问过"这层由
    :func:`_assistant_asked_repeatedly` 单独判定，两者在
    :func:`_build_llm_messages` 与 :func:`_finalize_reply` 里合流。

    实现上刻意**不**直接使用 ``UserStyle.AGITATED`` 标签：该标签的判据是
    "高情绪强度 + 短消息/感叹号"，实测会把 "我好难受"（负性 0.989、
    零感叹号）也判成 AGITATED，而"难受"恰恰是草案里最该被温柔追问一句的
    表达（草案模板 1：难受 → "这种难受是什么样的感觉？"）。
    因此这里按**是否有激动表达**判定：

        - 高负性 + 感叹号/⌈≤6 字且含激动或回避措辞⌉ → 先安抚，不追问；
        - 高负性但单纯难过（"我好难受"）→ 允许追问一句；
        - 明确表示不想谈 → 立刻停止深挖（"往下走"一节的留白要求）；
        - 寒暄/短消息但平静 → 允许邀请式追问，否则模型会退化成
          只会说"我在这里"的复读机。

    高风险/危机轮次一律不追问：此时首要动作是安全确认与线下转介，
    模板里那句安全确认问句由 ``_generate_empathy_reply`` 单独负责。

    Args:
        style: 用户表达风格；仅用于日志与兼容，不参与放行判定。
        risk_level: 本轮风险等级。
        user_message: 用户当前发言（识别"不愿多谈"与激动表达）。
        negative_intensity: 负面情绪强度（0-1），见 :func:`_negative_emotion_intensity`。

    Returns:
        bool: True 表示本轮允许提出一个开放式追问。
    """
    # 危机/高风险：只做安全确认与转介，不做认知探索
    if risk_level in ("high", "crisis"):
        return False

    # 明确表示不想谈 —— 立刻停止深挖（"往下走"一节的留白要求）
    if _is_avoidance_or_refusal(user_message):
        return False

    # 激动：高负性 + 感叹号；或极短且带激动措辞
    exclamation = user_message.count("！") + user_message.count("!")
    if negative_intensity >= 0.7:
        if exclamation > 0:
            return False
        if len(user_message.strip()) <= 6 and any(
            kw in user_message for kw in _AGITATION_KEYWORDS
        ):
            return False

    return True


# ============================================================
# 大模型调用：地址、模型名、消息装配
# ============================================================

# 基址与模型名走环境变量，便于把整条链路指向本地/自建 OpenAI 兼容服务做验证，
# 不改代码即可切换。默认值与本次改造前硬编码的值完全一致。
_LLM_BASE_URL = os.environ.get(
    'DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'
)
_LLM_MODEL = os.environ.get('DASHSCOPE_MODEL', 'qwen-turbo')
_LLM_TIMEOUT = 15


def _sampling_params(risk_level: str) -> dict:
    """按风险等级给采样参数 —— 日常共情求"多变鲜活"，危机求"稳准不出格"。

    - 高风险 / 危机：低温(0.7)，不加 frequency_penalty。此时要的是措辞稳妥、
      安全确认到位，宁可平实也不要"活"到跑偏。
    - 低 / 中风险（日常）：高温(0.95) + frequency_penalty=0.3，抑制同一会话内
      反复出现同样的词和句式，从采样层面削弱"复读机"感。
    """
    if risk_level in ("high", "crisis"):
        return {"temperature": 0.7}
    return {"temperature": 0.95, "frequency_penalty": 0.3}

# 连续追问的容忍轮数：允许连着两轮追问，只拦"连着三轮"。
#
# 取值经过两轮用户反馈校准：
#   2026-09-26 第一版设 2（一轮不追问就要插一轮留白）→ 用户反馈"追问的环节也没有了"；
#   改成 3 之后，对话节奏是「问、问、接一句」，既不像审问，也不会让用户
#   觉得你只是在附和不往下走。
# ⚠️ 这个常量同时被 prompt 侧（写不写"本轮不要提问"）和定稿侧（删不删结尾问句）
# 使用 —— 两处必须同源，否则会出现"prompt 说不许问、代码却照问"的静默分裂。
_QUESTION_RUN_LENGTH = 3


def _question_count(text: str) -> int:
    """一段回复里的问句数（中英文问号都算）。

    只看问号是刻意为之：追问失败的典型形态不是"没问号"，
    而是**每轮都恰好挂一个问号**，把"共情+提问"变成固定节拍。

    Args:
        text: 一段回复文本。

    Returns:
        int: 问号总数。
    """
    return text.count("？") + text.count("?")


def _assistant_asked_repeatedly(
    conversation_history: list,
    run_length: int = _QUESTION_RUN_LENGTH,
) -> bool:
    """最近 ``run_length`` 条 AI 回复是否**都**追问了。

    用途：连着追问是"套路感"的来源之一（用户 2026-09-26 第一张截图里
    4 轮全在问），但**拦得太狠会走向另一个极端** —— 第二版限制成
    "隔一轮才能问一次"后，用户反馈"追问的环节也没有了"。
    所以现在只拦连着三轮，具体取值见 :data:`_QUESTION_RUN_LENGTH`。

    只统计 ``role == "assistant"`` 的条目，且只回看对话尾部；
    历史里没有 AI 发言（首轮）时返回 False。

    Args:
        conversation_history: 对话历史 ``[{role, content}]``。
        run_length: 判定为"连续追问"所需的最小连续轮数，默认 2。

    Returns:
        bool: True 表示本轮应当避免再提问。
    """
    recent = [
        str(msg.get("content", ""))
        for msg in conversation_history
        if msg.get("role") == "assistant"
    ][-run_length:]
    if len(recent) < run_length:
        return False
    return all(_question_count(text) > 0 for text in recent)


# 把提问式表达改写成反思式表达。左列是问句中的"提问部件"，
# 右列是替换后的说法 —— 替换完句子自然变成陈述句。
#
# ⚠️ 只在**整段回复就是一个问句**时才会用到（见 _strip_trailing_question）：
# 那种情况删掉问句就什么都不剩，只能改写。刻意用固定替换而不是随机生成，
# 是为了让"本轮禁止提问"成为可测的不变量。
# 顺序即优先级：**先匹配具体的"提问部件"**（是什么感觉 / 什么滋味），
# 再匹配前半句的"你当时 / 这时候你"。反过来的话
# 「那你当时是什么感觉？」会先命中 "你当时"，被改成
# 「我猜你当时是什么感觉。」—— 问号没了，但还是个问题。
_QUESTION_TO_REFLECTION = (
    ("发生了什么事", "应该是发生了什么"),
    ("是什么感觉", "一定很不好受"),
    ("是什么滋味", "肯定不是滋味"),
    ("什么滋味", "肯定不是滋味"),
    ("是什么情况", "大概不好受"),
    ("是什么事", "一定有原因"),
    ("这时候你", "这时候你心里肯定"),
    ("那时候你", "我猜那时候的你"),
    ("你当时", "我猜是你"),
    ("你现在", "我想你大概是"),
    ("你心里", "我心里猜你"),
)


def _question_to_reflection(text: str) -> str:
    """把单个问句改写成陈述句（无法改写时返回空串）。

    只处理"整段就是一个问句"的兜底场景，因此不做通用句式分析，
    只按 :data:`_QUESTION_TO_REFLECTION` 做定向替换。

    Args:
        text: 一个以问号结尾的句子。

    Returns:
        str: 改写后的陈述句；没有可用的替换项时返回空串（调用方保留原句）。
    """
    body = text.rstrip().rstrip("？?").rstrip()
    for src, dst in _QUESTION_TO_REFLECTION:
        if src in body:
            return body.replace(src, dst, 1) + "。"
    return ""


# 把"吗/吧"类问句改写成陈述句用的替换表。
#
# 这类问句**没有 wh-词**（不是"什么/怎么/为什么"），所以没法用
# _QUESTION_TO_REFLECTION 那套"提问部件"替换；只能按句式对调。
# 覆盖实测里出现过的说法，够用即可；没命中的一律不改写（宁可留问句也不改坏）。
_DECLARATIVE_FROM_QUESTION = (
    ("是不是", "我猜是"),
    ("是不是有点", "我猜有点"),
    ("会不会", "我猜会"),
    ("有没有", "我想有"),
    ("在不在", "我想你在"),
    ("真的吗", "我有点不敢信"),
    ("是吧", "我猜是"),
    ("对吗", "我猜是"),
    ("好吗", "我明白"),
    ("行吗", "我明白"),
    ("还是", "，"),
    ("还是说", "，"),
    ("还是你", "，你"),
    ("呢吗", "呢"),
)


def _declarative_from_question(clause: str) -> str:
    """把"吗/吧"类问句改写成陈述句。

    两级处理：
        1. 先按 :data:`_DECLARATIVE_FROM_QUESTION` 做定向替换（更自然）；
        2. 都没命中时，删掉句尾的"吗/吧"并补句号 —— 这是兜底，
           只在剩下的内容还够长（≥6 字）时才用，否则读起来像半截话
           （「你信吗？」→「你信。」不该出现）。

    Args:
        clause: 一个以问号结尾的分句。

    Returns:
        str: 改写后的陈述句；无可用处理时返回空串。
    """
    body = clause.rstrip().rstrip("？?").rstrip()
    if not body:
        return ""
    for src, dst in _DECLARATIVE_FROM_QUESTION:
        if src in body:
            return body.replace(src, dst, 1).rstrip("，") + "。"
    if body.endswith(("吗", "吧")):
        # 「…，你信吗？」这类**附加问句**：问句只是挂在前面那句话的尾巴上，
        # 该丢的是整条尾巴（"你信吗"），不是单删一个"吗"字。
        if "，" in body:
            main = body.rsplit("，", 1)[0].rstrip()
            # 前截只是"…的时候 / …的话"这种状语框架时不能留，会成半截话
            if len(main) >= 4 and not main.endswith(
                ("的时候", "的话", "之后", "之前", "时", "以后")
            ):
                return main + "。"
        trimmed = body[:-1]
        if len(trimmed) >= 6:
            return trimmed + "。"
    return ""


# 判断"逗号前那一截"是不是一句**说完的事**，用来决定能不能丢掉后面的问句分句。
# 只认"完句标记"：了/没/过/已经 这类完成标记，或时间词。
# 例：
#   「你妈说要不是你，他们早就离了」有"了" → 是说完的事，可以只留这一截；
#   「老师骂你的时候，是在教室里吗」前截只是时间状语 → 不能砍，砍了成半截话。
_COMPLETION_MARKERS = (
    "了", "没", "过", "已经", "曾经",
    "昨天", "今天", "刚才", "以前", "后来", "当时", "那天", "这时候", "那时候",
)


def _strip_trailing_clause_question(text: str) -> tuple[str, bool]:
    """丢掉"已经说完的陈述 + 逗号 + 结尾问句"里的最后那个问句分句。

    只在逗号前那一截**自己就是一句说完的话**时才动手（判定见
    :data:`_COMPLETION_MARKERS`），且前截带"…的时候 / …的话"这类
    状语尾巴时一律不动 —— 那种前截离开后半个分句就讲不通了。

    Args:
        text: 模型回复。

    Returns:
        tuple[str, bool]: ``(处理后的文本, 是否发生了改动)``。
    """
    if "，" not in text:
        return text, False
    pieces = [p for p in text.split("，") if p.strip()]
    if len(pieces) < 2:
        return text, False
    if not pieces[-1].rstrip().endswith(("？", "?")):
        return text, False
    head = "，".join(pieces[:-1]).rstrip()
    if len(head) < 5:
        return text, False
    # 逗号前最后那一小截自己也得有分量：「…这话是他说的」后面跟着「你信吗」时，
    # 砍掉问句会剩下半截话；有 4 个字以上才算一个能独立站住的分句。
    if len(pieces[-2].strip()) < 4:
        return text, False
    if head.endswith(("的时候", "的话", "之后", "之前", "时")):
        return text, False
    if not any(m in head for m in _COMPLETION_MARKERS):
        return text, False
    if not head.endswith(("。", "！", "!", "…")):
        head += "。"
    return head, True


def _join_sentences(parts: list[str]) -> str:
    """把切好的句子拼回去，去掉各自两侧的空白。

    必须逐句 strip：切句正则允许句子里带空白，直接 ``"".join`` 会让
    「…挺难受的。   」这种**句尾空格**留在最终回复里（实测出现过）。

    Args:
        parts: :func:`_split_sentences` 切出的句子。

    Returns:
        str: 拼接后的文本。
    """
    return "".join(p.strip() for p in parts).strip()


def _strip_trailing_question(text: str) -> tuple[str, bool]:
    """去掉结尾的问句，用于"本轮不要提问"时的兜底。

    为什么需要代码兜底：即使 prompt 末尾已写明「只回应，不提问 / 把问号删掉」，
    实测 qwen-turbo 在一段短消息密集的对话里仍有约一半轮次照问不误
    （2026-09-26 复现：场景一连续 4 轮带问号）。这类"风格约束"在
    LLM 上是概率性的，**要变成不变量就只能落到代码里**。

    处理顺序（越靠前越保守）：
        1. 丢掉**整句**结尾的问句，保留前面的陈述内容；
        2. 丢掉「**已经说完的陈述**，逗号，结尾问句」里最后那个问句分句
           （判定见 :func:`_strip_trailing_clause_question`）；
        3. **只有当整条回复本来就是一个问句**时，才改写成陈述句；
        4. 其余情况原样返回。

    ⚠️ 第 3 条的范围必须守死。实测教训：早期版本对"就一个问句"的回复一律改写，
    结果把「老师骂你的时候，是在什么情况下发生的？」改成了
    「老师骂你的时候，大概不好受。」—— 语义跑偏、读起来像坏掉了。
    改写只在"整段就是一句问话、删掉就什么都没有"时才值得冒险；
    多句回复里宁可留一个问句，也不要动刀。

    Args:
        text: 模型回复。

    Returns:
        tuple[str, bool]: ``(处理后的文本, 是否发生了改动)``。
    """
    stripped = text.rstrip()
    if not stripped or ("？" not in stripped and "?" not in stripped):
        return text, False

    parts = _split_sentences(stripped)
    if not parts or not parts[-1].rstrip().endswith(("？", "?")):
        return text, False

    head = _join_sentences(parts[:-1])
    if head:
        return head, True

    clause_head, clause_removed = _strip_trailing_clause_question(stripped)
    if clause_removed:
        return clause_head, True

    # 到这里说明整条回复就是一个问句 —— 才轮到改写
    for rewrite in (_question_to_reflection, _declarative_from_question):
        rewritten = rewrite(stripped)
        if rewritten:
            return rewritten, True
    return text, False


def _split_sentences(text: str) -> list[str]:
    """按中英文句末标点切句，保留标点。

    Args:
        text: 任意文本。

    Returns:
        list[str]: 句子列表（已去掉纯空白项）。
    """
    parts = re.findall(r"[^。！？!?…]*[。！？!?…]+|[^。！？!?…]+$", text)
    return [p for p in parts if p.strip()]


def _cap_questions(text: str, max_questions: int = 1) -> str:
    """把一条回复里的问句数量压到 ``max_questions`` 以内。

    提示词里写了"一次只问一个"，实测仍会出现一轮抛两个问句
    （2026-09-26 复现：「是有什么特别的事要发生吗？还是最近在学校遇到什么
    烦心事了？」），连着追问的观感就是这么来的。

    **只做删除，不改写**：早期版本在"删完只剩一个问句"时会把它改写成陈述句，
    结果把「老师当着全班说的？那次是什么事？」压成了
    「那次一定有原因。」—— 用户看到的是一句与 ta 说的话无关的通用安慰。
    宁可少删一句，也不能动文本本身。所以这里只做三件事：
        1. 超出上限时，丢掉**后面**那些问句；
        2. 删完如果什么都不剩，保留**第一个**问句（宁可留一个问句）；
        3. 其余情况一律原样返回。

    Args:
        text: 模型回复。
        max_questions: 允许保留的问句数，默认 1。

    Returns:
        str: 处理后的回复；未超出上限时原样返回。
    """
    parts = _split_sentences(text)
    if len(parts) <= 1:
        return text.strip()
    question_idx = [
        i for i, p in enumerate(parts) if p.rstrip().endswith(("？", "?"))
    ]
    if len(question_idx) <= max_questions:
        return text.strip()

    keep = set(question_idx[:max_questions])
    kept_parts = [
        p for i, p in enumerate(parts) if i in keep or i not in question_idx
    ]
    result = _join_sentences(kept_parts)
    if not result:
        # 整段都是问句，删完什么都不剩 —— 保留第一个，别返回空回复
        result = parts[question_idx[0]].strip()
    logger.info("一轮里问了 %d 个问题，已压到 %d 个", len(question_idx), max_questions)
    return result


# 纯寒暄/客套（"你好""在吗""谢谢"）。这类输入**没有任何内容**可回应，
# 也检不到相关情境 —— 语料库里能匹配到的只会是"今天有点没精神"这种别的
# 场景，注进去等于逼模型替用户编状态（2026-09-26 截图：用户只说了"你好"，
# AI 回"你今天看起来有点没精神啊？"）。
_SOCIAL_ONLY_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"^(你好|您好|哈喽|哈啰|嗨|hi|hello|在吗|在不在|有人吗|早上好|早安|中午好|下午好|晚上好|晚安)"
        r"[呀啊哈呢吗~！!。.，,\s]*$",
        r"^(谢谢|多谢|感谢|谢了|thx|thanks)[呀啊啦了~！!。.，,\s]*$",
        r"^(嗯|哦|噢|好|好的|行|收到|知道了|没事|没什么)[呀啊啦了~！!。.，,\s]*$",
    )
)


def _is_social_only(text: str) -> bool:
    """本轮是否只是寒暄/客套，没有任何可回应的内容。

    用于两件事：不注入情境检索结果、以及提示词里指定寒暄打法。

    Args:
        text: 用户当前发言。

    Returns:
        bool: True 表示只有寒暄/客套，没有实质内容。
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    return any(p.match(stripped) for p in _SOCIAL_ONLY_PATTERNS)


_GREETING_PATTERN = re.compile(
    r"^(你好|您好|哈喽|哈啰|嗨|hi|hello|在吗|在不在|有人吗"
    r"|早上好|早安|中午好|下午好|晚上好)[呀啊哈呢吗~！!。.，,\s]*$"
)


def _is_greeting(text: str) -> bool:
    """本轮是否只是一句问候（用于避免连着问好两次）。

    Args:
        text: 用户当前发言。

    Returns:
        bool: True 表示是纯问候。
    """
    return bool(_GREETING_PATTERN.match((text or "").strip()))


def _assistant_greeted_before(conversation_history: list) -> bool:
    """本次对话里 AI 是否已经打过招呼。

    "你好"连回两次"你好呀"会显得很假；已经打过招呼时改口到
    「嗯，我在。」更自然（这一点与聊天历史有关，提示词自己看不到，
    所以由代码判定后写进 prompt）。

    Args:
        conversation_history: 对话历史 ``[{role, content}]``。

    Returns:
        bool: True 表示 AI 已经问过好。
    """
    prefix = re.compile(
        r"^(你好|您好|哈喽|哈啰|嗨|hi|hello|早上好|早安|中午好|下午好|晚上好)[呀啊哈呢~！!，,。. ]"
    )
    for msg in conversation_history:
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", "")).strip()
        if _is_greeting(content) or prefix.match(content):
            return True
    return False


# 在两句没说完的时候用来承接（见 _expand_bare_echo）。
_ECHO_INVITATIONS = (
    "那你当时是怎么想的？",
    "能多说一点吗？",
    "这让你心里是什么滋味？",
)


def _edit_distance(a: str, b: str) -> int:
    """两个字符串的 Levenshtein 距离（只用于短文本，写法取朴素 DP）。

    Args:
        a: 字符串一。
        b: 字符串二。

    Returns:
        int: 编辑距离。
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            )
        prev = cur
    return prev[-1]


# 被反复判定为"套路"的开场短语。模型每次都能"理解"禁止，但换一轮就忘；
# 所以除了 prompt 里的负面清单，还在**用过后**点名提醒它换说法。
_STALE_OPENINGS = ("我能感受到", "听起来你", "我理解你", "我懂你")


def _used_openings(conversation_history: list) -> list[str]:
    """本段对话里 AI 已经用过的套路开场。

    用于 prompt 末尾的"你已经用过这些了，换一个"提醒。光在规则里禁止
    实测不够（模型换一轮就忘），点名到具体短语的提醒遵守率更高。

    Args:
        conversation_history: 对话历史 ``[{role, content}]``。

    Returns:
        list[str]: 已经用过的开场短语（按定义顺序，去重）。
    """
    used: list[str] = []
    for phrase in _STALE_OPENINGS:
        for msg in conversation_history:
            if msg.get("role") != "assistant":
                continue
            head = str(msg.get("content", "")).strip()[:12]
            if phrase in head:
                used.append(phrase)
                break
    return used


# 反复出现的"陪伴/鼓励"套话。它们单说一次是暖的，但连着每轮都堆同一句，
# 就变成机械感来源（2026-09-27 用户截图：连续两轮都出现"我会一直在这里
# 陪着你"+"慢慢来"）。与 _STALE_OPENINGS 的区别：开场只看句首，这些扫正文。
_CLICHE_SENTENCES = (
    "我会一直在这里", "我会一直陪", "我在这里陪着你", "在这里陪着你", "会一直在这里",
    "慢慢来", "不着急", "按你的节奏",
    "你不需要马上", "不用马上回答", "不需要强迫自己",
    "你已经很棒", "你已经很勇敢", "这本身就是一种勇气", "本身就是一种释放",
    "不会评判你", "不评判你", "认真听你说", "愿意说就说", "想说的时候随时说",
)

# 套话只算"最近几轮"的即时重复。⚠️ 不能扫整段历史 —— 那样一句暖话说过一次
# 就永久进黑名单，模型没有陪伴表达可用，会退化成"嗯，怎么了？"这种光秃秃的
# 语气词+提问（2026-09-27 用户第二轮反馈：矫枉过正、太没人情味）。
# 隔了几轮再自然说一句同样的陪伴话，是合理的，不该拦。
_CLICHE_RECENCY_TURNS = 2


def _used_cliches(conversation_history: list) -> list[str]:
    """**最近 _CLICHE_RECENCY_TURNS 条 assistant 发言**里出现过的陪伴/鼓励套话。

    只看最近的回复，目的是拦"连着每轮重复同一句"，而不是永久封杀温暖表达。

    Args:
        conversation_history: 对话历史 ``[{role, content}]``。

    Returns:
        list[str]: 最近说过的套话（按定义顺序，去重）。
    """
    recent = [
        str(m.get("content", ""))
        for m in conversation_history
        if m.get("role") == "assistant"
    ][-_CLICHE_RECENCY_TURNS:]
    used: list[str] = []
    for phrase in _CLICHE_SENTENCES:
        if any(phrase in body for body in recent):
            used.append(phrase)
    return used


# 日常共情回复的正文长度上限（危机/高风险的安全文本不受此限）。
_MAX_REPLY_CHARS = 95


def _trim_reply(text: str, used_cliches: list[str]) -> str:
    """定稿后处理：删掉**本轮回复里**重复上轮已说过的套话句，并压到长度上限。

    只做删除，不改写（与 _cap_questions 同一原则：宁可少删也不改坏语义）。
    - 逐句扫描：若某句含有"上轮已用过"的套话，删掉该句（这是重复的陪伴宣言）。
    - 若删空了，回退到原句（绝不返回空回复）。
    - 再按 _MAX_REPLY_CHARS 截断到完整句：保留前若干句、总字数不超限，至少留一句。

    Args:
        text: 模型回复。
        used_cliches: :func:`_used_cliches` 的结果（上轮已说过的套话）。

    Returns:
        str: 处理后的回复。
    """
    parts = _split_sentences(text)
    if not parts:
        return text.strip()

    # 1) 删除含"最近说过套话"的句子。⚠️ 只有删完还剩 ≥2 句才删 —— 短回复
    #    （本就一两句）若被删到只剩个光杆，会比留着套话更没人情味，宁可留。
    if used_cliches:
        kept = [p for p in parts if not any(c in p for c in used_cliches)]
        if len(kept) >= 2:
            parts = kept

    # 2) 长度截断：按整句累加，超过上限就停在上一句，至少保留第一句
    result = ""
    for p in parts:
        candidate = (result + p.strip()) if result else p.strip()
        if len(candidate) > _MAX_REPLY_CHARS and result:
            break
        result = candidate
    if not result:
        result = parts[0].strip()
    if len(result) != len(text.strip()):
        logger.info("回复去套话/压长度：%r → %r", text[:40], result[:40])
    return result


def _is_bare_echo(text: str, user_message: str) -> bool:
    """回复是否只是把用户的话原样抛回来（顶多把"我"换成"你"）。

    实测复现（2026-09-26 第二张截图后的重放）：
        用户：老师骂我了     →  AI：老师骂你了？
        用户：爸妈昨天又吵架了 →  AI：爸妈又吵架了？
        用户：我妈说要不是我他们早就离了 → AI：我妈说要不是你他们早就离了？
    这种"复读 + 问号"是比套路开场更空的一轮：ta 刚说完的话被你重复了一遍，
    什么信息都没增加。

    判定用**编辑距离**而不是"包含"关系，因为回声往往改人称
    （"我"→"你"），还可能顺手删掉一两个词（"昨天"）。阈值取
    ``max(1, 长度差的 1/4)``：足够容忍人称/虚词差异，又不会把
    "被这么说肯定难受。这话是他随口说的吗？"这种真回应误判成回声。

    Args:
        text: 模型回复。
        user_message: 用户本轮发言。

    Returns:
        bool: True 表示这是空回声。
    """
    reply = re.sub(r"[\s，。！？!?、…~～：:；;]+", "", text or "")
    user = re.sub(r"[\s，。！？!?、…~～：:；;]+", "", user_message or "")
    # 太短的一律不动：回声至少要有一段完整的话，才有"复读"的观感
    if len(reply) < 4 or len(user) < 4:
        return False
    return _edit_distance(reply, user) <= max(1, len(user) // 4)


def _expand_bare_echo(text: str, user_message: str) -> str:
    """把"复读 + 问号"改写成一句真正的回应 + 一个追问。

    改写用的都是固定话术（不是模型生成的），因此这一层不会引入新幻觉：
    如果说错了，错的是"追问方向"，而不是"凭空捏造用户状态"。
    样本已经过人工挑选，语气统一。

    Args:
        text: 模型回复（已判定为空回声）。
        user_message: 用户本轮发言。

    Returns:
        str: 改写后的回复；无法改写时返回原文（调用方保留原样）。
    """
    stripped_user = (user_message or "").strip().rstrip("。！!，,？?…")
    if not stripped_user:
        return text
    has_question = text.rstrip().endswith(("？", "?"))
    statement = f"「{stripped_user}」，这句话我记住了。"
    if not has_question:
        return statement
    # 追问方向按用户发言里有没有明确情绪词来选，避免文不对题
    if any(kw in stripped_user for kw in ("难受", "委屈", "伤心", "害怕", "烦", "累", "痛")):
        question = "这种感觉，是从什么时候开始的？"
    else:
        question = "这件事里，最让你放不下的是哪一点？"
    logger.info("回复只是复读用户原话，已改写为共情+追问：%r", text[:30])
    return f"{statement}{question}"


def _finalize_reply(
    text: str,
    allow_question: bool,
    max_questions: int = 1,
    user_message: str = "",
    conversation_history: Optional[list] = None,
    risk_level: str = "low",
) -> str:
    """定稿一条模型回复：压问句数 → 按需删尾问句 → 去重复套话 + 压长度。

    单独抽出来是为了让**流式与非流式两条路径共用**同一套定稿规则；
    已经吐出去的字收不回来，所以流式路径只能在整段生成后定稿（见
    ``_chat_stream_events``：定稿结果与已上屏内容不一致时补一个 REVISE 事件）。

    顺序上**先压数量、再判尾句、最后去套话/压长度**：删掉多余问句后
    才做句子级裁剪，避免裁剪把该保留的问句又带出来。

    ⚠️ ``allow_question=False`` **只应来自"内容层面不该问"的判断**
    （用户拒绝深谈 / 高危机 / 寒暄无内容可问），**不要**因为"连着问了几轮"
    就传 False：实测那样会把「老师骂你了？」这类只有一句问话的回复
    削成「老师骂你了。」—— 问句没了，只剩回声，比原来更糟。
    连着追问只由 prompt 侧的语气规则降温（见 :data:`_QUESTION_RUN_LENGTH`）。

    ⚠️ 去套话/压长度**只对低/中风险生效**：危机/高风险的回复含安全确认与
    热线号码，一个字都不能删（见 :func:`_trim_reply`）。

    Args:
        text: 模型原始回复。
        allow_question: 本轮是否允许以问句结尾。
        max_questions: 单轮问句数上限，默认 1。
        user_message: 用户本轮发言；传入后可以识别并改写"复读用户原话"的回复。
        conversation_history: 对话历史；传入后可删除上轮已说过的套话。
        risk_level: 本轮风险等级，决定要不要做去套话/压长度。

    Returns:
        str: 定稿后的回复。
    """
    # 0) 纯复读先改写：这是比套路开场更空的一轮（见 _is_bare_echo）
    if user_message and _is_bare_echo(text, user_message):
        text = _expand_bare_echo(text, user_message)

    capped = _cap_questions(text, max_questions=max_questions)
    if not allow_question:
        cleaned, removed = _strip_trailing_question(capped)
        if removed:
            logger.info("本轮禁止提问，已删去结尾问句：%r → %r", capped[:40], cleaned[:40])
        capped = cleaned

    # 去重复套话 + 压长度：仅日常（低/中风险）；危机/高风险的安全文本原样保留
    if risk_level in ("high", "crisis"):
        return capped
    used_cliches = _used_cliches(conversation_history or [])
    return _trim_reply(capped, used_cliches)


def _build_llm_messages(
    user_message: str,
    conversation_history: list,
    emotion_probs: list[float],
    risk_level: str,
    style: Optional[UserStyle] = None,
    scene_context: str = "",
    ask_question: bool = True,
    memory_context: str = "",
) -> list[dict]:
    """装配千问调用的消息列表（system prompt + 最近 8 条历史 + 本轮发言）。

    从 :func:`_call_llm` 中抽出，是为了让**流式与非流式两条路径共用同一段
    prompt 构造逻辑**。prompt 决定模型说什么，是本项目安全与疗效的第一道
    闸门；若流式端点自己再拼一份，两边必然随时间漂移，而漂移是静默的。

    Args:
        user_message: 用户当前发言。
        conversation_history: 历史对话（``[{role, content}]``）。
        emotion_probs: 文本情绪概率 ``[快乐, 悲伤, 焦虑, 愤怒, 中性]``。
        risk_level: 风险等级。
        style: 用户表达风格，决定本轮是否追问。
        scene_context: C2D2 场景检索上下文；为空串时不注入，避免让模型对
            用户没说过的处境共情。
        ask_question: 本轮是否允许追问。由调用方用
            :func:`_ask_question_this_turn` 算好传入，保证注入的 ``ask_rule``
            与实际话术规则同源。**为 True 也可能被降级**：若最近两轮 AI 回复
            都追问了（:func:`_assistant_asked_repeatedly`），本轮仍会写明
            「不要提问」，把追问改成隔轮出现。

    Returns:
        list[dict]: OpenAI 兼容的 messages 列表。
    """
    # ⚠️ 刻意**不把情绪标签的字面值喂进 prompt**。过去注入「主导情绪=焦虑」，
    # 模型会忍不住复述这个词（"听起来你很焦虑"）—— 这恰恰是"复读情绪标签"
    # 这一死板来源。这里只保留**负性强度数值**与风险等级：够模型把握分量，
    # 又不会被某个具体情绪词锚定。
    neg_intensity = _negative_emotion_intensity(emotion_probs)
    # 节拍轮次：对话已进行的轮数（决定本轮"接一句"还是"往下问"）。
    turn_count = len(conversation_history) // 2

    style_desc = {
        UserStyle.INTROVERTED: '用户偏内向安静，回复要短、留白多，别连着追问',
        UserStyle.EXTROVERTED: '用户偏外向健谈，你可以接得活一点、多来回几句',
        UserStyle.AGITATED: '用户此刻激动/烦躁，先让 ta 落地，别急着往下走',
        UserStyle.CALM: '用户此刻较平静，可以多聊一层',
    }.get(style, '')

    # 本轮输入是不是"纯寒暄/客套" —— 这决定 prompt 里换一套打法。
    # 不处理的话，模型会拿情境检索的语料替用户编状态（截图里的
    # "你今天看起来有点没精神啊"就是这么来的）。
    #
    # ⚠️ 必须**先判"是否已经问过好"**再判通用寒暄：问候本身就属于
    # social_only，反过来写的话第二次"你好"会落进通用分支，又去问一遍好。
    social_only = _is_social_only(user_message)
    if _is_greeting(user_message) and _assistant_greeted_before(conversation_history):
        social_rule = (
            "⚠️ 本轮是打招呼，但**前面已经问过好了** —— 不要再重复问好，"
            "回一句「嗯，我在。」或「在的，你说。」就够了，不要提情绪、不要追问。"
        )
    elif social_only:
        social_rule = (
            "⚠️ 本轮用户只说了寒暄/客套话，**什么内容都没有**。"
            "只回应这句话本身："
            "打招呼就自然地打个招呼（「你好呀，今天想说点什么？」）；"
            "道谢就回一句得体的（「不用谢，我在。」）；"
            "应声「嗯/好的」就接一句（「好，你慢慢说。」）。"
            "**绝对不要**猜测或描述 ta 的状态（不许说「看你有点累/没精神/心情不好」），"
            "**也不要**追问「发生什么事了」—— ta 什么都还没说，问了就是硬造。"
        )
    else:
        social_rule = ""

    # 连续追问：**允许连着两轮**（用户明确要求"追问要回来"），
    # 只拦"连着三轮都在问" —— 那时对话就变成审问了。
    #
    # ⚠️ social_only 必须排在最前：寒暄轮次没有内容可追问，若还注入
    # "本轮要追问"，模型就会对着"你好"问"发生什么事了"。
    asked_repeatedly = _assistant_asked_repeatedly(conversation_history)
    # 节拍：默认「问、问、接一句」——每三轮留一轮**只承接不提问**，
    # 打断"陈述句+问号"的机械循环。ask_rule 与 final_reminder 必须同源，
    # 否则会出现"中段说不用问、末尾又叫它问"的自相矛盾。beat_no_question 一处决定两者。
    beat_no_question = (turn_count % 3 == 2)
    if not ask_question:
        ask_rule = (
            "本轮**不要提问**：用户情绪还没平复，或此刻不想被追问。"
            "只回应 ta 说的内容，让 ta 感到被听到就够了，结尾不要出现问号。"
        )
        beat_no_question = True
    elif social_only:
        ask_rule = "本轮**不要提问**：ta 只说了寒暄/客套话，没有内容可追问。"
        beat_no_question = True
    elif asked_repeatedly:
        ask_rule = (
            "⚠️ 本轮**不要提问**：前面已经连着三轮用问句结尾了。"
            "这一轮换成一句陈述 —— 接住 ta 刚说的内容、或者说一句你的感受。"
            "连着问会把对话变成审问。"
        )
        beat_no_question = True
    elif beat_no_question:
        ask_rule = (
            "这一轮**先不往下问**：把 ta 接住 —— 给一两句你自己的感受、"
            "或点出 ta 没说出口的那层。不提问不等于敷衍，别只回一个语气词。"
        )
    else:
        ask_rule = (
            "本轮**要追问**（这是默认动作）：顺着 ta 刚说的内容问一个开放式问题，"
            "把话头往下递。不要停在「我懂你」上就不动了。"
        )

    # 末尾硬约束：放在 prompt 最靠近"这一轮"的位置。
    # 实测（2026-09-26 重放截图对话）：把"本轮不要提问"只写在中段的 ask_rule 里，
    # 模型约一半轮次照问不误；末尾再重复一条可明显提高遵守率，
    # 但仍是概率性的 —— 真正的不变量由 :func:`_finalize_reply` 保证。
    if beat_no_question:
        final_reminder = (
            "这一轮：不提问，但要把 ta 接住 —— 用一两句陈述把 ta 说的东西"
            "回应到位（你的感受、或点出 ta 没说出口的那层），别只丢个语气词。"
        )
    else:
        final_reminder = (
            "这一轮：先用一句实打实的话接住 ta 说的具体内容（提到 ta 说的人或事），"
            "再顺着往下问**一个**问题，让 ta 有话可接。"
        )

    # 已经用过的套路开场 + 上轮说过的陪伴套话 —— 合成一个"硬禁用"块，
    # 放在最靠近"这一轮"的位置。开场只看句首，套话扫整段正文：
    # 模型爱每轮补一句"我会一直陪着你/慢慢来/你已经很棒了"，说一次是暖，
    # 轮轮说就是机械。这里点名 + 定稿处 :func:`_trim_reply` 兜底删除。
    used_openings = _used_openings(conversation_history)
    used_cliches = _used_cliches(conversation_history)
    ban_lines = []
    if used_openings:
        ban_lines.append(
            "禁止用作**开场**（已用过，含近义改写也不行）："
            + "、".join(f"「{o}」" for o in used_openings) + "。"
        )
    if used_cliches:
        ban_lines.append(
            "这几句陪伴/鼓励话**上一轮刚说过，本轮别再原样重复**："
            + "、".join(f"「{c}」" for c in used_cliches) + "。"
            "该给的温暖继续给，只是换个说法、或落到 ta 这件具体的事上，别复制粘贴。"
        )
    if ban_lines:
        openings_ban = "# 本轮硬性禁用\n" + "\n".join(ban_lines)
    else:
        openings_ban = ""

    # 人物画像连续性：把长期记忆里提炼出的稳定事实回注，让模型"记得这个人"
    # （叫得出细节，而不是每轮都像第一次见面）。memory_context 来自
    # _prepare_chat_turn 的记忆召回，为空时这一段不注入。
    if memory_context:
        persona_line = (
            "关于这个人的**旧背景**（以前听 ta 提过，仅供你理解 ta 是谁）：\n"
            + memory_context.strip()
            + "\n⚠️ 这**不是** ta 本轮说的话。除非 ta 这一轮自己提到相关的事，"
            "否则绝不要拿它来回应、点破或追问——把 ta 没说的场景当成事实是最伤人的。"
        )
    else:
        persona_line = ""

    system_prompt = f"""# 角色定位
你是一个会听人说话的 AI 陪伴者，**不是医生，也不是老师**。
你唯一的任务：让对面这个青少年觉得「有人真的听懂我在说什么」。
你不需要表现得专业，更不需要解决 ta 的问题。

# 核心原则（一个字都别违背）
1. **只回应 ta 说的话，不要替 ta 编话。**
   ta 没提过的事、没说过的状态，你一概不知道。
   反例（严重禁止）：ta 只说「你好」，你回「你今天看起来有点没精神」——
   这是凭空捏造，比不回应更伤人。
2. **共情是每个情绪轮次的必做动作，但不要每轮都用同一句话。**
   要承接 ta 的感受，方式要换着来：有时放在开头，有时放在后半句，
   有时只用一句「这不怪你」。
   反例（**禁止用这个句式开头**）：「我能感受到你现在很难受。」
   想表达同样的意思，换成**指着 ta 刚说的那件事说**：
   ✅「你一说这事我就听出来了，搁谁身上都难受。」（接住 ta 说的，不补细节）
   ✅「这话太重了。」（就事论事，感受自然带出来）
   也可以是「我听出来了」「这事儿换谁也受不了」。
   ⚠️ 共情只能落在 **ta 本轮真的说过**的内容上。ta 只说「难受」没说原因，
   就**不要**替 ta 编一个场景（被批评、考试、吵架都不行）——先接住情绪，再问发生了什么。
3. **追问是默认动作。** ta 给出了内容，你就顺着问一个开放式问题，
   把话头往下递。连着两轮都追问是可以的，但**不要连着三轮都在问**。
4. 不讲道理、不给建议、不灌鸡汤、不替 ta 下结论。ta 自己的答案才算数。
5. 不否定感受：绝不说「想开一点」「别难过」「这点小事不算什么」「很多人比你更惨」。
6. 尊重节奏：ta 说不想聊，就停下来，不硬追。

# 底线（硬性边界）
1. 不做任何诊断、不提任何药物，不说「你有抑郁症/焦虑症」这类话。
2. 提到自伤、自杀、伤人时，先做安全确认，并建议联系线下心理危机热线或医院心理科。
3. 被问身份时说明：我是 AI 陪伴者，**不能替代线下持证心理咨询师与精神科医生**。

# 怎么说话
- **每一轮都必须先用一句实打实的话接住 ta**，让 ta 知道你在听、你懂那是怎么回事，然后才谈要不要往下问。
  - ta 说了具体的事/人/原话 → 就**指着那句**接（「这话太重了」「憋了一个月真够受的」）。
  - ta 只报了个情绪、没说原因（「我难受」「心情不好」）→ **别编原因**。就接住这份情绪本身：
    「难受的时候能说出来，本身就挺难的。」「我在。是心里堵，还是身上也不舒服？」
  ⚠️ 不许只丢一个语气词加提问就交差 —— "嗯，怎么了？"这种没有承接的短问，
  比长一点的套话更让人心凉。至少先有一句真正接住 ta 的话。
- 共情要**指着 ta 说过的话说**，不是贴情绪标签：与其"你很伤心"，不如"这话搁谁身上都受不了"。
- 温暖的话该说就说，只是**别连着每轮都复制同一句**；这轮陪一句、下轮换个说法或落到 ta 这件具体的事上。
- 长度：一般 2~3 句、30~90 字，把温度和承接说够；ta 说得少你可以短一点，但**不能短到敷衍**。别写成一大段、别分点。口语、可以停顿。

# 两段示范（只学**语气和节奏**，句子和场景都不要照抄）
⚠️ 示范里的「作业堆成山」「老师当众骂」都是**举例**，跟眼前这个来访者毫无关系。
绝不要把示范中出现过的情节当成 ta 说的话——ta 没讲的就是没说。
【示范一 · 学业压力】
ta：作业堆成山了，我怎么写都写不完，好烦。
你：怎么写都写不完这种感觉，是真的能把人磨没脾气。是今天特别多，还是一直这么堆着？
ta：一直这样，快一个月了，晚上都睡不好。
你：连着快一个月，白天硬撑、晚上还睡不成——这哪是矫情，是真扛久了。
ta：反正没人觉得这算个事。
你：我听着是。这么重的日子一个人顶着，本身就不轻。是哪件事最先压上来的？
（注意节奏：问、问、**第三句不急着问**，先陪一句；也从不重复同一句开场。）

【示范二 · 被老师当众批评】
ta：老师当着全班的面骂我，说我什么都做不好。
你：当着全班被这么说——光是想想脸上都发烫。那句话到现在还卡在脑子里？
ta：嗯，我可能就真的很差。
你：他一句话就给你整个人判了分？我不认。气归气，先别拿他的话往自己身上贴。
ta：其实我也没气，就是难受。
你：那就是伤着你了，跟气不气没关系。被信任的大人那样说，最扎人。
（注意：ta 自我贬低时**不跟着复读**——不说"你觉得你很差"，而是挡回去。）

# 要避免（这几条最伤"活人感"）
1. 每轮同一句式开头（"我能感受到…""听起来你…""我理解你…"），或每轮都走"复述情绪→发生了什么？"这个固定节拍。
2. 把 ta 的原话加个问号抛回去（"老师骂你了？"）——等于没说；也别把 ta 原话直接当你自己的句子（人称都串了）。
3. 堆共情词（我懂/抱抱/你真勇敢/我一直陪你），一两次就够；同一句留白（"嗯，我在"）全篇只能用一次。
4. 猜测 ta 没说的状态、或替 ta 编处境。ta 没讲的，你就是不知道——**尤其不许**把示范里的情节、
   或"旧背景"里记的东西，当成 ta 本轮说的话来回应。
5. 跟着 ta 贬低自己的话复读。ta 说"他说我很废物"，你接"他那样说你，听着确实难受"，别回"你觉得你是废物"。

# 追问方向（要问时挑一个，别每轮都是"什么感觉"）
- **探事件**：发生了什么？当时是什么情况？—— ta 只说"难受"时往往自己也没头绪，事件类问题最好接。
- **探感受**：那是心里堵，还是身上也不舒服？
- **探想法**：那一刻脑子里冒出来的第一个念头是什么？
上面这几行、以及示范里"（注意…）"的括号说明都是**给你的**，不要输出给来访者。

# 当前轮次信息（内部参考，别把这些词或分析过程说出来）
当下负面情绪强度约 {neg_intensity:.0%}；风险等级={risk_level}。
（强度只用来把握回应分量，**不要**据此点名 ta 的情绪，也不要输出"焦虑/悲伤"这类标签。）
{style_desc}
{persona_line}
{ask_rule}
{social_rule}
{openings_ban}
{final_reminder}
"""
    if scene_context:
        system_prompt = f"{system_prompt}\n{scene_context}"

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    # 加入最近4轮对话历史
    for msg in conversation_history[-8:]:
        role = msg.get("role", "user")
        if role not in ("user", "assistant", "system"):
            role = "user"
        messages.append({"role": role, "content": msg.get("content", "")})
    messages.append({"role": "user", "content": user_message})
    return messages


def _call_llm(
    user_message: str,
    conversation_history: list,
    emotion_probs: list[float],
    risk_level: str,
    style: Optional[UserStyle] = None,
    dialogue_mode: str = "EMPATHY",
    scene_context: str = "",
    memory_context: str = "",
) -> str:
    """调用千问大模型生成回复，失败时回退模板。

    话术形态由**一段统一的人本取向提示词**决定，不再按 ``dialogue_mode``
    在共情/苏格拉底两套 prompt 之间二分：那两套约束互相冲突（一套禁止反问、
    另一套强制每段反问），叠加"1-2 句"上限后，模型只剩"复述情绪 + 陪伴宣言"
    一个合法动作。``dialogue_mode`` 仍由调用方计算并写入审计与前端标签，
    但**不再影响任何一句话术**；是否追问改由 :func:`_ask_question_this_turn`
    按用户风格与风险等级决定。

    Args:
        user_message: 用户当前发言。
        conversation_history: 历史对话（``[{role, content}]``）。
        emotion_probs: 文本情绪概率 ``[快乐, 悲伤, 焦虑, 愤怒, 中性]``。
        risk_level: 风险等级。
        style: 用户表达风格，决定本轮是否追问。
        dialogue_mode: 对话模式。**已不再参与 prompt 选择**，仅为兼容旧调用方
            保留（审计与前端标签仍使用它）。
        scene_context: C2D2 场景检索上下文，由
            ``intervention.scene_retrieval.build_scene_context`` 生成；
            为空串（相关度不足）时不注入，避免让模型对用户没说过的处境共情。

    Returns:
        str: 模型回复；调用失败时回退到模板回复。
    """
    api_key = os.environ.get('DASHSCOPE_API_KEY', '')
    negative_intensity = _negative_emotion_intensity(emotion_probs)
    ask_question = _ask_question_this_turn(
        style, risk_level, user_message=user_message,
        negative_intensity=negative_intensity,
    )
    # 是否允许**以问句结尾** —— 只由内容层面决定（安全 / 放行 / 寒暄），
    # 不看"连着问了几轮"（见 _finalize_reply 的说明：按轮数删问句会削出回声）。
    allow_question_ending = ask_question and not _is_social_only(user_message)
    if not api_key:
        return _generate_empathy_reply(
            emotion_probs, risk_level, style=style, ask_question=ask_question,
        )

    messages = _build_llm_messages(
        user_message,
        conversation_history,
        emotion_probs,
        risk_level,
        style=style,
        scene_context=scene_context,
        ask_question=ask_question,
        memory_context=memory_context,
    )
    try:
        resp = requests.post(
            f'{_LLM_BASE_URL}/chat/completions',
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
            json={
                'model': _LLM_MODEL, 'messages': messages, 'max_tokens': 200,
                **_sampling_params(risk_level),
            },
            timeout=_LLM_TIMEOUT,
        )
        if resp.ok:
            data = resp.json()
            reply = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            if reply.strip():
                return _finalize_reply(
                    reply.strip(), allow_question_ending, user_message=user_message,
                    conversation_history=conversation_history, risk_level=risk_level,
                )
        else:
            logger.warning("千问调用失败 HTTP %s，回退模板回复", resp.status_code)
    except Exception as e:  # noqa: BLE001
        logger.warning("千问调用异常，回退模板回复: %s", e)

    # LLM 调用失败，回退模板
    return _generate_empathy_reply(
        emotion_probs, risk_level, style=style, ask_question=ask_question,
    )


def _generate_empathy_reply(
    emotion_probs: list[float],
    risk_level: str,
    memory_context: str = "",
    style: Optional[UserStyle] = None,
    ask_question: bool = True,
    crisis_turns: int = 1,
) -> str:
    """基于情绪和风险等级生成共情回复（模板降级）

    模板分层与 :func:`_ask_question_this_turn` 保持一致：

        - ``ask_question=False``（拒绝深挖 / 情绪激动 / 高风险）：
          只从 :data:`_COMPANIONSHIP_TEMPLATES` 或
          ``_STYLE_REPLY_TEMPLATES[INTROVERTED/AGITATED]`` 里取，
          这两池**全部不含问句**，绝不会出现"本轮不要提问却丢回一个问题"。
        - ``ask_question=True``：可用 ``_EMPATHY_TEMPLATES``（含开放式追问）。

    ⚠️ 高风险/危机的模板含安全确认问句，属于"风险确认"而非"认知追问"，
    因此不受 ``ask_question=False`` 限制；但仅在确实高风险时才可取到。

    Args:
        emotion_probs: 情绪概率（当前仅用于兼容签名，模板按风险等级选）。
        risk_level: 风险等级。
        memory_context: 预留的记忆上下文参数（当前不参与选句）。
        style: 用户表达风格。
        ask_question: 本轮是否允许追问。
        crisis_turns: 危机状态持续轮次（用于 C-SSRS 阶梯式响应，默认 1）。

    Returns:
        str: 模板回复。
    """
    # 安全确认优先：高风险/危机无论风格都必须走这一支
    if risk_level in ("crisis", "high"):
        templates = _EMPATHY_TEMPLATES.get(risk_level, _EMPATHY_TEMPLATES["crisis"])

        # 危机模板现在是 dict（按轮次分级），需要特殊处理
        if isinstance(templates, dict):
            # C-SSRS 阶梯式响应：turn 1=接纳探索, turn 2=评估严重度, turn 3+=提供资源
            turn_key = min(crisis_turns, max(templates.keys()))  # 超出最大键则用最大键
            turn_templates = templates.get(turn_key, templates[1])
            return turn_templates[hash(risk_level) % len(turn_templates)]
        else:
            # 向后兼容：如果模板仍是 list
            return templates[hash(risk_level) % len(templates)]

    if not ask_question:
        # 内向/激动两池天然无问句，其余风格用通用陪伴池
        templates = _STYLE_REPLY_TEMPLATES.get(style)
        if not templates or style in (UserStyle.CALM, UserStyle.EXTROVERTED):
            templates = _COMPANIONSHIP_TEMPLATES
        return templates[hash(risk_level) % len(templates)]

    templates = _EMPATHY_TEMPLATES.get(risk_level, _EMPATHY_TEMPLATES["low"])
    return templates[hash(risk_level) % len(templates)]


# 危机关键词快速检测（独立于 perception 服务，覆盖直接/间接/隐晦表达）
_CRISIS_KEYWORDS = [
    # ⚠️ 本表与 perception/perception_service.py 的 _DEFAULT_CRISIS_KEYWORDS 必须保持一致。
    # 两处各自独立筛查（本处是 /smart-chat 生产路径的权威前哨），任一处漏词都等于
    # 该词在对应入口上不被识别。新增词条请同时更新两处。
    # ── 直接自杀/自伤意图 ──
    '自杀', '自残', '自伤', '不想活', '想死', '去死',
    '跳楼', '割腕', '划手', '划手臂', '服毒', '上吊',
    '活不下去', '死了算了', '不想活了', '不想醒来',
    '伤害自己', '吃药自杀', '结束生命', 'overdose',
    '烧自己', '刺自己', '掐自己',
    # ── 间接危机表达（绝望/无价值/告别）──
    '活着没意思', '活着没什么意义', '没有活下去的理由',
    '活着是一种负担', '活着好累', '不想存在',
    '消失就好了', '消失好了', '如果我不在了',
    '没人会在意', '世界没有我会更好', '没有我会更好',
    '我是废物', '没人要我', '我是个负担',
    '了结', '解脱', '一了百了', '了断',
    '结束一切', '结束这一切',
    '撑不住了', '撑不下去',
    '不想努力了', '放弃一切',
    # ── 告别/后事安排暗示 ──
    '交代后事', '最后的告别', '写遗书',
    '把我的东西', '留给你们', '对不起大家',
    '请原谅我', '我不值得', '我配不上',
    '你们会过得更好', '没有我的日子',
    # ── 隐晦/哲学化危机表达 ──
    '不想面对', '看不到希望', '看不到出路',
    '一切都没意义', '一切都没有意义',
    '好想逃', '无尽的黑暗', '深渊',
    '存在没有意义', '存在的荒谬', '虚无',
    '精神内耗到极限', '倦怠到极限',
    '人间不值得', '生而为人我很抱歉',
    '活着像行尸走肉', '灵魂已经死了',
    '心已经空了', '什么都感觉不到了',
    # ── 青少年常见危机表达 ──
    '不想上学', '好累不想动',
    '累了', '没有意义',
    '活着干嘛', '有什么用',
    '反正没人在乎', '多我一个不多',
    '我就是个笑话', '谁都不会心疼',
]


def _detect_crisis(text: str) -> bool:
    """快速危机关键词检测，用于覆盖 perception 情绪分析的盲区"""
    text_lower = text.lower()
    return any(kw in text_lower for kw in _CRISIS_KEYWORDS)


# 量表严重程度 → 风险贡献。severity 是**中文**分级名，取自
# assessment/scale 的 PHQ9_CUTOFFS / GAD7_CUTOFFS / PSS10_CUTOFFS，
# 例如「中度抑郁」「重度焦虑」「高压力」「轻度焦虑」「中等压力」「无抑郁」。
# ⚠️ 此前用英文 ("moderate","severe","mild") 比较，恒不命中 —— 除危机关键词
# 外风险永远算成 low，既漏报了该升级的高风险，也让流式增量在高危轮关不掉。
#
# 分级要贴合临床阈值：只有**中重度及以上**（PHQ-9≥15、GAD-7 重度、PSS 高压力）
# 才算 high（该升级、关增量流式）；中度/轻度是"该关注"的 medium。
# 否则一句"我好难受"（悲伤概率高→PHQ-9 中度抑郁）就被判成 high，既误触发升级、
# 又不该关掉的流式也被关掉。注意"中重度"含子串"重度"，两者都落在 high。
_HIGH_SEVERITY_MARKERS = ("中重度", "重度", "严重", "高压力")
_MEDIUM_SEVERITY_MARKERS = ("中度", "轻度", "中等压力")


def _severity_to_risk(severity: str) -> str:
    """把中文量表分级名映射为风险贡献 'high' / 'medium' / 'low'。

    用**子串标记**而非精确匹配：既兼容三张量表各自的命名，也不会在
    新增分级时静默漏判（宁可命中面宽一点，配合上层"high 优先"聚合）。
    """
    if not severity:
        return "low"
    if any(m in severity for m in _HIGH_SEVERITY_MARKERS):
        return "high"
    if any(m in severity for m in _MEDIUM_SEVERITY_MARKERS):
        return "medium"
    return "low"


# ============================================================
# 流式对话：LLM 增量输出 + 增量安全闸门
# ============================================================
#
# 为什么不能「模型吐一个字就上屏一个字」：
#   本项目的安全审计是**整段回复级**的判据（五轴审计需要完整回复 + 对话历史
#   + 状态机上下文），不可能在第一个字之前就拿到结论。逐字上屏等于在审计之前
#   就把未经检查的内容交付给了用户。
#
# 采用「先说、后校」的三重折中：
#   1. 句子级缓冲 —— 只有完整句子才上屏，闸门手上至少有一个语义单位可判；
#   2. 增量闸门 —— 复用 audit_rules.yaml 既有的词法红线，命中即停止生成；
#   3. 定稿改稿 —— 整段审计一旦否掉已生成内容，用 REVISE 事件整体替换。
#   另外 high / crisis 风险**完全关闭增量流式**：升级流程必须原子下发，
#   逐字吐出的安全确认与热线会看起来像普通闲聊。

# 句末标点 + 换行：一个「完整语义单位」的切分依据。
_SENTENCE_BOUNDARY_RE = re.compile(r'[。！？!?；;\n]')

# 逗号级提前放行：缓冲超过此长度且暂无句末标点时，在最近一个小句边界放行。
# 用途是**聊天的流式观感** —— 阈值取 50 时，一句 20~30 字的中文要整句憋完才吐，
# 前端只收到 1~2 个 delta，看着就像"转圈→整段蹦出来"，不像在逐字生成。
# 降到 12 后，逗号/顿号/冒号处即切，一句也能拆成几个小片段陆续上屏。
# ⚠️ 安全性不降：红线词都是不含逗号的连续短语（"你有抑郁症"等），小句边界切不开
# 它们，每个放行片段仍各自过 :meth:`_StreamingSafetyGate.screen` 筛查。
_COMMA_EARLY_RELEASE_CHARS = 12
_COMMA_BOUNDARY_RE = re.compile(r'[，,、：:]')

# 单个语义单位的字符上限。中文一句话通常 10~30 字，正常路径碰不到这个上限；
# 它的作用是兜底：模型若不输出任何句末标点，不能让内容永久卡在缓冲里。
_STREAM_UNIT_MAX_CHARS = 200


def _sse_pack(event_type: ChatStreamEventType, payload: dict) -> str:
    """把一次事件编码为 SSE 文本帧。

    Args:
        event_type: 事件类型。
        payload: JSON 可序列化的载荷。

    Returns:
        str: ``event: <type>`` + ``data: <json>`` + 空行的完整帧。
    """
    return (
        f"event: {event_type.value}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


def _iter_llm_deltas(
    messages: list[dict],
    api_key: str,
    timeout: int = _LLM_TIMEOUT,
    sampling: Optional[dict] = None,
) -> Iterator[str]:
    """以流式方式调用千问，逐块产出回复增量文本（OpenAI 兼容 SSE）。

    本函数**不吞异常**：调用方需要区分「一个字都没拿到」和「流到一半断了」，
    两者兜底策略不同（前者退回模板，后者把已生成部分交给审计）。

    Args:
        messages: 由 :func:`_build_llm_messages` 装配的消息列表。
        api_key: DashScope API Key。
        timeout: 连接与读取超时（秒）。
        sampling: 由 :func:`_sampling_params` 给的采样参数；为空回退温度 0.85。

    Yields:
        str: 增量文本片段。

    Raises:
        requests.RequestException: 网络层失败。
        RuntimeError: 服务返回非 2xx 状态码。
    """
    payload_sampling = sampling or {"temperature": 0.85}
    with requests.post(
        f'{_LLM_BASE_URL}/chat/completions',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        json={
            'model': _LLM_MODEL,
            'messages': messages,
            'max_tokens': 200,
            'stream': True,
            **payload_sampling,
        },
        stream=True,
        timeout=timeout,
    ) as resp:
        if not resp.ok:
            raise RuntimeError(f"千问流式调用失败 HTTP {resp.status_code}")
        for raw_line in resp.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode('utf-8', errors='replace').strip()
            if not line.startswith('data:'):
                continue
            payload = line[5:].strip()
            if payload == '[DONE]':
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = chunk.get('choices') or []
            if not choices:
                continue
            text = (choices[0].get('delta') or {}).get('content')
            if text:
                yield text


class _StreamingSafetyGate:
    """增量安全闸门 —— 在增量文本上屏之前做**既有规则**的前置筛查。

    设计边界（重要，改动前请先读）：
        * 本闸门**不引入任何新的安全规则或阈值**。它只复用
          ``config/audit_rules.yaml`` 里既有的 ``stigma_rejection`` 词表
          （含 ``teen_specific.stigma_rejection`` 的扩展词表），把原本在整段
          审计时才生效的词法红线**提前到句子级**。按 AGENTS.md「自动化流程
          不得自行修改安全阈值或升级条件」，这里改变的是既有规则的生效时机，
          不是规则本身。
        * 本闸门**不是**审计的替代品。整段回复仍必须过
          :func:`_audit_and_finalize` 里完整的 SafetyLoop 五轴审计；闸门的
          产物只是「要不要立刻停止继续生成」。
        * 因此闸门**不可能**拦住所有不该说的话：它只认词表。真正兜底的是定稿
          时的整段审计，其改稿通过 REVISE 事件生效。
        * 已知缺口：``audit_rules.yaml`` 目前没有药物相关词法轴，而 AGENTS.md
          明令禁止药物建议。补这一轴属于**新增安全规则**，需人工审核后方可
          合入，故此处不擅自添加。
    """

    def __init__(self) -> None:
        cfg = load_audit_config()
        stigma = cfg.get('stigma_rejection') or {}
        teen = (cfg.get('teen_specific') or {}).get('stigma_rejection') or {}
        terms = (
            list(stigma.get('diagnostic_labels') or [])
            + list(stigma.get('stigmatizing_terms') or [])
            + list(teen.get('additional_diagnostic_labels') or [])
            + list(teen.get('additional_stigmatizing_terms') or [])
        )
        # 统一小写以支持英文词条大小写不敏感匹配；去重排序保证行为可复现。
        self._terms: tuple[str, ...] = tuple(sorted({t.lower() for t in terms if t}))
        self._pending: str = ''
        self._tripped: bool = False
        self._hits: list[str] = []
        self._released_chars: int = 0
        self._scanned_units: int = 0

    @property
    def tripped(self) -> bool:
        """是否已命中红线（命中后 :meth:`feed` 一律返回空列表）"""
        return self._tripped

    @property
    def hits(self) -> list[str]:
        """命中的红线词条（可能重复）"""
        return list(self._hits)

    @property
    def released_chars(self) -> int:
        """累计放行的字符数"""
        return self._released_chars

    @property
    def scanned_units(self) -> int:
        """累计筛查过的完整语义单位数"""
        return self._scanned_units

    @property
    def terms_count(self) -> int:
        """载入的红线词条数量（自检用：应为非零）"""
        return len(self._terms)

    def screen(self, text: str) -> list[str]:
        """对一段文本做词法筛查。

        Args:
            text: 待筛查文本。

        Returns:
            list[str]: 命中的词条；空列表表示通过。
        """
        lowered = text.lower()
        return [t for t in self._terms if t in lowered]

    def feed(self, delta: str) -> list[str]:
        """追加一段增量，返回本次**可以上屏**的完整句子列表。

        只有完整的语义单位会被放行：缓冲到出现句末标点、或长度触及
        :data:`_STREAM_UNIT_MAX_CHARS` 上限才切出一个单位。这正是闸门有意义
        的前提 —— 拿到「你」或「你可」这样的前缀无法判断任何事。

        Args:
            delta: 本次从模型拿到的增量片段。

        Returns:
            list[str]: 通过筛查、可以直接上屏的文本片段（按顺序）。命中红线后
                一律返回空列表，且该单位及之后的内容永不放行。
        """
        if self._tripped:
            return []
        self._pending += delta
        out: list[str] = []
        while True:
            match = _SENTENCE_BOUNDARY_RE.search(self._pending)
            if match:
                cut = match.end()
            elif len(self._pending) >= _COMMA_EARLY_RELEASE_CHARS:
                # 逗号级兜底：缓冲够长但无句末标点时，在最近一个逗号处放行。
                # 这让长句（如"你好，我是AI心理咨询师，今天我们可以聊聊你的感受"）
                # 不必等到句号就能开始合成，减少首声延迟。
                comma_match = None
                for m in _COMMA_BOUNDARY_RE.finditer(self._pending):
                    comma_match = m
                if comma_match:
                    cut = comma_match.end()
                elif len(self._pending) >= _STREAM_UNIT_MAX_CHARS:
                    cut = len(self._pending)
                else:
                    break
            elif len(self._pending) >= _STREAM_UNIT_MAX_CHARS:
                cut = len(self._pending)
            else:
                break
            unit, self._pending = self._pending[:cut], self._pending[cut:]
            self._scanned_units += 1
            hits = self.screen(unit)
            if hits:
                self._tripped = True
                self._hits.extend(hits)
                logger.warning("流式闸门命中红线词: %s | 片段: %s", hits, unit[:40])
                return out
            out.append(unit)
            self._released_chars += len(unit)
        return out

    def flush(self) -> list[str]:
        """流结束时取出缓冲里剩下的残句（同样要过筛查）。

        Returns:
            list[str]: 通过筛查的尾段；已被拦截或缓冲为空时返回空列表。
        """
        if self._tripped or not self._pending:
            return []
        unit, self._pending = self._pending, ''
        self._scanned_units += 1
        hits = self.screen(unit)
        if hits:
            self._tripped = True
            self._hits.extend(hits)
            logger.warning("流式闸门命中红线词（尾段）: %s", hits)
            return []
        self._released_chars += len(unit)
        return [unit]


def _chat_stream_events(request: SmartChatRequest) -> Iterator[str]:
    """流式对话的事件生成器（同步生成器，由 Starlette 放线程池迭代）。

    事件序列见 :class:`shared.dataclasses.ChatStreamEventType`。

    Args:
        request: 智能对话请求。

    Yields:
        str: SSE 文本帧。
    """
    start = time.time()
    try:
        prep = _prepare_chat_turn(request)
    except Exception as e:  # noqa: BLE001
        logger.error("流式对话前处理失败: %s", e, exc_info=True)
        yield _sse_pack(ChatStreamEventType.ERROR, {
            'error': f"算法前处理异常: {str(e)[:200]}",
        })
        return

    streaming_allowed = (
        prep.risk_level in ("low", "medium") and not prep.crisis_detected
    )
    yield _sse_pack(ChatStreamEventType.META, {
        'risk_level': prep.risk_level,
        'dialogue_mode': prep.dialogue_mode,
        'emotion_probs': prep.emotion_result.text_emotion_probs,
        'streaming': streaming_allowed,
    })

    api_key = os.environ.get('DASHSCOPE_API_KEY', '')
    # 与非流式路径同源：是否允许以问句结尾只看"本轮该不该问"，
    # 不看"连着问了几轮"（连着追问由 prompt 语气规则降温，见 _finalize_reply）。
    # 流式已经吐出去的字收不回来，所以定稿若删掉尾问句，由下面的 REVISE 事件替换。
    allow_question = prep.ask_question and not _is_social_only(request.message)
    raw_parts: list[str] = []
    shown_parts: list[str] = []
    gate = _StreamingSafetyGate()

    if api_key:
        messages = _build_llm_messages(
            request.message,
            request.conversation_history,
            prep.emotion_result.text_emotion_probs,
            prep.risk_level,
            style=prep.user_style,
            scene_context=prep.scene_context,
            ask_question=prep.ask_question,
            memory_context=prep.memory_context,
        )
        try:
            for delta in _iter_llm_deltas(
                messages, api_key, sampling=_sampling_params(prep.risk_level)
            ):
                raw_parts.append(delta)
                if not streaming_allowed:
                    continue  # 高风险/危机：整段生成，不增量上屏
                for unit in gate.feed(delta):
                    shown_parts.append(unit)
                    yield _sse_pack(ChatStreamEventType.DELTA, {'text': unit})
                if gate.tripped:
                    logger.warning(
                        "闸门命中红线词，提前中止生成（已上屏 %d 字）",
                        len(''.join(shown_parts)),
                    )
                    break
        except Exception as e:  # noqa: BLE001
            # 流中断（网络抖动/超时）：已生成的部分照常交给审计，
            # 由审计决定最终文本，不让用户停在半句话上。
            logger.warning("流式生成中断，改用已生成内容走审计: %s", e)

    raw_reply = ''.join(raw_parts).strip()

    if not raw_reply:
        # 与 _call_llm 的兜底完全同源：没有 key、或流一个字都没拿到。
        raw_reply = _generate_empathy_reply(
            prep.emotion_result.text_emotion_probs,
            prep.risk_level,
            style=prep.user_style,
            ask_question=prep.ask_question,
        )
        if streaming_allowed:
            for unit in gate.feed(raw_reply) + gate.flush():
                shown_parts.append(unit)
                yield _sse_pack(ChatStreamEventType.DELTA, {'text': unit})
    else:
        # 与非流式路径用同一个定稿函数：本轮不许提问时删掉结尾问句，
        # 并做去套话/压长度（危机/高风险内部会跳过裁剪）。
        raw_reply = _finalize_reply(
            raw_reply, allow_question, user_message=request.message,
            conversation_history=request.conversation_history,
            risk_level=prep.risk_level,
        )
        if streaming_allowed and not gate.tripped:
            # 取出缓冲里的残句（闸门已拦下时不得再放行）。
            for unit in gate.flush():
                shown_parts.append(unit)
                yield _sse_pack(ChatStreamEventType.DELTA, {'text': unit})

    try:
        final = _audit_and_finalize(prep, request, raw_reply)
    except Exception as e:  # noqa: BLE001
        logger.error("流式对话定稿失败: %s", e, exc_info=True)
        yield _sse_pack(ChatStreamEventType.ERROR, {
            'error': f"算法定稿异常: {str(e)[:200]}",
        })
        return

    # 审计是最终权威：只要它给出的文本与已上屏的内容不一致，就整体改稿。
    # 两种情形会走到这里：(1) 闸门拦下后审计给出了安全的替代文本；
    # (2) 五轴审计（谄媚/漂移/污名化等）判定需要重写。
    shown = ''.join(shown_parts)
    if final.reply.strip() != shown.strip():
        yield _sse_pack(ChatStreamEventType.REVISE, {'text': final.reply})

    payload = final.model_dump()
    payload['text'] = payload.pop('reply')
    payload['streaming'] = streaming_allowed
    payload['latency_ms'] = round((time.time() - start) * 1000, 1)
    yield _sse_pack(ChatStreamEventType.DONE, payload)


@router.post("/smart-chat/stream", summary="智能对话（SSE 流式）")
async def smart_chat_stream(request: SmartChatRequest) -> StreamingResponse:
    """智能对话的流式版本 —— 逐句上屏，安全审计仍是最终权威。

    与非流式 ``/smart-chat`` 共享 :func:`_prepare_chat_turn` 与
    :func:`_audit_and_finalize`，因此两者的风险判定、话术规则与审计结论同源。

    事件序列（``text/event-stream``）：:

        meta → delta* → [revise] → done
        meta → error                  # 前处理失败，未产生任何文本

    关于 ``revise``：本项目的审计是整段级的，无法在吐字之前拿到结论，因此
    采用「先说、后校」。审计一旦否掉已生成内容，就用 ``revise`` 整体替换，
    而不是把改稿追加在错误文本后面 —— 消费端必须区分「追加」与「替换」。

    Args:
        request: 智能对话请求（与非流式端点同构）。

    Returns:
        StreamingResponse: SSE 响应。同步生成器由 Starlette 放线程池迭代，
            不会阻塞事件循环。
    """
    return StreamingResponse(
        _chat_stream_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # 关掉反向代理的缓冲，否则 Nginx 会攒完整段 SSE 再吐，
            # 流式就退化成一次性返回。
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# 单轮对话共享管线（非流式 / 流式两个端点共用）
# ============================================================
#
# 流式端点（/smart-chat/stream）与非流式端点（/smart-chat）必须走**同一套**
# 前处理（感知 → 风险 → 记忆 → 风格 → 状态机 → 场景检索）和**同一套**定稿
# （记忆提取 → SafetyLoop 五轴审计 → 高风险强制升级）。否则两条路径的审计
# 结论会随时间漂移 —— 改了一处忘另一处，而审计结论是本项目的安全兜底。
#
# 抽出之后，两个端点唯一的差别只剩「第 3 步怎么拿到 llm_reply」：一次拿全，
# 还是边流边拼。定稿永远由 _audit_and_finalize 完成，因此流式路径最终下发的
# 文本与审计结论，与非流式路径逐字段同源。


@dataclass
class _ChatTurnPrep:
    """单轮对话的前处理结果（模块内部结构，不作为跨模块契约）。

    定义在本模块而非 ``shared/dataclasses.py``：它只在本模块的两个端点之间
    传递，字段类型（SafetyLoop / UserStyle / EmotionResult）都属于 intervention
    与 perception 的内部实现，放进共享层会把 API 层的私有结构反向耦合进去。
    对外的跨模块契约是 :class:`shared.dataclasses.ChatStreamEvent`。

    Attributes:
        emotion_result: 感知层输出（含 evidence 与 crisis_keywords）。
        risk_level: 本轮风险等级 low / medium / high / crisis。
        crisis_detected: 是否命中危机关键词（快表或感知层结构化标记）。
        user_style: 用户表达风格。
        dialogue_mode: 对话模式 EMPATHY / SOCRATIC；仅用于审计与前端标签，
            **不参与话术生成**。
        scene_context: C2D2 场景检索上下文；空串表示相关度不足、不注入。
        scene_meta: 场景检索元信息，透出给咨询师端与审计链路。
        distortion_meta: 认知扭曲 8 分类结果，同上。
        ask_question: 本轮是否允许提出一个开放式追问。
        loop: 该用户的 SafetyLoop 实例（含对话状态机），定稿时要用。
    """
    emotion_result: Any
    risk_level: str
    crisis_detected: bool
    user_style: UserStyle
    dialogue_mode: str
    scene_context: str
    scene_meta: Optional[dict]
    distortion_meta: Optional[dict]
    ask_question: bool
    loop: SafetyLoop
    memory_context: str = ""


def _prepare_chat_turn(request: SmartChatRequest) -> _ChatTurnPrep:
    """步骤 1 ~ 2.9：文本感知 → 风险评估 → 危机快检 → 记忆召回 → 风格检测
    → 对话状态机推进 → C2D2 场景检索。

    **只读输入 + 推进该用户的对话状态机**，不产生回复文本、不写审计日志。

    Args:
        request: 智能对话请求。

    Returns:
        _ChatTurnPrep: 后续定稿所需的全部上下文。

    Raises:
        Exception: 感知/评估等依赖失败时向上抛出，由调用方决定降级策略
            （非流式端点回退降级响应，流式端点发 error 事件让 Node 层回退）。
    """
    # 步骤 1：文本情感感知
    perception = get_perception_service()
    emotion_result = perception.analyze_text(request.message)

    # 步骤 2：量表映射 → 风险评估
    scale_results = map_all(emotion_result)
    # 从量表结果推断风险等级（severity 是中文分级名，见 _severity_to_risk）
    risk_level = "low"
    for scale_name, result in scale_results.items():
        contrib = _severity_to_risk(getattr(result, 'severity', ''))
        if contrib == "high":
            risk_level = "high"
        elif contrib == "medium" and risk_level != "high":
            risk_level = "medium"

    # 步骤 2.5：独立危机关键词检测（覆盖 perception 情绪分析的盲区）
    # 双通道：本层快表 + 感知层结构化危机标记（EmotionResult.crisis_keywords）。
    # 任一命中即视为危机，避免因两侧词表细微差异而漏判。
    crisis_detected = _detect_crisis(request.message) or bool(
        getattr(emotion_result, "crisis_keywords", None)
    )
    if crisis_detected:
        risk_level = "crisis"
        # 在 evidence 中追加危机检测记录
        emotion_result.evidence.append("[危机检测] 用户输入包含危机关键词，已强制提升风险等级至 crisis")

    # 步骤 2.6：记忆召回 —— 获取与当前话题相关的用户记忆
    memory_context = ""
    try:
        memory_context = build_memory_context(
            user_id=request.user_id,
            current_topic=request.message[:50],  # 用前 50 字作为话题
            top_k=5,
        )
        if memory_context:
            emotion_result.evidence.append(f"[记忆召回] 召回 {memory_context.count(chr(10))} 条相关记忆")
    except Exception:
        pass  # 记忆召回失败不影响主流程

    # 步骤 2.7：用户风格检测 —— 根据前 3 轮消息判断回复节奏
    user_style = UserStyle.CALM
    style_params = None
    try:
        detector = StyleDetector()
        user_style = detector.detect(
            messages=request.conversation_history + [
                {"role": "user", "content": request.message}
            ],
            # ⚠️ 这里要做的是「负性情绪强度」，**不能传 max(probs)**。
            # max(probs) 是分类器置信度：说「你好」时中性概率 0.80，
            # 传进去会被判成"高情绪强度 + 短消息"→ AGITATED，
            # 于是每个寒暄轮都被要求"先安抚、不提问"。
            # 真正的强度取负面三类（悲伤/焦虑/愤怒）之和的归一化值。
            emotion_intensities=[_negative_emotion_intensity(
                emotion_result.text_emotion_probs
            )],
        )
        style_params = get_reply_style_params(user_style)
        emotion_result.evidence.append(f"[风格检测] 用户风格: {user_style.value}")
    except Exception:
        pass  # 风格检测失败不影响主流程

    # 步骤 2.8：维护对话状态机（供审计与前端标签使用）
    #
    # ⚠️ 自统一提示词起，dialogue_mode **不再参与话术生成**：EMPATHY/SOCRATIC
    # 两套 prompt 已合并为一段人本取向提示词（见 _call_llm）。
    # 这里继续计算，是因为：
    #   1. 响应体与 message.sentiment 仍要带 dialogue_mode 供前端标签与审计；
    #   2. 第六轴 socratic_timing 审计仍按它判定（auditor.py）。
    # 追问与否改由 _ask_question_this_turn（用户风格 + 风险等级）决定，
    # 这解决了两套 prompt 约束冲突造成的"每轮必反问"。
    if request.user_id not in _safety_loops:
        _safety_loops[request.user_id] = SafetyLoop(user_id=request.user_id)
    loop = _safety_loops[request.user_id]
    try:
        from intervention.state_machine import RiskLevel as _RL
        _rl_map = {"low": _RL.LOW, "medium": _RL.MEDIUM, "high": _RL.HIGH, "crisis": _RL.CRISIS}
        loop.dialog_engine.evaluate_and_update_mode(
            user_input=request.message,
            risk_level=_rl_map.get(risk_level, _RL.LOW),
        )
    except Exception:
        pass
    dialogue_mode = loop.dialog_engine.current_mode.value

    # 步骤 2.9：C2D2 场景检索 —— 从青少年认知扭曲语料库中检索与该用户当前
    # 表达最接近的真实情境与想法，注入 system prompt 作为 few-shot 参考。
    #
    # 与「认知扭曲类型」正交：场景决定提问措辞的贴合度（发生了什么），
    # 扭曲类型决定提问方向（怎么解读的）。设计依据见
    # intervention/scene_retrieval 模块 docstring。
    #
    # 仅低/中风险检索：高风险与危机必须走升级流程，不应把注意力引向
    # 认知重构（与 actions.yaml 中 COGNITIVE_RESTRUCTURE.allowed_risk 一致）。
    scene_context = ""
    scene_meta: Optional[dict] = None
    distortion_meta: Optional[dict] = None

# 本轮是否允许追问 —— 判据与实参和 _call_llm 内部完全一致。
# 提到 `if risk_level in (...)` 之前计算：高风险/危机轮次会整体跳过场景
# 检索，需要一个已定义的值返回给调用方（_ask_question_this_turn 对高风险
# 恒返回 False，提前计算不改变任何一轮的行为）。
    ask_question = _ask_question_this_turn(
        user_style,
        risk_level,
        user_message=request.message,
        negative_intensity=_negative_emotion_intensity(
            emotion_result.text_emotion_probs
        ),
    )

    if risk_level in ("low", "medium"):
        # 轴一「扭曲」：8 分类模型逐条判定。
        # 这里替换掉上一版"检索命中条目的标签统计"——模型判定精度更高
        # （官方 test acc 0.68–0.69，塌缩成有/无扭曲时 0.94）。
        distortion_hint = ""
        try:
            prediction = predict_distortion_safe(request.message)
            if prediction is not None:
                distortion_hint = format_distortion_hint(prediction)
                distortion_meta = {
                    "label": prediction.label,
                    "label_zh": prediction.label_zh,
                    "confidence": round(prediction.confidence, 4),
                    "is_distorted": prediction.is_distorted,
                    "margin": round(prediction.margin, 4),
                }
                emotion_result.evidence.append(
                    f"[认知扭曲] {prediction.label_zh}"
                    f"（置信度 {prediction.confidence:.2f}，8 分类模型）"
                    )
        except Exception:
            pass  # 分类器不可用不影响主流程（prompt 会退回检索标签投票）

        # 轴二「情境」：检索真实青少年语料，给追问提供**用户自己提过的**
        # 处境参照。相关度不足时 build_scene_context 返回空串（阈值见
        # intervention/scene_retrieval/index.py 的 MIN_SCENE_SCORE），
        # 此时不注入 —— 宁可不给参照，也不让模型对用户没说过的处境表达理解。
        #
        # socratic 形参已不再表示"苏格拉底模式"，而是"本轮是否允许追问"，
        # 与 _ask_question_this_turn 同源，保证注入的 [使用要求] 与
        # system prompt 的追问规则不打架。
        scene_context = build_scene_context(
            request.message,
            # 收紧注入：只给最贴合的 1 个情境、每个情境 1 条想法。
            # 实测「我好难受」在 max_scenes=3 时会注入 6 条无关语料
            # （怀孕/背痛/疫情），模型虽然没照搬，但也毫无增益，
            # 只是白烧 token。few-shot 的价值在精度不在条数。
            top_k=3,
            max_per_scene=1,
            max_scenes=1,
            socratic=ask_question,
            distortion_hint=distortion_hint,
        )
        # 纯寒暄/客套一律不注入情境。
        #
        # 2026-09-26 用户截图：ta 只说了"你好"，AI 回"你今天看起来有点没精神啊？"
        # —— 那句"没精神"就是从检索到的语料里搬过来的。用户没有任何内容时，
        # 任何"相似处境"都是硬塞，只会让模型替 ta 编状态。
        # 阈值挡不住这种情况：相似度只跟"像不像"有关，跟"该不该说"无关。
        if _is_social_only(request.message):
            if scene_context:
                emotion_result.evidence.append(
                    "[场景检索] 本轮为寒暄/客套，已跳过情境注入"
                )
            scene_context = ""
        if scene_context:
            scene_hits = scene_context.count("- 情境：")
            scene_meta = {"scenes": scene_hits, "injected": True}
            emotion_result.evidence.append(
                f"[场景检索] 命中 {scene_hits} 个相近情境，已注入 prompt"
            )

    return _ChatTurnPrep(
        emotion_result=emotion_result,
        risk_level=risk_level,
        crisis_detected=crisis_detected,
        user_style=user_style,
        dialogue_mode=dialogue_mode,
        scene_context=scene_context,
        scene_meta=scene_meta,
        distortion_meta=distortion_meta,
        ask_question=ask_question,
        loop=loop,
        memory_context=memory_context,
    )


def _audit_and_finalize(
    prep: _ChatTurnPrep,
    request: SmartChatRequest,
    llm_reply: str,
) -> SmartChatResponse:
    """步骤 3.5 ~ 5：记忆提取 → SafetyLoop 五轴审计 → 高风险强制升级 → 构建响应。

    这是**唯一**的定稿出口：无论回复是一次性生成的还是逐句流式拼出来的，最终
    对用户生效的文本都由这里给出。流式端点把拼接结果交给它，再把返回值作为
    DONE 事件的载荷下发 —— 审计因此仍然是最终权威。

    Args:
        prep: :func:`_prepare_chat_turn` 的结果。
        request: 智能对话请求。
        llm_reply: 待审计的模型回复原文（流式路径传拼接后的完整文本）。

    Returns:
        SmartChatResponse: 与非流式端点完全同构的响应。
    """
    # 以下正文自 smart_chat 原样搬入：把 prep 的字段绑定为局部名，
    # 让下游代码逐字不变（搬运不引入任何静默的行为差异）。
    emotion_result = prep.emotion_result
    risk_level = prep.risk_level
    crisis_detected = prep.crisis_detected
    scene_meta = prep.scene_meta
    distortion_meta = prep.distortion_meta
    loop = prep.loop

    # 步骤 3.5：记忆提取 —— 从对话中提取并存储长期记忆
    try:
        extractor = MemoryExtractor()
        extractor.extract_and_store(
            user_id=request.user_id,
            conversation_history=request.conversation_history + [
                {"role": "user", "content": request.message}
            ],
            conversation_id=request.session_id or "",
            max_facts=3,
        )
    except Exception:
        pass  # 记忆提取失败不影响主流程

    # 步骤 4：SafetyLoop 安全审计
    turn_count = len(request.conversation_history) // 2
    is_crisis = risk_level in ("high", "crisis")

    context = {
        "dialog_state": "CRISIS" if is_crisis else "EXPLORE",
        "turn_count": turn_count,
        "crisis_turns": 1 if is_crisis else 0,
        "risk_level": risk_level,
        "conversation_history": request.conversation_history,
        "llm_confidence": emotion_result.confidence,
        "current_topic": "",
        "is_teen": True,  # 青少年平台默认启用青少年检测
    }

    safety_result = loop.process(
        user_input=request.message,
        llm_response=llm_reply,
        context=context,
    )

    # 步骤 4.5：高风险强制升级
    # ⚠️ 比赛 Demo 为模拟实现，生产环境需接入真实危机干预流程
    #
    # AGENTS.md 安全红线：risk_level 为 high 或 crisis 时必须触发升级流程。
    # SafetyLoop 仅在「审计不通过且严重度为 SEVERE」时才调用 escalate()；
    # 若 LLM 回复恰好通过审计，高风险会被静默降级为普通回复 —— 这里补强制路径。
    escalation_status = ""
    escalation_alert_id = ""
    if risk_level in ("high", "crisis"):
        handled = safety_result.action_taken in (
            "crisis_escalate", "crisis_escalate_fallback",
        )
        if handled:
            escalation_status = "handled_by_safety_loop"
            escalation_alert_id = safety_result.escalation_alert_id
        else:
            try:
                from escalation.crisis import escalate as crisis_escalate
                from shared.dataclasses import CrisisAlert

                alert = CrisisAlert(
                    user_id=request.user_id,
                    risk_score=1.0 if risk_level == "crisis" else 0.8,
                    trigger_evidence=list(emotion_result.evidence),
                    timestamp=time.time(),
                    recommended_action="立即人工介入与安全确认",
                )
                esc = crisis_escalate(alert)
                escalation_status = esc.status.value
                escalation_alert_id = esc.alert_id
            except Exception as e:  # noqa: BLE001
                escalation_status = "failed"
                logger.error("高风险强制升级失败: %s", e)
            emotion_result.evidence.append(
                f"[强制升级] risk_level={risk_level}，已触发升级流程"
                f"（status={escalation_status}）"
                )

    # 步骤 5：构建响应
    requires_escalation = (
        safety_result.severity == "severe"
        or risk_level in ("high", "crisis")
        or crisis_detected
    )
    audit_passed = safety_result.severity == "none" and not crisis_detected
    # 与步骤 2.8 同源：仅用于响应/审计/前端标签，不影响话术
    dialogue_mode = loop.dialog_engine.current_mode.value

    return SmartChatResponse(
        reply=safety_result.final_response,
        dialog_state=safety_result.state_after,
        dialogue_mode=dialogue_mode,
        action_type=safety_result.action_taken,
        risk_level=risk_level,
        audit_passed=audit_passed,
        requires_escalation=requires_escalation,
        emotion_probs=emotion_result.text_emotion_probs,
        evidence=emotion_result.evidence,
        fallback=False,
        scene_retrieval=scene_meta,
        cognitive_distortion=distortion_meta,
    )


@router.post("/smart-chat", response_model=SmartChatResponse, summary="智能对话")
async def smart_chat(request: SmartChatRequest) -> SmartChatResponse:
    """智能对话 —— 感知 → 评估 → 回复生成 → 安全审计 完整闭环

    流程：
        1. 文本情感感知 → EmotionResult（:func:`_prepare_chat_turn`）
        2. 多任务风险评估 → RiskAssessment（同上）
        3. 基于风险等级生成共情回复（一次拿全）
        4. SafetyLoop 安全审计（:func:`_audit_and_finalize`）
        5. 返回安全回复

    需要逐句上屏时用 ``/smart-chat/stream``；两者共享前处理与定稿，
    唯一差别是第 3 步的取文本方式。
    """
    try:
        prep = _prepare_chat_turn(request)

        # 与 _audit_and_finalize 相同的绑定手法：把 prep 的字段绑成局部名，
        # 让下面这段自旧 smart_chat 搬来的代码逐字不变。
        emotion_result = prep.emotion_result
        risk_level = prep.risk_level
        user_style = prep.user_style
        dialogue_mode = prep.dialogue_mode
        scene_context = prep.scene_context

        # 步骤 3：调用千问大模型生成共情回复（失败时回退模板）
        llm_reply = _call_llm(
            request.message,
            request.conversation_history,
            emotion_result.text_emotion_probs,
            risk_level,
            style=user_style,
            dialogue_mode=dialogue_mode,
            scene_context=scene_context,
            memory_context=prep.memory_context,
        )

        return _audit_and_finalize(prep, request, llm_reply)

    except Exception as e:
        # 降级模式
        return SmartChatResponse(
            reply="",
            dialog_state="INIT",
            action_type="OPEN_QUESTION",
            risk_level="low",
            audit_passed=True,
            requires_escalation=False,
            emotion_probs=[0.2, 0.2, 0.2, 0.2, 0.2],
            evidence=[f"算法处理异常，降级模式: {str(e)[:100]}"],
            fallback=True,
        )


# ============================================================
# CBT 认知重构记录端点
# ============================================================

class CBTRecordRequest(BaseModel):
    """CBT 记录查询请求"""
    conversation_id: str = Field(..., description="对话 ID")


class CBTRecordResponse(BaseModel):
    """CBT 记录响应"""
    conversation_id: str
    status: str
    automatic_thought: str
    distortion_type: str
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    alternative_thought: str
    initial_emotion_intensity: Optional[int]
    final_emotion_intensity: Optional[int]
    action_plan: str
    depth_reached: str
    turns_in_socratic: int
    summary: str


@router.post("/cbt-record", response_model=CBTRecordResponse, summary="获取 CBT 认知重构记录")
async def get_cbt_restructuring_record(request: CBTRecordRequest):
    """获取指定对话的 CBT 认知重构记录

    当对话走过完整 DEEP 流程（SHALLOW → MEDIUM → DEEP）时，
    生成结构化记录供咨询师端查看。

    返回字段：
        - status: 记录状态（COMPLETE/PARTIAL/NOT_STARTED）
        - automatic_thought: 识别的自动思维
        - distortion_type: 认知扭曲类型
        - supporting_evidence: 支持证据
        - contradicting_evidence: 反对证据
        - alternative_thought: 替代想法
        - initial_emotion_intensity: 初始情绪强度
        - final_emotion_intensity: 最终情绪强度
        - action_plan: 行动计划
        - depth_reached: 达到的最深深度
        - turns_in_socratic: 苏格拉底模式总轮次
        - summary: 给咨询师的总结
    """
    record = get_cbt_record(request.conversation_id)

    if record is None:
        # 返回空记录
        return CBTRecordResponse(
            conversation_id=request.conversation_id,
            status="NOT_STARTED",
            automatic_thought="",
            distortion_type="",
            supporting_evidence=[],
            contradicting_evidence=[],
            alternative_thought="",
            initial_emotion_intensity=None,
            final_emotion_intensity=None,
            action_plan="",
            depth_reached="",
            turns_in_socratic=0,
            summary="本次对话未进入苏格拉底引导模式。",
        )

    return CBTRecordResponse(
        conversation_id=record.conversation_id,
        status=record.status.value,
        automatic_thought=record.automatic_thought,
        distortion_type=record.distortion_type,
        supporting_evidence=record.supporting_evidence,
        contradicting_evidence=record.contradicting_evidence,
        alternative_thought=record.alternative_thought,
        initial_emotion_intensity=record.initial_emotion_intensity,
        final_emotion_intensity=record.final_emotion_intensity,
        action_plan=record.action_plan,
        depth_reached=record.depth_reached,
        turns_in_socratic=record.turns_in_socratic,
        summary=record.summary,
    )


class UserCBTRecordsRequest(BaseModel):
    """用户 CBT 记录查询请求"""
    user_id: str = Field(..., description="用户 ID")


@router.post("/cbt-records", summary="获取用户的所有 CBT 认知重构记录")
async def get_user_cbt_records(request: UserCBTRecordsRequest):
    """获取指定用户的所有 CBT 认知重构记录

    用于咨询师端查看用户的历史认知重构进展。
    """
    records = get_user_cbt_records(request.user_id)

    return {
        "user_id": request.user_id,
        "total": len(records),
        "records": [
            {
                "conversation_id": r.conversation_id,
                "status": r.status.value,
                "depth_reached": r.depth_reached,
                "turns_in_socratic": r.turns_in_socratic,
                "summary": r.summary,
            }
            for r in records
        ],
    }


# ============================================================
# 用户长期记忆管理端点
# ============================================================

class MemoryQueryRequest(BaseModel):
    """记忆查询请求"""
    user_id: str = Field(..., description="用户 ID")
    topic: str = Field("", description="当前话题（用于相关性匹配）")
    top_k: int = Field(5, description="返回前 k 条记忆")


class MemoryDeleteRequest(BaseModel):
    """记忆删除请求"""
    user_id: str = Field(..., description="用户 ID")
    memory_id: str = Field(..., description="记忆 ID")


class MemoryClearRequest(BaseModel):
    """清空用户所有记忆请求"""
    user_id: str = Field(..., description="用户 ID")


@router.post("/memory/query", summary="查询用户记忆")
async def query_user_memories(request: MemoryQueryRequest):
    """查询用户的相关记忆

    根据当前话题返回最相关的记忆，用于注入对话上下文。
    """
    store = get_memory_store()
    memories = store.get_relevant_memories(
        user_id=request.user_id,
        current_topic=request.topic,
        top_k=request.top_k,
    )

    return {
        "user_id": request.user_id,
        "total": len(memories),
        "memories": [
            {
                "id": m.id,
                "fact": m.fact,
                "category": m.category,
                "confidence": m.confidence,
                "reference_count": m.reference_count,
                "created_at": m.created_at,
                "last_referenced": m.last_referenced,
            }
            for m in memories
        ],
    }


@router.post("/memory/list", summary="获取用户所有记忆")
async def list_user_memories(request: MemoryQueryRequest):
    """获取用户的所有记忆（用于记忆管理界面）"""
    store = get_memory_store()
    memories = store.get_all_memories(request.user_id)

    return {
        "user_id": request.user_id,
        "total": len(memories),
        "memories": [
            {
                "id": m.id,
                "fact": m.fact,
                "category": m.category,
                "confidence": m.confidence,
                "reference_count": m.reference_count,
                "created_at": m.created_at,
                "last_referenced": m.last_referenced,
            }
            for m in memories
        ],
    }


@router.post("/memory/delete", summary="删除单条记忆")
async def delete_memory(request: MemoryDeleteRequest):
    """删除用户的一条记忆

    用户可随时删除自己的记忆。
    """
    store = get_memory_store()
    deleted = store.delete_memory(request.user_id, request.memory_id)

    return {
        "deleted": deleted,
        "memory_id": request.memory_id,
    }


@router.post("/memory/clear", summary="清空用户所有记忆")
async def clear_user_memories(request: MemoryClearRequest):
    """清空用户的所有记忆"""
    store = get_memory_store()
    count = store.clear_user_memories(request.user_id)

    return {
        "user_id": request.user_id,
        "deleted_count": count,
    }


@router.post("/memory/decay", summary="执行记忆降权")
async def decay_old_memories():
    """对超期未引用的记忆执行降权

    超过 90 天未被引用的记忆，置信度降低 50%。
    """
    store = get_memory_store()
    count = store.decay_old_memories(days=90)

    return {
        "decayed_count": count,
    }
