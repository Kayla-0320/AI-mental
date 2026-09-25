"""
干预层路由 —— AI 心理干预 API 端点

包含：
    1. 安全审计事件查询（咨询师端实时审计面板）
    2. 结构化策略生成（Action → 回复文本）
    3. CBT 认知重构会话管理
"""
from __future__ import annotations

import os
import time
import random
import requests
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from intervention.strategy import generate_reply, get_generator, Action, ActionType
from intervention.auditor import SafetyAuditor, AuditContext
from intervention.safety_loop import SafetyLoop, SafetyLoopResult
from intervention.cbt.cbt_record import (
    get_cbt_record,
    get_user_cbt_records,
    CBTRestructuringRecord,
)
from intervention.style_detector import StyleDetector, UserStyle, get_reply_style_params
from perception.perception_service import get_perception_service
from assessment.scale import map_all
from assessment.user_memory import get_memory_store
from assessment.memory_extractor import MemoryExtractor, build_memory_context

router = APIRouter(prefix="/intervention", tags=["干预层"])


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


# 简易共情回复生成器（不依赖外部 LLM 服务）
_EMPATHY_TEMPLATES = {
    "low": [
        "我听到你说的了。能再多告诉我一些你的感受吗？",
        "谢谢你愿意分享。你最近还有什么让你开心的事情吗？",
        "我理解你的感受。你觉得什么事情最让你困扰？",
    ],
    "medium": [
        "听起来你现在承受着不少压力。你愿意说说是什么让你感到不安吗？",
        "我能感受到你的痛苦。这些感受是正常的，你不需要独自承受。",
        "你正在经历的事情确实不容易。让我们一起想想有什么方法可以帮助你。",
    ],
    "high": [
        "我能感受到你现在非常痛苦。你的感受很重要，你值得被倾听和帮助。",
        "谢谢你告诉我这些。你安全吗？有没有想过伤害自己？",
        "你现在经历的这些一定很困难。我想让你知道，有专业的人员可以帮助你。",
    ],
    "crisis": [
        "我非常担心你的安全。你现在是否处于危险中？请立即拨打24小时心理援助热线：400-161-9995。",
        "你的安全是最重要的。如果你正在考虑伤害自己，请立刻联系信任的人或拨打急救电话。",
    ],
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


def _call_llm(user_message: str, conversation_history: list, emotion_probs: list[float], risk_level: str, style: Optional[UserStyle] = None, dialogue_mode: str = "EMPATHY") -> str:
    """调用千问大模型生成共情回复，失败时回退模板"""
    api_key = os.environ.get('DASHSCOPE_API_KEY', '')
    if not api_key:
        return _generate_empathy_reply(emotion_probs, risk_level, style=style)

    emotion_labels = ['快乐', '悲伤', '焦虑', '愤怒', '中性']
    dominant = emotion_labels[max(range(len(emotion_probs)), key=lambda i: emotion_probs[i])]
    confidence = max(emotion_probs)

    style_desc = {
        UserStyle.INTROVERTED: '用户偏内向安静，回复要简短温和，给足空间',
        UserStyle.EXTROVERTED: '用户偏外向健谈，回复可以活泼一些，多互动',
        UserStyle.AGITATED: '用户当前情绪激动/烦躁，回复要简短、先安抚再引导',
        UserStyle.CALM: '用户当前较平静，可以深入探讨',
    }.get(style, '')

    system_prompt = f"""你是一位温暖、专业的AI心理咨询师，服务于青少年心理健康平台。

核心原则：
1. 共情优先：先认可感受，再探索原因
2. 不给建议：用反问引导自我觉察，不说"你应该"
3. 简短自然：每次回复1-2句话，像朋友聊天
4. 危机敏感：如检测到自伤/自杀信号，温柔引导求助

当前用户情绪分析：主导情绪={dominant}（置信度{confidence:.0%}），风险等级={risk_level}
{style_desc}

请用温暖、自然的语气回复用户，不要使用模板化语言。""" if dialogue_mode != "SOCRATIC" else f"""你是一个"话釶树洞"——温暖但不给建议的倾听者，擅长苏格拉底式反问。

核心原则：
1. 绝不给建议：不说"你应该"、"我建议你"、"你可以试试"
2. 苏格拉底式反问：用反问引导用户自我剖析
   - 用户说"我很失败" → "这个结论，是你自己得出的，还是别人替你盖章的？"
   - 用户说"没人理解我" → "你曾经尝试过让他们理解吗？还是你已经开始替他们做决定了？"
3. 共情但不沉溺：先简短认可感受（1句），然后立刻抛出反问
4. 保持对话感：像朋友聊天，语气轻松自然
5. 适度话釶：可以多说几句，但每段话最后都要落脚到一个反问
6. 危机敏感：如检测到自伤/自杀信号，温柔但直接地引导寻求专业帮助

当前用户情绪分析：主导情绪={dominant}（置信度{confidence:.0%}），风险等级={risk_level}
{style_desc}

回复格式：先共情（1-2句），再反问（1个核心问题）。总长度控制在100字以内。"""

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    # 加入最近4轮对话历史
    for msg in conversation_history[-8:]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": user_message})

    try:
        resp = requests.post(
            'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
            json={'model': 'qwen-turbo', 'messages': messages, 'temperature': 0.85, 'max_tokens': 200},
            timeout=15,
        )
        if resp.ok:
            data = resp.json()
            reply = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            if reply.strip():
                return reply.strip()
    except Exception:
        pass

    # LLM 调用失败，回退模板
    return _generate_empathy_reply(emotion_probs, risk_level, style=style)


def _generate_empathy_reply(
    emotion_probs: list[float],
    risk_level: str,
    memory_context: str = "",
    style: Optional[UserStyle] = None,
) -> str:
    """基于情绪和风险等级生成共情回复（模板降级）"""
    # 根据风格选择模板
    if style and risk_level not in ("crisis", "high"):
        style_templates = _STYLE_REPLY_TEMPLATES.get(style, _EMPATHY_TEMPLATES.get(risk_level, _EMPATHY_TEMPLATES["low"]))
        idx = hash(risk_level) % len(style_templates)
        return style_templates[idx]

    templates = _EMPATHY_TEMPLATES.get(risk_level, _EMPATHY_TEMPLATES["low"])
    idx = hash(risk_level) % len(templates)
    base_reply = templates[idx]

    # 如果有记忆上下文，追加记忆引用提示
    if memory_context and risk_level not in ("crisis", "high"):
        return base_reply

    return base_reply


# 危机关键词快速检测（独立于 perception 服务，覆盖直接/间接/隐晦表达）
_CRISIS_KEYWORDS = [
    # ── 直接自杀/自伤意图 ──
    '自杀', '自残', '自伤', '不想活', '想死', '去死',
    '跳楼', '割腕', '划手', '划手臂', '服毒', '上吊',
    '活不下去', '死了算了', '不想活了', '不想醒来',
    '伤害自己', '吃药自杀', '结束生命',
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


@router.post("/smart-chat", response_model=SmartChatResponse, summary="智能对话")
async def smart_chat(request: SmartChatRequest) -> SmartChatResponse:
    """智能对话 —— 感知 → 评估 → 回复生成 → 安全审计 完整闭环

    流程：
        1. 文本情感感知 → EmotionResult
        2. 多任务风险评估 → RiskAssessment
        3. 基于风险等级生成共情回复
        4. SafetyLoop 安全审计
        5. 返回安全回复
    """
    try:
        # 步骤 1：文本情感感知
        perception = get_perception_service()
        emotion_result = perception.analyze_text(request.message)

        # 步骤 2：量表映射 → 风险评估
        scale_results = map_all(emotion_result)
        # 从量表结果推断风险等级
        risk_level = "low"
        for scale_name, result in scale_results.items():
            severity = result.severity if hasattr(result, 'severity') else "mild"
            if severity in ("moderate", "severe"):
                risk_level = "high"
            elif severity == "mild" and risk_level != "high":
                risk_level = "medium"

        # 步骤 2.5：独立危机关键词检测（覆盖 perception 情绪分析的盲区）
        crisis_detected = _detect_crisis(request.message)
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
                emotion_intensities=[max(emotion_result.text_emotion_probs)],
            )
            style_params = get_reply_style_params(user_style)
            emotion_result.evidence.append(f"[风格检测] 用户风格: {user_style.value}")
        except Exception:
            pass  # 风格检测失败不影响主流程

        # 步骤 2.8：提前初始化状态机，预判对话模式用于 LLM prompt
        if request.user_id not in _safety_loops:
            _safety_loops[request.user_id] = SafetyLoop(user_id=request.user_id)
        loop = _safety_loops[request.user_id]
        # 用当前输入和风险预判对话模式（LLM 需要据此选择 prompt）
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

        # 步骤 3：调用千问大模型生成共情回复（失败时回退模板）
        llm_reply = _call_llm(
            request.message,
            request.conversation_history,
            emotion_result.text_emotion_probs,
            risk_level,
            style=user_style,
            dialogue_mode=dialogue_mode,
        )

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

        # 步骤 5：构建响应
        requires_escalation = safety_result.severity == "severe" or crisis_detected
        audit_passed = safety_result.severity == "none" and not crisis_detected
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
        )

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
