"""
干预层路由 —— AI 心理干预 API 端点

包含：
    1. 安全审计事件查询（咨询师端实时审计面板）
    2. 结构化策略生成（Action → 回复文本）
    3. CBT 认知重构会话管理
"""
from __future__ import annotations

import time
import random
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from intervention.strategy import generate_reply, get_generator, Action, ActionType
from intervention.auditor import SafetyAuditor, AuditContext

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
