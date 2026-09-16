"""
紧急升级 API —— /emergency/escalate 和 /emergency/report

⚠️ 声明：比赛 Demo 为模拟实现，非真实紧急呼叫系统。

接口：
    - POST /emergency/escalate: 触发危机升级
    - GET  /emergency/report: 获取审计报告
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from shared.dataclasses import CrisisAlert, EscalationChannel
from escalation.crisis import escalate, generate_escalation_report
from escalation.audit_log import get_audit_store


router = APIRouter(prefix="/emergency", tags=["紧急升级"])


# ============================================================
# 请求/响应模型
# ============================================================

class EscalateRequest(BaseModel):
    """危机升级请求"""
    user_id: str = Field(..., description="触发警报的用户 ID")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="综合风险评分 [0, 1]")
    trigger_evidence: list[str] = Field(..., description="触发证据列表")
    recommended_action: str = Field(default="立即联系专业危机干预人员", description="建议干预动作")
    channels: Optional[list[str]] = Field(
        default=None,
        description="通知通道列表（webhook/in_app/console），默认全部",
    )


class EscalateResponse(BaseModel):
    """危机升级响应"""
    alert_id: str
    status: str
    channels_notified: list[str]
    assigned_counselor: str
    response_time_seconds: float
    within_target: bool
    message: str


class ReportResponse(BaseModel):
    """审计报告响应"""
    report: str
    total_entries: int
    chain_valid: bool


# ============================================================
# 接口实现
# ============================================================

@router.post("/escalate", response_model=EscalateResponse)
async def api_escalate(request: EscalateRequest) -> EscalateResponse:
    """触发危机升级

    ⚠️ 比赛 Demo 为模拟实现。

    将危机警报升级为多通道紧急响应：
    - webhook: 向本地 mock server 发送通知
    - in_app: 站内通知
    - console: 控制台告警

    响应时间目标：< 30 秒
    """
    # 构建 CrisisAlert
    alert = CrisisAlert(
        user_id=request.user_id,
        risk_score=request.risk_score,
        trigger_evidence=request.trigger_evidence,
        timestamp=time.time(),
        recommended_action=request.recommended_action,
    )

    # 解析通道
    channels = None
    if request.channels:
        channels = []
        for ch in request.channels:
            try:
                channels.append(EscalationChannel(ch))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效通道: {ch}，可选: webhook/in_app/console",
                )

    # 执行升级
    result = escalate(alert, channels=channels)

    return EscalateResponse(
        alert_id=result.alert_id,
        status=result.status.value,
        channels_notified=[c.value for c in result.channels_notified],
        assigned_counselor=result.assigned_counselor,
        response_time_seconds=result.response_time_seconds,
        within_target=result.response_time_seconds < 30.0,
        message=(
            f"危机升级完成，响应时间 {result.response_time_seconds:.3f}s"
            if result.response_time_seconds < 30.0
            else f"⚠️ 响应时间 {result.response_time_seconds:.3f}s 超过 30s 目标！"
        ),
    )


@router.get("/report", response_model=ReportResponse)
async def api_report() -> ReportResponse:
    """获取危机升级审计报告

    ⚠️ 比赛 Demo 为模拟实现。
    """
    audit_store = get_audit_store()
    report = generate_escalation_report(audit_store)

    return ReportResponse(
        report=report,
        total_entries=audit_store.total_entries,
        chain_valid=audit_store.verify_chain(),
    )
