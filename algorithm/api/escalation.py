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
from escalation.proactive_care import (
    get_proactive_care_manager,
    TriggerCondition,
    TriggerSignal,
)


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


# ============================================================
# 主动关怀 API
# ============================================================

class ConsentRequest(BaseModel):
    """用户授权请求"""
    user_id: str = Field(..., description="用户 ID")
    enable: bool = Field(..., description="是否开启主动关怀")


class ConsentResponse(BaseModel):
    """授权响应"""
    user_id: str
    enabled: bool
    message: str


class TriggerEvaluationRequest(BaseModel):
    """触发条件评估请求"""
    user_id: str = Field(..., description="用户 ID")
    baseline_deviations: Optional[list[float]] = Field(None, description="最近 N 天的基线偏离值")
    last_app_open: Optional[float] = Field(None, description="上次打开 App 的时间戳")
    historical_crisis: bool = Field(False, description="是否有历史危机信号")


class CareMessageResponse(BaseModel):
    """关怀消息响应"""
    message_id: str
    content: str
    trigger_condition: str
    should_escalate: bool
    referenced_memory_id: str


@router.post("/proactive-care/consent", response_model=ConsentResponse)
async def api_proactive_care_consent(request: ConsentRequest) -> ConsentResponse:
    """用户开启/关闭主动关怀

    用户首次使用 App 时询问是否开启。
    用户可随时在设置中关闭。
    """
    manager = get_proactive_care_manager()

    if request.enable:
        manager.grant_consent(request.user_id)
        return ConsentResponse(
            user_id=request.user_id,
            enabled=True,
            message="已开启主动关怀，我们会在你需要的时候发送温和的问候。",
        )
    else:
        manager.revoke_consent(request.user_id)
        return ConsentResponse(
            user_id=request.user_id,
            enabled=False,
            message="已关闭主动关怀。如需帮助，随时联系我们。",
        )


@router.get("/proactive-care/consent/{user_id}", response_model=ConsentResponse)
async def api_get_proactive_care_consent(user_id: str) -> ConsentResponse:
    """查询用户主动关怀授权状态"""
    manager = get_proactive_care_manager()
    enabled = manager.has_consent(user_id)

    return ConsentResponse(
        user_id=user_id,
        enabled=enabled,
        message="已开启主动关怀" if enabled else "未开启主动关怀",
    )


@router.post("/proactive-care/evaluate", response_model=list[CareMessageResponse])
async def api_evaluate_proactive_care(request: TriggerEvaluationRequest) -> list[CareMessageResponse]:
    """评估触发条件并生成关怀消息

    每天最多评估一次。
    每天最多 1 条关怀消息。
    涉及危机信号必须转人工。
    """
    manager = get_proactive_care_manager()

    # 评估触发条件
    triggers = manager.evaluate_triggers(
        user_id=request.user_id,
        baseline_deviations=request.baseline_deviations,
        last_app_open=request.last_app_open,
        historical_crisis=request.historical_crisis,
    )

    responses = []
    for trigger in triggers:
        # 检查是否需要转人工
        should_escalate = manager.should_escalate_to_human(trigger)

        # 生成关怀消息
        care_msg = manager.generate_care_message(request.user_id, trigger)

        if care_msg:
            responses.append(CareMessageResponse(
                message_id=care_msg.id,
                content=care_msg.content,
                trigger_condition=care_msg.trigger_condition.value,
                should_escalate=should_escalate,
                referenced_memory_id=care_msg.referenced_memory_id,
            ))

    return responses
