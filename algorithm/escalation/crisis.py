"""
危机升级引擎 —— 将 CrisisAlert 升级为多通道紧急响应

⚠️ 声明：比赛 Demo 为模拟实现，非真实紧急呼叫系统。

核心接口：
    - escalate(alert) -> EscalationResult

三通道通知：
    1. webhook: 向本地 mock server 发送 HTTP POST
    2. in_app: 站内通知（模拟写入内存队列）
    3. console: 控制台告警（打印到 stderr）

响应时间目标：< 30 秒触发所有通知
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from shared.dataclasses import (
    CrisisAlert,
    EscalationChannel,
    EscalationResult,
    EscalationStatus,
)

from .audit_log import AuditLogStore, AuditEventType, get_audit_store
from .scheduler import CounselorScheduler, get_scheduler


# ============================================================
# 响应时间目标
# ============================================================

RESPONSE_TIME_TARGET_SECONDS = 30.0  # 升级响应时间目标


# ============================================================
# 通知通道处理器
# ============================================================

@dataclass
class NotificationResult:
    """单通道通知结果"""
    channel: EscalationChannel
    success: bool
    message: str = ""
    latency_ms: float = 0.0


def _notify_webhook(alert: CrisisAlert, alert_id: str) -> NotificationResult:
    """Webhook 通道 —— 向本地 mock server 发送 POST

    ⚠️ 比赛 Demo 为模拟实现，不实际发送 HTTP 请求。
    真实部署应使用 requests 库发送 POST 到配置的 webhook URL。

    Args:
        alert: 危机警报
        alert_id: 升级事件 ID

    Returns:
        NotificationResult: 通知结果
    """
    start = time.time()

    # 模拟 webhook payload
    payload = {
        "alert_id": alert_id,
        "user_id": alert.user_id,
        "risk_score": alert.risk_score,
        "trigger_evidence": alert.trigger_evidence,
        "recommended_action": alert.recommended_action,
        "timestamp": alert.timestamp,
    }

    # ⚠️ Demo 模拟：不实际发送 HTTP
    # 真实实现：
    # import requests
    # response = requests.post(WEBHOOK_URL, json=payload, timeout=10)
    # success = response.status_code == 200

    latency = (time.time() - start) * 1000

    print(
        f"[Webhook] ⚠️ 危机警报 → mock server "
        f"(alert_id={alert_id}, risk={alert.risk_score:.2f})",
        file=sys.stderr,
    )

    return NotificationResult(
        channel=EscalationChannel.WEBHOOK,
        success=True,
        message=f"Webhook 已发送到 mock server: {json.dumps(payload, ensure_ascii=False)[:100]}...",
        latency_ms=round(latency, 2),
    )


def _notify_in_app(alert: CrisisAlert, alert_id: str) -> NotificationResult:
    """站内通知通道 —— 模拟写入内存通知队列

    ⚠️ 比赛 Demo 为模拟实现，仅打印到控制台。
    真实部署应写入消息队列（如 Redis Pub/Sub）。

    Args:
        alert: 危机警报
        alert_id: 升级事件 ID

    Returns:
        NotificationResult: 通知结果
    """
    start = time.time()

    notification = {
        "type": "crisis_alert",
        "alert_id": alert_id,
        "user_id": alert.user_id,
        "title": "⚠️ 紧急危机警报",
        "body": (
            f"用户 {alert.user_id} 触发危机警报，"
            f"风险评分 {alert.risk_score:.2f}。"
            f"建议操作：{alert.recommended_action}"
        ),
        "priority": "critical",
        "timestamp": time.time(),
    }

    # ⚠️ Demo 模拟：仅打印
    # 真实实现：redis.publish("notifications", json.dumps(notification))

    latency = (time.time() - start) * 1000

    print(
        f"[InApp] 📢 站内通知 → 用户 {alert.user_id} "
        f"(alert_id={alert_id})",
        file=sys.stderr,
    )

    return NotificationResult(
        channel=EscalationChannel.IN_APP,
        success=True,
        message=f"站内通知已发送: {notification['title']}",
        latency_ms=round(latency, 2),
    )


def _notify_console(alert: CrisisAlert, alert_id: str) -> NotificationResult:
    """控制台告警通道 —— 输出到 stderr

    Args:
        alert: 危机警报
        alert_id: 升级事件 ID

    Returns:
        NotificationResult: 通知结果
    """
    start = time.time()

    # 控制台告警格式
    alert_banner = (
        f"\n{'='*60}\n"
        f"🚨 危机告警 🚨\n"
        f"{'='*60}\n"
        f"Alert ID:     {alert_id}\n"
        f"User ID:      {alert.user_id}\n"
        f"Risk Score:   {alert.risk_score:.2f}\n"
        f"Evidence:     {'; '.join(alert.trigger_evidence[:3])}\n"
        f"Action:       {alert.recommended_action}\n"
        f"Timestamp:    {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*60}\n"
    )

    print(alert_banner, file=sys.stderr)

    latency = (time.time() - start) * 1000

    return NotificationResult(
        channel=EscalationChannel.CONSOLE,
        success=True,
        message="控制台告警已输出",
        latency_ms=round(latency, 2),
    )


# ============================================================
# 通道处理器注册表
# ============================================================

_CHANNEL_HANDLERS = {
    EscalationChannel.WEBHOOK: _notify_webhook,
    EscalationChannel.IN_APP: _notify_in_app,
    EscalationChannel.CONSOLE: _notify_console,
}


# ============================================================
# 核心升级引擎
# ============================================================

def escalate(
    alert: CrisisAlert,
    channels: Optional[list[EscalationChannel]] = None,
    audit_store: Optional[AuditLogStore] = None,
    scheduler: Optional[CounselorScheduler] = None,
) -> EscalationResult:
    """危机升级 —— 将 CrisisAlert 升级为多通道紧急响应

    ⚠️ 比赛 Demo 为模拟实现。

    流程：
        1. 记录警报接收审计日志
        2. 分配可用咨询师
        3. 依次触发三通道通知
        4. 记录升级完成审计日志
        5. 返回 EscalationResult

    Args:
        alert: 危机警报
        channels: 要触发的通道列表（默认全部）
        audit_store: 审计日志存储（默认全局实例）
        scheduler: 排班系统（默认全局实例）

    Returns:
        EscalationResult: 升级结果

    响应时间目标：< 30 秒
    """
    start_time = time.time()

    if channels is None:
        channels = list(EscalationChannel)

    if audit_store is None:
        audit_store = get_audit_store()

    if scheduler is None:
        scheduler = get_scheduler()

    # 生成升级事件 ID
    alert_id = f"esc_{uuid.uuid4().hex[:12]}"

    # 1. 记录警报接收
    audit_store.append(
        event_type=AuditEventType.ALERT_RECEIVED,
        alert_id=alert_id,
        user_id=alert.user_id,
        details={
            "risk_score": alert.risk_score,
            "trigger_evidence": alert.trigger_evidence,
            "recommended_action": alert.recommended_action,
        },
    )

    # 2. 分配咨询师
    assigned_counselor = ""
    counselor = scheduler.get_available_counselor()
    if counselor:
        assigned_counselor = counselor.counselor_id
        audit_store.append(
            event_type=AuditEventType.COUNSELOR_ASSIGNED,
            alert_id=alert_id,
            user_id=alert.user_id,
            details={
                "counselor_id": counselor.counselor_id,
                "counselor_name": counselor.name,
                "specialty": counselor.specialty,
            },
        )
        print(
            f"[Scheduler] 分配咨询师: {counselor.name} "
            f"({counselor.specialty})",
            file=sys.stderr,
        )
    else:
        print(
            "[Scheduler] ⚠️ 当前无可用咨询师，仅触发通道通知",
            file=sys.stderr,
        )

    # 3. 触发通知通道
    notified_channels = []
    for channel in channels:
        handler = _CHANNEL_HANDLERS.get(channel)
        if handler:
            result = handler(alert, alert_id)
            if result.success:
                notified_channels.append(channel)
                audit_store.append(
                    event_type=AuditEventType.CHANNEL_NOTIFIED,
                    alert_id=alert_id,
                    user_id=alert.user_id,
                    details={
                        "channel": channel.value,
                        "latency_ms": result.latency_ms,
                        "message": result.message,
                    },
                )

    # 4. 计算响应时间
    response_time = time.time() - start_time

    # 5. 记录升级完成
    status = EscalationStatus.DISPATCHED if notified_channels else EscalationStatus.FAILED
    audit_entry = audit_store.append(
        event_type=AuditEventType.ESCALATION_COMPLETED,
        alert_id=alert_id,
        user_id=alert.user_id,
        details={
            "status": status.value,
            "channels_notified": [c.value for c in notified_channels],
            "assigned_counselor": assigned_counselor,
            "response_time_seconds": round(response_time, 3),
            "within_target": response_time < RESPONSE_TIME_TARGET_SECONDS,
        },
    )

    # 6. 响应时间检查
    if response_time >= RESPONSE_TIME_TARGET_SECONDS:
        print(
            f"[Escalation] ⚠️ 响应时间 {response_time:.2f}s "
            f"超过目标 {RESPONSE_TIME_TARGET_SECONDS}s！",
            file=sys.stderr,
        )

    return EscalationResult(
        alert_id=alert_id,
        alert=alert,
        status=status,
        channels_notified=notified_channels,
        assigned_counselor=assigned_counselor,
        response_time_seconds=round(response_time, 3),
        timestamp=time.time(),
        audit_log_id=audit_entry.log_id,
    )


# ============================================================
# 报告生成
# ============================================================

def generate_escalation_report(
    audit_store: Optional[AuditLogStore] = None,
) -> str:
    """生成危机升级审计报告

    ⚠️ 比赛 Demo 为模拟实现。

    Args:
        audit_store: 审计日志存储

    Returns:
        str: Markdown 格式审计报告
    """
    if audit_store is None:
        audit_store = get_audit_store()

    entries = audit_store.get_all_entries()

    lines = [
        "# 危机升级审计报告",
        "",
        f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> ⚠️ 比赛 Demo 为模拟实现，非真实紧急呼叫系统",
        "",
        f"## 总览",
        "",
        f"- 审计日志总数：{audit_store.total_entries}",
        f"- 哈希链完整性：{'✅ 通过' if audit_store.verify_chain() else '❌ 异常'}",
        "",
    ]

    # 按 alert_id 分组
    alert_groups: dict[str, list] = {}
    for entry in entries:
        alert_groups.setdefault(entry.alert_id, []).append(entry)

    if alert_groups:
        lines.extend([
            "## 升级事件列表",
            "",
            "| Alert ID | 用户 | 事件数 | 首条时间 | 最后事件 |",
            "|----------|------|--------|----------|----------|",
        ])

        for alert_id, group in alert_groups.items():
            user_id = group[0].user_id
            n_events = len(group)
            first_time = time.strftime(
                "%H:%M:%S", time.localtime(group[0].timestamp)
            )
            last_event = group[-1].event_type.value
            lines.append(
                f"| {alert_id} | {user_id} | {n_events} | "
                f"{first_time} | {last_event} |"
            )

        lines.append("")

    # 响应时间统计
    response_times = []
    for entry in entries:
        if entry.event_type == AuditEventType.ESCALATION_COMPLETED:
            rt = entry.details.get("response_time_seconds", 0)
            response_times.append(rt)

    if response_times:
        avg_rt = sum(response_times) / len(response_times)
        max_rt = max(response_times)
        within_target = sum(1 for rt in response_times if rt < RESPONSE_TIME_TARGET_SECONDS)

        lines.extend([
            "## 响应时间统计",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 升级次数 | {len(response_times)} |",
            f"| 平均响应时间 | {avg_rt:.3f} 秒 |",
            f"| 最大响应时间 | {max_rt:.3f} 秒 |",
            f"| 达标率（< {RESPONSE_TIME_TARGET_SECONDS}s） | "
            f"{within_target}/{len(response_times)} "
            f"({100*within_target/len(response_times):.0f}%) |",
            "",
        ])

    lines.extend([
        "## 声明",
        "",
        "本报告为模拟系统的审计日志，仅供比赛演示使用。",
        "真实部署需要 WORM 存储和区块链存证确保不可篡改性。",
        "",
    ])

    return "\n".join(lines)
