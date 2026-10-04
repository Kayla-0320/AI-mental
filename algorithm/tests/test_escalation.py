"""
危机升级单元测试

覆盖：
- 审计日志（append-only、哈希链、查询）
- 咨询师排班（工作时间/非工作时间分配）
- 升级引擎（三通道通知、响应时间）
- 审计报告生成
- 数据类完整性
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from shared.dataclasses import (
    CrisisAlert,
    EscalationChannel,
    EscalationResult,
    EscalationStatus,
)

from escalation.audit_log import (
    AuditLogStore,
    AuditEventType,
    reset_audit_store,
)
from escalation.scheduler import (
    Counselor,
    CounselorScheduler,
    ScheduleEntry,
)
from escalation.crisis import (
    escalate,
    generate_escalation_report,
    RESPONSE_TIME_TARGET_SECONDS,
    _notify_webhook,
    _notify_in_app,
    _notify_console,
)


# ============================================================
# 辅助数据
# ============================================================

@pytest.fixture
def sample_alert():
    """样本危机警报"""
    return CrisisAlert(
        user_id="user_test_001",
        risk_score=0.92,
        trigger_evidence=["检测到自伤关键词", "情绪急剧恶化"],
        timestamp=time.time(),
        recommended_action="立即通知监护人并推送危机热线",
    )


@pytest.fixture
def audit_store(tmp_path):
    """测试用审计日志存储"""
    return AuditLogStore(persist_path=str(tmp_path / "test_audit.jsonl"))


# ============================================================
# 审计日志测试
# ============================================================

class TestAuditLog:
    def test_append_entry(self, audit_store):
        """追加审计日志"""
        entry = audit_store.append(
            event_type=AuditEventType.ALERT_RECEIVED,
            alert_id="test_001",
            user_id="user_001",
            details={"risk_score": 0.9},
        )
        assert entry.log_id.startswith("audit_")
        assert entry.event_type == AuditEventType.ALERT_RECEIVED
        assert audit_store.total_entries == 1

    def test_hash_chain_valid(self, audit_store):
        """哈希链完整性"""
        audit_store.append(AuditEventType.ALERT_RECEIVED, "a1", "u1")
        audit_store.append(AuditEventType.CHANNEL_NOTIFIED, "a1", "u1")
        audit_store.append(AuditEventType.ESCALATION_COMPLETED, "a1", "u1")
        assert audit_store.verify_chain() is True

    def test_query_by_alert_id(self, audit_store):
        """按 alert_id 查询"""
        audit_store.append(AuditEventType.ALERT_RECEIVED, "a1", "u1")
        audit_store.append(AuditEventType.ALERT_RECEIVED, "a2", "u2")
        audit_store.append(AuditEventType.CHANNEL_NOTIFIED, "a1", "u1")

        results = audit_store.query(alert_id="a1")
        assert len(results) == 2

    def test_query_by_user_id(self, audit_store):
        """按 user_id 查询"""
        audit_store.append(AuditEventType.ALERT_RECEIVED, "a1", "u1")
        audit_store.append(AuditEventType.ALERT_RECEIVED, "a2", "u2")

        results = audit_store.query(user_id="u1")
        assert len(results) == 1

    def test_query_by_event_type(self, audit_store):
        """按事件类型查询"""
        audit_store.append(AuditEventType.ALERT_RECEIVED, "a1", "u1")
        audit_store.append(AuditEventType.CHANNEL_NOTIFIED, "a1", "u1")
        audit_store.append(AuditEventType.ALERT_RECEIVED, "a2", "u2")

        results = audit_store.query(event_type=AuditEventType.ALERT_RECEIVED)
        assert len(results) == 2

    def test_delete_raises_error(self, audit_store):
        """删除操作必须抛出 PermissionError"""
        audit_store.append(AuditEventType.ALERT_RECEIVED, "a1", "u1")
        with pytest.raises(PermissionError, match="不可删除"):
            audit_store.delete("audit_000001")

    def test_update_raises_error(self, audit_store):
        """修改操作必须抛出 PermissionError"""
        audit_store.append(AuditEventType.ALERT_RECEIVED, "a1", "u1")
        with pytest.raises(PermissionError, match="不可修改"):
            audit_store.update("audit_000001", details={})

    def test_persistence(self, tmp_path):
        """日志持久化"""
        path = str(tmp_path / "persist_test.jsonl")
        store1 = AuditLogStore(persist_path=path)
        store1.append(AuditEventType.ALERT_RECEIVED, "a1", "u1", {"key": "value"})
        assert store1.total_entries == 1

        # 重新加载
        store2 = AuditLogStore(persist_path=path)
        assert store2.total_entries == 1
        assert store2.get_all_entries()[0].details["key"] == "value"


# ============================================================
# 咨询师排班测试
# ============================================================

class TestScheduler:
    def test_weekday_daytime_assignment(self):
        """工作日白天有咨询师"""
        scheduler = CounselorScheduler()
        # 周三 14:00
        dt = datetime(2026, 9, 16, 14, 0, 0)
        counselor = scheduler.get_available_counselor(dt)
        assert counselor is not None
        assert counselor.counselor_id == "counselor_001"  # 最高优先级

    def test_weekend_assignment(self):
        """周末有咨询师"""
        scheduler = CounselorScheduler()
        # 周六 12:00
        dt = datetime(2026, 9, 19, 12, 0, 0)
        counselor = scheduler.get_available_counselor(dt)
        assert counselor is not None
        assert counselor.counselor_id == "counselor_003"

    def test_night_no_counselor(self):
        """夜间无咨询师"""
        scheduler = CounselorScheduler()
        # 周三 23:00
        dt = datetime(2026, 9, 16, 23, 0, 0)
        counselor = scheduler.get_available_counselor(dt)
        assert counselor is None

    def test_is_during_office_hours(self):
        """工作时间判断"""
        scheduler = CounselorScheduler()
        # 周三 14:00 → 是工作时间
        assert scheduler.is_during_office_hours(datetime(2026, 9, 16, 14, 0, 0)) is True
        # 周六 → 不是工作时间
        assert scheduler.is_during_office_hours(datetime(2026, 9, 19, 14, 0, 0)) is False
        # 周三 20:00 → 不是工作时间
        assert scheduler.is_during_office_hours(datetime(2026, 9, 16, 20, 0, 0)) is False

    def test_get_all_on_duty(self):
        """获取所有值班咨询师"""
        scheduler = CounselorScheduler()
        # 周三 14:00
        dt = datetime(2026, 9, 16, 14, 0, 0)
        on_duty = scheduler.get_all_on_duty(dt)
        assert len(on_duty) >= 2  # 张咨询师 + 李咨询师

    def test_priority_ordering(self):
        """优先级排序"""
        scheduler = CounselorScheduler()
        dt = datetime(2026, 9, 16, 14, 0, 0)
        on_duty = scheduler.get_all_on_duty(dt)
        # 优先级应该递增
        for i in range(len(on_duty) - 1):
            assert on_duty[i].priority <= on_duty[i + 1].priority


# ============================================================
# 通知通道测试
# ============================================================

class TestChannels:
    def test_webhook_channel(self, sample_alert):
        """Webhook 通道"""
        result = _notify_webhook(sample_alert, "test_001")
        assert result.success is True
        assert result.channel == EscalationChannel.WEBHOOK

    def test_in_app_channel(self, sample_alert):
        """站内通知通道"""
        result = _notify_in_app(sample_alert, "test_001")
        assert result.success is True
        assert result.channel == EscalationChannel.IN_APP

    def test_console_channel(self, sample_alert):
        """控制台告警通道"""
        result = _notify_console(sample_alert, "test_001")
        assert result.success is True
        assert result.channel == EscalationChannel.CONSOLE


# ============================================================
# 升级引擎测试
# ============================================================

class TestEscalate:
    def test_escalate_returns_result(self, sample_alert, audit_store):
        """升级返回 EscalationResult"""
        result = escalate(sample_alert, audit_store=audit_store)
        assert isinstance(result, EscalationResult)
        assert result.alert_id.startswith("esc_")
        assert result.status == EscalationStatus.DISPATCHED

    def test_escalate_all_channels(self, sample_alert, audit_store):
        """三通道全部通知"""
        result = escalate(sample_alert, audit_store=audit_store)
        assert len(result.channels_notified) == 3
        assert EscalationChannel.WEBHOOK in result.channels_notified
        assert EscalationChannel.IN_APP in result.channels_notified
        assert EscalationChannel.CONSOLE in result.channels_notified

    def test_escalate_selective_channels(self, sample_alert, audit_store):
        """选择性通道"""
        result = escalate(
            sample_alert,
            channels=[EscalationChannel.CONSOLE],
            audit_store=audit_store,
        )
        assert len(result.channels_notified) == 1
        assert EscalationChannel.CONSOLE in result.channels_notified

    def test_response_time_within_target(self, sample_alert, audit_store):
        """响应时间在 30 秒内"""
        result = escalate(sample_alert, audit_store=audit_store)
        assert result.response_time_seconds < RESPONSE_TIME_TARGET_SECONDS

    def test_counselor_assigned(self, sample_alert, audit_store):
        """咨询师分配"""
        result = escalate(sample_alert, audit_store=audit_store)
        # 取决于当前时间是否有值班咨询师
        # 如果有，assigned_counselor 不为空
        assert isinstance(result.assigned_counselor, str)

    def test_audit_log_created(self, sample_alert, audit_store):
        """审计日志已创建"""
        result = escalate(sample_alert, audit_store=audit_store)
        # 至少有：ALERT_RECEIVED + CHANNEL_NOTIFIED * 3 + ESCALATION_COMPLETED
        assert audit_store.total_entries >= 5

    def test_audit_chain_valid_after_escalation(self, sample_alert, audit_store):
        """升级后哈希链完整"""
        escalate(sample_alert, audit_store=audit_store)
        assert audit_store.verify_chain() is True


# ============================================================
# 审计报告测试
# ============================================================

class TestReport:
    def test_report_generation(self, sample_alert, audit_store):
        """报告生成"""
        escalate(sample_alert, audit_store=audit_store)
        report = generate_escalation_report(audit_store)
        assert "# 危机升级审计报告" in report
        assert "响应时间统计" in report

    def test_report_contains_alert_info(self, sample_alert, audit_store):
        """报告包含警报信息"""
        escalate(sample_alert, audit_store=audit_store)
        report = generate_escalation_report(audit_store)
        assert "user_test_001" in report

    def test_report_chain_validation(self, sample_alert, audit_store):
        """报告包含链验证"""
        escalate(sample_alert, audit_store=audit_store)
        report = generate_escalation_report(audit_store)
        assert "✅ 通过" in report


# ============================================================
# 数据类完整性测试
# ============================================================

class TestDataclasses:
    def test_escalation_channel_values(self):
        """EscalationChannel 枚举值"""
        assert EscalationChannel.WEBHOOK.value == "webhook"
        assert EscalationChannel.IN_APP.value == "in_app"
        assert EscalationChannel.CONSOLE.value == "console"

    def test_escalation_status_values(self):
        """EscalationStatus 枚举值"""
        assert EscalationStatus.PENDING.value == "pending"
        assert EscalationStatus.DISPATCHED.value == "dispatched"
        assert EscalationStatus.RESOLVED.value == "resolved"

    def test_escalation_result_fields(self, sample_alert):
        """EscalationResult 字段完整"""
        result = EscalationResult(
            alert_id="test_001",
            alert=sample_alert,
            status=EscalationStatus.DISPATCHED,
        )
        assert result.alert_id == "test_001"
        assert result.assigned_counselor == ""
        assert result.response_time_seconds == 0.0
