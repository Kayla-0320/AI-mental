"""
升级协议模块

核心功能：
    定义 4 级危机响应协议，每级包含：
    1. 触发条件（Trigger Conditions）
    2. 响应动作（Response Actions）
    3. 超时处理（Timeout Handling）
    4. 回退策略（Fallback Strategy）

协议级别：
    L1 - 监控观察（Monitor）：轻度消极，持续监控
    L2 - 主动介入（Intervene）：明显消极/自伤暗示，AI主动干预
    L3 - 人工转介（Refer）：高风险，转介咨询师
    L4 - 紧急响应（Emergency）：自伤/自杀明确意图，紧急通知

设计原则：
    1. 逐级升级：低级别超时未处理自动升级到更高级别
    2. 并行通知：高级别协议同时通知多个渠道
    3. 最小伤害：优先 AI 自助，其次咨询师，最后紧急服务
    4. 全程审计：每个协议执行步骤均记录审计日志

⚠️ 比赛 Demo 为模拟实现，生产环境需接入真实危机干预流程
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ============================================================
# 数据结构
# ============================================================

class ProtocolLevel(str, Enum):
    """协议级别"""
    L1_MONITOR = "L1_monitor"           # 监控观察
    L2_INTERVENE = "L2_intervene"       # 主动介入
    L3_REFER = "L3_refer"               # 人工转介
    L4_EMERGENCY = "L4_emergency"       # 紧急响应


class ProtocolStatus(str, Enum):
    """协议状态"""
    PENDING = "pending"          # 等待触发
    ACTIVE = "active"            # 执行中
    ACKNOWLEDGED = "acknowledged"  # 已确认
    RESOLVED = "resolved"        # 已解决
    ESCALATED = "escalated"      # 已升级
    TIMEOUT = "timeout"          # 超时
    FAILED = "failed"            # 失败


@dataclass
class TriggerCondition:
    """触发条件"""
    risk_level: str                          # 风险等级阈值
    self_harm_keywords: bool = False         # 是否检测到自伤关键词
    emotion_threshold: float = 0.0           # 情绪概率阈值
    consecutive_negative: int = 0            # 连续消极轮次
    duration_minutes: float = 0              # 持续时间阈值


@dataclass
class ResponseAction:
    """响应动作"""
    action_type: str                         # 动作类型
    target: str                              # 目标（user/consultant/admin/hotline）
    channel: str                             # 通知渠道（in_app/sms/phone/webhook）
    message_template: str = ""               # 消息模板
    priority: int = 0                        # 优先级（0=低，3=紧急）
    timeout_seconds: float = 0               # 超时时间


@dataclass
class ProtocolExecution:
    """协议执行记录"""
    execution_id: str
    protocol_level: ProtocolLevel
    user_id: str
    status: ProtocolStatus = ProtocolStatus.PENDING
    triggered_at: float = field(default_factory=time.time)
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    actions_taken: list[dict] = field(default_factory=list)
    escalation_count: int = 0
    timeout_count: int = 0
    notes: str = ""


# ============================================================
# 协议定义
# ============================================================

@dataclass
class EscalationProtocol:
    """升级协议定义"""
    level: ProtocolLevel
    name: str
    description: str
    trigger: TriggerCondition
    actions: list[ResponseAction]
    timeout_action: Optional[ResponseAction] = None    # 超时后升级动作
    fallback_action: Optional[ResponseAction] = None   # 回退动作
    auto_escalate_to: Optional[ProtocolLevel] = None   # 自动升级到
    max_duration_minutes: float = 0                    # 最大持续时间


# 预定义协议
PROTOCOLS: dict[ProtocolLevel, EscalationProtocol] = {
    ProtocolLevel.L1_MONITOR: EscalationProtocol(
        level=ProtocolLevel.L1_MONITOR,
        name="监控观察",
        description="轻度消极情绪，持续监控状态变化",
        trigger=TriggerCondition(
            risk_level="low",
            emotion_threshold=0.4,
            consecutive_negative=2,
        ),
        actions=[
            ResponseAction(
                action_type="ai_response",
                target="user",
                channel="in_app",
                message_template="我注意到你最近好像有些不太开心。想聊聊吗？我一直在这里。",
                priority=0,
                timeout_seconds=300,  # 5分钟
            ),
            ResponseAction(
                action_type="log_event",
                target="system",
                channel="audit_log",
                message_template="L1 监控协议触发：用户 {user_id} 连续 {n} 轮消极情绪",
                priority=0,
            ),
        ],
        auto_escalate_to=ProtocolLevel.L2_INTERVENE,
        max_duration_minutes=1440,  # 24小时
    ),

    ProtocolLevel.L2_INTERVENE: EscalationProtocol(
        level=ProtocolLevel.L2_INTERVENE,
        name="主动介入",
        description="明显消极或自伤暗示，AI 主动干预并提供安全计划",
        trigger=TriggerCondition(
            risk_level="medium",
            emotion_threshold=0.6,
            consecutive_negative=3,
            self_harm_keywords=True,
        ),
        actions=[
            ResponseAction(
                action_type="crisis_intervention",
                target="user",
                channel="in_app",
                message_template="你说的话让我很在意。我想直接问你：你有没有想过伤害自己？",
                priority=1,
                timeout_seconds=180,  # 3分钟
            ),
            ResponseAction(
                action_type="safety_plan",
                target="user",
                channel="in_app",
                message_template="我来帮你制定一个安全计划，好吗？",
                priority=1,
            ),
            ResponseAction(
                action_type="resource_push",
                target="user",
                channel="in_app",
                message_template="如果你需要帮助，可以拨打24小时心理援助热线：400-161-9995",
                priority=1,
            ),
            ResponseAction(
                action_type="notify_consultant",
                target="consultant",
                channel="in_app",
                message_template="用户 {user_id} 需要关注，风险等级：中等",
                priority=1,
                timeout_seconds=600,  # 10分钟
            ),
        ],
        auto_escalate_to=ProtocolLevel.L3_REFER,
        max_duration_minutes=120,  # 2小时
    ),

    ProtocolLevel.L3_REFER: EscalationProtocol(
        level=ProtocolLevel.L3_REFER,
        name="人工转介",
        description="高风险，转介给专业咨询师",
        trigger=TriggerCondition(
            risk_level="high",
            emotion_threshold=0.7,
            self_harm_keywords=True,
            consecutive_negative=5,
        ),
        actions=[
            ResponseAction(
                action_type="urgent_notify_consultant",
                target="consultant",
                channel="in_app",
                message_template="⚠️ 紧急：用户 {user_id} 高风险，请立即介入",
                priority=2,
                timeout_seconds=300,  # 5分钟
            ),
            ResponseAction(
                action_type="notify_admin",
                target="admin",
                channel="in_app",
                message_template="用户 {user_id} 已触发 L3 转介协议",
                priority=2,
            ),
            ResponseAction(
                action_type="crisis_resources",
                target="user",
                channel="in_app",
                message_template="你的安全对我们非常重要。以下是可以帮助你的资源：\n- 24小时心理援助热线：400-161-9995\n- 12355青少年服务热线\n- 生命热线：400-821-1215",
                priority=2,
            ),
            ResponseAction(
                action_type="schedule_followup",
                target="system",
                channel="audit_log",
                message_template="安排 24 小时内跟进",
                priority=2,
                timeout_seconds=86400,  # 24小时
            ),
        ],
        auto_escalate_to=ProtocolLevel.L4_EMERGENCY,
        max_duration_minutes=60,  # 1小时
    ),

    ProtocolLevel.L4_EMERGENCY: EscalationProtocol(
        level=ProtocolLevel.L4_EMERGENCY,
        name="紧急响应",
        description="自伤/自杀明确意图，紧急通知所有相关方",
        trigger=TriggerCondition(
            risk_level="crisis",
            self_harm_keywords=True,
            emotion_threshold=0.8,
        ),
        actions=[
            ResponseAction(
                action_type="emergency_user",
                target="user",
                channel="in_app",
                message_template="你现在的感受让我非常担心。你的安全是最重要的。请立刻：\n1. 拨打 120 或 110\n2. 拨打24小时心理援助热线：400-161-9995\n3. 告诉身边的大人\n我会一直在这里陪着你。",
                priority=3,
            ),
            ResponseAction(
                action_type="emergency_notify_consultant",
                target="consultant",
                channel="sms",
                message_template="🚨 紧急：用户 {user_id} 存在自伤风险，请立即联系！",
                priority=3,
                timeout_seconds=120,  # 2分钟
            ),
            ResponseAction(
                action_type="emergency_notify_admin",
                target="admin",
                channel="sms",
                message_template="🚨 紧急：用户 {user_id} 触发 L4 紧急响应协议",
                priority=3,
                timeout_seconds=120,
            ),
            ResponseAction(
                action_type="emergency_notify_guardian",
                target="guardian",
                channel="phone",
                message_template="您的孩子/被监护人可能存在安全风险，请尽快联系。",
                priority=3,
                timeout_seconds=180,  # 3分钟
            ),
            ResponseAction(
                action_type="create_incident",
                target="system",
                channel="audit_log",
                message_template="L4 紧急事件创建：用户 {user_id}",
                priority=3,
            ),
        ],
        max_duration_minutes=30,  # 30分钟
    ),
}


# ============================================================
# 协议执行引擎
# ============================================================

class ProtocolEngine:
    """升级协议执行引擎

    职责：
    1. 根据风险评估结果匹配协议级别
    2. 执行协议中的所有响应动作
    3. 监控超时并自动升级
    4. 记录全程审计日志
    """

    def __init__(self):
        self._executions: list[ProtocolExecution] = []
        self._active_protocols: dict[str, ProtocolExecution] = {}  # user_id -> execution

    def match_protocol(
        self,
        risk_level: str,
        self_harm_detected: bool = False,
        max_emotion_prob: float = 0.0,
        consecutive_negative: int = 0,
    ) -> Optional[ProtocolLevel]:
        """根据条件匹配协议级别

        从高到低匹配，返回第一个满足条件的级别。

        Args:
            risk_level: 风险等级
            self_harm_detected: 是否检测到自伤关键词
            max_emotion_prob: 最高消极情绪概率
            consecutive_negative: 连续消极轮次

        Returns:
            ProtocolLevel 或 None
        """
        # 从高到低检查
        for level in reversed(list(ProtocolLevel)):
            protocol = PROTOCOLS.get(level)
            if not protocol:
                continue

            trigger = protocol.trigger
            matched = True

            # 风险等级检查
            risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "crisis": 4}
            if risk_order.get(risk_level, 0) < risk_order.get(trigger.risk_level, 0):
                matched = False

            # 自伤关键词检查
            if trigger.self_harm_keywords and not self_harm_detected:
                # 不是必须条件，只是加分项
                pass

            # 情绪阈值检查
            if trigger.emotion_threshold > 0 and max_emotion_prob < trigger.emotion_threshold:
                matched = False

            # 连续消极检查
            if trigger.consecutive_negative > 0 and consecutive_negative < trigger.consecutive_negative:
                matched = False

            if matched:
                return level

        return None

    def execute_protocol(
        self,
        user_id: str,
        level: ProtocolLevel,
        context: Optional[dict] = None,
    ) -> ProtocolExecution:
        """执行升级协议

        Args:
            user_id: 用户 ID
            level: 协议级别
            context: 上下文信息

        Returns:
            ProtocolExecution: 执行记录
        """
        protocol = PROTOCOLS.get(level)
        if not protocol:
            raise ValueError(f"未知协议级别: {level}")

        execution_id = f"pe_{uuid.uuid4().hex[:12]}"
        execution = ProtocolExecution(
            execution_id=execution_id,
            protocol_level=level,
            user_id=user_id,
            status=ProtocolStatus.ACTIVE,
        )

        # 执行所有响应动作
        for action in protocol.actions:
            action_record = self._execute_action(action, user_id, context)
            execution.actions_taken.append(action_record)

        self._executions.append(execution)
        self._active_protocols[user_id] = execution

        return execution

    def acknowledge(self, user_id: str, acknowledged_by: str = "system") -> bool:
        """确认协议已处理

        Args:
            user_id: 用户 ID
            acknowledged_by: 确认者

        Returns:
            bool: 是否成功
        """
        execution = self._active_protocols.get(user_id)
        if not execution:
            return False

        execution.status = ProtocolStatus.ACKNOWLEDGED
        execution.acknowledged_at = time.time()
        execution.notes += f"\n[{acknowledged_by}] 已确认处理"
        return True

    def resolve(self, user_id: str, resolved_by: str = "system", notes: str = "") -> bool:
        """解决协议

        Args:
            user_id: 用户 ID
            resolved_by: 解决者
            notes: 备注

        Returns:
            bool: 是否成功
        """
        execution = self._active_protocols.pop(user_id, None)
        if not execution:
            return False

        execution.status = ProtocolStatus.RESOLVED
        execution.resolved_at = time.time()
        execution.notes += f"\n[{resolved_by}] 已解决: {notes}"
        return True

    def check_timeouts(self) -> list[ProtocolExecution]:
        """检查超时协议并自动升级

        Returns:
            list[ProtocolExecution]: 超时的协议列表
        """
        timed_out = []
        now = time.time()

        for user_id, execution in list(self._active_protocols.items()):
            protocol = PROTOCOLS.get(execution.protocol_level)
            if not protocol:
                continue

            # 检查每个动作的超时
            for action_record in execution.actions_taken:
                if action_record.get("status") == "pending":
                    timeout = action_record.get("timeout_seconds", 0)
                    if timeout > 0 and (now - action_record["executed_at"]) > timeout:
                        action_record["status"] = "timeout"
                        execution.timeout_count += 1

            # 检查是否需要自动升级
            if execution.timeout_count > 0 and protocol.auto_escalate_to:
                next_level = protocol.auto_escalate_to
                execution.status = ProtocolStatus.ESCALATED
                execution.escalation_count += 1
                execution.notes += f"\n[系统] 超时自动升级到 {next_level.value}"

                # 执行升级协议
                new_execution = self.execute_protocol(user_id, next_level)
                timed_out.append(execution)

        return timed_out

    def get_active_protocols(self) -> dict[str, dict]:
        """获取所有活跃协议"""
        return {
            user_id: {
                "execution_id": ex.execution_id,
                "level": ex.protocol_level.value,
                "status": ex.status.value,
                "triggered_at": ex.triggered_at,
                "duration_minutes": round((time.time() - ex.triggered_at) / 60, 1),
                "actions_count": len(ex.actions_taken),
                "escalation_count": ex.escalation_count,
            }
            for user_id, ex in self._active_protocols.items()
        }

    def get_stats(self) -> dict:
        """获取统计信息"""
        total = len(self._executions)
        if total == 0:
            return {"total_executions": 0, "active_count": len(self._active_protocols)}

        level_counts = {}
        status_counts = {}
        for level in ProtocolLevel:
            level_counts[level.value] = sum(
                1 for e in self._executions if e.protocol_level == level
            )
        for status in ProtocolStatus:
            status_counts[status.value] = sum(
                1 for e in self._executions if e.status == status
            )

        return {
            "total_executions": total,
            "active_count": len(self._active_protocols),
            "level_distribution": level_counts,
            "status_distribution": status_counts,
            "total_escalations": sum(e.escalation_count for e in self._executions),
            "total_timeouts": sum(e.timeout_count for e in self._executions),
        }

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _execute_action(
        self,
        action: ResponseAction,
        user_id: str,
        context: Optional[dict] = None,
    ) -> dict:
        """执行单个响应动作

        生产环境应对接真实通知系统（短信/电话/Webhook）。
        当前为模拟实现，记录动作到审计日志。
        """
        # 填充消息模板
        message = action.message_template.format(
            user_id=user_id,
            n=context.get("consecutive_negative", 0) if context else 0,
        )

        record = {
            "action_type": action.action_type,
            "target": action.target,
            "channel": action.channel,
            "message": message,
            "priority": action.priority,
            "timeout_seconds": action.timeout_seconds,
            "status": "executed",  # 模拟：立即执行成功
            "executed_at": time.time(),
        }

        # ⚠️ 比赛 Demo 为模拟实现，生产环境需接入真实通知系统
        if action.channel in ("sms", "phone"):
            record["status"] = "simulated"  # 标记为模拟
            record["note"] = "生产环境需接入真实 SMS/电话通知系统"

        return record


# ============================================================
# 便捷函数
# ============================================================

_engine: Optional[ProtocolEngine] = None


def get_engine() -> ProtocolEngine:
    """获取全局引擎实例"""
    global _engine
    if _engine is None:
        _engine = ProtocolEngine()
    return _engine


def auto_escalate(
    user_id: str,
    risk_level: str,
    self_harm_detected: bool = False,
    max_emotion_prob: float = 0.0,
    consecutive_negative: int = 0,
) -> Optional[ProtocolExecution]:
    """便捷函数：自动匹配并执行升级协议"""
    engine = get_engine()

    level = engine.match_protocol(
        risk_level=risk_level,
        self_harm_detected=self_harm_detected,
        max_emotion_prob=max_emotion_prob,
        consecutive_negative=consecutive_negative,
    )

    if level:
        return engine.execute_protocol(user_id, level, context={
            "risk_level": risk_level,
            "self_harm_detected": self_harm_detected,
            "max_emotion_prob": max_emotion_prob,
            "consecutive_negative": consecutive_negative,
        })

    return None


# ============================================================
# 演示
# ============================================================

def main():
    """演示升级协议执行"""
    print("=" * 60)
    print("升级协议演示")
    print("=" * 60)

    engine = ProtocolEngine()

    test_cases = [
        {
            "user_id": "user_001",
            "risk_level": "low",
            "self_harm_detected": False,
            "max_emotion_prob": 0.5,
            "consecutive_negative": 3,
            "description": "轻度消极，连续3轮",
        },
        {
            "user_id": "user_002",
            "risk_level": "medium",
            "self_harm_detected": True,
            "max_emotion_prob": 0.7,
            "consecutive_negative": 4,
            "description": "中度风险，检测到自伤暗示",
        },
        {
            "user_id": "user_003",
            "risk_level": "high",
            "self_harm_detected": True,
            "max_emotion_prob": 0.8,
            "consecutive_negative": 6,
            "description": "高风险，需要转介",
        },
        {
            "user_id": "user_004",
            "risk_level": "crisis",
            "self_harm_detected": True,
            "max_emotion_prob": 0.9,
            "consecutive_negative": 8,
            "description": "紧急！明确自伤意图",
        },
    ]

    for case in test_cases:
        print(f"\n{'─' * 50}")
        print(f"📝 场景：{case['description']}")
        print(f"👤 用户：{case['user_id']}")
        print(f"⚠️ 风险等级：{case['risk_level']}")

        level = engine.match_protocol(
            risk_level=case["risk_level"],
            self_harm_detected=case["self_harm_detected"],
            max_emotion_prob=case["max_emotion_prob"],
            consecutive_negative=case["consecutive_negative"],
        )

        if level:
            protocol = PROTOCOLS[level]
            print(f"\n📋 匹配协议：{protocol.name} ({level.value})")
            print(f"📖 描述：{protocol.description}")

            execution = engine.execute_protocol(case["user_id"], level, context=case)
            print(f"\n🔧 执行动作：")
            for action in execution.actions_taken:
                priority_emoji = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}.get(action["priority"], "⚪")
                print(f"  {priority_emoji} [{action['channel']}] {action['action_type']} → {action['target']}")
                print(f"     消息：{action['message'][:60]}...")
        else:
            print("✅ 未触发任何协议")

    # 统计
    print(f"\n{'=' * 60}")
    print(f"📊 统计：{engine.get_stats()}")
    print(f"📋 活跃协议：{engine.get_active_protocols()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
