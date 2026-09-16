"""
不可删除审计日志 —— 危机升级事件的全链路追踪

⚠️ 声明：比赛 Demo 为模拟实现，使用内存 + 文件存储。
真实部署需要 WORM（Write Once Read Many）存储或区块链存证。

设计约束：
    - 只追加（append-only），不允许删除或修改
    - 每条日志带时间戳和不可变哈希链
    - 支持按 alert_id / user_id / 时间范围查询
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ============================================================
# 路径配置
# ============================================================

_MODULE_DIR = Path(__file__).resolve().parent
LOG_DIR = _MODULE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG_PATH = str(LOG_DIR / "escalation_audit.jsonl")


# ============================================================
# 数据结构
# ============================================================

class AuditEventType(str, Enum):
    """审计事件类型"""
    ALERT_RECEIVED = "alert_received"           # 警报接收
    CHANNEL_NOTIFIED = "channel_notified"       # 通道通知
    COUNSELOR_ASSIGNED = "counselor_assigned"   # 咨询师分配
    ESCALATION_COMPLETED = "escalation_completed"  # 升级完成
    ESCALATION_FAILED = "escalation_failed"     # 升级失败
    MANUAL_OVERRIDE = "manual_override"         # 人工覆盖


@dataclass
class AuditLogEntry:
    """审计日志条目

    Attributes:
        log_id: 日志唯一标识
        event_type: 事件类型
        alert_id: 关联的升级事件 ID
        user_id: 关联的用户 ID
        timestamp: Unix 时间戳
        details: 事件详情字典
        prev_hash: 上一条日志的哈希（形成哈希链）
        hash: 本条日志的哈希
    """
    log_id: str
    event_type: AuditEventType
    alert_id: str
    user_id: str
    timestamp: float
    details: dict = field(default_factory=dict)
    prev_hash: str = ""
    hash: str = ""


# ============================================================
# 审计日志存储（不可删除）
# ============================================================

class AuditLogStore:
    """不可删除审计日志存储

    ⚠️ 比赛 Demo 为模拟实现，使用内存 + JSONL 文件。
    真实部署应使用 WORM 存储或区块链。

    约束：
        - 只能 append，不能 delete 或 update
        - 每条日志形成哈希链，确保不可篡改
    """

    def __init__(self, persist_path: str = AUDIT_LOG_PATH):
        """初始化审计日志存储

        Args:
            persist_path: 持久化文件路径
        """
        self._entries: list[AuditLogEntry] = []
        self._persist_path = persist_path
        self._last_hash = "genesis"
        self._counter = 0

        # 从文件恢复已有日志
        self._load_from_file()

    def _load_from_file(self) -> None:
        """从 JSONL 文件恢复日志"""
        path = Path(self._persist_path)
        if not path.exists():
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entry = AuditLogEntry(
                        log_id=data["log_id"],
                        event_type=AuditEventType(data["event_type"]),
                        alert_id=data["alert_id"],
                        user_id=data["user_id"],
                        timestamp=data["timestamp"],
                        details=data.get("details", {}),
                        prev_hash=data.get("prev_hash", ""),
                        hash=data.get("hash", ""),
                    )
                    self._entries.append(entry)
                    self._last_hash = entry.hash
                    self._counter += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    def _compute_hash(self, entry: AuditLogEntry) -> str:
        """计算日志条目哈希

        将关键字段序列化后计算 SHA-256，形成哈希链。
        """
        content = (
            f"{entry.log_id}|"
            f"{entry.event_type.value}|"
            f"{entry.alert_id}|"
            f"{entry.user_id}|"
            f"{entry.timestamp}|"
            f"{json.dumps(entry.details, sort_keys=True)}|"
            f"{entry.prev_hash}"
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def append(
        self,
        event_type: AuditEventType,
        alert_id: str,
        user_id: str,
        details: Optional[dict] = None,
    ) -> AuditLogEntry:
        """追加审计日志（唯一写入方式）

        Args:
            event_type: 事件类型
            alert_id: 关联的升级事件 ID
            user_id: 关联的用户 ID
            details: 事件详情

        Returns:
            AuditLogEntry: 新写入的日志条目

        Raises:
            PermissionError: 不允许删除或修改操作（此方法只支持追加）
        """
        self._counter += 1
        log_id = f"audit_{self._counter:06d}"

        entry = AuditLogEntry(
            log_id=log_id,
            event_type=event_type,
            alert_id=alert_id,
            user_id=user_id,
            timestamp=time.time(),
            details=details or {},
            prev_hash=self._last_hash,
        )
        entry.hash = self._compute_hash(entry)

        self._entries.append(entry)
        self._last_hash = entry.hash

        # 持久化到文件
        self._persist_entry(entry)

        return entry

    def _persist_entry(self, entry: AuditLogEntry) -> None:
        """持久化单条日志到 JSONL 文件"""
        data = {
            "log_id": entry.log_id,
            "event_type": entry.event_type.value,
            "alert_id": entry.alert_id,
            "user_id": entry.user_id,
            "timestamp": entry.timestamp,
            "details": entry.details,
            "prev_hash": entry.prev_hash,
            "hash": entry.hash,
        }
        with open(self._persist_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def query(
        self,
        alert_id: Optional[str] = None,
        user_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
    ) -> list[AuditLogEntry]:
        """查询审计日志

        Args:
            alert_id: 按升级事件 ID 过滤
            user_id: 按用户 ID 过滤
            event_type: 按事件类型过滤

        Returns:
            list[AuditLogEntry]: 匹配的日志条目列表
        """
        results = self._entries
        if alert_id:
            results = [e for e in results if e.alert_id == alert_id]
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        return results

    def verify_chain(self) -> bool:
        """验证哈希链完整性

        Returns:
            bool: 哈希链是否完整
        """
        for i, entry in enumerate(self._entries):
            expected_hash = self._compute_hash(entry)
            if entry.hash != expected_hash:
                return False
            if i > 0 and entry.prev_hash != self._entries[i - 1].hash:
                return False
        return True

    @property
    def total_entries(self) -> int:
        """总日志条数"""
        return len(self._entries)

    def get_all_entries(self) -> list[AuditLogEntry]:
        """获取所有日志条目"""
        return list(self._entries)

    # ============================================================
    # 安全约束：禁止删除和修改
    # ============================================================

    def delete(self, log_id: str) -> None:
        """删除操作 —— 永远不允许

        Raises:
            PermissionError: 审计日志不可删除
        """
        raise PermissionError("审计日志不可删除（append-only 约束）")

    def update(self, log_id: str, **kwargs) -> None:
        """修改操作 —— 永远不允许

        Raises:
            PermissionError: 审计日志不可修改
        """
        raise PermissionError("审计日志不可修改（append-only 约束）")


# ============================================================
# 全局审计日志实例
# ============================================================

_global_store: Optional[AuditLogStore] = None


def get_audit_store() -> AuditLogStore:
    """获取全局审计日志存储实例"""
    global _global_store
    if _global_store is None:
        _global_store = AuditLogStore()
    return _global_store


def reset_audit_store() -> AuditLogStore:
    """重置全局审计日志存储（仅用于测试）"""
    global _global_store
    _global_store = AuditLogStore(persist_path=str(LOG_DIR / "test_audit.jsonl"))
    return _global_store
