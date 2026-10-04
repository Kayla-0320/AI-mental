"""
主动关怀机制 —— 在用户状态异常时主动发起温和问候

功能：
    1. 检测触发条件（基线偏离、App 未使用、历史危机信号）
    2. 生成关怀消息（引用用户记忆，避免空洞问候）
    3. 用户授权管理（开启/关闭）
    4. 审计日志记录

伦理边界：
    - 必须获得用户明确授权
    - 每天最多 1 条关怀消息
    - 涉及危机信号必须转人工，不能只靠 AI
    - 用户可随时关闭
    - 所有操作记录到审计日志

设计依据：
    - 真正需要帮助的青少年往往不会主动求助
    - 主动关怀需要温和、不压迫，引用用户之前提过的事实
    - 避免"你还好吗"这类空洞问候
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from assessment.user_memory import get_memory_store, MemoryCategory


# ============================================================
# 路径配置
# ============================================================

_MODULE_DIR = Path(__file__).resolve().parent
LOG_DIR = _MODULE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CARE_LOG_PATH = str(LOG_DIR / "proactive_care.jsonl")
CONSENT_PATH = str(LOG_DIR / "proactive_care_consent.json")


# ============================================================
# 触发条件枚举
# ============================================================

class TriggerCondition(str, Enum):
    """触发条件类型"""
    BASELINE_DEVIATION = "baseline_deviation"      # 基线连续偏离
    APP_INACTIVITY = "app_inactivity"              # App 长时间未使用
    HISTORICAL_CRISIS = "historical_crisis"        # 历史危机信号（未升级）
    TWIN_PREDICTION = "twin_prediction"            # 数字孪生预测危机风险升高


# ============================================================
# 数据结构
# ============================================================

@dataclass
class TriggerSignal:
    """触发信号

    Attributes:
        condition: 触发条件类型
        user_id: 用户 ID
        severity: 严重程度 (0-1)
        details: 触发详情
        detected_at: 检测时间戳
    """
    condition: TriggerCondition
    user_id: str
    severity: float = 0.5
    details: dict = field(default_factory=dict)
    detected_at: float = 0.0


@dataclass
class CareMessage:
    """关怀消息

    Attributes:
        id: 消息唯一标识
        user_id: 目标用户 ID
        content: 消息内容
        trigger_condition: 触发条件
        referenced_memory_id: 引用的记忆 ID（可选）
        sent_at: 发送时间戳
        acknowledged: 用户是否已读
    """
    id: str
    user_id: str
    content: str
    trigger_condition: TriggerCondition
    referenced_memory_id: str = ""
    sent_at: float = 0.0
    acknowledged: bool = False


@dataclass
class UserConsent:
    """用户授权

    Attributes:
        user_id: 用户 ID
        enabled: 是否开启主动关怀
        granted_at: 授权时间戳
        revoked_at: 撤销时间戳（0 表示未撤销）
    """
    user_id: str
    enabled: bool = False
    granted_at: float = 0.0
    revoked_at: float = 0.0


# ============================================================
# 主动关怀管理器
# ============================================================

class ProactiveCareManager:
    """主动关怀管理器

    负责：
    1. 评估触发条件
    2. 生成关怀消息
    3. 管理用户授权
    4. 记录审计日志

    伦理约束：
    - 必须获得用户明确授权
    - 每天最多 1 条关怀消息
    - 涉及危机信号必须转人工
    """

    # 触发阈值
    BASELINE_DEVIATION_THRESHOLD = 2.0      # 基线偏离阈值（|Z| > 2.0）
    BASELINE_CONSECUTIVE_DAYS = 3           # 连续偏离天数
    APP_INACTIVITY_DAYS = 5                 # App 未使用天数
    MAX_CARE_PER_DAY = 1                    # 每天最多关怀消息数

    def __init__(self, care_log_path: str = CARE_LOG_PATH, consent_path: str = CONSENT_PATH):
        """初始化主动关怀管理器

        Args:
            care_log_path: 关怀日志路径
            consent_path: 用户授权存储路径
        """
        self._care_log_path = care_log_path
        self._consent_path = consent_path
        self._consents: dict[str, UserConsent] = {}
        self._daily_care_count: dict[str, int] = {}  # user_id -> count today
        self._last_evaluation: dict[str, float] = {}  # user_id -> timestamp
        self._care_history: list[CareMessage] = []

        # 加载用户授权
        self._load_consents()

    # ============================================================
    # 用户授权管理
    # ============================================================

    def grant_consent(self, user_id: str) -> UserConsent:
        """用户开启主动关怀

        Args:
            user_id: 用户 ID

        Returns:
            UserConsent: 授权记录
        """
        consent = UserConsent(
            user_id=user_id,
            enabled=True,
            granted_at=time.time(),
            revoked_at=0.0,
        )
        self._consents[user_id] = consent
        self._save_consents()
        self._log_audit(user_id, "consent_granted", {"enabled": True})
        return consent

    def revoke_consent(self, user_id: str) -> UserConsent:
        """用户关闭主动关怀

        Args:
            user_id: 用户 ID

        Returns:
            UserConsent: 更新后的授权记录
        """
        if user_id in self._consents:
            self._consents[user_id].enabled = False
            self._consents[user_id].revoked_at = time.time()
        else:
            self._consents[user_id] = UserConsent(
                user_id=user_id,
                enabled=False,
                granted_at=0.0,
                revoked_at=time.time(),
            )
        self._save_consents()
        self._log_audit(user_id, "consent_revoked", {"enabled": False})
        return self._consents[user_id]

    def has_consent(self, user_id: str) -> bool:
        """检查用户是否已授权

        Args:
            user_id: 用户 ID

        Returns:
            bool: 是否已授权且未撤销
        """
        consent = self._consents.get(user_id)
        return consent is not None and consent.enabled

    # ============================================================
    # 触发条件评估
    # ============================================================

    def evaluate_triggers(
        self,
        user_id: str,
        baseline_deviations: Optional[list[float]] = None,
        last_app_open: Optional[float] = None,
        historical_crisis: bool = False,
    ) -> list[TriggerSignal]:
        """评估触发条件

        每天最多评估一次。

        Args:
            user_id: 用户 ID
            baseline_deviations: 最近 N 天的基线偏离值（|Z|）
            last_app_open: 上次打开 App 的时间戳
            historical_crisis: 是否有历史危机信号（未升级）

        Returns:
            list[TriggerSignal]: 触发的信号列表
        """
        # 检查用户授权
        if not self.has_consent(user_id):
            return []

        # 检查每天评估一次限制
        now = time.time()
        last_eval = self._last_evaluation.get(user_id, 0)
        if now - last_eval < 86400:  # 24 小时
            return []
        self._last_evaluation[user_id] = now

        triggers: list[TriggerSignal] = []

        # 1. 基线连续偏离
        if baseline_deviations:
            consecutive_deviation = sum(
                1 for d in baseline_deviations
                if abs(d) > self.BASELINE_DEVIATION_THRESHOLD
            )
            if consecutive_deviation >= self.BASELINE_CONSECUTIVE_DAYS:
                triggers.append(TriggerSignal(
                    condition=TriggerCondition.BASELINE_DEVIATION,
                    user_id=user_id,
                    severity=min(1.0, consecutive_deviation / 5),
                    details={"consecutive_days": consecutive_deviation, "deviations": baseline_deviations},
                    detected_at=now,
                ))

        # 2. App 长时间未使用
        if last_app_open:
            days_inactive = (now - last_app_open) / 86400
            if days_inactive >= self.APP_INACTIVITY_DAYS:
                triggers.append(TriggerSignal(
                    condition=TriggerCondition.APP_INACTIVITY,
                    user_id=user_id,
                    severity=min(1.0, days_inactive / 10),
                    details={"days_inactive": days_inactive},
                    detected_at=now,
                ))

        # 3. 历史危机信号（仅作为触发，不作为内容）
        if historical_crisis:
            triggers.append(TriggerSignal(
                condition=TriggerCondition.HISTORICAL_CRISIS,
                user_id=user_id,
                severity=0.8,  # 高严重程度
                details={"has_historical_crisis": True},
                detected_at=now,
            ))

        # 记录审计
        if triggers:
            self._log_audit(user_id, "triggers_evaluated", {
                "trigger_count": len(triggers),
                "conditions": [t.condition.value for t in triggers],
            })

        return triggers

    # ============================================================
    # 关怀消息生成
    # ============================================================

    def generate_care_message(
        self,
        user_id: str,
        trigger: TriggerSignal,
    ) -> Optional[CareMessage]:
        """生成关怀消息

        优先引用用户之前提过的事实（来自长期记忆）。
        避免"你还好吗"这类空洞问候。
        消息长度 < 50 字。

        Args:
            user_id: 用户 ID
            trigger: 触发信号

        Returns:
            CareMessage: 生成的关怀消息，如果无法生成则返回 None
        """
        # 检查每天限制
        if self._get_daily_care_count(user_id) >= self.MAX_CARE_PER_DAY:
            return None

        # 获取用户记忆用于引用
        memory_store = get_memory_store()
        memories = memory_store.get_all_memories(user_id)

        # 选择一条记忆作为引用
        referenced_memory = None
        if memories:
            # 优先选择最近创建的、置信度高的记忆
            sorted_memories = sorted(
                memories,
                key=lambda m: (m.confidence, m.created_at),
                reverse=True,
            )
            referenced_memory = sorted_memories[0]

        # 生成消息内容
        content = self._build_care_content(trigger, referenced_memory)

        # 创建关怀消息
        care_msg = CareMessage(
            id=str(uuid.uuid4())[:8],
            user_id=user_id,
            content=content,
            trigger_condition=trigger.condition,
            referenced_memory_id=referenced_memory.id if referenced_memory else "",
            sent_at=time.time(),
            acknowledged=False,
        )

        # 记录
        self._care_history.append(care_msg)
        self._increment_daily_care_count(user_id)
        self._log_care_message(care_msg)
        self._log_audit(user_id, "care_message_sent", {
            "message_id": care_msg.id,
            "trigger": trigger.condition.value,
            "content_length": len(content),
        })

        return care_msg

    def _build_care_content(
        self,
        trigger: TriggerSignal,
        memory: Optional[object] = None,
    ) -> str:
        """构建关怀消息内容

        优先引用用户之前提过的事实。
        避免空洞问候。
        消息长度 < 50 字。

        Args:
            trigger: 触发信号
            memory: 引用的记忆（可选）

        Returns:
            str: 关怀消息内容
        """
        # 数字孪生预测触发：引用因果链，温和提示
        if trigger.condition == TriggerCondition.TWIN_PREDICTION:
            chain = trigger.details.get("causal_chain")
            if memory and hasattr(memory, "fact") and memory.fact:
                fact = memory.fact
                if chain:
                    return f"想起你之前说的{fact[:15]}，我注意到一些变化，想关心你一下。我在。"
                return f"想起你之前说的{fact[:15]}，最近怎么样？想聊随时找我。"
            if chain:
                return "我注意到你最近有些变化，想关心一下。随时可以找我聊聊。"
            return "我一直在，想聊的时候随时找我。"

        # 如果有记忆，优先引用
        if memory and hasattr(memory, "fact") and memory.fact:
            fact = memory.fact
            # 根据触发条件构建关怀语
            if trigger.condition == TriggerCondition.BASELINE_DEVIATION:
                return f"上次你提到{fact[:20]}，最近还好吗？想聊聊随时找我。"
            elif trigger.condition == TriggerCondition.APP_INACTIVITY:
                return f"好久不见，想起你之前说的{fact[:15]}，最近怎么样？"
            else:
                return f"想起你之前说的{fact[:20]}，有空聊聊？"

        # 无记忆时的通用关怀（但避免空洞）
        if trigger.condition == TriggerCondition.BASELINE_DEVIATION:
            return "最近感觉你有些变化，想关心一下。我在，随时可以聊。"
        elif trigger.condition == TriggerCondition.APP_INACTIVITY:
            return "好久没看到你了，希望你一切都好。想聊的时候我在。"
        else:
            return "我一直在，想聊的时候随时找我。"

    # ============================================================
    # 危机信号处理
    # ============================================================

    def evaluate_digital_twin_trigger(
        self,
        user_id: str,
        crisis_probability: float,
        causal_chain: Optional[str] = None,
        deterioration_time: Optional[int] = None,
    ) -> Optional[TriggerSignal]:
        """基于数字孪生预测评估触发条件

        当数字孪生预测的危机概率超过阈值时，触发主动关怀。
        这是“感知→融合→预测→干预”闭环的关键一环。

        Args:
            user_id: 用户 ID
            crisis_probability: 数字孪生预测的危机概率 [0, 1]
            causal_chain: 预测的因果链（如 "circadian → cognitive → text_emotion"）
            deterioration_time: 预计恶化时间步

        Returns:
            TriggerSignal 或 None
        """
        if not self.has_consent(user_id):
            return None

        # 危机概率超过 0.4 触发关怀（比基线偏离更敏感，因为是预测性的）
        TWIN_CRISIS_THRESHOLD = 0.4
        if crisis_probability < TWIN_CRISIS_THRESHOLD:
            return None

        now = time.time()
        severity = min(1.0, crisis_probability)

        details = {
            "crisis_probability": crisis_probability,
            "causal_chain": causal_chain,
            "deterioration_time": deterioration_time,
        }

        self._log_audit(user_id, "twin_trigger_evaluated", {
            "crisis_probability": crisis_probability,
            "triggered": True,
        })

        return TriggerSignal(
            condition=TriggerCondition.TWIN_PREDICTION,
            user_id=user_id,
            severity=severity,
            details=details,
            detected_at=now,
        )

    def should_escalate_to_human(self, trigger: TriggerSignal) -> bool:
        """判断是否需要转人工

        涉及危机信号时，必须转人工，不能只靠 AI。

        Args:
            trigger: 触发信号

        Returns:
            bool: 是否需要转人工
        """
        # 历史危机信号必须转人工
        if trigger.condition == TriggerCondition.HISTORICAL_CRISIS:
            return True

        # 数字孪生预测危机概率极高时转人工
        if trigger.condition == TriggerCondition.TWIN_PREDICTION and trigger.severity > 0.7:
            return True

        # 基线偏离严重（severity > 0.7）建议转人工
        if trigger.condition == TriggerCondition.BASELINE_DEVIATION and trigger.severity > 0.7:
            return True

        return False

    # ============================================================
    # 内部方法
    # ============================================================

    def _get_daily_care_count(self, user_id: str) -> int:
        """获取今天已发送的关怀消息数"""
        today = int(time.time() / 86400)
        key = f"{user_id}_{today}"
        return self._daily_care_count.get(key, 0)

    def _increment_daily_care_count(self, user_id: str) -> None:
        """增加今天已发送的关怀消息数"""
        today = int(time.time() / 86400)
        key = f"{user_id}_{today}"
        self._daily_care_count[key] = self._daily_care_count.get(key, 0) + 1

    def _load_consents(self) -> None:
        """从文件加载用户授权"""
        path = Path(self._consent_path)
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for user_id, consent_data in data.items():
                    self._consents[user_id] = UserConsent(
                        user_id=user_id,
                        enabled=consent_data.get("enabled", False),
                        granted_at=consent_data.get("granted_at", 0),
                        revoked_at=consent_data.get("revoked_at", 0),
                    )
        except (json.JSONDecodeError, IOError):
            pass

    def _save_consents(self) -> None:
        """保存用户授权到文件"""
        data = {}
        for user_id, consent in self._consents.items():
            data[user_id] = {
                "enabled": consent.enabled,
                "granted_at": consent.granted_at,
                "revoked_at": consent.revoked_at,
            }
        with open(self._consent_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _log_audit(self, user_id: str, event: str, details: dict) -> None:
        """记录审计日志"""
        log_entry = {
            "timestamp": time.time(),
            "user_id": user_id,
            "event": event,
            "details": details,
        }
        with open(self._care_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _log_care_message(self, care_msg: CareMessage) -> None:
        """记录关怀消息"""
        log_entry = {
            "timestamp": care_msg.sent_at,
            "user_id": care_msg.user_id,
            "event": "care_message",
            "message_id": care_msg.id,
            "content": care_msg.content,
            "trigger": care_msg.trigger_condition.value,
            "referenced_memory_id": care_msg.referenced_memory_id,
        }
        with open(self._care_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# ============================================================
# 全局实例
# ============================================================

_global_manager: Optional[ProactiveCareManager] = None


def get_proactive_care_manager() -> ProactiveCareManager:
    """获取全局主动关怀管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = ProactiveCareManager()
    return _global_manager


def reset_proactive_care_manager() -> ProactiveCareManager:
    """重置全局主动关怀管理器（仅用于测试）"""
    global _global_manager
    _global_manager = ProactiveCareManager(
        care_log_path=str(LOG_DIR / "test_proactive_care.jsonl"),
        consent_path=str(LOG_DIR / "test_proactive_care_consent.json"),
    )
    return _global_manager
