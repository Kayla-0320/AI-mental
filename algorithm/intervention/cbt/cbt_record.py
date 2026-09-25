"""
CBT 认知重构记录生成器

当对话走过完整 DEEP 流程（SHALLOW → MEDIUM → DEEP）时，
在对话结束时生成结构化"认知重构记录"，供咨询师端查看。

记录结构：
    - 自动思维（来自 SHALLOW 阶段）
    - 认知扭曲类型（来自 SHALLOW 阶段）
    - 支持证据 / 反对证据（来自 MEDIUM 阶段）
    - 替代想法（来自 MEDIUM 阶段）
    - 情绪变化（来自 DEEP 阶段）
    - 行动计划（来自 DEEP 阶段）

设计理由：
    - 咨询师需要了解 AI 对话过程中用户的认知重构进展
    - 结构化记录便于咨询师快速了解情况
    - 为后续人工咨询提供参考
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# CBT 记录数据结构
# ============================================================

class CBTRecordStatus(str, Enum):
    """CBT 记录状态

    - COMPLETE: 完整走过 SHALLOW → MEDIUM → DEEP
    - PARTIAL: 部分完成（未到 DEEP）
    - NOT_STARTED: 未进入苏格拉底模式
    """
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    NOT_STARTED = "NOT_STARTED"


@dataclass
class CBTRestructuringRecord:
    """CBT 认知重构记录

    结构化记录对话过程中的认知重构进展。

    Attributes:
        user_id: 用户 ID
        conversation_id: 对话 ID
        status: 记录状态
        automatic_thought: 识别的自动思维
        distortion_type: 认知扭曲类型
        supporting_evidence: 支持证据列表
        contradicting_evidence: 反对证据列表
        alternative_thought: 替代想法
        initial_emotion_intensity: 初始情绪强度 (0-10)
        final_emotion_intensity: 最终情绪强度 (0-10)
        action_plan: 行动计划
        depth_reached: 达到的最深深度
        turns_in_socratic: 苏格拉底模式总轮次
        summary: 总结（给咨询师看）
    """
    user_id: str
    conversation_id: str
    status: CBTRecordStatus = CBTRecordStatus.NOT_STARTED
    automatic_thought: str = ""
    distortion_type: str = ""
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    alternative_thought: str = ""
    initial_emotion_intensity: Optional[int] = None
    final_emotion_intensity: Optional[int] = None
    action_plan: str = ""
    depth_reached: str = ""
    turns_in_socratic: int = 0
    summary: str = ""


# ============================================================
# CBT 记录跟踪器
# ============================================================

class CBTRecordTracker:
    """CBT 认知重构记录跟踪器

    在对话过程中跟踪认知重构进展，
    对话结束时生成结构化记录。

    使用方式：
        tracker = CBTRecordTracker(user_id="user_123", conversation_id="conv_456")

        # 每轮对话更新
        tracker.update_depth_reached("SHALLOW")
        tracker.record_automatic_thought("我觉得没人喜欢我")
        tracker.increment_socratic_turns()

        # 对话结束时生成记录
        record = tracker.generate_record()
    """

    def __init__(self, user_id: str, conversation_id: str) -> None:
        self.user_id = user_id
        self.conversation_id = conversation_id
        self._automatic_thought: str = ""
        self._distortion_type: str = ""
        self._supporting_evidence: list[str] = []
        self._contradicting_evidence: list[str] = []
        self._alternative_thought: str = ""
        self._initial_emotion: Optional[int] = None
        self._final_emotion: Optional[int] = None
        self._action_plan: str = ""
        self._depth_reached: str = ""
        self._turns_in_socratic: int = 0
        self._depth_history: list[str] = []  # 深度变化历史

    def update_depth_reached(self, depth: str) -> None:
        """更新达到的最深深度

        Args:
            depth: 当前深度 (SHALLOW/MEDIUM/DEEP)
        """
        self._depth_history.append(depth)
        # 按深度排序：DEEP > MEDIUM > SHALLOW
        depth_order = {"SHALLOW": 1, "MEDIUM": 2, "DEEP": 3}
        current_max = depth_order.get(self._depth_reached, 0)
        new_depth = depth_order.get(depth, 0)
        if new_depth > current_max:
            self._depth_reached = depth

    def increment_socratic_turns(self) -> None:
        """苏格拉底模式轮次 +1"""
        self._turns_in_socratic += 1

    def record_automatic_thought(self, thought: str) -> None:
        """记录自动思维（SHALLOW 阶段）"""
        self._automatic_thought = thought

    def record_distortion_type(self, distortion: str) -> None:
        """记录认知扭曲类型（SHALLOW 阶段）"""
        self._distortion_type = distortion

    def record_evidence(
        self,
        supporting: list[str],
        contradicting: list[str],
    ) -> None:
        """记录证据（MEDIUM 阶段）"""
        self._supporting_evidence = supporting
        self._contradicting_evidence = contradicting

    def record_alternative_thought(self, thought: str) -> None:
        """记录替代想法（MEDIUM 阶段）"""
        self._alternative_thought = thought

    def record_emotion_intensity(
        self,
        initial: Optional[int] = None,
        final: Optional[int] = None,
    ) -> None:
        """记录情绪强度（DEEP 阶段）"""
        if initial is not None:
            self._initial_emotion = max(0, min(10, initial))
        if final is not None:
            self._final_emotion = max(0, min(10, final))

    def record_action_plan(self, plan: str) -> None:
        """记录行动计划（DEEP 阶段）"""
        self._action_plan = plan

    def generate_record(self) -> CBTRestructuringRecord:
        """生成 CBT 认知重构记录

        Returns:
            CBTRestructuringRecord: 结构化记录
        """
        # 判断状态
        if self._depth_reached == "DEEP" and self._turns_in_socratic >= 3:
            status = CBTRecordStatus.COMPLETE
        elif self._depth_reached in ("SHALLOW", "MEDIUM"):
            status = CBTRecordStatus.PARTIAL
        else:
            status = CBTRecordStatus.NOT_STARTED

        # 生成总结
        summary = self._generate_summary(status)

        return CBTRestructuringRecord(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            status=status,
            automatic_thought=self._automatic_thought,
            distortion_type=self._distortion_type,
            supporting_evidence=self._supporting_evidence,
            contradicting_evidence=self._contradicting_evidence,
            alternative_thought=self._alternative_thought,
            initial_emotion_intensity=self._initial_emotion,
            final_emotion_intensity=self._final_emotion,
            action_plan=self._action_plan,
            depth_reached=self._depth_reached,
            turns_in_socratic=self._turns_in_socratic,
            summary=summary,
        )

    def _generate_summary(self, status: CBTRecordStatus) -> str:
        """生成给咨询师看的总结

        Args:
            status: 记录状态

        Returns:
            str: 总结文本
        """
        if status == CBTRecordStatus.NOT_STARTED:
            return "本次对话未进入苏格拉底引导模式。"

        parts = []

        # 深度和轮次
        parts.append(
            f"对话中 AI 进入了苏格拉底引导模式，"
            f"共 {self._turns_in_socratic} 轮，最深达到 {self._depth_reached} 级别。"
        )

        # 自动思维
        if self._automatic_thought:
            parts.append(f"用户识别的自动思维：「{self._automatic_thought}」")

        # 认知扭曲
        if self._distortion_type:
            parts.append(f"识别的认知扭曲类型：{self._distortion_type}")

        # 情绪变化
        if self._initial_emotion is not None and self._final_emotion is not None:
            change = self._initial_emotion - self._final_emotion
            if change > 0:
                parts.append(
                    f"情绪强度从 {self._initial_emotion}/10 降至 {self._final_emotion}/10，"
                    f"改善了 {change} 分。"
                )
            elif change == 0:
                parts.append(
                    f"情绪强度保持在 {self._initial_emotion}/10，无明显变化。"
                )
            else:
                parts.append(
                    f"情绪强度从 {self._initial_emotion}/10 升至 {self._final_emotion}/10，"
                    f"有所上升，需关注。"
                )

        # 行动计划
        if self._action_plan:
            parts.append(f"用户制定的行动计划：{self._action_plan}")

        # 状态判断
        if status == CBTRecordStatus.COMPLETE:
            parts.append("用户完整走过了认知重构流程，建议后续咨询中可继续深化。")
        else:
            parts.append("用户未完成完整认知重构流程，可能在后续对话中继续。")

        return "\n".join(parts)


# ============================================================
# 全局 CBT 记录存储（简化版，实际应使用数据库）
# ============================================================

# 内存存储：conversation_id -> CBTRestructuringRecord
_cbt_records: dict[str, CBTRestructuringRecord] = {}


def store_cbt_record(record: CBTRestructuringRecord) -> None:
    """存储 CBT 记录

    Args:
        record: CBT 认知重构记录
    """
    _cbt_records[record.conversation_id] = record


def get_cbt_record(conversation_id: str) -> Optional[CBTRestructuringRecord]:
    """获取 CBT 记录

    Args:
        conversation_id: 对话 ID

    Returns:
        Optional[CBTRestructuringRecord]: CBT 记录，不存在则返回 None
    """
    return _cbt_records.get(conversation_id)


def get_user_cbt_records(user_id: str) -> list[CBTRestructuringRecord]:
    """获取用户的所有 CBT 记录

    Args:
        user_id: 用户 ID

    Returns:
        list[CBTRestructuringRecord]: 用户的 CBT 记录列表
    """
    return [r for r in _cbt_records.values() if r.user_id == user_id]


def clear_cbt_records() -> None:
    """清空所有 CBT 记录（用于测试）"""
    _cbt_records.clear()
