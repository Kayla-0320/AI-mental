"""
用户长期记忆系统 —— AI 倾诉的跨会话记忆

功能：
    1. 存储用户主动透露的稳定事实（家庭、学业、兴趣等）
    2. 按话题相关性召回记忆，注入对话上下文
    3. 超期未引用的记忆自动降权
    4. 用户可随时删除自己的记忆

安全约束：
    - 不存储危机信号（属于审计器职责）
    - 不存储诊断性信息（属于咨询师职责）
    - 不存储情绪状态（属于基线系统职责）
    - 所有记忆提取经过内容安全检查

设计依据：
    - 心理咨询中的"个案概念化"需要积累来访者背景信息
    - 跨会话记忆让用户感到"被记住"，增强治疗联盟
    - 参考：Kazdin (2017) 青少年心理治疗中的个案概念化

存储：JSON 文件持久化（algorithm/assessment/outputs/user_memories.json）
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ============================================================
# 记忆分类
# ============================================================

class MemoryCategory(str, Enum):
    """记忆分类

    只存储用户主动透露的稳定事实，不存储情绪/诊断/危机信息。
    """
    FAMILY = "family"           # 家庭（父母职业、兄弟姐妹、家庭结构等）
    ACADEMIC = "academic"       # 学业（年级、学校、考试、成绩等）
    INTEREST = "interest"       # 兴趣爱好（运动、音乐、游戏等）
    SOCIAL = "social"           # 社交（朋友、同学、师生关系等）
    HEALTH = "health"           # 健康（睡眠、饮食、运动习惯等）
    OTHER = "other"             # 其他


# ============================================================
# 数据结构
# ============================================================

@dataclass
class UserMemory:
    """用户记忆条目

    Attributes:
        id: 记忆唯一标识
        user_id: 用户 ID
        fact: 记忆内容（一条事实陈述）
        category: 记忆分类
        confidence: 提取置信度 (0-1)
        created_at: 创建时间戳
        last_referenced: 最近一次被引用的时间戳
        reference_count: 被引用次数
        source_conversation_id: 来源对话 ID
    """
    id: str
    user_id: str
    fact: str
    category: str = "other"
    confidence: float = 0.8
    created_at: float = 0.0
    last_referenced: float = 0.0
    reference_count: int = 0
    source_conversation_id: str = ""

    def to_dict(self) -> dict:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> UserMemory:
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================
# 内容安全检查
# ============================================================

# 禁止存储的内容关键词（危机信号、诊断信息、情绪状态）
_FORBIDDEN_PATTERNS = [
    # 危机信号
    "自杀", "自残", "不想活", "想死", "死了算了", "活不下去",
    "伤害自己", "结束生命", "跳楼", "割腕",
    # 诊断性信息
    "患有", "确诊", "诊断", "抑郁症", "焦虑症", "双相", "精神分裂",
    "PTSD", "强迫症", "多动症", "ADHD",
    # 情绪状态（属于基线系统职责）
    "感到悲伤", "感到焦虑", "感到愤怒", "情绪低落",
]


def is_memory_safe(fact: str) -> bool:
    """检查记忆内容是否安全可存储

    拒绝存储：危机信号、诊断信息、情绪状态。

    Args:
        fact: 待检查的记忆内容

    Returns:
        bool: True 表示安全可存储，False 表示应拒绝
    """
    fact_lower = fact.lower()
    for pattern in _FORBIDDEN_PATTERNS:
        if pattern in fact_lower:
            return False
    return True


# ============================================================
# 记忆存储
# ============================================================

# 默认存储路径
_STORAGE_DIR = Path(__file__).resolve().parent / "outputs"
_STORAGE_PATH = _STORAGE_DIR / "user_memories.json"


class MemoryStore:
    """用户记忆存储

    使用 JSON 文件持久化，支持：
    - 添加/删除记忆
    - 按话题相关性召回
    - 超期记忆降权
    - 按用户 ID 查询

    使用方式：
        store = MemoryStore()
        memory = store.add_memory("user_123", "正在准备中考", "academic")
        memories = store.get_relevant_memories("user_123", "学习压力")
        store.delete_memory("user_123", memory.id)
    """

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self._storage_path = Path(storage_path) if storage_path else _STORAGE_PATH
        self._memories: list[UserMemory] = []
        self._load()

    def _load(self) -> None:
        """从 JSON 文件加载记忆"""
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._memories = [UserMemory.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError):
                self._memories = []
        else:
            self._memories = []

    def _save(self) -> None:
        """保存记忆到 JSON 文件"""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, "w", encoding="utf-8") as f:
            json.dump(
                [m.to_dict() for m in self._memories],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def add_memory(
        self,
        user_id: str,
        fact: str,
        category: str = "other",
        confidence: float = 0.8,
        conversation_id: str = "",
    ) -> Optional[UserMemory]:
        """添加一条记忆

        安全检查：拒绝存储危机信号、诊断信息、情绪状态。

        Args:
            user_id: 用户 ID
            fact: 记忆内容
            category: 记忆分类
            confidence: 提取置信度 (0-1)
            conversation_id: 来源对话 ID

        Returns:
            Optional[UserMemory]: 创建的记忆，安全检查不通过时返回 None
        """
        # 内容安全检查
        if not is_memory_safe(fact):
            return None

        now = time.time()
        memory = UserMemory(
            id=str(uuid.uuid4()),
            user_id=user_id,
            fact=fact,
            category=category,
            confidence=max(0.0, min(1.0, confidence)),
            created_at=now,
            last_referenced=now,
            reference_count=0,
            source_conversation_id=conversation_id,
        )
        self._memories.append(memory)
        self._save()
        return memory

    def get_relevant_memories(
        self,
        user_id: str,
        current_topic: str = "",
        top_k: int = 5,
    ) -> list[UserMemory]:
        """获取与当前话题相关的记忆

        排序策略：
            1. 关键词匹配得分（话题与记忆内容的重叠度）
            2. 时间衰减因子（最近引用的记忆权重更高）
            3. 引用次数加成（被多次引用的记忆更稳定）
            4. 置信度加权

        Args:
            user_id: 用户 ID
            current_topic: 当前话题（用于相关性匹配）
            top_k: 返回前 k 条记忆

        Returns:
            list[UserMemory]: 按相关性排序的记忆列表
        """
        user_memories = [m for m in self._memories if m.user_id == user_id]
        if not user_memories:
            return []

        now = time.time()
        scored = []
        for mem in user_memories:
            # 关键词匹配得分
            topic_score = self._compute_topic_score(current_topic, mem.fact)

            # 时间衰减因子（30 天半衰期）
            days_since_ref = (now - mem.last_referenced) / 86400
            recency_score = 0.5 ** (days_since_ref / 30)

            # 引用次数加成（每次引用 +0.1，上限 0.5）
            ref_bonus = min(0.5, mem.reference_count * 0.1)

            # 综合得分
            score = (
                topic_score * 0.5
                + recency_score * 0.3
                + ref_bonus * 0.1
                + mem.confidence * 0.1
            )
            scored.append((score, mem))

        # 按得分降序排列
        scored.sort(key=lambda x: x[0], reverse=True)

        # 更新引用时间
        result = [mem for _, mem in scored[:top_k]]
        for mem in result:
            mem.last_referenced = now
            mem.reference_count += 1
        self._save()

        return result

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """删除一条记忆

        用户可随时删除自己的记忆。

        Args:
            user_id: 用户 ID
            memory_id: 记忆 ID

        Returns:
            bool: 是否成功删除
        """
        original_len = len(self._memories)
        self._memories = [
            m for m in self._memories
            if not (m.id == memory_id and m.user_id == user_id)
        ]
        if len(self._memories) < original_len:
            self._save()
            return True
        return False

    def get_all_memories(self, user_id: str) -> list[UserMemory]:
        """获取用户的所有记忆

        Args:
            user_id: 用户 ID

        Returns:
            list[UserMemory]: 用户的所有记忆
        """
        return [m for m in self._memories if m.user_id == user_id]

    def decay_old_memories(self, days: int = 90) -> int:
        """超期未引用的记忆降权

        超过指定天数未被引用的记忆，置信度降低 50%。

        Args:
            days: 超期天数阈值（默认 90 天）

        Returns:
            int: 被降权的记忆数量
        """
        now = time.time()
        threshold = days * 86400
        decayed_count = 0

        for mem in self._memories:
            if now - mem.last_referenced > threshold:
                mem.confidence *= 0.5
                decayed_count += 1

        if decayed_count > 0:
            self._save()

        return decayed_count

    def clear_user_memories(self, user_id: str) -> int:
        """清空用户的所有记忆

        Args:
            user_id: 用户 ID

        Returns:
            int: 被删除的记忆数量
        """
        original_len = len(self._memories)
        self._memories = [m for m in self._memories if m.user_id != user_id]
        if len(self._memories) < original_len:
            self._save()
        return original_len - len(self._memories)

    @staticmethod
    def _compute_topic_score(topic: str, fact: str) -> float:
        """计算话题与记忆内容的关键词匹配得分

        使用简单的字符重叠率作为相关性指标。
        生产环境可替换为 embedding 余弦相似度。

        Args:
            topic: 当前话题
            fact: 记忆内容

        Returns:
            float: 匹配得分 (0-1)
        """
        if not topic or not fact:
            return 0.3  # 无话题时给基础分

        # 分词：按字符 bigram 匹配（中文适用）
        topic_chars = set(topic)
        fact_chars = set(fact)
        if not topic_chars:
            return 0.0

        overlap = topic_chars & fact_chars
        return len(overlap) / len(topic_chars)


# ============================================================
# 全局单例（避免重复加载）
# ============================================================

_global_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """获取全局记忆存储实例

    Returns:
        MemoryStore: 全局记忆存储
    """
    global _global_store
    if _global_store is None:
        _global_store = MemoryStore()
    return _global_store


def reset_memory_store() -> None:
    """重置全局记忆存储（用于测试）"""
    global _global_store
    _global_store = None
