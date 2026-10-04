"""
干预效果追踪 —— 让系统知道"什么对这个用户有效"

核心问题：现在的干预策略是"一刀切"的。CBT、苏格拉底提问、认知重构——
这些技术对不同人效果不同。系统不知道上次的认知重构有没有帮到这个用户。

本模块解决：
    1. 记录每次干预使用的策略
    2. 记录用户反应（情绪变化、语言反馈）
    3. 计算每种策略对该用户的有效率
    4. 推荐下次优先使用的策略

设计依据：
    - 心理治疗中的"反馈知情治疗"（Feedback-Informed Treatment）
    - 不同用户对不同干预的反应差异很大（Lambert, 2010）
    - 系统应该学习"什么对这个用户有效"，而不是套用通用策略

存储：JSON 文件持久化（assessment/outputs/intervention_effectiveness.json）
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ============================================================
# 干预策略类型
# ============================================================

class InterventionStrategy(str, Enum):
    """干预策略类型"""
    COGNITIVE_RESTRUCTURE = "cognitive_restructure"  # 认知重构
    SOCRATIC_QUESTIONING = "socratic_questioning"    # 苏格拉底提问
    BEHAVIORAL_ACTIVATION = "behavioral_activation"  # 行为激活
    EMPATHIC_LISTENING = "empathic_listening"        # 共情倾听
    MINDFULNESS = "mindfulness"                      # 正念
    PSYCHOEDUCATION = "psychoeducation"              # 心理教育
    PROBLEM_SOLVING = "problem_solving"              # 问题解决
    VALIDATION = "validation"                        # 情感确认
    REFERRAL = "referral"                            # 转介


# ============================================================
# 干预记录
# ============================================================

@dataclass
class InterventionRecord:
    """干预记录
    
    Attributes:
        id: 记录 ID
        user_id: 用户 ID
        strategy: 使用的策略
        timestamp: 干预时间
        user_response: 用户反应（positive/negative/neutral）
        emotion_before: 干预前情绪（负性情绪强度 0-1）
        emotion_after: 干预后情绪（负性情绪强度 0-1）
        session_id: 对话 ID
        notes: 备注
    """
    id: str
    user_id: str
    strategy: str
    timestamp: float = 0.0
    user_response: str = "neutral"  # positive / negative / neutral
    emotion_before: float = 0.5
    emotion_after: float = 0.5
    session_id: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> InterventionRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================
# 策略效果统计
# ============================================================

@dataclass
class StrategyEffectiveness:
    """策略效果统计
    
    Attributes:
        strategy: 策略名称
        total_uses: 总使用次数
        positive_count: 正面反应次数
        effectiveness_rate: 有效率 (0-1)
        avg_emotion_improvement: 平均情绪改善度
    """
    strategy: str
    total_uses: int = 0
    positive_count: int = 0
    effectiveness_rate: float = 0.0
    avg_emotion_improvement: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> StrategyEffectiveness:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================
# 存储
# ============================================================

_STORAGE_DIR = Path(__file__).resolve().parent / "outputs"
_STORAGE_PATH = _STORAGE_DIR / "intervention_effectiveness.json"


class InterventionEffectivenessStore:
    """干预效果存储"""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._path = storage_path or _STORAGE_PATH
        self._records: list[InterventionRecord] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._records = [InterventionRecord.from_dict(r) for r in raw]
            except (json.JSONDecodeError, KeyError):
                self._records = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        raw = [r.to_dict() for r in self._records]
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    def add_record(self, record: InterventionRecord) -> None:
        self._records.append(record)
        self._save()

    def get_user_records(self, user_id: str, limit: int = 50) -> list[InterventionRecord]:
        """获取用户的干预记录"""
        user_records = [r for r in self._records if r.user_id == user_id]
        # 按时间倒序
        user_records.sort(key=lambda r: r.timestamp, reverse=True)
        return user_records[:limit]


# ============================================================
# 全局单例
# ============================================================

_store: Optional[InterventionEffectivenessStore] = None


def _get_store() -> InterventionEffectivenessStore:
    global _store
    if _store is None:
        _store = InterventionEffectivenessStore()
    return _store


# ============================================================
# 核心接口
# ============================================================

def record_intervention(
    user_id: str,
    strategy: str,
    user_response: str = "neutral",
    emotion_before: float = 0.5,
    emotion_after: float = 0.5,
    session_id: str = "",
) -> InterventionRecord:
    """记录一次干预
    
    Args:
        user_id: 用户 ID
        strategy: 使用的策略
        user_response: 用户反应（positive/negative/neutral）
        emotion_before: 干预前情绪
        emotion_after: 干预后情绪
        session_id: 对话 ID
    
    Returns:
        InterventionRecord: 创建的记录
    """
    import uuid
    
    record = InterventionRecord(
        id=str(uuid.uuid4()),
        user_id=user_id,
        strategy=strategy,
        timestamp=time.time(),
        user_response=user_response,
        emotion_before=emotion_before,
        emotion_after=emotion_after,
        session_id=session_id,
    )
    _get_store().add_record(record)
    return record


def get_strategy_effectiveness(user_id: str) -> dict[str, StrategyEffectiveness]:
    """获取用户各策略的效果统计
    
    Args:
        user_id: 用户 ID
    
    Returns:
        dict[str, StrategyEffectiveness]: 策略名 → 效果统计
    """
    records = _get_store().get_user_records(user_id, limit=100)
    
    # 按策略分组统计
    strategy_stats: dict[str, list[InterventionRecord]] = {}
    for record in records:
        if record.strategy not in strategy_stats:
            strategy_stats[record.strategy] = []
        strategy_stats[record.strategy].append(record)
    
    # 计算效果
    result = {}
    for strategy, recs in strategy_stats.items():
        total = len(recs)
        positive = sum(1 for r in recs if r.user_response == "positive")
        avg_improvement = sum(r.emotion_before - r.emotion_after for r in recs) / total if total > 0 else 0
        
        result[strategy] = StrategyEffectiveness(
            strategy=strategy,
            total_uses=total,
            positive_count=positive,
            effectiveness_rate=positive / total if total > 0 else 0,
            avg_emotion_improvement=avg_improvement,
        )
    
    return result


def recommend_strategy(user_id: str) -> str:
    """推荐下次使用的策略
    
    基于历史效果推荐最有效的策略。如果没有历史数据，返回默认的共情倾听。
    
    Args:
        user_id: 用户 ID
    
    Returns:
        str: 推荐的策略名称
    """
    effectiveness = get_strategy_effectiveness(user_id)
    
    if not effectiveness:
        # 没有历史数据，使用默认策略
        return InterventionStrategy.EMPATHIC_LISTENING.value
    
    # 按有效率排序，选择最有效的
    sorted_strategies = sorted(
        effectiveness.values(),
        key=lambda s: (s.effectiveness_rate, s.avg_emotion_improvement),
        reverse=True,
    )
    
    # 如果所有策略都没用过或效果一样，返回默认
    if sorted_strategies[0].effectiveness_rate == 0:
        return InterventionStrategy.EMPATHIC_LISTENING.value
    
    return sorted_strategies[0].strategy


def build_intervention_context(user_id: str) -> str:
    """构建干预效果上下文，用于注入 system prompt
    
    让 LLM 知道"什么对这个用户有效"。
    
    Args:
        user_id: 用户 ID
    
    Returns:
        str: 格式化的干预效果上下文
    """
    effectiveness = get_strategy_effectiveness(user_id)
    
    if not effectiveness:
        return ""
    
    parts = ["【干预效果历史】以下是你对这个用户用过的策略及其效果："]
    
    for strategy, stats in effectiveness.items():
        if stats.total_uses >= 2:  # 只显示用过 2 次以上的
            strategy_label = _strategy_label(strategy)
            rate_pct = int(stats.effectiveness_rate * 100)
            parts.append(
                f"- {strategy_label}：用了 {stats.total_uses} 次，"
                f"有效率 {rate_pct}%，"
                f"{'效果好' if stats.effectiveness_rate >= 0.6 else '效果一般' if stats.effectiveness_rate >= 0.3 else '效果差'}"
            )
    
    if len(parts) == 1:
        return ""  # 没有足够的历史数据
    
    # 推荐
    recommended = recommend_strategy(user_id)
    recommended_label = _strategy_label(recommended)
    parts.append(f"【建议】基于历史，{recommended_label} 对这个用户最有效。")
    
    return "\n".join(parts)


def _strategy_label(strategy: str) -> str:
    """策略标签"""
    labels = {
        "cognitive_restructure": "认知重构",
        "socratic_questioning": "苏格拉底提问",
        "behavioral_activation": "行为激活",
        "empathic_listening": "共情倾听",
        "mindfulness": "正念",
        "psychoeducation": "心理教育",
        "problem_solving": "问题解决",
        "validation": "情感确认",
        "referral": "转介",
    }
    return labels.get(strategy, strategy)


def reset_store() -> None:
    """重置存储（测试用）"""
    global _store
    _store = None
