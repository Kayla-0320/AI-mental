"""
共同活动模块 —— 非对话互动

核心问题：青少年不是通过"理性对话"建立关系的，他们通过**共同活动**。
一起打游戏、一起听音乐、一起画画、一起发呆。

现在的系统只有"对话"这一种互动方式。但很多青少年在"说话"时会封闭自己，
在"一起做点什么"时才会打开。

本模块提供：
    1. 共同活动类型定义
    2. 活动状态追踪
    3. 活动建议（基于用户兴趣和关系阶段）
    4. 活动后的对话衔接

设计依据：
    - 青少年发展心理学：同伴活动是青少年建立关系的核心方式
    - 表达性艺术治疗：绘画、音乐等非语言活动可以绕过防御
    - 行为激活：抑郁症治疗中，重新参与愉快活动是关键

存储：JSON 文件持久化（assessment/outputs/shared_activities.json）
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ============================================================
# 活动类型
# ============================================================

class ActivityType(str, Enum):
    """共同活动类型"""
    MUSIC = "music"               # 一起听音乐
    DRAWING = "drawing"           # 一起画画
    BREATHING = "breathing"       # 一起正念呼吸
    MOOD_CARD = "mood_card"       # 情绪卡牌
    FREE_WRITE = "free_write"     # 自由书写
    DREAM_BOARD = "dream_board"   # 梦想板
    GRATITUDE = "gratitude"       # 感恩练习
    MOVEMENT = "movement"         # 身体活动


# ============================================================
# 活动记录
# ============================================================

@dataclass
class ActivityRecord:
    """活动记录
    
    Attributes:
        id: 记录 ID
        user_id: 用户 ID
        activity_type: 活动类型
        timestamp: 活动时间
        duration_seconds: 持续时间（秒）
        user_engagement: 用户参与度 (1-5)
        emotion_after: 活动后情绪
        notes: 备注
    """
    id: str
    user_id: str
    activity_type: str
    timestamp: float = 0.0
    duration_seconds: int = 0
    user_engagement: int = 3
    emotion_after: str = "neutral"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ActivityRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================
# 活动建议
# ============================================================

@dataclass
class ActivitySuggestion:
    """活动建议
    
    Attributes:
        activity_type: 活动类型
        reason: 推荐理由
        prompt: 邀请话术
    """
    activity_type: str
    reason: str
    prompt: str


# ============================================================
# 存储
# ============================================================

_STORAGE_DIR = Path(__file__).resolve().parent / "outputs"
_STORAGE_PATH = _STORAGE_DIR / "shared_activities.json"


class ActivityStore:
    """活动存储"""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._path = storage_path or _STORAGE_PATH
        self._records: list[ActivityRecord] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._records = [ActivityRecord.from_dict(r) for r in raw]
            except (json.JSONDecodeError, KeyError):
                self._records = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        raw = [r.to_dict() for r in self._records]
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    def add_record(self, record: ActivityRecord) -> None:
        self._records.append(record)
        self._save()

    def get_user_records(self, user_id: str, limit: int = 20) -> list[ActivityRecord]:
        user_records = [r for r in self._records if r.user_id == user_id]
        user_records.sort(key=lambda r: r.timestamp, reverse=True)
        return user_records[:limit]


# ============================================================
# 全局单例
# ============================================================

_store: Optional[ActivityStore] = None


def _get_store() -> ActivityStore:
    global _store
    if _store is None:
        _store = ActivityStore()
    return _store


# ============================================================
# 活动建议逻辑
# ============================================================

# 活动类型 → 适用场景
_ACTIVITY_AFFORDANCES = {
    ActivityType.MUSIC: {
        "mood_match": ["sad", "anxious"],  # 适合的情绪
        "relationship_min": "stranger",     # 最低关系要求
        "prompt_template": "要不要一起听首歌？有时候音乐比说话更能表达心情。",
    },
    ActivityType.DRAWING: {
        "mood_match": ["anxious", "angry"],
        "relationship_min": "acquainted",
        "prompt_template": "要不要随便画点什么？不用画得好，就是让手动起来。",
    },
    ActivityType.BREATHING: {
        "mood_match": ["anxious", "angry"],
        "relationship_min": "stranger",
        "prompt_template": "要不要一起做几个深呼吸？就一分钟。",
    },
    ActivityType.MOOD_CARD: {
        "mood_match": ["neutral", "sad"],
        "relationship_min": "acquainted",
        "prompt_template": "要不要抽一张情绪卡？看看今天是什么颜色。",
    },
    ActivityType.FREE_WRITE: {
        "mood_match": ["sad", "angry"],
        "relationship_min": "acquainted",
        "prompt_template": "要不要随便写点什么？写完可以给我看，也可以不看。",
    },
    ActivityType.GRATITUDE: {
        "mood_match": ["neutral", "happy"],
        "relationship_min": "trusted",
        "prompt_template": "今天有没有什么小事让你觉得还不错的？",
    },
    ActivityType.MOVEMENT: {
        "mood_match": ["sad", "anxious"],
        "relationship_min": "stranger",
        "prompt_template": "要不要站起来动一动？伸个懒腰也行。",
    },
}


def suggest_activity(
    user_id: str,
    current_mood: str = "neutral",
    relationship_depth: str = "stranger",
) -> Optional[ActivitySuggestion]:
    """建议一个共同活动
    
    Args:
        user_id: 用户 ID
        current_mood: 当前情绪
        relationship_depth: 关系深度
    
    Returns:
        Optional[ActivitySuggestion]: 活动建议，没有合适活动时返回 None
    """
    # 关系深度排序
    depth_order = {"stranger": 0, "acquainted": 1, "trusted": 2}
    current_depth_level = depth_order.get(relationship_depth, 0)
    
    # 查找合适的活动
    suitable_activities = []
    for activity_type, affordance in _ACTIVITY_AFFORDANCES.items():
        # 检查关系要求
        required_depth = depth_order.get(affordance["relationship_min"], 0)
        if current_depth_level < required_depth:
            continue
        
        # 检查情绪匹配
        if current_mood in affordance["mood_match"]:
            suitable_activities.append((activity_type, affordance))
    
    if not suitable_activities:
        return None
    
    # 检查历史：避免重复推荐最近做过的活动
    history = _get_store().get_user_records(user_id, limit=5)
    recent_activities = {r.activity_type for r in history}
    
    # 过滤掉最近做过的
    fresh_activities = [
        (t, a) for t, a in suitable_activities
        if t.value not in recent_activities
    ]
    
    if not fresh_activities:
        # 如果都做过了，从全部中选
        fresh_activities = suitable_activities
    
    # 选第一个
    activity_type, affordance = fresh_activities[0]
    
    return ActivitySuggestion(
        activity_type=activity_type.value,
        reason=f"适合当前{current_mood}情绪",
        prompt=affordance["prompt_template"],
    )


def record_activity(
    user_id: str,
    activity_type: str,
    duration_seconds: int = 0,
    user_engagement: int = 3,
    emotion_after: str = "neutral",
) -> ActivityRecord:
    """记录一次活动
    
    Args:
        user_id: 用户 ID
        activity_type: 活动类型
        duration_seconds: 持续时间
        user_engagement: 用户参与度
        emotion_after: 活动后情绪
    
    Returns:
        ActivityRecord: 创建的记录
    """
    import uuid
    
    record = ActivityRecord(
        id=str(uuid.uuid4()),
        user_id=user_id,
        activity_type=activity_type,
        timestamp=time.time(),
        duration_seconds=duration_seconds,
        user_engagement=user_engagement,
        emotion_after=emotion_after,
    )
    _get_store().add_record(record)
    return record


def build_activity_context(user_id: str) -> str:
    """构建活动上下文，用于注入 system prompt
    
    让 LLM 知道"这个用户做过什么活动"。
    
    Args:
        user_id: 用户 ID
    
    Returns:
        str: 格式化的活动上下文
    """
    history = _get_store().get_user_records(user_id, limit=10)
    
    if not history:
        return ""
    
    parts = ["【共同活动历史】你们一起做过这些活动："]
    
    activity_labels = {
        "music": "听音乐",
        "drawing": "画画",
        "breathing": "正念呼吸",
        "mood_card": "情绪卡牌",
        "free_write": "自由书写",
        "dream_board": "梦想板",
        "gratitude": "感恩练习",
        "movement": "身体活动",
    }
    
    for record in history[:5]:
        label = activity_labels.get(record.activity_type, record.activity_type)
        engagement_label = {1: "很低", 2: "较低", 3: "一般", 4: "较高", 5: "很高"}
        eng = engagement_label.get(record.user_engagement, "一般")
        parts.append(f"- {label}：参与度{eng}，活动后情绪{record.emotion_after}")
    
    return "\n".join(parts)


def reset_store() -> None:
    """重置存储（测试用）"""
    global _store
    _store = None
