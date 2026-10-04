"""
治疗关系状态追踪 —— 让系统有"时间感"

核心问题：现在的系统每次对话都是"第一次见面"。它不知道这是第几次对话、
上次聊到哪里、这个用户是好转了还是恶化了。

本模块解决：
    1. 追踪对话次数（session_count）
    2. 追踪用户的核心议题（core_issues）—— 不是关键词，是"议题"
    3. 追踪每个议题的进展（trajectory）
    4. 追踪关系深度（relationship_depth）—— 陌生/熟悉/信任
    5. 追踪上次对话的主题（last_topic）—— 用于"上次你说..."的连续性

设计依据：
    - 心理咨询中的"治疗联盟"（Therapeutic Alliance）需要时间建立
    - 真人治疗师的核心能力之一是"连续性"——"上次你提到你爸住院了，现在怎么样了？"
    - 参考：Norcross & Lambert (2018) 治疗关系对治疗效果的影响

存储：JSON 文件持久化（assessment/outputs/therapeutic_state.json）
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ============================================================
# 关系深度阶段
# ============================================================

class RelationshipDepth(str, Enum):
    """关系深度阶段
    
    影响说话方式、话题深度、干预策略。
    """
    STRANGER = "stranger"       # 陌生期（1-3 次）：礼貌、探索、建立信任
    ACQUAINTED = "acquainted"   # 熟悉期（4-10 次）：可以开玩笑、有共同话题
    TRUSTED = "trusted"         # 信任期（10+ 次）：可以直接指出模式、有内部梗


# ============================================================
# 议题进展
# ============================================================

class IssueTrajectory(str, Enum):
    """议题进展方向"""
    WORSENING = "worsening"     # 恶化
    STABLE = "stable"           # 稳定
    IMPROVING = "improving"     # 好转
    UNKNOWN = "unknown"         # 未知（数据不足）


# ============================================================
# 核心议题
# ============================================================

@dataclass
class CoreIssue:
    """核心议题
    
    Attributes:
        topic: 议题名称（如"学业压力"、"亲子关系"）
        first_seen: 首次出现时间戳
        last_seen: 最近出现时间戳
        mention_count: 提及次数
        trajectory: 进展方向
        risk_contribution: 对风险的贡献度 (0-1)
        keywords: 相关关键词（用于匹配）
    """
    topic: str
    first_seen: float = 0.0
    last_seen: float = 0.0
    mention_count: int = 0
    trajectory: str = "unknown"
    risk_contribution: float = 0.5
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CoreIssue:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================
# 治疗状态
# ============================================================

@dataclass
class TherapeuticState:
    """治疗状态
    
    Attributes:
        user_id: 用户 ID
        session_count: 对话次数
        first_session_at: 首次对话时间
        last_session_at: 最近对话时间
        relationship_depth: 关系深度阶段
        core_issues: 核心议题列表
        last_topic: 上次对话主题
        last_risk_level: 上次风险等级
        intervention_history: 干预历史（最近 10 次）
        trust_signals: 信任信号（用户主动分享的私密信息数量）
    """
    user_id: str
    session_count: int = 0
    first_session_at: float = 0.0
    last_session_at: float = 0.0
    relationship_depth: str = "stranger"
    core_issues: list[dict] = field(default_factory=list)
    last_topic: str = ""
    last_risk_level: str = "low"
    intervention_history: list[dict] = field(default_factory=list)
    trust_signals: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TherapeuticState:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================
# 存储
# ============================================================

_STORAGE_DIR = Path(__file__).resolve().parent / "outputs"
_STORAGE_PATH = _STORAGE_DIR / "therapeutic_state.json"


class TherapeuticStateStore:
    """治疗状态存储"""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._path = storage_path or _STORAGE_PATH
        self._data: dict[str, TherapeuticState] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for uid, state_data in raw.items():
                    self._data[uid] = TherapeuticState.from_dict(state_data)
            except (json.JSONDecodeError, KeyError):
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        raw = {uid: state.to_dict() for uid, state in self._data.items()}
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    def get(self, user_id: str) -> Optional[TherapeuticState]:
        return self._data.get(user_id)

    def put(self, state: TherapeuticState) -> None:
        self._data[state.user_id] = state
        self._save()


# ============================================================
# 全局单例
# ============================================================

_store: Optional[TherapeuticStateStore] = None


def _get_store() -> TherapeuticStateStore:
    global _store
    if _store is None:
        _store = TherapeuticStateStore()
    return _store


# ============================================================
# 核心接口
# ============================================================

def get_or_create_state(user_id: str) -> TherapeuticState:
    """获取或创建用户的治疗状态"""
    store = _get_store()
    state = store.get(user_id)
    if state is None:
        now = time.time()
        state = TherapeuticState(
            user_id=user_id,
            session_count=0,
            first_session_at=now,
            last_session_at=now,
        )
        store.put(state)
    return state


def record_session(
    user_id: str,
    topic: str = "",
    risk_level: str = "low",
    trust_signal: bool = False,
) -> TherapeuticState:
    """记录一次对话会话
    
    Args:
        user_id: 用户 ID
        topic: 本轮主题
        risk_level: 本轮风险等级
        trust_signal: 是否有信任信号（用户主动分享私密信息）
    
    Returns:
        TherapeuticState: 更新后的状态
    """
    state = get_or_create_state(user_id)
    now = time.time()
    
    # 更新会话计数
    state.session_count += 1
    state.last_session_at = now
    
    # 更新关系深度
    if state.session_count >= 10:
        state.relationship_depth = RelationshipDepth.TRUSTED.value
    elif state.session_count >= 4:
        state.relationship_depth = RelationshipDepth.ACQUAINTED.value
    else:
        state.relationship_depth = RelationshipDepth.STRANGER.value
    
    # 更新主题和风险
    state.last_topic = topic
    state.last_risk_level = risk_level
    
    # 更新信任信号
    if trust_signal:
        state.trust_signals += 1
    
    _get_store().put(state)
    return state


def update_core_issues(
    user_id: str,
    detected_topics: list[str],
    risk_level: str = "low",
) -> TherapeuticState:
    """更新核心议题
    
    根据本轮检测到的话题，更新用户的核心议题列表。
    
    Args:
        user_id: 用户 ID
        detected_topics: 本轮检测到的话题列表
        risk_level: 本轮风险等级
    
    Returns:
        TherapeuticState: 更新后的状态
    """
    state = get_or_create_state(user_id)
    now = time.time()
    
    # 话题 → 核心议题的映射
    topic_mapping = {
        "学业": ["考试", "成绩", "作业", "学习", "学校", "老师", "上课"],
        "家庭": ["爸爸", "妈妈", "父母", "家", "弟", "妹", "哥", "姐"],
        "社交": ["朋友", "同学", "闺蜜", "兄弟", "社交", "聊天"],
        "情感": ["喜欢", "爱", "分手", "暗恋", "表白"],
        "自我": ["自己", "我", "感觉", "情绪", "心情"],
    }
    
    for detected_topic in detected_topics:
        # 找到匹配的核心议题类别
        matched_category = None
        for category, keywords in topic_mapping.items():
            if any(kw in detected_topic for kw in keywords):
                matched_category = category
                break
        
        if not matched_category:
            matched_category = detected_topic[:10]  # 用前 10 个字符作为议题名
        
        # 查找或创建该议题
        existing_issue = None
        for issue_data in state.core_issues:
            if issue_data.get("topic") == matched_category:
                existing_issue = issue_data
                break
        
        if existing_issue:
            # 更新已有议题
            existing_issue["last_seen"] = now
            existing_issue["mention_count"] += 1
            # 更新进展（简化：基于风险等级推断）
            if risk_level in ("high", "crisis"):
                existing_issue["trajectory"] = IssueTrajectory.WORSENING.value
            elif risk_level == "low" and existing_issue.get("trajectory") == IssueTrajectory.WORSENING.value:
                existing_issue["trajectory"] = IssueTrajectory.IMPROVING.value
        else:
            # 创建新议题
            new_issue = CoreIssue(
                topic=matched_category,
                first_seen=now,
                last_seen=now,
                mention_count=1,
                trajectory=IssueTrajectory.UNKNOWN.value,
                risk_contribution=0.5,
            )
            state.core_issues.append(new_issue.to_dict())
    
    _get_store().put(state)
    return state


def build_continuity_context(user_id: str) -> str:
    """构建连续性上下文，用于注入 system prompt
    
    让 LLM 知道"这个用户不是第一次来"。
    
    Args:
        user_id: 用户 ID
    
    Returns:
        str: 格式化的连续性上下文
    """
    state = get_or_create_state(user_id)
    
    if state.session_count <= 1:
        return ""  # 第一次对话，不需要连续性上下文
    
    parts = []
    
    # 关系深度
    depth_labels = {
        "stranger": "刚认识",
        "acquainted": "比较熟悉了",
        "trusted": "很信任你",
    }
    depth_label = depth_labels.get(state.relationship_depth, "刚认识")
    parts.append(f"【关系阶段】你们已经聊了 {state.session_count} 次，ta 对你{depth_label}。")
    
    # 核心议题
    if state.core_issues:
        issue_names = [issue.get("topic", "") for issue in state.core_issues[:3]]
        parts.append(f"【核心议题】ta 反复提到的话题：{', '.join(issue_names)}。")
    
    # 上次主题
    if state.last_topic:
        topic_preview = state.last_topic[:30]
        parts.append(f'【上次聊到】上次 ta 提到的是「{topic_preview}」。')
    
    # 进展提示
    worsening_issues = [
        issue for issue in state.core_issues
        if issue.get("trajectory") == IssueTrajectory.WORSENING.value
    ]
    if worsening_issues:
        issue_names = [issue.get("topic", "") for issue in worsening_issues[:2]]
        parts.append(f"【注意】这些议题在恶化：{', '.join(issue_names)}。")
    
    improving_issues = [
        issue for issue in state.core_issues
        if issue.get("trajectory") == IssueTrajectory.IMPROVING.value
    ]
    if improving_issues:
        issue_names = [issue.get("topic", "") for issue in improving_issues[:2]]
        parts.append(f"【好消息】这些议题在好转：{', '.join(issue_names)}。可以顺着这个势头。")
    
    return "\n".join(parts)


def reset_store() -> None:
    """重置存储（测试用）"""
    global _store
    _store = None
