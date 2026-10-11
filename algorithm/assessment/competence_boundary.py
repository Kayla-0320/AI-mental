"""
能力边界感知 —— 让系统知道自己什么时候"不行"

核心问题：现在的危机升级是"检测到关键词 → 弹热线号码"。但真正的临床判断是：
"我觉得我帮不了你。你需要一个真人治疗师。"

本模块解决：
    1. 追踪连续恶化次数（连续 N 次高风险 → 建议转介）
    2. 识别超出 AI 能力范围的问题（创伤、虐待、严重精神疾病）
    3. 尊重用户的"你帮不了我"信号
    4. 生成温和的转介建议（不是冷冰冰地弹号码）

设计依据：
    - AI 不是万能的，必须知道自己的局限
    - 真人治疗师会说"我觉得你需要一个比我更专业的人"
    - 转介不是失败，是负责任的表现

存储：JSON 文件持久化（assessment/outputs/competence_boundary.json）
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ============================================================
# 超出能力范围的信号
# ============================================================

# 这些信号表明用户的问题可能超出 AI 的能力范围
_OUT_OF_SCOPE_SIGNALS = [
    # 创伤相关
    "被强奸", "被性侵", "猥亵", "性侵犯",
    # 虐待相关
    "被虐待", "家暴", "被打", "被妈妈打", "被爸爸打",
    # 严重精神疾病
    "精神分裂", "双相情感障碍", "躁郁症", "幻觉", "妄想",
    # 物质滥用
    "吸毒", "药物滥用", "酗酒", "酒瘾",
    # 进食障碍
    "厌食症", "暴食症", "催吐",
]

# 用户明确表达"你帮不了我"的信号
_USER_REJECTION_SIGNALS = [
    "你帮不了我", "你不懂", "你根本不明白",
    "AI 没用", "跟你说了也没用",
    "我想找真人", "我想找真人咨询师",
    "你只是个 AI", "机器不懂人",
]


# ============================================================
# 边界状态
# ============================================================

@dataclass
class BoundaryState:
    """能力边界状态
    
    Attributes:
        user_id: 用户 ID
        consecutive_high_risk_count: 连续高风险次数
        last_high_risk_at: 最近一次高风险时间
        out_of_scope_detected: 是否检测到超出能力范围的信号
        out_of_scope_signals: 超出范围的信号列表
        user_rejection_count: 用户拒绝次数
        should_refer: 是否应该转介
        referral_reason: 转介原因
    """
    user_id: str
    consecutive_high_risk_count: int = 0
    last_high_risk_at: float = 0.0
    out_of_scope_detected: bool = False
    out_of_scope_signals: list[str] = field(default_factory=list)
    user_rejection_count: int = 0
    should_refer: bool = False
    referral_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BoundaryState:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================
# 存储
# ============================================================

_STORAGE_DIR = Path(__file__).resolve().parent / "outputs"
_STORAGE_PATH = _STORAGE_DIR / "competence_boundary.json"


class BoundaryStateStore:
    """边界状态存储"""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self._path = storage_path or _STORAGE_PATH
        self._data: dict[str, BoundaryState] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for uid, state_data in raw.items():
                    self._data[uid] = BoundaryState.from_dict(state_data)
            except (json.JSONDecodeError, KeyError):
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        raw = {uid: state.to_dict() for uid, state in self._data.items()}
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    def get(self, user_id: str) -> Optional[BoundaryState]:
        return self._data.get(user_id)

    def put(self, state: BoundaryState) -> None:
        self._data[state.user_id] = state
        self._save()


# ============================================================
# 全局单例
# ============================================================

_store: Optional[BoundaryStateStore] = None


def _get_store() -> BoundaryStateStore:
    global _store
    if _store is None:
        _store = BoundaryStateStore()
    return _store


def get_or_create_state(user_id: str) -> BoundaryState:
    """获取或创建用户的边界状态"""
    store = _get_store()
    state = store.get(user_id)
    if state is None:
        state = BoundaryState(user_id=user_id)
        store.put(state)
    return state


# ============================================================
# 核心接口
# ============================================================

def update_boundary_state(
    user_id: str,
    user_message: str,
    risk_level: str,
) -> BoundaryState:
    """更新能力边界状态
    
    Args:
        user_id: 用户 ID
        user_message: 用户本轮消息
        risk_level: 本轮风险等级
    
    Returns:
        BoundaryState: 更新后的状态
    """
    state = get_or_create_state(user_id)
    now = time.time()
    message_lower = user_message.lower()
    
    # 1. 追踪连续高风险
    if risk_level in ("high", "crisis"):
        state.consecutive_high_risk_count += 1
        state.last_high_risk_at = now
    else:
        # 风险降低，重置计数
        state.consecutive_high_risk_count = 0
    
    # 2. 检测超出能力范围的信号
    detected_out_of_scope = []
    for signal in _OUT_OF_SCOPE_SIGNALS:
        if signal in message_lower:
            detected_out_of_scope.append(signal)
    
    if detected_out_of_scope:
        state.out_of_scope_detected = True
        state.out_of_scope_signals.extend(detected_out_of_scope)
        # 去重
        state.out_of_scope_signals = list(set(state.out_of_scope_signals))
    
    # 3. 检测用户拒绝信号
    for signal in _USER_REJECTION_SIGNALS:
        if signal in message_lower:
            state.user_rejection_count += 1
            break
    
    # 4. 判断是否应该转介
    should_refer = False
    referral_reason = ""
    
    # 条件 1：连续 5 次高风险
    if state.consecutive_high_risk_count >= 5:
        should_refer = True
        referral_reason = (
            f"你已经连续 {state.consecutive_high_risk_count} 次对话都处于高风险状态。"
            "我觉得你需要一个比我更专业的人来帮助你。"
        )
    
    # 条件 2：检测到超出能力范围的信号
    if state.out_of_scope_detected:
        should_refer = True
        referral_reason = (
            "你提到了一些我觉得自己可能帮不了你的事情。"
            "这些事情需要更专业的帮助，我建议你联系一个真人咨询师。"
        )
    
    # 条件 3：用户明确拒绝
    if state.user_rejection_count >= 2:
        should_refer = True
        referral_reason = (
            "我感觉你不太信任我能帮到你。"
            "这完全可以理解。也许一个真人咨询师会更适合你。"
        )
    
    state.should_refer = should_refer
    state.referral_reason = referral_reason
    
    _get_store().put(state)
    return state


def build_boundary_context(user_id: str) -> str:
    """构建能力边界上下文，用于注入 system prompt
    
    让 LLM 知道"什么时候该建议转介"。
    
    Args:
        user_id: 用户 ID
    
    Returns:
        str: 格式化的边界上下文
    """
    state = get_or_create_state(user_id)
    
    if not state.should_refer:
        return ""
    
    parts = [
        "【能力边界提醒】",
        "这个用户的情况可能超出了你的能力范围。",
        f"原因：{state.referral_reason}",
        "",
        "【你应该做的】",
        "1. 不要假装你能解决所有问题。",
        "2. 温和但明确地建议转介：'我觉得你需要一个比我更专业的人。我帮你联系？'",
        "3. 不要冷冰冰地弹热线号码，要像朋友一样建议。",
        "4. 如果用户拒绝转介，尊重 ta 的选择，但不要假装问题不存在。",
    ]
    
    return "\n".join(parts)


def should_refer_now(user_id: str) -> tuple[bool, str]:
    """判断当前是否应该转介
    
    Args:
        user_id: 用户 ID
    
    Returns:
        tuple[bool, str]: (是否应该转介, 转介原因)
    """
    state = get_or_create_state(user_id)
    return state.should_refer, state.referral_reason


def reset_store() -> None:
    """重置存储（测试用）"""
    global _store
    _store = None
