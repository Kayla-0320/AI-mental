"""
转介系统模块 —— 将高风险用户转介至专业机构或咨询师

转介场景：
    1. AI 咨询中检测到高风险 → 转介至人工咨询师
    2. 咨询师发现超出能力范围 → 转介至专科机构
    3. 危机情况 → 转介至紧急医疗服务

转介流程：
    1. 生成转介建议（基于风险等级和专科匹配）
    2. 匹配可用接收方（咨询师/机构）
    3. 生成转介摘要（包含关键上下文，不含原始对话）
    4. 发送转介请求并追踪状态

核心接口：
    - create_referral(user_id, reason, risk_level) -> ReferralResult
    - match_receiver(referral) -> list[ReceiverCandidate]
    - generate_referral_summary(referral) -> str
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from shared.dataclasses import CrisisAlert, RiskLevel


# ============================================================
# 转介类型
# ============================================================

class ReferralType(str, Enum):
    """转介类型"""
    AI_TO_CONSULTANT = "ai_to_consultant"       # AI → 人工咨询师
    CONSULTANT_TO_SPECIALIST = "consultant_to_specialist"  # 咨询师 → 专科
    CRISIS_TO_EMERGENCY = "crisis_to_emergency"  # 危机 → 紧急医疗
    PEER_SUPPORT = "peer_support"               # 同伴支持转介


class ReferralStatus(str, Enum):
    """转介状态"""
    PENDING = "pending"             # 待处理
    MATCHING = "matching"           # 匹配中
    ACCEPTED = "accepted"           # 已接收
    REJECTED = "rejected"           # 已拒绝
    IN_PROGRESS = "in_progress"     # 进行中
    COMPLETED = "completed"         # 已完成
    CANCELLED = "cancelled"         # 已取消


# ============================================================
# 专科类型
# ============================================================

class SpecialtyType(str, Enum):
    """专科类型"""
    GENERAL = "general"                 # 一般心理咨询
    DEPRESSION = "depression"           # 抑郁专科
    ANXIETY = "anxiety"                 # 焦虑专科
    TRAUMA = "trauma"                   # 创伤专科
    EATING_DISORDER = "eating_disorder" # 进食障碍
    SELF_HARM = "self_harm"            # 自伤干预
    FAMILY = "family"                   # 家庭治疗
    PSYCHIATRY = "psychiatry"           # 精神科


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ReceiverCandidate:
    """接收方候选

    Attributes:
        receiver_id: 接收方 ID
        name: 名称
        specialty: 专科类型
        availability: 可用度 [0, 1]
        current_load: 当前负荷（个案数）
        match_score: 匹配分数
    """
    receiver_id: str
    name: str
    specialty: str
    availability: float
    current_load: int
    match_score: float


@dataclass
class ReferralResult:
    """转介结果

    Attributes:
        referral_id: 转介 ID
        user_id: 用户 ID
        referral_type: 转介类型
        reason: 转介原因
        risk_level: 风险等级
        status: 转介状态
        matched_receivers: 匹配的接收方列表
        selected_receiver: 选定的接收方
        summary: 转介摘要（脱敏后的上下文）
        created_at: 创建时间
        updated_at: 更新时间
        notes: 备注
    """
    referral_id: str
    user_id: str
    referral_type: ReferralType
    reason: str
    risk_level: RiskLevel
    status: ReferralStatus = ReferralStatus.PENDING
    matched_receivers: list[ReceiverCandidate] = field(default_factory=list)
    selected_receiver: Optional[ReceiverCandidate] = None
    summary: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    notes: list[str] = field(default_factory=list)


# ============================================================
# 模拟接收方数据库
# ============================================================

_MOCK_RECEIVERS = [
    {"id": "CONSULTANT-001", "name": "李咨询师", "specialty": SpecialtyType.DEPRESSION, "load": 3, "avail": 0.8},
    {"id": "CONSULTANT-002", "name": "王咨询师", "specialty": SpecialtyType.ANXIETY, "load": 5, "avail": 0.5},
    {"id": "CONSULTANT-003", "name": "张咨询师", "specialty": SpecialtyType.TRAUMA, "load": 2, "avail": 0.9},
    {"id": "CONSULTANT-004", "name": "刘咨询师", "specialty": SpecialtyType.SELF_HARM, "load": 1, "avail": 0.95},
    {"id": "CONSULTANT-005", "name": "陈咨询师", "specialty": SpecialtyType.GENERAL, "load": 4, "avail": 0.6},
    {"id": "HOSPITAL-001", "name": "市心理卫生中心", "specialty": SpecialtyType.PSYCHIATRY, "load": 0, "avail": 1.0},
    {"id": "HOTLINE-001", "name": "24h心理援助热线", "specialty": SpecialtyType.GENERAL, "load": 0, "avail": 1.0},
]


# ============================================================
# 风险等级 → 专科匹配规则
# ============================================================

_RISK_SPECIALTY_MAP: dict[RiskLevel, list[SpecialtyType]] = {
    RiskLevel.LOW: [SpecialtyType.GENERAL],
    RiskLevel.MEDIUM: [SpecialtyType.GENERAL, SpecialtyType.DEPRESSION, SpecialtyType.ANXIETY],
    RiskLevel.HIGH: [SpecialtyType.DEPRESSION, SpecialtyType.ANXIETY, SpecialtyType.SELF_HARM],
    RiskLevel.CRISIS: [SpecialtyType.SELF_HARM, SpecialtyType.PSYCHIATRY],
}


# ============================================================
# 核心接口
# ============================================================

def create_referral(
    user_id: str,
    reason: str,
    risk_level: RiskLevel,
    referral_type: Optional[ReferralType] = None,
    crisis_alert: Optional[CrisisAlert] = None,
) -> ReferralResult:
    """创建转介请求

    Args:
        user_id: 用户 ID
        reason: 转介原因描述
        risk_level: 风险等级
        referral_type: 转介类型（自动推断）
        crisis_alert: 危机警报（如有）

    Returns:
        ReferralResult: 转介结果（含匹配接收方）
    """
    # 自动推断转介类型
    if referral_type is None:
        if risk_level == RiskLevel.CRISIS:
            referral_type = ReferralType.CRISIS_TO_EMERGENCY
        elif risk_level == RiskLevel.HIGH:
            referral_type = ReferralType.AI_TO_CONSULTANT
        else:
            referral_type = ReferralType.AI_TO_CONSULTANT

    referral_id = f"REF-{uuid.uuid4().hex[:8].upper()}"

    # 匹配接收方
    candidates = match_receiver(risk_level, referral_type)

    # 生成转介摘要
    summary = generate_referral_summary(user_id, reason, risk_level, crisis_alert)

    # 自动选择最佳匹配
    selected = candidates[0] if candidates else None

    return ReferralResult(
        referral_id=referral_id,
        user_id=user_id,
        referral_type=referral_type,
        reason=reason,
        risk_level=risk_level,
        status=ReferralStatus.MATCHING if candidates else ReferralStatus.PENDING,
        matched_receivers=candidates,
        selected_receiver=selected,
        summary=summary,
        created_at=time.time(),
        updated_at=time.time(),
        notes=[f"系统自动匹配 {len(candidates)} 个接收方"],
    )


def match_receiver(
    risk_level: RiskLevel,
    referral_type: ReferralType,
) -> list[ReceiverCandidate]:
    """匹配可用接收方

    根据风险等级匹配对应专科，按可用度和负荷排序。

    Args:
        risk_level: 风险等级
        referral_type: 转介类型

    Returns:
        按匹配分数排序的接收方列表
    """
    target_specialties = _RISK_SPECIALTY_MAP.get(risk_level, [SpecialtyType.GENERAL])

    # 危机情况优先匹配热线和精神科
    if referral_type == ReferralType.CRISIS_TO_EMERGENCY:
        target_specialties = [SpecialtyType.SELF_HARM, SpecialtyType.PSYCHIATRY]

    candidates = []
    for r in _MOCK_RECEIVERS:
        specialty = r["specialty"]
        if specialty not in target_specialties:
            continue

        # 匹配分数 = 可用度 * (1 / (1 + 负荷))
        match_score = r["avail"] * (1.0 / (1.0 + r["load"]))

        candidates.append(ReceiverCandidate(
            receiver_id=r["id"],
            name=r["name"],
            specialty=specialty.value if isinstance(specialty, SpecialtyType) else specialty,
            availability=r["avail"],
            current_load=r["load"],
            match_score=round(match_score, 3),
        ))

    # 按匹配分数降序
    candidates.sort(key=lambda c: c.match_score, reverse=True)
    return candidates


def generate_referral_summary(
    user_id: str,
    reason: str,
    risk_level: RiskLevel,
    crisis_alert: Optional[CrisisAlert] = None,
) -> str:
    """生成转介摘要（脱敏）

    仅包含必要的临床信息，不包含原始对话内容。

    Args:
        user_id: 用户 ID
        reason: 转介原因
        risk_level: 风险等级
        crisis_alert: 危机警报

    Returns:
        脱敏的转介摘要文本
    """
    level_cn = {
        RiskLevel.LOW: "低风险",
        RiskLevel.MEDIUM: "中风险",
        RiskLevel.HIGH: "高风险",
        RiskLevel.CRISIS: "危机",
    }

    lines = [
        f"转介摘要",
        f"用户编号: {user_id}",
        f"风险等级: {level_cn.get(risk_level, risk_level.value)}",
        f"转介原因: {reason}",
        f"转介时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    ]

    if crisis_alert:
        lines.extend([
            f"",
            f"危机证据:",
        ])
        for ev in crisis_alert.trigger_evidence[:5]:
            lines.append(f"  - {ev}")
        lines.append(f"建议动作: {crisis_alert.recommended_action}")

    lines.append("")
    lines.append("⚠️ 本摘要仅包含必要临床信息，原始对话内容请通过安全渠道获取。")

    return "\n".join(lines)
