"""
知情同意管理模块 —— 用户数据使用授权的全生命周期管理

管理内容：
    1. 数据采集同意：多模态数据（文本/语音/面部/行为）的采集授权
    2. 数据分析同意：情绪分析、风险评估的授权
    3. 数据共享同意：向咨询师/第三方共享数据的授权
    4. 危机例外：危机情况下无需同意即可共享必要信息

法规依据：
    - 《个人信息保护法》第 13-16 条：知情同意原则
    - 《未成年人保护法》第 72 条：未成年人个人信息保护
    - 监护人同意：14 岁以下需监护人授权

核心接口：
    - create_consent(user_id, consents) -> ConsentRecord
    - check_consent(user_id, data_type, purpose) -> bool
    - revoke_consent(user_id, consent_id) -> bool
    - crisis_exception_check(user_id) -> CrisisConsentResult
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# 数据类型枚举
# ============================================================

class DataType(str, Enum):
    """数据类型"""
    TEXT = "text"                       # 文本对话
    VOICE = "voice"                     # 语音数据
    FACIAL = "facial"                   # 面部数据
    BEHAVIOR = "behavior"               # 行为数据（击键/活跃模式）
    SLEEP = "sleep"                     # 睡眠数据
    ASSESSMENT = "assessment"           # 测评数据
    PHYSIOLOGICAL = "physiological"     # 生理数据（心率/步数）
    EMOTION_RESULT = "emotion_result"   # 情绪分析结果
    RISK_ASSESSMENT = "risk_assessment" # 风险评估结果
    PHENOTYPE = "phenotype"             # 数字表型


class DataPurpose(str, Enum):
    """数据使用目的"""
    EMOTION_ANALYSIS = "emotion_analysis"           # 情绪分析
    RISK_ASSESSMENT = "risk_assessment"             # 风险评估
    CONSULTANT_VIEW = "consultant_view"             # 咨询师查看
    RESEARCH = "research"                           # 研究用途
    FEDERATED_LEARNING = "federated_learning"       # 联邦学习
    CRISIS_INTERVENTION = "crisis_intervention"     # 危机干预


class ConsentStatus(str, Enum):
    """同意状态"""
    ACTIVE = "active"                   # 有效
    REVOKED = "revoked"                 # 已撤回
    EXPIRED = "expired"                 # 已过期
    PENDING_GUARDIAN = "pending_guardian"  # 待监护人确认


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ConsentItem:
    """单项同意

    Attributes:
        data_type: 数据类型
        purpose: 使用目的
        granted: 是否同意
        description: 同意说明
    """
    data_type: DataType
    purpose: DataPurpose
    granted: bool
    description: str = ""


@dataclass
class ConsentRecord:
    """同意记录

    Attributes:
        consent_id: 同意记录 ID
        user_id: 用户 ID
        items: 同意项列表
        status: 整体状态
        guardian_consent: 监护人同意（未成年人）
        guardian_id: 监护人 ID
        created_at: 创建时间
        expires_at: 过期时间
        version: 同意书版本号
    """
    consent_id: str
    user_id: str
    items: list[ConsentItem] = field(default_factory=list)
    status: ConsentStatus = ConsentStatus.ACTIVE
    guardian_consent: bool = False
    guardian_id: str = ""
    created_at: float = 0.0
    expires_at: float = 0.0
    version: str = "1.0"


@dataclass
class CrisisConsentResult:
    """危机例外同意检查结果

    Attributes:
        can_share: 是否可以在危机时共享数据
        shareable_types: 可共享的数据类型
        reason: 原因说明
        legal_basis: 法律依据
    """
    can_share: bool
    shareable_types: list[DataType] = field(default_factory=list)
    reason: str = ""
    legal_basis: str = ""


# ============================================================
# 同意存储（内存模拟）
# ============================================================

_consent_store: dict[str, list[ConsentRecord]] = {}


def _get_user_consents(user_id: str) -> list[ConsentRecord]:
    """获取用户的所有同意记录"""
    return _consent_store.get(user_id, [])


# ============================================================
# 核心接口
# ============================================================

def create_consent(
    user_id: str,
    items: list[ConsentItem],
    guardian_consent: bool = False,
    guardian_id: str = "",
    expires_days: int = 365,
) -> ConsentRecord:
    """创建同意记录

    Args:
        user_id: 用户 ID
        items: 同意项列表
        guardian_consent: 是否获得监护人同意
        guardian_id: 监护人 ID
        expires_days: 有效天数

    Returns:
        ConsentRecord: 同意记录
    """
    consent_id = f"CON-{uuid.uuid4().hex[:8].upper()}"
    now = time.time()

    record = ConsentRecord(
        consent_id=consent_id,
        user_id=user_id,
        items=items,
        status=ConsentStatus.ACTIVE,
        guardian_consent=guardian_consent,
        guardian_id=guardian_id,
        created_at=now,
        expires_at=now + expires_days * 86400,
        version="1.0",
    )

    if user_id not in _consent_store:
        _consent_store[user_id] = []
    _consent_store[user_id].append(record)

    return record


def check_consent(
    user_id: str,
    data_type: DataType,
    purpose: DataPurpose,
) -> bool:
    """检查用户是否同意特定数据使用

    Args:
        user_id: 用户 ID
        data_type: 数据类型
        purpose: 使用目的

    Returns:
        是否已授权
    """
    records = _get_user_consents(user_id)
    now = time.time()

    for record in records:
        # 检查记录是否有效
        if record.status != ConsentStatus.ACTIVE:
            continue
        if record.expires_at > 0 and now > record.expires_at:
            record.status = ConsentStatus.EXPIRED
            continue

        # 检查具体同意项
        for item in record.items:
            if item.data_type == data_type and item.purpose == purpose and item.granted:
                return True

    return False


def revoke_consent(
    user_id: str,
    consent_id: str,
) -> bool:
    """撤回同意

    Args:
        user_id: 用户 ID
        consent_id: 同意记录 ID

    Returns:
        是否成功撤回
    """
    records = _get_user_consents(user_id)
    for record in records:
        if record.consent_id == consent_id:
            record.status = ConsentStatus.REVOKED
            return True
    return False


def revoke_all_consents(
    user_id: str,
    data_type: Optional[DataType] = None,
) -> int:
    """撤回用户的所有（或指定类型的）同意

    Args:
        user_id: 用户 ID
        data_type: 指定数据类型（None 表示全部）

    Returns:
        撤回的记录数
    """
    records = _get_user_consents(user_id)
    count = 0
    for record in records:
        if record.status != ConsentStatus.ACTIVE:
            continue
        if data_type is not None:
            # 只撤回包含指定数据类型的记录
            has_type = any(item.data_type == data_type for item in record.items)
            if has_type:
                record.status = ConsentStatus.REVOKED
                count += 1
        else:
            record.status = ConsentStatus.REVOKED
            count += 1
    return count


def crisis_exception_check(
    user_id: str,
    risk_level: str = "crisis",
) -> CrisisConsentResult:
    """危机例外同意检查

    在危机情况下，即使未获得用户同意，也可以共享必要数据。
    法律依据：《个人信息保护法》第 13 条第 4 款
    "为应对突发公共卫生事件，或者紧急情况下为保护自然人的生命健康和财产安全所必需"

    Args:
        user_id: 用户 ID
        risk_level: 风险等级

    Returns:
        CrisisConsentResult: 危机例外结果
    """
    if risk_level not in ("high", "crisis"):
        return CrisisConsentResult(
            can_share=False,
            reason="非危机情况，需遵循正常同意流程",
        )

    # 危机情况下可共享的必要数据
    shareable = [
        DataType.EMOTION_RESULT,
        DataType.RISK_ASSESSMENT,
        DataType.TEXT,
        DataType.BEHAVIOR,
    ]

    return CrisisConsentResult(
        can_share=True,
        shareable_types=shareable,
        reason=f"危机情况（{risk_level}），依据生命健康保护例外原则，"
               f"可共享必要数据给咨询师和紧急联系人",
        legal_basis="《个人信息保护法》第13条第4款："
                    "紧急情况下为保护自然人的生命健康和财产安全所必需",
    )


def get_consent_summary(user_id: str) -> dict:
    """获取用户同意状态摘要

    Args:
        user_id: 用户 ID

    Returns:
        同意状态摘要字典
    """
    records = _get_user_consents(user_id)
    active_records = [r for r in records if r.status == ConsentStatus.ACTIVE]

    granted_items = set()
    for r in active_records:
        for item in r.items:
            if item.granted:
                granted_items.add((item.data_type.value, item.purpose.value))

    return {
        "user_id": user_id,
        "total_records": len(records),
        "active_records": len(active_records),
        "granted_permissions": [
            {"data_type": dt, "purpose": p} for dt, p in granted_items
        ],
        "has_guardian_consent": any(r.guardian_consent for r in active_records),
    }


def create_default_consent(user_id: str, age: int = 18) -> ConsentRecord:
    """创建默认同意记录（首次注册时）

    默认授权：
    - 文本/行为数据的情绪分析和风险评估
    - 咨询师查看评估结果

    默认不授权：
    - 研究用途
    - 联邦学习

    Args:
        user_id: 用户 ID
        age: 用户年龄

    Returns:
        ConsentRecord
    """
    items = [
        # 默认授权
        ConsentItem(DataType.TEXT, DataPurpose.EMOTION_ANALYSIS, True, "文本情感分析"),
        ConsentItem(DataType.BEHAVIOR, DataPurpose.EMOTION_ANALYSIS, True, "行为数据分析"),
        ConsentItem(DataType.EMOTION_RESULT, DataPurpose.CONSULTANT_VIEW, True, "咨询师查看情绪分析"),
        ConsentItem(DataType.RISK_ASSESSMENT, DataPurpose.CONSULTANT_VIEW, True, "咨询师查看风险评估"),
        ConsentItem(DataType.ASSESSMENT, DataPurpose.RISK_ASSESSMENT, True, "测评数据用于风险评估"),
        # 默认不授权
        ConsentItem(DataType.TEXT, DataPurpose.RESEARCH, False, "文本数据用于研究（需单独授权）"),
        ConsentItem(DataType.EMOTION_RESULT, DataPurpose.FEDERATED_LEARNING, False, "参与联邦学习（需单独授权）"),
    ]

    need_guardian = age < 14
    guardian_consent = need_guardian  # 实际需要监护人确认

    return create_consent(
        user_id=user_id,
        items=items,
        guardian_consent=guardian_consent,
        expires_days=365,
    )
