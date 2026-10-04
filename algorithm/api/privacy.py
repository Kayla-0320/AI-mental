"""
隐私架构路由 —— 数据隐私保护 API 端点

暴露底层 privacy/ 子模块的功能：
- POST /privacy/anonymize → privacy/anonymization/ 的 K-匿名处理
- POST /privacy/consent   → privacy/consent/ 的知情同意管理
- POST /privacy/encrypt   → privacy/encryption/ 的 AES 加密
"""
from __future__ import annotations

import base64
import uuid
from dataclasses import asdict, field
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from privacy.anonymization import (
    anonymize_dataset,
    AnonymizationConfig,
    AnonymizationResult,
)
from privacy.consent import (
    create_consent,
    check_consent,
    revoke_consent,
    get_consent_summary,
    crisis_exception_check,
    ConsentItem,
    ConsentRecord,
    DataType,
    DataPurpose,
    ConsentStatus,
)
from privacy.encryption import (
    encrypt_data,
    decrypt_data,
    generate_encryption_key,
    EncryptionResult,
    EncryptedData,
)

router = APIRouter(prefix="/privacy", tags=["隐私架构"])


# ============================================================
# 匿名化 请求/响应模型
# ============================================================

class AnonymizeRequest(BaseModel):
    records: list[dict] = Field(..., description="原始数据记录列表")
    k_anonymity: int = Field(5, description="K-匿名度参数")
    epsilon: float = Field(1.0, description="差分隐私预算")
    quasi_identifiers: Optional[list[str]] = Field(None, description="准标识符字段")
    sensitive_fields: Optional[list[str]] = Field(None, description="敏感字段")


class AnonymizeResponse(BaseModel):
    success: bool
    original_count: int
    anonymized_count: int
    k_anonymity_achieved: int
    epsilon_used: float
    removed_records: int
    records: list[dict]
    error: str = ""


# ============================================================
# 知情同意 请求/响应模型
# ============================================================

class ConsentItemRequest(BaseModel):
    data_type: str = Field(..., description="数据类型 (text/voice/facial/...)")
    purpose: str = Field(..., description="使用目的 (emotion_analysis/risk_assessment/...)")
    granted: bool = Field(..., description="是否同意")
    description: str = Field("", description="同意说明")


class ConsentCreateRequest(BaseModel):
    user_id: str = Field(..., description="用户 ID")
    items: list[ConsentItemRequest] = Field(..., description="同意项列表")
    guardian_consent: bool = Field(False, description="监护人同意")
    guardian_id: str = Field("", description="监护人 ID")
    expires_days: int = Field(365, description="有效天数")


class ConsentCreateResponse(BaseModel):
    consent_id: str
    user_id: str
    status: str
    items: list[dict]
    guardian_consent: bool
    created_at: float
    expires_at: float


class ConsentCheckRequest(BaseModel):
    user_id: str
    data_type: str
    purpose: str


class ConsentSummaryResponse(BaseModel):
    user_id: str
    total_records: int
    active_records: int
    granted_permissions: list[dict]
    has_guardian_consent: bool


# ============================================================
# 加密 请求/响应模型
# ============================================================

class EncryptRequest(BaseModel):
    plaintext: str = Field(..., description="待加密明文")
    key: Optional[str] = Field(None, description="Base64 编码密钥（不传则自动生成）")


class EncryptResponse(BaseModel):
    success: bool
    ciphertext: str = ""
    nonce: str = ""
    salt: str = ""
    tag: str = ""
    algorithm: str = ""
    key_version: int = 1
    generated_key: Optional[str] = None  # 自动生成时返回 base64 密钥
    error: str = ""


# ============================================================
# 端点
# ============================================================

@router.post("/anonymize", response_model=AnonymizeResponse)
async def privacy_anonymize(request: AnonymizeRequest):
    """K-匿名化处理

    调用 privacy/anonymization/ 的完整匿名化管道：
    伪名化 → 泛化 → K-匿名检查 → 差分隐私噪声注入。
    """
    if not request.records:
        raise HTTPException(status_code=400, detail="records 不能为空")

    config = AnonymizationConfig(
        k_anonymity=request.k_anonymity,
        epsilon=request.epsilon,
    )
    if request.quasi_identifiers:
        config.quasi_identifiers = request.quasi_identifiers
    if request.sensitive_fields:
        config.sensitive_fields = request.sensitive_fields

    result: AnonymizationResult = anonymize_dataset(request.records, config)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)

    ds = result.dataset
    return AnonymizeResponse(
        success=True,
        original_count=ds.original_count,
        anonymized_count=ds.anonymized_count,
        k_anonymity_achieved=ds.k_anonymity_achieved,
        epsilon_used=ds.epsilon_used,
        removed_records=ds.removed_records,
        records=ds.records,
    )


@router.post("/consent", response_model=ConsentCreateResponse)
async def privacy_consent(request: ConsentCreateRequest):
    """创建知情同意记录

    调用 privacy/consent/ 的 create_consent()，支持多数据类型、
    多使用目的的细粒度授权。
    """
    if not request.items:
        raise HTTPException(status_code=400, detail="items 不能为空")

    # 转换枚举
    consent_items = []
    for item in request.items:
        try:
            dt = DataType(item.data_type)
            dp = DataPurpose(item.purpose)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"枚举值无效: {e}")
        consent_items.append(
            ConsentItem(
                data_type=dt,
                purpose=dp,
                granted=item.granted,
                description=item.description,
            )
        )

    record: ConsentRecord = create_consent(
        user_id=request.user_id,
        items=consent_items,
        guardian_consent=request.guardian_consent,
        guardian_id=request.guardian_id,
        expires_days=request.expires_days,
    )

    return ConsentCreateResponse(
        consent_id=record.consent_id,
        user_id=record.user_id,
        status=record.status.value,
        items=[
            {
                "data_type": ci.data_type.value,
                "purpose": ci.purpose.value,
                "granted": ci.granted,
                "description": ci.description,
            }
            for ci in record.items
        ],
        guardian_consent=record.guardian_consent,
        created_at=record.created_at,
        expires_at=record.expires_at,
    )


@router.get("/consent/{user_id}", response_model=ConsentSummaryResponse)
async def privacy_consent_summary(user_id: str):
    """获取用户同意状态摘要"""
    summary = get_consent_summary(user_id)
    return ConsentSummaryResponse(**summary)


@router.post("/encrypt", response_model=EncryptResponse)
async def privacy_encrypt(request: EncryptRequest):
    """AES-256-GCM 加密

    调用 privacy/encryption/ 的 encrypt_data()。
    未提供密钥时自动生成 32 字节随机密钥。
    """
    if not request.plaintext:
        raise HTTPException(status_code=400, detail="plaintext 不能为空")

    # 密钥处理
    generated_key_b64 = None
    if request.key:
        try:
            key = base64.b64decode(request.key)
        except Exception:
            raise HTTPException(status_code=400, detail="密钥 base64 解码失败")
    else:
        key = generate_encryption_key()
        generated_key_b64 = base64.b64encode(key).decode()

    result: EncryptionResult = encrypt_data(request.plaintext, key)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)

    ed = result.encrypted_data
    return EncryptResponse(
        success=True,
        ciphertext=ed.ciphertext,
        nonce=ed.nonce,
        salt=ed.salt,
        tag=ed.tag,
        algorithm=ed.algorithm,
        key_version=ed.key_version,
        generated_key=generated_key_b64,
    )
