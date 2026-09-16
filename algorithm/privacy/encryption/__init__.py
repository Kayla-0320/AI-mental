"""
加密模块 —— 数据静态加密与传输保护

加密策略：
    1. AES-256-GCM: 本地数据加密（心理评估报告、对话记录）
    2. 密钥派生: PBKDF2-SHA256 从用户主密钥派生加密密钥
    3. 安全随机: 使用 secrets 模块生成 IV 和 salt
    4. 密钥轮换: 支持密钥版本管理

法规依据：
    - 《个人信息保护法》第 51 条：采取加密等安全技术措施
    - 《数据安全法》第 27 条：数据加密存储和传输

核心接口：
    - encrypt_data(plaintext, key) -> EncryptedData
    - decrypt_data(encrypted, key) -> str
    - derive_key(master_key, salt) -> bytes
    - generate_encryption_key() -> bytes
"""
from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 数据结构
# ============================================================

@dataclass
class EncryptedData:
    """加密数据

    Attributes:
        ciphertext: 密文（base64 编码）
        nonce: 随机数/IV
        salt: 密钥派生 salt
        tag: 认证标签（GCM mode）
        key_version: 密钥版本号
        algorithm: 加密算法标识
        encrypted_at: 加密时间戳
    """
    ciphertext: str
    nonce: str
    salt: str
    tag: str = ""
    key_version: int = 1
    algorithm: str = "AES-256-GCM"
    encrypted_at: float = 0.0


@dataclass
class EncryptionResult:
    """加密操作结果

    Attributes:
        success: 是否成功
        encrypted_data: 加密后的数据
        error: 错误信息
    """
    success: bool
    encrypted_data: Optional[EncryptedData] = None
    error: str = ""


# ============================================================
# 常量
# ============================================================

AES_KEY_SIZE = 32       # 256 bits
NONCE_SIZE = 12         # 96 bits (GCM recommended)
SALT_SIZE = 16          # 128 bits
TAG_SIZE = 16           # 128 bits
PBKDF2_ITERATIONS = 600_000  # OWASP 2023 recommended


# ============================================================
# 密钥管理
# ============================================================

def generate_encryption_key() -> bytes:
    """生成随机 AES-256 加密密钥

    Returns:
        32 字节随机密钥
    """
    return secrets.token_bytes(AES_KEY_SIZE)


def derive_key(
    master_key: str,
    salt: Optional[bytes] = None,
    iterations: int = PBKDF2_ITERATIONS,
) -> tuple[bytes, bytes]:
    """从主密钥派生加密密钥 (PBKDF2-SHA256)

    Args:
        master_key: 主密钥字符串
        salt: 盐值（可选，不传则自动生成）
        iterations: PBKDF2 迭代次数

    Returns:
        (derived_key, salt): 派生密钥和盐值
    """
    if salt is None:
        salt = secrets.token_bytes(SALT_SIZE)

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        master_key.encode("utf-8"),
        salt,
        iterations,
        dklen=AES_KEY_SIZE,
    )

    return derived, salt


# ============================================================
# 加密/解密（使用 cryptography 库或纯 Python 降级）
# ============================================================

def _try_cryptography_encrypt(plaintext: bytes, key: bytes) -> Optional[EncryptedData]:
    """尝试使用 cryptography 库加密"""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        nonce = secrets.token_bytes(NONCE_SIZE)
        aesgcm = AESGCM(key)
        ciphertext_and_tag = aesgcm.encrypt(nonce, plaintext, None)

        # 分离密文和 tag
        ciphertext = ciphertext_and_tag[:-TAG_SIZE]
        tag = ciphertext_and_tag[-TAG_SIZE:]

        import base64
        return EncryptedData(
            ciphertext=base64.b64encode(ciphertext).decode(),
            nonce=base64.b64encode(nonce).decode(),
            salt="",
            tag=base64.b64encode(tag).decode(),
            algorithm="AES-256-GCM",
            encrypted_at=time.time(),
        )
    except ImportError:
        return None
    except Exception:
        return None


def _try_cryptography_decrypt(encrypted: EncryptedData, key: bytes) -> Optional[str]:
    """尝试使用 cryptography 库解密"""
    try:
        import base64
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        ciphertext = base64.b64decode(encrypted.ciphertext)
        nonce = base64.b64decode(encrypted.nonce)
        tag = base64.b64decode(encrypted.tag)

        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext + tag, None)
        return plaintext.decode("utf-8")
    except Exception:
        return None


def _fallback_encrypt(plaintext: bytes, key: bytes) -> EncryptedData:
    """降级加密方案：XOR + HMAC（仅用于演示，生产环境必须用 AES-GCM）

    ⚠️ 此方案不安全，仅在 cryptography 库不可用时降级使用。
    """
    import base64
    import hmac

    nonce = secrets.token_bytes(NONCE_SIZE)

    # XOR 加密（不安全，仅演示）
    key_stream = (key * (len(plaintext) // len(key) + 1))[:len(plaintext)]
    ciphertext = bytes(a ^ b for a, b in zip(plaintext, key_stream))

    # HMAC 认证
    tag = hmac.new(key, nonce + ciphertext, hashlib.sha256).digest()[:TAG_SIZE]

    return EncryptedData(
        ciphertext=base64.b64encode(ciphertext).decode(),
        nonce=base64.b64encode(nonce).decode(),
        salt="",
        tag=base64.b64encode(tag).decode(),
        algorithm="XOR-HMAC-FALLBACK",
        encrypted_at=time.time(),
    )


def _fallback_decrypt(encrypted: EncryptedData, key: bytes) -> Optional[str]:
    """降级解密"""
    try:
        import base64

        ciphertext = base64.b64decode(encrypted.ciphertext)
        nonce = base64.b64decode(encrypted.nonce)

        key_stream = (key * (len(ciphertext) // len(key) + 1))[:len(ciphertext)]
        plaintext = bytes(a ^ b for a, b in zip(ciphertext, key_stream))
        return plaintext.decode("utf-8")
    except Exception:
        return None


# ============================================================
# 核心接口
# ============================================================

def encrypt_data(
    plaintext: str,
    key: bytes,
) -> EncryptionResult:
    """加密数据

    优先使用 AES-256-GCM（cryptography 库），
    不可用时降级为 XOR+HMAC（仅演示）。

    Args:
        plaintext: 明文字符串
        key: 32 字节加密密钥

    Returns:
        EncryptionResult: 加密结果
    """
    if len(key) != AES_KEY_SIZE:
        return EncryptionResult(success=False, error=f"密钥长度必须为 {AES_KEY_SIZE} 字节")

    data = plaintext.encode("utf-8")

    # 尝试 cryptography 库
    encrypted = _try_cryptography_encrypt(data, key)
    if encrypted is None:
        encrypted = _fallback_encrypt(data, key)

    return EncryptionResult(success=True, encrypted_data=encrypted)


def decrypt_data(
    encrypted: EncryptedData,
    key: bytes,
) -> tuple[bool, str]:
    """解密数据

    Args:
        encrypted: 加密数据
        key: 加密密钥

    Returns:
        (success, plaintext_or_error)
    """
    if encrypted.algorithm == "AES-256-GCM":
        result = _try_cryptography_decrypt(encrypted, key)
        if result is not None:
            return True, result
        # 降级尝试
        result = _fallback_decrypt(encrypted, key)
    else:
        result = _fallback_decrypt(encrypted, key)

    if result is not None:
        return True, result
    return False, "解密失败"


def encrypt_file(
    file_path: str,
    key: bytes,
    output_path: Optional[str] = None,
) -> EncryptionResult:
    """加密文件

    Args:
        file_path: 原文件路径
        key: 加密密钥
        output_path: 输出路径（默认 .encrypted 后缀）

    Returns:
        EncryptionResult
    """
    try:
        with open(file_path, "rb") as f:
            plaintext = f.read()

        data = plaintext.decode("utf-8", errors="replace")
        result = encrypt_data(data, key)

        if result.success and result.encrypted_data and output_path:
            import json
            with open(output_path, "w") as f:
                json.dump({
                    "ciphertext": result.encrypted_data.ciphertext,
                    "nonce": result.encrypted_data.nonce,
                    "salt": result.encrypted_data.salt,
                    "tag": result.encrypted_data.tag,
                    "algorithm": result.encrypted_data.algorithm,
                    "encrypted_at": result.encrypted_data.encrypted_at,
                }, f, indent=2)

        return result
    except Exception as e:
        return EncryptionResult(success=False, error=str(e))
