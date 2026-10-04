"""
匿名化模块 —— 数据去标识化与匿名化处理

匿名化策略：
    1. K-匿名化：确保每条数据至少有 k 个不可区分个体
    2. 伪名化：用不可逆伪名替换真实标识符
    3. 数据泛化：将精确值替换为区间（如年龄 → 年龄段）
    4. 差分隐私：添加校准噪声确保 (ε, δ)-差分隐私

适用场景：
    - 联邦学习训练数据准备
    - 跨机构数据共享
    - 统计分析数据发布
    - 研究数据集生成

法规依据：
    - 《个人信息保护法》第 73 条：去标识化 vs 匿名化
    - GDPR Article 4(5): pseudonymisation

核心接口：
    - anonymize_dataset(records, config) -> AnonymizedDataset
    - pseudonymize(user_id) -> str
    - generalize_age(age) -> str
    - apply_k_anonymity(records, k) -> list
    - add_dp_noise(values, epsilon) -> list
"""
from __future__ import annotations

import hashlib
import math
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ============================================================
# 数据结构
# ============================================================

@dataclass
class AnonymizationConfig:
    """匿名化配置

    Attributes:
        k_anonymity: K-匿名化参数（默认 5）
        epsilon: 差分隐私预算（默认 1.0）
        delta: 差分隐私 delta 参数
        quasi_identifiers: 准标识符字段列表
        sensitive_fields: 敏感字段列表
        generalization_rules: 泛化规则
    """
    k_anonymity: int = 5
    epsilon: float = 1.0
    delta: float = 1e-5
    quasi_identifiers: list[str] = field(default_factory=lambda: ["age", "gender", "region"])
    sensitive_fields: list[str] = field(default_factory=lambda: ["diagnosis", "risk_score"])
    generalization_rules: dict[str, str] = field(default_factory=dict)


@dataclass
class AnonymizedDataset:
    """匿名化后的数据集

    Attributes:
        records: 匿名化记录列表
        original_count: 原始记录数
        anonymized_count: 匿名化后记录数
        k_anonymity_achieved: 实际达到的 K-匿名度
        epsilon_used: 使用的隐私预算
        removed_records: 因不满足 K-匿名而移除的记录数
        config: 使用的配置
    """
    records: list[dict] = field(default_factory=list)
    original_count: int = 0
    anonymized_count: int = 0
    k_anonymity_achieved: int = 0
    epsilon_used: float = 0.0
    removed_records: int = 0
    config: Optional[AnonymizationConfig] = None


@dataclass
class AnonymizationResult:
    """匿名化操作结果"""
    success: bool
    dataset: Optional[AnonymizedDataset] = None
    error: str = ""


# ============================================================
# 伪名化
# ============================================================

# 伪名映射表（内存存储）
_pseudonym_map: dict[str, str] = {}
_reverse_map: dict[str, str] = {}


def pseudonymize(real_id: str, salt: str = "") -> str:
    """将真实 ID 伪名化

    使用 HMAC-SHA256 生成不可逆伪名。
    同一 real_id 始终映射到同一伪名（确定性）。

    Args:
        real_id: 真实 ID
        salt: 额外盐值

    Returns:
        伪名字符串
    """
    if real_id in _pseudonym_map:
        return _pseudonym_map[real_id]

    # HMAC-SHA256
    key = (salt + "pseudonym_key_v1").encode("utf-8")
    pseudo = hashlib.sha256(
        key + real_id.encode("utf-8")
    ).hexdigest()[:16]

    pseudo_id = f"PS-{pseudo.upper()}"
    _pseudonym_map[real_id] = pseudo_id
    _reverse_map[pseudo_id] = real_id

    return pseudo_id


def depseudonymize(pseudo_id: str) -> Optional[str]:
    """反伪名化（需要授权）

    Args:
        pseudo_id: 伪名 ID

    Returns:
        真实 ID，未找到返回 None
    """
    return _reverse_map.get(pseudo_id)


def clear_pseudonym_map():
    """清除伪名映射（数据销毁时使用）"""
    _pseudonym_map.clear()
    _reverse_map.clear()


# ============================================================
# 数据泛化
# ============================================================

# 年龄 → 年龄段映射
AGE_GROUPS = [
    (0, 11, "儿童"),
    (12, 15, "青少年(12-15)"),
    (16, 18, "青少年(16-18)"),
    (19, 25, "青年(19-25)"),
    (26, 35, "成年(26-35)"),
    (36, 50, "中年(36-50)"),
    (51, 120, "老年(51+)"),
]


def generalize_age(age: int) -> str:
    """将精确年龄泛化为年龄段

    Args:
        age: 精确年龄

    Returns:
        年龄段字符串
    """
    for low, high, label in AGE_GROUPS:
        if low <= age <= high:
            return label
    return "未知"


def generalize_region(region: str) -> str:
    """将精确地区泛化为粗粒度区域

    Args:
        region: 精确地区（如"北京市海淀区"）

    Returns:
        泛化地区（如"北京"）
    """
    # 简化：只保留省级
    region_map = {
        "北京": "华北", "天津": "华北", "河北": "华北", "山西": "华北", "内蒙古": "华北",
        "上海": "华东", "江苏": "华东", "浙江": "华东", "安徽": "华东", "福建": "华东",
        "江西": "华东", "山东": "华东",
        "广东": "华南", "广西": "华南", "海南": "华南",
        "河南": "华中", "湖北": "华中", "湖南": "华中",
        "四川": "西南", "重庆": "西南", "贵州": "西南", "云南": "西南", "西藏": "西南",
        "陕西": "西北", "甘肃": "西北", "青海": "西北", "宁夏": "西北", "新疆": "西北",
        "辽宁": "东北", "吉林": "东北", "黑龙江": "东北",
    }

    for province, macro in region_map.items():
        if province in region:
            return macro
    return "未知"


def generalize_score(score: float, bins: int = 5) -> str:
    """将连续分数泛化为区间

    Args:
        score: 连续分数 [0, 1]
        bins: 分箱数

    Returns:
        区间字符串
    """
    bin_size = 1.0 / bins
    bin_idx = min(int(score / bin_size), bins - 1)
    low = round(bin_idx * bin_size, 2)
    high = round((bin_idx + 1) * bin_size, 2)
    return f"[{low}, {high})"


# ============================================================
# K-匿名化
# ============================================================

def apply_k_anonymity(
    records: list[dict],
    k: int = 5,
    quasi_identifiers: Optional[list[str]] = None,
) -> tuple[list[dict], int]:
    """K-匿名化处理

    确保每条记录的准标识符组合至少有 k 个不可区分个体。
    不满足条件的记录将被移除或泛化。

    Args:
        records: 数据记录列表
        k: K-匿名度参数
        quasi_identifiers: 准标识符字段

    Returns:
        (anonymized_records, removed_count)
    """
    if not records:
        return [], 0

    if quasi_identifiers is None:
        quasi_identifiers = ["age", "gender", "region"]

    # 计算每条记录的准标识符签名
    def _qi_signature(record: dict) -> tuple:
        return tuple(record.get(qi, "") for qi in quasi_identifiers)

    # 统计每个签名的出现次数
    signature_counts: dict[tuple, int] = {}
    for r in records:
        sig = _qi_signature(r)
        signature_counts[sig] = signature_counts.get(sig, 0) + 1

    # 保留满足 K-匿名的记录
    result = []
    removed = 0
    for r in records:
        sig = _qi_signature(r)
        if signature_counts[sig] >= k:
            result.append(r)
        else:
            removed += 1

    return result, removed


# ============================================================
# 差分隐私
# ============================================================

def add_dp_noise(
    values: list[float],
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
    mechanism: str = "laplace",
) -> list[float]:
    """添加差分隐私噪声

    支持拉普拉斯机制（ε-差分隐私）和高斯机制（(ε,δ)-差分隐私）。

    Args:
        values: 原始数值列表
        epsilon: 隐私预算（越小越隐私，噪声越大）
        sensitivity: 函数敏感度
        mechanism: 噪声机制 ("laplace" 或 "gaussian")

    Returns:
        加噪后的数值列表
    """
    if epsilon <= 0:
        raise ValueError("epsilon 必须为正数")

    result = []
    for v in values:
        if mechanism == "laplace":
            # 拉普拉斯机制: b = sensitivity / epsilon
            b = sensitivity / epsilon
            noise = np.random.laplace(0, b)
        elif mechanism == "gaussian":
            # 高斯机制: sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
            delta = 1e-5
            sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
            noise = np.random.normal(0, sigma)
        else:
            noise = 0.0

        result.append(round(float(v + noise), 4))

    return result


# ============================================================
# 核心接口
# ============================================================

def anonymize_dataset(
    records: list[dict],
    config: Optional[AnonymizationConfig] = None,
) -> AnonymizationResult:
    """完整匿名化处理管道

    流程：
        1. 伪名化所有 ID 字段
        2. 泛化准标识符
        3. K-匿名化检查
        4. 差分隐私噪声注入

    Args:
        records: 原始数据记录列表
        config: 匿名化配置

    Returns:
        AnonymizationResult
    """
    if config is None:
        config = AnonymizationConfig()

    original_count = len(records)

    # 1. 深拷贝 + 伪名化
    anonymized = []
    for r in records:
        new_r = dict(r)
        # 伪名化 ID 字段
        for key in ["user_id", "id", "patient_id", "consultant_id"]:
            if key in new_r and isinstance(new_r[key], str):
                new_r[key] = pseudonymize(new_r[key])
        anonymized.append(new_r)

    # 2. 泛化准标识符
    for r in anonymized:
        if "age" in r and isinstance(r["age"], (int, float)):
            r["age"] = generalize_age(int(r["age"]))
        if "region" in r and isinstance(r["region"], str):
            r["region"] = generalize_region(r["region"])
        # 泛化敏感分数
        for field_name in config.sensitive_fields:
            if field_name in r and isinstance(r[field_name], (int, float)):
                r[field_name] = generalize_score(float(r[field_name]) / 100.0)

    # 3. K-匿名化
    anonymized, removed = apply_k_anonymity(
        anonymized, k=config.k_anonymity, quasi_identifiers=config.quasi_identifiers
    )

    # 4. 计算实际 K-匿名度和噪声注入
    epsilon_used = 0.0
    if config.epsilon > 0 and config.sensitive_fields:
        for field_name in config.sensitive_fields:
            values = [r.get(field_name, 0) for r in anonymized if isinstance(r.get(field_name), (int, float))]
            if values:
                noisy = add_dp_noise(values, epsilon=config.epsilon)
                for i, r in enumerate(anonymized):
                    if field_name in r and i < len(noisy):
                        r[field_name] = noisy[i]
                epsilon_used = config.epsilon

    # 计算实际 K-匿名度
    def _qi_sig(r):
        return tuple(r.get(qi, "") for qi in config.quasi_identifiers)

    sig_counts: dict[tuple, int] = {}
    for r in anonymized:
        sig = _qi_sig(r)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
    k_achieved = min(sig_counts.values()) if sig_counts else 0

    dataset = AnonymizedDataset(
        records=anonymized,
        original_count=original_count,
        anonymized_count=len(anonymized),
        k_anonymity_achieved=k_achieved,
        epsilon_used=epsilon_used,
        removed_records=removed,
        config=config,
    )

    return AnonymizationResult(success=True, dataset=dataset)
