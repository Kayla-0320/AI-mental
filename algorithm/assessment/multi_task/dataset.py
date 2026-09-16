"""
数据加载器 —— 支持 CMDC/EATD 格式，当前使用假数据

数据格式（CMDC/EATD 兼容）：
{
    "samples": [
        {
            "features": [0.1, 0.2, ...],   # 感知层输出特征向量
            "depression": 0,                 # 抑郁标签 (0/1)
            "anxiety": 1,                    # 焦虑标签 (0/1)
            "sleep": 0                       # 睡眠标签 (0/1)
        },
        ...
    ]
}

切换真实数据只需替换 JSON 文件路径。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np

from assessment.multi_task.config import (
    DATA_DIR,
    FEATURE_DIM,
    N_SAMPLES,
    SEED,
    TASK_NAMES,
)


# ============================================================
# 假数据生成
# ============================================================

def generate_fake_data(
    n_samples: int = N_SAMPLES,
    feature_dim: int = FEATURE_DIM,
    output_path: Optional[str] = None,
) -> str:
    """生成模拟数据用于训练流程验证

    模拟感知层输出特征 + 三任务标签，
    标签之间存在共病相关性（模拟真实临床场景）。
    特征与标签之间建立更强的非线性关系，
    包含噪声特征以模拟真实感知层输出。

    Args:
        n_samples: 样本数量
        feature_dim: 特征维度
        output_path: 输出路径

    Returns:
        生成的文件路径
    """
    rng = np.random.RandomState(SEED)

    samples = []
    for _ in range(n_samples):
        # 生成潜在风险因子（偏向低风险但有一定比例高风险）
        latent_risk = rng.beta(2, 5)

        # 共病因子：影响多个任务同时为阳性的概率
        comorbidity = rng.beta(1.5, 4)

        # 各任务标签与潜在因子 + 共病因子相关
        dep_prob = 0.08 + 0.65 * latent_risk + 0.15 * comorbidity + rng.normal(0, 0.03)
        anx_prob = 0.12 + 0.60 * latent_risk + 0.18 * comorbidity + rng.normal(0, 0.03)
        slp_prob = 0.10 + 0.55 * latent_risk + 0.12 * comorbidity + rng.normal(0, 0.03)

        depression = int(rng.random() < np.clip(dep_prob, 0, 1))
        anxiety = int(rng.random() < np.clip(anx_prob, 0, 1))
        sleep = int(rng.random() < np.clip(slp_prob, 0, 1))

        # 特征与标签之间的非线性关系（模拟感知层输出）
        risk_signal = 0.40 * depression + 0.35 * anxiety + 0.25 * sleep

        # 信号特征（与标签强相关）
        n_signal = feature_dim // 2
        signal_features = rng.randn(n_signal) * 0.3 + risk_signal * rng.randn(n_signal) * 1.5

        # 噪声特征（与标签弱相关，模拟感知层噪声）
        n_noise = feature_dim - n_signal
        noise_features = rng.randn(n_noise) * 1.0 + risk_signal * rng.randn(n_noise) * 0.3

        features = np.concatenate([signal_features, noise_features])

        samples.append({
            "features": features.tolist(),
            "depression": depression,
            "anxiety": anxiety,
            "sleep": sleep,
        })

    if output_path is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(DATA_DIR / "fake_train.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)

    return output_path


# ============================================================
# 数据加载
# ============================================================

def load_dataset(data_path: str) -> dict:
    """加载数据集

    Args:
        data_path: JSON 数据文件路径

    Returns:
        {
            "X": np.ndarray (n_samples, feature_dim),
            "y_depression": np.ndarray (n_samples,),
            "y_anxiety": np.ndarray (n_samples,),
            "y_sleep": np.ndarray (n_samples,),
        }
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = raw["samples"]
    X = np.array([s["features"] for s in samples], dtype=np.float32)
    y_dict = {
        f"y_{task}": np.array([s[task] for s in samples], dtype=np.int64)
        for task in TASK_NAMES
    }

    return {"X": X, **y_dict}


def split_train_val(
    data: dict,
    val_ratio: float = 0.2,
    seed: int = SEED,
) -> tuple[dict, dict]:
    """划分训练集和验证集

    Args:
        data: load_dataset 的输出
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        (train_data, val_data)
    """
    rng = np.random.RandomState(seed)
    n = len(data["X"])
    indices = rng.permutation(n)
    val_size = int(n * val_ratio)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}

    return train_data, val_data
