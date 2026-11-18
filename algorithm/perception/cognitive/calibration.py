"""认知八分类的独立校准配置；权重不匹配时明确退回原始概率。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_temperature(path: Path, weights: Path, class_order: list[int]) -> dict[str, Any]:
    """验证适用任务、拟合集、类别顺序和权重摘要；禁止跨任务套用温度。"""
    fallback: dict[str, Any] = {"status": "uncalibrated", "temperature": 1.0}
    if not path.is_file():
        return {**fallback, "reason": "artifact_missing"}
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        temperature = config.get("temperature")
        if (config.get("schema_version") != "cognitive_temperature.v1"
                or config.get("task") != "c2d2_8class" or config.get("fit_split") != "valid"
                or config.get("class_order") != class_order or len(class_order) != 8
                or type(temperature) not in (float, int) or not math.isfinite(temperature)
                or temperature <= 0 or type(config.get("n_calibration")) is not int
                or config["n_calibration"] <= 0):
            return {**fallback, "reason": "invalid_contract"}
        digest = hashlib.sha256()
        with weights.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != config.get("checkpoint_sha256"):
            return {**fallback, "reason": "checkpoint_mismatch"}
        return {**config, "status": "validation_temperature"}
    except (OSError, ValueError, TypeError, AttributeError):
        return {**fallback, "reason": "invalid_artifact"}


def calibrated_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """数值稳定的八分类温度缩放；排序不变，不能解读为临床概率。"""
    values = np.asarray(logits, dtype=np.float64)
    if (values.shape != (8,) or not np.all(np.isfinite(values))
            or type(temperature) not in (int, float) or not math.isfinite(temperature)
            or temperature <= 0):
        raise ValueError("认知校准输入无效")
    values = (values - values.max()) / temperature
    exp = np.exp(values)
    return (exp / exp.sum()).astype(np.float32)
