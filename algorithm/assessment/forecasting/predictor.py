"""无未来信息的短期预测和带版本验证的研究产物加载。"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from shared import EmotionForecast, EmotionObservation

TARGETS = ["positive_self_report", "negative_self_report"]
SCHEMA = "esm_forecast.v1"
CONTEXT = 8


def validate_history(history: list[EmotionObservation], context: int = CONTEXT) -> None:
    """拒绝重复、乱序、未知来源、非有限值和不足的有效历史。"""
    if len(history) < context or len(history) > 512:
        raise ValueError(f"history must contain {context} to 512 observations")
    previous = 0
    for point in history:
        if isinstance(point.step, bool) or not isinstance(point.step, int) or point.step <= previous:
            raise ValueError("steps must be strictly increasing positive integers")
        if point.source != "self_report":
            raise ValueError("only self_report observations match this model task")
        if not all(not isinstance(value, bool) and isinstance(value, Real)
                   and math.isfinite(value) and 0 <= value <= 100
                   for value in (point.positive, point.negative)):
            raise ValueError("self-report values must be finite and in [0,100]")
        previous = point.step
    if any(b.step != a.step + 1 for a, b in zip(history[-context:-1], history[-context + 1:])):
        raise ValueError("the most recent context must have consecutive sampling steps")


def history_features(sequence: np.ndarray) -> np.ndarray:
    """仅从给定过去窗口计算末值、均值、标准差与趋势，不接受全程统计。"""
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (CONTEXT, 2) or not np.isfinite(values).all():
        raise ValueError(f"expected finite [N,{CONTEXT},2] history array")
    centered_time = np.arange(CONTEXT) - (CONTEXT - 1) / 2
    slope = (values * centered_time[None, :, None]).sum(axis=1) / (centered_time ** 2).sum()
    return np.concatenate([values.reshape(len(values), -1), values.mean(axis=1),
                           values.std(axis=1), slope], axis=1)


def ema_prediction(sequence: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """固定系数的指数平滑基线，只消费过去观测。"""
    result = sequence[:, 0].copy()
    for index in range(1, sequence.shape[1]):
        result = alpha * sequence[:, index] + (1 - alpha) * result
    return result


class ForecastPredictor:
    """从 JSON/NPZ 加载已验证任务产物，不执行 pickle。"""

    def __init__(self, artifact_path: Path) -> None:
        metadata = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("forecast metadata must be an object")
        if metadata.get("schema") != SCHEMA or metadata.get("targets") != TARGETS:
            raise ValueError("forecast artifact task/schema mismatch")
        if metadata.get("context") != CONTEXT or metadata.get("scale") != [0, 100]:
            raise ValueError("forecast context/scale mismatch")
        if metadata.get("selection_split") != "validation" or metadata.get("fit_split") != "train":
            raise ValueError("invalid model selection provenance")
        if not isinstance(metadata.get("model_id"), str) or not 1 <= len(metadata["model_id"]) <= 256:
            raise ValueError("invalid model identifier")
        horizons = metadata.get("horizons")
        if not isinstance(horizons, dict) or not horizons or not set(horizons) <= {"1", "3"}:
            raise ValueError("invalid forecast horizons")
        for settings in horizons.values():
            if not isinstance(settings, dict) or settings.get("selected_method") not in {
                    "persistence", "ema", "past_mean", "ridge", "gru"}:
                raise ValueError("invalid selected forecast method")
            residual = settings.get("validation_absolute_error_quantile")
            if not isinstance(residual, list) or len(residual) != 2 or not all(
                    not isinstance(value, bool) and isinstance(value, Real)
                    and math.isfinite(value) and 0 <= value <= 1 for value in residual):
                raise ValueError("invalid validation uncertainty estimate")
        weight_file = str(metadata["weights_file"])
        if Path(weight_file).name != weight_file:
            raise ValueError("weights file must be a sibling filename")
        weights_path = artifact_path.with_name(weight_file)
        if hashlib.sha256(weights_path.read_bytes()).hexdigest() != metadata["weights_sha256"]:
            raise ValueError("forecast weight digest mismatch")
        with np.load(weights_path, allow_pickle=False) as archive:
            self.weights = {key: archive[key] for key in archive.files}
        if not all(value.dtype.kind in "fiu" and np.isfinite(value).all() for value in self.weights.values()):
            raise ValueError("non-finite forecast weights")
        gru_shapes = {"gru.weight_ih_l0": (48, 2), "gru.weight_hh_l0": (48, 16),
                      "gru.bias_ih_l0": (48,), "gru.bias_hh_l0": (48,),
                      "head.weight": (2, 16), "head.bias": (2,)}
        for horizon, settings in horizons.items():
            shapes = {}
            if settings["selected_method"] == "ridge":
                shapes = {f"h{horizon}_coef": (2, CONTEXT * 2 + 6), f"h{horizon}_intercept": (2,)}
            elif settings["selected_method"] == "gru":
                shapes = {f"h{horizon}_gru_{name}": shape for name, shape in gru_shapes.items()}
            if any(name not in self.weights or self.weights[name].shape != shape for name, shape in shapes.items()):
                raise ValueError("forecast weight keys or shapes mismatch")
        self.metadata: dict[str, Any] = metadata

    def predict(self, history: list[EmotionObservation], horizon: int = 1) -> EmotionForecast:
        """预测下一指定采样步；时间步不是小时或天，也不是临床恶化风险。"""
        validate_history(history)
        if type(horizon) is not int or str(horizon) not in self.metadata["horizons"]:
            raise ValueError("unsupported sampling-step horizon")
        settings = self.metadata["horizons"][str(horizon)]
        sequence = np.array([[[p.positive / 100, p.negative / 100]
                              for p in history[-CONTEXT:]]], dtype=np.float64)
        method = settings["selected_method"]
        if method == "persistence":
            prediction = sequence[:, -1]
        elif method == "ema":
            prediction = ema_prediction(sequence)
        elif method == "past_mean":
            prediction = sequence.mean(axis=1)
        elif method == "ridge":
            prefix = f"h{horizon}_"
            features = history_features(sequence)
            prediction = features @ self.weights[prefix + "coef"].T + self.weights[prefix + "intercept"]
        elif method == "gru":
            prediction = self._gru_prediction(sequence, horizon)
        else:
            raise ValueError("unsupported artifact model method")
        prediction = np.clip(prediction[0] * 100, 0, 100)
        residuals = np.asarray(settings["validation_absolute_error_quantile"], dtype=np.float64) * 100
        if residuals.shape != (2,) or not np.isfinite(residuals).all() or (residuals < 0).any():
            raise ValueError("invalid validation uncertainty estimate")
        intervals = {name: [float(max(0, prediction[i] - residuals[i])),
                            float(min(100, prediction[i] + residuals[i]))]
                     for i, name in enumerate(TARGETS)}
        return EmotionForecast(
            positive=float(prediction[0]), negative=float(prediction[1]),
            origin_step=history[-1].step, target_step=history[-1].step + horizon,
            method=method, history_count=len(history), model_id=self.metadata["model_id"],
            interval=intervals,
            warnings=["research_only_not_clinical_forecast", "population_transfer_not_validated",
                      "validation_error_bands_not_guaranteed_personal_coverage"],
        )

    def _gru_prediction(self, sequence: np.ndarray, horizon: int) -> np.ndarray:
        import torch
        from assessment.forecasting.sequence import SmallEmotionGRU

        model = SmallEmotionGRU()
        prefix = f"h{horizon}_gru_"
        state = {key[len(prefix):]: torch.from_numpy(value.copy())
                 for key, value in self.weights.items() if key.startswith(prefix)}
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(sequence, dtype=torch.float32)).numpy()
