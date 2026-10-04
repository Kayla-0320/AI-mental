"""
概率校准与可靠性评估
====================
CT-PLEW 输出的危机概率需要**可临床采信**——即"预测 80% 的那批用户，
实际约 80% 真的发生危机"。本模块提供：

1. CalibratedLeadTime：把原始分数（Kramers 概率 / 复合分）通过等渗回归
   （isotonic）或 Platt 缩放映射到校准概率；未拟合时恒等透传。
2. reliability_curve：可靠性曲线（分箱预测概率 vs 实际频率）。
3. expected_calibration_error (ECE)：单一校准误差指标。
4. brier_score：校准 + 判别联合的proper评分。

设计原则
--------
- 无 scipy/sklearn 也可降级：等渗回归优先用 sklearn，缺失时退化为
  分箱保序（PAV 简化）。
- 校准器可 JSON 序列化（保存分位点），便于持久化与跨进程复用。

理论依据
--------
- Isotonic calibration (Zadrozny & Elkan, 2002)
- Platt scaling (Platt, 1999)
- Expected Calibration Error (Guo et al., 2017; Naeini et al., 2015 ECE)
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 可靠性曲线与校准误差（无需拟合，直接评估）
# ============================================================

def reliability_curve(probs: list[float], labels: list[int],
                      n_bins: int = 10) -> list[dict]:
    """可靠性曲线：按预测概率等宽分箱，统计每箱 [预测均值, 实际频率, 样本数]"""
    if not probs:
        return []
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    curve = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            curve.append({
                "bin_lo": round(float(lo), 3), "bin_hi": round(float(hi), 3),
                "mean_predicted": None, "observed_frequency": None, "count": 0,
            })
        else:
            curve.append({
                "bin_lo": round(float(lo), 3), "bin_hi": round(float(hi), 3),
                "mean_predicted": round(float(p[mask].mean()), 4),
                "observed_frequency": round(float(y[mask].mean()), 4),
                "count": n,
            })
    return curve


def expected_calibration_error(probs: list[float], labels: list[int],
                               n_bins: int = 10) -> float:
    """ECE = Σ_b (n_b / N) · |acc_b - conf_b|，越低越好（0=完美校准）"""
    if not probs:
        return 0.0
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    N = len(p)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += (n / N) * abs(acc - conf)
    return float(ece)


def brier_score(probs: list[float], labels: list[int]) -> float:
    """Brier = mean((p - y)²)，越低越好"""
    if not probs:
        return 0.0
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2))


# ============================================================
# 校准器（等渗 / Platt）
# ============================================================

@dataclass
class CalibratedLeadTime:
    """把原始危机分数映射到校准概率

    method="isotonic"：保序回归（非参、单调，样本足时首选）。
    method="platt"：逻辑回归 Platt 缩放（参数量少，样本少时更稳）。
    未 fit 时 transform 恒等透传（clip 到 [0,1]）。

    可序列化：isotonic 存 (x, y) 分位点，platt 存 (a, b)。
    """
    method: str = "isotonic"
    _fitted: bool = False
    _iso_x: list[float] = field(default_factory=list)
    _iso_y: list[float] = field(default_factory=list)
    _platt_a: float = 1.0
    _platt_b: float = 0.0

    # ----------------------------------------------------------
    def fit(self, probs: list[float], labels: list[int]) -> "CalibratedLeadTime":
        if len(probs) < 8:
            logger.warning("校准样本不足（<8），跳过拟合")
            return self
        p = np.asarray(probs, dtype=float)
        y = np.asarray(labels, dtype=float)
        if self.method == "platt":
            self._fit_platt(p, y)
        else:
            self._fit_isotonic(p, y)
        self._fitted = True
        return self

    def _fit_isotonic(self, p: np.ndarray, y: np.ndarray):
        try:
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p, y)
            # 存单调映射的分位点（去重）
            xs = np.linspace(p.min(), p.max(), 64)
            ys = iso.predict(xs)
            self._iso_x = [float(v) for v in xs]
            self._iso_y = [float(v) for v in ys]
        except Exception as e:  # sklearn 不可用则退化为分箱保序
            logger.info(f"isotonic sklearn 失败({e})，退化分箱保序")
            self._fit_binned_pav(p, y)

    def _fit_binned_pav(self, p: np.ndarray, y: np.ndarray, n_bins: int = 12):
        order = np.argsort(p)
        ps, ys = p[order], y[order]
        edges = np.linspace(0, len(ps), n_bins + 1).astype(int)
        xs, means = [], []
        for b in range(n_bins):
            s, e = edges[b], edges[b + 1]
            if e > s:
                xs.append(float(ps[s:e].mean()))
                means.append(float(ys[s:e].mean()))
        # 保序化（PAV 简化：累计最大单调）
        mono = []
        run_max = 0.0
        for m in means:
            run_max = max(run_max, m)
            mono.append(run_max)
        self._iso_x = xs
        self._iso_y = mono

    def _fit_platt(self, p: np.ndarray, y: np.ndarray):
        # 一维逻辑回归：p_cal = sigmoid(a * logit(p) + b)
        eps = 1e-6
        pc = np.clip(p, eps, 1 - eps)
        logit = np.log(pc / (1 - pc))
        X = np.column_stack([logit, np.ones_like(logit)])
        # 牛顿/梯度下降拟合逻辑回归（无 sklearn 依赖）
        w = np.zeros(2)
        for _ in range(200):
            z = X @ w
            mu = 1.0 / (1.0 + np.exp(-z))
            grad = X.T @ (mu - y) / len(y)
            Wd = mu * (1 - mu) + 1e-9
            H = X.T @ (X * Wd[:, None]) / len(y) + 1e-6 * np.eye(2)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            w -= step
            if np.linalg.norm(step) < 1e-8:
                break
        self._platt_a, self._platt_b = float(w[0]), float(w[1])

    # ----------------------------------------------------------
    def transform(self, prob: float) -> float:
        """原始分数 -> 校准概率"""
        p = float(np.clip(prob, 0.0, 1.0))
        if not self._fitted:
            return p
        if self.method == "platt":
            eps = 1e-6
            pc = min(max(p, eps), 1 - eps)
            logit = math.log(pc / (1 - pc))
            return float(1.0 / (1.0 + math.exp(-(self._platt_a * logit + self._platt_b))))
        # isotonic：线性插值单调映射
        if not self._iso_x:
            return p
        return float(np.clip(np.interp(p, self._iso_x, self._iso_y), 0.0, 1.0))

    def transform_batch(self, probs: list[float]) -> list[float]:
        return [self.transform(p) for p in probs]

    # ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "fitted": self._fitted,
            "iso_x": [round(v, 5) for v in self._iso_x],
            "iso_y": [round(v, 5) for v in self._iso_y],
            "platt_a": round(self._platt_a, 5),
            "platt_b": round(self._platt_b, 5),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "CalibratedLeadTime":
        obj = cls(method=d.get("method", "isotonic"))
        obj._fitted = bool(d.get("fitted", False))
        obj._iso_x = list(d.get("iso_x", []))
        obj._iso_y = list(d.get("iso_y", []))
        obj._platt_a = float(d.get("platt_a", 1.0))
        obj._platt_b = float(d.get("platt_b", 0.0))
        return obj

    @classmethod
    def from_json(cls, s: str) -> "CalibratedLeadTime":
        return cls.from_dict(json.loads(s))
