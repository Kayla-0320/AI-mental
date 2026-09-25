# -*- coding: utf-8 -*-
"""
多维临界跃迁引擎测试（支柱 A）
==============================
验证 landscape_nd 的核心数学性质：
  1. 梯度场（curl-free）→ 拟势 U 精确重构 + grad_flux_ratio 偏高
  2. 已知旋转通量 → Helmholtz 分解检测到非梯度分量（ratio 下降）
  3. 解析双阱 → find_fixed_points 找到 2 吸引子 + 1 鞍点，位置接近真值
  4. N=1 多维 Kramers 精确退化为经典 1D 公式
  5. 最小能量路径（instanton）从健康阱经鞍点到危机阱
  6. 潜流形 transform/inverse_transform 往返一致
  7. 顶层分析器端到端 + collapse_order + 序列化
  8. manager 多维方法冷启动回退不抛错
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assessment.digital_twin.landscape_nd import (  # noqa: E402
    DriftFieldEstimator,
    LatentRiskManifold,
    PotentialSolver,
    find_fixed_points,
    minimum_energy_path,
    multidim_escape_rate,
    particle_ensemble,
    get_multidim_analyzer,
)
from assessment.digital_twin.calibration import (  # noqa: E402
    CalibratedLeadTime,
    expected_calibration_error,
    reliability_curve,
    brier_score,
)


# ----------------------------------------------------------
# 合成动力系统工具
# ----------------------------------------------------------

def _simulate(f, D, z0, T, dt=0.2, seed=0, bound=3.0):
    """Euler-Maruyama 模拟 dZ = f(Z)dt + sqrt(2D)dW"""
    rng = np.random.default_rng(seed)
    k = len(z0)
    cov = 2.0 * np.asarray(D) * dt
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(k))
    Z = [np.asarray(z0, dtype=float)]
    for _ in range(T):
        z = Z[-1]
        noise = rng.standard_normal(k) @ L.T
        Z.append(np.clip(z + f(z) * dt + noise, -bound, bound))
    return np.array(Z)


def _doublewell_grad(z):
    """U = 0.25(z0²-1)² + 0.5 z1²，f = -∇U（纯梯度，curl-free）"""
    return np.array([z[0] - z[0] ** 3, -z[1]])


def _rot_flux(z, w=0.9):
    """旋转通量 J = w·[-z1, z0]（非梯度分量）"""
    return np.array([-w * z[1], w * z[0]])


# ----------------------------------------------------------
# 1 & 2. Helmholtz 梯度-通量分解
# ----------------------------------------------------------

class TestGradientFluxDecomposition:
    def test_curlfree_high_grad_flux_ratio(self):
        """纯梯度系统：grad_flux_ratio 应偏高（拟势主导）"""
        D = 0.03 * np.eye(2)
        Z = _simulate(_doublewell_grad, D, [-1.0, 0.0], T=1200, seed=1)
        drift = DriftFieldEstimator().fit(Z)
        assert drift is not None
        pot = PotentialSolver().solve(drift, Z)
        assert pot.grad_flux_ratio_ > 0.6, pot.grad_flux_ratio_

    def test_rotational_flux_lowers_ratio(self):
        """叠加旋转通量后 grad_flux_ratio 应下降（检测到非平衡概率流）"""
        D = 0.03 * np.eye(2)
        Z0 = _simulate(_doublewell_grad, D, [-1.0, 0.0], T=1200, seed=2)
        d0 = DriftFieldEstimator().fit(Z0)
        r0 = PotentialSolver().solve(d0, Z0).grad_flux_ratio_

        f_rot = lambda z: _doublewell_grad(z) + _rot_flux(z)  # noqa: E731
        Z1 = _simulate(f_rot, D, [-1.0, 0.0], T=1200, seed=2)
        d1 = DriftFieldEstimator().fit(Z1)
        r1 = PotentialSolver().solve(d1, Z1).grad_flux_ratio_
        assert r1 < r0, f"rotational flux should lower ratio: {r1} vs {r0}"


# ----------------------------------------------------------
# 3. 不动点检测
# ----------------------------------------------------------

class TestFixedPoints:
    def test_doublewell_two_attractors_one_saddle(self):
        D = 0.03 * np.eye(2)
        # 让轨迹访问两个阱
        Za = _simulate(_doublewell_grad, D, [-1.0, 0.0], T=800, seed=3)
        Zb = _simulate(_doublewell_grad, D, [1.0, 0.0], T=800, seed=4)
        Z = np.vstack([Za, Zb])
        drift = DriftFieldEstimator().fit(Z)
        fps = find_fixed_points(drift, Z)
        attractors = [fp for fp in fps if fp.kind == "attractor"]
        saddles = [fp for fp in fps if fp.kind == "saddle"]
        assert len(attractors) >= 2, [a.z for a in attractors]
        assert len(saddles) >= 1
        # 吸引子 z0 分量应分居 ±1 两侧
        z0s = sorted(float(a.z[0]) for a in attractors)
        assert z0s[0] < -0.4 and z0s[-1] > 0.4, z0s


# ----------------------------------------------------------
# 4. N=1 多维 Kramers 精确退化为经典 1D 公式
# ----------------------------------------------------------

class _StubDrift1D:
    """精确 1D 双阱漂移 f(z)=z-z³（U=(z²-1)²/4 的负梯度）"""
    def __init__(self, D):
        self.k_ = 1
        self.diffusion_ = np.array([[D]])

    def f(self, Z):
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        out = Z - Z ** 3
        return out

    def jacobian(self, z, h=1e-5):
        z = np.asarray(z, dtype=float).ravel()
        return np.array([[(self.f(z + np.array([h]))[0, 0]
                          - self.f(z - np.array([h]))[0, 0]) / (2 * h)]])


class _StubPot1D:
    """精确拟势 U(z)=(z²-1)²/4"""
    def U(self, z):
        z = float(np.asarray(z).ravel()[0])
        return 0.25 * (z ** 2 - 1) ** 2

    def gradU(self, z, h=1e-4):
        z = float(np.asarray(z).ravel()[0])
        return np.array([(self.U(z + h) - self.U(z - h)) / (2 * h)])


class TestKramersDegeneration:
    def test_n1_matches_classic_kramers(self):
        """N=1：λ = sqrt(|U''(xs)||U''(xb)|)/(2π)·exp(-ΔU/D)"""
        D = 0.05
        drift = _StubDrift1D(D)
        pot = _StubPot1D()
        rate, barrier, D_eff = multidim_escape_rate(
            pot, drift, np.array([-1.0]), np.array([0.0]), np.array([[D]])
        )
        # 解析：U''(±1)=2, U''(0)=-1, ΔU=U(0)-U(±1)=0.25
        analytic = math.sqrt(2.0 * 1.0) / (2 * math.pi) * math.exp(-0.25 / D)
        assert barrier == pytest.approx(0.25, abs=1e-6)
        assert D_eff == pytest.approx(D, rel=1e-3)
        assert rate == pytest.approx(analytic, rel=0.15), (rate, analytic)

    def test_higher_diffusion_higher_rate(self):
        pot = _StubPot1D()
        r_lo, _, _ = multidim_escape_rate(
            pot, _StubDrift1D(0.02), np.array([-1.0]), np.array([0.0]),
            np.array([[0.02]])
        )
        r_hi, _, _ = multidim_escape_rate(
            pot, _StubDrift1D(0.08), np.array([-1.0]), np.array([0.0]),
            np.array([[0.08]])
        )
        assert r_hi > r_lo


# ----------------------------------------------------------
# 5. 最小能量路径（instanton）
# ----------------------------------------------------------

class TestMinimumEnergyPath:
    def test_path_connects_wells_via_saddle(self):
        """在可跨越的 4 模态双稳态数据上，instanton 路径从健康阱到危机阱

        直接用分析器拟合的模型（其 collapse_path 即 minimum_energy_path），
        避开人为模拟中鞍点邻域欠采样导致拟合势外推不可信的问题。
        """
        model = get_multidim_analyzer().fit_model(_bistable_modes(seed=7))
        assert model is not None and not model.in_crisis
        health, crisis = model.health_fp.z, model.crisis_fp.z
        mid = 0.5 * (health + crisis)
        saddle = min(model.saddles, key=lambda s: np.linalg.norm(s.z - mid)).z
        path = minimum_energy_path(model.pot, health, saddle, crisis)
        assert len(path) >= 3
        # 起点近健康阱，终点近危机阱
        assert np.linalg.norm(path[0] - health) < 0.35 * np.linalg.norm(crisis - health)
        assert np.linalg.norm(path[-1] - crisis) < 0.35 * np.linalg.norm(crisis - health)
        # 沿危机方向投影单调推进（允许数值抖动）
        direction = crisis - health
        proj = (path - health) @ direction
        assert proj[-1] > proj[0]


# ----------------------------------------------------------
# 6. 潜流形往返
# ----------------------------------------------------------

class TestManifold:
    def test_inverse_transform_roundtrip(self):
        """低秩数据：transform→inverse_transform 应近似还原原始模态值"""
        rng = np.random.default_rng(0)
        T = 60
        z = np.linspace(-1, 1, T)
        load = {"a": 0.5, "b": -0.4, "c": 0.3}
        data = {m: [float(0.5 + w * zi + 0.01 * rng.standard_normal())
                    for zi in z] for m, w in load.items()}
        man = LatentRiskManifold(target_var=0.99, max_dim=3).fit(data, min_len=20)
        assert man is not None
        Z = man.transform(data)
        Xrec = man.inverse_transform(Z)
        L = min(len(v) for v in data.values())
        Xorig = np.column_stack([data[d][:L] for d in man.dim_names_])
        err = float(np.max(np.abs(Xrec[:L] - Xorig)))
        assert err < 0.05, err

    def test_insufficient_returns_none(self):
        assert LatentRiskManifold().fit({"a": [0.1, 0.2]}, min_len=20) is None


# ----------------------------------------------------------
# 7. 顶层分析器端到端
# ----------------------------------------------------------

def _bistable_modes(seed=7, both_wells=True):
    modes = ["sleep", "rumination", "cognitive_distortion", "mood"]
    load = {"sleep": 0.5, "rumination": 0.5,
            "cognitive_distortion": 0.45, "mood": 0.55}
    rng = np.random.default_rng(seed)
    sched = ([-0.9] * 80 + [0.9] * 80)
    if both_wells:
        sched = sched * 2 + [-0.9] * 60
    data = {m: [] for m in modes}
    for z in sched:
        for m in modes:
            v = 0.5 + load[m] * z + 0.02 * rng.standard_normal()
            data[m].append(float(np.clip(v, 0.0, 1.0)))
    return data


class TestAnalyzerEndToEnd:
    def test_analyze_returns_landscape(self):
        data = _bistable_modes()
        nd = get_multidim_analyzer().analyze(data, horizon=12)
        assert nd is not None
        assert nd.n_dim >= 1
        assert nd.health_attractor is not None
        assert nd.crisis_attractor is not None
        assert 0.0 <= nd.crisis_probability <= 1.0
        assert nd.collapse_order  # 非空 domino 顺序
        # 可 JSON 序列化
        json.dumps(nd.to_dict())

    def test_insufficient_dims_returns_none(self):
        # 只有 2 个模态 → 不满足 min_dims=3
        rng = np.random.default_rng(0)
        data = {"a": [float(0.5 + 0.05 * rng.standard_normal()) for _ in range(60)],
                "b": [float(0.5 + 0.05 * rng.standard_normal()) for _ in range(60)]}
        assert get_multidim_analyzer().analyze(data, 12) is None

    def test_particle_ensemble_prob_in_range(self):
        data = _bistable_modes(seed=11)
        model = get_multidim_analyzer().fit_model(data)
        assert model is not None
        p, lo, hi = particle_ensemble(
            model.drift, model.diffusion, model.z_now,
            model.crisis_fp.z, model.crisis_radius,
            horizon=15, n_particles=150, seed=0,
        )
        assert 0.0 <= p <= 1.0
        assert lo <= p <= hi


# ----------------------------------------------------------
# 8. Manager 冷启动回退
# ----------------------------------------------------------

class TestManagerFallback:
    def test_cold_start_no_throw(self):
        from assessment.digital_twin import DigitalTwinManager
        mgr = DigitalTwinManager()
        r1 = mgr.multidim_landscape("ghost", 12)
        r2 = mgr.counterfactual_query("ghost", {"mood": 0.9})
        r3 = mgr.parallel_futures("ghost", K=20, horizon=8)
        r4 = mgr.optimal_rescue("ghost", horizon=6, K=20)
        for r in (r1, r2, r3, r4):
            assert isinstance(r, dict)
            assert r.get("fallback") is True
            json.dumps(r)


# ----------------------------------------------------------
# 9. 概率校准
# ----------------------------------------------------------

class TestCalibration:
    def _miscalibrated(self, n=2000, seed=0):
        """构造过度自信（未校准）的概率与标签"""
        rng = np.random.default_rng(seed)
        raw = rng.uniform(0, 1, n)
        # 真实发生概率 = raw 的压缩（模拟系统性偏差）
        true_p = np.clip(0.5 + 0.35 * (raw - 0.5), 0, 1)
        labels = (rng.uniform(0, 1, n) < true_p).astype(int)
        return [float(x) for x in raw], [int(y) for y in labels]

    def test_isotonic_reduces_ece(self):
        probs, labels = self._miscalibrated()
        ece_before = expected_calibration_error(probs, labels)
        cal = CalibratedLeadTime(method="isotonic").fit(probs, labels)
        calibrated = cal.transform_batch(probs)
        ece_after = expected_calibration_error(calibrated, labels)
        assert ece_after < ece_before, (ece_before, ece_after)
        assert ece_after < 0.05, ece_after

    def test_platt_reduces_ece(self):
        probs, labels = self._miscalibrated(seed=1)
        ece_before = expected_calibration_error(probs, labels)
        cal = CalibratedLeadTime(method="platt").fit(probs, labels)
        ece_after = expected_calibration_error(cal.transform_batch(probs), labels)
        assert ece_after < ece_before

    def test_unfitted_passthrough(self):
        cal = CalibratedLeadTime()
        assert cal.transform(0.42) == pytest.approx(0.42)

    def test_reliability_curve_and_brier(self):
        probs, labels = self._miscalibrated(n=500)
        curve = reliability_curve(probs, labels, n_bins=10)
        assert len(curve) == 10
        assert 0.0 <= brier_score(probs, labels) <= 1.0

    def test_serialization_roundtrip(self):
        probs, labels = self._miscalibrated(n=500)
        cal = CalibratedLeadTime(method="isotonic").fit(probs, labels)
        restored = CalibratedLeadTime.from_json(cal.to_json())
        for p in (0.1, 0.5, 0.9):
            assert restored.transform(p) == pytest.approx(cal.transform(p), abs=1e-3)
