"""
CT-PLEW 精确引擎测试
====================
验证心理势能景观与临界跃迁预警算法的核心性质：
  1. 多稳态检测：双稳态序列 → 检测到 ≥2 个吸引子 + 势能壁垒
  2. Kramers 闭式解：韧性越高逃逸率越低；horizon 单调性
  3. 无壁垒穿越率：漂移直指危机态时速率 = 速度/距离
  4. 临界慢化预警：自相关上升序列的 EWS 分显著高于平稳对照
  5. 因果倾斜：负倾斜（干预）抬高危机方向壁垒
  6. 反事实干预：干预父维度可降低子系统危机概率
  7. 管理器集成：precision_analysis 输出契约完整
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assessment.digital_twin.precision_engine import (  # noqa: E402
    CTPLEWEngine,
    CriticalSlowingDetector,
    KramersEscapeModel,
    PotentialLandscape,
    get_precision_engine,
)
from assessment.digital_twin import (  # noqa: E402
    CausalEdge,
    CausalGraph,
    DigitalTwinManager,
)

RNG = np.random.default_rng(42)


# ----------------------------------------------------------
# 合成数据工具
# ----------------------------------------------------------

def ou_series(mean: float, phi: float, sigma: float, n: int,
              rng: np.random.Generator = RNG) -> list[float]:
    """OU 过程（单稳态）"""
    x = mean
    out = []
    for _ in range(n):
        x = mean + phi * (x - mean) + sigma * rng.standard_normal()
        out.append(float(np.clip(x, 0.0, 1.0)))
    return out


def bistable_series(n_per_well: int = 250, phi: float = 0.7,
                    sigma: float = 0.04,
                    rng: np.random.Generator = RNG) -> list[float]:
    """双稳态序列：先停留健康阱(0.3)再切换危机阱(0.7)"""
    return (
        ou_series(0.3, phi, sigma, n_per_well, rng)
        + ou_series(0.7, phi, sigma, n_per_well, rng)
    )


def drifting_series(start: float, slope: float, sigma: float, n: int,
                    rng: np.random.Generator = RNG) -> list[float]:
    """带正漂移的序列（朝危机态滑行）"""
    t = np.arange(n)
    x = start + slope * t + sigma * rng.standard_normal(n)
    return [float(np.clip(v, 0.0, 1.0)) for v in x]


def ar1_ramp_series(n: int = 140, phi0: float = 0.3, phi1: float = 0.95,
                    sigma: float = 0.03,
                    rng: np.random.Generator = RNG) -> list[float]:
    """自相关系数线性上升的序列（临界慢化前兆）"""
    phis = np.linspace(phi0, phi1, n)
    x = 0.0
    out = []
    for phi in phis:
        x = phi * x + sigma * rng.standard_normal()
        out.append(float(np.clip(0.5 + x, 0.0, 1.0)))
    return out


# ----------------------------------------------------------
# 1. 多稳态检测
# ----------------------------------------------------------

class TestMultistability:
    def test_bistable_attractors_detected(self):
        series = bistable_series()
        land = PotentialLandscape.analyze("mood", series, x_now=0.32)
        assert land is not None
        assert land.n_attractors >= 2, "双稳态序列应检测到至少 2 个吸引子"
        assert len(land.barriers) >= 1, "双稳态序列应存在势能壁垒"
        assert land.barrier_height > 0.0

    def test_monostable_single_attractor(self):
        series = ou_series(0.35, 0.7, 0.03, 300)
        land = PotentialLandscape.analyze("mood", series, x_now=0.35)
        assert land is not None
        # 单阱 OU：内部壁垒至多 1 个（噪声伪迹），吸引子接近真实均值
        assert min(abs(a - 0.35) for a in land.attractors) < 0.2

    def test_insufficient_data_returns_none(self):
        assert PotentialLandscape.analyze("mood", [0.3, 0.4, 0.5], 0.4) is None


# ----------------------------------------------------------
# 2. Kramers 闭式解性质
# ----------------------------------------------------------

class TestKramersClosedForm:
    def test_higher_resilience_lower_rate(self):
        """韧性（ΔU/D）越高 → 逃逸率指数下降"""
        base = PotentialLandscape.analyze(
            "x", ou_series(0.3, 0.7, 0.05, 300), x_now=0.3
        )
        calm = PotentialLandscape.analyze(
            "x", ou_series(0.3, 0.7, 0.02, 300), x_now=0.3
        )
        assert base is not None and calm is not None
        r_noisy, _ = KramersEscapeModel.escape_rate(base)
        r_calm, _ = KramersEscapeModel.escape_rate(calm)
        assert calm.resilience > base.resilience
        assert r_calm <= r_noisy

    def test_horizon_monotonicity(self):
        rate = 0.05
        probs = [KramersEscapeModel.crisis_probability(rate, h)
                 for h in (1, 6, 12, 24, 48)]
        assert all(a <= b for a, b in zip(probs, probs[1:]))
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_no_barrier_drift_crossing_rate(self):
        """无壁垒 + 正漂移：速率 = 速度/距离（确定性穿越）"""
        series = drifting_series(0.5, 0.004, 0.01, 80)
        land = PotentialLandscape.analyze("x", series, x_now=series[-1])
        assert land is not None
        rate, method = KramersEscapeModel.escape_rate(land)
        if method == "no_barrier_drift":
            grid = np.asarray(land.grid)
            f_now = float(np.interp(land.x_now, grid, np.asarray(land.drift)))
            gap = land.crisis_barrier - land.x_now
            assert rate == pytest.approx(min(0.5, f_now / gap), rel=1e-6)
        else:
            # 若估计出壁垒，则必须走 Kramers 分支且概率合法
            assert method == "kramers"
            assert 0.0 <= KramersEscapeModel.crisis_probability(rate, 12) <= 1.0


# ----------------------------------------------------------
# 3. 临界慢化预警
# ----------------------------------------------------------

class TestCriticalSlowing:
    def test_ramping_ar1_triggers_ews(self):
        ramp = ar1_ramp_series(rng=np.random.default_rng(123))
        rep = CriticalSlowingDetector().analyze("x", ramp)
        assert rep is not None
        assert rep.ar1_trend_tau > 0.2, "自相关上升序列 τ 应显著为正"
        assert rep.ews_score > 0.3

    def test_stationary_control_low_ews(self):
        control = ou_series(0.5, 0.5, 0.03, 140, rng=np.random.default_rng(9))
        ramp = ar1_ramp_series(rng=np.random.default_rng(123))
        rep_c = CriticalSlowingDetector().analyze("x", control)
        rep_r = CriticalSlowingDetector().analyze("x", ramp)
        assert rep_c is not None and rep_r is not None
        assert rep_r.ews_score > rep_c.ews_score

    def test_short_series_returns_none(self):
        assert CriticalSlowingDetector().analyze("x", [0.5] * 10) is None


# ----------------------------------------------------------
# 4. 因果倾斜与反事实干预
# ----------------------------------------------------------

class TestCausalTiltAndCounterfactual:
    def test_negative_tilt_raises_barrier(self):
        """负倾斜（干预改善）应抬高危机方向壁垒"""
        series = drifting_series(0.45, 0.003, 0.02, 90)
        base = PotentialLandscape.analyze("child", series, x_now=series[-1])
        tilted = PotentialLandscape.analyze(
            "child", series, x_now=series[-1], tilt=-0.09
        )
        assert base is not None and tilted is not None
        assert tilted.resilience > base.resilience

    def test_counterfactual_reduces_crisis_prob(self):
        """父维度干预经因果链传导后，系统危机概率下降"""
        rng = np.random.default_rng(7)
        n = 60
        parent = [0.7 + 0.03 * rng.standard_normal() for _ in range(n)]
        child = drifting_series(0.5, 0.006, 0.008, n, rng)
        history = [{"parent": p, "child": c} for p, c in zip(parent, child)]
        graph = CausalGraph(user_id="u", edges=[
            CausalEdge(cause="parent", effect="child",
                       strength=0.6, lag=1, p_value=0.01, confidence=0.9),
        ], n_observations=n)

        engine = CTPLEWEngine(min_history=20)
        report = engine.analyze("u", history, graph, horizon=12)
        assert report.crisis_probability > 0.0
        if report.counterfactual is not None:
            assert report.counterfactual.post_crisis_prob < \
                report.counterfactual.baseline_crisis_prob
            assert report.counterfactual.benefit > 0.01


# ----------------------------------------------------------
# 5. 引擎整体行为
# ----------------------------------------------------------

class TestEngineBehavior:
    def test_cold_start_empty_report(self):
        engine = CTPLEWEngine()
        rep = engine.analyze("u", [{"a": 0.3}] * 5, None, 12)
        assert rep.crisis_probability == 0.0
        assert rep.escapes == {}

    def test_horizon_monotone_system_prob(self):
        rng = np.random.default_rng(11)
        history = [{"risk": v} for v in drifting_series(0.5, 0.006, 0.01, 60, rng)]
        engine = CTPLEWEngine(min_history=20)
        p6 = engine.analyze("u", history, None, 6).crisis_probability
        p24 = engine.analyze("u", history, None, 24).crisis_probability
        assert p24 >= p6

    def test_report_serializable(self):
        rng = np.random.default_rng(3)
        history = [{"m": v} for v in bistable_series(150, rng=rng)]
        rep = CTPLEWEngine(min_history=20).analyze("u", history, None, 12)
        d = rep.to_dict()
        assert d["algorithm"].startswith("CT-PLEW")
        import json
        json.dumps(d)  # 必须可 JSON 序列化


# ----------------------------------------------------------
# 6. 管理器集成
# ----------------------------------------------------------

class TestManagerIntegration:
    def test_precision_analysis_contract(self):
        mgr = DigitalTwinManager()
        rng = np.random.default_rng(5)
        for t in range(30):
            mgr.update_observation(
                "ptest",
                {"sleep": float(np.clip(0.4 + 0.05 * rng.standard_normal(), 0, 1)),
                 "anxiety": float(np.clip(0.5 + 0.002 * t + 0.04 * rng.standard_normal(), 0, 1))},
                timestamp=1000.0 + t,
            )
        d = mgr.precision_analysis("ptest", horizon=12)
        assert d["algorithm"].startswith("CT-PLEW")
        for key in ("landscapes", "ews", "escapes", "crisis_probability",
                    "critical_dim", "tipping_eta_steps", "counterfactual",
                    "ews_alert", "multistable"):
            assert key in d

    def test_singleton_engine(self):
        assert get_precision_engine() is get_precision_engine()
