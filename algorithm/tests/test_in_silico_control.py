# -*- coding: utf-8 -*-
"""
In-Silico Patient 与最优控制测试（支柱 B）
==========================================
验证生成式模拟与最优干预：
  1. InSilicoPatient.rollout 输出合法的平行未来（危机率∈[0,1]、CI 覆盖）
  2. 朝健康吸引子的救援 policy 降低（不升高）危机率
  3. InterventionMPC 在 toy 景观上找到不劣于基线的干预策略
  4. minimum_rescue_path 给出逆向 instanton（能量≥0、靶点有效）
  5. domino 命中频率归一化；结果可 JSON 序列化
  6. manager parallel_futures / optimal_rescue 冷启动回退
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assessment.digital_twin.in_silico_simulator import InSilicoPatient  # noqa: E402
from assessment.digital_twin.optimal_control import (  # noqa: E402
    InterventionMPC,
    minimum_rescue_path,
)

MODES = ["sleep", "rumination", "cognitive_distortion", "mood"]
LOAD = {"sleep": 0.5, "rumination": 0.5, "cognitive_distortion": 0.45, "mood": 0.55}


def _bistable_series(seed=7):
    """当前处于健康阱、但噪声足以偶尔跨越势垒的 4 模态双稳态数据"""
    rng = np.random.default_rng(seed)
    sched = ([-0.9] * 80 + [0.9] * 80) * 2 + [-0.9] * 60
    data = {m: [] for m in MODES}
    for z in sched:
        for m in MODES:
            v = 0.5 + LOAD[m] * z + 0.02 * rng.standard_normal()
            data[m].append(float(np.clip(v, 0.0, 1.0)))
    return data


@pytest.fixture(scope="module")
def sim():
    s = InSilicoPatient.from_history(_bistable_series(seed=7))
    assert s is not None
    return s


class TestRollout:
    def test_parallel_futures_valid(self, sim):
        fut = sim.rollout(K=200, horizon=22, seed=3)
        assert 0.0 <= fut.crisis_rate <= 1.0
        lo, hi = fut.crisis_ci
        assert lo <= fut.crisis_rate <= hi
        assert sorted(fut.mode_names) == sorted(MODES)
        # 分位轨迹长度 = horizon+1
        for m in MODES:
            assert len(fut.mean_mode_traj[m]) == 23
            assert len(fut.quantile_mode_traj[m]["q50"]) == 23
        json.dumps(fut.to_dict())

    def test_domino_freq_normalized(self, sim):
        fut = sim.rollout(K=200, horizon=25, seed=5)
        if fut.first_domino_freq:
            total = sum(fut.first_domino_freq.values())
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_reproducible_with_seed(self, sim):
        a = sim.rollout(K=120, horizon=15, seed=42).crisis_rate
        b = sim.rollout(K=120, horizon=15, seed=42).crisis_rate
        assert a == b


class TestRescuePolicy:
    def test_push_toward_health_reduces_crisis(self, sim):
        base = sim.rollout(K=250, horizon=25, seed=3)
        health = sim.model.health_fp.z

        def rescue(t, Z):
            diff = health - Z
            n = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-9
            return 0.15 * diff / n

        ctrl = sim.rollout(K=250, horizon=25, policy=rescue, seed=3)
        assert ctrl.intervention_applied
        assert ctrl.crisis_rate <= base.crisis_rate + 1e-9, \
            f"{base.crisis_rate} -> {ctrl.crisis_rate}"


class TestMPC:
    def test_mpc_not_worse_than_baseline(self, sim):
        mpc = InterventionMPC(sim, lam=0.05, u_max=0.3, n_candidates=16,
                              n_iters=3, K=60, seed=0)
        plan = mpc.optimize(horizon=12)
        assert plan.horizon == 12
        assert len(plan.control_latent) == 12
        # 控制量受 u_max 约束
        flat = [v for u in plan.control_latent for v in u]
        assert max(flat) <= 0.3 + 1e-9 and min(flat) >= -0.3 - 1e-9
        # 经验最优不应显著劣于基线
        assert plan.controlled_crisis_rate <= plan.baseline_crisis_rate + 0.03, \
            (plan.baseline_crisis_rate, plan.controlled_crisis_rate)
        assert plan.energy_cost >= 0.0
        assert plan.target_dim in MODES or plan.target_dim is None
        json.dumps(plan.to_dict())


class TestMinimumRescuePath:
    def test_reverse_instanton_valid(self, sim):
        rp = minimum_rescue_path(sim.model)
        assert rp is not None
        assert rp.n_steps > 0
        assert rp.energy >= 0.0
        assert len(rp.latent_path) == rp.n_steps + 1
        assert len(rp.control_latent) == rp.n_steps
        assert rp.target_dim in MODES
        json.dumps(rp.to_dict())


class TestManagerControlFallback:
    def test_parallel_futures_and_rescue_fallback(self):
        from assessment.digital_twin import DigitalTwinManager
        mgr = DigitalTwinManager()
        r_fut = mgr.parallel_futures("ghost", K=20, horizon=6)
        r_res = mgr.optimal_rescue("ghost", horizon=5, K=20)
        assert r_fut.get("fallback") is True
        assert r_res.get("fallback") is True
        json.dumps(r_fut)
        json.dumps(r_res)
