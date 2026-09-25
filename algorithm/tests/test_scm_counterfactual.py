# -*- coding: utf-8 -*-
"""
结构因果模型与反事实测试（支柱 B）
==================================
验证 scm_counterfactual 的因果正确性：
  1. VAR(1) 拟合恢复已知结构系数（sleep→mood 负、rumination→mood 正）
  2. do-算子正确截断入边：干预父节点后子节点分布按因果方向移动
  3. Pearl 三步反事实：无干预（pulse 到实际值）时反事实恒等于事实
  4. 持续干预 do 与反事实方向一致
  5. 数据不足 → None；结果可 JSON 序列化
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assessment.digital_twin.scm_counterfactual import (  # noqa: E402
    StructuralCausalModel,
)


def _make_scm_data(n=250, seed=0):
    """已知滞后结构：mood_{t+1} = 0.2 mood_t - 0.4 sleep_t + 0.5 rumin_t"""
    rng = np.random.default_rng(seed)
    hist = []
    s = r = m = 0.5
    for _ in range(n):
        hist.append({'sleep': float(s), 'rumination': float(r), 'mood': float(m)})
        ns = np.clip(0.5 + 0.5 * (s - 0.5) + 0.05 * rng.standard_normal(), 0, 1)
        nr = np.clip(0.5 + 0.5 * (r - 0.5) + 0.05 * rng.standard_normal(), 0, 1)
        nm = np.clip(0.5 + 0.2 * (m - 0.5) - 0.4 * (s - 0.5)
                     + 0.5 * (r - 0.5) + 0.03 * rng.standard_normal(), 0, 1)
        s, r, m = ns, nr, nm
    return hist


class TestSCMFit:
    def test_recovers_known_coefficients(self):
        scm = StructuralCausalModel.fit(_make_scm_data())
        assert scm is not None
        i_mood = scm.dims.index('mood')
        i_sleep = scm.dims.index('sleep')
        i_rum = scm.dims.index('rumination')
        # mood 行：sleep 系数应显著为负，rumination 应显著为正
        assert scm.A[i_mood, i_sleep] < -0.15, scm.A[i_mood]
        assert scm.A[i_mood, i_rum] > 0.15, scm.A[i_mood]

    def test_insufficient_data_returns_none(self):
        assert StructuralCausalModel.fit([{'a': 0.5}] * 5) is None

    def test_serializable(self):
        scm = StructuralCausalModel.fit(_make_scm_data())
        json.dumps(scm.to_dict())


class TestDoOperator:
    def test_do_truncates_and_propagates(self):
        """do(sleep 高) 应降低 mood（风险），且不受 sleep 的父节点影响"""
        scm = StructuralCausalModel.fit(_make_scm_data())
        res = scm.do({'sleep': 0.9}, horizon=12, clamp=True, n_samples=400, seed=1)
        assert res is not None
        assert res.effect['mood'] < -0.05, res.effect
        # do(rumination 高) 应提高 mood
        res2 = scm.do({'rumination': 0.9}, horizon=12, clamp=True, n_samples=400, seed=1)
        assert res2.effect['mood'] > 0.05, res2.effect
        json.dumps(res.to_dict())

    def test_do_empty_or_unknown_returns_none(self):
        scm = StructuralCausalModel.fit(_make_scm_data())
        assert scm.do({}, horizon=5) is None
        assert scm.do({'nonexistent_dim': 0.9}, horizon=5) is None


class TestCounterfactual:
    def test_noop_counterfactual_equals_factual(self):
        """Pearl 三步自洽：pulse 到该步实际值 → 反事实 == 事实"""
        hist = _make_scm_data()
        scm = StructuralCausalModel.fit(hist)
        step = 120
        cfn = scm.counterfactual(
            hist, {'sleep': float(hist[step]['sleep'])},
            intervention_step=step, horizon=8, clamp=False,
        )
        assert cfn is not None
        assert max(abs(v) for v in cfn.effect.values()) < 1e-9, cfn.effect

    def test_sustained_sleep_improves_mood(self):
        hist = _make_scm_data()
        scm = StructuralCausalModel.fit(hist)
        cf = scm.counterfactual(hist, {'sleep': 0.95}, intervention_step=120,
                                horizon=8, clamp=True)
        assert cf is not None
        assert cf.effect['mood'] < -0.02, cf.effect
        assert cf.narrative
        json.dumps(cf.to_dict())

    def test_pulse_effect_visible_short_horizon(self):
        """单步脉冲在短窗口内立即可见（快速均值回复系统）"""
        hist = _make_scm_data()
        scm = StructuralCausalModel.fit(hist)
        cf = scm.counterfactual(hist, {'sleep': 0.95}, intervention_step=120,
                                horizon=1)
        assert cf.effect['mood'] < -0.05, cf.effect

    def test_counterfactual_insufficient_returns_none(self):
        scm = StructuralCausalModel.fit(_make_scm_data())
        assert scm.counterfactual([{'sleep': 0.5, 'rumination': 0.5, 'mood': 0.5}],
                                  {'sleep': 0.9}) is None
