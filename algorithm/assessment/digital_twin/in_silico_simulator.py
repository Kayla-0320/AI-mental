"""
In-Silico Patient：生成式平行未来模拟器（支柱 B 之二）
=====================================================
**景观即模拟器**——直接复用支柱 A 估出的个体化漂移场 f(Z)、扩散张量 Σ
与危机区域，做潜空间随机微分方程的 Euler-Maruyama 积分：

    dZ = [f(Z) + u(Z,t)] dt + sqrt(2 D) dW

从当前状态出发模拟 K 条**平行未来**轨迹，统计：
  - 危机率分布（进入危机吸引子邻域的轨迹比例）+ Wilson 置信区间；
  - 各模态的分位轨迹（10/50/90 百分位扇形带）；
  - domino 命中频率（哪条轨迹里"哪个模态先崩"的分布）；
  - 平均首次崩溃时间。

policy 接口
-----------
policy(t, Z) -> 控制增量 u (K, k)，叠加到漂移上，用于 MPC 最优干预
（见 optimal_control.py）。policy=None 时为自然演化（无干预基线）。

诚实性约束
----------
- 潜空间 SDE 是文档化近似；模态轨迹经 PCA 伪逆重构，仅保留主成分子空间。
- K 与 horizon 受性能上限约束；结果附带经验置信区间而非点估计。
- 数据不足或无清晰双稳态时 from_history 返回 None，由上层回退。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .landscape_nd import (
    LandscapeModel,
    MultidimLandscapeAnalyzer,
    _wilson_ci,
    get_multidim_analyzer,
)

logger = logging.getLogger(__name__)

# policy: (step_idx, Z (K,k)) -> 控制增量 (K,k) 或 (k,) 或 None
PolicyFn = Callable[[int, np.ndarray], Optional[np.ndarray]]


@dataclass
class ParallelFutures:
    """K 条平行未来的统计汇总"""
    n_particles: int
    horizon: int
    crisis_rate: float                      # 进入危机区域的轨迹比例
    crisis_ci: tuple                        # (lo, hi) Wilson 区间
    mode_names: list[str]
    mean_mode_traj: dict[str, list[float]] = field(default_factory=dict)
    quantile_mode_traj: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    first_domino_freq: dict[str, float] = field(default_factory=dict)
    time_to_crisis_mean: Optional[float] = None
    intervention_applied: bool = False
    method: str = "in_silico_latent_sde"

    def to_dict(self) -> dict:
        return {
            "n_particles": self.n_particles,
            "horizon": self.horizon,
            "crisis_rate": round(self.crisis_rate, 4),
            "crisis_ci": [round(self.crisis_ci[0], 4), round(self.crisis_ci[1], 4)],
            "mode_names": self.mode_names,
            "mean_mode_traj": {
                k: [round(float(v), 4) for v in seq] for k, seq in self.mean_mode_traj.items()
            },
            "quantile_mode_traj": {
                k: {q: [round(float(v), 4) for v in seq] for q, seq in qs.items()}
                for k, qs in self.quantile_mode_traj.items()
            },
            "first_domino_freq": {k: round(float(v), 4) for k, v in self.first_domino_freq.items()},
            "time_to_crisis_mean": (
                round(self.time_to_crisis_mean, 3)
                if self.time_to_crisis_mean is not None else None
            ),
            "intervention_applied": self.intervention_applied,
            "method": self.method,
        }


class InSilicoPatient:
    """生成式平行未来模拟器（复用支柱 A 的 f、Σ、危机区域）"""

    def __init__(self, model: LandscapeModel, dt: float = 1.0):
        self.model = model
        self.dt = dt
        self.drift = model.drift
        self.diffusion = model.diffusion
        self.manifold = model.manifold
        self.k = model.drift.k_
        # 危机/健康吸引子的模态空间投影，用于逐模态 domino 阈值
        self._crisis_modes = self.manifold.inverse_transform(model.crisis_fp.z)
        self._health_modes = self.manifold.inverse_transform(model.health_fp.z)
        # 每模态"崩溃阈值" = 健康值与危机值的中点（朝危机方向跨越即算 domino）
        self._mode_thr = 0.5 * (self._health_modes + self._crisis_modes)
        # 朝危机方向：crisis - health 的符号（+1 表示该模态上升=恶化）
        self._mode_dir = np.sign(self._crisis_modes - self._health_modes + 1e-9)
        self._chol: Optional[np.ndarray] = None

    # ----------------------------------------------------------
    @classmethod
    def from_history(
        cls,
        data: dict[str, list[float]],
        min_history: int = 40,
        min_dims: int = 3,
        analyzer: Optional[MultidimLandscapeAnalyzer] = None,
    ) -> Optional["InSilicoPatient"]:
        """从 history 拟合；数据不足或无双稳态时返回 None。"""
        an = analyzer or get_multidim_analyzer()
        model = an.fit_model(data)
        if model is None:
            return None
        return cls(model)

    def _noise_chol(self) -> np.ndarray:
        if self._chol is None:
            cov = 2.0 * self.diffusion * self.dt + 1e-9 * np.eye(self.k)
            try:
                self._chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                w, V = np.linalg.eigh(self.diffusion)
                w = np.clip(w, 1e-9, None)
                self._chol = V @ np.diag(np.sqrt(2.0 * w * self.dt)) @ V.T
        return self._chol

    # ----------------------------------------------------------
    def rollout(
        self,
        K: int = 200,
        horizon: int = 20,
        policy: Optional[PolicyFn] = None,
        state0: Optional[np.ndarray] = None,
        seed: int = 0,
        quantiles: tuple = (10, 50, 90),
    ) -> ParallelFutures:
        """模拟 K 条平行未来，返回统计汇总

        Args:
            K: 轨迹数（性能上限建议 <=500）
            horizon: 前向步数
            policy: 干预策略 (t, Z)->u；None=自然演化
            state0: 起始潜状态 (k,)；默认用当前观测 z_now
            seed: 随机种子（可复现）
        """
        K = int(max(1, min(K, 500)))
        horizon = int(max(1, min(horizon, 200)))
        rng = np.random.default_rng(seed)
        Lmat = self._noise_chol()
        z0 = self.model.z_now if state0 is None else np.asarray(state0, dtype=float)

        Z = np.tile(z0, (K, 1))                       # (K, k)
        crisis_center = self.model.crisis_fp.z
        crisis_radius = self.model.crisis_radius

        mode_traj = np.zeros((horizon + 1, K, len(self.manifold.dim_names_)))
        mode_traj[0] = self.manifold.inverse_transform(Z)

        hit = np.zeros(K, dtype=bool)
        ttc = np.full(K, np.nan)
        crossed = np.zeros((K, len(self.manifold.dim_names_)), dtype=bool)
        first_domino = np.full(K, -1, dtype=int)

        for t in range(horizon):
            fz = self.drift.f(Z)
            u = policy(t, Z) if policy is not None else None
            if u is not None:
                u = np.asarray(u, dtype=float)
                if u.ndim == 1:
                    u = np.tile(u, (K, 1))
                fz = fz + u
            noise = rng.standard_normal((K, self.k)) @ Lmat.T
            Z = Z + fz * self.dt + noise
            clip = getattr(self.drift, "clip_domain", None)
            if clip is not None:
                Z = clip(Z)          # 钳制到训练域，防潜状态漂出致多项式漂移溢出
            Xm = self.manifold.inverse_transform(Z)
            mode_traj[t + 1] = Xm

            # 危机命中（首次进入危机吸引子邻域）
            d = np.linalg.norm(Z - crisis_center, axis=1)
            newly = (d < crisis_radius) & (~hit)
            if np.any(newly):
                ttc[newly] = t + 1
            hit |= newly

            # domino：某模态朝危机方向首次跨越阈值
            advance = (Xm - self._health_modes) * self._mode_dir   # >0 = 朝危机恶化
            thr_advance = (self._mode_thr - self._health_modes) * self._mode_dir
            newly_cross = (advance > thr_advance) & (~crossed)
            crossed |= newly_cross
            for j in range(newly_cross.shape[1]):
                first_new = newly_cross[:, j] & (first_domino < 0)
                first_domino[first_new] = j

        return self._summarize(mode_traj, hit, ttc, first_domino, horizon, K,
                               policy is not None, quantiles)

    def _summarize(self, mode_traj, hit, ttc, first_domino, horizon, K,
                   intervened, quantiles) -> ParallelFutures:
        names = list(self.manifold.dim_names_)
        hits = int(hit.sum())
        crisis_rate = hits / K
        lo, hi = _wilson_ci(hits, K)

        mean_traj = {
            names[j]: [float(mode_traj[t, :, j].mean()) for t in range(horizon + 1)]
            for j in range(len(names))
        }
        q_traj = {}
        for j, nm in enumerate(names):
            q_traj[nm] = {}
            for q in quantiles:
                q_traj[nm][f"q{q}"] = [
                    float(np.percentile(mode_traj[t, :, j], q)) for t in range(horizon + 1)
                ]
        # first domino 频率（仅在记录到首张 domino 的轨迹中归一化）
        freq: dict[str, float] = {}
        valid = first_domino[first_domino >= 0]
        if len(valid) > 0:
            for j, nm in enumerate(names):
                c = int(np.sum(valid == j))
                if c > 0:
                    freq[nm] = c / len(valid)
        ttc_mean = float(np.nanmean(ttc)) if np.any(~np.isnan(ttc)) else None

        return ParallelFutures(
            n_particles=K, horizon=horizon,
            crisis_rate=crisis_rate, crisis_ci=(lo, hi),
            mode_names=names, mean_mode_traj=mean_traj,
            quantile_mode_traj=q_traj, first_domino_freq=freq,
            time_to_crisis_mean=ttc_mean, intervention_applied=intervened,
        )
