"""
最优干预控制（In-Silico Patient 支柱 B 之三）
============================================
在生成式模拟器（复用支柱 A 的 f、Σ）上求解**最省力的救援**，回答：
"要在虚拟世界里把这个孩子从崩溃边缘拉回来，应该在何时、对哪个模态、
用多大力度干预？"

两条互补路径
------------
1. InterventionMPC（模型预测控制 / 随机最优控制）：
   决策变量 = 潜空间控制序列 u_{0:H-1}；目标
       J(u) = E[危机命中率(u)] + λ · Σ_t ||u_t||²
   用交叉熵法（CEM）在模拟器上滚动优化，比固定时机网格搜索更强，
   能同时优化"时机 + 靶点 + 剂量"。

2. minimum_rescue_path（最小能量救援 / 逆向 instanton）：
   崩溃的最小作用量路径（instanton）反向即"最省力救援轨迹"。沿反向
   路径逐点计算抵消自然漂移所需的控制 u = (z_{t+1}-z_t)/dt - f(z_t)，
   闭式给出救援的剂量方向，并反投影为"该推哪个模态、推多少"。

诚实性约束
----------
- 控制量在潜空间施加并经 PCA 伪逆映射到模态，属文档化近似。
- MPC 用有限粒子/有限候选评估，结果为经验最优而非全局最优；返回中如实
  标注基线 vs 干预后的危机率与置信区间，包含"干预不占优"的情形。
- 救援能量 Σ||u||² 反映 Freidlin-Wentzell 作用量的离散近似。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .in_silico_simulator import InSilicoPatient
from .landscape_nd import LandscapeModel, minimum_energy_path

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class InterventionPlan:
    """MPC 最优干预方案"""
    horizon: int
    control_latent: list[list[float]]              # 每步潜空间控制 (horizon, k)
    mode_dose: dict[str, float]                    # 聚合到模态的平均干预剂量（带符号）
    target_dim: Optional[str]                      # 剂量最大的靶模态
    baseline_crisis_rate: float
    controlled_crisis_rate: float
    baseline_ci: tuple
    controlled_ci: tuple
    reduction: float                               # 危机率下降（正=有效）
    energy_cost: float                             # Σ||u||²
    lam: float
    iterations: int
    converged: bool
    narrative: str = ""
    method: str = "cem_mpc"

    def to_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "control_latent": [
                [round(float(v), 4) for v in u] for u in self.control_latent
            ],
            "mode_dose": {k: round(float(v), 4) for k, v in self.mode_dose.items()},
            "target_dim": self.target_dim,
            "baseline_crisis_rate": round(self.baseline_crisis_rate, 4),
            "controlled_crisis_rate": round(self.controlled_crisis_rate, 4),
            "baseline_ci": [round(self.baseline_ci[0], 4), round(self.baseline_ci[1], 4)],
            "controlled_ci": [round(self.controlled_ci[0], 4), round(self.controlled_ci[1], 4)],
            "reduction": round(self.reduction, 4),
            "energy_cost": round(self.energy_cost, 5),
            "lam": self.lam,
            "iterations": self.iterations,
            "converged": self.converged,
            "narrative": self.narrative,
            "method": self.method,
        }


@dataclass
class RescuePath:
    """最小能量救援轨迹（逆向 instanton）"""
    latent_path: list[list[float]]                 # crisis->health 有序潜轨迹
    control_latent: list[list[float]]              # 每点所需控制
    mode_dose: dict[str, float]                    # 峰值救援剂量（模态空间，带符号）
    target_dim: Optional[str]
    energy: float                                  # Σ||u||²
    barrier_restored: float                        # 健康阱壁垒高度（救援后重新抬高）
    n_steps: int
    narrative: str = ""
    method: str = "reverse_instanton"

    def to_dict(self) -> dict:
        return {
            "latent_path": [[round(float(v), 4) for v in z] for z in self.latent_path],
            "control_latent": [[round(float(v), 4) for v in u] for u in self.control_latent],
            "mode_dose": {k: round(float(v), 4) for k, v in self.mode_dose.items()},
            "target_dim": self.target_dim,
            "energy": round(self.energy, 5),
            "barrier_restored": round(self.barrier_restored, 5),
            "n_steps": self.n_steps,
            "narrative": self.narrative,
            "method": self.method,
        }


# ============================================================
# MPC 最优干预（交叉熵法）
# ============================================================

class InterventionMPC:
    """在 In-Silico 模拟器上用 CEM 求解最优干预控制序列"""

    def __init__(self, sim: InSilicoPatient, lam: float = 0.05,
                 u_max: float = 0.3, n_candidates: int = 24,
                 n_iters: int = 5, elite_frac: float = 0.25,
                 K: int = 80, seed: int = 0):
        self.sim = sim
        self.lam = lam
        self.u_max = u_max
        self.n_candidates = n_candidates
        self.n_iters = n_iters
        self.elite_frac = elite_frac
        self.K = K
        self.seed = seed

    def _cost(self, U: np.ndarray, horizon: int, seed: int) -> tuple[float, tuple]:
        """给定控制序列 U (horizon, k)，返回 (J, ci)"""
        U = np.clip(U, -self.u_max, self.u_max)

        def policy(t, Z):
            return U[min(t, len(U) - 1)]

        fut = self.sim.rollout(K=self.K, horizon=horizon, policy=policy, seed=seed)
        energy = float(np.sum(U ** 2))
        J = fut.crisis_rate + self.lam * energy / max(1, horizon)
        return J, fut.crisis_ci

    def optimize(self, horizon: int = 15, seed: Optional[int] = None) -> InterventionPlan:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        k = self.sim.k
        horizon = int(max(1, min(horizon, 60)))
        dim = horizon * k

        # CEM 初始化：零均值控制 + 适度方差
        mean = np.zeros(dim)
        std = np.full(dim, self.u_max * 0.6)
        n_elite = max(2, int(self.n_candidates * self.elite_frac))
        best_U = np.zeros((horizon, k))
        best_J = math.inf
        converged = False

        for it in range(self.n_iters):
            samples = mean + std * rng.standard_normal((self.n_candidates, dim))
            samples = np.clip(samples, -self.u_max, self.u_max)
            costs = []
            for c in range(self.n_candidates):
                U = samples[c].reshape(horizon, k)
                J, _ = self._cost(U, horizon, seed=(self.seed + it * 100 + c))
                costs.append(J)
                if J < best_J:
                    best_J = J
                    best_U = U.copy()
            order = np.argsort(costs)[:n_elite]
            elites = samples[order]
            new_mean = elites.mean(axis=0)
            new_std = elites.std(axis=0) + 1e-3
            if np.linalg.norm(new_mean - mean) < 1e-3 and it > 0:
                mean, std = new_mean, new_std
                converged = True
                break
            mean, std = new_mean, np.maximum(new_std, self.u_max * 0.05)

        best_U = np.clip(best_U, -self.u_max, self.u_max)

        # 用同一随机种子评估基线 vs 最优，保证可比
        base_fut = self.sim.rollout(K=self.K, horizon=horizon, policy=None, seed=self.seed)
        ctrl_fut = self.sim.rollout(
            K=self.K, horizon=horizon,
            policy=lambda t, Z: best_U[min(t, len(best_U) - 1)], seed=self.seed
        )
        energy = float(np.sum(best_U ** 2))
        mode_dose, target = self._latent_to_mode_dose(best_U.mean(axis=0))
        reduction = base_fut.crisis_rate - ctrl_fut.crisis_rate
        narr = self._narrative(target, mode_dose, base_fut.crisis_rate,
                               ctrl_fut.crisis_rate, reduction)
        return InterventionPlan(
            horizon=horizon,
            control_latent=best_U.tolist(),
            mode_dose=mode_dose, target_dim=target,
            baseline_crisis_rate=base_fut.crisis_rate,
            controlled_crisis_rate=ctrl_fut.crisis_rate,
            baseline_ci=base_fut.crisis_ci, controlled_ci=ctrl_fut.crisis_ci,
            reduction=reduction, energy_cost=energy, lam=self.lam,
            iterations=self.n_iters, converged=converged, narrative=narr,
        )

    def _latent_to_mode_dose(self, u_mean: np.ndarray) -> tuple[dict, Optional[str]]:
        """平均潜控制 -> 各模态剂量（带符号，+ = 提升该模态值）"""
        man = self.sim.manifold
        dmode = man.latent_to_mode_displacement(u_mean)     # (n_modes,)
        dose = {nm: float(dmode[j]) for j, nm in enumerate(man.dim_names_)}
        target = max(dose.items(), key=lambda kv: abs(kv[1]))[0] if dose else None
        return dose, target

    def _narrative(self, target, dose, base, ctrl, reduction) -> str:
        if target is None:
            return "无可用干预靶点。"
        d = dose.get(target, 0.0)
        direction = "提升" if d >= 0 else "降低"
        if reduction <= 0:
            return (f"MPC 未找到显著降低危机率的干预（基线 {base:.2%} → "
                    f"干预后 {ctrl:.2%}）；建议维持观察，本区间干预不占优。")
        return (f"最优救援：在预测时域内{direction}「{target}」约 {abs(d):.2f}（模态单位/步），"
                f"预期危机率 {base:.1%} → {ctrl:.1%}（下降 {reduction:.1%}，"
                f"经验估计附置信区间）。")


# ============================================================
# 最小能量救援轨迹（逆向 instanton）
# ============================================================

def minimum_rescue_path(model: LandscapeModel, dt: float = 1.0,
                        max_steps: int = 60) -> Optional[RescuePath]:
    """求把状态从危机阱推回健康阱、并重新抬高壁垒的最省力干预

    崩溃 instanton（health->saddle->crisis）反向即最小能量救援轨迹。沿反向
    路径逐点计算抵消自然漂移所需的控制 u = (z_{t+1}-z_t)/dt - f(z_t)，
    反投影为模态剂量方向。
    """
    pot, drift, man = model.pot, model.drift, model.manifold
    health = model.health_fp.z
    crisis = model.crisis_fp.z
    # 选择分隔危机与健康阱的鞍点
    if model.saddles:
        mid = 0.5 * (health + crisis)
        saddle = min(model.saddles, key=lambda s: np.linalg.norm(s.z - mid)).z
    else:
        saddle = mid
    # 正向 instanton（health->crisis），再反向得救援轨迹（crisis->health）
    fwd = minimum_energy_path(pot, health, saddle, crisis)
    path = fwd[::-1].copy()
    if len(path) > max_steps:
        idx = np.linspace(0, len(path) - 1, max_steps).astype(int)
        path = path[idx]

    # 逐点所需控制：抵消漂移 + 提供路径前进速度
    control = []
    for i in range(len(path) - 1):
        z = path[i]
        z_next = path[i + 1]
        f = drift.f(z)
        u = (z_next - z) / dt - f
        control.append(u)
    if not control:
        return None
    control_arr = np.array(control)
    energy = float(np.sum(control_arr ** 2))

    # 峰值救援剂量：控制幅度最大处的模态投影（决定"该推哪个模态"）
    norms = np.linalg.norm(control_arr, axis=1)
    peak = int(np.argmax(norms))
    dmode = man.latent_to_mode_displacement(control_arr[peak])
    # 对全程控制求平均，给出稳健的剂量方向
    dmode_mean = man.latent_to_mode_displacement(control_arr.mean(axis=0))
    dose = {nm: float(dmode_mean[j]) for j, nm in enumerate(man.dim_names_)}
    target = max(dose.items(), key=lambda kv: abs(kv[1]))[0] if dose else None

    # 健康阱壁垒（救援后重新抬高的势垒，来自模型闭式）
    from .landscape_nd import multidim_escape_rate
    try:
        _, barrier, _ = multidim_escape_rate(pot, drift, health, saddle, drift.diffusion_)
    except Exception:
        barrier = 0.0

    narr = _rescue_narrative(target, dose, energy, barrier, len(path) - 1)
    return RescuePath(
        latent_path=path.tolist(),
        control_latent=control_arr.tolist(),
        mode_dose=dose, target_dim=target,
        energy=energy, barrier_restored=float(barrier),
        n_steps=len(path) - 1, narrative=narr,
    )


def _rescue_narrative(target, dose, energy, barrier, steps) -> str:
    if target is None:
        return "无可用救援靶点。"
    d = dose.get(target, 0.0)
    direction = "提升" if d >= 0 else "降低"
    return (f"最小能量救援：沿逆向 instanton（{steps} 步）主要{direction}「{target}」"
            f"约 {abs(d):.2f}/步，总干预能量 {energy:.3f}，救援后健康阱壁垒恢复至 "
            f"ΔU≈{barrier:.4f}（闭式，鞍点邻域近似）。")
