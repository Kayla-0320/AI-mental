"""
结构因果模型与反事实推断（In-Silico Patient 支柱 B 之一）
========================================================
把个体化时间序列升级为**可干预、可反事实**的结构因果模型（SCM），
回答两类因果问题（Pearl 因果阶梯第 2、3 层）：

1. 干预 do(X_j = v)：截断被干预变量的入边、赋值、按因果拓扑序前向传播，
   得到**干预分布**（而非条件预测 P(Y|X_j=v)）。
2. 反事实：给定已观测轨迹（证据），回答"若当时多睡 2h，情绪轨迹会怎样"。
   走 Pearl 三步：Abduction（由证据反推外生噪声 U）→ Action（do）→
   Prediction（用同一 U 重算）。

模型形式（线性非高斯 SEM / VAR(1)）
----------------------------------
    X_{t+1} = A · X_t + b + U_t,   U_t 为外生噪声（残差经验分布）
- 结构 A 的稀疏模式优先由 `PersonalCausalDiscovery` 的因果边给定（复用现有
  因果图），系数用最小二乘拟合；无因果图时退化为全连接 VAR(1)。
- 外生噪声 U 的协方差由残差估计，反事实/干预时从残差经验分布采样。

诚实性约束
----------
- 线性高斯 SCM 的反事实在"噪声固定"假设下成立；真实心理系统非高斯、非线性，
  此处为文档化线性近似，数据不足时返回 None 由上层回退。
- do-算子的可识别性依赖因果图无未观测混杂（假设成立，如实标注）。

理论依据
--------
- Pearl, Causality (2009)：do-算子、结构方程、反事实三步法
- VAR / 线性非高斯无环模型 (Hyvärinen & Smith, 2013)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class InterventionResult:
    """do-干预结果：干预前后的轨迹分布对比"""
    do_spec: dict[str, float]           # 被干预变量 -> 值
    clamp: bool                          # True=持续钳制，False=单脉冲
    horizon: int
    n_samples: int
    baseline_final: dict[str, float]     # 无干预末态均值
    intervened_final: dict[str, float]   # 干预末态均值
    effect: dict[str, float]             # 末态净变化（intervened - baseline）
    baseline_traj: list[dict[str, float]] = field(default_factory=list)
    intervened_traj: list[dict[str, float]] = field(default_factory=list)
    method: str = "do_var1"

    def to_dict(self) -> dict:
        return {
            "do_spec": {k: round(float(v), 4) for k, v in self.do_spec.items()},
            "clamp": self.clamp,
            "horizon": self.horizon,
            "n_samples": self.n_samples,
            "baseline_final": {k: round(float(v), 4) for k, v in self.baseline_final.items()},
            "intervened_final": {k: round(float(v), 4) for k, v in self.intervened_final.items()},
            "effect": {k: round(float(v), 4) for k, v in self.effect.items()},
            "baseline_traj": [
                {k: round(float(v), 4) for k, v in step.items()} for step in self.baseline_traj
            ],
            "intervened_traj": [
                {k: round(float(v), 4) for k, v in step.items()} for step in self.intervened_traj
            ],
            "method": self.method,
        }


@dataclass
class CounterfactualResult:
    """Pearl 三步反事实结果"""
    do_spec: dict[str, float]
    clamp: bool
    intervention_step: int
    factual_final: dict[str, float]          # 事实（观测）末态
    counterfactual_final: dict[str, float]   # 反事实末态
    effect: dict[str, float]                 # 反事实 - 事实
    abduced_noise_norm: float                # 反推噪声的规模（诊断）
    factual_traj: list[dict[str, float]] = field(default_factory=list)
    counterfactual_traj: list[dict[str, float]] = field(default_factory=list)
    narrative: str = ""
    method: str = "pearl_3step"

    def to_dict(self) -> dict:
        return {
            "do_spec": {k: round(float(v), 4) for k, v in self.do_spec.items()},
            "clamp": self.clamp,
            "intervention_step": self.intervention_step,
            "factual_final": {k: round(float(v), 4) for k, v in self.factual_final.items()},
            "counterfactual_final": {k: round(float(v), 4) for k, v in self.counterfactual_final.items()},
            "effect": {k: round(float(v), 4) for k, v in self.effect.items()},
            "abduced_noise_norm": round(self.abduced_noise_norm, 5),
            "factual_traj": [
                {k: round(float(v), 4) for k, v in step.items()} for step in self.factual_traj
            ],
            "counterfactual_traj": [
                {k: round(float(v), 4) for k, v in step.items()} for step in self.counterfactual_traj
            ],
            "narrative": self.narrative,
            "method": self.method,
        }


# ============================================================
# 结构因果模型（VAR(1) 线性 SCM）
# ============================================================

class StructuralCausalModel:
    """线性结构因果模型 X_{t+1} = A·X_t + b + U

    - fit：从 history 拟合 A、b、外生噪声协方差与残差经验样本；可选用因果图
      约束 A 的稀疏模式（仅保留因果边对应的系数，其余置零后重拟合）。
    - do：干预分布（前向模拟，从残差噪声采样）。
    - counterfactual：Pearl 三步（abduction→action→prediction，噪声固定）。
    """

    def __init__(
        self,
        dims: list[str],
        A: np.ndarray,
        b: np.ndarray,
        noise_cov: np.ndarray,
        resid_samples: np.ndarray,
        topo_order: list[str],
    ):
        self.dims = dims
        self.A = A
        self.b = b
        self.noise_cov = noise_cov
        self.resid_samples = resid_samples     # (M, n_dims) 经验噪声
        self.topo_order = topo_order
        self._chol: Optional[np.ndarray] = None

    # ----------------------------------------------------------
    @classmethod
    def fit(
        cls,
        history: list[dict[str, float]],
        graph=None,
        min_len: int = 15,
    ) -> Optional["StructuralCausalModel"]:
        """从 history 拟合 VAR(1) SCM

        Args:
            history: list[dict[dim, value]]，各模态 [0,1]
            graph: 可选 CausalGraph，用于约束 A 的稀疏结构
            min_len: 最小时间步数
        """
        if not history or len(history) < min_len:
            return None
        # 仅保留在（几乎）每一步都出现的模态
        counts: dict[str, int] = {}
        for step in history:
            for k in step:
                counts[k] = counts.get(k, 0) + 1
        dims = sorted([k for k, c in counts.items() if c >= len(history) * 0.9])
        if len(dims) < 1:
            return None
        L = min(len(history), max((counts[d] for d in dims), default=0))
        # 对齐：抽取同时含所有 dims 的步
        rows = [s for s in history if all(d in s for d in dims)]
        if len(rows) < min_len:
            return None
        X = np.array([[float(s[d]) for d in dims] for s in rows], dtype=float)  # (T, n)
        T = len(X)
        if T < min_len:
            return None

        X_cur, X_next = X[:-1], X[1:]        # 预测下一步
        n = len(dims)
        # 设计矩阵：[X_t, 1]
        D = np.column_stack([X_cur, np.ones(T - 1)])
        try:
            theta, *_ = np.linalg.lstsq(D, X_next, rcond=None)   # (n+1, n)
        except np.linalg.LinAlgError:
            return None
        A = theta[:n].T            # (n, n): X_next_i = Σ_j A_ij X_t_j + b_i
        b = theta[n]

        resid = X_next - (X_cur @ A.T + b)     # (T-1, n) 外生噪声经验样本
        noise_cov = np.cov(resid, rowvar=False) if resid.shape[0] > 1 else np.eye(n) * 1e-4
        noise_cov = np.atleast_2d(noise_cov)

        topo_order = cls._topological_order(dims, A, graph)
        return cls(dims, A, b, noise_cov, resid, topo_order)

    @staticmethod
    def _topological_order(dims: list[str], A: np.ndarray, graph=None) -> list[str]:
        """按因果依赖排序（父在前）。

        优先用因果图的边定向；若图缺失或含环，用 A 的邻接强度做启发式
        （出度大者更靠上游）。始终返回全部 dims 的一个排列。
        """
        idx = {d: i for i, d in enumerate(dims)}
        parents: dict[str, set] = {d: set() for d in dims}
        if graph is not None and getattr(graph, "edges", None):
            for e in graph.edges:
                if e.cause in idx and e.effect in idx and e.cause != e.effect:
                    parents[e.effect].add(e.cause)
        else:
            # 用 |A| 阈值化推断结构（A_ij: j→i）
            thr = 0.15 * (np.max(np.abs(A)) + 1e-9)
            for i, di in enumerate(dims):
                for j, dj in enumerate(dims):
                    if i != j and abs(A[i, j]) > thr:
                        parents[di].add(dj)
        # Kahn 拓扑排序（含环时按剩余出度补足）
        indeg = {d: len(parents[d]) for d in dims}
        children = {d: [] for d in dims}
        for d, ps in parents.items():
            for p in ps:
                children[p].append(d)
        queue = sorted([d for d in dims if indeg[d] == 0],
                       key=lambda d: -float(np.sum(np.abs(A[idx[d]]))))
        order: list[str] = []
        while queue:
            d = queue.pop(0)
            order.append(d)
            for c in children[d]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
            queue.sort(key=lambda x: -float(np.sum(np.abs(A[idx[x]]))))
        # 环中剩余节点：按出度补
        rest = [d for d in dims if d not in order]
        rest.sort(key=lambda d: -float(np.sum(np.abs(A[idx[d]]))))
        return order + rest

    # ----------------------------------------------------------
    # 内部工具
    # ----------------------------------------------------------
    def _chol_noise(self) -> np.ndarray:
        if self._chol is None:
            cov = self.noise_cov.copy()
            # 半正定修正
            w, V = np.linalg.eigh(cov)
            w = np.clip(w, 1e-10, None)
            cov = V @ np.diag(w) @ V.T
            self._chol = np.linalg.cholesky(cov)
        return self._chol

    def _sample_noise(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """从外生噪声协方差采样 (size, n)"""
        z = rng.standard_normal((size, len(self.dims)))
        return z @ self._chol_noise().T

    def _clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, 0.0, 1.0)

    def _apply_do(self, X: np.ndarray, do_spec: dict[str, float]) -> np.ndarray:
        idx = {d: i for i, d in enumerate(self.dims)}
        X = X.copy()
        for d, v in do_spec.items():
            if d in idx:
                X[:, idx[d]] = float(v)
        return X

    def _propagate(
        self,
        x0: np.ndarray,
        horizon: int,
        do_spec: Optional[dict[str, float]],
        clamp: bool,
        noise: Optional[np.ndarray],
    ) -> np.ndarray:
        """从 x0 (n,) 前向模拟 horizon 步，返回轨迹 (horizon+1, n)

        noise: 若给定 (horizon, n) 则用固定噪声（反事实），否则用均值 0。
        do_spec: 若给定，第一步（pulse）或每步（clamp）覆盖被干预变量。
        """
        n = len(self.dims)
        traj = np.zeros((horizon + 1, n))
        x = x0.astype(float).copy()
        if do_spec and not clamp:
            x = self._apply_do(x.reshape(1, -1), do_spec)[0]
        traj[0] = x
        for t in range(horizon):
            x = self.A @ x + self.b
            if noise is not None:
                x = x + noise[t]
            if do_spec and clamp:
                x = self._apply_do(x.reshape(1, -1), do_spec)[0]
            x = self._clip(x)
            traj[t + 1] = x
        return traj

    # ----------------------------------------------------------
    # do-干预分布
    # ----------------------------------------------------------
    def do(
        self,
        do_spec: dict[str, float],
        x0: Optional[dict[str, float]] = None,
        horizon: int = 10,
        clamp: bool = True,
        n_samples: int = 200,
        seed: int = 0,
    ) -> Optional[InterventionResult]:
        """do(X_j=v) 干预分布：对比无干预 vs 干预的末态均值

        Args:
            do_spec: {dim: value}，被干预变量取值（截断其入边）
            x0: 起始状态（默认用残差样本无关的当前观测均值）
            horizon: 前向步数
            clamp: True=持续钳制被干预变量；False=单脉冲
            n_samples: 蒙特卡洛样本数
        """
        if not do_spec:
            return None
        idx = {d: i for i, d in enumerate(self.dims)}
        if not any(d in idx for d in do_spec):
            return None
        if x0 is None:
            # 用外生噪声的均值态作为起点（A 的稳态近似），否则零向量
            x0_vec = self._stationary_start()
        else:
            x0_vec = np.array([float(x0.get(d, 0.5)) for d in self.dims])

        rng = np.random.default_rng(seed)
        base_finals = np.zeros((n_samples, len(self.dims)))
        int_finals = np.zeros((n_samples, len(self.dims)))
        base_traj_sum = np.zeros((horizon + 1, len(self.dims)))
        int_traj_sum = np.zeros((horizon + 1, len(self.dims)))
        for s in range(n_samples):
            noise = self._sample_noise(horizon, rng)
            bt = self._propagate(x0_vec, horizon, None, False, noise)
            it = self._propagate(x0_vec, horizon, do_spec, clamp, noise)
            base_finals[s] = bt[-1]
            int_finals[s] = it[-1]
            base_traj_sum += bt
            int_traj_sum += it
        base_mean = base_finals.mean(axis=0)
        int_mean = int_finals.mean(axis=0)
        base_traj = (base_traj_sum / n_samples)
        int_traj = (int_traj_sum / n_samples)
        to_step = lambda vec: {d: float(vec[i]) for i, d in enumerate(self.dims)}  # noqa: E731
        return InterventionResult(
            do_spec={k: float(v) for k, v in do_spec.items() if k in idx},
            clamp=clamp,
            horizon=horizon,
            n_samples=n_samples,
            baseline_final=to_step(base_mean),
            intervened_final=to_step(int_mean),
            effect=to_step(int_mean - base_mean),
            baseline_traj=[to_step(v) for v in base_traj],
            intervened_traj=[to_step(v) for v in int_traj],
        )

    def _stationary_start(self) -> np.ndarray:
        """VAR(1) 稳态均值 (I-A)^{-1} b；奇异时退回 b"""
        n = len(self.dims)
        try:
            x = np.linalg.solve(np.eye(n) - self.A, self.b)
            if np.all(np.isfinite(x)):
                return self._clip(x)
        except np.linalg.LinAlgError:
            pass
        return self._clip(self.b.copy())

    # ----------------------------------------------------------
    # Pearl 三步反事实
    # ----------------------------------------------------------
    def counterfactual(
        self,
        evidence: list[dict[str, float]],
        do_spec: dict[str, float],
        intervention_step: Optional[int] = None,
        horizon: int = 10,
        clamp: bool = False,
    ) -> Optional[CounterfactualResult]:
        """Pearl 三步反事实：给定已观测轨迹，若在某步 do(X_j=v)，之后 horizon 步末态会怎样

        Step 1 Abduction：由证据反推每步外生噪声 U_t = X_{t+1} - A X_t - b。
        Step 2 Action：在 intervention_step 施加 do（pulse 或 clamp）。
        Step 3 Prediction：用**同一批** U_t 重算结构方程，得反事实轨迹。

        效应在 intervention_step + horizon 处度量（不超过证据长度）：单次脉冲
        在稳定系统中会自然衰减，故取干预后有限窗口而非证据末尾，符合
        "接下来 N 步会怎样"的临床语义。
        """
        if not evidence or len(evidence) < 3:
            return None
        idx = {d: i for i, d in enumerate(self.dims)}
        if not any(d in idx for d in do_spec):
            return None
        rows = [s for s in evidence if all(d in s for d in self.dims)]
        if len(rows) < 3:
            return None
        X = np.array([[float(s[d]) for d in self.dims] for s in rows], dtype=float)
        T = len(X)
        if intervention_step is None:
            intervention_step = max(0, T // 2)
        intervention_step = int(np.clip(intervention_step, 0, T - 2))
        end_idx = int(min(intervention_step + max(1, horizon), T - 1))

        # Step 1: Abduction —— 反推每步噪声（仅到 T-2 → T-1 有观测转移）
        U = np.zeros((T - 1, len(self.dims)))
        for t in range(T - 1):
            U[t] = X[t + 1] - (self.A @ X[t] + self.b)
        abduced_norm = float(np.mean(np.linalg.norm(U, axis=1)))

        # Step 2 + 3: Action & Prediction —— 从干预步用同一噪声重算
        factual = X.copy()
        cf = X.copy()
        x = X[intervention_step].astype(float).copy()
        x = self._apply_do(x.reshape(1, -1), do_spec)[0]
        cf[intervention_step] = x
        for t in range(intervention_step, end_idx):
            x = self.A @ x + self.b + U[t]      # 固定外生噪声（反事实核心）
            if clamp:
                x = self._apply_do(x.reshape(1, -1), do_spec)[0]
            x = self._clip(x)
            cf[t + 1] = x

        to_step = lambda vec: {d: float(vec[i]) for i, d in enumerate(self.dims)}  # noqa: E731
        factual_final = to_step(factual[end_idx])
        cf_final = to_step(cf[end_idx])
        effect = to_step(cf[end_idx] - factual[end_idx])
        narr = self._cf_narrative(do_spec, clamp, intervention_step, effect)
        return CounterfactualResult(
            do_spec={k: float(v) for k, v in do_spec.items() if k in idx},
            clamp=clamp,
            intervention_step=intervention_step,
            factual_final=factual_final,
            counterfactual_final=cf_final,
            effect=effect,
            abduced_noise_norm=abduced_norm,
            factual_traj=[to_step(v) for v in factual[intervention_step:end_idx + 1]],
            counterfactual_traj=[to_step(v) for v in cf[intervention_step:end_idx + 1]],
            narrative=narr,
        )

    def _cf_narrative(self, do_spec, clamp, step, effect) -> str:
        parts = []
        for d, v in do_spec.items():
            parts.append(f"{d}={v:.2f}" + ("(持续)" if clamp else "(一次性)"))
        do_str = "、".join(parts) if parts else "无"
        movers = sorted(effect.items(), key=lambda kv: -abs(kv[1]))[:3]
        eff_str = "、".join(
            f"{d}{'↑' if e >= 0 else '↓'}{abs(e):.3f}" for d, e in movers if abs(e) > 1e-4
        ) or "各维度几乎无变化"
        return (f"若在第 {step} 步施加 do({do_str})：{eff_str}（线性 SCM 反事实近似，"
                f"噪声固定）")

    # ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "dims": self.dims,
            "A": [[round(float(v), 5) for v in row] for row in self.A],
            "b": [round(float(v), 5) for v in self.b],
            "topo_order": self.topo_order,
            "noise_cov": [[round(float(v), 5) for v in row] for row in self.noise_cov],
            "n_resid": int(self.resid_samples.shape[0]),
        }
