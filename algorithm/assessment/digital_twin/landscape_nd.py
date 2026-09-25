"""
多维临界跃迁引擎（CT-PLEW 的 N 维扩展）
========================================
把逐维 1D 的势能景观升级为**多维耦合**的个体化随机动力系统分析：

    dX = f(X) dt + sqrt(2 D) dW,   X ∈ R^k（潜在风险流形）

科学流程
--------
1. LatentRiskManifold：11 模态 → 2~3 维潜在风险流形（PCA，保留 >=85% 方差），
   并保存模态<->潜维映射，用于把潜空间崩溃路径反投影为"哪个模态先崩"。
2. DriftFieldEstimator：多项式 Ridge 回归非参估计漂移场 f(Z) 与扩散张量 D。
3. gradient_flux_decomposition：最小二乘 Helmholtz 分解 f = -∇U + J，
   U 为拟势（非平衡系统的自由能景观），J 为旋转概率通量。
4. find_fixed_points：f(Z)=0 的零点 + Jacobian 特征值分类（吸引子/鞍点）。
5. minimum_energy_path：简化 string method 松弛 health→saddle→crisis，得 instanton
   （最小作用量崩溃路径），反投影为模态 domino 顺序。
6. multidim_escape_rate：Langer / 多维 Kramers 闭式逃逸率
       λ = (Λ₊/2π)·sqrt(det H_s / |det H_b|)·exp(-ΔU / D_eff)
   N=1 时精确退化为经典 Kramers 公式（单元测试保证）。
7. particle_ensemble：Fleming-Viot 粒子群经验估计逃逸概率与置信区间，
   交叉验证闭式解。

理论依据
--------
- Freidlin-Wentzell 大偏差与拟势 (Freidlin & Wentzell, 1998)
- Helmholtz 梯度-通量分解 / 非平衡景观 (Qian, 2006; Wang et al., 2011)
- Langer 多维逃逸率 (Langer, 1969)；Kramers (1940)
- String method / 最速下降即时子 (E, Ren, Vanden-Eijnden, 2002)

诚实性约束
----------
- 潜空间降维是文档化近似；数据不足或无清晰双稳态时返回 None，
  由上层回退到 1D CT-PLEW。
- 非平衡系统的 U 为**拟势**而非严格势函数，J≠0 时闭式逃逸率仅在鞍点邻域近似。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import optimize
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class FixedPoint:
    """潜空间不动点"""
    z: np.ndarray
    kind: str                    # attractor / saddle / unstable
    eigenvalues: np.ndarray      # Jacobian 特征值

    def to_dict(self) -> dict:
        return {
            "z": [round(float(v), 4) for v in self.z],
            "kind": self.kind,
            "eigenvalues": [
                [round(float(e.real), 4), round(float(e.imag), 4)]
                for e in self.eigenvalues
            ],
        }


@dataclass
class LandscapeND:
    """多维势能景观分析结果"""
    n_dim: int                            # 潜空间维度 k
    dim_names: list[str]                  # 参与分析的模态名
    latent_var_explained: float           # PCA 累计解释方差
    fixed_points: list[FixedPoint] = field(default_factory=list)
    health_attractor: Optional[np.ndarray] = None
    crisis_attractor: Optional[np.ndarray] = None
    current_attractor: Optional[np.ndarray] = None   # 当前所处吸引子
    in_crisis: bool = False                          # 当前是否已在危机阱
    saddle: Optional[np.ndarray] = None
    barrier_height: float = 0.0           # ΔU = U(saddle) - U(health)
    diffusion_eff: float = 0.0            # 不稳定方向有效扩散
    escape_rate: float = 0.0              # 多维 Kramers/Langer λ
    crisis_probability: float = 0.0       # 1 - exp(-λT)
    gradient_flux_ratio: float = 1.0      # ||∇U|| / (||∇U||+||J||)
    collapse_path: list[list[float]] = field(default_factory=list)  # 潜空间路径
    collapse_order: list[str] = field(default_factory=list)         # 模态 domino 顺序
    particle_prob: Optional[float] = None       # 粒子群经验概率
    particle_ci: Optional[tuple] = None         # (lo, hi) Wilson 区间
    coupling: Optional[dict] = None             # 模态耦合矩阵序列化
    method: str = "multidim"

    def to_dict(self, downsample: int = 24) -> dict:
        step = max(1, len(self.collapse_path) // downsample)
        return {
            "n_dim": self.n_dim,
            "dim_names": self.dim_names,
            "latent_var_explained": round(self.latent_var_explained, 4),
            "fixed_points": [fp.to_dict() for fp in self.fixed_points],
            "health_attractor": (
                [round(float(v), 4) for v in self.health_attractor]
                if self.health_attractor is not None else None
            ),
            "crisis_attractor": (
                [round(float(v), 4) for v in self.crisis_attractor]
                if self.crisis_attractor is not None else None
            ),
            "current_attractor": (
                [round(float(v), 4) for v in self.current_attractor]
                if self.current_attractor is not None else None
            ),
            "in_crisis": self.in_crisis,
            "saddle": (
                [round(float(v), 4) for v in self.saddle]
                if self.saddle is not None else None
            ),
            "barrier_height": round(self.barrier_height, 5),
            "diffusion_eff": round(self.diffusion_eff, 6),
            "escape_rate": round(self.escape_rate, 6),
            "crisis_probability": round(self.crisis_probability, 4),
            "gradient_flux_ratio": round(self.gradient_flux_ratio, 4),
            "collapse_path": [
                [round(float(v), 4) for v in node]
                for node in self.collapse_path[::step]
            ],
            "collapse_order": self.collapse_order,
            "particle_prob": (
                round(self.particle_prob, 4) if self.particle_prob is not None else None
            ),
            "particle_ci": (
                [round(self.particle_ci[0], 4), round(self.particle_ci[1], 4)]
                if self.particle_ci is not None else None
            ),
            "coupling": self.coupling,
            "method": self.method,
        }


# ============================================================
# 1. 潜在风险流形（降维）
# ============================================================

class LatentRiskManifold:
    """11 模态 → k 维潜在风险流形（PCA）

    单用户历史稀疏（<=200 步），无法直接非参估计高维景观。
    先标准化再 PCA，保留累计解释方差 >= target_var 的前 k 维（clamp 到 [1,3]）。
    保存 components_ 用于把潜空间位移反投影回各模态，得到崩溃 domino 顺序。
    """

    def __init__(self, target_var: float = 0.85, max_dim: int = 3):
        self.target_var = target_var
        self.max_dim = max_dim
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.center_: Optional[np.ndarray] = None     # 训练期潜空间中心（≈ 0）
        self.components_: Optional[np.ndarray] = None   # (k, n_modes)
        self.dim_names_: list[str] = []
        self.explained_: float = 0.0
        self.k_: int = 0
        self.min_len_: int = 0

    def fit(self, data: dict[str, list[float]], min_len: int = 20,
            causal_weights: Optional[dict[str, float]] = None,
            confidence_weights: Optional[dict[str, float]] = None) -> Optional["LatentRiskManifold"]:
        """拟合潜流形
            
        Args:
            data: {模态名：时间序列}
            min_len: 最小历史长度
            causal_weights: 可选，{模态名：因果出度权重}，根因模态优先
            confidence_weights: 可选，{模态名：感知置信度}，低置信模态降权
        """
        # 只保留长度足够的模态，且各模态等长对齐到最短
        series = {k: np.asarray(v, dtype=float) for k, v in data.items() if len(v) >= min_len}
        if len(series) < 2:
            return None
        L = min(len(v) for v in series.values())
        if L < min_len:
            return None
        self.dim_names_ = sorted(series.keys())
        X = np.column_stack([series[d][:L] for d in self.dim_names_])   # (L, n_modes)
    
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        Xs = (X - self.mean_) / self.std_
    
        # 统一加权：因果权重 × 置信度权重（缺失则默认 1.0）
        combined_weights = np.ones(len(self.dim_names_))
        if causal_weights:
            cw = np.array([causal_weights.get(d, 1.0) for d in self.dim_names_])
            cw = np.clip(cw, 0.1, 10.0)
            combined_weights *= cw / cw.mean()
        if confidence_weights:
            cw2 = np.array([confidence_weights.get(d, 1.0) for d in self.dim_names_])
            cw2 = np.clip(cw2, 0.1, 10.0)
            combined_weights *= cw2 / cw2.mean()
        if not (causal_weights or confidence_weights):
            combined_weights = None
            
        if combined_weights is not None:
            Xs = Xs * np.sqrt(combined_weights)  # 加权 PCA 等价于特征缩放后标准 PCA
    
        # PCA via SVD（保存训练期中心，transform 时复用，不逐批重定心）
        self.center_ = Xs.mean(axis=0)
        U, S, Vt = np.linalg.svd(Xs - self.center_, full_matrices=False)
        var_ratio = (S ** 2) / np.sum(S ** 2 + 1e-12)
        cum = np.cumsum(var_ratio)
        k = int(np.searchsorted(cum, self.target_var) + 1)
        k = int(np.clip(k, 1, min(self.max_dim, len(self.dim_names_))))
        self.k_ = k
        self.components_ = Vt[:k]                       # (k, n_modes)
        self.explained_ = float(cum[k - 1])
        self.min_len_ = L
        self.combined_weights_ = combined_weights  # 保存用于 transform 时复用
        return self

    def transform(self, data: dict[str, list[float]]) -> Optional[np.ndarray]:
        """history dict -> 潜变量矩阵 Z (L, k)；使用训练期 mean/std/center"""
        if self.components_ is None:
            return None
        present = [d for d in self.dim_names_ if d in data and len(data[d]) > 0]
        if len(present) != len(self.dim_names_):
            return None
        L = min(len(data[d]) for d in self.dim_names_)
        if L < 2:
            return None
        X = np.column_stack([np.asarray(data[d][:L], dtype=float) for d in self.dim_names_])
        Xs = (X - self.mean_) / self.std_
        # 复用训练期统一权重（因果 × 置信度）
        if getattr(self, 'combined_weights_', None) is not None:
            Xs = Xs * np.sqrt(self.combined_weights_)
        Z = (Xs - self.center_) @ self.components_.T   # (L, k)
        return Z

    def transform_point(self, values: dict[str, float]) -> Optional[np.ndarray]:
        """单个模态值 dict -> 潜坐标 (k,)（用于定位高风险角）"""
        if self.components_ is None:
            return None
        try:
            x = np.array([float(values[d]) for d in self.dim_names_])
        except KeyError:
            return None
        xs = (x - self.mean_) / self.std_
        if getattr(self, 'combined_weights_', None) is not None:
            xs = xs * np.sqrt(self.combined_weights_)
        return (xs - self.center_) @ self.components_.T

    def latent_to_mode_displacement(self, dz: np.ndarray) -> np.ndarray:
        """潜空间位移 dz (k,) -> 各模态位移贡献 (n_modes,)"""
        return self.components_.T @ dz

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """潜坐标 Z (..., k) -> 各模态近似值 (..., n_modes)，限在 [0,1]

        PCA 正交行的伪逆重构：Xs ≈ Z @ components_ + center_，X = Xs*std_ + mean_。
        仅保留主成分子空间信息（降维近似的逆映射）。
        """
        if self.components_ is None:
            raise ValueError("manifold not fitted")
        Z = np.asarray(Z, dtype=float)
        Xs = Z @ self.components_ + self.center_
        X = Xs * self.std_ + self.mean_
        return np.clip(X, 0.0, 1.0)

    def collapse_order_from_path(self, path: np.ndarray) -> list[str]:
        """沿潜空间路径（health->crisis）反投影，给出模态上升 domino 顺序

        对路径每一段位移反投影到模态空间，累计各模态的正向（恶化）位移，
        按"首次显著上升"的先后排序。
        """
        if path is None or len(path) < 2:
            return []
        first_rise: dict[str, float] = {}
        cumulative = np.zeros(len(self.dim_names_))
        for i in range(len(path) - 1):
            dz = path[i + 1] - path[i]
            dmode = self.latent_to_mode_displacement(dz)
            cumulative += np.maximum(dmode, 0.0)   # 只累计恶化方向
            for j, name in enumerate(self.dim_names_):
                if name not in first_rise and cumulative[j] > 0.05 * (self.std_[j] + 1e-9):
                    first_rise[name] = float(i)
        # 按首次上升步序排序，未触发的按累计幅度排在后面
        ordered = sorted(first_rise.items(), key=lambda kv: kv[1])
        result = [name for name, _ in ordered]
        remaining = [n for n in self.dim_names_ if n not in result]
        return result + remaining


# ============================================================
# 2. 漂移场 / 扩散张量估计
# ============================================================

class DriftFieldEstimator:
    """多项式 Ridge 回归非参估计 dZ = f(Z)dt + sqrt(2D)dW

    - f：每个潜维用一个 degree 阶多项式 Ridge 拟合 E[ΔZ|Z]。
      degree>=3 才能表达双稳态（f 有 3 个零点）；样本少时降到 2。
    - D：残差协方差的一半，作为常数扩散张量（半正定，特征分解修正）。
    """

    def __init__(self, degree: int = 3, alpha: float = 1.0):
        self.degree = degree
        self.alpha = alpha
        self.poly_: Optional[PolynomialFeatures] = None
        self.models_: list[Ridge] = []
        self.diffusion_: Optional[np.ndarray] = None    # (k, k)
        self.k_: int = 0
        # 训练潜空间域（含外扩），用于把 f 的输入钳制在有效域内
        self.z_lo_: Optional[np.ndarray] = None
        self.z_hi_: Optional[np.ndarray] = None
        self.z_pad_lo_: Optional[np.ndarray] = None
        self.z_pad_hi_: Optional[np.ndarray] = None

    def fit(self, Z: np.ndarray, dt: float = 1.0) -> Optional["DriftFieldEstimator"]:
        T, k = Z.shape
        if T < 30:
            return None
        self.k_ = k
        # 记录训练潜空间范围：拟合的 degree-3 多项式仅在此域邻内可信，
        # SDE 积分/控制输入可能把状态推出该域致多项式外推发散（→ inf）。
        self.z_lo_ = Z.min(axis=0)
        self.z_hi_ = Z.max(axis=0)
        span = np.maximum(self.z_hi_ - self.z_lo_, 1e-6)
        self.z_pad_lo_ = self.z_lo_ - 0.5 * span
        self.z_pad_hi_ = self.z_hi_ + 0.5 * span
        dZ = (Z[1:] - Z[:-1]) / dt
        Zt = Z[:-1]

        # 样本量决定多项式阶数（特征数 = C(degree+k, k)）
        degree = self.degree
        while degree >= 2 and self._n_features(degree, k) > max(20, T // 3):
            degree -= 1
        self.poly_ = PolynomialFeatures(degree=degree, include_bias=False)
        Phi = self.poly_.fit_transform(Zt)

        self.models_ = []
        resid = np.zeros_like(dZ)
        for i in range(k):
            m = Ridge(alpha=self.alpha, fit_intercept=True)
            m.fit(Phi, dZ[:, i])
            self.models_.append(m)
            resid[:, i] = dZ[:, i] - m.predict(Phi)

        # 扩散张量 D = 0.5 * Cov(残差)（去掉已解释的漂移）
        D = 0.5 * np.cov(resid, rowvar=False) if k > 1 else np.array([[0.5 * resid.var()]])
        D = np.atleast_2d(D)
        # 对称化 + 半正定修正
        D = 0.5 * (D + D.T)
        w, V = np.linalg.eigh(D)
        w = np.clip(w, 1e-6, None)
        self.diffusion_ = V @ np.diag(w) @ V.T
        return self

    @staticmethod
    def _n_features(degree: int, k: int) -> int:
        return int(math.comb(degree + k, k)) - 1

    def clip_domain(self, Z: np.ndarray) -> np.ndarray:
        """把潜状态投影钳制到训练域（含 50% 外扩），域内为恒等映射。

        多项式漂移场只在训练数据覆盖的潜空间邻域内有效；一旦 SDE 积分或控制
        输入把状态推出该域，degree-3 外推会急剧发散甚至溢出为 inf。此钳制是
        标准的投影式正则（把越界状态拉回边界），既防溢出又不改变域内动力学。
        """
        Z = np.nan_to_num(np.asarray(Z, dtype=float), nan=0.0,
                          posinf=0.0, neginf=0.0)
        if self.z_pad_lo_ is None or self.z_pad_hi_ is None:
            return Z
        return np.clip(Z, self.z_pad_lo_, self.z_pad_hi_)

    def f(self, Z: np.ndarray) -> np.ndarray:
        """漂移场 f(Z)；Z 可为 (k,) 或 (n,k)。输入先钳制到训练域，避免外推溢出。"""
        single = np.ndim(Z) == 1
        Zin = self.clip_domain(np.atleast_2d(Z))
        Phi = self.poly_.transform(Zin)
        out = np.column_stack([m.predict(Phi) for m in self.models_])
        return out[0] if single else out

    def jacobian(self, z: np.ndarray, h: float = 1e-4) -> np.ndarray:
        """数值 Jacobian J[i,j] = ∂f_i/∂z_j"""
        k = len(z)
        J = np.zeros((k, k))
        for j in range(k):
            zp = z.copy(); zp[j] += h
            zm = z.copy(); zm[j] -= h
            J[:, j] = (self.f(zp) - self.f(zm)) / (2 * h)
        return J


# ============================================================
# 2b. 模态耦合矩阵估计（VAR(1) 显式耦合）
# ============================================================

class ModalityCouplingEstimator:
    """在模态空间拟合 VAR(1): x[t+1] = A @ x[t] + b + noise
    
    输出显式耦合矩阵 A（n_modes × n_modes），其中 A[i,j] 表示
    模态 j 对模态 i 的因果驱动强度。用于解释“睡眠恶化 → 情绪下滑”
    等具体模态间耦合关系。
    """

    def __init__(self):
        self.A_: Optional[np.ndarray] = None  # (n, n) 耦合矩阵
        self.b_: Optional[np.ndarray] = None  # (n,) 截距
        self.dim_names_: list[str] = []
        self.residual_var_: Optional[np.ndarray] = None  # 残差方差

    def fit(self, data: dict[str, list[float]], min_len: int = 20) -> Optional["ModalityCouplingEstimator"]:
        """拟合 VAR(1) 耦合矩阵"""
        series = {k: np.asarray(v, dtype=float) for k, v in data.items() if len(v) >= min_len}
        if len(series) < 2:
            return None
        L = min(len(v) for v in series.values())
        if L < min_len + 1:
            return None
        self.dim_names_ = sorted(series.keys())
        X = np.column_stack([series[d][:L] for d in self.dim_names_])  # (L, n)
        
        # VAR(1): X[t+1] = A @ X[t] + b + noise
        X_past = X[:-1]  # (L-1, n)
        X_future = X[1:]  # (L-1, n)
        
        # 最小二乘：X_future = X_past @ A.T + b
        # 增广矩阵 [X_past, 1] @ [A.T; b.T] = X_future
        ones = np.ones((X_past.shape[0], 1))
        X_aug = np.hstack([X_past, ones])  # (L-1, n+1)
        # 求解 X_aug @ coef = X_future，coef shape = (n+1, n)
        coef, residuals, _, _ = np.linalg.lstsq(X_aug, X_future, rcond=None)
        self.A_ = coef[:-1].T  # (n, n)
        self.b_ = coef[-1]  # (n,)
        
        # 残差方差
        predicted = X_past @ self.A_.T + self.b_
        residuals_actual = X_future - predicted
        self.residual_var_ = residuals_actual.var(axis=0)  # (n,)
        
        return self

    def coupling_strength(self, cause: str, effect: str) -> float:
        """获取 cause → effect 的耦合强度"""
        if self.A_ is None:
            return 0.0
        try:
            i = self.dim_names_.index(effect)
            j = self.dim_names_.index(cause)
            return float(self.A_[i, j])
        except ValueError:
            return 0.0

    def top_couplings(self, k: int = 5) -> list[tuple[str, str, float]]:
        """返回最强的 k 个耦合关系 [(cause, effect, strength), ...]"""
        if self.A_ is None:
            return []
        n = len(self.dim_names_)
        couplings = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    strength = abs(self.A_[i, j])
                    if strength > 1e-4:
                        couplings.append((self.dim_names_[j], self.dim_names_[i], float(self.A_[i, j])))
        couplings.sort(key=lambda x: abs(x[2]), reverse=True)
        return couplings[:k]

    def to_dict(self) -> dict:
        """序列化用于 API 响应"""
        if self.A_ is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "dim_names": self.dim_names_,
            "coupling_matrix": [[round(float(v), 4) for v in row] for row in self.A_],
            "intercept": [round(float(v), 4) for v in self.b_],
            "residual_var": [round(float(v), 6) for v in self.residual_var_],
            "top_couplings": [
                {"cause": c, "effect": e, "strength": round(s, 4)}
                for c, e, s in self.top_couplings(8)
            ],
        }


# ============================================================
# 3. Helmholtz 梯度-通量分解
# ============================================================

class PotentialSolver:
    """在潜空间网格上最小二乘求解拟势 U，使 -∇U ≈ f，残差为通量 J。

    求解 min_U Σ_grid ||∇U + f||²（中心差分，gauge: mean(U)=0）。
    提供 U(z) 与 ∇U(z) 的插值（RegularGridInterpolator + 中心差分）。
    注：未启用显式平滑正则（实测会压平双阱壁垒、恶化逃逸率）；
    smooth_beta 仅作预留接口，默认不生效。
    """

    def __init__(self, n_grid: int = 20, smooth_beta: float = 0.0):
        self.n_grid = n_grid
        self.smooth_beta = smooth_beta
        self.axes_: list[np.ndarray] = []
        self.U_grid_: Optional[np.ndarray] = None
        self._interp: Optional[RegularGridInterpolator] = None
        self.grad_flux_ratio_: float = 1.0

    def solve(self, drift: DriftFieldEstimator, Z: np.ndarray) -> "PotentialSolver":
        k = drift.k_
        # 潜空间包围盒（数据范围 + padding）
        span = np.ptp(Z, axis=0) + 1e-6
        lo = Z.min(axis=0) - 0.15 * span
        hi = Z.max(axis=0) + 0.15 * span
        ng = self.n_grid if k <= 2 else max(10, self.n_grid - 6)
        self.axes_ = [np.linspace(lo[d], hi[d], ng) for d in range(k)]
        spacing = [self.axes_[d][1] - self.axes_[d][0] for d in range(k)]

        mesh = np.meshgrid(*self.axes_, indexing="ij")
        grid_pts = np.column_stack([m.ravel() for m in mesh])   # (M, k)
        M = grid_pts.shape[0]
        f_vals = drift.f(grid_pts)                              # (M, k)

        # 构建梯度算子（中心差分）稀疏系统：对每个内部点每个维度
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        shape = [ng] * k
        idx_grid = np.arange(M).reshape(shape)

        rows, cols, vals, rhs = [], [], [], []
        eq = 0

        def neighbor(index_arr, d, sign):
            shifted = index_arr.copy()
            shifted[..., d] += sign
            return shifted

        for d in range(k):
            # 只在 d 方向有两侧邻居的内部点做中心差分
            sl = [slice(None)] * k
            interior = np.zeros(shape, dtype=bool)
            sl_d = [slice(None)] * k
            sl_d[d] = slice(1, ng - 1)
            interior[tuple(sl_d)] = True
            idx_in = idx_grid[interior]
            ip = neighbor_index(idx_in, shape, d, +1)
            im = neighbor_index(idx_in, shape, d, -1)
            f_in = f_vals[interior.ravel()][:, d]
            n_eq = idx_in.size
            hd = spacing[d]
            # (U[ip] - U[im]) / (2h) = -f_d
            rows.extend(range(eq, eq + n_eq))
            cols.extend(ip.ravel().tolist())
            vals.extend([1.0 / (2 * hd)] * n_eq)
            rows.extend(range(eq, eq + n_eq))
            cols.extend(im.ravel().tolist())
            vals.extend([-1.0 / (2 * hd)] * n_eq)
            rhs.extend((-f_in).ravel().tolist())
            eq += n_eq

        n_eq_grad = eq
        # gauge: mean(U) = 0
        rows.extend([eq] * M)
        cols.extend(list(range(M)))
        vals.extend([1.0 / M] * M)
        rhs.append(0.0)
        eq += 1

        A = sp.csr_matrix((vals, (rows, cols)), shape=(eq, M))
        b = np.array(rhs)
        # 加权：梯度方程权重 1，gauge 权重较大以稳定
        w = np.ones(eq); w[n_eq_grad:] = 10.0
        A = sp.diags(w) @ A
        b = w * b

        sol = spla.lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=5000)[0]
        self.U_grid_ = sol.reshape(shape)
        self._interp = RegularGridInterpolator(
            self.axes_, self.U_grid_, bounds_error=False, fill_value=None
        )

        # 梯度-通量比：在内部网格点上比较 ||∇U|| 与 ||J=f+∇U||
        gradU = self._grad_grid(spacing)
        J = f_vals.reshape(M, k) + gradU
        # 只统计内部点
        norm_grad = float(np.mean(np.linalg.norm(gradU, axis=1)))
        norm_J = float(np.mean(np.linalg.norm(J, axis=1)))
        self.grad_flux_ratio_ = norm_grad / (norm_grad + norm_J + 1e-12)
        return self

    def _grad_grid(self, spacing) -> np.ndarray:
        """网格所有点上的 ∇U（中心差分，边界用单侧）"""
        M = self.U_grid_.size
        k = self.U_grid_.ndim
        grads = np.gradient(self.U_grid_, *spacing)   # list of arrays per dim
        G = np.column_stack([g.ravel() for g in grads])
        return G

    def U(self, z: np.ndarray) -> float:
        return float(self._interp(np.atleast_2d(z))[0])

    def gradU(self, z: np.ndarray, h: float = 1e-3) -> np.ndarray:
        k = len(z)
        g = np.zeros(k)
        for d in range(k):
            zp = z.copy(); zp[d] += h
            zm = z.copy(); zm[d] -= h
            g[d] = (self.U(zp) - self.U(zm)) / (2 * h)
        return g


def neighbor_index(flat_idx: np.ndarray, shape: list, d: int, sign: int) -> np.ndarray:
    """把扁平索引在维度 d 上移动 sign，返回新的扁平索引（flat_idx 为多维子数组）"""
    nd = len(shape)
    # 还原多维坐标
    coords = np.array(np.unravel_index(flat_idx, shape))   # (nd, ...)
    coords[d] = coords[d] + sign
    coords[d] = np.clip(coords[d], 0, shape[d] - 1)
    return np.ravel_multi_index(tuple(coords), shape)


# ============================================================
# 4. 不动点检测与分类
# ============================================================

def find_fixed_points(drift: DriftFieldEstimator, Z: np.ndarray,
                      max_points: int = 12) -> list[FixedPoint]:
    """多起点求 f(z)=0，去重后按 Jacobian 特征值分类"""
    k = drift.k_
    lo = Z.min(axis=0); hi = Z.max(axis=0)
    starts = []
    # 网格上 ||f|| 最小的若干点作为起点
    mesh_pts = _coarse_grid(lo, hi, k, per_dim=8)
    fv = np.linalg.norm(drift.f(mesh_pts), axis=1)
    order = np.argsort(fv)[: max(20, max_points * 3)]
    starts.extend(mesh_pts[order])
    # 数据点采样
    rng = np.random.default_rng(0)
    if len(Z) > 10:
        starts.extend(Z[rng.choice(len(Z), size=min(20, len(Z)), replace=False)])

    found: list[FixedPoint] = []
    for s in starts:
        try:
            res = optimize.root(lambda z: drift.f(z), s, method="hybr", tol=1e-8)
        except Exception:
            continue
        if not res.success:
            continue
        z = res.x
        # 拒绝数据包围盒外的伪不动点（多项式漂移在范围外不可信）
        margin = 0.05 * (hi - lo + 1e-6)
        if np.any(z < lo - margin) or np.any(z > hi + margin):
            continue
        # 去重
        if any(np.linalg.norm(z - fp.z) < 1e-2 * (np.linalg.norm(hi - lo) + 1e-6) for fp in found):
            continue
        J = drift.jacobian(z)
        eig = np.linalg.eigvals(J)
        real_parts = eig.real
        if np.all(real_parts < -1e-6):
            kind = "attractor"
        elif np.sum(real_parts > 1e-6) == 1:
            kind = "saddle"
        else:
            kind = "unstable"
        found.append(FixedPoint(z=z, kind=kind, eigenvalues=eig))
        if len(found) >= max_points:
            break
    return found


def _coarse_grid(lo, hi, k, per_dim) -> np.ndarray:
    axes = [np.linspace(lo[d], hi[d], per_dim) for d in range(k)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh])


# ============================================================
# 5. 最小能量路径（instanton / 最速下降）
# ============================================================

def steepest_descent_path(pot: PotentialSolver, z_start: np.ndarray,
                          z_target: np.ndarray, max_steps: int = 400,
                          step: float = 0.05) -> np.ndarray:
    """从 z_start 沿 -∇U 积分到 z_target 附近（最速下降线）"""
    path = [z_start.copy()]
    z = z_start.copy()
    target_tol = 0.08 * (np.linalg.norm(z_start - z_target) + 1e-6)
    for _ in range(max_steps):
        g = pot.gradU(z)
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-5:
            break
        # 自适应步长
        ds = step / (gnorm + 1e-6)
        ds = min(ds, 0.15)
        z = z - g * ds
        path.append(z.copy())
        if np.linalg.norm(z - z_target) < target_tol:
            break
    return np.array(path)


def _reparameterize(path: np.ndarray) -> np.ndarray:
    """按弧长等距重采样路径（端点固定），string method 的核心步骤。

    重参数化消除梯度步的切向分量、仅保留驱动路径形状的法向分量，
    是简化 string method 收敛到最小能量路径的关键。
    """
    n, k = path.shape
    if n < 3:
        return path.copy()
    diffs = np.diff(path, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-12:
        return path.copy()
    target = np.linspace(0.0, total, n)
    out = np.empty_like(path)
    out[0] = path[0]
    out[-1] = path[-1]
    for d in range(k):
        out[1:-1, d] = np.interp(target[1:-1], cum, path[:, d])
    return out


def minimum_energy_path(pot: PotentialSolver, z_health: np.ndarray,
                        z_saddle: np.ndarray, z_crisis: np.ndarray,
                        n_nodes: int = 40, n_iters: int = 400,
                        dtau: float = 0.2, max_disp: float = 0.2,
                        seed_eps: float = 0.12) -> np.ndarray:
    """最小能量路径（instanton）：简化 string method（E-Ren-Vanden-Eijnden 2002）

    以 health→saddle→crisis 的分段线性曲线为初值，迭代：
      (1) 每个内部节点沿 -∇U 松弛（位移上限 max_disp，防粗网格线性插值过冲）；
      (2) 按弧长重参数化，节点等距、端点固定。
    收敛曲线即最小作用量路径（Freidlin-Wentzell）。相较裸最速下降拼接，
    string method 不依赖鞍点邻域的不定下降方向，在高维稀疏景观上更稳健
    （裸下降在 3D 会被鞍点邻域的伪结构困住而无法抵达两阱）。

    seed_eps 保留以兼容旧签名（string method 无需种子偏移）。
    返回从 health 到 crisis 的有序路径 (n_nodes, k)。
    """
    z_health = np.asarray(z_health, dtype=float).ravel()
    z_saddle = np.asarray(z_saddle, dtype=float).ravel()
    z_crisis = np.asarray(z_crisis, dtype=float).ravel()
    n_nodes = max(8, int(n_nodes))
    half = n_nodes // 2
    seg1 = np.linspace(z_health, z_saddle, half + 1)         # health .. saddle
    seg2 = np.linspace(z_saddle, z_crisis, n_nodes - half)   # saddle .. crisis
    path = np.vstack([seg1, seg2[1:]])

    prev = path.copy()
    for _ in range(max(1, int(n_iters))):
        interior = path[1:-1]
        if len(interior) == 0:
            break
        grads = np.array([pot.gradU(z) for z in interior])
        gnorm = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-12
        step = dtau * grads
        scale = np.minimum(1.0, max_disp / (dtau * gnorm))
        path[1:-1] = interior - step * scale
        path = _reparameterize(path)
        shift = float(np.max(np.linalg.norm(path - prev, axis=1)))
        prev = path.copy()
        if shift < 1e-5:
            break
    return path


# ============================================================
# 6. 多维 Kramers / Langer 逃逸率
# ============================================================

def multidim_escape_rate(pot: PotentialSolver, drift: DriftFieldEstimator,
                         z_health: np.ndarray, z_saddle: np.ndarray,
                         diffusion: np.ndarray) -> tuple[float, float, float]:
    """Langer 多维逃逸率

    λ = (Λ₊/2π)·sqrt(det H_s / |det H_b|)·exp(-ΔU / D_eff)

    H_s/H_b 取不动点处 Hessian(U) ≈ -Jacobian(f)（平滑多项式，见下）；
    ΔU、Λ₊、D_eff 分别来自网格拟势、鞍点 Jacobian 不稳定特征值与该方向噪声强度。
    Returns: (rate, barrier_height ΔU, diffusion_eff D_eff)
    N=1 时精确退化为经典 Kramers：sqrt(|f'(xs)||f'(xb)|)/(2π)·exp(-ΔU/D)。
    """
    U_s = pot.U(z_health)
    U_b = pot.U(z_saddle)
    dU = max(0.0, U_b - U_s)

    # 不动点处 Hessian(U) ≈ -Jacobian(f)（梯度主导系统）。直接用平滑多项式漂移的
    # Jacobian（精确稳定），而非对分段线性的网格拟势做二阶差分——后者以远小于格距
    # 的步长求导会得到近零/噪声的曲率，使前因子 sqrt(det H_s/|det H_b|) 严重失真。
    # 注意 |det(-J)| = |det J|，故直接取 Jacobian 行列式绝对值。
    Js = drift.jacobian(z_health)
    Jb = drift.jacobian(z_saddle)

    det_Hs = float(abs(np.linalg.det(Js)))
    det_Hb = float(abs(np.linalg.det(Jb)))
    det_Hs = det_Hs if det_Hs > 1e-12 else 1e-12
    det_Hb = det_Hb if det_Hb > 1e-12 else 1e-12

    # 鞍点不稳定方向：Jacobian 正实部特征值/向量
    eigvals, eigvecs = np.linalg.eig(Jb)
    real_parts = eigvals.real
    unstable_idx = int(np.argmax(real_parts))
    Lam_plus = float(abs(real_parts[unstable_idx]))
    Lam_plus = max(Lam_plus, 1e-3)
    v_u = np.real(eigvecs[:, unstable_idx])
    v_u = v_u / (np.linalg.norm(v_u) + 1e-12)

    # 有效扩散 = 不稳定方向上的噪声强度 v_u^T D v_u
    D_eff = float(v_u @ diffusion @ v_u)
    D_eff = max(D_eff, 1e-8)

    prefactor = (Lam_plus / (2 * math.pi)) * math.sqrt(det_Hs / det_Hb)
    exponent = -dU / D_eff
    exponent = max(exponent, -50.0)   # 数值下溢保护
    rate = prefactor * math.exp(exponent)
    rate = float(min(max(rate, 0.0), 0.5))   # 单步逃逸率上限（物理约束）
    return rate, float(dU), D_eff


# ============================================================
# 7. Fleming-Viot 粒子群（经验逃逸率 + 置信区间）
# ============================================================

def particle_ensemble(drift: DriftFieldEstimator, diffusion: np.ndarray,
                      z0: np.ndarray, crisis_region_center: np.ndarray,
                      crisis_radius: float, horizon: int = 30,
                      n_particles: int = 300, dt: float = 1.0,
                      seed: int = 0) -> tuple[float, float, float]:
    """从 z0 模拟 n_particles 条轨迹，统计 horizon 步内进入危机区域的比例

    Euler-Maruyama 积分：dz = f(z)dt + sqrt(2 D) dW。
    Returns: (empirical_prob, ci_lo, ci_hi)  Wilson 置信区间。
    """
    rng = np.random.default_rng(seed)
    k = len(z0)
    # sqrt(2D) via Cholesky（对称半正定）
    try:
        Lmat = np.linalg.cholesky(2.0 * diffusion * dt + 1e-9 * np.eye(k))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(diffusion)
        w = np.clip(w, 1e-9, None)
        Lmat = V @ np.diag(np.sqrt(2.0 * w * dt)) @ V.T

    Z = np.tile(z0, (n_particles, 1))
    escaped = np.zeros(n_particles, dtype=bool)
    for _ in range(horizon):
        fz = drift.f(Z)
        noise = rng.standard_normal((n_particles, k)) @ Lmat.T
        Z = drift.clip_domain(Z + fz * dt + noise)   # 钳制到训练域防漂移外推溢出
        d = np.linalg.norm(Z - crisis_region_center, axis=1)
        escaped |= (d < crisis_radius)
    hits = int(escaped.sum())
    p = hits / n_particles
    lo, hi = _wilson_ci(hits, n_particles)
    return p, lo, hi


def _wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = hits / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ============================================================
# 8. 顶层分析器
# ============================================================

@dataclass
class LandscapeModel:
    """已拟合的多维动力系统组件（供 analyze / InSilicoPatient / MPC 共享）

    同一个个体化 f、Σ、拟势 U 与不动点结构，既用于预测（支柱 A），
    也用于生成式模拟与最优控制（支柱 B）——“景观即模拟器”。
    """
    manifold: LatentRiskManifold
    drift: DriftFieldEstimator
    pot: PotentialSolver
    attractors: list
    saddles: list
    health_fp: FixedPoint
    crisis_fp: FixedPoint
    current_fp: FixedPoint
    in_crisis: bool
    z_now: np.ndarray
    crisis_radius: float
    dim_names: list
    coupling: Optional[ModalityCouplingEstimator] = None  # 模态显式耦合矩阵

    @property
    def diffusion(self) -> np.ndarray:
        return self.drift.diffusion_


class MultidimLandscapeAnalyzer:
    """整合：降维 → 漂移场 → 拟势 → 不动点 → instanton → 逃逸率 → 粒子群

    fit_model(data) 返回可复用的 LandscapeModel（或数据不足时 None）；
    analyze(data, horizon) 在其上计算 LandscapeND。
    """

    def __init__(self, min_history: int = 40, min_dims: int = 3,
                 n_particles: int = 250):
        self.min_history = min_history
        self.min_dims = min_dims
        self.n_particles = n_particles

    def fit_model(self, data: dict[str, list[float]],
                  risk_direction: Optional[np.ndarray] = None
                  ) -> Optional[LandscapeModel]:
        """拟合多维动力系统组件；数据不足或无清晰双稳态时返回 None。"""
        usable = {k: v for k, v in data.items() if len(v) >= self.min_history}
        if len(usable) < self.min_dims:
            return None
        L = min(len(v) for v in usable.values())
        if L < self.min_history:
            return None

        manifold = LatentRiskManifold().fit(usable, min_len=self.min_history)
        if manifold is None or manifold.k_ < 1:
            return None
        Z = manifold.transform(usable)
        if Z is None or len(Z) < self.min_history:
            return None

        # 模态耦合矩阵（VAR(1) 显式耦合）
        coupling = ModalityCouplingEstimator().fit(usable, min_len=self.min_history)

        drift = DriftFieldEstimator().fit(Z)
        if drift is None:
            return None

        pot = PotentialSolver().solve(drift, Z)
        fps = find_fixed_points(drift, Z)
        attractors = [fp for fp in fps if fp.kind == "attractor"]
        saddles = [fp for fp in fps if fp.kind == "saddle"]
        if len(attractors) < 2 or len(saddles) < 1:
            # 无清晰双稳态：多维分析不适用，回退
            return None

        # 危机吸引子判定：心理语义下"全模态高风险角"（各维=1.0）的潜坐标，
        # 距离该角最近的吸引子=危机态，最远=健康态（比 PCA 方向符号更鲁棒）。
        z_worst = manifold.transform_point({d: 1.0 for d in manifold.dim_names_})
        if z_worst is not None:
            dist_worst = np.array([float(np.linalg.norm(fp.z - z_worst)) for fp in attractors])
            crisis_fp = attractors[int(np.argmin(dist_worst))]
            health_fp = attractors[int(np.argmax(dist_worst))]
        else:
            if risk_direction is None:
                risk_direction = np.ones(drift.k_)
            proj = np.array([float(fp.z @ risk_direction) for fp in attractors])
            crisis_fp = attractors[int(np.argmax(proj))]
            health_fp = attractors[int(np.argmin(proj))]
        if crisis_fp is health_fp:
            return None

        # 当前所处吸引子（逃逸源）：离当前潜状态最近的吸引子——与 1D 语义一致
        z_now = Z[-1]
        dist_now = np.array([float(np.linalg.norm(fp.z - z_now)) for fp in attractors])
        current_fp = attractors[int(np.argmin(dist_now))]
        in_crisis = current_fp is crisis_fp
        crisis_radius = 0.25 * float(np.linalg.norm(crisis_fp.z - health_fp.z) + 1e-6)

        return LandscapeModel(
            manifold=manifold, drift=drift, pot=pot,
            attractors=attractors, saddles=saddles,
            health_fp=health_fp, crisis_fp=crisis_fp, current_fp=current_fp,
            in_crisis=in_crisis, z_now=z_now, crisis_radius=crisis_radius,
            dim_names=manifold.dim_names_,
            coupling=coupling,
        )

    def analyze(self, data: dict[str, list[float]], horizon: int = 12,
                risk_direction: Optional[np.ndarray] = None) -> Optional[LandscapeND]:
        model = self.fit_model(data, risk_direction)
        if model is None:
            return None
        return self.analyze_model(model, data, horizon)

    def analyze_model(self, model: LandscapeModel,
                      data: dict[str, list[float]],
                      horizon: int = 12) -> Optional[LandscapeND]:
        """在已拟合模型上计算逃逸率/instanton/粒子群，产出 LandscapeND。"""
        drift, pot, manifold = model.drift, model.pot, model.manifold
        attractors, saddles = model.attractors, model.saddles
        health_fp, crisis_fp, current_fp = model.health_fp, model.crisis_fp, model.current_fp
        z_now, in_crisis = model.z_now, model.in_crisis
        crisis_radius = model.crisis_radius
        all_fps = [fp for fp in attractors + saddles]

        if in_crisis:
            # 已在危机阱：壁垒≈ 0，闭式逃逸率不再适用，危机概率以粒子经验为主
            saddle_fp = min(saddles, key=lambda s: np.linalg.norm(s.z - health_fp.z))
            barrier = 0.0
            D_eff = float(np.mean(np.diag(drift.diffusion_)))
            rate = 0.5
            path = np.vstack([z_now.reshape(1, -1), crisis_fp.z.reshape(1, -1)])
        else:
            # 从当前（健康）吸引子逃逸：选择分隔 current 与 crisis 的 index-1 鞍点
            mid = 0.5 * (current_fp.z + crisis_fp.z)
            saddle_fp = min(saddles, key=lambda s: np.linalg.norm(s.z - mid))
            rate, barrier, D_eff = multidim_escape_rate(
                pot, drift, current_fp.z, saddle_fp.z, drift.diffusion_
            )
            path = minimum_energy_path(pot, current_fp.z, saddle_fp.z, crisis_fp.z)

        crisis_prob = 1.0 - math.exp(-rate * max(1, horizon))
        collapse_order = manifold.collapse_order_from_path(path)

        # 粒子群经验验证（从当前态出发，与闭式同一起点）
        p_emp, ci_lo, ci_hi = particle_ensemble(
            drift, drift.diffusion_, z_now, crisis_fp.z, crisis_radius,
            horizon=max(1, horizon), n_particles=self.n_particles,
        )
        # 已在危机阱时，闭式与粒子均指向高概率，以粒子经验为准
        if in_crisis:
            crisis_prob = max(crisis_prob, p_emp)

        return LandscapeND(
            n_dim=drift.k_,
            dim_names=manifold.dim_names_,
            latent_var_explained=manifold.explained_,
            fixed_points=all_fps,
            health_attractor=health_fp.z,
            crisis_attractor=crisis_fp.z,
            current_attractor=current_fp.z,
            in_crisis=in_crisis,
            saddle=saddle_fp.z,
            barrier_height=barrier,
            diffusion_eff=D_eff,
            escape_rate=rate,
            crisis_probability=crisis_prob,
            gradient_flux_ratio=pot.grad_flux_ratio_,
            collapse_path=path.tolist(),
            collapse_order=collapse_order,
            particle_prob=p_emp,
            particle_ci=(ci_lo, ci_hi),
            coupling=model.coupling.to_dict() if model.coupling is not None else None,
            method="multidim_langer",
        )


# ============================================================
# 单例
# ============================================================

_analyzer: Optional[MultidimLandscapeAnalyzer] = None


def get_multidim_analyzer() -> MultidimLandscapeAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = MultidimLandscapeAnalyzer()
    return _analyzer
