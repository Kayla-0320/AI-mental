"""
CT-PLEW: Critical-Transition Potential-Landscape Early Warning
心理势能景观与临界跃迁预警算法（数字孪生精确引擎）
================================================================

原创性声明
----------
本算法是本平台原创的精确预测引擎，与既有数字孪生的线性状态空间模型互补：
线性模型只能外推趋势，而心理系统本质上是非线性多稳态系统——
"健康态"与"危机态"是势能景观上的两个吸引子（attractor），
危机爆发 = 状态粒子翻越两态之间的势能壁垒（临界跃迁）。

核心思想（统计物理 → 个性化心理健康）
--------------------------------------
1. Kramers-Moyal 展开：从个体时间序列非参数估计漂移 f(x) 与扩散 D，
   即 dX = f(X)dt + sqrt(2D)dW 的个性化随机动力学。
2. 势能景观：U(x) = -∫f dx，吸引子 = U 的极小值，壁垒 = U 的极大值。
3. Kramers 逃逸率（闭式精确解）：
       λ = sqrt(|f'(xs)| · |f'(xb)|) / (2π) · exp(-ΔU / D)
   其中 xs 为当前吸引子、xb 为危机方向壁垒、ΔU = U(xb) - U(xs)。
    horizon T 内危机概率 P = 1 - exp(-λT)  ——  解析精确，无需蒙特卡洛。
4. 临界慢化预警（EWS）：滑窗 lag-1 自相关 / 方差 / 偏度的 Kendall τ 趋势，
   在跃迁前捕捉"恢复力下降"信号（critical slowing down）。
5. 因果耦合倾斜：个人因果图中父维度的偏离作为常数倾斜 c 叠加到子维度漂移，
   f_eff = f + c，精确重算壁垒与逃逸率。
6. 反事实干预规划：对每个候选靶点施加位移 δ，闭式重算 Kramers 逃逸率，
   选择使危机概率下降最大的靶点 —— 干预决策有解析依据。

与线性数字孪生的分工
--------------------
- PsychologicalDigitalTwin（线性）：短时程趋势外推、观测融合。
- CT-PLEW（本模块）：韧性度量（壁垒高度）、临界跃迁概率、预警、干预靶点。

临床/物理依据
-------------
- Kramers escape rate (Kramers, 1940)：势阱逃逸的解析理论
- Critical slowing down as early warning (Scheffer et al., 2009, Nature)
- 抑郁态双稳态与临界跃迁证据 (van de Leemput et al., 2014, PNAS)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from assessment.digital_twin.landscape_nd import (
    LandscapeND,
    get_multidim_analyzer,
)
from assessment.digital_twin.calibration import CalibratedLeadTime

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class LandscapeAnalysis:
    """单维度的势能景观分析结果"""
    dim: str
    diffusion: float                 # 扩散系数 D（个体噪声强度）
    attractors: list[float]          # 稳定吸引子位置（心理稳态）
    barriers: list[float]            # 势能壁垒位置（临界点）
    current_attractor: float         # 当前所处吸引子
    crisis_barrier: float            # 危机方向壁垒位置 xb
    barrier_height: float            # ΔU = U(xb) - U(xs)（漂移单位）
    resilience: float                # ΔU / D（无量纲韧性：越大越难跃迁）
    recovery_rate: float             # |f'(xs)| 局部恢复率（临界慢化指标）
    n_attractors: int                # 吸引子个数（>1 表示多稳态）
    x_now: float = 0.5               # 当前状态位置
    grid: list[float] = field(default_factory=list)
    potential: list[float] = field(default_factory=list)
    drift: list[float] = field(default_factory=list)

    def to_dict(self, downsample: int = 41) -> dict:
        step = max(1, len(self.grid) // downsample)
        return {
            "dim": self.dim,
            "diffusion": round(self.diffusion, 6),
            "attractors": [round(a, 4) for a in self.attractors],
            "barriers": [round(b, 4) for b in self.barriers],
            "current_attractor": round(self.current_attractor, 4),
            "crisis_barrier": round(self.crisis_barrier, 4),
            "barrier_height": round(self.barrier_height, 5),
            "resilience": round(self.resilience, 4),
            "recovery_rate": round(self.recovery_rate, 5),
            "n_attractors": self.n_attractors,
            "x_now": round(self.x_now, 4),
            "grid": [round(g, 4) for g in self.grid[::step]],
            "potential": [round(u, 5) for u in self.potential[::step]],
            "drift": [round(f, 5) for f in self.drift[::step]],
        }


@dataclass
class EWSReport:
    """临界慢化早期预警报告（单维度）"""
    dim: str
    ar1_trend_tau: float        # lag-1 自相关趋势 Kendall τ（>0 = 临界慢化）
    ar1_trend_p: float
    var_trend_tau: float        # 方差趋势 Kendall τ
    var_trend_p: float
    current_ar1: float          # 当前窗口自相关
    current_variance: float
    ews_score: float            # 综合预警分 [0,1]
    warning: bool               # 是否触发预警

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "ar1_trend_tau": round(self.ar1_trend_tau, 4),
            "ar1_trend_p": round(self.ar1_trend_p, 4),
            "var_trend_tau": round(self.var_trend_tau, 4),
            "var_trend_p": round(self.var_trend_p, 4),
            "current_ar1": round(self.current_ar1, 4),
            "current_variance": round(self.current_variance, 5),
            "ews_score": round(self.ews_score, 4),
            "warning": self.warning,
        }


@dataclass
class EscapeRisk:
    """Kramers 临界跃迁（逃逸）风险"""
    dim: str
    escape_rate: float           # λ（每步逃逸概率速率）
    crisis_probability: float    # horizon 内危机概率 1-exp(-λT)
    resilience: float            # 无量纲壁垒高度
    method: str                  # kramers / no_barrier / insufficient_data

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "escape_rate": round(self.escape_rate, 6),
            "crisis_probability": round(self.crisis_probability, 4),
            "resilience": round(self.resilience, 4),
            "method": self.method,
        }


@dataclass
class CounterfactualPlan:
    """反事实干预规划（闭式 Kramers 重算）"""
    target_dim: str
    shift: float                       # 施加的位移（负=改善）
    baseline_crisis_prob: float
    post_crisis_prob: float
    benefit: float                     # 危机概率下降量
    affected_dims: list[str]
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "target_dim": self.target_dim,
            "shift": round(self.shift, 4),
            "baseline_crisis_prob": round(self.baseline_crisis_prob, 4),
            "post_crisis_prob": round(self.post_crisis_prob, 4),
            "benefit": round(self.benefit, 4),
            "affected_dims": self.affected_dims,
            "reasoning": self.reasoning,
        }


@dataclass
class PrecisionReport:
    """CT-PLEW 精确分析报告（用户级）"""
    user_id: str
    horizon: int
    landscapes: dict[str, LandscapeAnalysis] = field(default_factory=dict)
    ews: dict[str, EWSReport] = field(default_factory=dict)
    escapes: dict[str, EscapeRisk] = field(default_factory=dict)
    crisis_probability: float = 0.0          # 全维度最大跃迁概率（1D 最弱环节）
    critical_dim: Optional[str] = None       # 风险最高维度
    tipping_eta_steps: Optional[float] = None  # 预计跃迁时间（步）
    counterfactual: Optional[CounterfactualPlan] = None
    ews_alert: bool = False
    multistable: bool = False                # 是否检测到多稳态
    # --- 多维临界跃迁引擎（CT-PLEW N 维扩展，数据不足时为 None） ---
    multidim: Optional[LandscapeND] = None
    latent_dim: Optional[int] = None
    collapse_order: list[str] = field(default_factory=list)   # 模态 domino 崩溃顺序
    collapse_path: list[list[float]] = field(default_factory=list)
    saddle: Optional[list[float]] = None
    multidim_crisis_prob: Optional[float] = None
    escape_ci: Optional[list[float]] = None       # 粒子群经验概率 Wilson 区间
    gradient_flux_ratio: Optional[float] = None   # 梯度主导度（≈1 近平衡）
    calibration: Optional[dict] = None            # 校准信息（附 calibrator 时）

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "horizon": self.horizon,
            "algorithm": "CT-PLEW (Kramers potential-landscape early warning)",
            "landscapes": {k: v.to_dict() for k, v in self.landscapes.items()},
            "ews": {k: v.to_dict() for k, v in self.ews.items()},
            "escapes": {k: v.to_dict() for k, v in self.escapes.items()},
            "crisis_probability": round(self.crisis_probability, 4),
            "critical_dim": self.critical_dim,
            "tipping_eta_steps": (
                round(self.tipping_eta_steps, 2)
                if self.tipping_eta_steps is not None else None
            ),
            "counterfactual": (
                self.counterfactual.to_dict() if self.counterfactual else None
            ),
            "ews_alert": self.ews_alert,
            "multistable": self.multistable,
            # --- 多维扩展 ---
            "multidim": self.multidim.to_dict() if self.multidim else None,
            "latent_dim": self.latent_dim,
            "collapse_order": self.collapse_order,
            "collapse_path": [[round(float(v), 4) for v in n] for n in self.collapse_path],
            "saddle": (
                [round(float(v), 4) for v in self.saddle] if self.saddle else None
            ),
            "multidim_crisis_prob": (
                round(self.multidim_crisis_prob, 4)
                if self.multidim_crisis_prob is not None else None
            ),
            "escape_ci": self.escape_ci,
            "gradient_flux_ratio": (
                round(self.gradient_flux_ratio, 4)
                if self.gradient_flux_ratio is not None else None
            ),
            "calibration": self.calibration,
        }


# ============================================================
# 统计工具
# ============================================================

def _kendall_tau(x: list[float]) -> tuple[float, float]:
    """Kendall τ 及其双侧 p 值（正态近似，无 scipy 依赖）"""
    n = len(x)
    if n < 3:
        return 0.0, 1.0
    conc = disc = 0
    for i in range(n):
        xi = x[i]
        for j in range(i + 1, n):
            d = x[j] - xi
            if d > 0:
                conc += 1
            elif d < 0:
                disc += 1
    denom = n * (n - 1) / 2.0
    tau = (conc - disc) / denom if denom else 0.0
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = (conc - disc) / math.sqrt(var) if var > 0 else 0.0
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(tau), float(min(1.0, p))


def _skewness(x: np.ndarray) -> float:
    s = x.std()
    if s < 1e-12:
        return 0.0
    return float(((x - x.mean()) ** 3).mean() / s ** 3)


def _smooth(y: np.ndarray, width: int = 5, passes: int = 2) -> np.ndarray:
    """移动平均平滑（保持端点）"""
    out = y.astype(float).copy()
    k = np.ones(width) / width
    for _ in range(passes):
        padded = np.pad(out, (width // 2, width // 2), mode="edge")
        out = np.convolve(padded, k, mode="valid")
    return out


# ============================================================
# 1. Kramers-Moyal 漂移/扩散估计
# ============================================================

class KramersMoyalEstimator:
    """从个体时间序列非参数估计随机动力学 dX = f(X)dt + sqrt(2D)dW

    方法：状态空间分箱 → 箱内 ΔX 的一阶矩（漂移）与二阶矩（扩散）
    → 插值到均匀网格 → 平滑。 bins 样本不足时返回 None（数据不够）。
    """

    def __init__(self, n_grid: int = 81, min_bin_samples: int = 3):
        self.n_grid = n_grid
        self.min_bin_samples = min_bin_samples

    def estimate(
        self, series: list[float]
    ) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
        x = np.asarray(series, dtype=float)
        if len(x) < 12:
            return None
        dx = np.diff(x)
        xt = x[:-1]

        # 分箱基于数据实际范围（窄范围序列在 [0,1] 全域分箱会导致占用箱不足）
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-6:
            return None
        pad = 0.02 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        n_bins = int(np.clip(len(x) // 8, 6, 14))
        edges = np.linspace(lo, hi, n_bins + 1)
        idx = np.clip(np.digitize(xt, edges[1:-1]), 0, n_bins - 1)

        centers, f_hat, v_hat = [], [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() >= self.min_bin_samples:
                centers.append(float(xt[m].mean()))
                f_hat.append(float(dx[m].mean()))
                v_hat.append(float(dx[m].var()))
        if len(centers) < 3:
            return None

        centers = np.array(centers)
        order = np.argsort(centers)
        centers, f_hat = centers[order], np.array(f_hat)[order]

        grid = np.linspace(0.0, 1.0, self.n_grid)
        f_grid = _smooth(np.interp(grid, centers, f_hat), width=5)
        diffusion = float(np.clip(np.median(v_hat) / 2.0, 1e-4, 0.05))
        return grid, f_grid, diffusion


# ============================================================
# 2. 势能景观与吸引子/壁垒分析
# ============================================================

class PotentialLandscape:
    """势能景观构建与多稳态结构分析

    U(x) = -∫ f dx（漂移单位）；吸引子 = f 由 + 变 - 的零点；
    壁垒（临界点）= f 由 - 变 + 的零点。
    tilt 参数用于因果耦合 / 反事实干预：f_eff = f + tilt。
    """

    @staticmethod
    def analyze(
        dim: str,
        series: list[float],
        x_now: float,
        tilt: float = 0.0,
        n_grid: int = 81,
    ) -> Optional[LandscapeAnalysis]:
        est = KramersMoyalEstimator(n_grid=n_grid).estimate(series)
        if est is None:
            return None
        grid, f_raw, diffusion = est
        f = f_raw + tilt

        # 势能：U = -∫ f dx（梯形积分），归一化最小值为 0
        potential = -np.concatenate(
            ([0.0], np.cumsum((f[1:] + f[:-1]) / 2.0 * np.diff(grid)))
        )
        potential = potential - potential.min()

        # 不动点：f 的符号变化
        attractors: list[float] = []
        barriers: list[float] = []
        for i in range(len(grid) - 1):
            if f[i] == 0.0 or f[i] * f[i + 1] >= 0.0:
                continue
            xz = float(
                grid[i] - f[i] * (grid[i + 1] - grid[i]) / (f[i + 1] - f[i])
            )
            if f[i] > 0:
                attractors.append(xz)   # + → - ：稳定
            else:
                barriers.append(xz)    # - → + ：壁垒

        # 无内部不动点：单调漂移 → 边界吸引子
        if not attractors:
            attractors = [1.0 if f.mean() > 0 else 0.0]

        current_attractor = min(attractors, key=lambda a: abs(a - x_now))

        # 危机方向（值越大风险越高）：x_now 上方最近的壁垒；无则取边界 1.0
        up_barriers = [b for b in barriers if b > x_now - 1e-9]
        crisis_barrier = min(up_barriers) if up_barriers else 1.0

        u_at = lambda xv: float(np.interp(xv, grid, potential))
        barrier_height = max(0.0, u_at(crisis_barrier) - u_at(current_attractor))

        # 局部恢复率 |f'(x)|（临界慢化的直接度量）
        fprime = np.gradient(f, grid)
        recovery_rate = float(abs(np.interp(current_attractor, grid, fprime)))

        return LandscapeAnalysis(
            dim=dim,
            diffusion=diffusion,
            attractors=attractors,
            barriers=barriers,
            current_attractor=current_attractor,
            crisis_barrier=crisis_barrier,
            barrier_height=barrier_height,
            resilience=barrier_height / diffusion if diffusion > 0 else 0.0,
            recovery_rate=recovery_rate,
            n_attractors=len(attractors),
            x_now=float(x_now),
            grid=grid.tolist(),
            potential=potential.tolist(),
            drift=f.tolist(),
        )


# ============================================================
# 3. 临界慢化早期预警（EWS）
# ============================================================

class CriticalSlowingDetector:
    """滑窗临界慢化检测：自相关/方差上升 = 恢复力下降 = 跃迁前兆

    趋势检验仅使用尾部窗口（trend_tail）：早期预警关心的是
    "近期"恢复力是否在下降，全历史趋势会被长平稳段稀释。
    """

    def __init__(self, window: int = 20, min_windows: int = 4,
                 trend_tail: int = 30):
        self.window = window
        self.min_windows = min_windows
        self.trend_tail = trend_tail

    def analyze(self, dim: str, series: list[float]) -> Optional[EWSReport]:
        x = np.asarray(series, dtype=float)
        n = len(x)
        if n < self.window + self.min_windows:
            return None

        ar1s, vars_, = [], []
        for start in range(0, n - self.window + 1):
            w = x[start:start + self.window]
            v = float(w.var())
            if w[:-1].std() > 1e-12 and w[1:].std() > 1e-12:
                ar1s.append(float(np.corrcoef(w[:-1], w[1:])[0, 1]))
            else:
                ar1s.append(0.0)
            vars_.append(v)
        if len(ar1s) < self.min_windows:
            return None

        # 尾部窗口趋势：避免长平稳段稀释近期临界慢化信号
        tail = ar1s[-self.trend_tail:]
        tail_var = vars_[-self.trend_tail:]
        tau_ar1, p_ar1 = _kendall_tau(tail)
        tau_var, p_var = _kendall_tau(tail_var)
        cur_ar1 = float(np.mean(ar1s[-3:]))
        cur_var = float(np.mean(vars_[-3:]))

        # 综合预警分：趋势显著性加权 + 当前自相关水平
        w_ar1 = 1.0 if p_ar1 < 0.1 else 0.5
        w_var = 1.0 if p_var < 0.1 else 0.5
        score = (
            0.45 * max(tau_ar1, 0.0) * w_ar1
            + 0.30 * max(tau_var, 0.0) * w_var
            + 0.25 * max(cur_ar1, 0.0)
        )
        score = float(min(1.0, score))
        return EWSReport(
            dim=dim,
            ar1_trend_tau=tau_ar1,
            ar1_trend_p=p_ar1,
            var_trend_tau=tau_var,
            var_trend_p=p_var,
            current_ar1=cur_ar1,
            current_variance=cur_var,
            ews_score=score,
            warning=bool(score > 0.5 and tau_ar1 > 0.0),
        )


# ============================================================
# 4. Kramers 逃逸率模型（闭式精确危机概率）
# ============================================================

class KramersEscapeModel:
    """Kramers (1940) 势阱逃逸率：λ = sqrt(|f'(xs)||f'(xb)|)/(2π)·exp(-ΔU/D)

    horizon T 内危机概率 P = 1 - exp(-λT)，解析精确。
    """

    MAX_RATE = 0.5  # 单步逃逸率上限（物理约束）

    @staticmethod
    def escape_rate(landscape: LandscapeAnalysis) -> tuple[float, str]:
        if landscape.barrier_height <= 1e-6:
            # 无势能壁垒：Kramers 指数形式不适用，改用穿越率模型
            grid = np.asarray(landscape.grid)
            drift = np.asarray(landscape.drift)
            f_now = float(np.interp(landscape.x_now, grid, drift))
            gap = max(1e-6, landscape.crisis_barrier - landscape.x_now)
            if f_now > 0:
                # 漂移直指危机态：确定性穿越率 = 速度 / 距离
                rate = f_now / gap
                method = "no_barrier_drift"
            else:
                # 逆漂移爬升：扩散主导，速率 ≈ D / gap²
                rate = landscape.diffusion / gap ** 2
                method = "no_barrier_diffusion"
            return min(float(rate), KramersEscapeModel.MAX_RATE), method

        # 壁垒处曲率：用漂移在壁垒附近的斜率近似
        grid = np.asarray(landscape.grid)
        drift = np.asarray(landscape.drift)
        fprime = np.gradient(drift, grid)
        slope_b = abs(float(np.interp(landscape.crisis_barrier, grid, fprime)))
        slope_b = max(slope_b, 1e-3)
        slope_s = max(landscape.recovery_rate, 1e-3)

        pref = math.sqrt(slope_s * slope_b) / (2 * math.pi)
        rate = pref * math.exp(-landscape.resilience)
        return min(float(rate), KramersEscapeModel.MAX_RATE), "kramers"

    @staticmethod
    def crisis_probability(rate: float, horizon: int) -> float:
        return float(1.0 - math.exp(-rate * max(1, horizon)))


# ============================================================
# 5. CT-PLEW 引擎（整合：景观 + EWS + 逃逸 + 反事实）
# ============================================================

class CTPLEWEngine:
    """CT-PLEW 主引擎

    输入：用户观测历史（list[dict]）+ 个人因果图（可选）
    输出：PrecisionReport（韧性 / 预警 / 跃迁概率 / 反事实干预规划）
    """

    def __init__(self, min_history: int = 20, intervention_shift: float = -0.15,
                 use_multidim: bool = True, min_dims_multidim: int = 3,
                 min_history_multidim: int = 40,
                 calibrator: Optional[CalibratedLeadTime] = None):
        self.min_history = min_history
        self.intervention_shift = intervention_shift
        # 多维临界跃迁引擎（数据/维度达阈时启用，否则回退 1D）
        self.use_multidim = use_multidim
        self.min_dims_multidim = min_dims_multidim
        self.min_history_multidim = min_history_multidim
        self.calibrator = calibrator

    # ----------------------------------------------------------
    # 主分析入口
    # ----------------------------------------------------------
    def analyze(
        self,
        user_id: str,
        history: list[dict],
        causal_graph=None,
        horizon: int = 12,
    ) -> PrecisionReport:
        report = PrecisionReport(user_id=user_id, horizon=horizon)
        if len(history) < self.min_history:
            return report

        # 收集各维度序列
        dims: dict[str, list[float]] = {}
        for h in history:
            for k, v in h.items():
                dims.setdefault(k, []).append(float(v))
        dims = {k: v for k, v in dims.items() if len(v) >= self.min_history}
        if not dims:
            return report

        x_now = {k: v[-1] for k, v in dims.items()}
        baseline = {k: float(np.mean(v)) for k, v in dims.items()}

        # 因果耦合倾斜：父维度偏离 → 子维度漂移常数偏移
        tilt = self._causal_tilt(causal_graph, x_now, baseline)

        # 逐维度：景观 + EWS + Kramers 逃逸
        for dim, series in dims.items():
            land = PotentialLandscape.analyze(
                dim, series, x_now[dim], tilt=tilt.get(dim, 0.0)
            )
            if land is None:
                continue
            report.landscapes[dim] = land
            if land.n_attractors > 1:
                report.multistable = True

            ews = CriticalSlowingDetector().analyze(dim, series)
            if ews is not None:
                report.ews[dim] = ews
                if ews.warning:
                    report.ews_alert = True

            rate, method = KramersEscapeModel.escape_rate(land)
            prob = KramersEscapeModel.crisis_probability(rate, horizon)
            report.escapes[dim] = EscapeRisk(
                dim=dim,
                escape_rate=rate,
                crisis_probability=prob,
                resilience=land.resilience,
                method=method,
            )

        if not report.escapes:
            return report

        # 全维度危机概率 = 最大值（最弱环节决定系统风险）
        worst = max(report.escapes.values(), key=lambda e: e.crisis_probability)
        report.crisis_probability = worst.crisis_probability
        report.critical_dim = worst.dim

        # 预计跃迁时间：确定性外推 与 随机逃逸 取较小者
        report.tipping_eta_steps = self._tipping_eta(
            report.landscapes.get(worst.dim), x_now.get(worst.dim, 0.5),
            worst.escape_rate,
        )

        # 反事实干预规划
        report.counterfactual = self._counterfactual(
            dims, x_now, baseline, causal_graph, horizon,
        )

        # 多维临界跃迁引擎：数据/维度达阈时叠加系统级耦合分析（不覆盖 1D 结果）
        self._attach_multidim(report, dims, history, horizon)
        return report

    # ----------------------------------------------------------
    # 多维临界跃迁（CT-PLEW N 维扩展）
    # ----------------------------------------------------------
    def _attach_multidim(self, report: "PrecisionReport", dims: dict,
                         history: list, horizon: int) -> None:
        """在 1D 逐维分析之上叠加多维耦合景观（数据不足时静默回退）

        多维结果作为独立字段（multidim_*），不覆盖 crisis_probability，
        保证向后兼容与现有契约。
        """
        if not self.use_multidim:
            return
        if (len(dims) < self.min_dims_multidim
                or len(history) < self.min_history_multidim):
            return
        try:
            nd = get_multidim_analyzer().analyze(dims, horizon)
        except Exception as e:  # 数值失败不拖垮主流程
            logger.warning(f"多维景观分析失败，回退 1D：{e}")
            return
        if nd is None:
            return
        report.multidim = nd
        report.latent_dim = nd.n_dim
        report.collapse_order = nd.collapse_order
        report.collapse_path = nd.collapse_path
        report.saddle = nd.saddle.tolist() if nd.saddle is not None else None
        report.multidim_crisis_prob = nd.crisis_probability
        report.gradient_flux_ratio = nd.gradient_flux_ratio
        if nd.particle_ci is not None:
            report.escape_ci = [round(nd.particle_ci[0], 4), round(nd.particle_ci[1], 4)]
        if nd.collapse_order:
            report.multistable = True
        # 校准：附校准器时给出校准后概率（ECE 在验证实验中计算）
        if self.calibrator is not None and self.calibrator._fitted:
            report.calibration = {
                "method": self.calibrator.method,
                "raw_prob": round(nd.crisis_probability, 4),
                "calibrated_prob": round(
                    self.calibrator.transform(nd.crisis_probability), 4
                ),
            }

    # ----------------------------------------------------------
    # 因果耦合倾斜
    # ----------------------------------------------------------
    @staticmethod
    def _causal_tilt(
        causal_graph, x_now: dict, baseline: dict, extra_dev: Optional[dict] = None,
    ) -> dict:
        """父维度偏离对子维度漂移的常数倾斜 c = Σ (strength/lag)·dev(parent)"""
        dev = {k: x_now[k] - baseline[k] for k in x_now}
        if extra_dev:
            for k, v in extra_dev.items():
                dev[k] = dev.get(k, 0.0) + v
        tilt: dict[str, float] = {}
        if causal_graph is None:
            return tilt
        for edge in causal_graph.edges:
            if edge.cause in dev and edge.effect in x_now:
                lag = max(1, int(getattr(edge, "lag", 1)))
                tilt[edge.effect] = (
                    tilt.get(edge.effect, 0.0)
                    + edge.strength * dev[edge.cause] / lag
                )
        return tilt

    # ----------------------------------------------------------
    # 跃迁时间估计
    # ----------------------------------------------------------
    @staticmethod
    def _tipping_eta(
        land: Optional[LandscapeAnalysis], x_now: float, rate: float,
    ) -> Optional[float]:
        if land is None:
            return None
        etas = []
        if rate > 1e-6:
            etas.append(1.0 / rate)
        # 确定性外推：当前漂移朝壁垒方向时的到达时间
        grid = np.asarray(land.grid)
        drift = np.asarray(land.drift)
        f_now = float(np.interp(x_now, grid, drift))
        gap = land.crisis_barrier - x_now
        if f_now > 1e-4 and gap > 0:
            etas.append(gap / f_now)
        return float(min(etas)) if etas else None

    # ----------------------------------------------------------
    # 反事实干预规划（闭式 Kramers 重算）
    # ----------------------------------------------------------
    def _counterfactual(
        self,
        dims: dict,
        x_now: dict,
        baseline: dict,
        causal_graph,
        horizon: int,
    ) -> Optional[CounterfactualPlan]:
        base_prob = self._system_crisis_prob(
            dims, x_now, baseline, causal_graph, horizon, None, 0.0,
        )
        if base_prob is None:
            return None

        best: Optional[CounterfactualPlan] = None
        for target in dims:
            post_prob, affected = self._system_crisis_prob(
                dims, x_now, baseline, causal_graph, horizon,
                target, self.intervention_shift, return_affected=True,
            )
            if post_prob is None:
                continue
            benefit = base_prob - post_prob
            if benefit > 0.01 and (best is None or benefit > best.benefit):
                best = CounterfactualPlan(
                    target_dim=target,
                    shift=self.intervention_shift,
                    baseline_crisis_prob=base_prob,
                    post_crisis_prob=post_prob,
                    benefit=benefit,
                    affected_dims=affected or [target],
                    reasoning=(
                        f"反事实仿真：将「{target}」改善 "
                        f"{abs(self.intervention_shift):.0%} 后，经因果链传导重算 "
                        f"Kramers 逃逸率，{horizon} 步内危机概率由 "
                        f"{base_prob:.0%} 降至 {post_prob:.0%}。"
                    ),
                )
        return best

    def _system_crisis_prob(
        self,
        dims: dict,
        x_now: dict,
        baseline: dict,
        causal_graph,
        horizon: int,
        target: Optional[str],
        shift: float,
        return_affected: bool = False,
    ):
        """施加反事实位移后的系统危机概率（闭式重算）"""
        extra_dev: dict[str, float] = {}
        affected: list[str] = []
        if target is not None:
            extra_dev[target] = shift
            affected.append(target)
            # 深度 2 因果传导（衰减 0.5）
            if causal_graph is not None:
                for edge in causal_graph.edges:
                    if edge.cause == target and edge.effect in dims:
                        lag = max(1, int(getattr(edge, "lag", 1)))
                        extra_dev[edge.effect] = (
                            extra_dev.get(edge.effect, 0.0)
                            + 0.5 * edge.strength * shift / lag
                        )
                        affected.append(edge.effect)

        tilt = self._causal_tilt(causal_graph, x_now, baseline, extra_dev)
        worst_prob = None
        for dim, series in dims.items():
            land = PotentialLandscape.analyze(
                dim, series, x_now[dim], tilt=tilt.get(dim, 0.0)
            )
            if land is None:
                continue
            rate, _ = KramersEscapeModel.escape_rate(land)
            prob = KramersEscapeModel.crisis_probability(rate, horizon)
            worst_prob = prob if worst_prob is None else max(worst_prob, prob)
        if return_affected:
            return worst_prob, affected
        return worst_prob


# ============================================================
# 全局单例
# ============================================================

_engine: Optional[CTPLEWEngine] = None


def get_precision_engine() -> CTPLEWEngine:
    """获取 CT-PLEW 引擎全局单例"""
    global _engine
    if _engine is None:
        _engine = CTPLEWEngine()
    return _engine
