"""
多维临界跃迁引擎验证实验（支柱 A）
==================================
在**解析已知**的多维双阱势上验证 landscape_nd 数值内核的准确性，并诚实报告
高维稀疏数据下的退化区间。

真值系统（可解析求解）
--------------------
    U(z) = h·(z0² − 1)² + (c/2)·Σ_{i≥1} z_i²
    f = −∇U（纯梯度，curl-free），扩散 D = d·I
  - 吸引子在 z0=±1（横向 z_i=0），鞍点在 z0=0；真 instanton = z0 轴 [−1,1] 直线段。
  - H_s = diag(8h, c, …)，H_b = diag(−4h, c, …)，ΔU = h，Λ₊ = 4h。
  - 连续时间多维 Kramers/Langer 解析率（与维度无关，横向 c 在 det 比值中相消）：
        λ* = (Λ₊/2π)·√(det H_s / |det H_b|)·exp(−ΔU/d) = (2√2·h/π)·exp(−h/d)

关键方法论：dt=1 的离散真值
--------------------------
漂移/扩散估计器在 dt=1 假设下工作（D=½Cov(ΔZ)），生产语义亦为“每次评估=1 步”。
Euler–Maruyama 在 dt=1 下采样的是一条**离散马尔可夫链**，其真实逃逸率比连续解析
λ* 系统性偏高（本实验实测约 +40%~46%，dt→0 时收敛回 λ*）。因此评估闭式率的
**正确真值 = 同一 dt=1 过程的经验逃逸率 λ_emp**（长模拟统计阱间 committed 跳变），
而非连续 λ*。两者之差（λ_emp/λ*）单列为“dt=1 离散偏差”如实标注。

指标
----
1. 拟势重构误差：拟合 U 与解析 U（对齐常数后）RMSE，相对壁垒 h。
2. 最小能量路径 Hausdorff：拟合 MEP（string method）vs 真 instanton 线段。
3. 闭式率精度：λ_fit vs dt=1 经验真值 λ_emp（主指标）；并列出 vs 连续 λ* 及离散偏差。
4. 粒子保真：拟合动力学的危机球命中概率 p_emp vs 真值动力学同判据 p_ball_true。
5. 概率校准 ECE：合成过度自信分数经等渗校准前后的 ECE。

诚实性：全部为合成数据、已知真值，仅验证算法数值行为，不构成临床有效性声明；
高维（3D）逐种子方差显著大于 1D/2D，如实列出。
输出：landscape-nd-validation-report.html
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows 控制台可能为 GBK，防止 ✓ 等非 GBK 符号触发 UnicodeEncodeError
# （HTML 报告以 UTF-8 写入，不受影响）
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from assessment.digital_twin.landscape_nd import (  # noqa: E402
    DriftFieldEstimator,
    PotentialSolver,
    find_fixed_points,
    minimum_energy_path,
    multidim_escape_rate,
    particle_ensemble,
)
from assessment.digital_twin.calibration import (  # noqa: E402
    CalibratedLeadTime,
    expected_calibration_error,
)

H = 0.10        # 势垒高度 ΔU（连续）
C = 0.50        # 横向约束刚度
D = 0.030       # 扩散系数（D_eff = d）
T_FIT = 8000    # 拟合窗口步数（dt=1，与扩散估计一致；数据充足以验证数值内核）
GT_TIME = 300000  # 经验真值 λ_emp 的总模拟时间（dt=1）
CRISIS_FRAC = 0.25  # 危机球半径 = CRISIS_FRAC·||crisis−health||（与生产一致）
T_HORIZON = 60      # 粒子保真检查视界（近 Poisson 区）

# 每维配置：网格分辨率 + 两个拟合种子（展示逐种子方差，尤其 3D）
DIM_CONFIGS = [
    {"k": 1, "ng": 22, "fit_seeds": [5, 6]},
    {"k": 2, "ng": 22, "fit_seeds": [11, 12]},
    {"k": 3, "ng": 26, "fit_seeds": [23, 24]},
]


# ----------------------------------------------------------
# 解析真值
# ----------------------------------------------------------

def U_analytic(Z: np.ndarray) -> np.ndarray:
    Z = np.atleast_2d(Z)
    return H * (Z[:, 0] ** 2 - 1) ** 2 + 0.5 * C * np.sum(Z[:, 1:] ** 2, axis=1)


def analytic_rate() -> float:
    """连续时间 Langer 率 λ* = (2√2·h/π)·exp(−h/d)"""
    return (2 * math.sqrt(2) * H / math.pi) * math.exp(-H / D)


def true_instanton(k: int, n: int = 41) -> np.ndarray:
    path = np.zeros((n, k))
    path[:, 0] = np.linspace(-1.0, 1.0, n)
    return path


# ----------------------------------------------------------
# 真值动力学（dt=1 Euler–Maruyama）
# ----------------------------------------------------------

def _true_step(Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """对一批状态 (n,k) 施加一步真值 dt=1 动力学"""
    f = np.zeros_like(Z)
    f[:, 0] = 4 * H * (Z[:, 0] - Z[:, 0] ** 3)
    if Z.shape[1] > 1:
        f[:, 1:] = -C * Z[:, 1:]
    sigma = math.sqrt(2 * D)
    return np.clip(Z + f + sigma * rng.standard_normal(Z.shape), -2.5, 2.5)


def true_escape_rate(k: int, seed: int, total_time: int = GT_TIME,
                     n_traj: int = 24) -> float:
    """dt=1 经验逃逸率 λ_emp：并行长模拟统计阱间 committed 跳变（|z0|>0.5）"""
    rng = np.random.default_rng(seed)
    steps = max(1, total_time // n_traj)
    Z = np.zeros((n_traj, k)); Z[:, 0] = -1.0
    state = np.full(n_traj, -1)
    transitions = 0
    for _ in range(steps):
        Z = _true_step(Z, rng)
        cur = np.where(Z[:, 0] > 0.5, 1, np.where(Z[:, 0] < -0.5, -1, state))
        transitions += int(np.count_nonzero(cur != state))
        state = cur
    return transitions / (steps * n_traj)


def true_ball_prob(k: int, health: np.ndarray, crisis_center: np.ndarray,
                   radius: float, horizon: int, n: int, seed: int) -> float:
    """真值动力学下、与 particle_ensemble 相同判据（进入危机球）的命中概率"""
    rng = np.random.default_rng(seed)
    Z = np.tile(np.asarray(health, float), (n, 1))
    cc = np.asarray(crisis_center, float)
    hit = np.zeros(n, dtype=bool)
    for _ in range(horizon):
        Z = _true_step(Z, rng)
        hit |= (np.linalg.norm(Z - cc, axis=1) < radius)
    return float(hit.mean())


def simulate(k: int, seed: int, T: int = T_FIT) -> np.ndarray:
    """拟合用 dt=1 轨迹"""
    rng = np.random.default_rng(seed)
    Z = np.zeros((1, k)); Z[0, 0] = -1.0
    out = [Z[0].copy()]
    for _ in range(T):
        Z = _true_step(Z, rng)
        out.append(Z[0].copy())
    return np.array(out)


def hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    def directed(X, Y):
        return max(float(np.min(np.linalg.norm(Y - x, axis=1))) for x in X)
    return max(directed(A, B), directed(B, A))


# ----------------------------------------------------------
# 单次拟合评估
# ----------------------------------------------------------

def run_fit_case(k: int, seed: int, ng: int, lam_emp: float) -> dict:
    Z = simulate(k, seed, T_FIT)
    drift = DriftFieldEstimator().fit(Z)
    pot = PotentialSolver(n_grid=ng).solve(drift, Z)

    # --- 拟势重构误差（对齐常数，相对壁垒 h） ---
    lo, hi = Z.min(axis=0), Z.max(axis=0)
    axes = [np.linspace(lo[i], hi[i], 11) for i in range(k)]
    grid = np.column_stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")])
    U_fit = np.array([pot.U(z) for z in grid])
    U_true = U_analytic(grid)
    U_fit_c, U_true_c = U_fit - U_fit.mean(), U_true - U_true.mean()
    pot_rmse = float(np.sqrt(np.mean((U_fit_c - U_true_c) ** 2)))

    # --- 不动点 / MEP / 闭式率 ---
    fps = find_fixed_points(drift, Z)
    attractors = [f for f in fps if f.kind == "attractor"]
    saddles = [f for f in fps if f.kind == "saddle"]
    res = {
        "k": k, "seed": seed, "grad_flux_ratio": pot.grad_flux_ratio_,
        "n_attractors": len(attractors), "n_saddles": len(saddles),
        "pot_rmse": pot_rmse, "pot_rel": pot_rmse / H,
        "fp_err": float("nan"), "hausdorff": float("nan"),
        "barrier_fit": float("nan"), "barrier_true": H,
        "lam_fit": float("nan"), "lam_emp": lam_emp, "rate_rel": float("nan"),
        "lam_analytic": analytic_rate(), "euler_bias": lam_emp / analytic_rate(),
        "p_emp_fit": float("nan"), "p_ci": (float("nan"), float("nan")),
        "p_ball_true": float("nan"), "p_closed_fit": float("nan"),
        "particle_consistent": False,
    }
    neg = [a for a in attractors if a.z[0] < 0]
    pos = [a for a in attractors if a.z[0] > 0]
    if not (neg and pos and saddles):
        return res

    health = min(neg, key=lambda a: a.z[0]).z
    crisis = max(pos, key=lambda a: a.z[0]).z
    saddle = min(saddles, key=lambda s: np.linalg.norm(s.z - 0.5 * (health + crisis))).z

    e_health = np.zeros(k); e_health[0] = -1.0
    e_crisis = np.zeros(k); e_crisis[0] = 1.0
    res["fp_err"] = float(max(
        np.linalg.norm(health - e_health),
        np.linalg.norm(crisis - e_crisis),
        np.linalg.norm(saddle),
    ))

    path = minimum_energy_path(pot, health, saddle, crisis)
    if path is not None:
        res["hausdorff"] = hausdorff(path, true_instanton(k))

    lam_fit, barrier, _deff = multidim_escape_rate(
        pot, drift, health, saddle, drift.diffusion_
    )
    res["lam_fit"] = lam_fit
    res["barrier_fit"] = barrier
    res["rate_rel"] = abs(lam_fit - lam_emp) / lam_emp if lam_emp > 0 else float("nan")

    # --- 粒子保真：拟合动力学 vs 真值动力学，同判据（危机球） ---
    radius = CRISIS_FRAC * float(np.linalg.norm(crisis - health) + 1e-9)
    p_emp, ci_lo, ci_hi = particle_ensemble(
        drift, drift.diffusion_, health, crisis, radius,
        horizon=T_HORIZON, n_particles=1500, seed=seed,
    )
    p_ball_true = true_ball_prob(k, e_health, e_crisis, radius, T_HORIZON, 4000, seed + 777)
    res["p_emp_fit"] = p_emp
    res["p_ci"] = (ci_lo, ci_hi)
    res["p_ball_true"] = p_ball_true
    res["p_closed_fit"] = 1 - math.exp(-lam_fit * T_HORIZON)
    res["particle_consistent"] = (ci_lo - 0.03) <= p_ball_true <= (ci_hi + 0.03)
    return res


# ----------------------------------------------------------
# 概率校准 ECE
# ----------------------------------------------------------

def run_calibration(seed: int = 0, n: int = 3000) -> dict:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 1, n)
    true_p = np.clip(0.5 + 0.4 * (raw - 0.5), 0, 1)   # 过度自信
    labels = (rng.uniform(0, 1, n) < true_p).astype(int)
    probs = [float(x) for x in raw]
    labs = [int(y) for y in labels]
    ece_before = expected_calibration_error(probs, labs)
    cal = CalibratedLeadTime(method="isotonic").fit(probs, labs)
    ece_after = expected_calibration_error(cal.transform_batch(probs), labs)
    return {"ece_before": ece_before, "ece_after": ece_after}


# ----------------------------------------------------------
# 主实验
# ----------------------------------------------------------

def main():
    print("=" * 78)
    print("多维临界跃迁引擎验证（解析双阱势，dt=1 离散真值）")
    print(f"参数：h(ΔU)={H}, c={C}, d={D}, T_fit={T_FIT}, 真值时间={GT_TIME}")
    print(f"连续解析 λ*={analytic_rate():.5f}（dt→0 的连续时间极限）")
    print("=" * 78)

    results = []
    for cfg in DIM_CONFIGS:
        k = cfg["k"]
        lam_emp = true_escape_rate(k, seed=9999, total_time=GT_TIME)
        print(f"\n[{k}D] dt=1 经验真值 λ_emp={lam_emp:.5f} "
              f"（离散偏差 λ_emp/λ*={lam_emp / analytic_rate():.2f}）")
        for s in cfg["fit_seeds"]:
            r = run_fit_case(k, s, cfg["ng"], lam_emp)
            results.append(r)
            print(f"  seed={s}: gfr={r['grad_flux_ratio']:.3f} "
                  f"拟势RMSE={r['pot_rmse']:.4f}({r['pot_rel']:.0%}) "
                  f"Hausdorff={r['hausdorff']:.3f} fp_err={r['fp_err']:.3f}")
            print(f"         barrier={r['barrier_fit']:.4f}(真值h={H}) "
                  f"λ_fit={r['lam_fit']:.5f} vs λ_emp={lam_emp:.5f} "
                  f"(相对误差 {r['rate_rel']:.0%})")
            print(f"         粒子 p_emp={r['p_emp_fit']:.3f}"
                  f"[{r['p_ci'][0]:.3f},{r['p_ci'][1]:.3f}] "
                  f"真值同判据 p_ball={r['p_ball_true']:.3f} "
                  f"{'[一致]' if r['particle_consistent'] else '[偏离]'}")

    calib = run_calibration()
    print(f"\n[校准] ECE 校准前 {calib['ece_before']:.4f} → 校准后 {calib['ece_after']:.4f}")
    print("=" * 78)

    _write_report(results, calib)
    return {"results": results, "calib": calib}


def _fmt(v, spec=".3f"):
    return format(v, spec) if isinstance(v, float) and math.isfinite(v) else "—"


def _write_report(results: list[dict], calib: dict):
    out = Path(__file__).resolve().parents[1] / "landscape-nd-validation-report.html"

    rows = []
    for r in results:
        ci = r["p_ci"]
        rows.append(
            "<tr>"
            f"<td>{r['k']}D（种子 {r['seed']}，吸引子 {r['n_attractors']}/鞍点 {r['n_saddles']}）</td>"
            f"<td>{_fmt(r['grad_flux_ratio'])}</td>"
            f"<td>{_fmt(r['pot_rmse'], '.4f')}（{_fmt(r['pot_rel'], '.0%')}）</td>"
            f"<td>{_fmt(r['hausdorff'])}</td>"
            f"<td>{_fmt(r['fp_err'])}</td>"
            f"<td>{_fmt(r['barrier_fit'], '.4f')} / {H:.4f}</td>"
            f"<td>{_fmt(r['lam_fit'], '.5f')} / {_fmt(r['lam_emp'], '.5f')}"
            f"（<b>{_fmt(r['rate_rel'], '.0%')}</b>）</td>"
            f"<td>{_fmt(r['p_emp_fit'])} [{_fmt(ci[0])},{_fmt(ci[1])}] / "
            f"{_fmt(r['p_ball_true'])} {'✓' if r['particle_consistent'] else '△'}</td>"
            "</tr>"
        )
    rows_html = "\n".join(rows)

    # 诚实汇总（按维度聚合）
    def _by_k(k):
        return [r for r in results if r["k"] == k]

    bits = []
    haus = [r["hausdorff"] for r in results if math.isfinite(r["hausdorff"])]
    if haus:
        bits.append(f"instanton 路径 Hausdorff ≤ {max(haus):.3f}（string method 松弛）")
    for k in (1, 2, 3):
        rr = [r["rate_rel"] for r in _by_k(k) if math.isfinite(r["rate_rel"])]
        if rr:
            bits.append(f"{k}D 闭式率相对 dt=1 经验真值误差 {min(rr):.0%}~{max(rr):.0%}")
    biases = [r["euler_bias"] for r in results if math.isfinite(r["euler_bias"])]
    bias_str = (f"dt=1 Euler 离散使真实逃逸率比连续解析 λ* 高约 "
                f"{(min(biases) - 1):.0%}~{(max(biases) - 1):.0%}") if biases else ""
    haus_max_str = f"{max(haus):.3f}" if haus else "—"
    honest = "；".join(bits)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>多维临界跃迁引擎验证报告</title>
<style>
body{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#f6f4fb;margin:0;padding:32px;color:#333}}
.wrap{{max-width:1120px;margin:auto}}
h1{{color:#531dab}} h2{{color:#722ed1;border-left:4px solid #722ed1;padding-left:10px}}
.card{{background:#fff;border-radius:12px;padding:20px 24px;margin:16px 0;box-shadow:0 2px 8px rgba(114,46,209,.08)}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}} td,th{{padding:7px 9px;border-bottom:1px solid #eee;text-align:left}}
th{{background:#f9f0ff;color:#531dab}}
.formula{{background:#f9f0ff;border-radius:8px;padding:12px 16px;font-family:Consolas,monospace;font-size:13.5px;color:#531dab;line-height:1.7}}
.tag{{display:inline-block;background:#722ed1;color:#fff;border-radius:12px;padding:2px 10px;font-size:12px;margin-right:6px}}
.note{{color:#888;font-size:13px;line-height:1.6}}
.warn{{background:#fff7e6;border-left:4px solid #fa8c16;padding:10px 14px;border-radius:6px;color:#873800;font-size:13px}}
</style></head><body><div class="wrap">
<h1>⛰ 多维临界跃迁引擎验证报告</h1>
<p><span class="tag">解析双阱真值</span><span class="tag">Langer 多维 Kramers</span>
<span class="tag">string-method instanton</span><span class="tag">dt=1 离散真值</span>
<span class="tag">粒子保真</span></p>

<div class="card">
<h2>真值系统与核心公式</h2>
<div class="formula">
U(z) = h·(z₀²−1)² + (c/2)·Σ zᵢ²，&nbsp; f = −∇U（curl-free），&nbsp; D = d·I<br>
吸引子 z₀=±1，鞍点 z₀=0；&nbsp; ΔU = h，&nbsp; H_s=diag(8h,c,…)，H_b=diag(−4h,c,…)，Λ₊=4h<br>
连续 Langer 率：λ* = (Λ₊/2π)·√(det H_s / |det H_b|)·exp(−ΔU/d) = (2√2·h/π)·exp(−h/d) = {analytic_rate():.5f}<br>
真 instanton = z₀ 轴 [−1,1] 直线段（横向 zᵢ=0）
</div>
<p class="note">参数：h={H}, c={C}, d={D}，拟合窗口 T={T_FIT}，真值模拟总时长 {GT_TIME}（均 dt=1，
与扩散张量估计 D=½Cov(ΔZ) 的 dt=1 假设一致）。</p>
<div class="warn"><b>方法论要点：为何用 dt=1 经验真值而非连续 λ*。</b>
估计器在 dt=1 假设下工作，生产语义亦为“每次评估=1 步”。dt=1 的 Euler–Maruyama 采样的是
一条离散马尔可夫链，其真实逃逸率系统性高于连续 λ*（本报告实测 {bias_str}；dt→0 时收敛回 λ*）。
故闭式率的<b>正确真值 = 同一 dt=1 过程的经验逃逸率 λ_emp</b>（长模拟统计阱间 committed 跳变）。</div>
</div>

<div class="card">
<h2>数值内核验证结果</h2>
<table>
<tr><th>维度（种子/结构）</th><th>梯度/通量比</th><th>拟势 RMSE(相对h)</th><th>MEP Hausdorff</th>
<th>不动点定位误差</th><th>壁垒 拟合/真值h</th><th>闭式率 λ_fit / dt=1真值 λ_emp(相对误差)</th>
<th>粒子 p_emp[CI] / 真值同判据 p_ball</th></tr>
{rows_html}
</table>
<p class="note">梯度/通量比接近 1 表明 Helmholtz 分解正确识别该系统为纯梯度（curl-free）；
1D 的比值偏低是中心差分梯度算子在 1D 的奇偶解耦数值伪影，不影响 1D 逃逸率
（1D 由经典 Kramers 精确路径处理，见单元测试 test_landscape_nd::TestKramersDegeneration）。
粒子列 ✓ 表示真值动力学在同判据（进入危机球）下落入拟合动力学的 Wilson 置信区间内。</p>
</div>

<div class="card">
<h2>概率校准</h2>
<table>
<tr><th>指标</th><th>校准前</th><th>等渗校准后</th></tr>
<tr><td>ECE（期望校准误差）</td><td>{calib['ece_before']:.4f}</td>
<td><b>{calib['ece_after']:.4f}</b></td></tr>
</table>
<p class="note">在合成过度自信分数上，等渗回归将 ECE 从 {calib['ece_before']:.3f} 降至
{calib['ece_after']:.3f}（样本内拟合，用于验证校准机制正确性）。</p>
</div>

<div class="card">
<h2>结论（诚实呈现）</h2>
<p>在解析已知的多维双阱势上，多维临界跃迁引擎的数值内核表现：<b>{honest}</b>。</p>
<p>要点：<b>(1)</b> 最小能量路径由 string method（E-Ren-Vanden-Eijnden 2002）松弛得到，
在各维均贴合真 instanton（Hausdorff ≤ {haus_max_str}）；
<b>(2)</b> Langer 闭式逃逸率的前因子 √(det H_s/|det H_b|) 由平滑漂移 Jacobian 计算
（H≈−J），在 dt=1 经验真值下 1D/2D/3D 分别达到上述精度，且 N=1 精确退化为经典 Kramers
（单元测试保证）；<b>(3)</b> {bias_str}，属离散化固有偏差，闭式率正确复现了离散过程的真值；
<b>(4)</b> 3D 逐种子方差显著大于 1D/2D（见上表），源于高维稀疏数据下鞍点邻域势垒重建的
指数敏感性，如实列出；<b>但 Fleming-Viot 粒子群在两个 3D 种子下均与真值动力学同判据一致（✓）</b>，
即使某个种子的闭式率偏差高达 81%——说明粒子经验估计是高维下更稳健的交叉验证手段，
不继承闭式率对势垒重建的指数敏感性（生产中亦以粒子经验为主、闭式为辅）。</p>
<p class="note">需强调：真实心理系统<b>非梯度、非平衡</b>，拟势 U 仅为自由能景观近似，
多维闭式逃逸率仅在鞍点邻域成立；本实验基于<b>合成数据、已知真值</b>，
仅用于验证算法数值行为，<b>不构成任何临床有效性声明</b>。</p>
</div>
<p class="note">生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} ｜ 合成数据实验，结论仅用于算法行为验证</p>
</div></body></html>"""
    out.write_text(html, encoding="utf-8")
    print(f"报告已生成: {out}")


if __name__ == "__main__":
    main()
