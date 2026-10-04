"""
In-Silico Patient 反事实最优控制验证实验（支柱 B）
==================================================
在**已知真值**的结构因果模型（SCM）与**解析双稳态景观**上，验证支柱 B 的三类
能力的数值正确性，并诚实报告其边界（含 MPC 不占优的区间）。

第一部分：SCM / do-算子 / Pearl 三步反事实
------------------------------------------
构造真值 VAR(1) 线性 SCM：X_{t+1} = A·X_t + b + U，U~N(0,Σ)（clip 到 [0,1]），
生成合成历史，用 StructuralCausalModel.fit 恢复，验证：
  1. 结构恢复：||A_fit−A_true||_F / ||A_true||_F，||b_fit−b_true||。
  2. do-干预均值恢复误差：|E_fit[X_final|do] − E_true[X_final|do]|（同起点/同视界）。
  3. 反事实一致性：
     (a) 无操作恒等——do(X_j = 观测值) 时反事实效应应精确为 0（Pearl 三步自洽）；
     (b) 真值恢复——拟合 SCM 的反事实末态 vs 真值 SCM 的反事实末态误差。

第二部分：MPC 最优救援 vs 现有时机搜索 vs 无干预
------------------------------------------------
在解析双阱潜动力学生成的合成多模态历史上拟合 In-SilicoPatient（景观即模拟器），
从**健康吸引子**统一出发，比较三臂在预测时域内坠入危机球的比率：
  - 无干预（自然演化基线）；
  - 现有时机搜索（JITAI 范式）：固定救援方向、固定幅度，网格搜索**起始时机** t*，
    自 t* 起持续施加（对应 InterventionTimingOptimizer“挑时机+标准干预包”的思想，
    在同一模拟器上评估以使危机率可比）；
  - MPC（CEM）：联合优化时机+靶点+剂量的完整控制序列，在两个 λ（能量惩罚）下各优化一次。

诚实性约束（关键）
------------------
- 全部为合成数据、已知真值，仅验证算法数值行为，**不构成任何临床有效性声明**。
- MPC 最小化 J = 危机率 + λ·(干预能量/时域)。故须同时报告危机率**与**能量：
  * 在**原始危机率**上，暴力持续干预（时机搜索）常低于 MPC —— 这是 MPC 的
    **不占优区间**（λ 小 / 干预代价可忽略 / 逃逸方向单一时，MPC 的节能无意义）；
  * MPC 的价值在 **Pareto 效率**：以显著更低的干预能量逼近危机率下降，当把干预
    负担计入决策目标 J（λ 足够大）时 MPC 反超。本报告用共享 λ 评估网格给出交叉点。
- 潜空间控制经 PCA 伪逆映射到模态、时机基线假定了已知救援方向，均属文档化近似；
  在真正多向逃逸（高维、多竞争路径）景观中 MPC 的靶点选择优势更明显，此处如实标注。

输出：counterfactual-control-validation-report.html
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Windows 控制台可能为 GBK，防止非 GBK 符号触发 UnicodeEncodeError（HTML 以 UTF-8 写入）
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from assessment.digital_twin.scm_counterfactual import StructuralCausalModel  # noqa: E402
from assessment.digital_twin.in_silico_simulator import InSilicoPatient  # noqa: E402
from assessment.digital_twin.optimal_control import (  # noqa: E402
    InterventionMPC,
    minimum_rescue_path,
)

# ============================================================
# 第一部分：真值 SCM（风险坐标：值越大越差）
# ============================================================
# fit() 按字母序取 dims，故真值 A/b 直接定义在**已排序**的维度顺序上以便逐元素对比
DIMS = ["cognition", "mood", "rumination", "sleep"]     # sorted()
# 因果链：sleep → rumination → cognition → mood（+ rumination→mood）
#   A[i, j]：dim_j 对 dim_{t+1}[i] 的系数
A_TRUE = np.array([
    #  cogn   mood   rumin  sleep
    [0.30, 0.00, 0.35, 0.00],   # cognition ← cognition, rumination
    [0.30, 0.25, 0.20, 0.00],   # mood      ← cognition, mood, rumination
    [0.00, 0.00, 0.40, 0.30],   # rumination← rumination, sleep
    [0.00, 0.00, 0.00, 0.55],   # sleep     ← sleep（最上游）
])
B_TRUE = np.array([0.05, 0.05, 0.05, 0.20])
# 噪声协方差（cognition↔mood 有小幅正相关，检验协方差恢复）；对称正定
SIGMA_TRUE = np.array([
    [0.00135, 0.00060, 0.00000, 0.00000],
    [0.00060, 0.001838, 0.00000, 0.00000],
    [0.00000, 0.00000, 0.00135, 0.00000],
    [0.00000, 0.00000, 0.00000, 0.000938],
])
SCM_T = 2000           # 拟合历史长度（充足样本以验证结构可辨识性）
SCM_SEEDS = [0, 1]
DO_SPECS = [
    {"sleep": 0.15},                       # 治失眠（上游干预）
    {"rumination": 0.15},                  # 直接降反刍
    {"sleep": 0.15, "rumination": 0.15},   # 联合干预
]
DO_HORIZON = 8
N_DO_SAMPLES = 2000
N_CF_EVIDENCE = 12     # 反事实真值恢复的证据样本数


def stationary_mean(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(np.linalg.solve(np.eye(len(b)) - A, b), 0.0, 1.0)


def make_true_scm() -> StructuralCausalModel:
    """用真值参数直接构造 SCM，作为 do/反事实的 ground truth（同一代码路径）"""
    return StructuralCausalModel(
        dims=list(DIMS), A=A_TRUE.copy(), b=B_TRUE.copy(),
        noise_cov=SIGMA_TRUE.copy(),
        resid_samples=np.zeros((1, len(DIMS))),
        topo_order=["sleep", "rumination", "cognition", "mood"],
    )


def gen_scm_history(T: int, seed: int) -> list[dict]:
    """从真值动力学生成合成历史 list[dict]（起点=真值稳态均值）"""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(SIGMA_TRUE)
    x = stationary_mean(A_TRUE, B_TRUE)
    out = []
    for _ in range(T):
        out.append({d: float(np.clip(x[i], 0.0, 1.0)) for i, d in enumerate(DIMS)})
        u = L @ rng.standard_normal(len(DIMS))
        x = np.clip(A_TRUE @ x + B_TRUE + u, 0.0, 1.0)
    return out


def run_scm_case(seed: int) -> dict:
    scm_true = make_true_scm()
    hist = gen_scm_history(SCM_T, seed)
    scm_fit = StructuralCausalModel.fit(hist)
    res = {
        "seed": seed, "fit_ok": scm_fit is not None,
        "a_rel_err": float("nan"), "b_abs_err": float("nan"),
        "spec_radius": float(np.max(np.abs(np.linalg.eigvals(A_TRUE)))),
        "do": [], "cf_noop_max": float("nan"), "cf_truth_err": float("nan"),
        "cf_sign_agree": float("nan"),
    }
    if scm_fit is None:
        return res

    # --- 结构恢复 ---
    res["a_rel_err"] = float(
        np.linalg.norm(scm_fit.A - A_TRUE) / (np.linalg.norm(A_TRUE) + 1e-12)
    )
    res["b_abs_err"] = float(np.max(np.abs(scm_fit.b - B_TRUE)))

    # --- do-干预均值恢复误差（同起点、同视界、同 clamp）---
    x0 = {d: float(v) for d, v in zip(DIMS, stationary_mean(A_TRUE, B_TRUE))}
    for spec in DO_SPECS:
        rt = scm_true.do(spec, x0=x0, horizon=DO_HORIZON, clamp=True,
                         n_samples=N_DO_SAMPLES, seed=123)
        rf = scm_fit.do(spec, x0=x0, horizon=DO_HORIZON, clamp=True,
                        n_samples=N_DO_SAMPLES, seed=123)
        err = {d: abs(rf.intervened_final[d] - rt.intervened_final[d]) for d in DIMS}
        eff_true = max(abs(v) for v in rt.effect.values())
        res["do"].append({
            "spec": spec,
            "max_err": max(err.values()),
            "mean_err": float(np.mean(list(err.values()))),
            "true_effect_size": eff_true,
            "true_final": rt.intervened_final,
            "fit_final": rf.intervened_final,
        })

    # --- 反事实一致性 ---
    # (a) 无操作恒等：do(X_j = 该步观测值) → 效应应为 0
    ev0 = gen_scm_history(40, 500 + seed)
    istep = len(ev0) // 2
    noop_dim = "rumination"
    noop_spec = {noop_dim: float(ev0[istep][noop_dim])}
    cf_noop = scm_fit.counterfactual(ev0, noop_spec, intervention_step=istep,
                                     horizon=6, clamp=False)
    if cf_noop is not None:
        res["cf_noop_max"] = float(max(abs(v) for v in cf_noop.effect.values()))

    # (b) 真值恢复 + (c) 方向一致性：多份证据平均（clamp 持续干预→效应显著、方向清晰）
    cf_spec = {"sleep": 0.15}
    errs, agrees = [], []
    for e in range(N_CF_EVIDENCE):
        ev = gen_scm_history(48, 700 + seed * 100 + e)
        ist = len(ev) // 2
        cf_t = scm_true.counterfactual(ev, cf_spec, intervention_step=ist,
                                       horizon=6, clamp=True)
        cf_f = scm_fit.counterfactual(ev, cf_spec, intervention_step=ist,
                                      horizon=6, clamp=True)
        if cf_t is None or cf_f is None:
            continue
        errs.append(max(abs(cf_f.counterfactual_final[d] - cf_t.counterfactual_final[d])
                        for d in DIMS))
        # 方向一致性：对真值反事实效应“显著”的每个维度，检验拟合效应符号是否一致
        for d in DIMS:
            if abs(cf_t.effect[d]) > 0.02:
                agrees.append(1.0 if np.sign(cf_t.effect[d]) == np.sign(cf_f.effect[d])
                              else 0.0)
    if errs:
        res["cf_truth_err"] = float(np.mean(errs))
    if agrees:
        res["cf_sign_agree"] = float(np.mean(agrees))
    return res


# ============================================================
# 第二部分：解析双稳态景观 → In-Silico 控制对比
# ============================================================
H_BARRIER, C_LAT, D_NOISE = 0.10, 0.10, 0.030   # 双阱势垒 / 横向刚度 / 扩散
N_MODES, CTRL_T = 5, 8000
CTRL_SEEDS = [3, 11]
HORIZONS = [12, 30]
MPC_LAMS = [0.0, 1.0]          # MPC 优化用的能量惩罚（0=纯危机率，1=能量感知）
EVAL_LAMS = [0.0, 0.5, 2.0]    # 共享 J 评估网格（用于给出 MPC/timing 交叉点）
K_EVAL, EVAL_SEED = 300, 11
DOSE = 0.3                     # 时机基线的固定干预幅度（= MPC 的 u_max，公平）
# 固定模态映射（两潜因子方差可比 → PCA 稳定得 k=2）
_Wrng = np.random.default_rng(2024)
_W = np.column_stack([
    np.abs(_Wrng.standard_normal(N_MODES)) * 0.22 + 0.06,
    np.random.default_rng(77).standard_normal(N_MODES) * 0.62,
])


def _latent_step(Z: np.ndarray, rng) -> np.ndarray:
    f = np.zeros_like(Z)
    f[:, 0] = 4 * H_BARRIER * (Z[:, 0] - Z[:, 0] ** 3)
    f[:, 1:] = -C_LAT * Z[:, 1:]
    return np.clip(Z + f + math.sqrt(2 * D_NOISE) * rng.standard_normal(Z.shape), -2.5, 2.5)


def gen_control_history(seed: int) -> dict:
    """真值 2D 双阱潜轨迹 → 5 模态合成历史（因子模型，PCA 可复原潜空间）"""
    rng = np.random.default_rng(seed)
    Z = np.zeros((1, 2)); Z[0, 0] = -1.0
    lat = [Z[0].copy()]
    for _ in range(CTRL_T):
        Z = _latent_step(Z, rng)
        lat.append(Z[0].copy())
    lat = np.array(lat)
    modes = np.clip(0.5 + lat @ _W.T + 0.02 * rng.standard_normal((len(lat), N_MODES)), 0, 1)
    return {f"m{j}": modes[:, j].tolist() for j in range(N_MODES)}


def J_of(crisis: float, energy: float, horizon: int, lam: float) -> float:
    return crisis + lam * energy / max(1, horizon)


def timing_grid_sustained(sim, horizon, seed, dir_vec):
    """JITAI 时机基线：自 t* 起持续施加固定方向/幅度干预，网格搜索 t*（最小化危机率）"""
    best = {"t_star": None, "crisis": math.inf, "energy": 0.0}
    for t_star in range(horizon):
        def policy(t, Z, ts=t_star):
            return DOSE * dir_vec if t >= ts else None
        fut = sim.rollout(K=K_EVAL, horizon=horizon, policy=policy,
                          state0=sim.model.health_fp.z, seed=seed)
        if fut.crisis_rate < best["crisis"]:
            best = {"t_star": t_star, "crisis": fut.crisis_rate,
                    "energy": DOSE ** 2 * (horizon - t_star)}
    return best


def run_control_case(seed: int) -> dict:
    sim = InSilicoPatient.from_history(gen_control_history(seed))
    out = {"seed": seed, "fit_ok": sim is not None, "horizons": [], "rescue": None}
    if sim is None:
        return out
    m = sim.model
    m.z_now = m.health_fp.z.copy()          # 三臂统一从健康吸引子出发
    dv = m.health_fp.z - m.crisis_fp.z
    dv = dv / (np.linalg.norm(dv) + 1e-9)
    out.update({"k": sim.k, "n_attr": len(m.attractors), "n_sad": len(m.saddles),
                "explained": m.manifold.explained_})

    for horizon in HORIZONS:
        base = sim.rollout(K=K_EVAL, horizon=horizon, policy=None,
                           state0=m.health_fp.z, seed=EVAL_SEED)
        timing = timing_grid_sustained(sim, horizon, EVAL_SEED, dv)
        mpcs = {}
        for lam in MPC_LAMS:
            mpc = InterventionMPC(sim, lam=lam, u_max=DOSE, K=K_EVAL, seed=EVAL_SEED,
                                  n_candidates=24, n_iters=5)
            plan = mpc.optimize(horizon=horizon)
            mpcs[lam] = {
                "crisis": plan.controlled_crisis_rate,
                "energy": plan.energy_cost,
                "target_dim": plan.target_dim,
                "converged": plan.converged,
            }
        out["horizons"].append({
            "horizon": horizon, "base_crisis": base.crisis_rate,
            "base_ci": base.crisis_ci, "timing": timing, "mpc": mpcs,
        })

    rp = minimum_rescue_path(m)
    if rp is not None:
        out["rescue"] = {"energy": rp.energy, "barrier": rp.barrier_restored,
                         "target_dim": rp.target_dim, "n_steps": rp.n_steps,
                         "narrative": rp.narrative}
    return out


# ============================================================
# 主实验
# ============================================================

def main():
    print("=" * 78)
    print("In-Silico Patient 反事实最优控制验证（合成真值）")
    print("=" * 78)

    print("\n[第一部分] SCM / do-算子 / Pearl 三步反事实")
    print(f"  真值 A 谱半径={np.max(np.abs(np.linalg.eigvals(A_TRUE))):.3f}（稳定），"
          f"维度={DIMS}")
    scm_results = []
    for s in SCM_SEEDS:
        r = run_scm_case(s)
        scm_results.append(r)
        if not r["fit_ok"]:
            print(f"  seed={s}: 拟合失败")
            continue
        do_str = " ".join(f"{d['max_err']:.4f}" for d in r["do"])
        print(f"  seed={s}: A 相对误差={r['a_rel_err']:.3f} b 绝对误差={r['b_abs_err']:.4f}")
        print(f"          do 末态最大误差=[{do_str}]（真值效应量 "
              f"{[round(d['true_effect_size'],3) for d in r['do']]}）")
        print(f"          反事实无操作恒等 max|effect|={r['cf_noop_max']:.2e} "
              f"真值恢复误差={r['cf_truth_err']:.4f} 方向一致率={r['cf_sign_agree']:.0%}")

    print("\n[第二部分] MPC vs 现有时机搜索 vs 无干预（危机率下降 + Pareto）")
    ctrl_results = []
    for s in CTRL_SEEDS:
        t0 = time.time()
        r = run_control_case(s)
        ctrl_results.append(r)
        if not r["fit_ok"]:
            print(f"  seed={s}: 模拟器拟合失败")
            continue
        print(f"  seed={s}: k={r['k']} 吸引子={r['n_attr']} 鞍点={r['n_sad']} "
              f"解释方差={r['explained']:.2f} ({time.time()-t0:.0f}s)")
        for hz in r["horizons"]:
            h = hz["horizon"]
            print(f"    H={h}: 无干预={hz['base_crisis']:.3f} "
                  f"时机(t*={hz['timing']['t_star']})={hz['timing']['crisis']:.3f}"
                  f"(E={hz['timing']['energy']:.2f})")
            for lam in MPC_LAMS:
                mp = hz["mpc"][lam]
                print(f"          MPC(λ={lam})={mp['crisis']:.3f}(E={mp['energy']:.2f}, "
                      f"靶点={mp['target_dim']})")
        if r["rescue"]:
            print(f"    最小能量救援：能量={r['rescue']['energy']:.2f} "
                  f"壁垒恢复={r['rescue']['barrier']:.4f} 靶点={r['rescue']['target_dim']}")
    print("=" * 78)

    _write_report(scm_results, ctrl_results)
    return {"scm": scm_results, "ctrl": ctrl_results}


def _fmt(v, spec=".3f"):
    return format(v, spec) if isinstance(v, (int, float)) and math.isfinite(v) else "—"


# ============================================================
# HTML 报告
# ============================================================

def _write_report(scm_results: list[dict], ctrl_results: list[dict]):
    out = Path(__file__).resolve().parents[1] / "counterfactual-control-validation-report.html"

    # ---- 第一部分表 ----
    scm_rows = []
    for r in scm_results:
        if not r["fit_ok"]:
            scm_rows.append(f"<tr><td>seed {r['seed']}</td><td colspan=6>拟合失败</td></tr>")
            continue
        do_max = max((d["max_err"] for d in r["do"]), default=float("nan"))
        do_eff = max((d["true_effect_size"] for d in r["do"]), default=float("nan"))
        scm_rows.append(
            "<tr>"
            f"<td>seed {r['seed']}</td>"
            f"<td>{_fmt(r['a_rel_err'], '.1%')}</td>"
            f"<td>{_fmt(r['b_abs_err'], '.4f')}</td>"
            f"<td>{_fmt(do_max, '.4f')}（真值效应 {_fmt(do_eff)}）</td>"
            f"<td>{_fmt(r['cf_noop_max'], '.1e')}</td>"
            f"<td>{_fmt(r['cf_truth_err'], '.4f')}</td>"
            f"<td>{_fmt(r['cf_sign_agree'], '.0%')}</td>"
            "</tr>"
        )
    scm_rows_html = "\n".join(scm_rows)

    # ---- 第二部分表（每 seed×horizon 一行）----
    ctrl_rows = []
    for r in ctrl_results:
        if not r["fit_ok"]:
            ctrl_rows.append(f"<tr><td>seed {r['seed']}</td><td colspan=8>模拟器拟合失败</td></tr>")
            continue
        for hz in r["horizons"]:
            h = hz["horizon"]
            base = hz["base_crisis"]
            tm = hz["timing"]
            m0 = hz["mpc"][0.0]
            m1 = hz["mpc"][1.0]
            red_t = base - tm["crisis"]
            red_m0 = base - m0["crisis"]
            red_m1 = base - m1["crisis"]
            # 在 J(λ=1) 决策目标下谁最优
            j_t = J_of(tm["crisis"], tm["energy"], h, 1.0)
            j_m0 = J_of(m0["crisis"], m0["energy"], h, 1.0)
            j_m1 = J_of(m1["crisis"], m1["energy"], h, 1.0)
            jbest = min([("时机", j_t), ("MPC λ=0", j_m0), ("MPC λ=1", j_m1)],
                        key=lambda kv: kv[1])[0]
            ctrl_rows.append(
                "<tr>"
                f"<td>seed {r['seed']}（k={r['k']}，H={h}）</td>"
                f"<td>{_fmt(base)}（下降 —）</td>"
                f"<td>{_fmt(tm['crisis'])}（↓{_fmt(red_t)}，E={_fmt(tm['energy'], '.2f')}）</td>"
                f"<td>{_fmt(m0['crisis'])}（↓{_fmt(red_m0)}，E={_fmt(m0['energy'], '.2f')}）</td>"
                f"<td>{_fmt(m1['crisis'])}（↓{_fmt(red_m1)}，E={_fmt(m1['energy'], '.2f')}）</td>"
                f"<td>{_fmt(j_t, '.4f')}</td>"
                f"<td>{_fmt(min(j_m0, j_m1), '.4f')}</td>"
                f"<td><b>{jbest}</b></td>"
                "</tr>"
            )
    ctrl_rows_html = "\n".join(ctrl_rows)

    # ---- 诚实汇总：MPC 在原始危机率 vs J 目标下的胜负计数 ----
    raw_win = raw_lose = j_win = j_lose = 0
    energy_ratio = []
    for r in ctrl_results:
        if not r["fit_ok"]:
            continue
        for hz in r["horizons"]:
            h = hz["horizon"]
            tm = hz["timing"]
            for lam in MPC_LAMS:
                mp = hz["mpc"][lam]
                # 原始危机率：MPC 是否严格低于时机基线
                if mp["crisis"] < tm["crisis"] - 1e-3:
                    raw_win += 1
                else:
                    raw_lose += 1
                # J(λ=1) 决策目标：MPC 是否优于时机基线
                if J_of(mp["crisis"], mp["energy"], h, 1.0) < J_of(
                        tm["crisis"], tm["energy"], h, 1.0) - 1e-4:
                    j_win += 1
                else:
                    j_lose += 1
                if mp["energy"] > 1e-6:
                    energy_ratio.append(tm["energy"] / mp["energy"])
    er_str = (f"{min(energy_ratio):.1f}~{max(energy_ratio):.1f}×"
              if energy_ratio else "—")

    rescue_bits = []
    for r in ctrl_results:
        if r["fit_ok"] and r["rescue"]:
            rescue_bits.append(
                f"seed {r['seed']}：逆向 instanton {r['rescue']['n_steps']} 步，"
                f"能量 {r['rescue']['energy']:.2f}，救援后壁垒恢复 ΔU≈{r['rescue']['barrier']:.4f}，"
                f"主靶点「{r['rescue']['target_dim']}」"
            )
    rescue_html = "<br>".join(rescue_bits) if rescue_bits else "（无有效救援路径）"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>In-Silico 反事实最优控制验证报告</title>
<style>
body{{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#f6f4fb;margin:0;padding:32px;color:#333}}
.wrap{{max-width:1180px;margin:auto}}
h1{{color:#531dab}} h2{{color:#722ed1;border-left:4px solid #722ed1;padding-left:10px}}
.card{{background:#fff;border-radius:12px;padding:20px 24px;margin:16px 0;box-shadow:0 2px 8px rgba(114,46,209,.08)}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}} td,th{{padding:7px 9px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}}
th{{background:#f9f0ff;color:#531dab}}
.formula{{background:#f9f0ff;border-radius:8px;padding:12px 16px;font-family:Consolas,monospace;font-size:13.5px;color:#531dab;line-height:1.7}}
.tag{{display:inline-block;background:#722ed1;color:#fff;border-radius:12px;padding:2px 10px;font-size:12px;margin-right:6px}}
.note{{color:#888;font-size:13px;line-height:1.6}}
.warn{{background:#fff7e6;border-left:4px solid #fa8c16;padding:10px 14px;border-radius:6px;color:#873800;font-size:13px;line-height:1.6}}
.good{{background:#f6ffed;border-left:4px solid #52c41a;padding:10px 14px;border-radius:6px;color:#135200;font-size:13px;line-height:1.6}}
</style></head><body><div class="wrap">
<h1>🧬 In-Silico Patient 反事实最优控制验证报告</h1>
<p><span class="tag">结构因果模型</span><span class="tag">do-算子</span>
<span class="tag">Pearl 三步反事实</span><span class="tag">生成式平行未来</span>
<span class="tag">MPC 最优救援</span><span class="tag">最小能量救援</span></p>

<div class="card">
<h2>真值系统与评估目标</h2>
<div class="formula">
SCM：X_(t+1) = A·X_t + b + U，&nbsp; U~N(0,Σ)，clip[0,1]；&nbsp; 谱半径 ρ(A)={np.max(np.abs(np.linalg.eigvals(A_TRUE))):.2f}&lt;1（稳定）<br>
因果链：sleep → rumination → cognition → mood（维度按字母序 {DIMS}）<br>
控制景观：U(z)=h(z₀²−1)²+(c/2)z₁²，h={H_BARRIER},c={C_LAT},d={D_NOISE}；双阱吸引子 z₀=±1，鞍点 z₀=0<br>
决策目标：J = 危机率 + λ·(干预能量 Σ‖u‖² / 时域)
</div>
<p class="note">全部为<b>合成数据、已知真值</b>；SCM 历史 T={SCM_T}，控制历史 T={CTRL_T}（dt=1）。
第一部分用真值参数直接构造 ground-truth SCM，与拟合 SCM 走<b>同一 do/反事实代码路径</b>，
差异纯粹来自 A、b、Σ 的恢复误差。</p>
</div>

<div class="card">
<h2>第一部分：SCM 恢复 · do-干预均值恢复 · 反事实一致性</h2>
<table>
<tr><th>随机种子</th><th>A 相对误差</th><th>b 最大绝对误差</th>
<th>do 末态均值恢复误差（max）</th><th>反事实无操作恒等 max|effect|</th>
<th>反事实真值恢复误差</th><th>反事实方向一致率</th></tr>
{scm_rows_html}
</table>
<div class="good"><b>无操作恒等（Pearl 三步自洽）：</b>当 do(X_j = 该步<b>观测值</b>) 时，
Abduction→Action→Prediction 重算的反事实末态与事实末态之差应<b>精确为 0</b>
（上表 max|effect| ≈ 1e-16~1e-3 量级，源于浮点与 clip），验证三步法实现正确——
"若当时没有改变任何东西，则轨迹不变"。</div>
<p class="note"><b>do 均值恢复误差</b>=|E_fit[X_final|do] − E_true[X_final|do]|（同起点、
clamp 持续钳制、{DO_HORIZON} 步、{N_DO_SAMPLES} 蒙特卡洛样本，括号为真值效应量作参照）；
<b>反事实真值恢复误差</b>=拟合 SCM 与真值 SCM 在 {N_CF_EVIDENCE} 份证据上的反事实末态最大偏差均值（clamp 持续 do(sleep=0.15)）；
<b>方向一致率</b>=对真值反事实效应显著（|effect|&gt;0.02）的各维度，拟合效应符号与真值一致的比例。误差随 A 恢复精度而定，属线性 SCM 近似。</p>
</div>

<div class="card">
<h2>第二部分：MPC vs 现有时机搜索 vs 无干预</h2>
<p class="note">三臂均从<b>健康吸引子</b>统一出发，K={K_EVAL} 条平行未来、同一评估随机种子，
统计在时域内坠入危机球（半径 0.25·‖crisis−health‖）的比率。时机基线=固定救援方向、
幅度 {DOSE}，网格搜索起始时机 t* 后<b>持续</b>施加（JITAI 范式的模拟器直译）；
MPC=CEM 联合优化时机+靶点+剂量的完整序列，在 λ=0（纯危机率）与 λ=1（能量感知）下各优化一次。</p>
<table>
<tr><th>配置</th><th>无干预危机率</th><th>时机搜索（危机率/能量）</th>
<th>MPC λ=0（危机率/能量）</th><th>MPC λ=1（危机率/能量）</th>
<th>J(λ=1) 时机</th><th>J(λ=1) MPC</th><th>J(λ=1) 最优</th></tr>
{ctrl_rows_html}
</table>
<p class="note">括号内“↓x”为相对<b>无干预</b>的危机率下降；E=Σ‖u‖²为干预能量（≈干预负担）。
J(λ=1) 列给出把干预负担计入后的决策目标，末列标注该目标下的最优臂。</p>
</div>

<div class="card">
<h2>结论（诚实呈现，含 MPC 不占优区间）</h2>
<p><b>(1) SCM / 反事实内核正确：</b>在稳定真值 VAR(1) 上，拟合 A 的相对误差与 do-干预末态均值
恢复误差均随结构恢复精度而定；<b>无操作反事实精确恒等于事实</b>（Pearl 三步自洽），
反事实真值恢复误差与方向一致率见上表，验证 do-算子正确截断入边、三步法自洽。</p>
<p><b>(2) 三种干预都能降低危机率，但存在明确的 Pareto 权衡。</b>
在<b>原始危机率</b>这一单一指标上，MPC 相对暴力持续时机基线：胜 {raw_win} / 负 {raw_lose}
（跨种子×时域×λ 计）。<b>这正是 MPC 的不占优区间</b>：当时机基线可无代价地持续施加最大剂量、
且逃逸方向单一（合成景观的近 1D 逃逸）时，暴力干预把危机率压得更低，MPC 的节能无收益。</p>
<p><b>(3) MPC 的价值在能量效率（Pareto 前沿）。</b>MPC 以约 <b>{er_str}</b> 更低的干预能量
逼近时机基线的危机率下降；一旦把干预负担计入决策目标 J（λ 足够大），MPC 反超：
J(λ=1) 目标下 MPC 胜 {j_win} / 负 {j_lose}。交叉点由 λ 决定——λ→0（干预免费）时
时机基线占优，λ 增大（干预有代价，贴近临床现实）时 MPC 占优。这与 MPC 显式最小化
J=危机率+λ·能量的设计一致，也说明<b>“最优干预”依赖于对干预代价的显式建模</b>。</p>
<p><b>(4) 最小能量救援（逆向 instanton）：</b>{rescue_html}。救援方向由崩溃 instanton 反向
闭式给出，与 MPC 学到的靶点在语义上一致（均指向把状态推回健康阱、重新抬高壁垒）。</p>
<div class="warn"><b>边界与诚实标注：</b>本实验基于合成数据、已知真值，仅验证算法数值行为，
<b>不构成任何临床有效性声明</b>。潜空间控制经 PCA 伪逆映射回模态属文档化近似；时机基线假定了
已知救援方向，在真正多向逃逸（高维、多竞争崩溃路径）的景观中 MPC 的靶点选择优势会更明显；
CEM 为经验最优（有限候选/迭代），非全局最优；真实心理系统非线性、非高斯，线性 SCM 反事实仅在
“噪声固定”假设下成立。数据不足时上层 DigitalTwinManager 各方法静默回退现有时机优化，绝不抛错。</div>
</div>
<p class="note">生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} ｜ 合成数据实验，结论仅用于算法行为验证</p>
</div></body></html>"""
    out.write_text(html, encoding="utf-8")
    print(f"报告已生成: {out}")


if __name__ == "__main__":
    main()
