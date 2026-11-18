"""从第三轮真实结果 JSON 生成可复核的静态研究图。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = ["text", "audio", "visual", "concat", "concat_knowledge", "moe",
        "moe_knowledge", "sparse", "sparse_knowledge"]
LABELS = ["Text", "Audio", "Visual", "Concat", "Concat +\nKG", "MoE",
          "MoE +\nKG", "Sparse\nMoE", "Sparse\nMoE + KG"]
COLORS = ["#9EAABC"] * 3 + ["#346F9D", "#AB9CBE", "#608A78", "#AB9CBE", "#C0803C", "#C0803C"]


def create(root: Path) -> None:
    """展示九臂最终测试、与第一轮同一任务的可比臂以及真实稀疏分派成本。"""
    destination = root / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    ablation = root / "model/ch_sims_v2_ablation"
    experiment = json.loads((ablation / "experiment.json").read_text(encoding="utf-8"))

    values = [experiment["arms"][arm]["conditions"]["clean"]["macro_f1"] for arm in ARMS]
    figure, axis = plt.subplots(figsize=(11.4, 4.6), layout="constrained")
    bars = axis.bar(LABELS, values, color=COLORS, width=.66)
    axis.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    axis.set(ylabel="Macro-F1 (higher is better)", ylim=(0, .72),
             title="Full 4,400-clip CH-SIMS v2 holdout: MoE nominally above concat, knowledge arms below")
    delta = experiment["group_bootstrap_f1"]["moe_minus_concat"]
    axis.text(.01, .96, "821 test clips / 30 source groups; frozen Chinese-CLIP + real audio features\n"
                        f"MoE - concat = {delta['difference']:+.4f}, grouped bootstrap 95% CI "
                        f"[{delta['ci95'][0]:+.4f}, {delta['ci95'][1]:+.4f}] (crosses zero)",
              transform=axis.transAxes, va="top", fontsize=8.5)
    figure.savefig(destination / "vlm_ablation_v2.png", dpi=180)
    figure.savefig(destination / "vlm_ablation_v2.svg")
    plt.close(figure)

    round2 = root.parent / "round2/model/ch_sims_v1/experiment.json"
    figure, axis = plt.subplots(figsize=(9.6, 4.4), layout="constrained")
    shared = ["text", "audio", "visual", "concat", "moe"]
    names = ["Text", "Audio", "Visual", "Concat", "MoE"]
    small = json.loads(round2.read_text(encoding="utf-8"))["arms"]
    x = np.arange(len(shared))
    for offset, source, label, color in [(-.19, small, "v1 author features, 472 test clips", "#A5AFBA"),
                                         (.19, experiment["arms"], "v2 full media, 821 test clips", "#346F9D")]:
        scores = [source[arm]["conditions"]["clean"]["macro_f1"] for arm in shared]
        bars = axis.bar(x + offset, scores, width=.36, label=label, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5)
    axis.set(xticks=x, xticklabels=names, ylabel="Macro-F1 (higher is better)", ylim=(0, .74),
             title="Same task, different holdouts: v2 full media is not uniformly better, and the MoE gap stays within noise")
    axis.legend(loc="upper left", frameon=False)
    figure.savefig(destination / "scale_comparison.png", dpi=180)
    figure.savefig(destination / "scale_comparison.svg")
    plt.close(figure)

    latency = json.loads((ablation / "sparse_latency_and_text_equivalence.json").read_text(encoding="utf-8"))
    batches = sorted(latency["batches"], key=int)
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), layout="constrained")
    x = np.arange(len(batches))
    sparse_rows = [latency["batches"][b]["sparse_expert_row_evaluations"] for b in batches]
    dense_rows = [latency["batches"][b]["dense_expert_row_evaluations"] for b in batches]
    axes[0].bar(x - .19, dense_rows, width=.36, label="Dense top-2 reference", color="#A5AFBA")
    axes[0].bar(x + .19, sparse_rows, width=.36, label="Actual sparse dispatch", color="#346F9D")
    axes[0].set(xticks=x, xticklabels=[f"batch {b}" for b in batches], ylabel="Expert row evaluations",
                title="Sparse dispatch really skips experts")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[0].bar_label(axes[0].containers[0], fmt="%d", padding=2, fontsize=8)
    axes[0].bar_label(axes[0].containers[1], fmt="%d", padding=2, fontsize=8)
    dense_ms = [latency["batches"][b]["timings"]["dense_reference"]["p50_ms"] for b in batches]
    sparse_ms = [latency["batches"][b]["timings"]["actual_sparse"]["p50_ms"] for b in batches]
    axes[1].bar(x - .19, dense_ms, width=.36, label="Dense top-2 reference", color="#A5AFBA")
    axes[1].bar(x + .19, sparse_ms, width=.36, label="Actual sparse dispatch", color="#346F9D")
    axes[1].set(xticks=x, xticklabels=[f"batch {b}" for b in batches], ylabel="Fusion-head P50 (ms, CPU 1 thread)",
                title="Fewer evaluations, no wall-clock win at this head size")
    axes[1].legend(frameon=False, fontsize=8.5)
    figure.suptitle("Sparse MoE head: dispatch is real, speed-up is not demonstrated", fontsize=11)
    figure.savefig(destination / "sparse_dispatch.png", dpi=180)
    figure.savefig(destination / "sparse_dispatch.svg")
    plt.close(figure)

    forecast = json.loads((root / "data/two_wave_prediction.json").read_text(encoding="utf-8"))
    methods = ["train_mean", "persistence_wave1", "ridge_baseline_features"]
    method_names = ["Train mean", "Keep wave-1 value", "Ridge:\nwave-1 + baseline"]
    targets = ["wave2_g", "wave2_h"]
    target_names = ["g total (9 items)", "h total (7 items)"]
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), layout="constrained")
    x = np.arange(len(targets))
    for offset, method, label, color in [(-.26, methods[0], method_names[0], "#A5AFBA"),
                                         (0.0, methods[1], method_names[1], "#8FB0CC"),
                                         (.26, methods[2], method_names[2], "#346F9D")]:
        scores = [forecast["targets"][target][method]["mae"] for target in targets]
        bars = axes[0].bar(x + offset, scores, width=.24, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    axes[0].set(xticks=x, xticklabels=target_names, ylabel="Test MAE (lower is better)", ylim=(0, 5.0),
                title="Follow-up MAE: keeping wave-1 value is already strong")
    axes[0].legend(frameon=False, fontsize=8)
    for offset, method, color in [(-.18, methods[1], "#8FB0CC"), (.18, methods[2], "#346F9D")]:
        scores = [forecast["targets"][target][method]["r2"] for target in targets]
        bars = axes[1].bar(x + offset, scores, width=.34, label=method, color=color)
        axes[1].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axes[1].set(xticks=x, xticklabels=target_names, ylabel="Test R² (higher is better)", ylim=(0, .42),
                title="R² rises with baseline covariates")
    g_delta = forecast["targets"]["wave2_g"]["delta_vs_persistence"]
    h_delta = forecast["targets"]["wave2_h"]["delta_vs_persistence"]
    figure.suptitle("Yunnan two-wave cohort, 5,206 participants: follow-up is predictable, but no stable MAE gain over persistence",
                    fontsize=10.5)
    axes[0].text(.02, .98, f"Ridge − persistence MAE, 95% CI:\ng {g_delta['ridge_minus_persistence_mae']:+.3f} "
                           f"[{g_delta['ci95'][0]:+.3f}, {g_delta['ci95'][1]:+.3f}]\nh "
                           f"{h_delta['ridge_minus_persistence_mae']:+.3f} "
                           f"[{h_delta['ci95'][0]:+.3f}, {h_delta['ci95'][1]:+.3f}]\nitem semantics unverified",
                  transform=axes[0].transAxes, va="top", ha="left", fontsize=7.5)
    figure.savefig(destination / "longitudinal_followup.png", dpi=180)
    figure.savefig(destination / "longitudinal_followup.svg")
    plt.close(figure)
    print("created 4 round3 research figures from recorded results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    create(parser.parse_args().results)
