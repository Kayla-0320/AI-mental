"""从第二轮真实结果 JSON 生成可复核的静态研究图。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def create(root: Path) -> None:
    """展示全臂结果、覆盖率一致的质量对照和参与者等权预测误差。"""
    destination = root / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    model = json.loads((root / "model/ch_sims_v1/experiment.json").read_text(encoding="utf-8"))
    keys = ["text", "audio", "visual", "concat", "concat_knowledge", "moe", "moe_knowledge"]
    labels = ["Text", "Audio", "Visual", "Concat", "Concat +\ncandidate KG", "MoE", "MoE +\ncandidate KG"]
    values = [model["arms"][name]["conditions"]["clean"]["macro_f1"] for name in keys]
    fig, ax = plt.subplots(figsize=(10.8, 4.5), layout="constrained")
    bars = ax.bar(labels, values, color=["#9EAABC"] * 3 + ["#346F9D", "#AB9CBE", "#608A78", "#AB9CBE"], width=.64)
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set(ylabel="Macro-F1 (higher is better)", ylim=(0, .74),
           title="Source-video holdout: learned KG/MoE did not outperform concatenation")
    ax.text(.01, .97, "472 test clips / 12 source groups; seed 42; candidate KG is not expert reviewed",
            transform=ax.transAxes, va="top", fontsize=9)
    fig.savefig(destination / "model_ablation.png", dpi=180)
    fig.savefig(destination / "model_ablation.svg")
    plt.close(fig)
    quality = json.loads((root / "data/quality_benchmark.json").read_text(encoding="utf-8"))
    scenarios = ["clean", "audio_missing_50pct", "visual_stale_50pct", "audio_misaligned_50pct", "audio_nonfinite_50pct"]
    names = ["Clean", "Audio absent", "Visual stale", "Audio misaligned", "Audio non-finite"]
    fig, ax = plt.subplots(figsize=(10.8, 4.5), layout="constrained")
    x = np.arange(len(names))
    for offset, key, label, color in [(-.18, "ignoring_fault_metadata", "Ignore fault metadata", "#A5AFBA"),
                                      (.18, "quality_aware", "Use quality masks", "#346F9D")]:
        scores = [quality["scenarios"][name][key]["macro_f1"] for name in scenarios]
        bars = ax.bar(x + offset, scores, width=.34, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=names, ylabel="Macro-F1 (higher is better)", ylim=(0, .66),
           title="Known injected faults: quality masks help some conditions, not all")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.savefig(destination / "quality_conditions.png", dpi=180)
    fig.savefig(destination / "quality_conditions.svg")
    plt.close(fig)
    forecast = json.loads((root / "forecast/metrics.json").read_text(encoding="utf-8"))
    methods = ["persistence", "past_mean", "ema", "ridge", "gru"]
    names = ["Last value", "Past mean", "EMA", "Ridge", "GRU"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), layout="constrained", sharey=True)
    for ax, horizon in zip(axes, ("1", "3")):
        record = forecast["horizons"][horizon]
        scores = [record["test"][method]["participant_equal_mae"] for method in methods]
        colors = ["#346F9D" if method == record["selected_method"] else "#A5AFBA" for method in methods]
        bars = ax.bar(names, scores, color=colors)
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set(title=f"Horizon: {horizon} sampling step(s)", ylim=(0, 18),
               xlabel="Blue: selected using validation only")
        ax.tick_params(axis="x", labelrotation=20)
    axes[0].set_ylabel("Participant-equal MAE, 0-100 (lower is better)")
    fig.suptitle("Real adolescent self-reports: 9 held-out people; GRU did not beat simple baselines")
    fig.savefig(destination / "forecast_comparison.png", dpi=180)
    fig.savefig(destination / "forecast_comparison.svg")
    plt.close(fig)
    print("created 3 research figures from recorded results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    create(parser.parse_args().results)
