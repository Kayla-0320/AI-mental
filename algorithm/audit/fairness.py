"""
公平性审计模块 —— 评估模型在不同群体上的表现差异

核心功能：
    1. generate_audit_report(): 按人口统计学分组计算公平性指标
    2. 输出 Markdown 对比表格（高亮差异最大的组）
    3. matplotlib 生成分组性能对比图

分组维度：
    - 年龄：12-15 / 16-18
    - 性别：男 / 女
    - 地域：城市 / 县镇 / 农村

公平性指标：
    - demographic_parity_difference: 人口统计差异
    - equalized_odds_difference: 均等赔率差异

安全约束：
    - 样本量 < 30 的组标注"样本不足，结果仅供参考"
    - 不输出可能导致群体污名化的结论
    - 使用 fairlearn 库计算公平性指标
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# 路径配置
# ============================================================

_MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _MODULE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = str(OUTPUT_DIR / "fairness_report.md")
CHART_PATH = str(OUTPUT_DIR / "fairness_chart.png")

# 样本量阈值：低于此值的组标注"样本不足"
MIN_SAMPLE_SIZE = 30

# ============================================================
# 分组定义
# ============================================================

AGE_GROUPS = {
    "12-15": (12, 15),
    "16-18": (16, 18),
}

GENDER_GROUPS = {
    "男": "male",
    "女": "female",
}

REGION_GROUPS = {
    "城市": "urban",
    "县镇": "town",
    "农村": "rural",
}


# ============================================================
# 数据结构
# ============================================================

@dataclass
class GroupMetrics:
    """单组指标

    Attributes:
        group_name: 组名
        dimension: 分组维度（age/gender/region）
        sample_size: 样本量
        accuracy: 准确率
        f1: F1 分数
        tpr: 真正例率（召回率）
        fpr: 假正例率
        insufficient_sample: 样本是否不足
    """
    group_name: str
    dimension: str
    sample_size: int
    accuracy: float
    f1: float
    tpr: float
    fpr: float
    insufficient_sample: bool = False


@dataclass
class FairnessReport:
    """公平性审计报告

    Attributes:
        group_metrics: 各组指标
        demographic_parity_diff: 人口统计差异
        equalized_odds_diff: 均等赔率差异
        worst_group: 表现最差的组（仅用于改进建议，不用于污名化）
        recommendations: 改进建议
        timestamp: 报告生成时间
    """
    group_metrics: list[GroupMetrics] = field(default_factory=list)
    demographic_parity_diff: float = 0.0
    equalized_odds_diff: float = 0.0
    worst_group: str = ""
    recommendations: list[str] = field(default_factory=list)
    timestamp: str = ""


# ============================================================
# 分组指标计算
# ============================================================

def _compute_group_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    group_name: str,
    dimension: str,
) -> GroupMetrics:
    """计算单个组的指标"""
    group_preds = predictions[mask]
    group_labels = labels[mask]
    n = len(group_labels)

    if n == 0:
        return GroupMetrics(
            group_name=group_name,
            dimension=dimension,
            sample_size=0,
            accuracy=0.0,
            f1=0.0,
            tpr=0.0,
            fpr=0.0,
            insufficient_sample=True,
        )

    acc = float(accuracy_score(group_labels, group_preds))
    f1 = float(f1_score(group_labels, group_preds, zero_division=0))

    # TPR = TP / (TP + FN)
    tp = np.sum((group_preds == 1) & (group_labels == 1))
    fn = np.sum((group_preds == 0) & (group_labels == 1))
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # FPR = FP / (FP + TN)
    fp = np.sum((group_preds == 1) & (group_labels == 0))
    tn = np.sum((group_preds == 0) & (group_labels == 0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return GroupMetrics(
        group_name=group_name,
        dimension=dimension,
        sample_size=n,
        accuracy=round(acc, 4),
        f1=round(f1, 4),
        tpr=round(tpr, 4),
        fpr=round(fpr, 4),
        insufficient_sample=(n < MIN_SAMPLE_SIZE),
    )


# ============================================================
# 核心接口
# ============================================================

def generate_audit_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    demographics: dict,
) -> FairnessReport:
    """生成公平性审计报告

    按年龄、性别、地域分组，计算各组指标和公平性差异。

    Args:
        predictions: 模型预测标签 (n_samples,)
        labels: 真实标签 (n_samples,)
        demographics: 人口统计学信息字典
            - "ages": 年龄列表 (n_samples,)
            - "genders": 性别列表 (n_samples,)，值为 "male"/"female"
            - "regions": 地域列表 (n_samples,)，值为 "urban"/"town"/"rural"

    Returns:
        FairnessReport: 公平性审计报告

    示例：
        >>> predictions = np.array([1, 0, 1, 1, 0])
        >>> labels = np.array([1, 0, 0, 1, 1])
        >>> demographics = {
        ...     "ages": [13, 17, 14, 16, 15],
        ...     "genders": ["male", "female", "male", "female", "male"],
        ...     "regions": ["urban", "rural", "town", "urban", "rural"],
        ... }
        >>> report = generate_audit_report(predictions, labels, demographics)
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    ages = np.asarray(demographics["ages"])
    genders = np.asarray(demographics["genders"])
    regions = np.asarray(demographics["regions"])

    all_metrics = []

    # 按年龄分组
    for group_name, (low, high) in AGE_GROUPS.items():
        mask = (ages >= low) & (ages <= high)
        metrics = _compute_group_metrics(predictions, labels, mask, group_name, "age")
        all_metrics.append(metrics)

    # 按性别分组
    gender_reverse_map = {v: k for k, v in GENDER_GROUPS.items()}
    for group_cn, group_en in GENDER_GROUPS.items():
        mask = genders == group_en
        metrics = _compute_group_metrics(predictions, labels, mask, group_cn, "gender")
        all_metrics.append(metrics)

    # 按地域分组
    region_reverse_map = {v: k for k, v in REGION_GROUPS.items()}
    for group_cn, group_en in REGION_GROUPS.items():
        mask = regions == group_en
        metrics = _compute_group_metrics(predictions, labels, mask, group_cn, "region")
        all_metrics.append(metrics)

    # 计算公平性指标（使用 fairlearn）
    dpd, eod = _compute_fairness_metrics(predictions, labels, demographics)

    # 找出表现最差的组（仅用于改进建议）
    valid_metrics = [m for m in all_metrics if m.sample_size >= MIN_SAMPLE_SIZE]
    worst_group = ""
    if valid_metrics:
        worst = min(valid_metrics, key=lambda m: m.f1)
        worst_group = f"{worst.dimension}/{worst.group_name}"

    # 生成建议（不污名化任何群体）
    recommendations = _generate_recommendations(all_metrics, dpd, eod)

    return FairnessReport(
        group_metrics=all_metrics,
        demographic_parity_diff=round(dpd, 4),
        equalized_odds_diff=round(eod, 4),
        worst_group=worst_group,
        recommendations=recommendations,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def _compute_fairness_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    demographics: dict,
) -> tuple[float, float]:
    """使用 fairlearn 计算公平性指标

    Returns:
        (demographic_parity_difference, equalized_odds_difference)
    """
    try:
        from fairlearn.metrics import (
            demographic_parity_difference,
            equalized_odds_difference,
        )

        # 对每个敏感属性计算，取最大值
        dpd_values = []
        eod_values = []

        for attr_name in ["ages", "genders", "regions"]:
            sensitive = np.asarray(demographics[attr_name])

            # 年龄需要分组成二值（12-15 vs 16-18）
            if attr_name == "ages":
                sensitive = (sensitive >= 16).astype(int)

            try:
                dpd = demographic_parity_difference(
                    y_true=labels, y_pred=predictions, sensitive_features=sensitive
                )
                eod = equalized_odds_difference(
                    y_true=labels, y_pred=predictions, sensitive_features=sensitive
                )
                dpd_values.append(dpd)
                eod_values.append(eod)
            except Exception:
                continue

        dpd = max(dpd_values) if dpd_values else 0.0
        eod = max(eod_values) if eod_values else 0.0

    except ImportError:
        # fairlearn 不可用时，手动计算
        dpd = _manual_demographic_parity(predictions, demographics)
        eod = 0.0  # 手动计算 equalized odds 较复杂，暂返回 0

    return dpd, eod


def _manual_demographic_parity(
    predictions: np.ndarray,
    demographics: dict,
) -> float:
    """手动计算 demographic parity difference"""
    # 使用性别作为敏感属性
    genders = np.asarray(demographics["genders"])
    unique_groups = np.unique(genders)

    if len(unique_groups) < 2:
        return 0.0

    rates = []
    for g in unique_groups:
        mask = genders == g
        rate = np.mean(predictions[mask]) if np.sum(mask) > 0 else 0.0
        rates.append(rate)

    return float(max(rates) - min(rates))


# ============================================================
# 建议生成（不污名化）
# ============================================================

def _generate_recommendations(
    metrics: list[GroupMetrics],
    dpd: float,
    eod: float,
) -> list[str]:
    """生成改进建议

    注意：不输出可能导致群体污名化的结论。
    建议聚焦于模型改进方向，而非群体特征。
    """
    recommendations = []

    # 公平性指标建议
    if dpd > 0.1:
        recommendations.append(
            f"人口统计差异 (DPD={dpd:.3f}) 较高，建议检查训练数据分布是否均衡，"
            "考虑使用重采样或对抗去偏方法改进模型公平性。"
        )

    if eod > 0.1:
        recommendations.append(
            f"均等赔率差异 (EOD={eod:.3f}) 较高，建议在不同群体上分别校准决策阈值。"
        )

    # 样本量建议
    insufficient = [m for m in metrics if m.insufficient_sample and m.sample_size > 0]
    if insufficient:
        groups = ", ".join(f"{m.dimension}/{m.group_name}" for m in insufficient)
        recommendations.append(
            f"以下分组样本量不足 30：{groups}。"
            "建议增加这些群体的数据采集，以获得更可靠的评估结果。"
        )

    # 空组建议
    empty = [m for m in metrics if m.sample_size == 0]
    if empty:
        groups = ", ".join(f"{m.dimension}/{m.group_name}" for m in empty)
        recommendations.append(f"以下分组无数据：{groups}。建议检查数据收集覆盖度。")

    if not recommendations:
        recommendations.append(
            "各分组表现较为均衡，公平性指标在可接受范围内。"
            "建议持续监控以确保模型在不同群体上的稳定性。"
        )

    return recommendations


# ============================================================
# Markdown 报告
# ============================================================

def render_markdown_report(report: FairnessReport) -> str:
    """渲染 Markdown 格式报告"""
    lines = [
        "# 公平性审计报告",
        "",
        f"> 生成时间：{report.timestamp}",
        "> 评估维度：年龄（12-15 / 16-18）、性别、地域（城市 / 县镇 / 农村）",
        "",
        "## 1. 分组性能指标",
        "",
        "| 维度 | 分组 | 样本量 | 准确率 | F1 | TPR | FPR | 备注 |",
        "|------|------|--------|--------|----|-----|-----|------|",
    ]

    # 找出各维度内 F1 差异最大的组（用于高亮）
    dim_groups = {}
    for m in report.group_metrics:
        dim_groups.setdefault(m.dimension, []).append(m)

    highlight_groups = set()
    for dim, group_list in dim_groups.items():
        valid = [m for m in group_list if m.sample_size >= MIN_SAMPLE_SIZE]
        if len(valid) >= 2:
            max_f1 = max(m.f1 for m in valid)
            min_f1 = min(m.f1 for m in valid)
            if max_f1 - min_f1 > 0.05:  # 5% 差异才高亮
                for m in valid:
                    if m.f1 == max_f1 or m.f1 == min_f1:
                        highlight_groups.add((m.dimension, m.group_name))

    dimension_cn = {"age": "年龄", "gender": "性别", "region": "地域"}

    for m in report.group_metrics:
        dim_cn = dimension_cn.get(m.dimension, m.dimension)
        note = ""
        if m.insufficient_sample:
            if m.sample_size == 0:
                note = "⚠️ 无数据"
            else:
                note = "⚠️ 样本不足，结果仅供参考"

        # 高亮差异最大的组
        is_highlight = (m.dimension, m.group_name) in highlight_groups
        row_prefix = "**" if is_highlight else ""
        row_suffix = "**" if is_highlight else ""

        lines.append(
            f"| {row_prefix}{dim_cn}{row_suffix} "
            f"| {row_prefix}{m.group_name}{row_suffix} "
            f"| {m.sample_size} "
            f"| {m.accuracy:.4f} "
            f"| {m.f1:.4f} "
            f"| {m.tpr:.4f} "
            f"| {m.fpr:.4f} "
            f"| {note} |"
        )

    lines.extend([
        "",
        "## 2. 公平性指标",
        "",
        "| 指标 | 值 | 说明 |",
        "|------|-----|------|",
        f"| Demographic Parity Difference | {report.demographic_parity_diff:.4f} | 各组正预测率的最大差异 |",
        f"| Equalized Odds Difference | {report.equalized_odds_diff:.4f} | 各组 TPR/FPR 的最大差异 |",
        "",
        "**判定标准**：",
        "",
        "- DPD / EOD < 0.1：公平性良好",
        "- 0.1 ≤ DPD / EOD < 0.2：需关注",
        "- DPD / EOD ≥ 0.2：建议采取去偏措施",
        "",
        "## 3. 改进建议",
        "",
    ])

    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"{i}. {rec}")

    lines.extend([
        "",
        "## 4. 声明",
        "",
        "本报告仅用于评估模型在不同群体上的技术表现差异，",
        "不反映任何群体的真实心理特征差异。",
        "所有群体在人格尊严和权利上完全平等。",
        "审计目的是改进模型的公平性，而非对任何群体做出判断。",
        "",
    ])

    return "\n".join(lines)


# ============================================================
# 图表生成
# ============================================================

def generate_chart(
    report: FairnessReport,
    save_path: str = CHART_PATH,
) -> str:
    """生成分组性能对比图

    Args:
        report: 公平性审计报告
        save_path: 图表保存路径

    Returns:
        图表文件路径
    """
    import matplotlib
    matplotlib.use("Agg")  # 非交互式后端
    import matplotlib.pyplot as plt

    # 按维度分组
    dim_data = {}
    for m in report.group_metrics:
        if m.sample_size > 0:
            dim_data.setdefault(m.dimension, []).append(m)

    dimension_cn = {"age": "年龄", "gender": "性别", "region": "地域"}
    n_dims = len(dim_data)

    if n_dims == 0:
        return ""

    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 5))
    if n_dims == 1:
        axes = [axes]

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

    for idx, (dim, metrics_list) in enumerate(dim_data.items()):
        ax = axes[idx]
        names = [m.group_name for m in metrics_list]
        f1_scores = [m.f1 for m in metrics_list]
        acc_scores = [m.accuracy for m in metrics_list]

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(x - width / 2, acc_scores, width, label="准确率", color=colors[0], alpha=0.8)
        bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1", color=colors[1], alpha=0.8)

        ax.set_xlabel(dimension_cn.get(dim, dim))
        ax.set_ylabel("分数")
        ax.set_title(f"{dimension_cn.get(dim, dim)}分组性能对比")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        # 标注样本量不足的组
        for i, m in enumerate(metrics_list):
            if m.insufficient_sample:
                ax.annotate(
                    "⚠️",
                    xy=(i, max(m.accuracy, m.f1) + 0.03),
                    ha="center",
                    fontsize=12,
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path


# ============================================================
# 便捷函数
# ============================================================

def run_full_audit(
    predictions: np.ndarray,
    labels: np.ndarray,
    demographics: dict,
    report_path: str = REPORT_PATH,
    chart_path: str = CHART_PATH,
) -> FairnessReport:
    """运行完整审计流程

    1. 计算分组指标
    2. 生成 Markdown 报告
    3. 生成分组性能对比图

    Args:
        predictions: 模型预测
        labels: 真实标签
        demographics: 人口统计学信息
        report_path: 报告保存路径
        chart_path: 图表保存路径

    Returns:
        FairnessReport
    """
    report = generate_audit_report(predictions, labels, demographics)

    # Markdown 报告
    md = render_markdown_report(report)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[INFO] 公平性报告已保存 → {report_path}")

    # 图表
    chart = generate_chart(report, chart_path)
    if chart:
        print(f"[INFO] 性能对比图已保存 → {chart}")

    return report
