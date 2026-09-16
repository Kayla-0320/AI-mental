"""
融合策略评估模块

对比三种融合策略的 AUC 和 F1，输出 Markdown 对比报告。

使用方式：
    from perception.fusion.evaluate import evaluate_fusion, generate_report

    results = evaluate_fusion(text_probs, audio_probs, labels)
    report = generate_report(results)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from perception.fusion.fusion import (
    StackingFusion,
    fuse_dynamic_weighting,
    fuse_weighted_average,
    load_fusion_config,
)


# ============================================================
# 评估函数
# ============================================================

def evaluate_fusion(
    text_probs: np.ndarray,
    audio_probs: np.ndarray,
    labels: np.ndarray,
    config: Optional[dict] = None,
) -> dict:
    """对比三种融合策略的 AUC 和 F1

    Args:
        text_probs: 文本概率分布 (n_samples, n_classes)
        audio_probs: 语音概率分布 (n_samples, n_classes)
        labels: 真实标签 (n_samples,)
        config: 融合配置

    Returns:
        {
            "weighted_average": {"auc": float, "f1_macro": float, "f1_weighted": float},
            "stacking": {"auc": float, "f1_macro": float, "f1_weighted": float},
            "dynamic_weighting": {"auc": float, "f1_macro": float, "f1_weighted": float},
            "text_only": {"auc": float, "f1_macro": float, "f1_weighted": float},  # 基线
            "audio_only": {"auc": float, "f1_macro": float, "f1_weighted": float},  # 基线
        }
    """
    if config is None:
        config = load_fusion_config()

    n_classes = text_probs.shape[1]
    results = {}

    # --- 基线 1：仅文本 ---
    results["text_only"] = _compute_metrics(text_probs, labels, n_classes)

    # --- 基线 2：仅语音 ---
    results["audio_only"] = _compute_metrics(audio_probs, labels, n_classes)

    # --- 策略 1：加权平均 ---
    fused_wa = fuse_weighted_average(text_probs, audio_probs, config)
    results["weighted_average"] = _compute_metrics(fused_wa, labels, n_classes)

    # --- 策略 2：Stacking ---
    # 使用交叉验证式的分割：前半训练，后半评估
    n = len(labels)
    split = max(n // 2, 1)
    stacking = StackingFusion(config)
    stacking.fit(text_probs[:split], audio_probs[:split], labels[:split])
    fused_stacking = stacking.predict(text_probs[split:], audio_probs[split:])
    results["stacking"] = _compute_metrics(fused_stacking, labels[split:], n_classes)

    # --- 策略 3：动态加权 ---
    # 使用每样本的最大概率作为置信度
    text_confidences = text_probs.max(axis=1)
    audio_confidences = audio_probs.max(axis=1)
    fused_dynamic = np.zeros_like(text_probs)
    for i in range(n):
        fused_dynamic[i] = fuse_dynamic_weighting(
            text_probs[i], audio_probs[i],
            float(text_confidences[i]), float(audio_confidences[i]),
            config,
        )
    results["dynamic_weighting"] = _compute_metrics(fused_dynamic, labels, n_classes)

    return results


def _compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> dict:
    """计算 AUC 和 F1 指标

    Args:
        probs: 预测概率 (n_samples, n_classes)
        labels: 真实标签 (n_samples,)
        n_classes: 类别数

    Returns:
        {"auc": float, "f1_macro": float, "f1_weighted": float}
    """
    preds = probs.argmax(axis=1)

    # F1
    f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    f1_weighted = float(f1_score(labels, preds, average="weighted", zero_division=0))

    # AUC（需要 one-hot 编码）
    try:
        auc = float(roc_auc_score(
            labels, probs,
            multi_class="ovr",
            average="macro",
            labels=list(range(n_classes)),
        ))
    except ValueError:
        # 某些类别可能没有样本，AUC 无法计算
        auc = 0.0

    return {
        "auc": round(auc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
    }


# ============================================================
# Markdown 报告生成
# ============================================================

def generate_report(results: dict, save_path: Optional[str] = None) -> str:
    """生成融合策略对比 Markdown 报告

    Args:
        results: evaluate_fusion 的输出
        save_path: 报告保存路径（可选）

    Returns:
        Markdown 格式的报告字符串
    """
    lines = [
        "# 多模态融合策略对比报告",
        "",
        f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 评估指标",
        "",
        "| 策略 | AUC (macro) | F1 (macro) | F1 (weighted) |",
        "|------|-------------|------------|---------------|",
    ]

    # 按策略顺序输出
    strategy_order = [
        "text_only", "audio_only",
        "weighted_average", "stacking", "dynamic_weighting",
    ]
    strategy_names = {
        "text_only": "文本单模态（基线）",
        "audio_only": "语音单模态（基线）",
        "weighted_average": "加权平均",
        "stacking": "Stacking (LR)",
        "dynamic_weighting": "动态加权",
    }

    best_auc = 0.0
    best_f1 = 0.0
    best_strategy = ""

    for strategy in strategy_order:
        if strategy not in results:
            continue
        m = results[strategy]
        name = strategy_names.get(strategy, strategy)
        lines.append(
            f"| {name} | {m['auc']:.4f} | {m['f1_macro']:.4f} | {m['f1_weighted']:.4f} |"
        )
        # 记录最优（排除基线）
        if strategy not in ("text_only", "audio_only"):
            score = m["auc"] + m["f1_macro"]
            if score > best_auc + best_f1:
                best_auc = m["auc"]
                best_f1 = m["f1_macro"]
                best_strategy = name

    lines.extend([
        "",
        "## 结论",
        "",
        f"**最优融合策略：{best_strategy}**（AUC={best_auc:.4f}, F1={best_f1:.4f}）",
        "",
        "### 策略说明",
        "",
        "- **加权平均**：简单高效，适合实时场景，权重可通过 config/fusion.yaml 调整",
        "- **Stacking (LR)**：学习模态间互补关系，需要训练数据，泛化性较好",
        "- **动态加权**：根据各模态置信度自适应分配权重，适合模态质量不稳定的场景",
        "",
        "### 风险等级阈值",
        "",
        "融合后风险评分 → 风险等级映射（来源：PHQ-9 临床分级标准）：",
        "",
        "| 评分范围 | 风险等级 | 处理建议 |",
        "|----------|---------|---------|",
        "| 0.0 ~ 0.3 | low | 正常情绪波动 |",
        "| 0.3 ~ 0.6 | medium | 需要关注，建议轻度干预 |",
        "| 0.6 ~ 0.8 | high | 建议专业评估 |",
        "| 0.8 ~ 1.0 | crisis | 立即触发升级流程 |",
        "",
    ])

    report = "\n".join(lines)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report
