"""
一键运行所有对比实验 + 生成汇总 Markdown 报告

⚠️ 声明：所有实验使用模拟数据，结果仅供技术报告参考。

实验列表：
    1. 规则引擎 vs 多任务模型（风险评估）
    2. 单模态 vs 双模态（情感感知融合）
    3. 云端推理 vs 端侧推理（隐私保护）

输出：
    - experiments/outputs/experiment_summary.md（汇总报告）
    - experiments/outputs/compare_risk_models.md（实验 1 详细报告）
    - experiments/outputs/compare_emotion_models.md（实验 2 详细报告）
    - experiments/outputs/compare_privacy.md（实验 3 详细报告）
"""
from __future__ import annotations

import time
from pathlib import Path

# ============================================================
# 路径配置
# ============================================================

_MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _MODULE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = str(OUTPUT_DIR / "experiment_summary.md")


def run_all_experiments() -> dict:
    """运行所有对比实验

    Returns:
        dict: 各实验结果汇总
    """
    all_results = {}
    total_start = time.time()

    # 实验 1：规则引擎 vs 多任务模型
    print("\n" + "#" * 60)
    print("# 开始运行对比实验")
    print("#" * 60)

    from experiments.compare_risk_models import (
        run_experiment as run_risk_exp,
        generate_report as gen_risk_report,
    )
    risk_results = run_risk_exp()
    risk_report = gen_risk_report(risk_results)
    risk_path = str(OUTPUT_DIR / "compare_risk_models.md")
    with open(risk_path, "w", encoding="utf-8") as f:
        f.write(risk_report)
    print(f"\n[Output] 实验 1 报告 → {risk_path}")
    all_results["risk_models"] = risk_results

    # 实验 2：单模态 vs 双模态
    from experiments.compare_emotion_models import (
        run_experiment as run_emotion_exp,
        generate_report as gen_emotion_report,
    )
    emotion_results = run_emotion_exp()
    emotion_report = gen_emotion_report(emotion_results)
    emotion_path = str(OUTPUT_DIR / "compare_emotion_models.md")
    with open(emotion_path, "w", encoding="utf-8") as f:
        f.write(emotion_report)
    print(f"\n[Output] 实验 2 报告 → {emotion_path}")
    all_results["emotion_models"] = emotion_results

    # 实验 3：云端 vs 端侧推理
    from experiments.compare_privacy import (
        run_experiment as run_privacy_exp,
        generate_report as gen_privacy_report,
    )
    privacy_results = run_privacy_exp()
    privacy_report = gen_privacy_report(privacy_results)
    privacy_path = str(OUTPUT_DIR / "compare_privacy.md")
    with open(privacy_path, "w", encoding="utf-8") as f:
        f.write(privacy_report)
    print(f"\n[Output] 实验 3 报告 → {privacy_path}")
    all_results["privacy"] = privacy_results

    total_elapsed = time.time() - total_start
    all_results["_meta"] = {
        "total_time_seconds": round(total_elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return all_results


def generate_summary_report(all_results: dict) -> str:
    """生成汇总 Markdown 报告"""
    meta = all_results["_meta"]

    risk = all_results["risk_models"]
    emotion = all_results["emotion_models"]
    privacy = all_results["privacy"]

    lines = [
        "# 量化实验汇总报告",
        "",
        f"> 生成时间：{meta['timestamp']}",
        f"> 总耗时：{meta['total_time_seconds']:.2f} 秒",
        f"> 随机种子：42",
        "> ⚠️ 所有实验使用模拟数据，结果仅供技术报告参考",
        "",
        "---",
        "",
        "## 实验总览",
        "",
        "| # | 实验名称 | 对比方案 | 核心发现 |",
        "|---|----------|----------|----------|",
    ]

    # 实验 1 摘要
    r_rule = risk["rule_engine"]["recall_high_risk"]
    r_mt = risk["multi_task"]["recall_high_risk"]
    exp1_finding = (
        f"多任务模型高风险召回率 {r_mt:.4f} vs 规则引擎 {r_rule:.4f}"
    )
    lines.append(f"| 1 | 风险评估方法 | 规则引擎 vs 多任务模型 | {exp1_finding} |")

    # 实验 2 摘要
    e_text = emotion["text_only"]["accuracy"]
    e_stack = emotion["stacking_fusion"]["accuracy"]
    exp2_finding = (
        f"Stacking 融合准确率 {e_stack:.4f} vs 纯文本 {e_text:.4f}"
    )
    lines.append(f"| 2 | 情感感知融合 | 单模态 vs 双模态 | {exp2_finding} |")

    # 实验 3 摘要
    p_cloud = privacy["cloud"]["privacy_score"]
    p_device = privacy["device_quantized"]["privacy_score"]
    exp3_finding = (
        f"端侧量化隐私评分 {p_device}/100 vs 云端 {p_cloud}/100"
    )
    lines.append(f"| 3 | 隐私保护方案 | 云端 vs 端侧推理 | {exp3_finding} |")

    lines.extend([
        "",
        "---",
        "",
        "## 实验 1：规则引擎 vs 多任务模型",
        "",
        "### 核心指标对比",
        "",
        "| 指标 | 规则引擎 | 多任务模型 |",
        "|------|----------|------------|",
    ])

    for metric, label in [
        ("accuracy", "Accuracy"),
        ("f1_macro", "F1 (macro)"),
        ("recall_high_risk", "Recall (高风险) ⭐"),
        ("specificity", "Specificity"),
    ]:
        lines.append(
            f"| {label} | {risk['rule_engine'][metric]:.4f} | "
            f"{risk['multi_task'][metric]:.4f} |"
        )

    lines.extend([
        "",
        "> ⭐ Recall (高风险) 是临床安全性的核心指标——漏诊代价远高于误诊。",
        "",
        "---",
        "",
        "## 实验 2：单模态 vs 双模态",
        "",
        "### 核心指标对比",
        "",
        "| 模型 | Accuracy | F1 (macro) | Robustness |",
        "|------|----------|------------|------------|",
        f"| 纯文本（单模态） | {emotion['text_only']['accuracy']:.4f} | "
        f"{emotion['text_only']['f1_macro']:.4f} | "
        f"{emotion['text_only']['robustness_noisy']:.4f} |",
        f"| 加权融合（双模态） | {emotion['weighted_fusion']['accuracy']:.4f} | "
        f"{emotion['weighted_fusion']['f1_macro']:.4f} | "
        f"{emotion['weighted_fusion']['robustness_noisy']:.4f} |",
        f"| Stacking 融合（双模态） | {emotion['stacking_fusion']['accuracy']:.4f} | "
        f"{emotion['stacking_fusion']['f1_macro']:.4f} | "
        f"{emotion['stacking_fusion']['robustness_noisy']:.4f} |",
        "",
        "> Robustness 为噪声条件下的准确率，衡量模型在真实干扰下的稳定性。",
        "",
        "---",
        "",
        "## 实验 3：云端推理 vs 端侧推理",
        "",
        "### 核心指标对比",
        "",
        "| 方案 | Accuracy | Model Size | Latency | Privacy |",
        "|------|----------|------------|---------|---------|",
        f"| 云端推理 | {privacy['cloud']['accuracy']:.4f} | "
        f"{privacy['cloud']['model_size_mb']}MB | "
        f"{privacy['cloud']['inference_latency_ms']:.1f}ms | "
        f"{privacy['cloud']['privacy_score']}/100 |",
        f"| 端侧推理（量化） | {privacy['device_quantized']['accuracy']:.4f} | "
        f"{privacy['device_quantized']['model_size_mb']}MB | "
        f"{privacy['device_quantized']['inference_latency_ms']:.1f}ms | "
        f"{privacy['device_quantized']['privacy_score']}/100 |",
        "",
        f"> 模型压缩率：{privacy['_meta']['size_reduction_pct']}%",
        "",
        "---",
        "",
        "## 综合结论",
        "",
        "### 技术选型建议",
        "",
        "1. **风险评估**：多任务模型在高风险召回率上优于规则引擎，",
        "   更符合「漏诊代价远高于误诊」的临床安全原则。",
        "",
        "2. **情感感知**：双模态融合（特别是 Stacking）在准确率和鲁棒性上",
        "   均优于单模态，建议在条件允许时启用双通道。",
        "",
        "3. **隐私保护**：端侧量化推理在保持较高准确率的同时，",
        "   隐私评分从 20 提升至 90，且模型大小降低 90%+，",
        "   满足移动端部署需求。",
        "",
        "### 局限性汇总",
        "",
        "| 局限 | 影响 | 改进方向 |",
        "|------|------|----------|",
        "| 模拟数据 | 结果不可直接推广到临床 | 接入真实数据集验证 |",
        "| 浅层模型 | 表达能力有限 | 引入预训练语言模型 |",
        "| 未调优 | 非最优超参数 | 网格搜索 / 贝叶斯优化 |",
        "| 单机测量 | 不含网络延迟 | 部署后端到端测量 |",
        "",
        "## 详细报告链接",
        "",
        "- [实验 1 详细报告](compare_risk_models.md)",
        "- [实验 2 详细报告](compare_emotion_models.md)",
        "- [实验 3 详细报告](compare_privacy.md)",
        "",
    ])

    return "\n".join(lines)


def main():
    """主函数"""
    print("=" * 60)
    print("心理健康平台 —— 量化实验套件")
    print("=" * 60)

    all_results = run_all_experiments()
    summary = generate_summary_report(all_results)

    summary_path = SUMMARY_PATH
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n{'=' * 60}")
    print(f"所有实验完成！")
    print(f"{'=' * 60}")
    print(f"\n[Output] 汇总报告 → {summary_path}")
    print(f"[Output] 实验 1 → {OUTPUT_DIR / 'compare_risk_models.md'}")
    print(f"[Output] 实验 2 → {OUTPUT_DIR / 'compare_emotion_models.md'}")
    print(f"[Output] 实验 3 → {OUTPUT_DIR / 'compare_privacy.md'}")

    return all_results


if __name__ == "__main__":
    main()
