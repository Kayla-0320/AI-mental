"""整理本轮代码、环境、实际结果和下一步条件，保留负结果与验证边界。"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
from typing import Any

from experiments.round11_common import digest
from experiments.round12_explore import OUTPUT, save


def charts(results: dict[str, Any], fallback: dict[str, Any]) -> None:
    """生成可直接导出的静态实验图，纵轴与任务口径明确。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    folder = OUTPUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    rows = results["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), layout="constrained")
    y = np.arange(len(rows))
    axes[0].barh(y-.17, [r["metrics"]["clean"]["macro_f1"] for r in rows], .32, color="#2166ac", label="Clean M3")
    axes[0].barh(y+.17, [r["robustness_mean"] for r in rows], .32, color="#63a892", label="Mean of 5 stress conditions")
    axes[0].set_yticks(y, [r["config"]["id"] for r in rows])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, .8)
    axes[0].set_title("11 fixed-budget controls | 4 source folds")
    axes[0].legend(loc="lower left", fontsize=8)
    selected = [r for r in fallback["rows"] if r["condition"] in ("clean", "missing_A", "text_only")]
    x = np.arange(len(selected))
    axes[1].bar(x-.18, [r["old"]["macro_f1"] for r in selected], .34, color="#2166ac", label="Round 11 7B")
    axes[1].bar(x+.18, [r["fallback"]["macro_f1"] for r in selected], .34, color="#63a892", label="Audio-absent fallback")
    axes[1].set_xticks(x, [r["condition"] for r in selected])
    axes[1].set_ylim(0, .8)
    axes[1].set_title("Supplemental availability route")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Macro-F1") if ax is axes[0] else ax.set_ylabel("Macro-F1")
        ax.grid(axis="x" if ax is axes[0] else "y", alpha=.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Previously exposed development data only | no independent test / no production promotion", fontsize=11)
    fig.savefig(folder / "development_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """打包前核查完成与数值，不把待人工采集事项写为完成。"""
    root = Path(__file__).resolve().parents[2]
    results = json.loads((OUTPUT / "results.json").read_text("utf-8"))
    fallback = json.loads((OUTPUT / "audio_absent_fallback/results.json").read_text("utf-8"))
    verification = json.loads((OUTPUT / "verification.json").read_text("utf-8"))
    run = json.loads((OUTPUT / "run_complete.json").read_text("utf-8"))
    if not run["complete"] or not verification["previous_inputs_preserved"]:
        raise ValueError("本轮验证尚未完成")
    source = OUTPUT / "source"
    files = list((root / "algorithm/experiments").glob("*.py"))
    files += list((root / "algorithm/perception/knowledge_moe").glob("*.py"))
    files += list((root / "algorithm/perception/knowledge_moe").glob("*.json"))
    files += [root / "algorithm/tests" / name for name in ("test_round12_exploration.py", "test_round11_protocol.py", "test_knowledge_moe.py")]
    for path in files:
        target = source / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "torch", "catboost", "pytest", "matplotlib")}
    save(OUTPUT / "environment.json", dict(python=platform.python_version(), platform=platform.platform(), packages=versions,
        execution="CPU training, torch threads2, seed42", original_run=run), immutable=False)
    charts(results, fallback)
    report = OUTPUT / "第十二轮鲁棒训练与知识MoE探索报告.md"
    text = report.read_text("utf-8")
    if "本轮可直接采用的结论" not in text:
        first, rest = text.split("\n", 1)
        intro = ("\n\n## 本轮可直接采用的结论\n\n"
            "- 固定11候选、4来源折、每头种子42/60轮均完成；增广、知识、MoE与蒸馏未超过预定综合标准下的普通轻量融合。保留负结果，不据此替换主模型。\n"
            "- 缺音频补充路由：缺A macro-F1 0.647107→0.667892，但来源区间跨0；仅文字0.555239→0.596470，开发条件区间为正。干净0.728403→0.727571，略回退；未知音轨错配仍未解决。\n"
            "- 仅保留离线研究候选；仍缺新封存来源、两位真人复核、线上自动转录/说话人关联、跨天连续真实观测。原821与外部1929测试本轮未复评。\n"
            "- 69项相关测试通过；28次头重放最大误差1.19e-7，从头重训误差0。\n\n"
            "![开发消融与缺音频路由](figures/development_comparison.png)\n")
        report.write_text(first + intro + rest, encoding="utf-8")
    (OUTPUT / "REPRODUCE.md").write_text(
        "# 第十二轮复现与工件导航\n\n"
        "固定开发条件、负结果和局部收益见《第十二轮鲁棒训练与知识MoE探索报告.md》。preregister.json在跑分前写入，results.json为原11候选锁定结果；audio_absent_fallback为主搜索后的独立说明补充，不能冒充原预注册胜者。\n\n"
        "- candidates/*/fold*/head.pt：来源留出头；predictions.npz和oof.npz：全部逐条概率。\n"
        "- audit/fold*.json：训练/留出来源、训练索引近邻和供体、输入缓存SHA256。\n"
        "- previous_baselines：旧7B/light在相同开发压力输入的重放。\n"
        "- verification.json：28次重放、折0从头重训及旧输入摘要核验。\n"
        "- annotation_audit/multimodal_A.csv、multimodal_B.csv：两份空白M3原媒体审阅模板；reference_private.json不得交给盲标者。\n"
        "- source：当前代码快照；environment.json：版本。依赖原round10/round11的冻结开发表征与配套专家，本包不重复拷贝基础大模型。\n\n"
        "在项目algorithm目录使用D:/Develop/AIC/.venv/Scripts/python.exe执行：\n\n"
        "```powershell\n"
        "& 'D:/Develop/AIC/.venv/Scripts/python.exe' -m pytest tests/test_round11_protocol.py tests/test_knowledge_moe.py tests/test_round12_exploration.py -q -p no:cacheprovider\n"
        "& 'D:/Develop/AIC/.venv/Scripts/python.exe' -m experiments.round12_reproduce --output D:/Develop/AIC/_国赛优化/round12_reproduction\n"
        "```\n\n"
        "重训输出目录必须不存在，以保护原锁定工件；`--smoke`可运行同样44头的一轮流程检查。原目录的verify和fallback已有锁定结果，不重复覆盖。\n\n"
        "离线推理使用experiments.round12_runtime.OfflinePredictor，predict不接收金标。所用轻量原专家必须与头的外层来源折一致，不能混用其他折或最终全训练专家来声称重放等价。需要记忆的候选在训练来源建立索引并排除同来源/同原文；默认胜者不依赖知识/记忆。该接口尚未接入平台线上流程。\n",
        encoding="utf-8")
    (OUTPUT / "下一步执行清单.md").write_text(
        "# 第十二轮之后的执行边界\n\n"
        "1. 先补真正的新封存来源。当前2876开发来源已经多轮接触，重分折不产生新独立数据。冻结本轮候选与音频缺失路由，再采集/接入来源未暴露的带授权中文T/A/V媒体；整对话/说话人分组，去重后封存，下一次评测前不查看目标。\n"
        "2. 知识增强先改善资源质量：现有六节点仅工程词表，打乱对照未显示稳定知识收益。下一实验分别比较候选词表、授权中文ATOMIC/COMET事件/反应表示、训练来源监督记忆；资源来源/许可、中文翻译质量、否定/反讽/覆盖率必须独立核查。保持移除知识、打乱知识、等维容量对照，不能把新增外部监督与同预算方法混写。\n"
        "3. MoE目前没有综合优势；先解决缺A收益与干净回退的权衡，再考虑更多专家。比较独立的缺失增广、错配一致性、质量路由因子；当前aug将缺失和OOF软教师错配一致性组合，不能将差距都归因于单一组件。若继续Top-k，分别测编码器/门控/专家/整体成本。\n"
        "4. 使用annotation_audit两份M3模板组织真人独立审阅。先盲标后计算一致率/kappa与中性分歧；官方标签保持原样。EmotionTalk的happy/连续极性语义需单独复核，不能从CH审阅推断。\n"
        "5. 本轮蒸馏未胜出：下一步用更严格的嵌套OOF最终融合教师，而非仅联合M专家；教师生成每条训练软分布时，感知专家和融合头都不得见过该来源。加自动ASR转录和说话人关联误差控制，重新实测原媒体端到端成本。\n"
        "6. 跨天趋势需新连续多日真实记录与个人时间边界，当前话轮跨度不可替代。以上待数据/真人事项没有计为已完成。\n",
        encoding="utf-8")
    save(OUTPUT / "COMPLETION-AUDIT.json", dict(completed=["11fixed_controls_x4sourcefolds_x60epochs", "7conditions_full2876_coverage",
        "knowledge_and_memory_shuffle_controls", "actual_top2_head_dispatch", "source_OOF7B_M_expert_distillation_control",
        "same_corruption_previous7B_light_baselines", "audio_absent_supplement", "28replays_and_fold0_independent_retrain",
        "69related_tests_passed", "120double_annotator_media_templates", "source_environment_report"],
        pending=["fresh_sealed_sources", "human_annotation_agreement", "full_fusion_teacher_distillation", "online_ASR_speaker_evaluation",
                 "raw_media_latency_after_change", "multiday_real_observations"], production_promoted=False), immutable=False)
    paths = [p for p in OUTPUT.rglob("*") if p.is_file() and p.name != "artifact_manifest.json" and "smoke" not in p.relative_to(OUTPUT).parts]
    save(OUTPUT / "artifact_manifest.json", dict(files=[dict(path=str(p.relative_to(OUTPUT)), bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(paths)]), immutable=False)


if __name__ == "__main__":
    main()
