"""从已落盘结果生成第十轮报告和独立复算记录，不写入虚构达标结论。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from experiments.round10_fusion import ROOT, OUTPUT, save


def figures(stages: list[tuple[str, float, float]], current_stage: str) -> None:
    """绘制原始历史样本成绩与类别变化，误差棒只表示种子波动。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import recall_score
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10})
    output = OUTPUT / "figures"
    output.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.errorbar(np.arange(len(stages)), [s[1] for s in stages], yerr=[s[2] for s in stages],
                fmt="o-", color="#245c8e", capsize=4, linewidth=1.5)
    ax.axhline(.70268, color="#ad4328", linestyle="--", label="相对提升10%目标：0.70268")
    ax.set_xticks(np.arange(len(stages)), [s[0] for s in stages], rotation=15)
    ax.set_ylabel("macro-F1，原821条，100%覆盖")
    ax.set_ylim(.59, max(.76, max(s[1] for s in stages)+.04))
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="upper left")
    for i, (_, value, _) in enumerate(stages):
        ax.annotate(f"{value:.4f}", (i, value), xytext=(0, -20), textcoords="offset points", ha="center")
    ax.set_title("历史诊断全记录；误差棒为五种子总体标准差，单次模型无误差棒")
    fig.tight_layout()
    fig.savefig(output / "historical_macro_f1.png", dpi=180)
    plt.close(fig)
    current_path = OUTPUT / current_stage / "historical_predictions.npz"
    if not current_path.exists():
        return
    current = np.load(current_path)
    if "probabilities" not in current:
        return
    probabilities = current["probabilities"]
    if probabilities.ndim == 2:
        probabilities = probabilities[None]
    old = np.load(OUTPUT / "final" / "historical_predictions.npz")
    y = current["labels"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, function, title in zip(axes, (f1_score, recall_score), ("逐类F1", "逐类召回")):
        baseline = np.mean([function(y, p.argmax(1), labels=[0, 1, 2], average=None) for p in old["baseline"]], 0)
        newer = np.mean([function(y, p.argmax(1), labels=[0, 1, 2], average=None) for p in probabilities], 0)
        x = np.arange(3)
        ax.bar(x-.18, baseline, .35, label="第九轮五种子均值", color="#a9bac7")
        ax.bar(x+.18, newer, .35, label=f"{current_stage}均值", color="#245c8e")
        ax.set_xticks(x, ["负向", "中性", "正向"])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        for j, value in enumerate(newer):
            ax.text(j+.18, value+.02, f"{value:.1%}", ha="center", fontsize=9)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "per_class.png", dpi=180)
    plt.close(fig)


def read(path: Path) -> Any:
    """读取实际工件。"""
    return json.loads(path.read_text("utf-8"))


def generate() -> None:
    """报告始终区分开发、历史诊断、目标和未完成工作。"""
    first = read(OUTPUT / "final" / "results.json")
    domain = read(OUTPUT / "domain" / "results.json")
    sequence = read(OUTPUT / "sequence" / "results.json")
    matched = read(OUTPUT / "final" / "matched_moe.json")
    checks = []
    for directory, key in (("final", "final3579"), ("domain", "probabilities")):
        values = np.load(OUTPUT / directory / "historical_predictions.npz")
        y, pred = values["labels"], values[key].argmax(1)
        expected = first["models"]["final3579"] if directory == "final" else domain
        f1 = float(f1_score(y, pred, labels=[0, 1, 2], average="macro"))
        acc = float(accuracy_score(y, pred))
        checks.append({"directory": directory, "macro_f1": f1, "accuracy": acc,
                       "matches_recorded": abs(f1-expected["macro_f1"]) < 1e-12 and abs(acc-expected["accuracy"]) < 1e-12})
    if not all(row["matches_recorded"] for row in checks):
        raise ValueError("独立复算与报告不一致")
    save(OUTPUT / "independent_recomputation.json", {"checks": checks})
    development = []
    for name, title in (("linear", "冻结表征分类器"), ("neural", "同专家池路由对照"),
                        ("text", "折内文字编码器训练"), ("aux_fusion", "辅助训练文字表征融合"),
                        ("deep_fusion", "深层人脸表征融合"), ("ordinal", "有序训练"),
                        ("external_fusion", "公开110M情感编码器"), ("external_large_fusion", "公开330M情感编码器"),
                        ("llm_fusion", "冻结3B语义表征"), ("evidence", "正确单模态监督证据"),
                        ("evidence5", "五档强度专家证据"), ("intensity", "连续强度回归"),
                        ("speech_evidence", "语音基础模型与三档专家"), ("speech_evidence5", "语音基础模型与五档专家"),
                        ("speech_bag", "内层来源专家袋装"), ("large_speech", "冻结7B语义与语音三档专家"),
                        ("large_speech5", "冻结7B语义与语音五档专家"),
                        ("direct_evidence", "三档专家加原始冻结输出"), ("direct_evidence5", "五档专家加原始冻结输出"),
                        ("qlora_prefix", "3B末六层监督适配与匹配冻结融合对照")):
        path = OUTPUT / name / "search_complete.json"
        if path.exists():
            rows = read(path)["results"]
            best = max(rows, key=lambda row: row["metrics"]["macro_f1"])
            development.append(f"| {title} | {len(rows)} | {best['metrics']['macro_f1']:.4f} |")
    winner_rows = [
        f"| 第九轮MoE 五种子均值 | 0.6388 | 78.17% | 23.29% | — |",
        f"| 第一阶段同配方2876训练对照 | {first['models']['development2876']['macro_f1']:.4f} | {first['models']['development2876']['accuracy']:.2%} | {first['models']['development2876']['per_class_recall'][1]:.2%} | {first['models']['development2876']['relative_gain_vs_reported_baseline']:.2%} |",
        f"| 第一阶段预定3579训练方案 | {first['models']['final3579']['macro_f1']:.4f} | {first['models']['final3579']['accuracy']:.2%} | {first['models']['final3579']['per_class_recall'][1]:.2%} | {first['models']['final3579']['relative_gain_vs_reported_baseline']:.2%} |",
        f"| 第二阶段来源风格惩罚与袋装 | {domain['macro_f1']:.4f} | {domain['accuracy']:.2%} | {domain['per_class_recall'][1]:.2%} | {domain['relative_gain']:.2%} |",
    ]
    later = []
    achieved = domain["target_achieved"]
    current_stage, current = "domain", domain
    plot_stages = [("第九轮基线", .6388, .0104),
                   ("首次锁定融合", first["models"]["final3579"]["macro_f1"], 0.),
                   ("来源惩罚", domain["macro_f1"], 0.)]
    stage_names = (("external_large_fusion", "公开330M表征方案"), ("external_fusion", "公开110M表征方案"),
                   ("cascade", "语义专家级联"), ("speech_final", "语音证据五种子组合"),
                   ("robust_final", "三档与五档证据五种子组合"), ("large_final", "7B语义与语音组合"),
                   ("holdout_final", "703保留来源输出适配"), ("direct_final", "原始冻结输出与专家联合融合"),
                   ("refined_final", "冻结配方的类别校正细化"))
    short_names = {"cascade": "语义级联", "speech_final": "语音专家", "robust_final": "强度组合",
                   "large_final": "7B专家", "holdout_final": "输出适配", "refined_final": "截距细化"}
    for name, title in stage_names:
        path = OUTPUT / name / "results.json"
        if path.exists():
            r = read(path)
            current_stage, current = name, r
            achieved = r["target_achieved"]
            if "runs" in r:
                score = r["mean_macro_f1"]
                acc = np.mean([v["accuracy"] for v in r["runs"]])
                recall = np.mean([v["per_class_recall"][1] for v in r["runs"]])
                gain = r["relative_gain_vs_reported_baseline"]
            else:
                score, acc, recall, gain = r["macro_f1"], r["accuracy"], r["per_class_recall"][1], r["relative_gain"]
            later.append(f"| {title} | {score:.4f} | {acc:.2%} | {recall:.2%} | {gain:.2%} |")
            plot_stages.append((short_names.get(name, title), score, r.get("std_macro_f1", 0.)))
    # 最后开发组合明确拒绝原始输出新增分支；主线选择依据是OOF，不是历史最高值。
    joint_path = OUTPUT / "joint_final" / "selection_locked.json"
    if (OUTPUT / "refined_final" / "results.json").exists():
        current_stage = "refined_final"
        current = read(OUTPUT / current_stage / "results.json")
        achieved = current["target_achieved"]
    elif joint_path.exists() and read(joint_path)["winner"]["right_weight"] == 0:
        current_stage = "large_final"
        current = read(OUTPUT / current_stage / "results.json")
        achieved = current["target_achieved"]
    figures(plot_stages, current_stage)
    status = "五种子历史点估计达到目标；独立来源泛化仍未确认" if achieved else "相对提升10%的目标尚未达到，目标模式继续运行"
    final_details = ""
    if "runs" in current:
        seed_rows = [f"| {row['seed']} | {row['macro_f1']:.6f} | {row['accuracy']:.2%} | {row['per_class_recall'][1]:.2%} |"
                     for row in current["runs"]]
        e = current["ensemble"]
        interval = current["bootstrap"]
        lock = read(OUTPUT / current_stage / "selection_locked.json")
        base_lock = lock.get("base_recipe", lock)
        pass_count = sum(row["macro_f1"] >= .70268 for row in current["runs"])
        ensemble_f1_text = " / ".join(f"{value:.4f}" for value in e["per_class_f1"])
        ensemble_recall_text = " / ".join(f"{value:.2%}" for value in e["per_class_recall"])
        refinement_text = ""
        if current_stage == "refined_final":
            refinement_text = (f"最终输出的负向/中性/正向截距为{lock['winner']['bias']}。此前粗网格负向最优值0.25位于上边界，"
                "因此预先把两个自由截距统一扩展到[-0.5,0.75]、步长0.05，正向固定为0，共676点；其余专家、混合权重、路由和训练参数保持固定。"
                f"来源OOF最优macro-F1为{lock['winner']['metrics']['macro_f1']:.6f}、来源等权F1为{lock['winner']['source_balanced_f1']:.6f}。"
                "独立审计用scikit-learn重算全部676点，确认锁定点确为相同开发目标的最优点，且锁定先于模型落盘和历史评测。"
                "该步骤在固定权重下使历史均值从0.694316变为0.703203，增加0.008887；这一部分收益来自输出校正，不应归于新MoE架构。")
        final_details = f"""## 当前最终配方的主验收

主指标为五种子macro-F1均值：**{current['mean_macro_f1']:.6f} ± {current['std_macro_f1']:.6f}**，相对原报告基线变化**{current['relative_gain_vs_reported_baseline']:.2%}**。±表示五种子总体标准差，不是统计置信区间。全部821条均有输出，类别支持数仍为负向326、中性85、正向410。每个种子的基专家训练过程相同，随机变化来自元分类器；这衡量元模型种子敏感性，不是五次独立数据采样或五次完整基础模型预训练。

| 种子 | macro-F1 | accuracy | 中性召回 |
|---|---:|---:|---:|
{chr(10).join(seed_rows)}

五种子概率集成的macro-F1为{e['macro_f1']:.6f}，accuracy为{e['accuracy']:.2%}，单列供参考，**没有拿集成分数代替主验收均值**。集成逐类F1为{ensemble_f1_text}，逐类召回为{ensemble_recall_text}，顺序均为负向、中性、正向。

**边界必须保留：{pass_count}/5个种子达到0.70268，概率集成没有达到该门槛；主验收均值高出门槛仅{current['mean_macro_f1']-.70268:.6f}。** 因此不能表述为“任意种子或集成模型稳定提升10%”。平均中性召回仍是短板，未随总体macro-F1同步改善。

{refinement_text}

当前融合权重为{base_lock['winner']['weights']}，候选成员顺序为{[m['stage'] for m in base_lock['members']]}。轻量专家最大概率低于{base_lock['winner']['route']}时执行相应证据分类头；测试实际调用比例为{current['costly_fraction']:.2%}。这是分类头调用比例，编码器特征本轮离线提前提取，不能等同于原始媒体逐请求的GPU/CPU总耗时下降。

30来源组、10000次配对bootstrap得到五种子均值的条件95%区间[{interval['new_mean_f1_ci95'][0]:.4f}, {interval['new_mean_f1_ci95'][1]:.4f}]，对基线的绝对增益区间[{interval['delta_ci95'][0]:+.4f}, {interval['delta_ci95'][1]:+.4f}]，相对增益区间[{interval['relative_gain_ci95'][0]:.2%}, {interval['relative_gain_ci95'][1]:.2%}]。这些区间不修正多轮历史测试暴露，不能作为“未来来源至少提升10%”的保证。
"""
    evidence_details = """## 新增的正确监督与语音证据

从数据作者公开的[官方元标注](https://drive.google.com/file/d/1OINbBnCat988BNPV0RnS5iphRlHd6SW2/view)恢复M/T/A/V四套监督。官方CSV共4403行，本地冻结镜像共4400条，逐条核验M三类标签均未改变。镜像中csbdgs2_与官方csbgs2_的拼写差异使用一条显式别名，按ID及文字例子核对；没有改写情感金标。T/A/V只在训练入口作为各自专家的监督，不在预测时读取测试单模态标签。

同一句话的整体标签不等于各模态各自的标签。新增M/T/A/V四个监督专家分别预测各自概率及连续强度，再用概率、熵、两两差异和可用性形成40维证据，由固定轮次CatBoost分类器输出最终三类结果。五档方案只把专家监督按[-0.5,-0.1,0.1,0.5]离散化，元证据48维；最终三类主任务与原821条始终保持不变。

选型采用严格嵌套来源划分：四折外层来源验证，每一外层训练集合再做三折来源交叉拟合来生成元训练特征。元训练没有使用基模型在自身训练样本上的预测；外层模型也不以验证损失早停。最终3579条的元训练特征使用四折来源交叉拟合，最终专家再按锁定规则拟合。五种子42至46全部保留，主验收是五种子macro-F1均值，概率集成分数单列。

语音采用冻结的[audeering wav2vec2情感模型](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)，固定提交6eba34a2485ea31cb03600241787c3a5edab8626。它在英语MSP-Podcast接受维度情绪监督，中文适用性由本次来源验证检验。16kHz原片语音经12层编码，提取1024维状态的有效帧均值和标准差，以及唤醒、支配、效价三维输出；只在训练有效语音上拟合标准化。长于10秒时取中间连续10秒，截断数量、可用性和失败数见speech_features/manifest.json。未观测语音保持独立mask，缺失专家不冒充中性。

该语音模型许可证为CC-BY-NC-SA-4.0，本轮产物用于研究；这不是可以直接商业部署的授权包。外部语言和表情模型的预训练语料重合仍未完全核实。本轮属于更强表征与正确监督的整体优化，不能把全部增益归因于MoE架构。

3B与7B语言特征使用同一固定中文提示、零少样本例子，提取回答开始前的隐藏状态，没有把目标答案放入提示。7B使用本机已有Qwen2.5-7B-Instruct，运行时NF4量化以适应8GB显存。尚未完成的7B实验不会提前写成历史收益。

原始输出联合融合进一步保留冻结语言模型三类概率及熵、语音模型三维情绪输出和可用性、表情网络24维统计及可用性，共33个目标标签无关特征。它们与40/48维折内专家证据拼接，由同一CatBoost有限网格选型；没有引入新的测试标签或全局测试统计。

新增监督适配分支使用本地3B NF4模型，仅在最后六层Q/V投影训练rank8适配器。整体M三类与独立文字T五档分别监督融合头和文字辅助头，辅助损失权重0.3；数值分支包含冻结的文字补充、语音及人脸观测。匹配冻结对照使用相同头部结构、监督、有效batch16、微批4和三轮学习率轨迹，比较第1/2/3轮的统一checkpoint。适配器、PCA、标准化及监督都限制在当前来源训练折。

前30层固定输出以BF16缓存，RoPE保持完整kbit模型的FP32精度，以减少重复训练计算。真实GPU冒烟验证记录：缓存与完整前向相对RMSE为0，适配权重实际更新，保存重载预测最大误差为0，峰值显存约3.4GB。它说明该缓存实现与保存流程一致，不是开发或历史成绩。实现参照[PEFT量化训练文档](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)，固定peft0.18.0；该分支仅在来源OOF支持时才进入历史验收。

专家证据读出的思路参考[冻结多模态模型的共享情感证据论文](https://arxiv.org/html/2610.19981v1)，但该论文使用的基础模型、28维输出证据和回归任务与本实现不同；本轮不是原论文复现，也不称为I²MoE或FuseMoE官方复现。

内层专家袋装与细粒度强度监督均保留完整开发与历史记录。开发分数提高没有稳定迁移到历史集，不能挑选历史分数最高的阶段重新包装成一次独立确认。

该阶段的开发组合同时比较7B专家主线与原始冻结输出分支，四分之一步长的组合没有超过原主线，锁定结果对原始输出分支权重为0，因此没有为其追加历史评测。该阶段保留large_final，依据见joint_final/selection_locked.json。输出适配分支单独列为协议对照，没有用其历史结果倒推或修改主线阈值。

QLoRA四折全量完成后，独立适配最好macro-F1为0.692587，匹配冻结对照最好为0.688813；与既有7B主线组合后，来源开发选择给新增分支的权重为0，故未追加历史选型。随后以开发最优截距位于原网格边界为依据，仅细化主线的两个输出截距，形成refined_final；准确率增益不归因于QLoRA。
"""
    audit_path = OUTPUT / "final_audit.json"
    audit_text = "最终审计仍在进行。"
    if audit_path.exists():
        audit = read(audit_path)
        audit_text = (f"最近完成的独立审计：五种子均值{audit['mean_macro_f1']:.6f}，模型序列化重放最大概率误差"
                      f"{audit['serialized_model_prediction_max_error']:.1g}。删除真实标签、来源ID及原始文本，并禁止读取官方监督后，"
                      "预测仍逐位一致。审计对应阶段和指纹以final_audit.json为准。")
    sequence_rows = []
    for horizon, row in sequence["horizons"].items():
        sequence_rows.append(f"| {horizon} | {row['test_n']} | {row['persistence']['macro_f1']:.4f} | {row['model']['macro_f1']:.4f} |")
    tests = sum(int(node.attrib.get("tests", 0)) for node in ET.parse(OUTPUT / "regression_junit.xml").getroot().iter("testsuite"))
    reconstruction_text = "固定配方重训复核仍在进行。"
    rebuild_path = OUTPUT / "locked_rebuild_check.json"
    if rebuild_path.exists() and read(rebuild_path)["passed"]:
        rebuild = read(rebuild_path)
        reconstruction_text = (f"从冻结输入重建了全部最终训练权重，没有复制原训练模型；{rebuild['n_predictions']}个种子×样本硬预测全部相同，"
            f"最大概率差{rebuild['max_probability_error']:.1g}，重训五种子均值{rebuild['rebuilt_mean_macro_f1']:.6f}。详见locked_rebuild_check.json及reproduction_locked_v2/v3。")
    report = f"""# 第十轮多模态情感模型优化报告

日期：2026年11月16日。状态：**{status}**。

本轮以第九轮同一821条历史样本、五种子平均macro-F1 0.6388为起点，目标为相对提升至少10%，即macro-F1至少0.70268。所有三类标签、测试ID和来源组保持不变，覆盖率100%。没有以减少测试样本、修改中性标签或更换指标实现目标。

当前按来源开发验证选择的主线为`{current_stage}`。本报告保留历次负结果；数值达标仅指历史点估计，不等于对未来数据保证10%收益。

{final_details}

## 已验证结果

| 方案 | 历史macro-F1 | accuracy | 中性召回 | 相对0.6388变化 |
|---|---:|---:|---:|---:|
{chr(10).join(winner_rows+later)}

![历史macro-F1及目标](D:/Develop/AIC/_国赛优化/round10/figures/historical_macro_f1.png)

![逐类表现](D:/Develop/AIC/_国赛优化/round10/figures/per_class.png)

第一阶段预定以3579条非测试样本拟合；2876条结果是提前声明的训练规模对照。因为历史分数更高而改选2876条方案，会引入测试选型，因此没有这么做。固定三轮、相同3579训练规模的第九轮式MoE对照，五种子平均macro-F1为{matched['mean_macro_f1']:.4f}；它仅用于训练规模对照，不代表该架构经过独立充分调优的最高性能。

第一阶段最终方案的来源组配对增益95%区间为[{first['bootstrap']['delta_ci95'][0]:+.4f}, {first['bootstrap']['delta_ci95'][1]:+.4f}]。第二阶段为[{domain['bootstrap']['delta_ci95'][0]:+.4f}, {domain['bootstrap']['delta_ci95'][1]:+.4f}]。这些是30个来源视频组、10000次配对重采样的条件区间，不能消除历史测试多轮暴露造成的适应性选择偏差，也不构成未来来源至少提升10%的保证。

## 数据与选择协议

开发选择仅使用原train和selection共2876条、107个来源视频组，采用固定种子42的四折来源分组OOF。标准化、缺失统计、PCA、分类头和监督文字编码器均在训练折拟合；监督文字编码器没有直接用见过整批目标标签的旧权重冒充OOF。

原calibration703在前期主线不用于验证、概率校准或阈值选择；配方锁定后作为额外监督训练样本加入最终拟合，得到3579条非测试训练集。后续holdout_final明确更改协议：基础模型重新只在2876条训练，703条与基础训练来源互斥，用三折来源OOF选择输出适配器。它的全部模型均重新排除703基础监督；703曾参与此前其他模型的训练，因此不是整个项目始终未见的新确认集。该协议改变和负结果均单独保存，不能与此前主线混称。

历史821只在各阶段配方锁定和模型落盘后评测；第一阶段之后的新增阶段明确属于适应性研究，不能称为全新独立确认。holdout_final之后的原始冻结输出分支恢复3579条主线训练方案，仍只用2876条来源OOF选型。

## 实验范围与开发结果

下表为各族开发候选中最高的逐样本OOF分数，属于选型成绩，不能与历史测试分数直接比较；后续来源泛化阶段实际同时考虑来源等权成绩。

| 实验族 | 候选含轮次数量 | 最高开发macro-F1 |
|---|---:|---:|
{chr(10).join(development)}

第一阶段锁定组合为75%辅助文字与多模态线性分类器、25%增加深层人脸表示的线性分类器，采用软投票。共享同一专家池的均匀投票、全局权重、逐样本路由与实际Top-2分派均已训练对照，复杂动态路由没有在本次开发协议中获胜。不能据此宣称MoE或Top-k普遍无效。

辅助文字编码器只使用ECISA官方训练14620条进行预训练，然后冻结提取目标特征。深层人脸使用相同五帧与YuNet检测，在冻结HSEmotion表情网络倒数层提取1280维表示，并汇总均值与标准差；PCA仅在当前训练折拟合。最大人脸尚未验证就是说话人。

第二阶段使用训练来源均值构造低秩风格惩罚，并把正则化强度按训练样本数统一。预测不使用测试来源均值，也不需要来源ID。最终采用四个固定来源折训练模型等权软投票。该阶段提高了历史点估计，但仍未自动解决中性识别。

公开情感编码器是额外表征对照。其二类情感输出仅用于特征，不被映射成新的三类金标；上游八个训练集合未完整公开核实，不能排除与目标数据或预训练语料重合。来源见[110M模型卡](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment)与[330M模型卡](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment)。

{evidence_details}

## 原实现修正

1. 严格固定轮次时采用最后一轮，不能再按selection最低损失回载旧checkpoint。原非固定轮次流程仍保留。
2. 反事实样本必须有合法捐赠者，且双方被替换模态均可观测；无配对样本不进入CF损失。不存在用负索引读取最后一条标签的路径。
3. CF与对照共享模态丢弃和加噪配方，质量随捐赠者观测转移。删除“捐赠者整体M硬标签等于混合样本标签”的断言；改为源完整输入与捐赠者仅被替换模态的教师软预测，并停止教师和路由目标梯度。它是有假设的自蒸馏代理，不是可验证的合成整体金标或已识别因果效应。旧硬目标代码已另存档。
4. 人脸未检出时mask为假，不把零向量冒充有效人脸；归一化只使用有效训练观测。
5. Laplace形状权重采用对数域归一化，有限大路由值不会让全部门控权重下溢为零；补齐有效观测与质量的有限值检查。
6. 修正第五轮缓存来源说明：原text_features来自冻结预训练RoBERTa，不能称为监督微调缓存。旧第九轮结果文件保留，没有覆盖成修复后的数字。

## 鲁棒性与选择性预测

第一阶段完整821条中，去人脸macro-F1为{first['stress']['without_face']['macro_f1']:.4f}，去音频为{first['stress']['without_audio']['macro_f1']:.4f}，去原CLIP视觉为{first['stress']['without_visual']['macro_f1']:.4f}。人脸跨样本错配下降至{first['stress']['face_swapped']['macro_f1']:.4f}，说明可靠配对仍是明显短板，不能声称纠偏问题已经解决。

80%覆盖率下accuracy达到{first['selection_curve'][2]['accuracy']:.2%}，但中性样本只保留{first['selection_curve'][2]['class_coverage'][1]:.2%}，中性召回只有{first['selection_curve'][2]['per_class_recall'][1]:.2%}。因此不能只用高覆盖子集的总体accuracy证明柔性判定有效。所有预测概率仍标记未验证校准。

## 会话状态序列实验

使用EmotionTalk的真实文字、视觉及多模态人工情绪状态标注，按官方对话和说话人形成序列，保留官方来源验证与测试组。输入仅包含目标前的历史状态，至少已有五次观测，比较5、10、20步历史和未来1、3、5个本人话轮。七类情绪任务与CH-SIMS三类极性任务分开计算。

| 未来本人话轮跨度 | 测试窗口 | 保持上一状态macro-F1 | 来源验证锁定模型macro-F1 |
|---|---:|---:|---:|
{chr(10).join(sequence_rows)}

这是以过去人工标注状态可观测为前提的条件实验，尚未接入原始媒体识别器的实际误差。顺序来自同一会话文件名中的话轮编号，缺绝对时间戳确认。它不能被表述为已经完成原始多模态长序列预测，更不能表述为跨天心理状态预测。数据来源见[EmotionTalk原论文](https://arxiv.org/abs/2505.23018)。

三个跨度的锁定模型均已保存为sequence/horizon_1_model.pkl、horizon_3_model.pkl及horizon_5_model.pkl，并与1111、808、545个既有测试窗口逐条重放一致；见sequence/model_archive_check.json。

## 验证与工件

协议、缺失、配对、门控、未来信息隔离与无标签推理的定向回归为**{tests}项通过**，另有1条旧测试的requires_grad张量转标量告警；见regression_junit.xml。独立使用scikit-learn从逐样本预测复算主要历史macro-F1和accuracy，与存档一致，见[复算工件](D:/Develop/AIC/_国赛优化/round10/independent_recomputation.json)。{audit_text}

新的软目标CF还在真实CH-SIMS来源训练折完成moe/i2moe有无正则的四配置单轮数值检查，参数均有限，有效配对分别2075和2078个。该检查不作为CF精度增益证据，最终0.703203模型没有使用这项CF正则，不能把主结果增益归于CF。

{reconstruction_text}

重训复核曾发现并修复一个实际问题：父进程导入实验模块后把线程环境改为1，错误继承使子进程的随机PCA出现微小数值差异；首次重建五种子均值为0.701696。重训器现保存并继承程序启动时环境，已补充回归测试。原运行环境中NumPy与SciPy OpenBLAS均为20线程，CatBoost固定2线程；具体DLL与版本记录在重建日志。失败重建reproduction_locked_v1保留，没有覆盖成成功结果。**不能假定任意BLAS线程配置都能得到相同分数**，数值环境是复现条件的一部分。

工件目录为[D:/Develop/AIC/_国赛优化/round10](D:/Develop/AIC/_国赛优化/round10)。第一阶段模型与逐样本预测位于final，第二阶段位于domain。所有搜索的候选、来源OOF、选择锁定、负结果与模型分开保存；复现时使用新输出目录，不能覆盖已完成结果。

主要复现入口位于项目algorithm目录：

```powershell
D:\\Develop\\AIC\\.venv\\Scripts\\python.exe -X utf8 -m experiments.round10_fusion
D:\\Develop\\AIC\\.venv\\Scripts\\python.exe -X utf8 -m experiments.round10_neural
D:\\Develop\\AIC\\.venv\\Scripts\\python.exe -X utf8 -m experiments.round10_text
D:\\Develop\\AIC\\.venv\\Scripts\\python.exe -X utf8 -m experiments.round10_represented_fusion
D:\\Develop\\AIC\\.venv\\Scripts\\python.exe -X utf8 -m experiments.round10_represented_fusion --deep
D:\\Develop\\AIC\\.venv\\Scripts\\python.exe -X utf8 -m experiments.round10_select
D:\\Develop\\AIC\\.venv\\Scripts\\python.exe -X utf8 -m experiments.round10_final fit
D:\\Develop\\AIC\\.venv\\Scripts\\python.exe -X utf8 -m experiments.round10_final evaluate
D:\\Develop\\AIC\\.venv\\Scripts\\python.exe -X utf8 -m experiments.round10_domain search
D:\\Develop\\AIC\\.venv\\Scripts\\python.exe -X utf8 -m experiments.round10_sequence
```

最终证据分支依赖顺序为：辅助文字和深层人脸缓存 → 冻结3B语义和语音缓存 → 官方标注ID核验 → round10_speech_evidence search（可选--fine）→ round10_speech_final refit/evaluate → round10_robust_final select/refit/evaluate。7B使用项目根目录scripts/round10_llm_encoder.py --large和algorithm中的round10_large_speech。最终运行环境版本记录在execution_environment.json；新输出根目录可通过ROUND10_OUTPUT环境变量指定。

推荐先用项目根目录`scripts/round10_verify.ps1 -Stage refined_final`执行测试、独立复算与保存模型重放。固定配方重训命令如下，目标目录必须尚不存在；它复用冻结输入和已锁定参数，重新拟合所有最终训练权重，不重新搜索历史分数：

```powershell
& 'D:\\Develop\\AIC\\.venv\\Scripts\\python.exe' -X utf8 'D:\\Develop\\AIC\\AI-mental-main (2)\\AI-mental-main\\AI-mental-main\\scripts\\round10_reproduce_selected.py' --destination 'D:\\Develop\\AIC\\_国赛优化\\round10\\reproduction_user'
```

这是从冻结特征开始的训练复现，需要原第五轮特征、官方元标注和本轮冻结缓存；不是从所有原片开始的一键部署包。

## 批注落实程度与后续价值

| 图片建议 | 本轮工作 | 证据边界 |
|---|---|---|
| 数据质量与跨模态纠偏 | 真实缺失mask、来源隔离归一化、官方单模态标注、单模态冲突证据 | 未完成说话人人脸跟踪或人工复核整批标注 |
| 增加vision | 真实表情倒数层特征、训练折PCA、去人脸与错配压力对照 | 泛用CLIP视觉与表情特征不等同于完整视频行为理解 |
| MoE、投票与Top-k | 同专家池统一投票、全局门控、逐样本门控、真正Top-2执行；置信路由组合 | 开发结果没有支持把复杂路由当作主要收益来源 |
| 反事实训练 | 修复无效捐赠者、质量继承和不公平增广；删除整体硬标签替换，改为路由软自蒸馏 | 没有合成M金标或因果识别证明；主模型精度增益不归因于CF |
| 长序列分析预测 | 真会话历史情绪状态的多跨度预测、未来信息隔离测试 | 人工历史状态条件实验，尚非原始媒体端到端时序预测 |
| 部署提速 | 保留实际分类头调用比例和缺失处理 | 冻结特征离线提取尚未形成完整逐请求计时，不能宣称端到端提速10% |

优先提升方向是增加真正未见的新来源确认集、对中性与模态冲突样本做盲审，以及将视频人脸与当前说话人可靠对齐。数值层面应进一步采用更高精度的PCA并固定训练线程配置，在新来源重新确认稳定性。开发OOF与历史结果的落差表明，再增加路由复杂度的依据不足。序列分支应下一步接入实际识别概率及误差，再评价长跨度预测；当前结果不支持跨天预测的表述。

以上命令保留完成结果防覆盖检查；正式从零复现需先改为新输出根目录，并按脚本参数提取相应特征。当前状态为：{status}。关键源码快照、环境与工件指纹见round10/source/current、execution_environment.json及artifact_manifest.json；只有实际核验通过的数值才作为达标结论。
"""
    target = ROOT / "第十轮多模态情感模型优化报告-20261116.md"
    target.write_text(report, "utf-8")
    artifact_rows = []
    for title, relative, purpose in (
        ("最终模型", f"{current_stage}/model.pkl", "冻结的五种子模型及输出校正"),
        ("最终结果", f"{current_stage}/results.json", "五种子、集成、来源组区间"),
        ("逐样本预测", f"{current_stage}/historical_predictions.npz", "原821条、五种子概率及标签"),
        ("参数锁定", f"{current_stage}/selection_locked.json", "来源开发选型结果"),
        ("独立审计", "final_audit.json", "原始ID/标签核对、全部676点复算、模型重放"),
        ("重训复核", "locked_rebuild_check.json", "冻结输入重建，4105个预测完全一致"),
        ("回归结果", "regression_junit.xml", f"{tests}项通过"),
        ("工件指纹", "artifact_manifest.json", "源码、结果及报告SHA-256"),
    ):
        artifact_rows.append(f"| [{title}]({(OUTPUT / relative).as_posix()}) | {purpose} |")
    index = f"""# 第十轮最终送审包

日期：2026年11月16日。完整分析见[最终报告]({target.as_posix()})。

**预设历史点估计门槛已达到：五种子平均macro-F1为{current['mean_macro_f1']:.6f}，相对0.6388提升{current['relative_gain_vs_reported_baseline']:.2%}。** 原821条全部保留，标签与ID不变，覆盖率100%。

主验收均值与概率集成分别报告：集成macro-F1为{current['ensemble']['macro_f1']:.6f}，未达到0.70268；五个种子中{pass_count}个达到门槛。条件来源组95%相对增益区间为[{current['bootstrap']['relative_gain_ci95'][0]:.2%}, {current['bootstrap']['relative_gain_ci95'][1]:.2%}]，历史集曾反复用于研究，新来源至少提升10%尚未验证。

| 文件 | 用途 |
|---|---|
{chr(10).join(artifact_rows)}

固定配方从冻结输入重建两次，均未复制训练权重，4105个种子×样本预测完全一致、最大概率误差0。首次错误继承线程环境的失败重建仍保留在reproduction_locked_v1。复现需使用报告记录的数值环境；完整版本、BLAS线程和DLL见execution_environment.json与重训日志。

验证命令：

```powershell
& 'D:\\Develop\\AIC\\AI-mental-main (2)\\AI-mental-main\\AI-mental-main\\scripts\\round10_verify.ps1' -Stage refined_final
```

从冻结输入重训的命令见完整报告。source/current保存当前代码，source目录另存旧硬标签CF及QLoRA搜索时的精确代码版本。负结果、来源OOF、训练协议和所有选择锁定文件均保留。

本轮主要收益来自更强的语音/语义表征、正确的单模态监督和融合输出校正。Top-k、QLoRA及反事实训练没有被包装成主结果精度增益；真实序列分支是过去人工情绪状态条件下的会话预测。下一优先级是独立新来源确认、中性和模态冲突样本盲审、说话人人脸对齐及数值精度稳定性。
"""
    (OUTPUT / "REVIEW-PACKAGE.md").write_text(index, "utf-8")
    print(str(target))


if __name__ == "__main__":
    generate()
