"""导出第七轮OOF成本曲线、反事实分解及评审回应，不重新选择模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "algorithm"))
from experiments.round7_decision_oof import paired_ci


def read(path: Path) -> Any:
    """读取冻结的实验元数据。"""
    return json.loads(path.read_text(encoding="utf-8"))


def export(root: Path) -> None:
    """只从已经锁定的工件生成材料，不回调阈值。"""
    result = read(root / "results.json")
    verification = read(root / "verification.json")
    suites = ET.parse(root / "regression.xml").getroot().findall("testsuite")
    total = sum(int(suite.get("tests", 0)) for suite in suites)
    skipped = sum(int(suite.get("skipped", 0)) for suite in suites)
    failed = sum(int(suite.get("failures", 0)) + int(suite.get("errors", 0)) for suite in suites)
    passed = total - skipped - failed
    floors = result["protocol"]["negative_recall_constraints"]
    with np.load(root / "historical_A.npz", allow_pickle=False) as saved:
        y, groups = saved["labels"], saved["source_groups"]
        base = saved["round5_predictions"]
        plain = saved["scores"].argmax(1)
        plain[~saved["face"]] = base[~saved["face"]]
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
    plain_metrics = {"macro_f1": float(f1_score(y, plain, labels=[0, 1, 2], average="macro", zero_division=0)),
                     "accuracy": float(accuracy_score(y, plain)), "neutral_f1": float(f1_score(y, plain, labels=[1], average=None, zero_division=0)[0]),
                     "negative_recall": float(recall_score(y, plain, labels=[0], average=None)[0]),
                     "neutral_correct": int(((y == 1) & (plain == 1)).sum()), "confusion_matrix": confusion_matrix(y, plain, labels=[0, 1, 2]).tolist()}
    decomposition = {"description": "post-lock exploratory decomposition: remove the old primary neutral bias -1 only; keep old neural fallback; not selected or promoted",
                     "metrics": plain_metrics, "delta_vs_round5_ci95": paired_ci(y, plain, base, groups)}
    (root / "zero_bias_decomposition.json").write_text(json.dumps(decomposition, indent=2), encoding="utf-8")
    np.savez_compressed(root / "zero_bias_decomposition.npz", predictions=plain, labels=y, source_groups=groups)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), layout="constrained")
    colors = {"A": "#2563eb", "B": "#b45309"}
    for track in ["A", "B"]:
        policy = result["policies"][track]
        nested = result["nested_oof"][track]["nested_cost_curve"]
        for column, field in enumerate(["macro_f1", "neutral", "negative"]):
            def value(record: dict[str, Any]) -> float:
                return record["macro_f1"] if field == "macro_f1" else record["per_class_f1"][1] if field == "neutral" else record["per_class_recall"][0]
            apparent = [value(policy["cost_curve"][str(floor)]["metrics"]) for floor in floors]
            heldout = [value(nested[str(floor)]) for floor in floors]
            axes[column].plot(floors, apparent, "--", color=colors[track], alpha=.55, label=f"{track}: selection OOF")
            axes[column].plot(floors, heldout, "o-", color=colors[track], markersize=4, label=f"{track}: nested held-out")
    for axis, title in zip(axes, ["Macro F1", "Neutral F1", "Observed negative recall"]):
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Required training-side negative recall", fontsize=9)
        axis.axvline(.85, color="#64748b", lw=.8, alpha=.5)
        axis.grid(alpha=.2)
        axis.set_xticks(floors)
        axis.tick_params(labelsize=9)
    axes[2].plot(floors, floors, ":", color="black", lw=1)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Decision cost curves from training data only | A: primary branch; B: full CPU pipeline", fontsize=11)
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(root / ("decision_cost_curve." + extension), dpi=180)
    plt.close(fig)
    h, e = result["historical"], result["external_ecisa"]
    def row(name: str, metric: dict[str, Any]) -> str:
        return f"| {name} | {metric['macro_f1']:.4f} | {metric['accuracy']*100:.2f}% | {metric['per_class_recall'][0]*100:.2f}% | {metric['per_class_f1'][1]:.4f} | {metric['confusion_matrix'][1][1]}/85 |"
    historical_table = "\n".join([row("第五轮锁定方案", h["round5"]), row("A：只改原主分支决策", h["A"]["historical"]),
                                   row("B：完整CPU OOF方案", h["B"]["historical"]), row("预设CPU argmax对照，未被选中", h["cpu_argmax"])])
    nested_table = []
    for track in ["A", "B"]:
        n = result["nested_oof"][track]
        chosen = n["nested_cost_curve"]["0.85"]
        nested_table.append(f"| {track} | {n['scope_n']} | {n['raw_argmax']['macro_f1']:.4f} | {n['raw_argmax']['per_class_recall'][0]*100:.2f}% | {chosen['macro_f1']:.4f} | {chosen['per_class_recall'][0]*100:.2f}% |")
    calibration_table = []
    for track in ["A", "B"]:
        b = result["nested_oof"][track]["nested_probability_before"]
        a = result["nested_oof"][track]["nested_probability_after"]
        calibration_table.append(f"| {track} | {b['nll']:.5f}→{a['nll']:.5f} | {b['brier']:.5f}→{a['brier']:.5f} | {b['ece10']:.4f}→{a['ece10']:.4f} |")
    report = f"""# 第七轮：决策层与来源分组OOF

日期：2026-11-16。承接第五／六轮评审，完成T1决策层与T2选择平台调整。全部使用CPU，正式数值运行约{result['seconds']:.1f}秒；没有训练或调用GPU编码器。折内重拟合17个分类流程，不应把本轮描述为“完全不训练”。

## 结论

**正式OOF选择没有复现评审中0.66–0.67的预期；不改选、不部署。** A历史macro-F1为0.6391，B为0.6329；两者相对第五轮的来源组配对区间均跨零。B在ECISA3363上还从原方案0.7019降至0.6543，负结果完整保留。

预先保留的CPU argmax对照历史macro-F1为0.6802、中性21/85，说明简化旧决策确有探索价值；但它的训练侧OOF负类召回只有77.57%，不满足预注册85%约束。**不能看到历史集高分后把它改成获选方案，也不能把约束要求偷偷调低。**

## 1. 对评审的核实与修正

- 方向上同意研究决策规则、让703退出新选择、保留负结果和完成人工复核。
- “决策从未作为变量”不准确：第四至六轮已有中性偏置网格，第五轮主分支实际选出的中性偏置是**−1.0**。本轮新增的是来源OOF、温度拟合、决策族和成本约束，而非首次研究决策。
- 评审的softmax值来自归一化margin及不同尺度的回退logit，不能直接称为已校准概率。温度拟合也不自动等于校准有效，见下方嵌套结果。
- 相对原拼接0.5657375提高10%，所需绝对增量是**0.0565738**，目标分数0.6223113。评审将所需增量写成0.0622有误；第五轮下界0.0179仍不足以支持“保证提高10%”。
- 0.6266→0.6738的相对变化以第五轮作分母约为7.53%；8.34%是以原拼接分数作分母的增量比例，需区分。
- 开发与历史排序反转提示选择不稳及来源差异，不能把不同模型的开发差距直接量化成“选择偏差的价格”。OOF不能抹去此前的架构选择和历史测试暴露。
- ECISA匹配配方差值+0.1916支持该文字任务及数据家族上的辅助训练效果，不等于证明人脸增益或未知三模态来源稳定提高。EmotionTalk线本轮停止扩充，“未证实增量”也不等于“证明任何用法都无效”。
- 双人“各标一半”无法计算彼此一致性κ；需要共同独立盲标的重叠样本。低κ也不自动允许把“中性”改成“不确定性”。本轮没有伪造真人复核。

## 2. 为什么有A、B两条范围

第五轮神经文字回退已经监督训练过全部2876条，不能直接拿它给同一批训练样本生成“OOF”。没有重新逐折训练该编码器，就无法声称完整第五轮模型做了严格OOF。

**A：原模型主分支决策调整。** 主头参数与第五轮逐项核对一致，旧−1偏置移除后再应用OOF锁定规则。规则只从人脸可用的2753条OOF选择，原神经文字回退保持不变。因此A的OOF结论只覆盖主分支，不能覆盖完整旧回退；其ECISA文字单路结果按设计不变。

**B：完整CPU流程。** 主分支同上；文字回退改成折内拟合的字符TF-IDF＋逻辑回归，只使用该折目标训练及固定ECISA官方训练14620条，覆盖全部2876条OOF。它与第五轮的回退不同，必须用B自身argmax作对照来隔离决策收益。

## 3. 预注册协议

目标数据只读原train＋selection共2876条、107个来源组；703不再进入本轮校准、阈值或选型。所有随机性固定种子42。

外层4折来源分组；每个外层训练内部再做3折OOF，拟合标准化、分类器、TF-IDF、温度和决策规则。外层留出来源从未进入这些步骤。最后使用四折OOF选择最终策略，再拟合全训练模型、落盘selection_locked，之后才运行本轮的821／3363固定评测。

固定293条规则：argmax、含负向保护的中性决策带、类别偏置、序数切点及全负可行兜底。成本曲线要求负类召回75/80/85/90/95%，主约束85%，目标为约束下最大macro-F1、accuracy打破平局。该“负类”指消极情感，**不是自伤风险，也没有改变业务风险阈值或升级流程**。

最终A/B都选中：温度变换后，在log概率上给负类+0.75、中性−0.5，正类不变。A主分支温度约0.98774；B文字回退温度约1.43764。不存在根据测试来源ID分配阈值的规则。

## 4. 嵌套OOF结果：约束不是跨来源保证

| 范围 | 条数 | 原始argmax F1 | 原始负召回 | 嵌套选择后F1 | 嵌套负召回 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(nested_table)}

B最终用于选择的OOF规则负召回85.76%，但外层未见来源只有84.44%。训练侧达到85%不代表未来来源必然达到85%。A也由训练侧85.59%降到外层84.87%。

概率质量三个量都是越低越好：

| 范围 | NLL：拟合前→后 | Brier：前→后 | ECE10：前→后 |
|---|---|---|---|
{chr(10).join(calibration_table)}

温度拟合的外层NLL/Brier/ECE没有改善，不能宣称已经获得可靠概率校准。规则工件仍标记研究用途、未验证校准、未部署。

## 5. 相同821条历史样本

| 方案 | macro-F1 | accuracy | 负类召回 | 中性F1 | 中性识别 |
|---|---:|---:|---:|---:|---:|
{historical_table}

A相对第五轮的macro-F1差值95%区间为[{h['A']['delta_vs_round5_ci95'][0]:+.5f},{h['A']['delta_vs_round5_ci95'][1]:+.5f}]；B为[{h['B']['delta_vs_round5_ci95'][0]:+.5f},{h['B']['delta_vs_round5_ci95'][1]:+.5f}]。两者都没有达到预定0.66的验收项。B相对自身argmax还下降，区间[{h['B']['delta_vs_cpu_argmax_ci95'][0]:+.5f},{h['B']['delta_vs_cpu_argmax_ci95'][1]:+.5f}]。

锁定后的补充诊断只移除第五轮主分支−1中性偏置、保持旧神经回退，macro-F1为{plain_metrics['macro_f1']:.4f}、accuracy为{plain_metrics['accuracy']*100:.2f}%、中性{plain_metrics['neutral_correct']}/85。它用于拆解“旧偏置”与“CPU回退替换”的贡献，属于探索性反事实，不作为新获选模型或独立确认。

## 6. 既有ECISA3363回归

| 方案 | macro-F1 | accuracy | 负类召回 | 中性F1 |
|---|---:|---:|---:|---:|
| 第五轮／A原回退 | {e['round5']['macro_f1']:.4f} | {e['round5']['accuracy']*100:.2f}% | {e['round5']['per_class_recall'][0]*100:.2f}% | {e['round5']['per_class_f1'][1]:.4f} |
| CPU argmax对照 | {e['cpu_argmax']['macro_f1']:.4f} | {e['cpu_argmax']['accuracy']*100:.2f}% | {e['cpu_argmax']['per_class_recall'][0]*100:.2f}% | {e['cpu_argmax']['per_class_f1'][1]:.4f} |
| B锁定成本规则 | {e['B']['macro_f1']:.4f} | {e['B']['accuracy']*100:.2f}% | {e['B']['per_class_recall'][0]*100:.2f}% | {e['B']['per_class_f1'][1]:.4f} |

B提高了负类召回，但牺牲了整体macro-F1及其他类。两个集合都已经被前几轮及本次评审查看，均不能称为本轮独立新测试；本轮未在它们上再扫描阈值。

## 7. 成本曲线与证据

![训练侧选择与外层留出成本曲线](D:/Develop/AIC/_国赛优化/round7/decision_cost_curve.png)

曲线只由2876条训练数据产生，没有在821或3363上扫成本。彩色虚线是选规则用的OOF表现，实线是外层留出表现；右图黑色点线是要求本身的参考线。提高负召回要求时，中性F1显著下降，不能同时承诺各项都提高。

独立复核{verification['passed']}项通过，覆盖来源组／样本隔离、四个外层模型预测重放、内部选择重放、最终约束最优性、821／3363身份与规则重放、原主头身份和第五六轮关键文件不变。完整Python回归：**{passed} passed、{skipped} skipped、{failed} failed**，见regression.log及regression.xml。

工件根目录：D:/Develop/AIC/_国赛优化/round7。主脚本：algorithm/experiments/round7_decision_oof.py；只读复核：scripts/verify_round7_decision.py。已锁定目录禁止覆盖。

## 8. 下一步

保持第五轮为已选研究参考，追加这次成本敏感负结果，不把0.6802或评审的0.6738写成正式获选分数。下一步优先同批双人盲标中性／错误队列，使用同步音视频及统一标注说明，计算一致性、来源差异并保留原金标。没有真人参与前不填写κ或“人工复核完成”。

之后如再研究序数模型或新的成本约束，继续用固定的嵌套OOF，并另找未查看的匹配来源确认。703退出选择是今后纪律调整，不能补回已经使用掉的独立性。
"""
    target = root.parent / "第七轮决策层与OOF报告-20261116.md"
    target.write_text(report, encoding="utf-8")
    claims = """# 主张、证据与边界

| 主张 | 证据 | 可讲边界 |
|---|---|---|
| 第五轮历史点估计超过原拼接10% | macro-F1 0.56574→0.62661；相对+10.76%，差值区间[+0.0179,+0.1033] | 不保证未知来源至少+10%；不把整体归因于人脸 |
| 决策规则会改变中性与负类取舍 | 原主分支旧中性偏置−1；CPU argmax历史0.6802，85%成本选择仅0.6329 | argmax对照未满足训练侧负召回约束，不按历史高分改选 |
| 已退出703新选择、建立来源嵌套OOF | 2876条、107组、4外折×3内折；折内拟合全部CPU流程 | 架构有既往选择历史；OOF不消除所有来源差异 |
| 成本约束不等于未来保证 | B选择侧负召回85.76%，外层84.44%，历史90.49% | 消极情感类别并非临床风险；不改升级阈值 |
| 概率校准尚未验证有效 | 外层NLL/Brier/ECE略变差 | 不把softmax或温度拟合本身称为可靠概率 |
| 新数据收益仍需限定 | EmotionTalk未超越更强无新数据对照，停止继续扩充 | 这是当前配方的负结果，不是证明所有使用方式无效 |
| 可信度流程可复核 | 原文件哈希不变、逐样本预测、约束重放、负结果保留 | 人工复核与真实效果证据仍缺 |
"""
    (root / "claims_evidence_boundary.md").write_text(claims, encoding="utf-8")
    print(json.dumps({"report": str(target), "zero_bias_diagnostic": plain_metrics, "cost_curve": str(root / 'decision_cost_curve.png')}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    export(parser.parse_args().root)
