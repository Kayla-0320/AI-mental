"""第十二轮固定预算开发实验：鲁棒训练、知识消融、稀疏路由及轻量蒸馏。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch
from torch import nn
from experiments.round10_fusion import folds, take
from experiments.round11_common import OUTPUT as PREVIOUS, digest
from experiments.round11_features import alignment_feature, load_data
from experiments.round12_model import EvidenceHead, change_inputs, cross_source_memory, evidence_inputs
from experiments.round7_decision_oof import metric
from perception.knowledge_moe.knowledge import KnowledgeIndex

OUTPUT = PREVIOUS.parent / "round12"
CONDITIONS = ("clean", "missing_A", "missing_V", "missing_T", "text_only", "audio_mismatch", "all_missing")
KNOWLEDGE = Path(__file__).resolve().parents[1] / "perception/knowledge_moe/knowledge_seed.json"


def save(path: Path, value: Any, immutable: bool = True) -> None:
    """新轮工件原子写入，锁定结果拒绝静默覆盖。"""
    if path.exists() and immutable:
        raise ValueError(f"拒绝覆盖已锁定工件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def configurations() -> list[dict[str, Any]]:
    """一次固定11个候选，每个候选相同更新步数、样本与随机种子。"""
    return [dict(id=i, mode=m, augment=a, knowledge=k, memory=r, shuffle=s, distill=d)
            for i, m, a, k, r, s, d in (
                ("mlp_clean", "mlp", False, False, False, False, False),
                ("mlp_aug", "mlp", True, False, False, False, False),
                ("moe_clean", "dense", False, False, False, False, False),
                ("moe_aug", "dense", True, False, False, False, False),
                ("top2_aug", "top2", True, False, False, False, False),
                ("moe_knowledge", "dense", True, True, False, False, False),
                ("moe_knowledge_shuffle", "dense", True, True, False, True, False),
                ("moe_memory", "dense", True, False, True, False, False),
                ("moe_memory_shuffle", "dense", True, False, True, True, False),
                ("mlp_distilled", "mlp", True, False, False, False, True),
                ("moe_knowledge_memory", "dense", True, True, True, False, False))]


def normalize(text: str) -> str:
    """按原文字母数字归一化，排除精确文字重复的检索捷径。"""
    return "".join(c.casefold() for c in text if c.isalnum())


def donors_for(groups: np.ndarray, partitions: np.ndarray) -> np.ndarray:
    """供体必须在同一OOF留出分区且跨来源，不参考情绪金标。"""
    rng = np.random.default_rng(42)
    result = np.empty(len(groups), int)
    for i in range(len(groups)):
        candidates = np.flatnonzero((groups != groups[i]) & (partitions == partitions[i]))
        if not len(candidates):
            raise ValueError("同一OOF分区缺少不同来源供体")
        result[i] = rng.choice(candidates)
    return result


def prepare() -> dict[str, np.ndarray]:
    """只打开2876条开发数据，既有历史与外部测试不进入本轮加载器。"""
    data = load_data("pts")
    keep = ("sample_ids", "labels", "source_groups", "raw_text", "quality", "bge_cls", "asr_alignment")
    result = {k: data[k] for k in keep}
    del data
    parts = []
    for split in ("train", "selection"):
        with np.load(PREVIOUS / "sensevoice_deterministic_features" / f"{split}.npz", allow_pickle=False) as z:
            parts.append((z["sample_ids"], z["transcript"]))
    if not np.array_equal(result["sample_ids"], np.concatenate([p[0] for p in parts])):
        raise ValueError("ASR转录未对齐")
    result["transcript"] = np.concatenate([p[1] for p in parts])
    result["normalized_text"] = np.array([normalize(str(t)) for t in result["raw_text"]])
    result["knowledge"], _ = KnowledgeIndex(KNOWLEDGE, allow_candidates=True).encode(list(result["raw_text"].astype(str)))
    return result


def fold_inputs(data: dict[str, np.ndarray], fold: int, tr: np.ndarray, va: np.ndarray
                ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """复用旧轮严格嵌套的轻量证据，训练教师逐条来自未见该来源的7B专家。"""
    folder = PREVIOUS / "deployment_light/evidence_deterministic/family10"
    path = folder / f"fold{fold}.npz"
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(data["sample_ids"][tr], z["train_ids"]) or not np.array_equal(data["sample_ids"][va], z["valid_ids"]):
            raise ValueError("原嵌套缓存ID或划分已变化")
        training, validation = z["inner_evidence"], z["outer_evidence"]
    teacher_path = PREVIOUS / f"evidence_deterministic/family10/fold{fold}.npz"
    with np.load(teacher_path, allow_pickle=False) as z:
        if not np.array_equal(z["train_ids"], data["sample_ids"][tr]):
            raise ValueError("教师ID未对齐")
        teacher = z["inner_evidence"][:, :3]
    td, vd = take(data, tr), take(data, va)
    if set(td["source_groups"]) & set(vd["source_groups"]):
        raise ValueError("外层来源泄漏")
    partitions = np.full(len(tr), -1)
    for f, (_, held) in enumerate(folds(td, 3)):
        partitions[held] = f
    result = []
    memories = []
    for part, evidence, partition in ((td, training, partitions), (vd, validation, np.zeros(len(va), int))):
        x, mask = evidence_inputs(evidence, part["quality"])
        memory, neighbors = cross_source_memory(part["bge_cls"], td["bge_cls"], td["labels"],
            part["source_groups"], td["source_groups"], part["normalized_text"], td["normalized_text"])
        donors = donors_for(part["source_groups"], partition)
        alignment = np.stack([alignment_feature(str(a), str(b)) for a, b in
                              zip(part["raw_text"], part["transcript"][donors])]).astype(np.float32)
        result.append(dict(x=x, mask=mask, alignment=part["asr_alignment"].astype(np.float32),
                           knowledge=part["knowledge"], memory=memory, donors=donors,
                           mismatch_alignment=alignment, labels=part["labels"], groups=part["source_groups"],
                           sample_ids=part["sample_ids"]))
        memories.append(neighbors)
    # 错配一致性默认也用同一个OOF教师；不是给合成音画对赋整体硬标签。
    result[0]["teacher"] = teacher.astype(np.float32)
    audit = dict(fold=fold, train_groups=np.unique(td["source_groups"]).tolist(),
                 valid_groups=np.unique(vd["source_groups"]).tolist(),
                 cache_sha256=digest(path), teacher_cache_sha256=digest(teacher_path),
                 memory_bank_ids=td["sample_ids"].tolist(), memory_train_neighbors=memories[0].tolist(),
                 memory_valid_neighbors=memories[1].tolist(), train_donors=result[0]["donors"].tolist(),
                 valid_donors=result[1]["donors"].tolist(), inner_partition=partitions.tolist())
    return result[0], result[1], audit


def altered(part: dict[str, Any], condition: str) -> tuple[np.ndarray, ...]:
    """训练与验证调用完全相同的观测变换。"""
    return change_inputs(*(part[k] for k in ("x", "mask", "alignment", "knowledge", "memory")),
                         condition, part["donors"], part["mismatch_alignment"])


def tensor_inputs(values: tuple[np.ndarray, ...]) -> tuple[torch.Tensor, ...]:
    """把只含观测的数组转换到CPU张量。"""
    return tuple(torch.from_numpy(v) for v in values)


def train(config: dict[str, Any], training: dict[str, Any], validation: dict[str, Any],
          epochs: int, folder: Path) -> dict[str, np.ndarray]:
    """固定60轮；缺失样本保留原场景监督，错配只作OOF软教师一致性。"""
    torch.manual_seed(42)
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    training, validation = dict(training), dict(validation)
    if config["shuffle"]:
        # 保留命中率及维数，独立打乱训练与验证行，不参考标签。
        for part in (training, validation):
            order = rng.permutation(len(part["x"]))
            if config["knowledge"]:
                part["knowledge"] = part["knowledge"][order]
            if config["memory"]:
                part["memory"] = part["memory"][order]
    y = torch.tensor(training["labels"], dtype=torch.long)
    counts = np.bincount(training["labels"], minlength=3)
    prior = counts / counts.sum()
    model = EvidenceHead(config, training["knowledge"].shape[1], prior)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    weights = torch.tensor(np.sqrt(len(y) / (3 * counts)), dtype=torch.float32)
    teacher = torch.tensor(training["teacher"])
    tensors = {c: tensor_inputs(altered(training, c)) for c in CONDITIONS}
    policy = np.array([.5, .1, .1, .1, .1, .1])
    soft_rows = 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(y))
        for start in range(0, len(y), 128):
            ix = order[start:start + 128]
            condition = str(rng.choice(CONDITIONS[:6], p=policy)) if config["augment"] else "clean"
            logp, gate = model(*(v[ix] for v in tensors[condition]))
            if condition == "audio_mismatch":
                loss = nn.functional.kl_div(logp, teacher[ix], reduction="batchmean")
                soft_rows += len(ix)
            else:
                loss = nn.functional.nll_loss(logp, y[ix], weight=weights)
                if config["distill"]:
                    loss = .75 * loss + .25 * nn.functional.kl_div(logp, teacher[ix], reduction="batchmean")
            if config["mode"] != "mlp":
                # 可用专家的平均占用作弱平衡；不把缺失专家强制路由为非零。
                eligible = (gate > 0).float().mean(0)
                target = eligible / eligible.sum().clamp_min(1)
                loss += .01 * (gate.mean(0) - target).square().sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
    model.eval()
    probabilities = {}
    routing = {}
    with torch.inference_mode():
        for condition in CONDITIONS:
            lp, gate = model(*tensor_inputs(altered(validation, condition)))
            probabilities[condition] = lp.exp().numpy()
            routing[condition] = dict(mean_weights=gate.mean(0).tolist(),
                mean_executed_heads=float((gate > 0).sum(1).float().mean()),
                all_missing_prior=prior.tolist() if condition == "all_missing" else None)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), config=config, knowledge_dim=training["knowledge"].shape[1],
                    prior=prior, epochs=epochs, input_contract="light frozen T/A/V expert evidence + observed alignment; no label input",
                    train_ids=training["sample_ids"], valid_ids=validation["sample_ids"]), folder / "head.pt")
    save(folder / "training.json", dict(parameter_count=sum(p.numel() for p in model.parameters()),
        mismatch_soft_target_rows=soft_rows, train_n=len(y), epochs=epochs, routing=routing))
    np.savez_compressed(folder / "predictions.npz", sample_ids=validation["sample_ids"], **probabilities)
    return probabilities


def paired_interval(y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: np.ndarray) -> list[float]:
    """按来源配对重采样；区间条件于已接触的开发数据。"""
    rng = np.random.default_rng(42)
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    differences = []
    for _ in range(1000):
        ix = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        differences.append(metric(y[ix], b[ix].argmax(1))["macro_f1"] - metric(y[ix], a[ix].argmax(1))["macro_f1"])
    return np.quantile(differences, [.025, .975]).tolist()


def annotation_kit(data: dict[str, np.ndarray], output: Path) -> None:
    """生成双人独立标注模板，保留原标签在独立对照表中；不伪造人工审核。"""
    rng = np.random.default_rng(42)
    ids = []
    for label in range(3):
        pool = np.flatnonzero(data["labels"] == label)
        ids.extend(rng.choice(pool, min(40, len(pool)), replace=False).tolist())
    rng.shuffle(ids)
    output.mkdir(parents=True, exist_ok=True)
    for person in ("A", "B"):
        with (output / f"annotator_{person}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["blind_id", "text", "M3_negative_neutral_positive", "uncertain", "comment"])
            writer.writerows([[f"R12_{j:03d}", data["raw_text"][i], "", "", ""] for j, i in enumerate(ids)])
    save(output / "reference_private.json", dict(rows=[dict(blind_id=f"R12_{j:03d}",
        sample_id=str(data["sample_ids"][i]), original_label=int(data["labels"][i])) for j, i in enumerate(ids)],
        status="awaiting_two_independent_human_annotations", modality="text_only_T_reference_not_M3_replacement"))


def report(data: dict[str, np.ndarray], configs: list[dict[str, Any]], predictions: dict[str, dict[str, np.ndarray]]) -> None:
    """完整记录正负对照，开发筛选不触发历史或外部测试复评。"""
    rows = []
    baseline = predictions["mlp_clean"]
    for config in configs:
        metrics = {c: metric(data["labels"], predictions[config["id"]][c].argmax(1)) for c in CONDITIONS}
        robust = float(np.mean([metrics[c]["macro_f1"] for c in CONDITIONS[1:6]]))
        rows.append(dict(config=config, metrics=metrics, robustness_mean=robust,
            clean_delta_interval=paired_interval(data["labels"], baseline["clean"], predictions[config["id"]]["clean"], data["source_groups"])))
    base_clean = rows[0]["metrics"]["clean"]["macro_f1"]
    eligible = [r for r in rows if not r["config"]["shuffle"] and r["metrics"]["clean"]["macro_f1"] >= base_clean - .01]
    winner = max(eligible, key=lambda r: .5 * r["metrics"]["clean"]["macro_f1"] + .5 * r["robustness_mean"])
    comparisons = []
    for a, b in (("mlp_clean", "mlp_aug"), ("moe_clean", "moe_aug"), ("moe_aug", "top2_aug"),
                 ("moe_aug", "moe_knowledge"), ("moe_knowledge_shuffle", "moe_knowledge"),
                 ("moe_aug", "moe_memory"), ("moe_memory_shuffle", "moe_memory"), ("mlp_aug", "mlp_distilled")):
        comparisons.append(dict(control=a, treatment=b, clean_paired_interval=paired_interval(data["labels"],
            predictions[a]["clean"], predictions[b]["clean"], data["source_groups"])))
    save(OUTPUT / "results.json", dict(rows=rows, comparisons=comparisons, development_candidate=winner["config"]["id"],
        selection_rule="clean F1 >= same-input mlp_clean -.01; maximize .5clean+.5mean5stress; shuffle controls excluded",
        independent_validation=False, promotion_to_production=False, seed=42,
        limitations=["previously exposed development sources; not new sealed validation", "no historical821/calibration703/external1929 evaluation",
                    "knowledge seed is engineering candidate lexicon; not COMET inference or expert-reviewed graph",
                    "memory uses train-source human labels; distinct from external commonsense knowledge",
                    "distillation target is nested source-OOF7B M expert, not the final fused teacher",
                    "frozen cached expert inputs; no raw-media end-to-end speed claim", "automatic ASR not used for text branch; alignment is an ASR proxy",
                    "text-only annotation template does not establish multimodal label consistency", "single head seed; bootstrap uncorrected for selection/multiple comparisons"]))
    lines = ["# 第十二轮鲁棒训练、知识增强与MoE探索报告", "", "本轮已完成固定预算开发实验。原第十一轮模型及测试结果保持封存；本轮结果仅用于下一步候选筛选，不能作为新独立验证或线上替换依据。", "",
        "输入为无生成式语言模型的轻量T/A/V冻结专家证据。每个外层训练集内的专家输入来自原3折来源OOF；4个外层留出来源均未进入对应专家/新融合头训练。仍沿用此前已接触开发来源，因此不是新的封存来源。", "",
        "全部候选固定种子42、60轮、相同训练样本和更新次数；直接argmax，无事后截距网格。训练缺失保留原场景目标；音轨错配仅作原干净OOF联合专家软分布一致性，不赋合成硬标签。", "",
        "|候选|干净M3|缺A|缺V|缺T|仅T|音频错配|五压力均值|中性F1(干净)|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["metrics"][c]["macro_f1"] for c in CONDITIONS[:6]]
        lines.append("|" + row["config"]["id"] + "|" + "|".join(f"{v:.6f}" for v in values) +
                     f"|{row['robustness_mean']:.6f}|{row['metrics']['clean']['per_class_f1'][1]:.6f}|")
    lines += ["", f"固定规则开发候选：**{winner['config']['id']}**。该选择经过开发多候选筛选，仍需新来源独立确认。", "",
        "知识分为两个不同条件：现有六节点工程候选词库（未专业审核），以及只含当前外层训练来源的BGE监督检索记忆。检索排除同来源和同归一化原文，保存每条近邻ID；未接入完整COMET/ATOMIC知识推理。知识/记忆打乱对照保留维数和覆盖率，不把额外容量直接归因为知识有效。", "",
        "Top-2实际只调用所选M专家头；冻结编码器与模态证据提取成本仍然存在。本轮未实测原媒体端到端延迟，不声称Top-2整体加速。蒸馏仅将7B的来源OOF联合专家软分布传给轻量融合头，不能声称完整7B系统已成功蒸馏。", "",
        "标注审计已生成两份独立空白模板及隔离参考表，待真人填写。它是文本侧T3复核准备，不替代M3多模态审阅；尚无多人一致性成绩。未修改happy或任何原始标签。", "",
        "跨天个人趋势和真实线上ASR/说话人关联需要新增连续观测与独立采集，本轮未宣称完成。", "",
        "实现参考：[COSMIC](https://aclanthology.org/2020.findings-emnlp.224/)讨论常识表示；[KET](https://aclanthology.org/D19-1016/)讨论外部常识注意力；[EMOE](https://openaccess.thecvf.com/content/CVPR2025/papers/Fang_EMOE_Modality-Specific_Enhanced_Dynamic_Emotion_Experts_CVPR_2025_paper.pdf)讨论模态专家与单模态蒸馏。这些是已有方法背景，本项目的候选词库/监督记忆不等同于复现这些系统。", "",
        "复现：在项目algorithm目录运行 `D:/Develop/AIC/.venv/Scripts/python.exe -m experiments.round12_explore`。完整逐条OOF概率、训练来源审计、邻居与供体索引、各类召回/混淆矩阵、模型权重和条件区间均在同目录工件中。"]
    (OUTPUT / "第十二轮鲁棒训练与知识MoE探索报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """在跑分前锁定协议和输入摘要，支持核对来源后恢复未完成候选。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    global OUTPUT
    if args.smoke:
        OUTPUT = OUTPUT / "smoke"
    configs = configurations()
    epochs = 1 if args.smoke else 60
    plan = dict(configs=configs, epochs=epochs, seed=42, conditions=list(CONDITIONS),
        data="existing development2876 only; 4source outer/3source inner caches; no fresh independent validation",
        selection="no shuffled control; clean >= mlp_clean-.01; max .5clean+.5mean5stress; no test driven search",
        mismatch="train donors same inner OOF fold and different source; validation donors same outer holdout and different source; soft teacher only",
        augmentation_probabilities=dict(clean=.5, missing_A=.1, missing_V=.1, missing_T=.1, text_only=.1, audio_mismatch=.1),
        deployment="light source-OOF evidence; KD teacher source-OOF7B M expert; no new raw media inference timing",
        code_sha256={"runner":digest(Path(__file__)), "model":digest(Path(__file__).with_name("round12_model.py")),
                     "knowledge":digest(KNOWLEDGE)}, knowledge_status="offline unreviewed engineering candidates only")
    lock = OUTPUT / "preregister.json"
    if lock.exists():
        if json.loads(lock.read_text("utf-8")) != plan:
            raise ValueError("代码或预注册条件变化，请使用新的输出目录")
    else:
        save(lock, plan)
    data = prepare()
    all_predictions = {c["id"]: {s: np.full((len(data["labels"]), 3), np.nan) for s in CONDITIONS} for c in configs}
    started = time.monotonic()
    for f, (tr, va) in enumerate(folds(data)):
        training, validation, audit = fold_inputs(data, f, tr, va)
        path = OUTPUT / "audit" / f"fold{f}.json"
        if not path.exists():
            save(path, audit)
        elif json.loads(path.read_text("utf-8")) != audit:
            raise ValueError("来源审计或输入缓存发生变化")
        for config in configs:
            folder = OUTPUT / "candidates" / config["id"] / f"fold{f}"
            path = folder / "predictions.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    if not np.array_equal(z["sample_ids"], data["sample_ids"][va]):
                        raise ValueError("断点预测ID变化")
                    p = {c: z[c] for c in CONDITIONS}
            else:
                p = train(config, training, validation, epochs, folder)
            for condition in CONDITIONS:
                all_predictions[config["id"]][condition][va] = p[condition]
            print(json.dumps(dict(fold=f, candidate=config["id"], seconds=round(time.monotonic()-started, 1))), flush=True)
    for config in configs:
        p = all_predictions[config["id"]]
        if not all(np.isfinite(a).all() and np.allclose(a.sum(1), 1, atol=1e-6) for a in p.values()):
            raise ValueError("OOF覆盖或概率归一化不完整")
        np.savez_compressed(OUTPUT / "candidates" / config["id"] / "oof.npz", sample_ids=data["sample_ids"],
                            labels=data["labels"], source_groups=data["source_groups"], **p)
    if not args.smoke:
        annotation_kit(data, OUTPUT / "annotation_audit")
        if not (OUTPUT / "results.json").exists():
            report(data, configs, all_predictions)
    save(OUTPUT / "run_complete.json", dict(complete=True, seconds=round(time.monotonic()-started, 1),
        n=len(data["labels"]), configs=len(configs), folds=4, independent_validation=False), immutable=False)


if __name__ == "__main__":
    main()
