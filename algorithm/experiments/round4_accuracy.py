"""来源分组开发评测、知识对照与中性类误差分析；不自动替换业务模型。"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from experiments.round2_moe_train import forward, grouped_delta_ci, load_split, metrics, seed_all, subset, temperature_fit, tensors
from perception.knowledge_moe.features import FeatureStandardizer, save_json, sha256_file
from perception.knowledge_moe.knowledge import KnowledgeIndex
from perception.knowledge_moe.model import ConcatenationHead, QualityAwareMoE

NAMES = ["text", "audio", "visual"]


def merge(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """共同拼接全部逐样本字段。"""
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def folds_for(data: dict[str, np.ndarray], count: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """固定种子分组分层；断言每折三类齐全且没有来源泄漏。"""
    splits = list(StratifiedGroupKFold(count, shuffle=True, random_state=42).split(
        data["labels"], data["labels"], data["source_groups"]))
    for train, valid in splits:
        if set(data["source_groups"][train]) & set(data["source_groups"][valid]):
            raise ValueError("来源泄漏")
        if set(data["labels"][train]) != {0, 1, 2} or set(data["labels"][valid]) != {0, 1, 2}:
            raise ValueError("折内类别缺失")
    return splits


def linear_features(data: dict[str, np.ndarray], names: list[str], scaler: FeatureStandardizer) -> np.ndarray:
    """使用本折训练统计，按模态维数平衡尺度，并显式附加掩码。"""
    columns = [NAMES.index(name) for name in names]
    q = data["quality"][:, columns]
    mask = data["modality_mask"][:, columns] & (q > 0)
    values = scaler.transform({name: data[name] for name in names}, mask)
    return np.column_stack([*(values[name] * q[:, j:j+1] / np.sqrt(values[name].shape[1])
                             for j, name in enumerate(names)), mask, q]).astype(np.float64)


def knowledge_batch(batch: tuple, mode: str) -> tuple:
    """知识负对照保持命中掩码、维度与参数量；不接触标签。"""
    x, mask, q, kg, present, y = batch
    kg, present = kg.clone(), present.clone()
    if mode == "null":
        kg.zero_(); present.zero_()
    elif mode == "shuffled":
        indices = torch.where(present)[0]
        order = np.random.default_rng(42).permutation(len(indices))
        kg[indices] = kg[indices[torch.tensor(order.copy())]]
    return x, mask, q, kg, present, y


def fit_candidate(config: dict[str, Any], train: dict[str, np.ndarray], index: KnowledgeIndex) -> dict[str, Any]:
    """训练固定候选，不用评估折选择训练轮数或归一化参数。"""
    seed_all()
    names = config["names"]
    mask = train["modality_mask"][:, [NAMES.index(name) for name in names]]
    scaler = FeatureStandardizer().fit({name: train[name] for name in names}, mask)
    if config["kind"] == "linear":
        model = LogisticRegression(C=config["C"], class_weight=config["weight"], max_iter=700,
                                   random_state=42, solver="lbfgs")
        model.fit(linear_features(train, names, scaler), train["labels"])
    else:
        kd = index.dimensions if config["kind"] == "moe" else 0
        constructor = QualityAwareMoE if config["kind"] == "moe" else ConcatenationHead
        model = constructor({name: train[name].shape[1] for name in names}, 3, config["hidden"], kd)
        batch = tensors(train, names, scaler, index)
        if kd:
            batch = knowledge_batch(batch, config["knowledge"])
        counts = torch.bincount(batch[-1], minlength=3).float()
        criterion = nn.CrossEntropyLoss(weight=len(train["labels"])/(3*counts))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
        for _ in range(30):
            model.train()
            order = torch.randperm(len(train["labels"]))
            for start in range(0, len(order), 64):
                ix = order[start:start+64]
                logits, _, balance, valid = forward(model, batch, ix, augment=True)
                if not valid.any():
                    continue
                loss = criterion(logits, batch[-1][ix][valid]) + .01*balance
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
        model.eval()
    return {"config": config, "scaler": scaler, "model": model}


def predict(fitted: dict[str, Any], data: dict[str, np.ndarray], index: KnowledgeIndex) -> np.ndarray:
    """预测固定模型；全缺失返回NaN，由调用方显式统计弃权。"""
    config, scaler, model = fitted["config"], fitted["scaler"], fitted["model"]
    if config["kind"] == "linear":
        columns = [NAMES.index(name) for name in config["names"]]
        valid = (data["modality_mask"][:, columns] & (data["quality"][:, columns] > 0)).any(axis=1)
        output = np.full((len(data["labels"]), 3), np.nan)
        if valid.any():
            output[valid] = model.decision_function(linear_features(data, config["names"], scaler)[valid])
        return output
    batch = tensors(data, config["names"], scaler, index)
    if config["kind"] == "moe":
        batch = knowledge_batch(batch, config["knowledge"])
    with torch.inference_mode():
        logits, _, _, valid = forward(model, batch, torch.arange(len(data["labels"])))
    output = np.full((len(data["labels"]), 3), np.nan)
    output[valid.numpy()] = logits.numpy()
    return output


def summarize(labels: np.ndarray, logits: np.ndarray, temperature: float = 1.) -> dict[str, Any]:
    """统一报告逐类精确率、召回、F1及混淆矩阵。"""
    valid = np.isfinite(logits).all(axis=1)
    result = metrics(labels[valid], logits[valid], 3, temperature)
    result["coverage"] = float(valid.mean())
    if valid.any():
        pred = logits[valid].argmax(axis=1)
        precision, recall, f1, count = precision_recall_fscore_support(labels[valid], pred, labels=[0,1,2], zero_division=0)
        result.update(confusion_matrix=confusion_matrix(labels[valid], pred, labels=[0,1,2]).tolist(),
                      per_class_precision=precision.tolist(), per_class_f1=f1.tolist(), support=count.tolist())
    return result


def choose_offset(labels: np.ndarray, logits: np.ndarray) -> tuple[float, dict[str, Any]]:
    """仅在开发OOF上选择中性偏置；记录选择分数而非独立性能估计。"""
    raw = summarize(labels, logits)
    candidates = []
    for offset in [-.6, -.3, 0., .3, .6, .9, 1.2]:
        values = logits.copy(); values[:,1] += offset
        score = summarize(labels, values)
        if score["accuracy"] >= raw["accuracy"]-.01:
            candidates.append((score["macro_f1"], score["accuracy"], -abs(offset), offset, score))
    best = max(candidates, key=lambda row: row[:3])
    return best[3], best[4]


def run(bundle: Path, previous: Path, output: Path) -> dict[str, Any]:
    """冻结候选和折，完成开发选择后才读历史测试；历史结果仅作诊断。"""
    output.mkdir(parents=True, exist_ok=True)
    if (output/"model_results.json").exists():
        raise ValueError("结果已存在，拒绝覆盖；新方案需要新实验目录")
    development = merge([load_split(bundle/f"{name}.npz") for name in ["train", "selection"]])
    fold_pairs = folds_for(development)
    configs = [{"id": f"lr_{name}_{c}_{weight}", "kind": "linear", "names": names, "C": c, "weight": weight}
               for name,names in [("text",["text"]),("concat",NAMES)] for c in [.1,1.,10.,100.] for weight in [None,"balanced"]]
    configs += [{"id":"neural_concat", "kind":"concat", "names":NAMES,"hidden":64}]
    configs += [{"id":"moe_"+mode,"kind":"moe","names":NAMES,"hidden":32,"knowledge":mode}
                for mode in ["null","real","shuffled"]]
    protocol = {"seed":42,"task":"sentiment_polarity_3","selection":"4-fold source-group OOF; max macro-F1; neutral offset accuracy loss at most .01 per candidate",
                "configs":configs,"neural_epochs":30,"early_stopping":False,"neutral_offset_grid":[-.6,-.3,0.,.3,.6,.9,1.2],
                "calibration":"separate round3 calibration split; temperature fits NLL only",
                "historical_test":"already examined in round3; diagnostics only, not new independent confirmation",
                "external_validation":"not_available","input_sha256":sha256_file(bundle/"metadata.json"),
                "code_sha256":sha256_file(Path(__file__)),
                "folds":[{"train_ids":development["sample_ids"][tr].tolist(),"valid_ids":development["sample_ids"][va].tolist(),
                          "valid_class_counts":np.bincount(development["labels"][va],minlength=3).tolist()} for tr,va in fold_pairs]}
    save_json(output/"preregister.json",protocol)
    index = KnowledgeIndex(previous/"user_attested_knowledge.json",allow_user_attested=True)
    with np.load(previous/"knowledge_vectors.npz",allow_pickle=False) as nodes:
        index.attach_semantic_vectors(dict(zip(nodes["node_ids"].astype(str),nodes["vectors"])))
    oof, results, fitted_models = {}, {}, {}
    for config in configs:
        logits = np.full((len(development["labels"]),3),np.nan)
        fold_scores = []
        for tr,va in fold_pairs:
            fitted = fit_candidate(config,subset(development,tr),index)
            logits[va] = predict(fitted,subset(development,va),index)
            fold_scores.append(summarize(development["labels"][va],logits[va]))
        if not np.isfinite(logits).all():
            raise ValueError("开发OOF包含无效预测")
        offset, score = choose_offset(development["labels"],logits)
        oof[config["id"]] = logits
        results[config["id"]] = {"raw_oof":summarize(development["labels"],logits),"selection_oof":score,
                                   "neutral_logit_offset":offset,"folds_raw":fold_scores}
        print(json.dumps({"candidate":config["id"],"macro_f1":score["macro_f1"],"accuracy":score["accuracy"],"neutral_recall":score["per_class_recall"][1]}),flush=True)
        save_json(output/"development_progress.json",results)
    winner = max(results,key=lambda key:(results[key]["selection_oof"]["macro_f1"],results[key]["selection_oof"]["accuracy"]))
    save_json(output/"selection_locked.json",{"winner":winner,"results":results,"policy":"no historical test read before this lock"})
    calibration = load_split(bundle/"calibration.npz")
    historical = load_split(bundle/"test.npz")
    for part in [calibration,historical]:
        if set(part["source_groups"]) & set(development["source_groups"]):
            raise ValueError("来源组泄漏")
    if set(calibration["source_groups"]) & set(historical["source_groups"]):
        raise ValueError("校准与历史测试来源交叉")
    evaluated = {}
    # 同时保留所有固定候选的历史诊断，禁止按其结果重新选择winner。
    for config in configs:
        fitted = fit_candidate(config,development,index)
        values = predict(fitted,calibration,index)
        offset = results[config["id"]]["neutral_logit_offset"]
        values[:,1] += offset
        temperature = temperature_fit(calibration["labels"],values,3)
        test_logits = predict(fitted,historical,index); test_logits[:,1] += offset
        evaluated[config["id"]] = {"raw":summarize(historical["labels"],test_logits),
                                    "calibrated":summarize(historical["labels"],test_logits,temperature),
                                    "temperature":temperature,"parameters":sum(p.numel() for p in fitted["model"].parameters()) if config["kind"]!="linear" else int(fitted["model"].coef_.size+fitted["model"].intercept_.size)}
        np.savez_compressed(output/(config["id"]+"_predictions.npz"),logits=test_logits,labels=historical["labels"],sample_ids=historical["sample_ids"])
        if config["id"] == winner:
            fitted_models[winner] = fitted
            with (output/"selected_research_model.pkl").open("wb") as handle:
                pickle.dump({**fitted,"neutral_offset":offset,"temperature":temperature,"research_only":True},handle)
    comparisons = {}
    with np.load(previous/"concat"/"test_predictions.npz",allow_pickle=False) as old:
        if not np.array_equal(old["sample_ids"],historical["sample_ids"]):
            raise ValueError("历史预测ID顺序不一致")
        reference = old["logits"].argmax(axis=1)
    selected_logits = np.load(output/(winner+"_predictions.npz"))["logits"]
    selected_pred = selected_logits.argmax(axis=1)
    comparisons["selected_minus_round3_concat"]={"delta_macro_f1":evaluated[winner]["raw"]["macro_f1"]-summarize(historical["labels"],old_logits(previous))["macro_f1"],
        "ci95":grouped_delta_ci(historical["labels"],selected_pred,reference,historical["source_groups"],3)}
    for a,b in [("moe_real","moe_null"),("moe_real","moe_shuffled")]:
        with np.load(output/(a+"_predictions.npz")) as za,np.load(output/(b+"_predictions.npz")) as zb:
            comparisons[a+"_minus_"+b]={"delta_macro_f1":evaluated[a]["raw"]["macro_f1"]-evaluated[b]["raw"]["macro_f1"],
                "ci95":grouped_delta_ci(historical["labels"],za["logits"].argmax(1),zb["logits"].argmax(1),historical["source_groups"],3)}
    error_rows = [{"id":str(historical["sample_ids"][i]),"text":str(historical["raw_text"][i]),"true":int(historical["labels"][i]),
                   "round3_prediction":int(reference[i]),"selected_prediction":int(selected_pred[i]),"source_group":str(historical["source_groups"][i]),
                   "human_label_review":"pending_not_model_generated_gold"}
                  for i in range(len(reference)) if historical["labels"][i]==1 or reference[i]!=historical["labels"][i]]
    save_json(output/"error_review_queue.json",error_rows)
    report={"protocol":protocol,"winner":winner,"development":results,"historical_diagnostic":evaluated,"comparisons":comparisons,
            "human_label_review_completed":False,"deployed":False,"knowledge_match_fraction":float(index.encode(development["raw_text"].tolist())[1].mean()),
            "limitations":["OOF participates in hyperparameter selection; selected OOF is optimistic", "historical test has been examined; new external confirmation still required",
                           "same seed and one four-fold partition", "knowledge controls have matched parameters; concat versus MoE not parameter matched",
                           "class labels describe clip sentiment, not mental health"]}
    save_json(output/"model_results.json",report)
    print(json.dumps({"winner":winner,"historical":evaluated[winner],"comparisons":comparisons}),flush=True)
    return report


def old_logits(previous: Path) -> np.ndarray:
    """读取冻结历史拼接预测。"""
    with np.load(previous/"concat"/"test_predictions.npz",allow_pickle=False) as values:
        return values["logits"].copy()


def main() -> None:
    """命令行入口。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--previous",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();run(args.bundle,args.previous,args.output)


if __name__=="__main__":
    main()
