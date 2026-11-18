"""EmotionTalk 中性辅助训练的匹配对照；开发锁定后再评测历史及外部留出。"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any

for _key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_key] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from transformers import AutoTokenizer

from experiments.round2_moe_train import load_split, grouped_delta_ci
from experiments.round4_accuracy import merge
from experiments.round5_finetune import MultimodalBERT, Rows, collate, external_data, logits_for, seed_all, train
from experiments.round5_face_search import features
from experiments.round5_frozen_search import summary
from experiments.round5_ensemble import normalized, select_bias
from experiments.round5_availability_router import route_scores
from perception.knowledge_moe.features import FeatureStandardizer, save_json, sha256_file


def binary_logits(logits: torch.Tensor) -> torch.Tensor:
    """把非中性的两个极性边缘化，辅助训练不指定其正负类别。"""
    return torch.stack([torch.logsumexp(logits[:, [0, 2]].float(), dim=1), logits[:, 1].float()], dim=1)


def binary_summary(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """二分类情绪中性指标，独立于目标三分类极性指标。"""
    values = torch.tensor(scores)
    pred = binary_logits(values).argmax(1).numpy()
    return {"n": len(labels), "accuracy": float(accuracy_score(labels, pred)),
            "macro_f1": float(f1_score(labels, pred, labels=[0, 1], average="macro", zero_division=0)),
            "neutral_recall": float(recall_score(labels, pred, pos_label=1, zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist()}


def auxiliary_epoch(model: MultimodalBERT, rows: Rows, output: Path) -> None:
    """固定一轮二分类辅助训练，不按来源验证或测试成绩选择轮次。"""
    seed_all()
    loader = DataLoader(rows, batch_size=32, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(42))
    counts = torch.bincount(rows.y, minlength=2).float().cuda()
    weights = (len(rows) / (2 * counts)).sqrt()
    weights /= weights.mean()
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 5e-6},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("encoder.")], "lr": 5e-5}
    ], weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda")
    total = 0.
    model.train()
    for step, row in enumerate(loader, 1):
        inputs = {key: value.cuda() for key, value in row.items()}
        labels = inputs.pop("labels")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = nn.functional.cross_entropy(binary_logits(model(**inputs)), labels, weight=weights, label_smoothing=.03)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(labels)
        if step % 100 == 0:
            print(json.dumps({"auxiliary_step": step, "steps": len(loader), "loss": total / min(step * 32, len(rows))}), flush=True)
    save_json(output, {"epochs": 1, "n": len(rows), "loss": total / len(rows), "lr": 5e-6, "labels": "binary text emotional neutrality only"})


def encode_mean(model: MultimodalBERT, data: dict[str, np.ndarray], tokenizer: Any) -> np.ndarray:
    """从实际适配的编码器提取均值表征，推理时不读标签。"""
    texts = data["raw_text"].astype(str).tolist()
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            batch = tokenizer(texts[start:start + 64], truncation=True, max_length=128, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = model.encoder(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1)
            output.append(((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu().numpy())
    return np.concatenate(output)


def infer(model: MultimodalBERT, data: dict[str, np.ndarray], tokenizer: Any, scaler: FeatureStandardizer) -> np.ndarray:
    """评估现有模型结构，保持原模态可用性掩码。"""
    return logits_for(model, DataLoader(Rows(data, tokenizer, scaler), batch_size=64, collate_fn=collate))


def primary_scores(data: dict[str, np.ndarray], head: dict[str, Any]) -> np.ndarray:
    """使用仅在训练样本上拟合的标准化和分类头。"""
    x, _ = features(data, head["names"], head["scalers"], .5)
    values = normalized(head["model"].decision_function(x))
    values[:, 1] += head["bias"]
    return values


def run(root: Path, previous: Path, bundle: Path, encoder: Path, output: Path) -> dict[str, Any]:
    """匹配起始权重、训练配方和融合头；只按既有开发集选择。"""
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection_locked.json").exists():
        raise ValueError("该轮已锁定，拒绝覆盖或重新选型")
    seed_all()
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(encoder, local_files_only=True)
    scaler = FeatureStandardizer()
    scaler.parameters = json.loads((previous / "finetune/side_standardizer.json").read_text(encoding="utf-8"))
    base = previous / "finetune/ecisa_1e-05_0.5.pt"
    train_data = merge([load_split(previous / "face_features" / (split + ".npz")) for split in ["train", "selection"]])
    valid = load_split(previous / "face_features/calibration.npz")
    tr = Rows(train_data, tokenizer, scaler)
    va = Rows(valid, tokenizer, scaler)
    neutral_root = root / "neutral_data"
    configs = ["no_new_auxiliary", "all_emotiontalk", "consensus_emotiontalk"]
    ft = json.loads((previous / "finetune/results.json").read_text(encoding="utf-8"))
    old_lock = json.loads((previous / "availability_router/selection_locked.json").read_text(encoding="utf-8"))
    threshold = old_lock["minimum_face_coverage"]
    keep_metric = old_lock["choices"][str(threshold)]["development"]
    protocol = {"seed": 42, "conditions": configs, "base_checkpoint_sha256": sha256_file(base),
                "target_training": "original train+selection2876; matched continuing adaptation for every condition",
                "target_validation": "historically reused703; no claim of independent calibration",
                "auxiliary": "one fixed epoch of text emotional neutrality vs other; 5e-6 encoder,5e-5 head; no polarity remapping",
                "consensus": "at least4/5 annotators agree on binary neutral status; training only",
                "target_config": {"lr": 1e-5, "weight_power": .5, "epochs_max": 6},
                "primary_head": "replace frozen RoBERTa mean with adapted mean; retain BGE,CLIP,MFCC,face; fixed C=.1,balanced,face_weight=.5",
                "routing": {"minimum_face_coverage": threshold, "unchanged_availability_rule": True},
                "selection": "max target development macro-F1 then accuracy; previous round5 router included as keep-current candidate",
                "historical": "same821, already examined, no use for this round checkpoint or configuration selection",
                "external_neutral": "official EmotionTalk test groups G03/G15, dedup1890, only final reporting",
                "code_sha256": sha256_file(Path(__file__)), "data_manifest_sha256": sha256_file(neutral_root / "manifest.json"),
                "deployed": False}
    save_json(output / "preregister.json", protocol)
    development: dict[str, Any] = {"keep_round5": {"development": keep_metric}}
    for condition in configs:
        candidate = output / condition
        candidate.mkdir(exist_ok=True)
        seed_all()
        model = MultimodalBERT(encoder).cuda()
        model.load_state_dict(torch.load(base, map_location="cpu", weights_only=True))
        if condition != "no_new_auxiliary":
            rows = [json.loads(line) for line in (neutral_root / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            if condition == "consensus_emotiontalk":
                rows = [row for row in rows if row["binary_vote_agreement"] >= .8]
            source = candidate / "auxiliary_train.jsonl"
            source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            auxiliary_epoch(model, Rows(external_data(source), tokenizer, scaler), candidate / "auxiliary_training.json")
            torch.save(model.state_dict(), candidate / "after_auxiliary.pt")
        state, bias, training = train(model, tr, va, {"lr": 1e-5, "weight_power": .5}, 6, candidate, "target")
        torch.save(state, candidate / "weights.pt")
        del state
        a = {**train_data, "tuned_mean": encode_mean(model, train_data, tokenizer)}
        b = {**valid, "tuned_mean": encode_mean(model, valid, tokenizer)}
        names = ["tuned_mean", "bge_cls", "visual", "audio", "face_summary"]
        x, scalers = features(a, names, face_weight=.5)
        v, _ = features(b, names, scalers, .5)
        head_model = LogisticRegression(C=.1, class_weight="balanced", max_iter=700, random_state=42).fit(x, a["labels"])
        raw = normalized(head_model.decision_function(v))
        primary_bias, _ = select_bias(valid["labels"], raw)
        primary = raw.copy()
        primary[:, 1] += primary_bias
        fallback = infer(model, valid, tokenizer, scaler)
        fallback[:, 1] += bias
        routed, mask = route_scores(valid, primary, fallback, threshold)
        metric = summary(valid["labels"], routed)
        head = {"names": names, "scalers": scalers, "model": head_model, "bias": primary_bias}
        with (candidate / "primary.pkl").open("wb") as handle:
            pickle.dump(head, handle)
        np.savez_compressed(candidate / "development_predictions.npz", scores=routed, labels=valid["labels"], sample_ids=valid["sample_ids"], used_face=mask)
        development[condition] = {"development": metric, "fallback_bias": bias, "primary_bias": primary_bias,
                                  "training": training, "weights_sha256": sha256_file(candidate / "weights.pt")}
        save_json(output / "development_progress.json", development)
        print(json.dumps({"condition": condition, "development": metric}), flush=True)
        del model, a, b, head_model, x, v
        gc.collect()
        torch.cuda.empty_cache()
    winner = max(development, key=lambda name: (development[name]["development"]["macro_f1"], development[name]["development"]["accuracy"]))
    save_json(output / "selection_locked.json", {"winner": winner, "development": development, "test_not_yet_loaded": True})
    # 到这里才读取历史测试和新来源测试，以下结果不能改动 winner。
    test = load_split(previous / "face_features/test.npz")
    with np.load(previous / "availability_router/historical_predictions.npz", allow_pickle=False) as saved:
        if not np.array_equal(saved["sample_ids"], test["sample_ids"]):
            raise ValueError("原历史样本顺序不符")
        old_scores = saved["scores"].copy()
    baseline_pred = old_scores.argmax(1)
    source_test = external_data(neutral_root / "test.jsonl")
    source_groups = np.array([json.loads(line)["dialogue"] for line in (neutral_root / "test.jsonl").read_text(encoding="utf-8").splitlines()])
    ecisa_source = previous / "router_confirmation/holdout.jsonl"
    ecisa = external_data(ecisa_source)
    reports: dict[str, Any] = {"keep_round5": {"historical": summary(test["labels"], old_scores)}}
    for condition in ["keep_round5", *configs]:
        model = MultimodalBERT(encoder).cuda()
        path = base if condition == "keep_round5" else output / condition / "weights.pt"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        bias = ft["development"][ft["winner"]]["bias"] if condition == "keep_round5" else development[condition]["fallback_bias"]
        if condition == "keep_round5":
            scores = old_scores
        else:
            data = {**test, "tuned_mean": encode_mean(model, test, tokenizer)}
            with (output / condition / "primary.pkl").open("rb") as handle:
                head = pickle.load(handle)
            fallback = infer(model, test, tokenizer, scaler)
            fallback[:, 1] += bias
            scores, _ = route_scores(data, primary_scores(data, head), fallback, threshold)
        source_scores = infer(model, source_test, tokenizer, scaler)
        source_scores[:, 1] += bias
        ecisa_scores = infer(model, ecisa, tokenizer, scaler)
        ecisa_scores[:, 1] += bias
        historical = summary(test["labels"], scores)
        reports[condition] = {"historical": historical, "emotiontalk_binary_neutral": binary_summary(source_test["labels"], source_scores),
                              "ecisa_3class": summary(ecisa["labels"], ecisa_scores),
                              "delta_vs_round5_ci95": grouped_delta_ci(test["labels"], scores.argmax(1), baseline_pred, test["source_groups"], 3)}
        np.savez_compressed(output / (condition + "_predictions.npz"), historical_scores=scores, historical_labels=test["labels"], sample_ids=test["sample_ids"],
                            source_scores=source_scores, source_labels=source_test["labels"], source_ids=source_test["sample_ids"], source_groups=source_groups,
                            ecisa_scores=ecisa_scores, ecisa_labels=ecisa["labels"], ecisa_ids=ecisa["sample_ids"])
        if condition != "keep_round5":
            reports[condition]["delta_vs_matched_control_ci95"] = None
        print(json.dumps({"condition": condition, **reports[condition]}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    with np.load(output / "no_new_auxiliary_predictions.npz", allow_pickle=False) as control:
        control_pred = control["historical_scores"].argmax(1)
    for condition in configs[1:]:
        with np.load(output / (condition + "_predictions.npz"), allow_pickle=False) as saved:
            reports[condition]["delta_vs_matched_control_ci95"] = grouped_delta_ci(test["labels"], saved["historical_scores"].argmax(1), control_pred, test["source_groups"], 3)
    report = {"protocol": protocol, "winner": winner, "development": development, "evaluations": reports,
              "selected_historical": reports[winner]["historical"], "seconds": time.monotonic() - started,
              "limitations": ["historical and development sets repeatedly used in prior rounds", "new source evaluation is text-only binary emotional neutrality",
                              "new source emotion neutrality and target sentiment neutrality have different annotation meanings",
                              "no independent full audio/video source validation", "uncalibrated research model; no real human efficacy or clinical conclusions"],
              "deployed": False}
    save_json(output / "results.json", report)
    print(json.dumps({"winner": winner, "historical": report["selected_historical"], "seconds": report["seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["root", "previous", "bundle", "encoder", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.previous, args.bundle, args.encoder, args.output)
