"""区分情绪中性与极性中性的多任务对照；辅助头只参与训练。"""
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer

from experiments.round2_moe_train import load_split, grouped_delta_ci
from experiments.round4_accuracy import merge
from experiments.round5_finetune import MultimodalBERT, Rows, collate, external_data, logits_for, choose_bias, seed_all
from experiments.round5_face_search import features
from experiments.round5_frozen_search import summary
from experiments.round5_ensemble import normalized, select_bias
from experiments.round5_availability_router import route_scores
from experiments.round6_neutral_transfer import encode_mean, infer, primary_scores, binary_summary
from perception.knowledge_moe.features import FeatureStandardizer, save_json, sha256_file


def auxiliary_logits(model: MultimodalBERT, head: nn.Module, row: dict[str, torch.Tensor]) -> torch.Tensor:
    """共享编码器，单独输出情绪中性标签，不向三分类头强塞二分类标签。"""
    hidden = model.encoder(input_ids=row["input_ids"], attention_mask=row["attention_mask"]).last_hidden_state
    mask = row["attention_mask"].unsqueeze(-1)
    pooled = torch.cat([hidden[:, 0], (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)], dim=1)
    return head(pooled)


def fit(model: MultimodalBERT, aux_head: nn.Module, target: Rows, auxiliary: Rows, valid: Rows,
        weight: float, output: Path) -> tuple[float, dict[str, Any]]:
    """配对批次联合优化，保留零轮起点，防止继续训练对照被人为做差。"""
    target_loader = DataLoader(target, batch_size=32, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(42))
    source_loader = DataLoader(auxiliary, batch_size=32, shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(42))
    source_iterator = iter(source_loader)
    valid_loader = DataLoader(valid, batch_size=64, collate_fn=collate)
    weights = []
    for rows, classes in [(target, 3), (auxiliary, 2)]:
        counts = torch.bincount(rows.y, minlength=classes).float().cuda()
        value = (len(rows) / (classes * counts)).sqrt()
        weights.append(value / value.mean())
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": [p for name, p in model.named_parameters() if not name.startswith("encoder.")], "lr": 1e-4},
        {"params": aux_head.parameters(), "lr": 1e-4}
    ], weight_decay=.01)
    steps = len(target_loader) * 6
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / max(1, int(steps * .1)), max(0, (steps - step) / max(1, int(steps * .9)))))
    amp = torch.amp.GradScaler("cuda")
    initial = logits_for(model, valid_loader)
    bias, metric = choose_bias(valid.y.numpy(), initial)
    best = (metric["macro_f1"], metric["accuracy"])
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_aux = {key: value.detach().cpu().clone() for key, value in aux_head.state_dict().items()}
    best_bias = bias
    history = [{"epoch": 0, "development": metric, "bias": bias}]
    stale = 0
    source_seen = 0
    seed_all()
    for epoch in range(1, 7):
        model.train()
        aux_head.train()
        losses = []
        for row in target_loader:
            batch = {key: value.cuda() for key, value in row.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                target_loss = nn.functional.cross_entropy(model(**batch).float(), labels, weight=weights[0], label_smoothing=.03)
                loss = target_loss
                if weight:
                    try:
                        source = next(source_iterator)
                    except StopIteration:
                        source_iterator = iter(source_loader)
                        source = next(source_iterator)
                    source = {key: value.cuda() for key, value in source.items()}
                    source_y = source.pop("labels")
                    source_loss = nn.functional.cross_entropy(auxiliary_logits(model, aux_head, source).float(), source_y, weight=weights[1], label_smoothing=.03)
                    loss = loss + weight * source_loss
                    source_seen += len(source_y)
            amp.scale(loss).backward()
            amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_([*model.parameters(), *aux_head.parameters()], 1.)
            previous_scale = amp.get_scale()
            amp.step(optimizer)
            amp.update()
            if amp.get_scale() >= previous_scale:
                scheduler.step()
            losses.append(loss.item())
        values = logits_for(model, valid_loader)
        bias, metric = choose_bias(valid.y.numpy(), values)
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "development": metric, "bias": bias}
        history.append(record)
        print(json.dumps({"aux_weight": weight, **record}), flush=True)
        key = (metric["macro_f1"], metric["accuracy"])
        if key > best:
            best = key
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_aux = {key: value.detach().cpu().clone() for key, value in aux_head.state_dict().items()}
            best_bias = bias
            stale = 0
        else:
            stale += 1
        save_json(output / "training.json", {"history": history, "source_examples_seen": source_seen})
        if stale >= 3:
            break
    model.load_state_dict(best_state)
    aux_head.load_state_dict(best_aux)
    torch.save(best_state, output / "weights.pt")
    torch.save(best_aux, output / "auxiliary_head.pt")
    return best_bias, {"history": history, "source_examples_seen": source_seen, "best": best, "zero_epoch_allowed": True}


def run(root: Path, previous: Path, encoder: Path, output: Path) -> dict[str, Any]:
    """使用固定的三种损失权重，保留前次结果作为探索诊断而非独立确认。"""
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection_locked.json").exists():
        raise ValueError("已锁定的实验不可覆盖")
    started = time.monotonic()
    seed_all()
    tokenizer = AutoTokenizer.from_pretrained(encoder, local_files_only=True)
    scaler = FeatureStandardizer()
    scaler.parameters = json.loads((previous / "finetune/side_standardizer.json").read_text(encoding="utf-8"))
    base = previous / "finetune/ecisa_1e-05_0.5.pt"
    training = merge([load_split(previous / "face_features" / (split + ".npz")) for split in ["train", "selection"]])
    valid = load_split(previous / "face_features/calibration.npz")
    source = external_data(root / "neutral_data/train.jsonl")
    target_rows, valid_rows, source_rows = [Rows(data, tokenizer, scaler) for data in [training, valid, source]]
    old = json.loads((previous / "availability_router/selection_locked.json").read_text(encoding="utf-8"))
    threshold = old["minimum_face_coverage"]
    keep = old["choices"][str(threshold)]["development"]
    protocol = {"seed": 42, "auxiliary_weights": [0., .1, .3], "auxiliary_task": "separate2-class text emotion-neutral head; discarded for target inference",
                "target_task": "unchanged3-class sentiment polarity head", "zero_epoch_candidate": True,
                "selection": "same703 development macro-F1 then accuracy; previous round5 included as keep-current",
                "source_training": "EmotionTalk official train groups, deduplicated14961; no source test selection",
                "diagnostic_context": "sequential neutral transfer and source test metrics have already been seen; external source is now exploratory, not fresh confirmation",
                "primary_head": "adapted mean+BGE+visual+audio+face; fixed C=.1,balanced,face_weight=.5",
                "base_checkpoint_sha256": sha256_file(base), "data_manifest_sha256": sha256_file(root / "neutral_data/manifest.json"),
                "code_sha256": sha256_file(Path(__file__)), "deployed": False}
    save_json(output / "preregister.json", protocol)
    development: dict[str, Any] = {"keep_round5": {"development": keep}}
    for weight in protocol["auxiliary_weights"]:
        condition = f"joint_{weight}"
        candidate = output / condition
        candidate.mkdir(exist_ok=True)
        seed_all()
        model = MultimodalBERT(encoder).cuda()
        model.load_state_dict(torch.load(base, map_location="cpu", weights_only=True))
        head = nn.Sequential(nn.LayerNorm(model.encoder.config.hidden_size * 2), nn.Linear(model.encoder.config.hidden_size * 2, 2)).cuda()
        bias, record = fit(model, head, target_rows, source_rows, valid_rows, weight, candidate)
        a = {**training, "tuned_mean": encode_mean(model, training, tokenizer)}
        b = {**valid, "tuned_mean": encode_mean(model, valid, tokenizer)}
        names = ["tuned_mean", "bge_cls", "visual", "audio", "face_summary"]
        x, scalers = features(a, names, face_weight=.5)
        v, _ = features(b, names, scalers, .5)
        classifier = LogisticRegression(C=.1, class_weight="balanced", max_iter=700, random_state=42).fit(x, a["labels"])
        primary = normalized(classifier.decision_function(v))
        pbias, _ = select_bias(valid["labels"], primary)
        primary[:, 1] += pbias
        fallback = infer(model, valid, tokenizer, scaler)
        fallback[:, 1] += bias
        scores, mask = route_scores(valid, primary, fallback, threshold)
        metric = summary(valid["labels"], scores)
        with (candidate / "primary.pkl").open("wb") as handle:
            pickle.dump({"names": names, "model": classifier, "scalers": scalers, "bias": pbias}, handle)
        np.savez_compressed(candidate / "development_predictions.npz", scores=scores, labels=valid["labels"], sample_ids=valid["sample_ids"], used_face=mask)
        development[condition] = {"development": metric, "fallback_bias": bias, "training": record}
        save_json(output / "development_progress.json", development)
        print(json.dumps({"condition": condition, "development": metric}), flush=True)
        del model, head, a, b, classifier, x, v
        gc.collect()
        torch.cuda.empty_cache()
    winner = max(development, key=lambda name: (development[name]["development"]["macro_f1"], development[name]["development"]["accuracy"]))
    save_json(output / "selection_locked.json", {"winner": winner, "development": development, "test_not_yet_loaded_this_stage": True})
    test = load_split(previous / "face_features/test.npz")
    with np.load(previous / "availability_router/historical_predictions.npz", allow_pickle=False) as saved:
        reference = saved["scores"].copy()
    source_test = external_data(root / "neutral_data/test.jsonl")
    ecisa = external_data(previous / "router_confirmation/holdout.jsonl")
    evaluations = {}
    for weight in protocol["auxiliary_weights"]:
        condition = f"joint_{weight}"
        candidate = output / condition
        model = MultimodalBERT(encoder).cuda()
        model.load_state_dict(torch.load(candidate / "weights.pt", map_location="cpu", weights_only=True))
        data = {**test, "tuned_mean": encode_mean(model, test, tokenizer)}
        with (candidate / "primary.pkl").open("rb") as handle:
            primary = pickle.load(handle)
        bias = development[condition]["fallback_bias"]
        fallback = infer(model, test, tokenizer, scaler)
        fallback[:, 1] += bias
        scores, mask = route_scores(data, primary_scores(data, primary), fallback, threshold)
        source_values = infer(model, source_test, tokenizer, scaler)
        source_values[:, 1] += bias
        ecisa_values = infer(model, ecisa, tokenizer, scaler)
        ecisa_values[:, 1] += bias
        evaluations[condition] = {"historical": summary(test["labels"], scores),
                                  "source_sentiment_neutral_proxy": binary_summary(source_test["labels"], source_values),
                                  "ecisa_3class": summary(ecisa["labels"], ecisa_values),
                                  "delta_vs_round5_ci95": grouped_delta_ci(test["labels"], scores.argmax(1), reference.argmax(1), test["source_groups"], 3)}
        np.savez_compressed(output / (condition + "_predictions.npz"), scores=scores, labels=test["labels"], sample_ids=test["sample_ids"], used_face=mask,
                            source_scores=source_values, source_labels=source_test["labels"], ecisa_scores=ecisa_values, ecisa_labels=ecisa["labels"])
        print(json.dumps({"condition": condition, **evaluations[condition]}), flush=True)
        del model, data
        gc.collect()
        torch.cuda.empty_cache()
    with np.load(output / "joint_0.0_predictions.npz", allow_pickle=False) as control:
        control_pred = control["scores"].argmax(1)
    for condition in evaluations:
        with np.load(output / (condition + "_predictions.npz"), allow_pickle=False) as values:
            evaluations[condition]["delta_vs_matched_control_ci95"] = grouped_delta_ci(test["labels"], values["scores"].argmax(1), control_pred, test["source_groups"], 3)
    report = {"protocol": protocol, "winner": winner, "development": development, "evaluations": evaluations,
              "selected_historical": summary(test["labels"], reference) if winner == "keep_round5" else evaluations[winner]["historical"],
              "seconds": time.monotonic() - started, "deployed": False}
    save_json(output / "results.json", report)
    print(json.dumps({"winner": winner, "historical": report["selected_historical"], "seconds": report["seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["root", "previous", "encoder", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.previous, args.encoder, args.output)
