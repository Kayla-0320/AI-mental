"""独立辅助预训练后，按来源折重训文字编码器与真实人脸分支。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import torch
from torch import nn, Tensor
from transformers import AutoModel, AutoTokenizer

from experiments.round10_fusion import ROOT, OUTPUT, load_data, folds, take, save, select_bias
from experiments.round10_neural import seed_all
from perception.knowledge_moe.features import FeatureStandardizer

ENCODER = Path(r"D:\ai\models\encoders\chinese-roberta-wwm-ext")


class TextFace(nn.Module):
    """可训练文字编码器、人脸与固定语义补充；缺脸时投影后置零。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ENCODER, local_files_only=True)
        self.text = nn.Sequential(nn.LayerNorm(1536), nn.Dropout(.2), nn.Linear(1536, 128), nn.GELU())
        self.face = nn.Sequential(nn.Linear(29, 32), nn.LayerNorm(32), nn.GELU())
        self.bge = nn.Sequential(nn.Linear(768, 32), nn.LayerNorm(32), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(193, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, ids: Tensor, attention: Tensor, face: Tensor, bge: Tensor, present: Tensor) -> Tensor:
        """只处理当前真实观测；未来片段和真实标签不作为输入。"""
        hidden = self.encoder(input_ids=ids, attention_mask=attention).last_hidden_state
        weights = attention.unsqueeze(-1)
        pooled = torch.cat([hidden[:, 0], (hidden * weights).sum(1) / weights.sum(1)], 1)
        return self.head(torch.cat([self.text(pooled), self.face(face) * present,
                                    self.bge(bge), present], 1))


def prepare(data: dict[str, np.ndarray], tokenizer: Any,
            scaler: FeatureStandardizer | None = None) -> tuple[dict[str, Tensor], FeatureStandardizer]:
    """所有数值统计只拟合训练折有效观测。"""
    present = data["face_summary"][:, 24] > 0
    raw = {"face": data["face_summary"], "bge": data["bge_cls"]}
    observed = np.column_stack([present, np.ones(len(present), bool)])
    if scaler is None:
        scaler = FeatureStandardizer().fit(raw, observed)
    values = scaler.transform(raw, observed)
    encoded = tokenizer(data["raw_text"].astype(str).tolist(), max_length=96, truncation=True,
                        padding="max_length", return_tensors="pt")
    tensors = {"ids": encoded["input_ids"], "attention": encoded["attention_mask"],
               "face": torch.tensor(values["face"]), "bge": torch.tensor(values["bge"]),
               "present": torch.tensor(present[:, None], dtype=torch.float32),
               "labels": torch.tensor(data["labels"], dtype=torch.long)}
    return tensors, scaler


def predict(model: TextFace, data: dict[str, Tensor]) -> np.ndarray:
    """评测只读特征，标签不传入模型。"""
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(data["ids"]), 64):
            row = {name: values[start:start+64].cuda() for name, values in data.items() if name != "labels"}
            width = int(row["attention"].sum(1).max())
            row["ids"], row["attention"] = row["ids"][:, :width], row["attention"][:, :width]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result.append(model(**row).float().cpu().numpy())
    return np.concatenate(result)


def optimize(model: TextFace, data: dict[str, Tensor], config: dict[str, Any], epochs: int,
             validation: dict[str, Tensor] | None = None, log_name: str = "train"
             ) -> dict[int, np.ndarray]:
    """固定轮次训练并记录OOF预测，不在留出折中选checkpoint。"""
    labels = data["labels"]
    counts = torch.bincount(labels, minlength=3).float().cuda()
    weights = (len(labels) / (3 * counts)).pow(config["power"])
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config["lr"]},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": config["lr"] * 10},
    ], weight_decay=.01)
    steps = epochs * ((len(labels) + 31) // 32)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min((step + 1) / max(1, int(.1 * steps)), max(0., (steps-step) / max(1, .9*steps))))
    predictions = {}
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(labels))
        total = 0.
        for start_ix in range(0, len(labels), 32):
            ix = order[start_ix:start_ix+32]
            row = {name: value[ix].cuda() for name, value in data.items()}
            y = row.pop("labels")
            width = int(row["attention"].sum(1).max())
            row["ids"], row["attention"] = row["ids"][:, :width], row["attention"][:, :width]
            if config.get("drop_face", True):
                row["present"] *= (torch.rand_like(row["present"]) > .1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**row)
                loss = nn.functional.cross_entropy(logits.float(), y, weight=weights, label_smoothing=.03)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(y)
        if validation is not None:
            predictions[epoch] = predict(model, validation)
        print(json.dumps({"stage": log_name, "epoch": epoch, "loss": total/len(labels),
                          "seconds": round(time.monotonic()-start, 1)}), flush=True)
    return predictions


def auxiliary(output: Path, tokenizer: Any) -> Path:
    """仅ECISA官方训练预训练，绝不从含全部目标标签的旧checkpoint启动OOF。"""
    checkpoint = output / "auxiliary_only.pt"
    if checkpoint.exists():
        return checkpoint
    source = ROOT / "round5" / "external" / "train.jsonl"
    rows = [json.loads(line) for line in source.read_text("utf-8").splitlines() if line]
    data = {"raw_text": np.array([row["text"] for row in rows]), "labels": np.array([row["label"] for row in rows]),
            "face_summary": np.zeros((len(rows), 29), np.float32), "bge_cls": np.zeros((len(rows), 768), np.float32)}
    encoded = tokenizer(data["raw_text"].tolist(), max_length=96, truncation=True, padding="max_length", return_tensors="pt")
    tensors = {"ids": encoded["input_ids"], "attention": encoded["attention_mask"],
        "face": torch.zeros(len(rows), 29), "bge": torch.zeros(len(rows), 768),
        "present": torch.zeros(len(rows), 1), "labels": torch.tensor(data["labels"])}
    seed_all()
    model = TextFace().cuda()
    optimize(model, tensors, {"lr": 2e-5, "power": .5, "drop_face": False}, 1, log_name="auxiliary_only")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint)
    save(output / "auxiliary_manifest.json", {"source": str(source), "n": len(rows),
         "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "epochs": 1, "target_labels_used": False})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return checkpoint


def search(output: Path) -> None:
    """每个来源折从同一目标无关辅助权重开始，保存完整OOF。"""
    output.mkdir(parents=True, exist_ok=True)
    if (output / "search_complete.json").exists():
        raise ValueError("拒绝覆盖已完成搜索")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
    configs = [{"id": "textface_sqrt", "lr": 1e-5, "power": .5},
               {"id": "textface_balanced", "lr": 1e-5, "power": 1.}]
    epochs = 5
    save(output / "preregister.json", {"configs": configs, "epochs": epochs,
         "training": "4-fold source-group OOF; all supervised encoding refitted per fold",
         "forbidden_for_selection": ["calibration703", "test821"], "encoder": str(ENCODER),
         "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    base = auxiliary(output, tokenizer)
    data = load_data()
    result, all_predictions = [], []
    for config in configs:
        oof = {epoch: np.full((len(data["labels"]), 3), np.nan) for epoch in range(1, epochs+1)}
        for fold, (tr, va) in enumerate(folds(data)):
            seed_all()
            training, scaler = prepare(take(data, tr), tokenizer)
            validation, _ = prepare(take(data, va), tokenizer, scaler)
            model = TextFace().cuda()
            model.load_state_dict(torch.load(base, map_location="cpu", weights_only=True))
            predictions = optimize(model, training, config, epochs, validation, f"{config['id']}_fold{fold}")
            for epoch, prediction in predictions.items():
                oof[epoch][va] = prediction
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config, "scaler": scaler.parameters, "epoch": epochs},
                        output / f"{config['id']}_fold{fold}_last.pt")
            del model
            gc.collect()
            torch.cuda.empty_cache()
        for epoch, prediction in oof.items():
            chosen = select_bias(data["labels"], prediction)
            result.append({"config": {**config, "epoch": epoch}, **chosen})
            all_predictions.append(prediction)
        save(output / "progress.json", result)
    np.savez_compressed(output / "candidates_oof.npz", logits=np.stack(all_predictions), labels=data["labels"],
                        sample_ids=data["sample_ids"], source_groups=data["source_groups"])
    save(output / "search_complete.json", {"results": result})


def main() -> None:
    """训练入口，全部输出与旧实验分离。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT / "text")
    args = parser.parse_args()
    search(args.output)


if __name__ == "__main__":
    main()
