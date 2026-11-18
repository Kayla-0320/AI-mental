"""固定来源OOF上的共享专家对照、概率投票和Top-k路由。"""
from __future__ import annotations

import argparse
import copy
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

from experiments.round10_fusion import OUTPUT, load_data, take, folds, save, select_bias, observed
from experiments.round7_decision_oof import metric
from perception.knowledge_moe.features import FeatureStandardizer


def seed_all(seed: int = 42) -> None:
    """固定初始化、批次及增广的所有随机源。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def inputs(data: dict[str, np.ndarray], config: dict[str, Any],
           scaler: FeatureStandardizer | None = None) -> tuple[dict[str, Tensor], Tensor, FeatureStandardizer]:
    """训练折内拟合归一化，缺失人脸不参与统计或专家执行。"""
    names = ["semantic", "face"] + (["visual", "audio"] if config["full"] else [])
    raw = {"semantic": np.column_stack([data["roberta_mean"], data["bge_cls"]]),
           "face": data["face_summary"][:, :24 if config.get("expression_only") else 29]}
    raw.update({name: data[name] for name in names if name in ("visual", "audio")})
    mask = np.column_stack([np.ones(len(data["labels"]), bool), observed(data, "face_summary")]
                          + [observed(data, name) for name in names[2:]])
    if scaler is None:
        scaler = FeatureStandardizer().fit(raw, mask)
    values = scaler.transform(raw, mask)
    return {name: torch.tensor(value) for name, value in values.items()}, torch.tensor(mask), scaler


class FusionHead(nn.Module):
    """同一专家池下比较均匀、全局、动态及真实Top-2概率投票。"""

    def __init__(self, dims: dict[str, int], config: dict[str, Any]) -> None:
        super().__init__()
        self.names = list(dims)
        self.mode = config["mode"]
        hidden = config.get("hidden", 64)
        self.projections = nn.ModuleDict({name: nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(.25)) for name, dim in dims.items()})
        self.requirements = [(i,) for i in range(len(dims))] + [tuple(range(len(dims)))]
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(hidden * len(req), hidden), nn.GELU(),
            nn.Dropout(.25), nn.Linear(hidden, 3)) for req in self.requirements])
        self.router = nn.Sequential(nn.Linear((hidden + 1) * len(dims), hidden), nn.GELU(),
                                    nn.Linear(hidden, len(self.requirements)))
        self.global_weights = nn.Parameter(torch.zeros(len(self.requirements)))
        self.concat = nn.Sequential(nn.Linear(hidden * len(dims), hidden), nn.GELU(),
                                    nn.Dropout(.25), nn.Linear(hidden, 3))

    def forward(self, x: dict[str, Tensor], mask: Tensor) -> tuple[Tensor, Tensor]:
        """无可用输入必须拒绝；Top-k仅执行被选中的样本与专家组合。"""
        if not mask.any(1).all():
            raise ValueError("全缺失输入")
        h = [self.projections[name](torch.where(mask[:, i:i+1], x[name], torch.zeros_like(x[name])))
             * mask[:, i:i+1] for i, name in enumerate(self.names)]
        if self.mode == "concat":
            return self.concat(torch.cat(h, 1)), torch.ones(len(mask), 1, device=mask.device)
        eligible = torch.stack([mask[:, req].all(1) for req in self.requirements], 1)
        if self.mode == "uniform":
            scores = torch.zeros_like(eligible, dtype=h[0].dtype)
        elif self.mode == "global":
            scores = self.global_weights.expand(len(mask), -1)
        else:
            scores = self.router(torch.cat([*h, mask.to(h[0].dtype)], 1))
        scores = scores.masked_fill(~eligible, -torch.inf)
        if self.mode == "top2":
            chosen = torch.zeros_like(eligible).scatter_(1, scores.topk(min(2, scores.shape[1]), 1).indices, True)
            scores = scores.masked_fill(~chosen, -torch.inf)
        weights = scores.softmax(1)
        result = torch.zeros(len(mask), 3, device=mask.device)
        for i, (expert, req) in enumerate(zip(self.experts, self.requirements)):
            rows = torch.where(weights[:, i] > 0)[0]
            if len(rows):
                prob = expert(torch.cat([h[j][rows] for j in req], 1)).softmax(1)
                result = result.index_add(0, rows, weights[rows, i:i+1] * prob)
        return result.clamp_min(1e-9).log(), weights


def options() -> list[dict[str, Any]]:
    """同池路由消融与有限损失/特征候选，在读取测试前固定。"""
    result = []
    for mode in ("concat", "uniform", "global", "dynamic", "top2"):
        for power in (.5, 1.):
            result.append({"mode": mode, "power": power, "full": False, "cf": 0., "expression_only": False})
    result.extend([
        {"mode": "dynamic", "power": .5, "full": False, "cf": .1, "expression_only": False},
        {"mode": "dynamic", "power": 1., "full": True, "cf": 0., "expression_only": False},
        {"mode": "concat", "power": 1., "full": False, "cf": 0., "expression_only": True},
    ])
    return [{**item, "id": f"neural_{i}_{item['mode']}", "lr": .001, "weight_decay": .01, "hidden": 64}
            for i, item in enumerate(result)]


def train(config: dict[str, Any], data: dict[str, np.ndarray], validation: dict[str, np.ndarray],
          epochs: tuple[int, ...], seed: int = 42, directory: Path | None = None
          ) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Tensor]], FeatureStandardizer]:
    """不在折内挑轮次；逐预定轮次输出留出预测，供统一OOF选择。"""
    seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, mask, scaler = inputs(data, config)
    vx, vm, _ = inputs(validation, config, scaler)
    x = {k: v.to(device) for k, v in x.items()}
    mask = mask.to(device)
    vx = {k: v.to(device) for k, v in vx.items()}
    vm = vm.to(device)
    y = torch.tensor(data["labels"], dtype=torch.long, device=device)
    model = FusionHead({name: values.shape[1] for name, values in x.items()}, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    counts = torch.bincount(y, minlength=3).float()
    weights = (len(y) / (3 * counts)).pow(config["power"])
    predictions, states = {}, {}
    for epoch in range(1, max(epochs) + 1):
        model.train()
        order = torch.randperm(len(y), device=device)
        for start in range(0, len(y), 128):
            ix = order[start:start+128]
            subset = {k: v[ix] for k, v in x.items()}
            visible = mask[ix].clone()
            # 仅丢弃非语义分支，保留目标定义；两对照臂共享完全相同配方。
            visible[:, 1:] &= torch.rand_like(visible[:, 1:].float()) >= .1
            logits, gate = model(subset, visible)
            loss = nn.functional.cross_entropy(logits, y[ix], weight=weights, label_smoothing=.03)
            if config["cf"]:
                with torch.no_grad():
                    teacher, _ = model(subset, visible)
                    target = teacher.softmax(1)
                perturbed = {k: v.clone() for k, v in subset.items()}
                perturbed["face"] += torch.randn_like(perturbed["face"]) * .5
                moved, _ = model(perturbed, visible)
                loss += config["cf"] * nn.functional.kl_div(moved.log_softmax(1), target, reduction="batchmean")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        if epoch in epochs:
            model.eval()
            with torch.inference_mode():
                predictions[epoch] = model(vx, vm)[0].cpu().numpy()
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": state, "scaler": scaler.parameters, "config": config,
                            "epoch": epoch, "seed": seed}, directory / f"epoch_{epoch}.pt")
            states[epoch] = state
    return predictions, states, scaler


def search(output: Path) -> None:
    """测试封存时完成同专家池架构对照。"""
    if (output / "search_complete.json").exists():
        raise ValueError("拒绝覆盖已完成搜索")
    data = load_data()
    grid = options()
    epochs = (5, 10, 20, 30)
    save(output / "preregister.json", {"configs": grid, "epochs": epochs,
         "protocol": "source-group 4-fold OOF; no 703/821 access; probabilities are not claimed calibrated"})
    all_predictions, results = [], []
    start = time.monotonic()
    for config in grid:
        oof = {epoch: np.full((len(data["labels"]), 3), np.nan) for epoch in epochs}
        for fold, (tr, va) in enumerate(folds(data)):
            predictions, _, _ = train(config, take(data, tr), take(data, va), epochs,
                                       directory=output / "models" / config["id"] / f"fold_{fold}")
            for epoch in epochs:
                oof[epoch][va] = predictions[epoch]
            print(json.dumps({"candidate": config["id"], "fold": fold,
                              "seconds": round(time.monotonic()-start, 1)}), flush=True)
        for epoch in epochs:
            chosen = select_bias(data["labels"], oof[epoch])
            results.append({"config": {**config, "epoch": epoch}, "argmax": metric(data["labels"], oof[epoch].argmax(1)), **chosen})
            all_predictions.append(oof[epoch])
        save(output / "progress.json", results)
        print(json.dumps({"best_oof": max(r["metrics"]["macro_f1"] for r in results), "candidate": config["id"]}), flush=True)
    np.savez_compressed(output / "candidates_oof.npz", logits=np.stack(all_predictions), labels=data["labels"],
                        sample_ids=data["sample_ids"], source_groups=data["source_groups"])
    save(output / "search_complete.json", {"results": results, "seconds": time.monotonic()-start})


def main() -> None:
    """运行来源分组开发实验。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT / "neural")
    args = parser.parse_args()
    search(args.output)


if __name__ == "__main__":
    main()
