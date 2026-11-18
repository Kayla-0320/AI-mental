"""只由来源OOF选择最终模型或软投票组合，先锁定后接触历史评测。"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import softmax
from experiments.round10_fusion import OUTPUT, save, select_bias


def select() -> None:
    """固定有限专家组合；没有任何历史标签或历史分数输入。"""
    target = OUTPUT / "final" / "selection_locked.json"
    if target.exists():
        raise ValueError("拒绝覆盖最终锁定")
    component_specs = [("linear", None), ("aux_fusion", None), ("deep_fusion", None),
                       ("neural", None), ("text", None), ("linear", "textonly"), ("linear", "faceonly")]
    components, probabilities = [], []
    ids = labels = None
    for stage, pack in component_specs:
        rows = json.loads((OUTPUT / stage / "search_complete.json").read_text("utf-8"))["results"]
        choices = [i for i, row in enumerate(rows) if pack is None or row["config"].get("pack") == pack]
        best = max(choices, key=lambda i: (rows[i]["metrics"]["macro_f1"], rows[i]["metrics"]["accuracy"]))
        with np.load(OUTPUT / stage / "candidates_oof.npz") as values:
            logits = values["logits"][best]
            if ids is not None and not np.array_equal(ids, values["sample_ids"]):
                raise ValueError("专家OOF未对齐")
            ids, labels = values["sample_ids"].copy(), values["labels"].copy()
        scale = max(.1, float(logits.std(axis=1).mean()))
        probabilities.append(softmax(logits/scale, axis=1))
        components.append({"stage": stage, "candidate_index": best, "candidate": rows[best], "logit_scale": scale})
    save(OUTPUT / "final" / "blend_preregister.json", {"components": components,
         "grid": "nonnegative quarter weights summing to1; global bias from fixed 15-member grid; macro-F1 then accuracy then fewer active experts",
         "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         "final_refit_protocol_sha256": hashlib.sha256((OUTPUT / "final_refit_protocol.json").read_bytes()).hexdigest()})
    results, best = [], None
    for choices in itertools.combinations_with_replacement(range(len(components)), 4):
        weights = np.bincount(choices, minlength=len(components)) / 4
        p = sum(weight * values for weight, values in zip(weights, probabilities))
        selected = select_bias(labels, np.log(np.maximum(p, 1e-12)))
        row = {"weights": weights.tolist(), **selected}
        results.append(row)
        key = (row["metrics"]["macro_f1"], row["metrics"]["accuracy"], -int((weights > 0).sum()))
        if best is None or key > best[0]:
            best = key, row, p
    selection = {"components": components, "winner": best[1], "development_only": True,
                 "training_final": ["train", "selection", "calibration"], "ablation_training": ["train", "selection"],
                 "caveat": "OOF used for selection is a development score; not independent confirmation"}
    save(target, selection)
    save(OUTPUT / "final" / "blend_candidates.json", results)
    np.savez_compressed(OUTPUT / "final" / "selected_oof.npz", probabilities=best[2], labels=labels, sample_ids=ids)
    print(json.dumps({"selected_oof": best[1]["metrics"], "weights": best[1]["weights"], "bias": best[1]["bias"]}), flush=True)


if __name__ == "__main__":
    select()
