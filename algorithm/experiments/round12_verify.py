"""第十二轮独立重放、同压力旧模型对照及多人盲标准备。"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
from scipy.special import softmax
from experiments.round10_fusion import folds, take
from experiments.round11_common import digest
from experiments.round11_evidence import expert_evidence
from experiments.round11_features import load_data
from experiments.round11_missing_route import route
from experiments.round12_explore import (OUTPUT, PREVIOUS, CONDITIONS, altered, configurations,
    fold_inputs, paired_interval, prepare, save, train)
from experiments.round12_runtime import OfflinePredictor
from experiments.round7_decision_oof import metric


def old_inputs(data: dict[str, np.ndarray], condition: str, donors: np.ndarray,
               mismatch_alignment: np.ndarray) -> dict[str, np.ndarray]:
    """旧模型消费相同的原始缺失/供体条件，不使用新候选的合成联合证据。"""
    result = {k: v.copy() for k, v in data.items()}
    absent = {"missing_T": (0,), "missing_A": (1,), "missing_V": (2,),
              "text_only": (1, 2), "all_missing": (0, 1, 2)}.get(condition, ())
    for j in absent:
        result["modality_mask"][:, j] = False
        result["quality"][:, j] = 0.
        if j == 2:
            for field in ("face_summary", "deep_face", "visual", "coverage"):
                result[field][:] = 0.
    if condition == "audio_mismatch":
        for field in ("audio", "speech", "speech_emotion", "speech_available", "sensevoice",
                      "sensevoice_query", "sensevoice_emotion", "sensevoice_available"):
            result[field] = data[field][donors].copy()
        result["modality_mask"][:, 1] = data["modality_mask"][donors, 1]
        result["quality"][:, 1] = data["quality"][donors, 1]
        result["asr_alignment"] = mismatch_alignment
    return result


def baselines(raw: dict[str, np.ndarray], parts: list[dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    """只重放开发来源旧7B/light已锁定模型，所有候选的压力供体完全一致。"""
    output = OUTPUT / "previous_baselines"
    output.mkdir(parents=True, exist_ok=True)
    policies = json.loads((PREVIOUS / "missing_route/selection_locked.json").read_text("utf-8"))
    predictions = {}
    for mode in ("7b", "light"):
        cache = output / f"{mode}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as z:
                predictions[mode] = {c: z[c] for c in CONDITIONS}
            continue
        folder = PREVIOUS / ("evidence_deterministic" if mode == "7b" else "deployment_light/evidence_deterministic") / "family10"
        rows = json.loads((folder / "complete.json").read_text("utf-8"))["results"]
        choice = max(range(len(rows)), key=lambda j: rows[j]["selection_score"])
        bias = rows[choice]["bias"]
        if mode == "7b":
            bias = json.loads((PREVIOUS / "blend/selection_locked.json").read_text("utf-8"))["winner"]["bias"]
        p = {c: np.full((len(raw["labels"]), 3), np.nan) for c in CONDITIONS}
        for f, (tr, va) in enumerate(folds(raw)):
            with (folder / f"fold{f}_models.pkl").open("rb") as handle:
                bundle = pickle.load(handle)
            if set(raw["source_groups"][va]) & set(bundle["training_groups"]):
                raise ValueError("旧模型基线来源泄漏")
            prior = np.bincount(raw["labels"][tr], minlength=3) / len(tr)
            for condition in CONDITIONS:
                part = old_inputs(take(raw, va), condition, parts[f]["donors"], parts[f]["mismatch_alignment"])
                x = expert_evidence(bundle["experts"], part)
                value = softmax(np.log(np.maximum(bundle["meta_models"][choice].predict_proba(x), 1e-12)) + bias, axis=1)
                missing = ~(part["modality_mask"][:, 0] & (part["quality"][:, 0] > 0))
                if policies.get("promoted", False) and missing.any():
                    value = route(value, x, missing, policies["winner"]["audio_weight"], bias, prior)
                p[condition][va] = value
            print(json.dumps(dict(baseline=mode, fold=f)), flush=True)
        np.savez_compressed(cache, sample_ids=raw["sample_ids"], **p)
        predictions[mode] = p
    return predictions


def multimodal_kit(data: dict[str, np.ndarray]) -> None:
    """提供M3原媒体路径，盲标文件不泄露官方极性或七类情绪。"""
    folder = OUTPUT / "annotation_audit"
    reference = json.loads((folder / "reference_private.json").read_text("utf-8"))
    lookup = {str(k): i for i, k in enumerate(data["sample_ids"])}
    media_root = PREVIOUS.parents[1] / "data/public/ch_sims_v2/supervised"
    for person in ("A", "B"):
        with (folder / f"multimodal_{person}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["blind_id", "media_path", "text", "M3_negative_neutral_positive", "uncertain", "comment"])
            for row in reference["rows"]:
                path = media_root / f"{row['sample_id']}.mp4"
                if not path.is_file():
                    raise ValueError(f"盲标媒体缺失: {path}")
                writer.writerow([row["blind_id"], str(path), data["raw_text"][lookup[row["sample_id"]]], "", "", ""])
    (folder / "标注说明.md").write_text(
        "# 第十二轮多人独立标注准备\n\nmultimodal_A.csv与multimodal_B.csv供两位标注者独立审阅原视频、原音轨和转录，填写整体M3：negative、neutral、positive。"
        "不查看reference_private.json或对方结果；不确定时保留uncertain及原因，不强行改变原标签。annotator_A/B.csv为较早生成的文本侧模板，只能用作T3复核，不能与M3混用。\n\n"
        "120条按旧M3每类40条抽样，属于分层审计样本，不代表自然类别比例。先计算双人一致率/Cohen kappa、各类分歧，再结合原标注审阅；目前模板为空，没有真人审核成绩。"
        "本批CH审计不解决EmotionTalk连续极性与七类情绪的语义差异，后者需另外组织发布标注者/独立人员复核。\n", encoding="utf-8")


def main() -> None:
    """重放与从头重训都通过后写验证记录，不对历史或外部测试跑分。"""
    results = json.loads((OUTPUT / "results.json").read_text("utf-8"))
    if (OUTPUT / "verification.json").exists():
        raise ValueError("已有锁定验证结果，拒绝重复覆盖")
    started = time.monotonic()
    data = prepare()
    candidate = results["development_candidate"]
    config = next(c for c in configurations() if c["id"] == candidate)
    parts = []
    replay_errors = []
    retrain_error = None
    old_paths = [PREVIOUS / "第十一轮完整优化与独立验证报告.md", PREVIOUS / "artifact_manifest_final.json"]
    old_paths += [PREVIOUS / f"evidence_deterministic/family10/fold{f}.npz" for f in range(4)]
    before = {str(p): digest(p) for p in old_paths}
    for f, (tr, va) in enumerate(folds(data)):
        training, validation, _ = fold_inputs(data, f, tr, va)
        parts.append(validation)
        folder = OUTPUT / "candidates" / candidate / f"fold{f}"
        predictor = OfflinePredictor(folder / "head.pt")
        with np.load(folder / "predictions.npz", allow_pickle=False) as z:
            for condition in CONDITIONS:
                actual, _ = predictor.predict_prepared(*altered(validation, condition))
                replay_errors.append(float(np.max(np.abs(actual - z[condition]))))
        # 监督记忆需保存到模型目录供离线推理使用；线上准入仍待新来源验证。
        np.savez_compressed(folder / "memory_bank.npz", vectors=data["bge_cls"][tr], labels=data["labels"][tr],
                            source_groups=data["source_groups"][tr], normalized_text=data["normalized_text"][tr],
                            sample_ids=data["sample_ids"][tr])
        if f == 0:
            p = train(config, training, validation, 60, OUTPUT / "independent_retrain/fold0")
            with np.load(folder / "predictions.npz", allow_pickle=False) as z:
                retrain_error = max(float(np.max(np.abs(p[c] - z[c]))) for c in CONDITIONS)
    raw = load_data("pts")
    previous = baselines(raw, parts)
    with np.load(OUTPUT / "candidates" / candidate / "oof.npz", allow_pickle=False) as z:
        current = {c: z[c] for c in CONDITIONS}
    summaries = {mode: {c: metric(data["labels"], p[c].argmax(1)) for c in CONDITIONS} for mode, p in previous.items()}
    comparison = dict(candidate=candidate, previous=summaries,
        paired_clean_vs_light=paired_interval(data["labels"], previous["light"]["clean"], current["clean"], data["source_groups"]),
        paired_clean_vs_7b=paired_interval(data["labels"], previous["7b"]["clean"], current["clean"], data["source_groups"]),
        status="same development2876 and same corruption donors; previous choices locked in round11; not a new selection grid")
    save(OUTPUT / "previous_baselines/comparison.json", comparison)
    multimodal_kit(data)
    after = {str(p): digest(p) for p in old_paths}
    if before != after or max(replay_errors) > 1e-6 or retrain_error is None or retrain_error > 1e-6:
        raise ValueError("旧工件保护、模型重放或独立重训未通过")
    save(OUTPUT / "verification.json", dict(replay_max_abs_error=max(replay_errors), independent_retrain_max_abs_error=retrain_error,
        source_folds=4, condition_replays=len(replay_errors), previous_inputs_preserved=True,
        previous_inputs_sha256=after, historical_or_external_test_evaluated=False, seconds=time.monotonic()-started))
    lines = ["\n## 与旧轮同压力条件对照及独立重放\n", "全部对照使用同2876条开发样本及本轮同一错配供体。7B与light均保留第十一轮已锁定开发配方和缺字退化路径；其骨干/联合M专家与本轮新融合不同，差距不能归于单一增广或MoE组件。\n",
        "|方案|干净|缺A|缺V|缺T|仅T|错配|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, metrics in summaries.items():
        lines.append("|第十一轮" + name + "|" + "|".join(f"{metrics[c]['macro_f1']:.6f}" for c in CONDITIONS[:6]) + "|")
    lines += [f"\n新开发候选为{candidate}。相对旧light干净F1差的来源配对95%条件区间：{comparison['paired_clean_vs_light']}；相对旧7B：{comparison['paired_clean_vs_7b']}。",
        f"\n4来源折×7观测条件共28次重放，最大概率误差{max(replay_errors):.9g}；来源折0从头重训最大误差{retrain_error:.9g}。已核对的旧报告、清单和输入证据摘要未变化。",
        "\n已补齐两份带原视频路径的M3独立盲标模板和标注说明；尚无真人标注结果。\n"]
    with (OUTPUT / "第十二轮鲁棒训练与知识MoE探索报告.md").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    main()
