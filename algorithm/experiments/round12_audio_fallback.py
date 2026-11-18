"""针对第十一轮已知缺音频短板的明确退化路径，完整音频观测逐位保留。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from experiments.round11_common import digest
from experiments.round12_explore import OUTPUT, CONDITIONS, paired_interval, prepare, save
from experiments.round7_decision_oof import metric


def audio_absent_route(original: np.ndarray, fallback: np.ndarray, modality_mask: np.ndarray,
                       quality: np.ndarray) -> np.ndarray:
    """文字可见且音频真实缺失时使用轻量头，其余行逐位保留。"""
    if original.shape != fallback.shape or original.shape != (len(modality_mask), 3):
        raise ValueError("概率与观测行数不一致")
    if modality_mask.shape != quality.shape or quality.shape != original.shape:
        raise ValueError("需要T/A/V可用性和质量")
    if not all(np.isfinite(p).all() and np.allclose(p.sum(1), 1., atol=1e-6) for p in (original, fallback)):
        raise ValueError("原始/退化概率无效")
    text = modality_mask[:, 0] & (quality[:, 0] > 0)
    audio = modality_mask[:, 1] & (quality[:, 1] > 0)
    use = text & ~audio
    result = original.copy()
    result[use] = fallback[use]
    return result


def register() -> None:
    """只针对前轮指定缺A与仅文字问题，不通过当前条件分数调整阈值。"""
    plan = dict(trigger="text present and audio absent from observed mask/quality; no numeric threshold search",
        model="fixed mlp_clean from the primary development selection; no new training or hyperparameter search",
        scope="supplemental exploratory fallback; priority follows round11 report; no production promotion",
        clean_audio_present="original decisions bitwise preserved", conditions=list(CONDITIONS), seed=42,
        code_sha256=digest(Path(__file__)))
    path = OUTPUT / "audio_absent_fallback/preregister.json"
    if not path.exists():
        save(path, plan)
    elif json.loads(path.read_text("utf-8")) != plan:
        raise ValueError("退化策略锁定条件变化")


def evaluate() -> None:
    """在统一开发压力输入上报告局部路由，不增加测试集访问。"""
    register()
    folder = OUTPUT / "audio_absent_fallback"
    if (folder / "results.json").exists():
        return
    data = prepare()
    from experiments.round10_fusion import folds
    from experiments.round12_explore import altered, fold_inputs
    with np.load(OUTPUT / "previous_baselines/7b.npz", allow_pickle=False) as z:
        old = {c: z[c] for c in CONDITIONS}
    with np.load(OUTPUT / "candidates/mlp_clean/oof.npz", allow_pickle=False) as z:
        new = {c: z[c] for c in CONDITIONS}
    result = {c: np.full_like(old[c], np.nan) for c in CONDITIONS}
    for f, (tr, va) in enumerate(folds(data)):
        _, part, _ = fold_inputs(data, f, tr, va)
        for condition in CONDITIONS:
            x, mask, _, _, _ = altered(part, condition)
            result[condition][va] = audio_absent_route(old[condition][va], new[condition][va], mask, x[:, :, -1])
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        p = result[condition]
        rows.append(dict(condition=condition, old=metric(data["labels"], old[condition].argmax(1)),
            fallback=metric(data["labels"], p.argmax(1)), changed_rows=int(np.any(p != old[condition], axis=1).sum()),
            paired_source_interval=paired_interval(data["labels"], old[condition], p, data["source_groups"])))
    np.savez_compressed(folder / "oof.npz", sample_ids=data["sample_ids"], labels=data["labels"],
                        source_groups=data["source_groups"], **result)
    save(folder / "results.json", dict(rows=rows, development_only=True, primary_search_winner_unchanged=True,
        independent_confirmation_pending=True, note="known absent audio only; does not detect unknown mismatch"))
    lines = ["\n## 缺音频专用退化补充实验\n", "主要11候选的综合规则仍选mlp_clean；增广/知识/MoE/蒸馏没有胜出。为第十一轮明确的缺A和仅文字短板，补充固定路由：仅在文字可见且音频真实不可用时改用mlp_clean；其他观测保留第十一轮7B决策。本项为主搜索之后的开发探索，未改变原主选型，也未读旧测试。\n",
        "|条件|旧7B|缺音频路由|变化样本数|来源配对差95%区间|", "|---|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"|{row['condition']}|{row['old']['macro_f1']:.6f}|{row['fallback']['macro_f1']:.6f}|{row['changed_rows']}|{row['paired_source_interval']}|")
    lines += ["\n该路由依据真实可用性，不把模型不确定性当作音频缺失；尚未解决未知音轨错配。区间仍条件于已接触开发来源，未经多重比较或开发选择校正。新来源确认前保留为离线候选。\n"]
    with (OUTPUT / "第十二轮鲁棒训练与知识MoE探索报告.md").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    register() if args.register_only else evaluate()
