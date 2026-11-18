"""冻结语言模型任务表征的来源OOF融合，不调用少样本标签或历史调参。"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.special import softmax

from experiments import round10_external_fusion as external
from experiments import round10_represented_fusion as represented
from experiments.round10_fusion import OUTPUT, load_npz, save

PACKS = {
    "llmface": ("llm_last", "bge_cls", "face_summary", "sent_confidence"),
    "llmfull": ("llm_last", "aux_mean", "bge_cls", "face_summary", "visual", "audio", "sent_confidence"),
    "llmmean": ("llm_last", "llm_mean", "face_summary", "sent_confidence"),
}


def data_for(splits: tuple[str, ...]) -> dict[str, np.ndarray]:
    """加载目标标签无关的冻结任务表示，三类logit只是当前文字特征。"""
    data = represented.data_for(splits, False)
    parts = [load_npz(OUTPUT / "llm_features" / f"{split}.npz") for split in splits]
    if not np.array_equal(data["sample_ids"], np.concatenate([part["sample_ids"] for part in parts])):
        raise ValueError("任务表征ID未对齐")
    for name in ("llm_last", "llm_mean", "llm_scores"):
        data[name] = np.concatenate([part[name] for part in parts])
    probability = softmax(data["llm_scores"], axis=1)
    delta = data["llm_scores"][:, 2]-data["llm_scores"][:, 0]
    data["sent_scores"] = data["llm_scores"][:, [0, 2]]
    data["sent_confidence"] = np.column_stack([data["llm_scores"], delta, abs(delta), delta**2,
        probability[:, 1], -(probability*np.log(np.maximum(probability, 1e-12))).sum(1)])
    return data


def configure() -> None:
    """复用同一来源划分、归一化、训练与选型实现，避免重复入口漂移。"""
    external.PACKS = PACKS
    represented.PACKS.update(PACKS)
    external.data_for = data_for
    external.PHASE = 5


def main() -> None:
    """开发搜索、固定来源袋装拟合及评测分开执行。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("search", "refit", "evaluate"))
    args = parser.parse_args()
    configure()
    {"search": external.search, "refit": external.refit, "evaluate": external.evaluate}[args.action](OUTPUT / "llm_fusion")


if __name__ == "__main__":
    main()
