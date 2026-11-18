"""在相同来源协议下对照本地7B冻结语义表征，不更改任务标签。"""
from __future__ import annotations

import argparse
import numpy as np

from experiments import round10_evidence as evidence
from experiments import round10_speech_evidence as speech
from experiments.round10_fusion import OUTPUT, load_npz


def data_for(splits: tuple[str, ...]) -> dict[str, np.ndarray]:
    """冻结7B与3B使用完全相同提示，逐行核对后替换语义特征。"""
    data = speech.data_for(splits)
    parts = [load_npz(OUTPUT / "llm_large_features" / f"{split}.npz") for split in splits]
    if not np.array_equal(data["sample_ids"], np.concatenate([part["sample_ids"] for part in parts])):
        raise ValueError("7B语言表征ID不一致")
    for name in ("llm_last", "llm_mean", "llm_scores"):
        data[name] = np.concatenate([part[name] for part in parts])
    return data


def configure() -> None:
    """仅M/T专家的冻结语义宽度变化，监督与元特征协议保持相同。"""
    speech.configure()
    evidence.data_for = data_for


def main() -> None:
    """三档与五档监督为同一预定义对照，选型不读取历史成绩。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("search", "refit"))
    parser.add_argument("--fine", action="store_true")
    args = parser.parse_args()
    configure()
    if args.fine:
        evidence.EXPERT_CLASSES = 5
    output = OUTPUT / ("large_speech5" if args.fine else "large_speech")
    output.mkdir(exist_ok=True)
    {"search": evidence.search, "refit": evidence.refit}[args.action](output)


if __name__ == "__main__":
    main()
