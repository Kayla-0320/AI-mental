"""替换语音证据表征后，复用严格嵌套的官方单模态监督协议。"""
from __future__ import annotations

import argparse
import numpy as np
from experiments import round10_evidence as evidence
from experiments import round10_represented_fusion as represented
from experiments.round10_fusion import OUTPUT, load_npz

original_data_for = evidence.data_for


def data_for(splits: tuple[str, ...]) -> dict[str, np.ndarray]:
    """语音缓存逐行对齐；失败或过短语音保持独立不可用标志。"""
    data = original_data_for(splits)
    parts = [load_npz(OUTPUT / "speech_features" / f"{split}.npz") for split in splits]
    if not np.array_equal(np.concatenate([part["sample_ids"] for part in parts]), data["sample_ids"]):
        raise ValueError("语音ID未对齐")
    for name in ("speech", "speech_emotion"):
        data[name] = np.concatenate([part[name] for part in parts])
    data["speech_available"] = np.concatenate([part["available"] for part in parts])
    return data


def configure() -> None:
    """只改变A和M的输入证据，其余代码与分组协议复用。"""
    evidence.data_for = data_for
    evidence.PACKS["view_A"] = ("audio", "speech", "speech_emotion")
    evidence.PACKS["view_M"] = ("llm_last", "aux_mean", "bge_cls", "face_summary", "audio", "visual", "speech", "speech_emotion")
    represented.PACKS.update(evidence.PACKS)


def main() -> None:
    """来源OOF训练、最终拟合、历史诊断分开执行。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("search", "refit", "evaluate"))
    parser.add_argument("--fine", action="store_true")
    args = parser.parse_args()
    configure()
    if args.fine:
        evidence.EXPERT_CLASSES = 5
    output = OUTPUT / ("speech_evidence5" if args.fine else "speech_evidence")
    output.mkdir(exist_ok=True)
    {"search": evidence.search, "refit": evidence.refit, "evaluate": evidence.evaluate}[args.action](output)


if __name__ == "__main__":
    main()
