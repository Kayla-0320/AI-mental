"""第九轮增补四：构建"强表征"特征包，用于检验换表征能否真正涨点。

来源（均与第三轮划分逐行对齐，构建时用 sample_ids 再核一次）：
- 文本：第五轮冻结预训练中文 RoBERTa 的均值表征（768 维），替换 CLIP 文本 512 维
- 声音：沿用 46 维 MFCC
- 视觉：沿用冻结 CLIP 三帧 512 维
- 人脸：第五轮 YuNet+HSEmotion 五帧统计 29 维（可选第四路）

产出两个包：
- bundle_text：三模态（text 768 / audio 46 / visual 512）
- bundle_text_face：四模态（再加 face 29）

边界：文本来自冻结预训练编码器，没有使用本批标签微调；历史测试集已经多轮查看，
因此新分数仍属于重复诊断，不作为全新独立来源的泛化确认。
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\Develop\AIC")
R3 = ROOT / "_国赛优化" / "round3" / "model" / "ch_sims_v2"
R5 = ROOT / "_国赛优化" / "round5"
OUT = ROOT / "_国赛优化" / "round9_additions"
SPLITS = ("train", "selection", "test")


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def build(name: str, with_face: bool) -> Path:
    target = OUT / name
    if target.exists() and any(target.glob("*.npz")):
        raise SystemExit("拒绝覆盖旧特征包；请使用--output指定新目录")
    target.mkdir(parents=True, exist_ok=True)
    dims: dict[str, int] = {}
    for split in SPLITS:
        base = load(R3 / f"{split}.npz")
        strong = load(R5 / "text_features" / f"{split}.npz")
        face = load(R5 / "face_features" / f"{split}.npz")
        if not (np.array_equal(base["sample_ids"], strong["sample_ids"])
                and np.array_equal(base["sample_ids"], face["sample_ids"])):
            raise SystemExit(f"{split} 划分未逐行对齐，拒绝构建")
        payload = {
            "text": strong["roberta_mean"].astype(np.float32),
            "audio": base["audio"].astype(np.float32),
            "visual": base["visual"].astype(np.float32),
            "modality_mask": base["modality_mask"].astype(bool),
            "quality": base["quality"].astype(np.float32),
            "labels": base["labels"],
            "sample_ids": base["sample_ids"],
            "source_groups": base["source_groups"],
            "raw_text": base["raw_text"],
            "original_split": base["original_split"],
        }
        if with_face:
            face_summary = face["face_summary"].astype(np.float32)
            if not np.isfinite(face_summary).all():
                raise SystemExit(f"{split} 人脸特征含非有限值")
            payload["face"] = face_summary
            present = face_summary[:, 24:25] > 0
            payload["modality_mask"] = np.concatenate([payload["modality_mask"], present], axis=1)
            payload["quality"] = np.concatenate([payload["quality"], present.astype(np.float32)], axis=1)
        np.savez_compressed(target / f"{split}.npz", **payload)
        dims = {k: int(v.shape[1]) for k, v in payload.items()
                if k in ("text", "audio", "visual", "face")}
    metadata = {
        "schema_version": "paired_features_v1",
        "task": "ch_sims_v2_sentiment_polarity_3",
        "dataset": "CH-SIMS-v2",
        "class_names": ["negative", "neutral", "positive"],
        "modality_order": ["text", "audio", "visual"] + (["face"] if with_face else []),
        "feature_dims": dims,
        "feature_source": {
            "text": "round5 frozen pretrained chinese-roberta mean pooling (768); no target-label training",
            "audio": "librosa mfcc46",
            "visual": "frozen chinese-clip 3-frame mean (512)",
            "face": "round5 yunet+hsemotion 5-frame summary (29)",
        },
        "limitations": [
            "文本编码器在同一批数据上微调过，本包分数不是未见数据泛化",
            "视觉仍为三帧池化，声音仍为手工特征",
        ],
    }
    (target / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), "utf-8")
    sizes = {split: len(load(target / f"{split}.npz")["labels"]) for split in SPLITS}
    print(f"{name}: dims={dims} sizes={sizes}")
    return target


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    OUT = parser.parse_args().output
    build("bundle_text", with_face=False)
    build("bundle_text_face", with_face=True)


if __name__ == "__main__":
    main()
