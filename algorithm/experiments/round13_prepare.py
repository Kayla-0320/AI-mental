"""将新公开语料转换为显式训练/验证与封存测试入口，保留原目标定义。"""
from __future__ import annotations

import csv
from pathlib import Path
import zipfile
import numpy as np
from experiments.round13_data import DATA, OUTPUT, load_pickle, save
from experiments.round11_common import digest


def pooled(values: np.ndarray) -> np.ndarray:
    """按真实非补零帧池化；NaN/Inf显式清理，保留有效性供质量审计。"""
    values = np.nan_to_num(np.asarray(values, np.float64), nan=0., posinf=0., neginf=0.)
    mask = np.any(values != 0., axis=2)
    count = mask.sum(1).clip(1)[:, None]
    mean = (values * mask[:, :, None]).sum(1) / count
    variance = ((values-mean[:, None])**2 * mask[:, :, None]).sum(1) / count
    return np.c_[mean, np.sqrt(variance)].astype(np.float32)


def write(dataset: str, split: str, inputs: dict[str, np.ndarray], labels: np.ndarray) -> None:
    """测试输入工件中不包含标签，最终评测必须先完成模型/逐条预测锁定。"""
    folder = OUTPUT / "datasets" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, int)
    if split == "test":
        np.savez_compressed(folder / "test_inputs.npz", **inputs)
        np.savez_compressed(folder / "sealed_test_labels.npz", sample_ids=inputs["sample_ids"], labels=labels)
    else:
        np.savez_compressed(folder / f"{split}.npz", **inputs, labels=labels)


def mosi() -> None:
    """MOSI按连续情感符号得到原负/中/正，不把零分并入任一极性。"""
    target = OUTPUT / "datasets/mosi/manifest.json"
    if target.exists():
        return
    obj = load_pickle(DATA / "mosi/Processed/aligned_50.pkl")
    rows = list(csv.DictReader((DATA / "mosi/label.csv").open(encoding="utf-8")))
    lookup = {r["video_id"] + "$_$" + r["clip_id"]: r for r in rows}
    groups = {}
    media_inputs = []
    checks = []
    for split, source in obj.items():
        ids = np.array(source["id"]).astype(str)
        groups[split] = set(str(i).split("$_$")[0] for i in ids)
        labels = np.sign(np.asarray(source["regression_labels"])).astype(int) + 1
        if len(np.unique(ids)) != len(ids):
            raise ValueError("MOSI重复ID")
        for i, key in enumerate(ids):
            row = lookup[key]
            if row["mode"] != split or np.sign(float(row["label"]))+1 != labels[i]:
                raise ValueError("MOSI媒体与特征标注不一致")
            media_inputs.append(dict(id=key.replace("$_$", "__"), split=split, text=row["text"],
                group=row["video_id"], media=f"Raw/{row['video_id']}/{row['clip_id']}.mp4"))
        inputs = dict(sample_ids=ids, source_groups=np.array([str(i).split("$_$")[0] for i in ids]),
            raw_text=np.array(source["raw_text"]).astype(str), T=pooled(source["text"]),
            A=pooled(source["audio"]), V=pooled(source["vision"]))
        write("mosi", split, inputs, labels)
        checks.append(dict(split=split, n=len(ids), source_count=len(groups[split])))
    overlaps = {a+"_"+b:len(groups[a]&groups[b]) for a,b in (("train","valid"),("train","test"),("valid","test"))}
    if any(overlaps.values()):
        raise ValueError("MOSI官方视频来源跨划分")
    with zipfile.ZipFile(DATA / "mosi/Raw.zip") as archive:
        names = set(archive.namelist())
        for row in media_inputs:
            if row["media"] not in names:
                raise ValueError("MOSI对应媒体缺失")
            destination = (DATA / "mosi" / row["media"]).resolve()
            if not destination.is_relative_to((DATA / "mosi").resolve()):
                raise ValueError("媒体路径越界")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(row["media"]) as source, destination.open("wb") as out:
                    import shutil
                    shutil.copyfileobj(source, out)
    save(OUTPUT / "datasets/mosi/media_inputs.json", media_inputs)
    save(target, dict(dataset="CMU-MOSI", language="English", task="M3 sentiment from sign of original continuous labels",
        splits=checks, source_overlap=overlaps, raw_media_n=len(media_inputs),
        raw_sha256=digest(DATA / "mosi/Raw.zip"), feature_sha256=digest(DATA / "mosi/Processed/aligned_50.pkl"),
        source="https://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/",
        mirror="https://huggingface.co/datasets/tamb2203579/CMU-MOSI", previous_use_audit="no MOSI references found in current experiment/scripts; not proof against undocumented historical access",
        label_boundary="preparation reads labels solely for integrity and quarantine; model selection never opens sealed_test_labels"))


def meld() -> None:
    """只使用发布者给出的整体Sentiment字段，不从七情绪反推极性。"""
    target = OUTPUT / "datasets/meld/manifest.json"
    if target.exists():
        return
    path = DATA / "meld/extracted"
    records = load_pickle(path / "data_sentiment.p")[0]
    text = load_pickle(path / "text_glove_average_sentiment.pkl")
    audio = load_pickle(path / "audio_embeddings_feature_selection_sentiment.pkl")
    mapping = {"negative":0, "neutral":1, "positive":2}
    summaries = []
    for j, split in enumerate(("train", "valid", "test")):
        rows = [r for r in records if r["split"] == ("dev" if split == "valid" else split)]
        if not rows and split == "valid":
            rows = [r for r in records if r["split"] == "val"]
        if not rows:
            raise ValueError(f"MELD划分名称未匹配: {split}, {set(r['split'] for r in records)}")
        keys = [str(r["dialog"])+"_"+str(r["utterance"]) for r in rows]
        if set(keys) != set(text[j]) or set(keys) != set(audio[j]):
            raise ValueError("MELD标注与特征ID未完整对齐")
        inputs = dict(sample_ids=np.array([split+"/"+k for k in keys]),
            source_groups=np.array([split+"/dialog"+str(r["dialog"]) for r in rows]),
            raw_text=np.array([r["text"] for r in rows]), T=np.stack([text[j][k] for k in keys]).astype(np.float32),
            A=np.stack([audio[j][k] for k in keys]).astype(np.float32), V=np.zeros((len(keys),1),np.float32))
        labels = np.array([mapping[r["y"]] for r in rows])
        write("meld", split, inputs, labels)
        summaries.append(dict(split=split, n=len(keys), groups=len(np.unique(inputs["source_groups"]))))
    save(target, dict(dataset="MELD", task="original human3 sentiment", language="English", modalities="released T/A features; V not provided in this experiment",
        splits=summaries, source="https://github.com/declare-lab/MELD", feature_archive_sha256=digest(DATA/"meld/MELD.Features.Models.tar.gz"),
        limits="official conversation split; recurring characters/episodes may overlap; not unseen-identity validation; audio feature selection may use official training supervision",
        label_boundary="test labels read only for preparation/quarantine until predictions/model lock"))


if __name__ == "__main__":
    mosi()
    meld()
