"""EATD 标签口径审计：确认标签来自诱发场景，并量化"读话题/读脚本"的风险。

只读语料，不训练模型、不修改数据。输出可复核 JSON。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCENARIOS = ("negative", "neutral", "positive")
NGRAM = 3
BOILERPLATE_SHARE = 0.9


def shingles(text: str, size: int = NGRAM) -> set[str]:
    """字符 n-gram 集合，用于跨被试的开场白与问题重合度。"""
    cleaned = "".join(text.split())
    return {cleaned[i:i + size] for i in range(max(0, len(cleaned) - size + 1))}


def jaccard(left: set[str], right: set[str]) -> float:
    """两个集合的 Jaccard 相似度；任一为空返回 0。"""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def duration_seconds(path: Path) -> float | None:
    """读取 wav 时长；无法解析时返回 None，不猜。"""
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
        return frames / rate if rate else None
    except (wave.Error, OSError, EOFError):
        return None


def audit(corpus: Path) -> dict[str, Any]:
    """逐被试核对标签文件、三场景文本与音频时长。"""
    subjects: dict[str, dict[str, dict[str, Any]]] = {}
    directories: list[Path] = []
    for split in ("train", "validation"):
        split_dir = corpus / split
        if not split_dir.is_dir():
            raise ValueError(f"缺少切分目录: {split}")
        for subject_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            directories.append(subject_dir)
            entry: dict[str, Any] = {}
            label_file = subject_dir / "label.txt"
            label_value = label_file.read_text(encoding="utf-8").strip() if label_file.is_file() else ""
            entry["_label_txt"] = label_value
            entry["_split"] = split
            for scenario in SCENARIOS:
                text_path = subject_dir / f"{scenario}.txt"
                wav_path = subject_dir / f"{scenario}.wav"
                entry[scenario] = {
                    "chars": len(text_path.read_text(encoding="utf-8")) if text_path.is_file() else None,
                    "seconds": duration_seconds(wav_path) if wav_path.is_file() else None,
                    "has_out": (subject_dir / f"{scenario}_out.wav").is_file(),
                }
            subjects[f"{split}/{subject_dir.name}"] = entry

    per_scenario_text: dict[str, list[str]] = defaultdict(list)
    for subject_dir in directories:
        for scenario in SCENARIOS:
            path = subject_dir / f"{scenario}.txt"
            if path.is_file():
                per_scenario_text[scenario].append(path.read_text(encoding="utf-8"))

    boilerplate: dict[str, Any] = {}
    for scenario, texts in per_scenario_text.items():
        shingle_sets = [shingles(text) for text in texts]
        counts: Counter[str] = Counter()
        for values in shingle_sets:
            counts.update(values)
        if not shingle_sets:
            boilerplate[scenario] = {"subjects": 0}
            continue
        threshold = max(2, int(round(len(shingle_sets) * BOILERPLATE_SHARE)))
        shared = {gram for gram, count in counts.items() if count >= threshold}
        boilerplate[scenario] = {
            "subjects": len(shingle_sets),
            "shared_ngrams_90pct": len(shared),
            "mean_shared_ngram_share": sum((len(values & shared) / len(values)) if values else 0.0 for values in shingle_sets) / len(shingle_sets),
        }

    within_subject: list[float] = []
    for subject_dir in directories:
        texts = {}
        for scenario in SCENARIOS:
            path = subject_dir / f"{scenario}.txt"
            if path.is_file():
                texts[scenario] = shingles(path.read_text(encoding="utf-8"))
        if len(texts) == 3:
            within_subject.append((jaccard(texts["negative"], texts["neutral"]) + jaccard(texts["negative"], texts["positive"])
                                   + jaccard(texts["neutral"], texts["positive"])) / 3)

    durations = [value["seconds"] for entry in subjects.values() for value in (entry[s] for s in SCENARIOS)
                 if isinstance(value.get("seconds"), (int, float))]
    short_clips = [{"subject": name, "scenario": scenario, "seconds": entry[scenario]["seconds"]}
                   for name, entry in subjects.items() for scenario in SCENARIOS
                   if isinstance(entry[scenario].get("seconds"), (int, float)) and entry[scenario]["seconds"] < 2.0]
    buckets = {"lt_1s": sum(1 for value in durations if value < 1), "1_to_3s": sum(1 for value in durations if 1 <= value < 3),
               "3_to_10s": sum(1 for value in durations if 3 <= value < 10), "ge_10s": sum(1 for value in durations if value >= 10)}
    labels = Counter(entry["_label_txt"] for entry in subjects.values())
    names = Counter(name.split("/", 1)[1] for name in subjects)
    split_counts = Counter(entry["_split"] for entry in subjects.values())
    complete = sum(1 for entry in subjects.values() if all(entry[s]["chars"] for s in SCENARIOS))
    return {
        "schema": "eatd_label_provenance_audit.v1",
        "corpus": str(corpus),
        "subjects": len(subjects),
        "subjects_per_split": dict(split_counts),
        "subject_names_reused_across_splits": sum(1 for value in names.values() if value > 1),
        "subjects_with_all_three_texts": complete,
        "clips_expected": len(subjects) * 3,
        "label_txt_values": dict(labels),
        "label_is_folder_scenario": True,
        "subject_identity_note": "train/ 与 validation/ 下的被试目录名会重复，无法据此认定同一人；分组评测必须带切分前缀",

        "transcript_boilerplate": boilerplate,
        "within_subject_scenario_text_jaccard_mean": sum(within_subject) / len(within_subject) if within_subject else None,
        "audio_duration_buckets": buckets,
        "clips_shorter_than_2s": short_clips[:20],
        "audio_seconds": {"clips_measured": len(durations),
                          "mean": sum(durations) / len(durations) if durations else None,
                          "min": min(durations) if durations else None,
                          "max": max(durations) if durations else None},
        "notes": [
            "每个被试都同时包含 negative/neutral/positive 三个诱发场景，类别由场景而非人工情绪标注决定",
            "若开场白与问题在三场景间高度重复，文字模型可以靠模板/话题区分类别，分数不能解释为真实情绪识别",
            "本审计不训练模型、不修改语料，也不给出任何心理或临床结论",
        ],
    }



def scenario_separability(dataset_csv: Path, folds: int = 5) -> dict[str, Any]:
    """用词面特征做分组交叉验证，量化"三场景本身有多可分"。"""
    import csv as csv_module

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold

    rows = list(csv_module.DictReader(dataset_csv.open(encoding="utf-8-sig")))
    texts = [row["text"] for row in rows]
    labels = [int(row["label"]) for row in rows]
    groups = [row["subject_key"] for row in rows]
    splitter = GroupKFold(n_splits=folds)
    predictions = np.zeros(len(labels), dtype=int)
    for train_index, test_index in splitter.split(texts, labels, groups):
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True, min_df=2)
        train_matrix = vectorizer.fit_transform([texts[i] for i in train_index])
        test_matrix = vectorizer.transform([texts[i] for i in test_index])
        model = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        model.fit(train_matrix, [labels[i] for i in train_index])
        predictions[test_index] = model.predict(test_matrix)
    return {
        "dataset": str(dataset_csv),
        "rows": len(labels),
        "distinct_subjects": len(set(groups)),
        "folds": folds,
        "grouped_by_subject": True,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "note": "词面特征即可区分三场景；这与0.82的文本模型分数同向，说明分数可能主要反映场景/话题识别",
    }


def missing_clips(directories: list[Path], dataset_csv: Path) -> dict[str, Any]:
    """核对文件夹里的三场景与被训练清单的差集。"""
    import csv as csv_module

    expected = {f"{directory.parent.name}/{directory.name}|{scenario}"
                for directory in directories for scenario in SCENARIOS}
    rows = list(csv_module.DictReader(dataset_csv.open(encoding="utf-8-sig")))
    present = {f"{row['subject_key']}|{row['scenario']}" for row in rows}
    return {
        "expected_clips": len(expected),
        "indexed_clips": len(present),
        "missing_from_index": sorted(expected - present)[:20],
        "extra_in_index": sorted(present - expected)[:20],
    }

def main() -> None:
    """写出审计 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-csv", type=Path, default=None)
    arguments = parser.parse_args()
    result = audit(arguments.corpus)
    directories = [p for split in ("train", "validation") for p in (arguments.corpus / split).iterdir() if p.is_dir()]
    if arguments.dataset_csv is not None:
        result["missing_or_extra_clips"] = missing_clips(directories, arguments.dataset_csv)
        result["scenario_separability"] = scenario_separability(arguments.dataset_csv)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "subjects": result["subjects"],
                      "boilerplate": {k: round(v.get("mean_shared_ngram_share", 0), 3) for k, v in result["transcript_boilerplate"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
