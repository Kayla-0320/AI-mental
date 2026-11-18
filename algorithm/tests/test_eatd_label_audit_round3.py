"""EATD 标签口径审计脚本的边界测试，只用合成目录。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / "scripts/round3_eatd_label_audit.py"
    spec = importlib.util.spec_from_file_location("round3_eatd_label_audit", path)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def corpus(root: Path, subjects: int = 2) -> Path:
    prompts = {"negative": "请描述让你难过的事情。", "neutral": "请描述你平常的一天。", "positive": "请描述让你开心的事情。"}
    bodies = {"negative": "被冷漠忽视", "neutral": "上课写作业", "positive": "庆祝得奖"}
    for split, names in (("train", range(1, subjects + 1)), ("validation", range(subjects + 1, subjects * 2 + 1))):
        for name in names:
            directory = root / split / str(name)
            directory.mkdir(parents=True)
            (directory / "label.txt").write_text("37.0", encoding="utf-8")
            for scenario in ("negative", "neutral", "positive"):
                text = prompts[scenario] + bodies[scenario] + f"{split}{name}"
                (directory / f"{scenario}.txt").write_text(text, encoding="utf-8")
    return root


def test_shingles_and_jaccard_reject_empty_input() -> None:
    audit = module()
    assert audit.shingles("ab") == set()
    assert audit.shingles("abcd") == {"abc", "bcd"}
    assert audit.jaccard(set(), {"abc"}) == 0.0
    assert audit.jaccard({"abc"}, {"abc"}) == 1.0


def test_audit_treats_labels_as_scenario_and_counts_subjects(tmp_path: Path) -> None:
    audit = module()
    result = audit.audit(corpus(tmp_path / "EATD"))
    assert result["subjects"] == 4
    assert result["subjects_with_all_three_texts"] == 4
    assert result["clips_expected"] == 12
    assert result["label_txt_values"] == {"37.0": 4}
    assert result["label_is_folder_scenario"] is True
    # 合成语料让同一场景共享前缀，因此模板 n-gram 必须被检出。
    assert result["transcript_boilerplate"]["negative"]["shared_ngrams_90pct"] > 0
    assert 0 < result["within_subject_scenario_text_jaccard_mean"] < 0.35
    assert result["audio_seconds"]["clips_measured"] == 0


def test_audit_rejects_missing_split(tmp_path: Path) -> None:
    audit = module()
    (tmp_path / "train").mkdir()
    with pytest.raises(ValueError, match="validation"):
        audit.audit(tmp_path)
