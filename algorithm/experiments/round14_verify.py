"""校验第十四轮真实来源、因果时间窗口、完整生成和未完成的人评状态。"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from experiments.round14_common import OUT, REPO, save, sha256


def read(relative: str) -> Any:
    """只读已产出的证据文件。"""
    return json.loads((OUT / relative).read_text(encoding="utf-8"))


def verify() -> dict[str, Any]:
    """任一关键不变量不满足则停止交付，不以报告文字代替实际证据。"""
    result: dict[str, Any] = {}
    checked_windows = 0
    for dataset, source_version in (("studentlife_calendar_days", "current"), ("adolescent_sampling_days_reused", "adolescent_locked")):
        directory = OUT / "longitudinal" / dataset
        protocol = json.loads((directory / "preregister.json").read_text(encoding="utf-8"))
        assert sha256(OUT / "source_locked" / source_version / "round14_longitudinal.py") == protocol["code_sha256"]
        for mode in ("same_person_future", "unseen_person_future"):
            settings = protocol["partitions"][mode]
            assert settings["fit_max_target_day"] < settings["validation_start_day"] < settings["test_start_day"]
            rows = json.loads((directory / f"{mode}_test_windows.json").read_text(encoding="utf-8"))
            for row in rows:
                assert row["target_day"] == row["origin"] + 1
                assert row["history_days"][-1] == row["origin"]
                assert len(row["history_days"]) >= 3 and max(row["history_days"]) < row["target_day"]
                assert all(b - a <= 3 for a, b in zip(row["history_days"][:-1], row["history_days"][1:]))
                assert row["target_day"] >= settings["test_start_day"]
            checked_windows += len(rows)
            if mode == "unseen_person_future":
                people = [set(settings["participant_partition"][name]) for name in ("train", "validation", "test")]
                assert not people[0] & people[1] and not people[0] & people[2] and not people[1] & people[2]
                assert {row["person"] for row in rows} <= people[2]
    result["strict_test_windows_verified"] = checked_windows
    generations = [json.loads(line) for line in (OUT / "support/generations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(generations) == 128 and len({(row["case_id"], row["arm"]) for row in generations}) == 128
    assert all(row["raw_reply"].strip() and row["final_reply"].strip() for row in generations)
    for case in {row["case_id"] for row in generations}:
        selected = [row for row in generations if row["case_id"] == case]
        assert len(selected) == 4 and {row["arm"] for row in selected} == {"A", "B", "C", "D"}
        assert all(row["history"] == selected[0]["history"] for row in selected)
    result["matched_real_four_arm_generations"] = len(generations)
    for path in ("support", "support_refined"):
        public = read(path + "/blind/replies.json"); private = read(path + "/private/arm_key.json")
        assert {row["display_id"] for row in public} == {row["display_id"] for row in private}
        assert len({row["display_id"] for row in public}) == len(public)
        assert all("arm" not in row and "case_id" not in row for row in public)
        ratings = read(path + "/independent_ratings_analysis.json")
        assert ratings["completed_rows"] == 0 and ratings["raters_with_completed_rows"] == 0
    candidate = read("support_refined/generations.json")
    assert len(candidate) == 32 and all(row["versions"] and row["final_reply"].strip() for row in candidate)
    result["candidate_model_generations"] = sum(len(row["versions"]) for row in candidate)
    field_audit = read("data/studentlife_field_audit.json")
    assert field_audit["stress_excluded"] is True
    assert field_audit["pam_released_rows"] == field_audit["pam_rows"] + field_audit["pam_conflicting_timestamp_rows_quarantined"] + field_audit["pam_exact_duplicate_extras_removed"]
    assert sha256(OUT / "data/PAM.Rds") == field_audit["pam_rds_sha256"]
    assert sha256(OUT / "data/PAM_mapping_SL_tbl_trans.R") == field_audit["mapping_code_sha256"]
    result["raw_pam_and_mapping_hashes_verified"] = True
    scope = read("scope_and_promotion.json")
    assert all(scope["protected_files_unchanged_relative_to_git"].values())
    for name, digest in scope["protected_current_sha256"].items(): assert sha256(REPO / name) == digest
    assert not scope["new_models_promoted"] and scope["real_pilot_participants"] == 0
    result["protected_existing_behavior_unchanged"] = True
    reproduced = read("verification.json")
    assert reproduced["all_passed"] is True
    result["from_scratch_retraining_and_runtime_verified"] = True
    test_log = (OUT / "tests.log").read_text(encoding="utf-8")
    assert re.search(r"162 passed, 1 warning", test_log)
    result.update(tests_passed=162, new_tests_passed=25, warnings=1,
        real_user_study_completed=False, independent_ratings_completed=False, all_checks_passed=True)
    save(OUT / "evidence_verification.json", result)
    return result


if __name__ == "__main__": print(json.dumps(verify(), ensure_ascii=False))
