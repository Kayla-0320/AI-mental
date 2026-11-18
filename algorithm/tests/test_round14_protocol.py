"""第十四轮时间泄漏、量表定义、来源真实性与盲评完整性回归。"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.round14_common import freeze
from experiments.round14_human import SCHEMAS, analyze_ratings, analyze_study, init_study, timestamp
from experiments.round14_longitudinal import daily_samples, select_partitions
from experiments.round14_runtime import predict_pam
from experiments.round14_studentlife import pam_valence
from experiments.round14_support import cases, prompt, write_blind_pack
from experiments.round14_tracking import tracking_summary


def write_table(path: Path, name: str, rows: list[dict]) -> None:
    """仅向pytest临时目录写入清楚标注的测试记录。"""
    with (path / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCHEMAS[name]); writer.writeheader(); writer.writerows(rows)


def artifact(kind: str = "ema") -> dict:
    """构建隔离测试配置，不进入研究结果或真实用户目录。"""
    return {"protocol": {"source": {"source": "https://zenodo.org/records/3529253"}},
        "config": {"id": "ema_0.3", "kind": kind, "alpha": .3}, "train_median": .5, "validation_error_radius": .2}


def consent() -> dict:
    """仅测试真实性校验路径的夹具，不能当作研究同意或人类证据。"""
    return {"participant_id": "fixture", "consented_at": "2026-10-25T08:00:00+08:00", "adult_confirmed": "1", "consent_version": "test_fixture", "withdrawn_at": "", "source": "real"}


def test_future_values_never_enter_features() -> None:
    frame = pd.DataFrame({"person": ["p"] * 10, "day": range(10), "value": np.linspace(.1, .9, 10)})
    first = [row for row in daily_samples(frame) if row["origin"] == 4][0]
    frame.loc[frame.day >= 5, "value"] = 0
    second = [row for row in daily_samples(frame) if row["origin"] == 4][0]
    assert first["x"] == second["x"] and first["history"] == second["history"]
    assert first["y"] != second["y"]
    assert max(first["history_days"]) < first["target_day"]


def test_missing_days_not_compressed_into_next_day() -> None:
    frame = pd.DataFrame({"person": ["p"] * 5, "day": [0, 1, 2, 8, 9], "value": [.2] * 5})
    assert daily_samples(frame) == []


def test_both_time_and_people_separated() -> None:
    samples = [{"person": str(person), "target_day": day, "origin": day - 1} for person in range(30) for day in range(3, 30)]
    parts, _ = select_partitions(samples, "unseen_person_future")
    groups = [{row["person"] for row in parts[name]} for name in ("train", "validation", "test")]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert max(row["target_day"] for row in parts["train"]) < min(row["target_day"] for row in parts["validation"])
    assert max(row["target_day"] for row in parts["validation"]) < min(row["target_day"] for row in parts["test"])


def test_duplicate_days_rejected() -> None:
    with pytest.raises(ValueError): daily_samples(pd.DataFrame({"person": ["p", "p"], "day": [1, 1], "value": [.2, .4]}))


def test_pam_mapping_matches_all_sixteen_primary_codes() -> None:
    assert pam_valence(np.arange(1, 17)).tolist() == [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]
    for values in ([0], [17], [1.5], [np.nan]):
        with pytest.raises(ValueError): pam_valence(np.array(values))


@pytest.mark.parametrize("change", ["future", "duplicate", "unsorted", "wrong_scale", "mixed_people", "nan"])
def test_runtime_rejects_invalid_or_leaking_history(change: str) -> None:
    rows = [{"person": "p", "day": i, "value": 2.} for i in range(3)]
    schema = "studentlife_pam_valence_1_4"
    if change == "future": rows[-1]["day"] = 3
    elif change == "duplicate": rows[-1]["day"] = 1
    elif change == "unsorted": rows.reverse()
    elif change == "wrong_scale": schema = "project_mood_0_100"
    elif change == "mixed_people": rows[-1]["person"] = "other"
    elif change == "nan": rows[-1]["value"] = float("nan")
    with pytest.raises(ValueError): predict_pam(artifact(), rows, 3, schema)


def test_runtime_abstains_on_gaps_and_cold_start() -> None:
    rows = [{"person": "p", "day": i, "value": 2.} for i in [0, 1, 2, 8, 9]]
    assert predict_pam(artifact(), rows, 10)["status"] == "abstain"
    assert predict_pam(artifact(), rows[:3], 5)["reason"] == "latest_day_missing"


def test_freeze_disallows_posthoc_protocol_change(tmp_path: Path) -> None:
    freeze(tmp_path / "p.json", {"test_used_for_selection": False})
    with pytest.raises(ValueError): freeze(tmp_path / "p.json", {"test_used_for_selection": True})


def test_all_arms_receive_identical_raw_information() -> None:
    case = cases()[0]
    all_messages = [prompt(case, arm) for arm in ("A", "B", "C", "D")]
    assert all(messages[1:] == all_messages[0][1:] for messages in all_messages)
    assert "本人已纠正" not in all_messages[0][0]["content"]
    assert "本人已纠正" in all_messages[1][0]["content"]
    assert "本轮目标计划" not in all_messages[1][0]["content"]
    assert "本轮目标计划" in all_messages[2][0]["content"]


def test_blind_materials_do_not_expose_arm_or_case(tmp_path: Path) -> None:
    rows = [{"case_id": "s", "arm": arm, "split": "test", "history": [], "final_reply": "reply", "final_status": "audit_passed"} for arm in ("A", "B", "C", "D")]
    write_blind_pack(rows, tmp_path)
    public = json.loads((tmp_path / "blind/replies.json").read_text(encoding="utf-8"))
    assert all("arm" not in row and "case_id" not in row for row in public)
    assert len(json.loads((tmp_path / "private/arm_key.json").read_text(encoding="utf-8"))) == 4
    result = analyze_ratings(tmp_path / "blind", tmp_path / "private/arm_key.json", tmp_path / "analysis.json")
    assert result["raters_with_completed_rows"] == 0 and result["completed_rows"] == 0


def test_empty_study_never_claims_effectiveness(tmp_path: Path) -> None:
    init_study(tmp_path)
    result = analyze_study(tmp_path, tmp_path / "analysis.json")
    assert result["consented_adults"] == 0 and result["acceptance_rate"] is None
    assert not result["clinical_effectiveness_evidence"]


def test_synthetic_rows_cannot_enter_real_study(tmp_path: Path) -> None:
    init_study(tmp_path); row = consent(); row["source"] = "synthetic"
    write_table(tmp_path, "consent", [row])
    with pytest.raises(ValueError, match="模拟"): analyze_study(tmp_path, tmp_path / "analysis.json")


def test_unconsented_and_future_observations_rejected(tmp_path: Path) -> None:
    init_study(tmp_path)
    diary = {"record_id": "fixture-1", "participant_id": "fixture", "observed_at": "2026-10-26T08:00:00+08:00", "submitted_at": "2026-10-26T08:01:00+08:00", "mood_0_100": "50", "stress_0_100": "40", "source": "real"}
    write_table(tmp_path, "diary", [diary])
    with pytest.raises(ValueError, match="同意"): analyze_study(tmp_path, tmp_path / "analysis.json")
    write_table(tmp_path, "consent", [consent()])
    diary["observed_at"] = "2028-10-26T08:00:00+08:00"; write_table(tmp_path, "diary", [diary])
    with pytest.raises(ValueError): analyze_study(tmp_path, tmp_path / "analysis.json")


def test_timestamps_require_timezone() -> None:
    with pytest.raises(ValueError): timestamp("2026-11-17T00:00:00")


def test_withdrawal_excludes_all_observations(tmp_path: Path) -> None:
    init_study(tmp_path); row = consent(); row["withdrawn_at"] = "2026-11-01T08:00:00+08:00"
    write_table(tmp_path, "consent", [row])
    write_table(tmp_path, "diary", [{"record_id": "fixture-1", "participant_id": "fixture", "observed_at": "2026-10-26T08:00:00+08:00", "submitted_at": "2026-10-26T08:01:00+08:00", "mood_0_100": "50", "stress_0_100": "40", "source": "real"}])
    result = analyze_study(tmp_path, tmp_path / "analysis.json")
    assert result["diary_rows"] == 0 and result["active_adults"] == 0
    assert tracking_summary(tmp_path, "fixture")["status"] == "withdrawn"


def test_current_day_excluded_from_personal_reference(tmp_path: Path) -> None:
    init_study(tmp_path); write_table(tmp_path, "consent", [consent()])
    start = datetime(2026, 10, 26, 8, tzinfo=timezone(timedelta(hours=8)))
    rows = [{"record_id": f"fixture-{i}", "participant_id": "fixture", "observed_at": (start + timedelta(days=i)).isoformat(),
        "submitted_at": (start + timedelta(days=i, minutes=1)).isoformat(), "mood_0_100": str(100 if i == 7 else 20), "stress_0_100": "40", "source": "real"} for i in range(8)]
    write_table(tmp_path, "diary", rows)
    result = tracking_summary(tmp_path, "fixture", start + timedelta(days=7, hours=1))
    assert result["personal_reference"]["mood"] == 20 and result["current_day"]["mood"] == 100
    assert result["current_day"]["date"] not in result["reference_dates"]
    assert result["prediction"] is None and not result["clinical_alert"]


def test_no_human_declaration_no_completed_ratings(tmp_path: Path) -> None:
    rows = [{"case_id": "s", "arm": arm, "split": "test", "history": [], "final_reply": "reply", "final_status": "audit_passed"} for arm in ("A", "B", "C", "D")]
    write_blind_pack(rows, tmp_path)
    path = tmp_path / "blind/rater_1.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream: ratings = list(csv.DictReader(stream))
    for row in ratings:
        row.update(fidelity_1_5="4", appropriateness_1_5="4", correction_violation="0", goal_violation="0", unsupported_details="0", repeat_question="0")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ratings[0])); writer.writeheader(); writer.writerows(ratings)
    with pytest.raises(ValueError, match="独立人评声明"):
        analyze_ratings(tmp_path / "blind", tmp_path / "private/arm_key.json", tmp_path / "analysis.json")


def test_complete_paired_ratings_and_constant_kappa(tmp_path: Path) -> None:
    rows = [{"case_id": "s", "arm": arm, "split": "test", "history": [], "final_reply": "reply", "final_status": "audit_passed"} for arm in ("A", "B", "C", "D")]
    write_blind_pack(rows, tmp_path)
    for rater in ("rater_1", "rater_2"):
        path = tmp_path / "blind" / f"{rater}.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream: ratings = list(csv.DictReader(stream))
        for row in ratings:
            row.update(fidelity_1_5="4", appropriateness_1_5="4", correction_violation="0", goal_violation="0", unsupported_details="0", repeat_question="0")
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ratings[0])); writer.writeheader(); writer.writerows(ratings)
    declarations = [{"rater_id": rater, "human_completed": True, "independent_of_development": True, "declared_at": "2026-11-16T12:00:00+08:00"} for rater in ("rater_1", "rater_2")]
    (tmp_path / "blind/rater_declarations.json").write_text(json.dumps(declarations), encoding="utf-8")
    result = analyze_ratings(tmp_path / "blind", tmp_path / "private/arm_key.json", tmp_path / "analysis.json")
    assert result["status"] == "complete" and result["fully_paired_cases"] == 1
    assert result["agreement"]["fidelity_1_5"]["weighted_kappa"] is None
    assert result["descriptive_differences"]["goal_violation"]["candidate_minus_reference"] == 0


def test_duplicate_real_diary_ids_rejected(tmp_path: Path) -> None:
    init_study(tmp_path); write_table(tmp_path, "consent", [consent()])
    row = {"record_id": "fixture-1", "participant_id": "fixture", "observed_at": "2026-10-26T08:00:00+08:00", "submitted_at": "2026-10-26T08:01:00+08:00", "mood_0_100": "50", "stress_0_100": "40", "source": "real"}
    write_table(tmp_path, "diary", [row, row])
    with pytest.raises(ValueError, match="重复"): analyze_study(tmp_path, tmp_path / "analysis.json")


def test_followup_denominator_excludes_future_study_days(tmp_path: Path) -> None:
    init_study(tmp_path); write_table(tmp_path, "consent", [consent()])
    row = {"record_id": "fixture-1", "participant_id": "fixture", "observed_at": "2026-10-26T08:00:00+08:00", "submitted_at": "2026-10-26T08:01:00+08:00", "mood_0_100": "50", "stress_0_100": "40", "source": "real"}
    write_table(tmp_path, "diary", [row])
    result = analyze_study(tmp_path, tmp_path / "analysis.json", timestamp("2026-10-27T08:00:00+08:00"))
    assert result["fully_elapsed_24h_study_windows"] == 2
    assert result["diary_completion_rate_in_elapsed_windows"] == .5
    assert not result["rqi_construct_validated"]
