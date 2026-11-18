"""审计中国青少年两波原表；题义、时距和身份未核实前机器阻止训练。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_SHA256 = "eeb680f188860ed54a1163c2f655886166c6c6e3241ee4bba267ccc14163555f"
SOURCE = "https://data.mendeley.com/datasets/d8h6gbs4jk/1"


def unresolved_definition() -> dict[str, Any]:
    """生成无法误启用训练的待核实任务模板，不根据题号猜量表。"""
    return {
        "schema": "two_wave_task.v1", "dataset_sha256": EXPECTED_SHA256,
        "task_id": "undefined_pending_primary_codebook", "participant_columns": ["cohort", "code"],
        "participant_identity_verified": False,
        "codebook": {"verified": False, "primary_source_url": "", "column_definitions": {}},
        "baseline_features": [], "target_column": None,
        "followup_interval": {"verified": False, "minimum_days": None, "maximum_days": None, "primary_source_url": ""},
        "missing_codes_verified": False, "missing_codes": {},
        "scoring_verified": False, "target_meaning": "", "target_unit": "",
        "duplicate_policy": "quarantine_all_duplicate_participants",
        "notes": "两波关联/预测候选；不能称长序列，不能把g1..g9题号直接命名为PHQ。",
    }


def training_blockers(frame: pd.DataFrame, definition: dict[str, Any], source_sha: str) -> list[str]:
    """检查训练前必须已被原始问卷/说明核实的条件，缺项一律拒绝训练。"""
    failures: list[str] = []
    if not isinstance(definition, dict) or definition.get("schema") != "two_wave_task.v1":
        return ["invalid_task_definition_schema"]
    if source_sha != definition.get("dataset_sha256"):
        failures.append("source_digest_mismatch")
    keys = definition.get("participant_columns")
    if not isinstance(keys, list) or not keys or not all(isinstance(key, str) and key in frame.columns for key in keys):
        failures.append("undefined_participant_columns")
    elif frame[keys].isna().any().any():
        failures.append("missing_participant_identity")
    if definition.get("participant_identity_verified") is not True:
        failures.append("participant_identity_semantics_unverified")
    codebook = definition.get("codebook")
    if not isinstance(codebook, dict) or codebook.get("verified") is not True or not str(codebook.get("primary_source_url", "")).startswith("https://"):
        failures.append("primary_codebook_unverified")
        mapping: dict[str, Any] = {}
    else:
        mapping = codebook.get("column_definitions", {})
        if not isinstance(mapping, dict):
            mapping = {}
            failures.append("invalid_column_dictionary")
    features = definition.get("baseline_features")
    if not isinstance(features, list) or not features or not all(isinstance(name, str) for name in features):
        failures.append("undefined_baseline_features")
        features = []
    elif len(set(features)) != len(features):
        failures.append("duplicate_feature_columns")
    target = definition.get("target_column")
    if not isinstance(target, str) or target not in frame.columns or not target.endswith("_2"):
        failures.append("undefined_followup_target")
    elif target in features:
        failures.append("target_in_predictors")
    for name in features:
        info = mapping.get(name, {})
        if name not in frame.columns:
            failures.append(f"unknown_feature:{name}")
        if not isinstance(info, dict) or info.get("wave") != 1 or info.get("available_at_baseline") is not True or not info.get("meaning") or not info.get("source_locator"):
            failures.append(f"feature_temporality_or_meaning_unverified:{name}")
        if name.endswith("_2") or name in {"wave2_followup", "date2"}:
            failures.append(f"followup_predictor_forbidden:{name}")
        if keys and isinstance(keys, list) and name in keys:
            failures.append(f"participant_identifier_as_feature:{name}")
    if isinstance(target, str):
        target_info = mapping.get(target, {})
        if not isinstance(target_info, dict) or target_info.get("wave") != 2 or not target_info.get("meaning") or not target_info.get("source_locator"):
            failures.append("target_meaning_or_wave_unverified")
    interval = definition.get("followup_interval")
    if not isinstance(interval, dict) or interval.get("verified") is not True or not str(interval.get("primary_source_url", "")).startswith("https://"):
        failures.append("followup_interval_unverified")
    else:
        low, high = interval.get("minimum_days"), interval.get("maximum_days")
        if type(low) not in (float, int) or type(high) not in (float, int) or not all(math.isfinite(v) for v in (low, high)) or not 0 < low <= high:
            failures.append("invalid_followup_interval")
    if definition.get("missing_codes_verified") is not True:
        failures.append("missing_value_codes_unverified")
    elif not isinstance(definition.get("missing_codes"), dict) or any(name not in definition["missing_codes"] for name in [*features, target] if isinstance(name, str)):
        failures.append("missing_value_policy_incomplete")
    if definition.get("scoring_verified") is not True or not definition.get("target_meaning") or not definition.get("target_unit"):
        failures.append("target_scoring_unverified")
    if definition.get("duplicate_policy") != "quarantine_all_duplicate_participants":
        failures.append("ambiguous_duplicate_policy")
    return sorted(set(failures))


def audit_workbook(source: Path, output_dir: Path, definition_path: Path | None = None) -> dict[str, Any]:
    """保持原表不变，输出来源、元数据、重复、缺失及可训练性审计。"""
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    if source_sha != EXPECTED_SHA256:
        raise ValueError("原始文件SHA变化，需重新审核版本")
    frame = pd.read_excel(source, engine="openpyxl")
    output_dir.mkdir(parents=True, exist_ok=True)
    definition = json.loads(definition_path.read_text(encoding="utf-8")) if definition_path else unresolved_definition()
    duplicate_mask = frame.duplicated(["cohort", "code"], keep=False)
    with zipfile.ZipFile(source) as archive:
        members = archive.namelist()
        comments = [name for name in members if "comment" in name.lower()]
        property_files = [name for name in members if name.startswith("docProps/") and name.endswith(".xml")]
        properties = {name: archive.read(name).decode("utf-8") for name in property_files}
    first = [name for name in frame.columns if str(name).endswith("_1")]
    paired = [name[:-2] for name in first if name[:-2] + "_2" in frame.columns]
    date_values = pd.to_datetime(frame["date2"], format="mixed", errors="coerce")
    candidate = ["g1_1", "g9_1", "h1_1", "g1_2", "g9_2", "h1_2"]
    blockers = training_blockers(frame, definition, source_sha)
    audit: dict[str, Any] = {
        "dataset_source": SOURCE, "source_sha256": source_sha, "license": "CC BY 4.0",
        "rows": len(frame), "columns": len(frame.columns), "wave_count": 2,
        "long_sequence": False, "age_min_max": [int(frame.age.min()), int(frame.age.max())],
        "participant_key_candidate": ["cohort", "code"],
        "candidate_unique_keys": int(len(frame.drop_duplicates(["cohort", "code"]))),
        "duplicate_key_rows_to_quarantine": int(duplicate_mask.sum()),
        "duplicate_extra_rows": int(frame.duplicated(["cohort", "code"]).sum()),
        "duplicate_rows_identical": bool(frame.loc[duplicate_mask].duplicated().any()),
        "cohort_counts": {str(k): int(v) for k, v in frame.cohort.value_counts().items()},
        "wave2_flag_counts_not_semantically_confirmed": {str(k): int(v) for k, v in frame.wave2_followup.value_counts().items()},
        "wave1_wave2_paired_field_names_count": len(paired),
        "date2_observed_range_not_assigned_to_wave": [date_values.min().isoformat(), date_values.max().isoformat()],
        "date2_parse_failures": int(date_values.isna().sum()), "followup_interval_days": None,
        "workbook_comments_found": comments, "workbook_properties": properties,
        "column_names": [str(name) for name in frame.columns],
        "coded_field_examples_no_scale_name_inferred": {
            name: {"missing": int(frame[name].isna().sum()), "observed_numeric_values": sorted(float(value) for value in frame[name].dropna().unique())}
            for name in candidate
        },
        "dictionary_primary_evidence_found": False,
        "training_permitted": not blockers, "training_blockers": blockers,
        "unresolved": ["questionnaire_item_meanings", "scoring_and_reverse_items", "missing_value_codes", "wave_timing_and_interval", "participant_and_cohort_key_semantics"],
        "scope": "read-only audit; no model trained; no PHQ/GAD or clinical threshold inferred from column counts",
    }
    (output_dir / "longitudinal_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not definition_path:
        (output_dir / "longitudinal_task_pending.json").write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    """命令行审计入口；请求训练时若前提未核实，非零退出且不写模型。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task-definition", type=Path)
    parser.add_argument("--require-trainable", action="store_true")
    args = parser.parse_args()
    report = audit_workbook(args.source, args.output_dir, args.task_definition)
    print(json.dumps({key: report[key] for key in ("rows", "columns", "training_permitted", "training_blockers")}, ensure_ascii=False, indent=2))
    if args.require_trainable and not report["training_permitted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
