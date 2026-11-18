"""未定义的两波预测任务必须被阻止，后测变量不能伪装成基线特征。"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


def module() -> ModuleType:
    """加载接入审核器，不读真实参与者或触发模型训练。"""
    path = Path(__file__).resolve().parents[2] / "scripts" / "round3_data_longitudinal.py"
    spec = importlib.util.spec_from_file_location("longitudinal_contract_review", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def frame() -> pd.DataFrame:
    """构造仅用于契约测试的小表，变量没有心理学含义。"""
    return pd.DataFrame({"cohort": [1, 1], "code": [101, 102], "item_1": [1, 2], "item_2": [2, 3], "date2": ["2024-01-01"] * 2, "wave2_followup": [1, 1]})


def verified_fixture() -> dict[str, object]:
    """自编的完整契约仅验证软件边界，不声称真实数据词典已核实。"""
    result = module().unresolved_definition()
    result.update(participant_identity_verified=True, missing_codes_verified=True, scoring_verified=True,
                  baseline_features=["item_1"], target_column="item_2", target_meaning="fixture coding", target_unit="fixture unit",
                  missing_codes={"item_1": [], "item_2": []})
    result["codebook"] = {"verified": True, "primary_source_url": "https://example.org/fixture",
                          "column_definitions": {"item_1": {"wave": 1, "available_at_baseline": True, "meaning": "fixture item", "source_locator": "fixture p1"},
                                                 "item_2": {"wave": 2, "meaning": "fixture item", "source_locator": "fixture p2"}}}
    result["followup_interval"] = {"verified": True, "minimum_days": 30, "maximum_days": 31, "primary_source_url": "https://example.org/fixture"}
    return result


def test_current_unknown_task_cannot_be_trained() -> None:
    checker = module()
    failures = checker.training_blockers(frame(), checker.unresolved_definition(), checker.EXPECTED_SHA256)
    assert {"primary_codebook_unverified", "followup_interval_unverified", "target_scoring_unverified", "undefined_followup_target"} <= set(failures)


def test_verified_fixture_checks_can_pass_without_using_real_unknown_columns() -> None:
    checker = module()
    assert checker.training_blockers(frame(), verified_fixture(), checker.EXPECTED_SHA256) == []


@pytest.mark.parametrize("feature", ["item_2", "date2", "wave2_followup", "cohort", "code"])
def test_future_and_identity_columns_are_rejected_even_if_dictionary_claims_baseline(feature: str) -> None:
    checker = module()
    task = verified_fixture()
    task["baseline_features"] = [feature]
    task["codebook"]["column_definitions"][feature] = {"wave": 1, "available_at_baseline": True, "meaning": "fixture", "source_locator": "fixture"}
    task["missing_codes"][feature] = []
    assert checker.training_blockers(frame(), task, checker.EXPECTED_SHA256)


def test_matching_suffix_is_not_enough_without_question_meaning() -> None:
    checker = module()
    task = verified_fixture()
    task["codebook"]["column_definitions"]["item_1"].pop("meaning")
    assert "feature_temporality_or_meaning_unverified:item_1" in checker.training_blockers(frame(), task, checker.EXPECTED_SHA256)


def test_changed_source_and_unapproved_duplicate_policy_are_blocked() -> None:
    checker = module()
    task = verified_fixture()
    task["duplicate_policy"] = "keep_first"
    failures = checker.training_blockers(frame(), task, "changed-sha")
    assert "source_digest_mismatch" in failures and "ambiguous_duplicate_policy" in failures


@pytest.mark.parametrize("flag", ["participant_identity_verified", "missing_codes_verified", "scoring_verified"])
def test_truthy_strings_do_not_bypass_verification(flag: str) -> None:
    checker = module()
    task = verified_fixture()
    task[flag] = "true"
    assert checker.training_blockers(frame(), task, checker.EXPECTED_SHA256)
