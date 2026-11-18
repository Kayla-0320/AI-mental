"""独立评分和成人使用先导的采集、校验与分析；空表绝不生成用户效果。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score

from experiments.round14_common import OUT, SEED, freeze, save

RATING_FIELDS = ["fidelity_1_5", "appropriateness_1_5", "correction_violation", "goal_violation", "unsupported_details", "repeat_question"]
SCHEMAS = {
    "consent": ["participant_id", "consented_at", "adult_confirmed", "consent_version", "withdrawn_at", "source"],
    "diary": ["record_id", "participant_id", "observed_at", "submitted_at", "mood_0_100", "stress_0_100", "context_optional", "source"],
    "strategy_events": ["event_id", "participant_id", "offered_at", "arm", "accepted_at", "started_at", "completed_at", "feedback_at", "helpfulness_1_5", "burden_1_5", "declined", "source"],
    "relationship_feedback": ["record_id", "participant_id", "observed_at", "rqi_score", "felt_understood_1_5", "choice_respected_1_5", "trust_1_5", "unpressured_1_5", "source"],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """读取带 UTF-8 BOM 的表格，不将空单元格当作零分。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def timestamp(value: str) -> datetime:
    """拒绝无时区或不合法时间，内部统一 UTC 比较。"""
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None: raise ValueError("时间必须带时区")
    return result.astimezone(timezone.utc)


def number(value: str, low: float, high: float, integer: bool = False) -> float:
    """拒绝缺失、非有限数、越界及离散评分中的小数。"""
    result = float(value)
    if not np.isfinite(result) or not low <= result <= high or (integer and not result.is_integer()):
        raise ValueError("评分或数值越界")
    return result


def init_study(output: Path = OUT / "human_study") -> None:
    """生成空白采集模板与预先平衡顺序，分配槽位不是已经招募的参与者。"""
    output.mkdir(parents=True, exist_ok=True)
    freeze(output / "protocol.json", {"schema": "adult_usability_pilot.v1", "seed": SEED,
        "planned_adult_slots": 24, "planned_days": 28, "actual_enrolled_at_creation": 0,
        "sample_size_is_power_calculated": False,
        "arm_definitions": {"A": "original", "B": "correction only", "C": "goal only", "D": "correction + goal"},
        "counterbalanced_order": ["ABDC", "BCAD", "CDBA", "DACB"],
        "scope": "adult voluntary usability and longitudinal feasibility; not treatment effectiveness",
        "daily_measures": "self-reported mood/stress, one daily opportunity, optional context; missed responses stay missing",
        "relational_items": "four authored usability items; not a validated trust/alliance scale",
        "causal_claims": "No causal benefit claim from accepted/completed users or before-after change; no MRT performed",
        "contact_recruitment_and_platform_integration": "not performed; no messages sent; offline preparation only",
        "analysis_excludes": "synthetic rows, test accounts, no adult consent, future dates, withdrawal and out-of-window records",
        "storage": "local research output; researcher must provide access control, retention and withdrawal deletion procedures before real collection"})
    for name, fields in SCHEMAS.items():
        path = output / f"{name}.csv"
        if not path.exists():
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                csv.DictWriter(stream, fieldnames=fields).writeheader()
    orders = np.random.default_rng(SEED).permutation(["ABDC", "BCAD", "CDBA", "DACB"] * 6)
    save(output / "allocation_slots.json", [{"slot_id": f"P{i + 1:03d}", "order": str(order), "enrolled": False} for i, order in enumerate(orders)])
    save(output / "status.json", {"actual_enrolled": 0, "completed_days": 0, "independent_raters": 0,
        "status": "prepared_awaiting_real_consent_and_observations", "efficacy_demonstrated": False})


def analyze_ratings(directory: Path, key_path: Path, output: Path) -> dict[str, Any]:
    """仅对真实独立评审填写的完整行统计，保持重复测量的场景聚类。"""
    key = {row["display_id"]: row for row in json.loads(key_path.read_text(encoding="utf-8"))}
    declarations_path = directory / "rater_declarations.json"
    declarations = json.loads(declarations_path.read_text(encoding="utf-8")) if declarations_path.exists() else []
    declarations = {row["rater_id"]: row for row in declarations}
    rows = []; seen = set(); incomplete = 0
    for path in sorted(directory.glob("rater_*.csv")):
        for row in read_csv(path):
            if row.get("display_id") not in key: raise ValueError("评分出现未知展示ID")
            fields = [row.get(name, "").strip() for name in RATING_FIELDS]
            if not any(fields): incomplete += 1; continue
            if not all(fields): raise ValueError("已填写的评分必须完成六个必填指标")
            identity = (row["rater_id"], row["display_id"])
            if identity in seen: raise ValueError("同一评审重复评分")
            seen.add(identity)
            declaration = declarations.get(row["rater_id"], {})
            if declaration.get("human_completed") is not True or declaration.get("independent_of_development") is not True:
                raise ValueError("评分缺少本人独立人评声明；不能把模型评分当独立用户评价")
            timestamp(declaration["declared_at"])
            scores = {name: number(row[name], 1 if name.endswith("1_5") else 0, 5 if name.endswith("1_5") else 1, True) for name in RATING_FIELDS}
            rows.append({**key[row["display_id"]], "rater_id": row["rater_id"], **scores})
    raters = sorted({row["rater_id"] for row in rows})
    result: dict[str, Any] = {"completed_rows": len(rows), "unfilled_rows": incomplete, "raters_with_completed_rows": len(raters),
        "human_independence": "self-declared, researcher must verify; cannot be established by software alone",
        "status": "awaiting_two_independent_raters", "clinical_effectiveness": False}
    if len(raters) >= 2:
        paired = {rater: {row["display_id"]: row for row in rows if row["rater_id"] == rater} for rater in raters}
        shared = sorted(set(paired[raters[0]]) & set(paired[raters[1]]))
        agreement = {}
        for metric in RATING_FIELDS:
            first = [paired[raters[0]][identifier][metric] for identifier in shared]
            second = [paired[raters[1]][identifier][metric] for identifier in shared]
            varying = len(set(first + second)) > 1
            agreement[metric] = {"paired_replies": len(shared), "exact_agreement": float(np.mean(np.array(first) == second)) if shared else None,
                "weighted_kappa": float(cohen_kappa_score(first, second, weights="quadratic", labels=[1, 2, 3, 4, 5])) if shared and varying and metric.endswith("1_5") else None}
        arms = sorted({item["arm"] for item in key.values()})
        arm_summary = {arm: {"ratings": len(selected), **{metric: float(np.mean([row[metric] for row in selected])) for metric in RATING_FIELDS}}
            for arm in arms if (selected := [row for row in rows if row["arm"] == arm])}
        # 只有同一场景四臂都有两名评审时才进入配对臂比较。
        complete_cases = []
        for case in sorted({row["case_id"] for row in rows}):
            if all(len({row["rater_id"] for row in rows if row["case_id"] == case and row["arm"] == arm}) >= 2 for arm in arms):
                complete_cases.append(case)
        differences = {}
        reference, candidate = ("A", "D") if "A" in arms and "D" in arms else (arms[0], arms[-1])
        for metric in RATING_FIELDS:
            changes_by_family: dict[str, list[float]] = {}
            for case in complete_cases:
                means = {arm: np.mean([row[metric] for row in rows if row["case_id"] == case and row["arm"] == arm]) for arm in (reference, candidate)}
                changes_by_family.setdefault(case.split("_")[0], []).append(float(means[candidate] - means[reference]))
            if changes_by_family:
                values = np.array([np.mean(changes) for changes in changes_by_family.values()]); rng = np.random.default_rng(SEED)
                sampled = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
                differences[metric] = {"candidate_minus_reference": float(values.mean()), "topic_family_cluster_ci95": np.quantile(sampled, [.025, .975]).tolist()}
        result.update(status="complete" if len(complete_cases) == len({item["case_id"] for item in key.values()}) else "partial_ratings",
            agreement=agreement, by_arm=arm_summary, fully_paired_cases=len(complete_cases), reference=reference, candidate=candidate, descriptive_differences=differences,
            multiple_comparisons="exploratory unadjusted intervals; no confirmatory efficacy inference")
    save(output, result)
    return result


def analyze_study(directory: Path, output: Path, now: datetime | None = None) -> dict[str, Any]:
    """先验证来源、时间顺序和同意，再报告真实参与和可用性，不推断因果疗效。"""
    now = now or datetime.now(timezone.utc)
    people = {}; withdrawn = set()
    for row in read_csv(directory / "consent.csv"):
        if row["source"] != "real": raise ValueError("真实先导分析拒绝模拟/测试同意记录")
        pid = row["participant_id"]
        if not pid or pid in people: raise ValueError("空参与者或重复同意记录")
        if row["adult_confirmed"] != "1" or not row["consent_version"]: raise ValueError("成人先导缺少显式成人同意")
        consented = timestamp(row["consented_at"])
        if consented > now: raise ValueError("同意时间在未来")
        people[pid] = consented
        if row["withdrawn_at"]:
            if not consented <= timestamp(row["withdrawn_at"]) <= now: raise ValueError("撤回时间无效")
            withdrawn.add(pid)
    active = set(people) - withdrawn

    def permitted(row: dict[str, str], time_field: str) -> bool:
        """拒绝未经同意来源；撤回者整体排除，不能只删除其负面反馈。"""
        if row["source"] != "real": raise ValueError("真实先导分析拒绝合成或fixture行")
        pid = row["participant_id"]
        if pid not in people: raise ValueError("记录没有对应同意")
        observed = timestamp(row[time_field])
        if not people[pid] <= observed <= now: raise ValueError("记录在同意之前或未来")
        if observed >= people[pid] + timedelta(days=28): raise ValueError("记录超出预注册28天窗口")
        return pid in active

    diary = []; seen = set()
    for row in read_csv(directory / "diary.csv"):
        if not row["record_id"] or row["record_id"] in seen: raise ValueError("日记ID为空或重复")
        seen.add(row["record_id"])
        if not permitted(row, "observed_at"): continue
        observed, submitted = timestamp(row["observed_at"]), timestamp(row["submitted_at"])
        if not observed <= submitted <= now: raise ValueError("日记提交顺序无效")
        diary.append({**row, "mood": number(row["mood_0_100"], 0, 100), "stress": number(row["stress_0_100"], 0, 100),
            "late_entry": submitted - observed > timedelta(hours=24)})
    events = []; seen = set()
    for row in read_csv(directory / "strategy_events.csv"):
        if not row["event_id"] or row["event_id"] in seen: raise ValueError("支持事件ID为空或重复")
        seen.add(row["event_id"])
        if not permitted(row, "offered_at"): continue
        if row["arm"] not in {"A", "B", "C", "D"} or row["declined"] not in {"0", "1"}: raise ValueError("支持臂或拒绝状态无效")
        previous = timestamp(row["offered_at"])
        for field in ("accepted_at", "started_at", "completed_at", "feedback_at"):
            if row[field]:
                current = timestamp(row[field])
                if not previous <= current <= now: raise ValueError("支持事件时间顺序无效")
                previous = current
        if row["declined"] == "1" and row["accepted_at"]: raise ValueError("拒绝与接受状态冲突")
        if row["started_at"] and not row["accepted_at"] or row["completed_at"] and not row["started_at"]: raise ValueError("支持流程缺少前置事件")
        if row["helpfulness_1_5"] or row["burden_1_5"]:
            if not row["feedback_at"]: raise ValueError("评分没有本人反馈时间")
            number(row["helpfulness_1_5"], 1, 5, True); number(row["burden_1_5"], 1, 5, True)
        events.append(row)
    relationship = []; seen = set()
    for row in read_csv(directory / "relationship_feedback.csv"):
        if not row["record_id"] or row["record_id"] in seen: raise ValueError("关系反馈ID为空或重复")
        seen.add(row["record_id"])
        if not permitted(row, "observed_at"): continue
        scores = {field: number(row[field], 1, 5, True) for field in ("felt_understood_1_5", "choice_respected_1_5", "trust_1_5", "unpressured_1_5")}
        relationship.append({**row, "rqi": number(row["rqi_score"], 0, 1), **scores})
    accepted = [row for row in events if row["accepted_at"]]
    helpful = [float(row["helpfulness_1_5"]) for row in events if row["helpfulness_1_5"]]
    burden = [float(row["burden_1_5"]) for row in events if row["burden_1_5"]]
    expected_windows = {pid: min(28, max(0, (now - people[pid]).days)) for pid in active}
    completed_windows = {pid: len({int((timestamp(row["observed_at"]) - people[pid]).total_seconds() // 86400)
        for row in diary if row["participant_id"] == pid and
        int((timestamp(row["observed_at"]) - people[pid]).total_seconds() // 86400) < expected_windows[pid]}) for pid in active}
    # 不把尚未过去的28天当作漏填；当前未结束的24小时窗口单独留待以后统计。
    expected_total = sum(expected_windows.values()); completed_total = sum(completed_windows.values())
    rqi_people = sorted({row["participant_id"] for row in relationship})
    rqi_means = {pid: {metric: float(np.mean([row[metric] for row in relationship if row["participant_id"] == pid]))
        for metric in ("rqi", "felt_understood_1_5", "choice_respected_1_5", "trust_1_5", "unpressured_1_5")} for pid in rqi_people}
    correlations = {}
    for metric in ("felt_understood_1_5", "choice_respected_1_5", "trust_1_5", "unpressured_1_5"):
        first = np.array([rqi_means[pid]["rqi"] for pid in rqi_people]); second = np.array([rqi_means[pid][metric] for pid in rqi_people])
        if len(first) >= 3 and np.ptp(first) > 0 and np.ptp(second) > 0:
            from scipy.stats import spearmanr
            value = float(spearmanr(first, second).statistic)
        else: value = None
        correlations[metric] = {"participant_mean_spearman": value, "people": len(rqi_people), "descriptive_not_construct_validation": True}
    result = {"status": "awaiting_real_observations" if not diary and not events else "descriptive_real_pilot_only",
        "consented_adults": len(people), "withdrawn_excluded": len(withdrawn), "active_adults": len(active),
        "diary_rows": len(diary), "late_diary_rows": sum(row["late_entry"] for row in diary),
        "fully_elapsed_24h_study_windows": expected_total, "windows_with_diary": completed_total,
        "diary_completion_rate_in_elapsed_windows": completed_total / expected_total if expected_total else None,
        "not_yet_elapsed_windows_counted_as_missing": False,
        "completed_28day_followup_windows_people": sum(windows == 28 for windows in expected_windows.values()),
        "observed_days_per_person": {pid: len({timestamp(row["observed_at"]).astimezone(timezone(timedelta(hours=8))).date() for row in diary if row["participant_id"] == pid}) for pid in sorted(active)},
        "offered": len(events), "accepted": len(accepted), "acceptance_rate": len(accepted) / len(events) if events else None,
        "completed": sum(bool(row["completed_at"]) for row in accepted),
        "completion_rate_among_accepted": sum(bool(row["completed_at"]) for row in accepted) / len(accepted) if accepted else None,
        "feedback_count": len(helpful), "mean_helpfulness": float(np.mean(helpful)) if helpful else None,
        "feedback_rate_among_offered": len(helpful) / len(events) if events else None,
        "mean_burden": float(np.mean(burden)) if burden else None,
        "relationship_feedback_rows": len(relationship), "rqi_descriptive_external_feedback": correlations, "rqi_construct_validated": False,
        "missing_feedback_is_not_positive_feedback": True, "causal_effect_estimated": False,
        "clinical_effectiveness_evidence": False, "note": "helpfulness conditional on submitted feedback; self-selection remains; withdrawal requires separate raw-data deletion by researcher"}
    save(output, result)
    return result


def main() -> None:
    """提供空模板初始化、双人评分分析和真实成人先导校验三个命令。"""
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=["init", "ratings", "study"])
    parser.add_argument("--directory", type=Path); parser.add_argument("--output", type=Path)
    parser.add_argument("--key", type=Path, default=OUT / "support" / "private" / "arm_key.json")
    args = parser.parse_args()
    if args.command == "init":
        init_study(args.directory or OUT / "human_study")
        result = analyze_study(args.directory or OUT / "human_study", args.output or OUT / "human_study" / "analysis.json")
    elif args.command == "ratings":
        result = analyze_ratings(args.directory or OUT / "support" / "blind", args.key,
            args.output or OUT / "support" / "independent_ratings_analysis.json")
    else:
        result = analyze_study(args.directory or OUT / "human_study", args.output or OUT / "human_study" / "analysis.json")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
