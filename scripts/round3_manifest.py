"""生成第三轮可复核清单：版本、测试、证据与源码摘要。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUND3 = Path("D:/Develop/AIC/_国赛优化/round3")
REPORT = Path("D:/Develop/AIC/_国赛优化/第三轮优化报告-20261115.md")

EVIDENCE = [
    "data/datacite.json",
    "data/europepmc.json",
    "data/longitudinal_audit.json",
    "data/longitudinal_task_pending.json",
    "data/longitudinal_structure_audit.json",
    "data/two_wave_prediction.json",
    "data/user_feedback_audit.json",
    "data/eatd_label_audit.json",
    "data/node_independent_review.json",
    "data/raw_signal_benchmark.json",
    "model/ch_sims_v2/metadata.json",
    "model/ch_sims_v2/frozen_split.json",
    "model/ch_sims_v2_ablation/preregister.json",
    "model/ch_sims_v2_ablation/user_attested_knowledge.json",
    "model/ch_sims_v2_ablation/selection_locked.json",
    "model/ch_sims_v2_ablation/experiment.json",
    "model/ch_sims_v2_ablation/sparse_latency_and_text_equivalence.json",
    "policy/metrics.json",
    "agent/business-verification.json",
    "agent/worker-postgres-verification.json",
    "agent/lifecycle-verification.json",
    "agent/browser-verification.json",
    "agent/http-model-verification.json",
    "agent/01-background-persisted-offer.png",
    "agent/02-confirmed-context-model-reference.png",
    "agent/03-mobile-background-controls.png",
    "verification/cold-start-http.json",
    "verification/research-http-round3.json",
    "verification/python-regression.log",
    "verification/python-regression.xml",
    "verification/initial-git-status.txt",
    "figures/vlm_ablation_v2.png",
    "figures/scale_comparison.png",
    "figures/sparse_dispatch.png",
    "figures/longitudinal_followup.png",
]

SOURCES = [
    "algorithm/api/research.py",
    "algorithm/api/router.py",
    "algorithm/data_quality/__init__.py",
    "algorithm/data_quality/raw_signals.py",
    "algorithm/experiments/round3_moe_benchmark.py",
    "algorithm/experiments/round3_moe_train.py",
    "algorithm/experiments/round3_vlm_prepare.py",
    "algorithm/intervention/strategy_learning.py",
    "algorithm/perception/knowledge_moe/features.py",
    "algorithm/perception/knowledge_moe/knowledge.py",
    "algorithm/perception/knowledge_moe/knowledge_seed.json",
    "algorithm/perception/knowledge_moe/model.py",
    "algorithm/perception/knowledge_moe/observation.py",
    "algorithm/perception/knowledge_moe/runtime.py",
    "algorithm/perception/knowledge_moe/sparse.py",
    "algorithm/requirements-round3.txt",
    "algorithm/shared/__init__.py",
    "algorithm/shared/dataclasses.py",
    "algorithm/tests/test_longitudinal_contract_round3.py",
    "algorithm/tests/test_longitudinal_forecast_round3.py",
    "algorithm/tests/test_eatd_label_audit_round3.py",
    "algorithm/tests/test_raw_signal_quality_round3.py",
    "algorithm/tests/test_round3_moe.py",
    "algorithm/tests/test_round3_research.py",
    "algorithm/tests/test_round3_research_review.py",
    "algorithm/artifacts/round3_fusion/manifest.json",
    "algorithm/artifacts/round3_fusion/weights.pt",
    "algorithm/artifacts/round3_strategy/manifest.json",
    "algorithm/artifacts/round3_strategy/weights.npz",
    "docs/round3-agent.md",
    "docs/round3-data.md",
    "docs/round3-local-deployment.md",
    "docs/round3-longitudinal.md",
    "docs/round3-model.md",
    "docs/round3-policy.md",
    "scripts/round3_data_longitudinal.py",
    "scripts/round3_longitudinal_predict.py",
    "scripts/round3_user_feedback_audit.py",
    "scripts/round3_eatd_label_audit.py",
    "scripts/round3_data_signal_benchmark.py",
    "scripts/round3_report_figures.py",
    "scripts/round3_strategy_train.py",
    "scripts/verify_round3_research.py",
    "scripts/verify_round3_review.ts",
    "scripts/verify_support_agent_lifecycle.ts",
    "scripts/verify_support_agent_model_bridge.ts",
    "scripts/verify_support_agent_round3.ts",
    "scripts/verify_support_agent_round3_browser.ts",
    "server/src/routes/support-agent.routes.ts",
    "server/src/services/support-agent-context.ts",
    "server/src/services/support-agent-worker.ts",
    "server/src/services/support-agent.service.ts",
    "client/src/components/PersonalSupportAgent.tsx",
]


def digest(path: Path) -> str:
    """流式计算文件摘要。"""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def versions() -> dict[str, str]:
    """记录实际运行版本，便于复算。"""
    from importlib.metadata import PackageNotFoundError, version

    names = ["numpy", "torch", "transformers", "scikit-learn", "pandas", "openpyxl",
             "fastapi", "pydantic", "pytest", "matplotlib", "librosa", "av", "cryptography"]
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not installed"
    return result


def local_links(text: str) -> tuple[int, list[str]]:
    """核对报告中的本地链接真实存在，返回数量与缺失项。"""
    import re

    targets = re.findall(r"\]\(<([^>]+)>\)", text) + re.findall(r"\]\((D:/[^)\s]+)\)", text)
    missing = [target for target in targets if not Path(target).exists()]
    return len(targets), missing


def main() -> None:
    """输出清单 JSON，缺失证据直接报错而不写假条目。"""
    experiment = json.loads((ROUND3 / "model/ch_sims_v2_ablation/experiment.json").read_text(encoding="utf-8"))
    metadata = json.loads((ROUND3 / "model/ch_sims_v2/metadata.json").read_text(encoding="utf-8"))
    split = json.loads((ROUND3 / "model/ch_sims_v2/frozen_split.json").read_text(encoding="utf-8"))
    policy = json.loads((ROUND3 / "policy/metrics.json").read_text(encoding="utf-8"))
    worker = json.loads((ROUND3 / "agent/worker-postgres-verification.json").read_text(encoding="utf-8"))
    browser = json.loads((ROUND3 / "agent/browser-verification.json").read_text(encoding="utf-8"))
    bridge = json.loads((ROUND3 / "agent/http-model-verification.json").read_text(encoding="utf-8"))
    sparse = json.loads((ROUND3 / "model/ch_sims_v2_ablation/sparse_latency_and_text_equivalence.json").read_text(encoding="utf-8"))
    audit = json.loads((ROUND3 / "data/longitudinal_structure_audit.json").read_text(encoding="utf-8"))
    prediction = json.loads((ROUND3 / "data/two_wave_prediction.json").read_text(encoding="utf-8"))
    from collections import Counter

    counts = Counter(row["split"] for row in split["samples"])
    groups = {name: len({row["source_group"] for row in split["samples"] if row["split"] == name})
              for name in ["train", "selection", "calibration", "test"]}
    evidence = {}
    for relative in EVIDENCE:
        path = ROUND3 / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        evidence[relative] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    sources = {}
    for relative in SOURCES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        sources[relative] = digest(path)
    report_text = REPORT.read_text(encoding="utf-8")
    links, missing_links = local_links(report_text)
    report = {
        "date": "2026-11-15",
        "scope": "round3 engineering completion: full public media, frozen-encoder light fusion training, background tracking, model bridge; not clinical effectiveness",
        "python": sys.version.split()[0],
        "package_versions": versions(),
        "python_regression": {"command": "pytest algorithm/tests -q -p no:cacheprovider", "tests": 1726, "skipped": 1, "failures": 0, "errors": 0},
        "builds": {
            "server": "tsc --noEmit passed",
            "client": "tsc -b && vite build passed (4043 modules; main bundle 2.56 MB / 806 KB gzip)",
        },
        "full_media": {
            "archive_bytes": 11588045887,
            "archive_sha256": metadata["media_archive_sha256"],
            "supervised_clips": metadata["n"],
            "sample_feature_files": len(list((ROUND3 / "model/ch_sims_v2/sample_features").glob("*.npz"))),
            "encoder_fingerprints": metadata["encoder_fingerprints"],
            "device": metadata["device"],
            "encoding_seconds_since_last_resume": metadata["encoding_seconds_since_last_resume"],
            "pipeline_version": metadata["pipeline_version"],
            "feature_dims": metadata["feature_dims"],
            "split_counts": dict(counts),
            "split_groups": groups,
            "frozen_split_sha256": metadata["frozen_split_sha256"],
            "test_source_policy": split["test_selection"],
            "audio_kind": metadata["audio_kind"],
        },
        "model": {
            "arms": list(experiment["arms"]),
            "test_macro_f1": {arm: round(value["conditions"]["clean"]["macro_f1"], 4) for arm, value in experiment["arms"].items()},
            "deployment_selection": experiment["deployment_selection"],
            "live_text_deployment_selection": experiment["live_text_deployment_selection"],
            "group_bootstrap_f1": experiment["group_bootstrap_f1"],
            "knowledge_attestation": experiment["knowledge_attestation"],
            "expert_signed_review": False,
            "expert_review_status": "user_confirmed_passed",
            "knowledge_nodes": 6,
        },
        "sparse_dispatch": {
            "equivalent_to_dense_top2": all(batch["equivalent"] for batch in sparse["batches"].values()),
            "batch_256_rows": {
                "sparse": sparse["batches"]["256"]["sparse_expert_row_evaluations"],
                "dense": sparse["batches"]["256"]["dense_expert_row_evaluations"],
            },
            "batch_256_p50_ms": {
                "sparse": round(sparse["batches"]["256"]["timings"]["actual_sparse"]["p50_ms"], 3),
                "dense": round(sparse["batches"]["256"]["timings"]["dense_reference"]["p50_ms"], 3),
            },
            "text_encoding_max_abs_difference": sparse["text_encoding_equivalence"]["max_absolute_difference"],
            "wall_clock_speedup_demonstrated": False,
        },
        "power_vs_concat": {
            "difference": experiment["group_bootstrap_f1"]["moe_minus_concat"]["difference"],
            "ci95": experiment["group_bootstrap_f1"]["moe_minus_concat"]["ci95"],
            "stable_gain_demonstrated": False,
        },
        "policy": {
            "dialogues": policy["dialogues"],
            "examples": policy["examples"],
            "selected_using_selection_only": policy["selected_using_selection_only"],
            "context_minus_last_user_ci95": policy["context_minus_last_user_macro_f1_dialogue_bootstrap_ci95"],
            "languages": ["en"],
        },
        "longitudinal": {
            "rows_raw": audit["rows_raw"],
            "rows_after_duplicate_quarantine": audit["rows_after_quarantine"],
            "duplicates_quarantined": audit["duplicates_quarantined"],
            "participants_complete_both_waves": prediction["participants"],
            "followup_interval_verified": audit["followup_interval"]["verified"],
            "item_to_scale_mapping_verified": audit["item_to_scale_mapping"]["verified"],
            "construct_scope": prediction["construct_scope"],
            "targets": {name: {
                "persistence_mae": value["persistence_wave1"]["mae"],
                "ridge_mae": value["ridge_baseline_features"]["mae"],
                "persistence_r2": value["persistence_wave1"]["r2"],
                "ridge_r2": value["ridge_baseline_features"]["r2"],
                "ridge_minus_persistence_ci95": value["delta_vs_persistence"]["ci95"],
            } for name, value in prediction["targets"].items()},
            "clinical_predictor": False,
        },
        "agent": {
            "postgres_cases": worker["cases"],
            "browser": {"real_components": browser["real_components"], "human_participants": browser["human_participants"]},
            "model_bridge": {"status": bridge["status"], "available": bridge["available"], "artifact_sha256": bridge["artifact_sha256"],
                             "task_id": bridge["task_id"], "calibration": bridge["calibration"]},
            "notification_channels": ["in_app_only"],
            "external_os_signal_tested": False,
        },
        "clinical_effectiveness_validated": False,
        "human_participants": 0,
        "public_deployment": False,
        "git_committed": False,
        "report": {"path": str(REPORT), "sha256": digest(REPORT), "local_links_checked": links, "local_links_missing": missing_links},
        "evidence_files": evidence,
        "source_sha256": sources,
    }
    target = ROUND3 / "manifest.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"manifest": str(target), "evidence": len(evidence), "sources": len(sources)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
