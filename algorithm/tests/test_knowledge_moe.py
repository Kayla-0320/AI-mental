"""检验缺失、路由、知识审核边界与特征来源，不以合成样例充当性能证据。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from perception.knowledge_moe.features import FeatureStandardizer, save_json, sha256_file
from perception.knowledge_moe.knowledge import KnowledgeIndex
from perception.knowledge_moe.model import ConcatenationHead, QualityAwareMoE
from perception.knowledge_moe.runtime import FusionRuntime
from shared import MultimodalFeatureBatch


def data() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    return {"text": torch.randn(6, 4), "visual": torch.randn(6, 3), "audio": torch.randn(6, 2)}, torch.ones(6, 3, dtype=torch.bool), torch.ones(6, 3)


def test_missing_experts_are_exactly_masked_and_masked_values_cannot_change_prediction() -> None:
    x, mask, q = data()
    model = QualityAwareMoE({"text": 4, "visual": 3, "audio": 2}, 3).eval()
    mask[:, 1] = False
    logits, gates, _ = model(x, mask, q)
    missing_names = [i for i, name in enumerate(model.expert_names) if "visual" in name]
    assert torch.count_nonzero(gates[:, missing_names]) == 0
    assert torch.allclose(gates.sum(dim=1), torch.ones(6))
    x["visual"][:] = float("nan")
    after, _, _ = model(x, mask, q)
    assert torch.equal(logits, after)


def test_router_and_real_experts_receive_gradients() -> None:
    x, mask, q = data()
    model = QualityAwareMoE({"text": 4, "visual": 3, "audio": 2}, 3)
    logits, _, regularizer = model(x, mask, q)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 0, 1, 2])) + regularizer * .01
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.grad.abs().sum().item() for p in model.router.parameters()) > 0
    assert sum(p.grad.abs().sum().item() for p in model.experts.parameters()) > 0


def test_all_missing_refuses_prediction_instead_of_neutral_imputation() -> None:
    x, mask, q = data()
    mask[2] = False
    for model in [QualityAwareMoE({"text": 4, "visual": 3, "audio": 2}, 3),
                  ConcatenationHead({"text": 4, "visual": 3, "audio": 2}, 3)]:
        with pytest.raises(ValueError, match="缺失|有效"):
            model(x, mask, q)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_quality_rejected(bad: float) -> None:
    x, mask, q = data()
    q[0, 1] = bad
    for cls in [QualityAwareMoE, ConcatenationHead]:
        with pytest.raises(ValueError):
            cls({"text": 4, "visual": 3, "audio": 2}, 3)(x, mask, q)


def test_baselines_reject_invalid_mask_and_observed_nonfinite_values() -> None:
    x, mask, q = data()
    for cls in [QualityAwareMoE, ConcatenationHead]:
        with pytest.raises(ValueError):
            cls({"text": 4, "visual": 3, "audio": 2}, 3)(x, mask.float(), q)
    x["text"][0, 0] = float("nan")
    for cls in [QualityAwareMoE, ConcatenationHead]:
        with pytest.raises(ValueError, match="非有限"):
            cls({"text": 4, "visual": 3, "audio": 2}, 3)(x, mask, q)


def test_zero_quality_is_missing_and_top_k_is_bounded_by_availability() -> None:
    x, mask, q = data()
    q[:, 1:] = 0
    model = QualityAwareMoE({"text": 4, "visual": 3, "audio": 2}, 3, top_k=2).eval()
    logits, weights, regularizer = model(x, mask, q)
    assert torch.isfinite(logits).all() and torch.isfinite(regularizer)
    assert torch.all(weights[:, 0] == 1) and torch.count_nonzero(weights[:, 1:]) == 0


def test_top_k_training_refused_instead_of_silent_zero_routing_gradient() -> None:
    x, mask, q = data()
    model = QualityAwareMoE({"text": 4, "visual": 3, "audio": 2}, 3, top_k=1)
    with pytest.raises(ValueError, match="仅用于推理"):
        model(x, mask, q)
    model.eval()
    _, weights, _ = model(x, mask, q)
    assert torch.all(torch.count_nonzero(weights, dim=1) == 1)


def test_knowledge_missing_mask_zeroes_expert_and_default_index_excludes_candidates() -> None:
    x, mask, q = data()
    model = QualityAwareMoE({"text": 4, "visual": 3, "audio": 2}, 3, knowledge_dim=8)
    _, weights, _ = model(x, mask, q, torch.zeros(6, 8), torch.zeros(6, dtype=torch.bool))
    assert torch.count_nonzero(weights[:, -1]) == 0
    path = Path(__file__).resolve().parents[1] / "perception/knowledge_moe/knowledge_seed.json"
    assert KnowledgeIndex(path).retrieve("我很开心，也有些担心") == []
    offline = KnowledgeIndex(path, allow_candidates=True)
    hits = offline.retrieve("我很开心，也有些担心")
    assert hits and all(x["review_status"] == "candidate" and x["source_url"] for x in hits)
    values, present = offline.encode(["我很开心", "完全无关的描述"])
    assert present.tolist() == [True, False] and np.isfinite(values).all()
    assert np.array_equal(values, offline.encode(["我很开心", "完全无关的描述"])[0])


def test_expert_reviewed_claim_requires_actual_review_identity(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.json"
    save_json(path, {"schema_version": "emotion_knowledge.v1", "nodes": [
        {"id": "a", "source_url": "https://example.org", "aliases": ["a"], "review_status": "expert_reviewed"}
    ]})
    with pytest.raises(ValueError, match="审核人"):
        KnowledgeIndex(path)


def test_training_only_standardizer_ignores_missing_nan_and_keeps_future_values_unseen() -> None:
    scaler = FeatureStandardizer().fit({"text": np.array([[1., 2.], [3., 4.], [np.nan, np.nan]])},
                                      np.array([[True], [True], [False]]))
    assert scaler.parameters["text"]["mean"] == [2., 3.]
    values = scaler.transform({"text": np.array([[100., 102.], [np.nan, np.nan]])}, np.array([[True], [False]]))
    assert np.array_equal(values["text"][1], [0, 0])
    assert scaler.parameters["text"]["mean"] == [2., 3.]
    with pytest.raises(ValueError):
        scaler.transform({"text": np.array([[np.nan, 1.]])}, np.array([[True]]))


@pytest.mark.parametrize("scale", [[-1., 1.], [0., 1.], [float("nan"), 1.], [1.], [float("inf"), 1.]])
def test_standardizer_rejects_invalid_loaded_scale(scale: list[float]) -> None:
    scaler = FeatureStandardizer()
    scaler.parameters = {"text": {"mean": [0., 0.], "scale": scale}}
    with pytest.raises(ValueError, match="参数非法"):
        scaler.transform({"text": np.array([[1., 2.]])}, np.array([[True]]))


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    torch.manual_seed(42)
    model = QualityAwareMoE({"text": 4}, 3)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    save_json(tmp_path / "manifest.json", {
        "schema_version": "knowledge_moe.v1", "dimensions": {"text": 4}, "class_names": ["negative", "neutral", "positive"],
        "task": "test_sentiment", "hidden": 32, "architecture": "moe", "knowledge_dim": 0,
        "checkpoint_sha256": sha256_file(tmp_path / "weights.pt"), "encoder_fingerprints": {"text": "sha256:source"},
        "standardizer": {"text": {"mean": [0.] * 4, "scale": [1.] * 4}}, "temperature": 1.2,
        "calibration": {"status": "calibrated", "fit_split": "calibration", "n": 3}
    })
    return tmp_path


def batch() -> MultimodalFeatureBatch:
    return MultimodalFeatureBatch(sample_ids=["a", "b"], modalities={"text": [[1.] * 4, [float("nan")] * 4]},
                                  observed={"text": [True, False]}, quality={"text": [1., 0.]},
                                  task="test_sentiment", class_names=["negative", "neutral", "positive"],
                                  encoder_fingerprints={"text": "sha256:source"})


def test_runtime_abstains_only_missing_samples_and_reports_task(artifact: Path) -> None:
    result = FusionRuntime(artifact).predict(batch())
    assert result.abstained == [False, True] and result.probabilities[1] == []
    assert sum(result.probabilities[0]) == pytest.approx(1)
    assert result.calibration_status == "calibrated" and result.task == "test_sentiment"


@pytest.mark.parametrize("field,value", [("task", "clinical"), ("class_names", ["x", "y", "z"]),
                                        ("encoder_fingerprints", {"text": "other"})])
def test_runtime_rejects_semantic_or_encoder_mismatch(artifact: Path, field: str, value: object) -> None:
    b = batch()
    setattr(b, field, value)
    with pytest.raises(ValueError):
        FusionRuntime(artifact).predict(b)


def test_runtime_rejects_modified_weights_or_test_fitted_temperature(artifact: Path) -> None:
    meta_path = artifact / "manifest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["calibration"]["fit_split"] = "test"
    save_json(meta_path, meta)
    with pytest.raises(ValueError, match="独立校准"):
        FusionRuntime(artifact)
    meta["calibration"]["fit_split"] = "calibration"
    save_json(meta_path, meta)
    with (artifact / "weights.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="摘要"):
        FusionRuntime(artifact)


@pytest.mark.parametrize("value", [[], None, {"schema_version": "knowledge_moe.v1"}])
def test_nonobject_or_incomplete_manifest_fails_during_load_with_valueerror(artifact: Path, value: object) -> None:
    save_json(artifact / "manifest.json", value)
    with pytest.raises(ValueError):
        FusionRuntime(artifact)


@pytest.mark.parametrize("field,value", [
    ("class_names", ["negative", "negative"]),
    ("dimensions", {"text": True}),
    ("dimensions", {"text": 10 ** 80}),
    ("hidden", "32"),
    ("architecture", {}),
    ("encoder_fingerprints", []),
    ("standardizer", {"text": {"mean": [0.] * 3, "scale": [1.] * 4}}),
    ("standardizer", {"text": {"mean": [10 ** 400] * 4, "scale": [1.] * 4}}),
    ("standardizer", {"text": {"mean": [0.] * 4, "scale": [1e-100] * 4}}),
    ("temperature", True),
    ("temperature", 10 ** 400),
    ("calibration", []),
    ("calibration", {"status": "unknown"}),
    ("calibration", {"status": "uncalibrated"}),
    ("calibration", {"status": "calibrated", "fit_split": "calibration", "n": 0}),
])
def test_malformed_manifest_never_reaches_prediction_or_leaks_shape_exceptions(
        artifact: Path, field: str, value: object) -> None:
    path = artifact / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    save_json(path, manifest)
    with pytest.raises(ValueError):
        FusionRuntime(artifact)


def test_all_missing_nan_rows_are_valid_abstentions_and_do_not_require_feature_computation(artifact: Path) -> None:
    b = batch()
    b.modalities["text"] = [[float("nan")] * 4] * 2
    b.observed["text"] = [False, False]
    b.quality["text"] = [1., 1.]
    result = FusionRuntime(artifact).predict(b)
    assert result.abstained == [True, True] and result.probabilities == [[], []]
    b.observed["text"][0] = True
    with pytest.raises(ValueError, match="非有限"):
        FusionRuntime(artifact).predict(b)


def test_uncalibrated_status_requires_identity_temperature_and_is_reported_honestly(artifact: Path) -> None:
    path = artifact / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(temperature=1., calibration={"status": "uncalibrated"})
    save_json(path, manifest)
    assert FusionRuntime(artifact).predict(batch()).calibration_status == "uncalibrated"


def test_damaged_tensor_archive_with_matching_hash_still_fails_as_valueerror(artifact: Path) -> None:
    path = artifact / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    (artifact / "weights.pt").write_bytes(b"broken tensor archive")
    manifest["checkpoint_sha256"] = sha256_file(artifact / "weights.pt")
    save_json(path, manifest)
    with pytest.raises(ValueError, match="权重内容"):
        FusionRuntime(artifact)
