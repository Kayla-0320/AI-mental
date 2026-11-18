"""校准适用性和数值行为：不加载模型，不把八分类校准泛化为临床效度。"""
import hashlib
import json

import numpy as np
import pytest

from perception.cognitive.calibration import calibrated_softmax, load_temperature


@pytest.fixture
def artifact(tmp_path):
    weights = tmp_path / "weights"
    weights.write_bytes(b"test checkpoint")
    config = {"schema_version": "cognitive_temperature.v1", "task": "c2d2_8class",
              "fit_split": "valid", "class_order": list(range(8)), "temperature": 1.5,
              "n_calibration": 1075, "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, weights, config


def test_matching_artifact_and_weight_activates_calibration(artifact):
    path, weights, _ = artifact
    result = load_temperature(path, weights, list(range(8)))
    assert result["status"] == "validation_temperature" and result["temperature"] == 1.5


@pytest.mark.parametrize("change", [
    {"fit_split": "test"}, {"task": "emotion"}, {"class_order": list(reversed(range(8)))},
    {"temperature": -1}, {"temperature": True}, {"temperature": float("nan")},
    {"n_calibration": 0}, {"checkpoint_sha256": "other"},
])
def test_mismatched_or_invalid_artifact_never_silently_calibrates(artifact, change):
    path, weights, config = artifact
    path.write_text(json.dumps({**config, **change}), encoding="utf-8")
    result = load_temperature(path, weights, list(range(8)))
    assert result["status"] == "uncalibrated" and result["temperature"] == 1.0


def test_missing_or_corrupt_artifact_has_explicit_status(artifact):
    path, weights, _ = artifact
    path.write_text("broken", encoding="utf-8")
    assert load_temperature(path, weights, list(range(8)))["reason"] == "invalid_artifact"
    path.unlink()
    assert load_temperature(path, weights, list(range(8)))["reason"] == "artifact_missing"


def test_temperature_preserves_ranking_and_softens_overconfidence():
    values = np.array([10000, 9999, 9998, 9997, 9996, 9995, 9994, 9993])
    raw, calibrated = calibrated_softmax(values), calibrated_softmax(values, 1.5)
    assert np.isfinite(calibrated).all() and calibrated.sum() == pytest.approx(1)
    assert np.array_equal(np.argsort(raw), np.argsort(calibrated))
    assert calibrated.max() < raw.max()


@pytest.mark.parametrize("values,temperature", [(np.zeros(5), 1), (np.full(8, np.nan), 1), (np.zeros(8), 0)])
def test_invalid_logits_fail_instead_of_fabricating_confidence(values, temperature):
    with pytest.raises(ValueError):
        calibrated_softmax(values, temperature)
