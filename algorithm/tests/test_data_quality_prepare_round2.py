"""特征预处理必须排除padding、显式标记缺失，并拒绝未知pickle。"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def load_preparer() -> ModuleType:
    """直接加载脚本函数，不触发命令行或下载。"""
    path = Path(__file__).resolve().parents[2] / "scripts" / "round2_data_prepare.py"
    spec = importlib.util.spec_from_file_location("round2_data_prepare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_padding_never_contributes_to_pooling() -> None:
    values = np.array([[[1, 2], [3, 4], [1e9, 1e9]]], dtype=float)
    pooled, mask, quality, stats = load_preparer().pool_features(values, np.array([2]))
    np.testing.assert_array_equal(pooled, [[2, 3]])
    assert mask.tolist() == [True]
    assert quality.tolist() == [1]
    assert stats["usable"] == 1


def test_nonfinite_padding_does_not_invalidate_real_observation() -> None:
    values = np.array([[[1.0], [float("nan")]]])
    pooled, mask, _, _ = load_preparer().pool_features(values, np.array([1]))
    assert mask.tolist() == [True]
    np.testing.assert_array_equal(pooled, [[1]])


def test_bad_actual_feature_is_missing_rather_than_silently_imputed() -> None:
    values = np.array([[[float("inf")], [1.0]]])
    pooled, mask, quality, stats = load_preparer().pool_features(values, np.array([2]))
    assert mask.tolist() == [False]
    assert quality.tolist() == [0]
    assert stats["non_finite"] == 1
    np.testing.assert_array_equal(pooled, [[0]])


@pytest.mark.parametrize("length", [0, -1, 3, float("nan"), 1.5])
def test_invalid_or_empty_length_does_not_fabricate_observation(length: float) -> None:
    pooled, mask, quality, stats = load_preparer().pool_features(np.ones((1, 2, 3)), np.array([length]))
    assert mask.tolist() == [False]
    assert quality.tolist() == [0]
    assert stats["usable"] == 0
    assert np.isfinite(pooled).all()


def test_unknown_pickle_is_rejected_before_loading(tmp_path: Path) -> None:
    path = tmp_path / "unknown.pkl"
    path.write_bytes(b"untrusted data")
    with pytest.raises(ValueError, match="SHA|摘要"):
        load_preparer().prepare(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
