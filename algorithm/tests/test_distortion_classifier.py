"""认知扭曲 8 分类推理模块测试

不加载真实权重：只验证「检查点解析、容错降级、提示串格式」等非模型逻辑。
模型前向由 tools/train_cognitive_distortion.py 的训练/评测链路覆盖。
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from perception.cognitive.distortion_classifier import (  # noqa: E402
    NON_DISTORTED_ZH,
    DistortionClassifier,
    DistortionPrediction,
    format_distortion_hint,
    get_distortion_classifier,
    predict_distortion_safe,
    reset_distortion_classifier,
    resolve_ckpt_dir,
)
from perception.cognitive import distortion_classifier as dc  # noqa: E402

_LABEL_MAP = {
    "id2label": {
        "0": "非黑即白", "1": "情绪化推理", "2": "算命", "3": "乱贴标签",
        "4": "读心术", "5": "过度泛化", "6": "个人化归责", "7": "非扭曲",
    },
    "best_epoch": 3,
    "val_macro_f1": 0.6532,
    "encoder": "chinese-roberta-wwm-ext",
    "max_len": 160,
    "num_labels": 8,
}


@pytest.fixture()
def fake_ckpt(tmp_path):
    """构造一个只有元信息的最小检查点目录（不含权重）。"""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "label_map.json").write_text(
        json.dumps(_LABEL_MAP, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


class TestCkptParsing:
    """检查点元信息解析（不触发权重加载）"""

    def test_reads_label_map(self, fake_ckpt):
        clf = DistortionClassifier(fake_ckpt)
        assert clf.num_labels == 8
        assert clf.id2label[0] == "非黑即白"
        assert clf.id2label[7] == NON_DISTORTED_ZH

    def test_reads_training_meta(self, fake_ckpt):
        clf = DistortionClassifier(fake_ckpt)
        assert clf.meta["best_epoch"] == 3
        assert clf.meta["val_macro_f1"] == pytest.approx(0.6532)
        assert clf.meta["encoder"] == "chinese-roberta-wwm-ext"

    def test_max_len_from_label_map(self, fake_ckpt):
        assert DistortionClassifier(fake_ckpt)._max_len == 160

    def test_does_not_load_weights_on_init(self, fake_ckpt):
        """__init__ 必须保持惰性，不得拉起 torch/transformers。"""
        clf = DistortionClassifier(fake_ckpt)
        assert clf._model is None
        assert clf._tokenizer is None

    def test_missing_label_map_raises(self, tmp_path):
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(FileNotFoundError):
            DistortionClassifier(tmp_path)


class TestResolveCkptDir:
    """检查点目录解析"""

    def test_repo_artifacts_path_is_preferred(self):
        """仓库内 artifacts 必须排在开发机路径之前，保证随代码分发可用。

        若此顺序被改回"开发机路径优先"，换机器部署时会静默找不到检查点，
        所以用测试锁死。
        """
        first = dc._DEFAULT_CKPT_DIRS[0]
        assert first.parts[-2:] == ("artifacts", "cognitive_distortion_ckpt")
        assert first.is_absolute()

    @pytest.mark.skipif(
        not (dc._DEFAULT_CKPT_DIRS[0] / "label_map.json").exists(),
        reason="仓库内尚未导出产物（先跑 tools/export_model_artifacts.py）",
    )
    def test_resolves_to_repo_artifacts_when_present(self, monkeypatch):
        monkeypatch.delenv("DISTORTION_CKPT_DIR", raising=False)
        assert resolve_ckpt_dir() == dc._DEFAULT_CKPT_DIRS[0]

    def test_env_var_wins(self, fake_ckpt, monkeypatch):
        monkeypatch.setenv("DISTORTION_CKPT_DIR", str(fake_ckpt))
        assert resolve_ckpt_dir() == fake_ckpt

    def test_returns_none_when_nothing_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dc, "_DEFAULT_CKPT_DIRS", ())
        monkeypatch.setenv("DISTORTION_CKPT_DIR", str(tmp_path / "nope"))
        assert resolve_ckpt_dir() is None

    def test_env_var_without_label_map_ignored(self, tmp_path, monkeypatch):
        """只有 config.json、没有 label_map.json 的目录不算可用检查点。"""
        monkeypatch.setattr(dc, "_DEFAULT_CKPT_DIRS", ())
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv("DISTORTION_CKPT_DIR", str(tmp_path))
        assert resolve_ckpt_dir() is None


class TestSafeEntrypoints:
    """容错入口（/smart-chat 依赖它们不抛异常）"""

    def test_get_classifier_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dc, "_DEFAULT_CKPT_DIRS", ())
        monkeypatch.setenv("DISTORTION_CKPT_DIR", str(tmp_path / "nope"))
        reset_distortion_classifier()
        assert get_distortion_classifier() is None
        reset_distortion_classifier()

    def test_predict_safe_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dc, "_DEFAULT_CKPT_DIRS", ())
        monkeypatch.setenv("DISTORTION_CKPT_DIR", str(tmp_path / "nope"))
        reset_distortion_classifier()
        assert predict_distortion_safe("我很失败") is None
        reset_distortion_classifier()

    def test_predict_safe_handles_blank_text(self):
        assert predict_distortion_safe("") is None
        assert predict_distortion_safe("   ") is None

    def test_load_raises_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dc, "_DEFAULT_CKPT_DIRS", ())
        monkeypatch.setenv("DISTORTION_CKPT_DIR", str(tmp_path / "nope"))
        with pytest.raises(FileNotFoundError, match="未找到认知扭曲分类器检查点"):
            DistortionClassifier.load()


class TestFormatting:
    """提示串格式"""

    def test_format_reads_as_tendency_not_diagnosis(self):
        pred = DistortionPrediction(
            label=5, label_zh="过度泛化", confidence=0.6234,
            probs=(0.0,) * 8, is_distorted=True, margin=0.11,
        )
        hint = format_distortion_hint(pred)
        assert hint == "过度泛化（置信度 0.62）"
        # 不得出现"诊断/确诊/患有"这类措辞
        for banned in ("诊断", "确诊", "患有", "障碍"):
            assert banned not in hint

    def test_non_distorted_flag(self):
        pred = DistortionPrediction(
            label=7, label_zh=NON_DISTORTED_ZH, confidence=0.9,
            probs=(0.0,) * 8, is_distorted=False, margin=0.8,
        )
        assert pred.is_distorted is False

    def test_prediction_is_frozen(self):
        pred = DistortionPrediction(
            label=0, label_zh="非黑即白", confidence=0.5,
            probs=(0.0,) * 8, is_distorted=True, margin=0.1,
        )
        with pytest.raises(Exception):
            pred.label = 1  # type: ignore[misc]
