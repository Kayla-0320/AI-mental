"""新来源标签边界、归档读取和中性辅助损失的回归。"""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
import tarfile

import pytest
import torch

from experiments.round6_neutral_transfer import binary_logits
from experiments.round6_multitask import auxiliary_logits


def script(name: str) -> object:
    """本地加载数据脚本，测试不访问网络与账号。"""
    source = Path(__file__).resolve().parents[2] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(source.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    return module


@pytest.mark.parametrize("value", [{}, {"emotion_result": "neutral", "data": {}},
                                   {"emotion_result": "unknown", "data": {str(i): {"emotion": "neutral"} for i in range(5)}}])
def test_unlabeled_or_invalid_annotations_never_become_neutral(value: dict) -> None:
    with pytest.raises(ValueError):
        script("round6_prepare_neutral").neutral_label(value)


def test_surprise_is_an_auxiliary_non_neutral_label_without_positive_polarity() -> None:
    value = {"emotion_result": "surprised", "data": {str(i): {"emotion": "surprised"} for i in range(5)}}
    label, agreement = script("round6_prepare_neutral").neutral_label(value)
    assert label == 0 and agreement == 1.
    assert "sentiment" not in value


def test_transcription_emotion_and_overlap_markers_are_not_model_input() -> None:
    module = script("round6_prepare_neutral")
    assert module.clean_text("[over/][laughing]我觉得还行[/over]。[interrupted]") == "我觉得还行。"
    assert module.key_for("ＡＢＣ， 我觉得还行。") == "abc我觉得还行"


def test_archive_reader_rejects_parent_traversal_and_duplicate_ids(tmp_path: Path) -> None:
    module = script("round6_emotiontalk_data")
    for index, names in enumerate([["../bad.json"], ["a/same.json", "b/same.json"]]):
        path = tmp_path / f"bad{index}.tar"
        with tarfile.open(path, "w") as archive:
            for name in names:
                member = tarfile.TarInfo(name)
                body = b'{}'
                member.size = len(body)
                archive.addfile(member, io.BytesIO(body))
        with pytest.raises(ValueError):
            module.read_local_annotations(path)


def test_binary_auxiliary_probability_preserves_three_class_neutral_probability() -> None:
    values = torch.tensor([[1.3, -.8, 2.], [-4., 1., -2.]], requires_grad=True)
    binary = binary_logits(values)
    assert torch.allclose(binary.softmax(1)[:, 1], values.softmax(1)[:, 1])
    torch.nn.functional.cross_entropy(binary, torch.tensor([0, 1])).backward()
    assert torch.isfinite(values.grad).all()


def test_separate_emotion_auxiliary_loss_does_not_train_sentiment_head() -> None:
    from types import SimpleNamespace

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(12, 4)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    encoder = Encoder()
    sentiment = torch.nn.Linear(8, 3)
    auxiliary = torch.nn.Linear(8, 2)
    model = SimpleNamespace(encoder=encoder, head=sentiment)
    row = {"input_ids": torch.tensor([[1, 2, 3], [1, 4, 0]]), "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]])}
    logits = auxiliary_logits(model, auxiliary, row)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert encoder.embedding.weight.grad is not None and torch.isfinite(encoder.embedding.weight.grad).all()
    assert auxiliary.weight.grad is not None and sentiment.weight.grad is None
