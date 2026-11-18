"""第二轮跨模块契约的独立默认值和序列化验证。"""
from dataclasses import asdict

from shared import EmotionForecast, EmotionObservation, MultimodalFeatureBatch, MultimodalPrediction
from shared import DatasetSampleRecord, DatasetAuditReport, ModalityObservation, ModalityQualityReport


def test_observation_and_forecast_roundtrip() -> None:
    point = EmotionObservation(step=3, positive=60.0, negative=20.0)
    assert EmotionObservation(**asdict(point)) == point
    first = EmotionForecast(50, 30, 3, 4, "persistence", 1)
    second = EmotionForecast(50, 30, 3, 4, "persistence", 1)
    first.warnings.append("research_only")
    assert second.warnings == []
    assert first.scope == "next_sampling_observation_research"


def test_multimodal_contract_preserves_absence() -> None:
    batch = MultimodalFeatureBatch(
        ["source:1"], {"text": [[1.0, 2.0]], "video": [[0.0]]},
        {"text": [True], "video": [False]}, {"text": [1.0], "video": [0.0]},
        "source_valence", ["negative", "positive"], {"text": "v1", "video": "v1"},
    )
    assert MultimodalFeatureBatch(**asdict(batch)).observed["video"] == [False]
    result = MultimodalPrediction("source_valence", ["negative", "positive"], [], [], [],
                                  ["source:1"], "unfitted", "v1", [True])
    assert asdict(result)["abstained"] == [True]


def test_quality_contracts_roundtrip_and_no_shared_hashes() -> None:
    left = DatasetSampleRecord("dataset", "1", "person1", "train", "valence", 1)
    right = DatasetSampleRecord("dataset", "2", "person2", "test", "valence", 0)
    left.content_hashes["text"] = "sha"
    assert right.content_hashes == {}
    objects = [left, DatasetAuditReport(True, 1, {"train": 1}, [], [], []),
               ModalityObservation("audio", [0.0], available=False),
               ModalityQualityReport("audio", False, 0.0, ["absent"], [0.0])]
    for item in objects:
        assert type(item)(**asdict(item)) == item
