"""危机严重程度分级器 + Suici-Urgency 数据加载器测试"""
import json
import tempfile
from pathlib import Path

from escalation.crisis_severity import (
    CrisisSeverity,
    CrisisSeverityResult,
    classify_severity,
    batch_classify,
    severity_to_risk_level,
    get_signal_stats,
)
from escalation.suici_urgency_loader import (
    CrisisSample,
    CrisisDataset,
    load_suici_urgency,
    split_dataset,
    evaluate_classifier,
    generate_demo_samples,
)
from shared.dataclasses import RiskLevel


# ============================================================
# 分级器测试
# ============================================================

def test_classify_no_crisis():
    result = classify_severity("今天天气不错")
    assert result.severity == CrisisSeverity.NONE
    assert result.risk_level == RiskLevel.LOW


def test_classify_low_risk():
    result = classify_severity("最近好累啊，不想上学")
    assert result.severity == CrisisSeverity.LOW
    assert result.risk_level == RiskLevel.LOW


def test_classify_moderate_risk():
    result = classify_severity("活着没意思，我是废物")
    assert result.severity == CrisisSeverity.MODERATE
    assert result.risk_level == RiskLevel.MEDIUM


def test_classify_high_risk():
    result = classify_severity("我不想活了，想自杀")
    assert result.severity == CrisisSeverity.HIGH
    assert result.risk_level == RiskLevel.HIGH


def test_classify_severe_risk():
    result = classify_severity("遗书写好了，把东西都送给朋友了")
    assert result.severity == CrisisSeverity.SEVERE
    assert result.risk_level == RiskLevel.CRISIS


def test_classify_imminent_risk():
    result = classify_severity("我已经站在楼顶了，再见了")
    assert result.severity == CrisisSeverity.IMMINENT
    assert result.risk_level == RiskLevel.CRISIS


def test_classify_empty():
    result = classify_severity("")
    assert result.severity == CrisisSeverity.NONE
    assert result.confidence == 1.0


def test_classify_priority():
    """高等级信号优先于低等级"""
    result = classify_severity("活着没意思，我想自杀，遗书都写好了")
    assert result.severity >= CrisisSeverity.SEVERE


def test_classify_confidence():
    """多信号命中提高置信度"""
    r1 = classify_severity("我想死")
    r2 = classify_severity("我想死，不想活了，想结束一切")
    assert r2.confidence >= r1.confidence


def test_batch_classify():
    results = batch_classify(["你好", "我想自杀", "今天很开心"])
    assert len(results) == 3
    assert results[0].severity == CrisisSeverity.NONE
    assert results[1].severity >= CrisisSeverity.HIGH
    assert results[2].severity == CrisisSeverity.NONE


def test_severity_to_risk_level():
    assert severity_to_risk_level(CrisisSeverity.NONE) == RiskLevel.LOW
    assert severity_to_risk_level(CrisisSeverity.LOW) == RiskLevel.LOW
    assert severity_to_risk_level(CrisisSeverity.MODERATE) == RiskLevel.MEDIUM
    assert severity_to_risk_level(CrisisSeverity.HIGH) == RiskLevel.HIGH
    assert severity_to_risk_level(CrisisSeverity.SEVERE) == RiskLevel.CRISIS
    assert severity_to_risk_level(CrisisSeverity.IMMINENT) == RiskLevel.CRISIS


def test_signal_stats():
    stats = get_signal_stats()
    assert "total" in stats
    assert stats["total"] > 50  # 应有足够多的信号词条


# ============================================================
# 数据加载器测试
# ============================================================

def test_generate_demo_samples():
    dataset = generate_demo_samples(n_per_level=3)
    assert dataset.total == 18  # 6 levels * 3 samples
    assert 1 in dataset.distribution
    assert 6 in dataset.distribution
    assert dataset.distribution[1] == 3


def test_split_dataset():
    dataset = generate_demo_samples(n_per_level=5)
    train, val, test = split_dataset(dataset.samples)
    assert len(train) + len(val) + len(test) == dataset.total
    assert len(train) > len(val)


def test_load_json_array():
    data = [
        {"text": "我想死", "severity": 4},
        {"text": "今天还行", "severity": 1},
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as f:
        json.dump(data, f, ensure_ascii=False)
        f.flush()
        dataset = load_suici_urgency(f.name)

    assert dataset.total == 2
    assert dataset.samples[0].severity == 4
    assert dataset.samples[1].severity == 1
    Path(f.name).unlink()


def test_load_jsonl():
    lines = [
        '{"text": "活着没意思", "severity": 3}',
        '{"text": "遗书写好了", "severity": 5}',
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", encoding="utf-8", delete=False
    ) as f:
        f.write("\n".join(lines))
        f.flush()
        dataset = load_suici_urgency(f.name)

    assert dataset.total == 2
    assert dataset.samples[0].severity == 3
    assert dataset.samples[1].severity == 5
    Path(f.name).unlink()


def test_load_json_object():
    data = {"data": [
        {"text": "好累", "severity": 2, "time": "最近"},
    ]}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as f:
        json.dump(data, f, ensure_ascii=False)
        f.flush()
        dataset = load_suici_urgency(f.name)

    assert dataset.total == 1
    assert dataset.samples[0].time_clue != ""
    Path(f.name).unlink()


def test_load_alternative_field_names():
    data = [{"content": "我想死", "level": 4}]
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", encoding="utf-8", delete=False
    ) as f:
        json.dump(data, f, ensure_ascii=False)
        f.flush()
        dataset = load_suici_urgency(f.name)

    assert dataset.samples[0].text == "我想死"
    assert dataset.samples[0].severity == 4
    Path(f.name).unlink()


def test_evaluate_classifier():
    dataset = generate_demo_samples(n_per_level=5)
    metrics = evaluate_classifier(dataset.samples)
    assert "accuracy" in metrics
    assert "total" in metrics
    assert "recall_by_level" in metrics
    assert metrics["total"] == 30
    assert metrics["accuracy"] > 0.5  # 规则匹配应有一定准确率


def test_crisis_sample_dataclass():
    sample = CrisisSample(text="测试", severity=3, time_clue="昨天")
    assert sample.text == "测试"
    assert sample.severity == 3
    assert sample.source == "suici_urgency"
