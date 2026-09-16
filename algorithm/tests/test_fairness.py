"""
公平性审计单元测试

覆盖：
- 分组指标计算
- 公平性指标（DPD / EOD）
- Markdown 报告生成
- 图表生成
- 样本不足标注
- 建议生成（不污名化）
"""
from __future__ import annotations

import numpy as np
import pytest

from audit.fairness import (
    GroupMetrics,
    FairnessReport,
    MIN_SAMPLE_SIZE,
    _compute_group_metrics,
    _generate_recommendations,
    generate_audit_report,
    generate_chart,
    render_markdown_report,
    run_full_audit,
)


# ============================================================
# 辅助数据
# ============================================================

@pytest.fixture
def sample_data():
    """样本数据"""
    rng = np.random.RandomState(42)
    n = 200

    predictions = rng.randint(0, 2, n)
    labels = rng.randint(0, 2, n)

    demographics = {
        "ages": rng.choice([12, 13, 14, 15, 16, 17, 18], n),
        "genders": rng.choice(["male", "female"], n),
        "regions": rng.choice(["urban", "town", "rural"], n, p=[0.4, 0.3, 0.3]),
    }

    return predictions, labels, demographics


@pytest.fixture
def small_sample_data():
    """小样本数据（某些组 < 30）"""
    rng = np.random.RandomState(42)
    n = 50

    predictions = rng.randint(0, 2, n)
    labels = rng.randint(0, 2, n)

    demographics = {
        "ages": rng.choice([12, 13, 16, 17], n),
        "genders": rng.choice(["male", "female"], n),
        "regions": rng.choice(["urban", "rural"], n, p=[0.9, 0.1]),  # rural 很少
    }

    return predictions, labels, demographics


# ============================================================
# 数据结构测试
# ============================================================

class TestGroupMetrics:
    def test_creation(self):
        """创建组指标"""
        m = GroupMetrics(
            group_name="12-15",
            dimension="age",
            sample_size=50,
            accuracy=0.8,
            f1=0.75,
            tpr=0.7,
            fpr=0.2,
        )
        assert m.group_name == "12-15"
        assert m.insufficient_sample is False

    def test_insufficient_sample(self):
        """样本不足标注"""
        m = GroupMetrics(
            group_name="rural",
            dimension="region",
            sample_size=10,
            accuracy=0.7,
            f1=0.6,
            tpr=0.5,
            fpr=0.3,
            insufficient_sample=True,
        )
        assert m.insufficient_sample is True


# ============================================================
# 分组指标计算测试
# ============================================================

class TestComputeGroupMetrics:
    def test_basic_computation(self):
        """基本指标计算"""
        preds = np.array([1, 0, 1, 1, 0, 1])
        labels = np.array([1, 0, 0, 1, 0, 1])
        mask = np.array([True, True, True, True, True, True])

        m = _compute_group_metrics(preds, labels, mask, "test", "test_dim")
        assert m.sample_size == 6
        assert 0 <= m.accuracy <= 1
        assert 0 <= m.f1 <= 1
        assert 0 <= m.tpr <= 1
        assert 0 <= m.fpr <= 1

    def test_empty_group(self):
        """空组"""
        preds = np.array([1, 0])
        labels = np.array([1, 0])
        mask = np.array([False, False])

        m = _compute_group_metrics(preds, labels, mask, "empty", "test")
        assert m.sample_size == 0
        assert m.insufficient_sample is True

    def test_insufficient_sample_flag(self):
        """样本不足标记"""
        preds = np.array([1, 0, 1])
        labels = np.array([1, 0, 0])
        mask = np.array([True, True, True])

        m = _compute_group_metrics(preds, labels, mask, "small", "test")
        assert m.sample_size == 3
        assert m.insufficient_sample is True  # 3 < 30


# ============================================================
# 核心接口测试
# ============================================================

class TestGenerateAuditReport:
    def test_report_structure(self, sample_data):
        """报告结构完整"""
        preds, labels, demo = sample_data
        report = generate_audit_report(preds, labels, demo)
        assert isinstance(report, FairnessReport)
        assert len(report.group_metrics) > 0
        assert report.timestamp != ""

    def test_report_has_all_dimensions(self, sample_data):
        """报告包含所有维度"""
        preds, labels, demo = sample_data
        report = generate_audit_report(preds, labels, demo)
        dims = {m.dimension for m in report.group_metrics}
        assert "age" in dims
        assert "gender" in dims
        assert "region" in dims

    def test_fairness_metrics_range(self, sample_data):
        """公平性指标在 [0, 1] 范围"""
        preds, labels, demo = sample_data
        report = generate_audit_report(preds, labels, demo)
        assert 0 <= report.demographic_parity_diff <= 1
        assert 0 <= report.equalized_odds_diff <= 1

    def test_small_sample_flagged(self, small_sample_data):
        """小样本组被标注"""
        preds, labels, demo = small_sample_data
        report = generate_audit_report(preds, labels, demo)
        insufficient = [m for m in report.group_metrics if m.insufficient_sample]
        assert len(insufficient) > 0

    def test_recommendations_present(self, sample_data):
        """报告包含建议"""
        preds, labels, demo = sample_data
        report = generate_audit_report(preds, labels, demo)
        assert len(report.recommendations) > 0


# ============================================================
# 建议生成测试
# ============================================================

class TestRecommendations:
    def test_no_stigmatizing_language(self):
        """建议不包含污名化语言"""
        metrics = [
            GroupMetrics("rural", "region", 20, 0.5, 0.4, 0.3, 0.4, True),
            GroupMetrics("urban", "region", 100, 0.8, 0.75, 0.7, 0.2, False),
        ]
        recs = _generate_recommendations(metrics, 0.15, 0.12)
        rec_text = " ".join(recs)
        # 不应包含对群体的负面定性描述
        assert "落后" not in rec_text
        assert "差" not in rec_text.split("差距")[0] if "差距" in rec_text else True

    def test_recommends_data_collection(self):
        """建议增加数据采集"""
        metrics = [
            GroupMetrics("rural", "region", 10, 0.6, 0.5, 0.4, 0.3, True),
        ]
        recs = _generate_recommendations(metrics, 0.05, 0.05)
        assert any("样本" in r for r in recs)

    def test_good_fairness_message(self):
        """公平性良好时的消息"""
        metrics = [
            GroupMetrics("A", "test", 100, 0.8, 0.75, 0.7, 0.2, False),
            GroupMetrics("B", "test", 100, 0.78, 0.73, 0.68, 0.22, False),
        ]
        recs = _generate_recommendations(metrics, 0.02, 0.03)
        assert any("均衡" in r or "稳定" in r for r in recs)


# ============================================================
# Markdown 报告测试
# ============================================================

class TestMarkdownReport:
    def test_report_contains_sections(self, sample_data):
        """报告包含所有章节"""
        preds, labels, demo = sample_data
        report = generate_audit_report(preds, labels, demo)
        md = render_markdown_report(report)
        assert "# 公平性审计报告" in md
        assert "分组性能指标" in md
        assert "公平性指标" in md
        assert "改进建议" in md
        assert "声明" in md

    def test_report_contains_disclaimer(self, sample_data):
        """报告包含不污名化声明"""
        preds, labels, demo = sample_data
        report = generate_audit_report(preds, labels, demo)
        md = render_markdown_report(report)
        assert "不反映任何群体的真实心理特征差异" in md

    def test_insufficient_sample_noted(self, small_sample_data):
        """样本不足组在报告中标注"""
        preds, labels, demo = small_sample_data
        report = generate_audit_report(preds, labels, demo)
        md = render_markdown_report(report)
        assert "样本不足" in md or "无数据" in md


# ============================================================
# 图表生成测试
# ============================================================

class TestChartGeneration:
    def test_chart_generated(self, sample_data, tmp_path):
        """图表成功生成"""
        preds, labels, demo = sample_data
        report = generate_audit_report(preds, labels, demo)
        chart_path = str(tmp_path / "chart.png")
        result = generate_chart(report, chart_path)
        assert result != ""
        assert Path(result).exists()

    def test_chart_with_small_samples(self, small_sample_data, tmp_path):
        """小样本数据也能生成图表"""
        preds, labels, demo = small_sample_data
        report = generate_audit_report(preds, labels, demo)
        chart_path = str(tmp_path / "chart.png")
        result = generate_chart(report, chart_path)
        assert result != ""


# ============================================================
# 完整流程测试
# ============================================================

class TestFullAudit:
    def test_full_audit_flow(self, sample_data, tmp_path):
        """完整审计流程"""
        preds, labels, demo = sample_data
        report_path = str(tmp_path / "report.md")
        chart_path = str(tmp_path / "chart.png")

        report = run_full_audit(preds, labels, demo, report_path, chart_path)
        assert isinstance(report, FairnessReport)
        assert Path(report_path).exists()
        assert Path(chart_path).exists()


# 需要 Path import
from pathlib import Path
