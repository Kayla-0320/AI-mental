"""
共病模式分析模块测试
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from assessment.comorbidity import (
    analyze_comorbidity,
    batch_analyze,
    get_comorbidity_prevalence,
    generate_heatmap_data,
    get_high_risk_users,
    ComorbidityType,
    COMORBIDITY_NAMES,
    COMORBIDITY_RECOMMENDATIONS,
)


class TestComorbidityType:
    """测试共病类型枚举"""

    def test_all_types_exist(self):
        assert ComorbidityType.NONE == "none"
        assert ComorbidityType.DEPRESSION_ONLY == "depression_only"
        assert ComorbidityType.ANXIETY_ONLY == "anxiety_only"
        assert ComorbidityType.SLEEP_ONLY == "sleep_only"
        assert ComorbidityType.DEP_ANX == "dep_anx"
        assert ComorbidityType.DEP_SLEEP == "dep_sleep"
        assert ComorbidityType.ANX_SLEEP == "anx_sleep"
        assert ComorbidityType.TRIPLE == "triple"

    def test_all_types_have_names(self):
        for ctype in ComorbidityType:
            assert ctype in COMORBIDITY_NAMES

    def test_all_types_have_recommendations(self):
        for ctype in ComorbidityType:
            assert ctype in COMORBIDITY_RECOMMENDATIONS


class TestAnalyzeComorbidity:
    """测试共病分析"""

    def test_no_risk(self):
        result = analyze_comorbidity("user_1", 0.1, 0.1, 0.1)
        assert result.comorbidity_type == ComorbidityType.NONE
        assert result.severity_score == pytest.approx(0.3, abs=0.01)

    def test_depression_only(self):
        result = analyze_comorbidity("user_1", 0.7, 0.1, 0.1)
        assert result.comorbidity_type == ComorbidityType.DEPRESSION_ONLY

    def test_anxiety_only(self):
        result = analyze_comorbidity("user_1", 0.1, 0.7, 0.1)
        assert result.comorbidity_type == ComorbidityType.ANXIETY_ONLY

    def test_sleep_only(self):
        result = analyze_comorbidity("user_1", 0.1, 0.1, 0.7)
        assert result.comorbidity_type == ComorbidityType.SLEEP_ONLY

    def test_dep_anx_comorbidity(self):
        result = analyze_comorbidity("user_1", 0.7, 0.8, 0.1)
        assert result.comorbidity_type == ComorbidityType.DEP_ANX

    def test_dep_sleep_comorbidity(self):
        result = analyze_comorbidity("user_1", 0.7, 0.1, 0.8)
        assert result.comorbidity_type == ComorbidityType.DEP_SLEEP

    def test_anx_sleep_comorbidity(self):
        result = analyze_comorbidity("user_1", 0.1, 0.7, 0.8)
        assert result.comorbidity_type == ComorbidityType.ANX_SLEEP

    def test_triple_comorbidity(self):
        result = analyze_comorbidity("user_1", 0.8, 0.7, 0.9)
        assert result.comorbidity_type == ComorbidityType.TRIPLE

    def test_custom_threshold(self):
        result = analyze_comorbidity("user_1", 0.4, 0.1, 0.1, threshold=0.3)
        assert result.comorbidity_type == ComorbidityType.DEPRESSION_ONLY

    def test_risk_combinations_populated(self):
        result = analyze_comorbidity("user_1", 0.7, 0.8, 0.1)
        assert len(result.risk_combinations) == 2

    def test_severity_score_range(self):
        result = analyze_comorbidity("user_1", 0.5, 0.6, 0.7)
        assert 0.0 <= result.severity_score <= 3.0

    def test_recommendation_not_empty(self):
        result = analyze_comorbidity("user_1", 0.8, 0.7, 0.9)
        assert len(result.recommendation) > 0


class TestBatchAnalyze:
    """测试批量分析"""

    def test_batch_analyze(self):
        data = [
            {"user_id": "u1", "depression": 0.7, "anxiety": 0.8, "sleep": 0.1},
            {"user_id": "u2", "depression": 0.1, "anxiety": 0.1, "sleep": 0.1},
            {"user_id": "u3", "depression": 0.8, "anxiety": 0.7, "sleep": 0.9},
        ]
        results = batch_analyze(data)
        assert len(results) == 3
        assert results[0].comorbidity_type == ComorbidityType.DEP_ANX
        assert results[1].comorbidity_type == ComorbidityType.NONE
        assert results[2].comorbidity_type == ComorbidityType.TRIPLE

    def test_empty_batch(self):
        results = batch_analyze([])
        assert len(results) == 0


class TestPrevalence:
    """测试流行率统计"""

    def test_prevalence_calculation(self):
        results = [
            analyze_comorbidity(f"u{i}", 0.7, 0.8, 0.1) for i in range(5)
        ] + [
            analyze_comorbidity(f"u{i+5}", 0.1, 0.1, 0.1) for i in range(5)
        ]
        prevalence = get_comorbidity_prevalence(results)
        assert "dep_anx" in prevalence
        assert prevalence["dep_anx"]["count"] == 5
        assert prevalence["dep_anx"]["prevalence"] == 0.5

    def test_empty_prevalence(self):
        prevalence = get_comorbidity_prevalence([])
        assert prevalence == {}


class TestHeatmapData:
    """测试热力图数据"""

    def test_heatmap_generation(self):
        results = [
            analyze_comorbidity(f"u{i}", 0.3 + i * 0.1, 0.4 + i * 0.1, 0.2 + i * 0.05)
            for i in range(10)
        ]
        data = generate_heatmap_data(results)
        assert data["labels"] == ["抑郁", "焦虑", "睡眠"]
        assert len(data["matrix"]) == 3
        assert data["total_users"] == 10

    def test_empty_heatmap(self):
        data = generate_heatmap_data([])
        assert data["matrix"] == []


class TestHighRiskUsers:
    """测试高风险用户筛选"""

    def test_filter_by_severity(self):
        results = [
            analyze_comorbidity("u1", 0.8, 0.7, 0.9),  # severity ~2.4
            analyze_comorbidity("u2", 0.1, 0.1, 0.1),  # severity ~0.3
            analyze_comorbidity("u3", 0.6, 0.7, 0.5),  # severity ~1.8
        ]
        high_risk = get_high_risk_users(results, min_severity=1.5)
        assert len(high_risk) == 2
        assert high_risk[0].severity_score >= high_risk[1].severity_score

    def test_filter_by_type(self):
        results = [
            analyze_comorbidity("u1", 0.8, 0.7, 0.9),  # triple
            analyze_comorbidity("u2", 0.7, 0.8, 0.1),  # dep_anx
            analyze_comorbidity("u3", 0.1, 0.1, 0.1),  # none
        ]
        high_risk = get_high_risk_users(
            results,
            min_severity=0.0,
            comorbidity_types=[ComorbidityType.TRIPLE],
        )
        assert len(high_risk) == 1
        assert high_risk[0].comorbidity_type == ComorbidityType.TRIPLE
