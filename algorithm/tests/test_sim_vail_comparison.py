"""
SIM-VAIL 对比实验单元测试

测试目标：
1. baseline_driven 参数正确控制审计器行为
2. run_comparison() 返回完整的对比报告
3. 对比报告 Markdown 渲染正确
4. FastAPI 路由返回 200
5. 不破坏现有测试
"""
import pytest

from sim_vail import (
    SimVailTestEngine,
    ComparisonReport,
    ProfileComparison,
    get_profiles,
)
from sim_vail.report import (
    generate_comparison_report,
    save_comparison_report,
)


# ============================================================
# 测试：baseline_driven 参数
# ============================================================

class TestBaselineDrivenParam:
    """测试 baseline_driven 参数"""

    def test_engine_default_no_baseline(self):
        """默认 baseline_driven=False"""
        engine = SimVailTestEngine()
        assert engine.baseline_driven is False

    def test_engine_baseline_driven_true(self):
        """baseline_driven=True 正确设置"""
        engine = SimVailTestEngine(baseline_driven=True)
        assert engine.baseline_driven is True

    def test_synthetic_deviation_empty_for_normal(self):
        """正常轮次的模拟偏离为空"""
        engine = SimVailTestEngine(baseline_driven=True)
        profiles = get_profiles()
        profile = profiles[0]
        # 第一轮是 NORMAL 状态
        script = profile.turn_scripts[0]

        deviation = engine._build_synthetic_deviation(profile, script)

        assert deviation.significant_deviations == []

    def test_synthetic_deviation_populated_for_crisis(self):
        """危机轮次的模拟偏离非空"""
        engine = SimVailTestEngine(baseline_driven=True)
        profiles = get_profiles()
        # 画像 2 (焦虑+学业压力) 有 somatic_symptoms=0.7
        profile = profiles[1]
        # 第 4 轮是 CRISIS 状态
        script = profile.turn_scripts[4]

        deviation = engine._build_synthetic_deviation(profile, script)

        assert len(deviation.significant_deviations) > 0
        # 焦虑画像应有 emotion_anxiety
        assert "emotion_anxiety" in deviation.significant_deviations


# ============================================================
# 测试：run_comparison
# ============================================================

class TestRunComparison:
    """测试 run_comparison 方法"""

    def test_comparison_returns_report(self):
        """run_comparison 返回 ComparisonReport"""
        engine = SimVailTestEngine()
        report = engine.run_comparison()

        assert isinstance(report, ComparisonReport)
        assert report.run_id.startswith("cmp_")
        assert len(report.profile_comparisons) == 5

    def test_comparison_profile_ids_match(self):
        """对比报告画像 ID 与原始画像一致"""
        engine = SimVailTestEngine()
        report = engine.run_comparison()
        profiles = get_profiles()

        expected_ids = {p.profile_id for p in profiles}
        actual_ids = {pc.profile_id for pc in report.profile_comparisons}

        assert actual_ids == expected_ids

    def test_comparison_rates_in_range(self):
        """对比报告的比率在 [0, 1] 范围内"""
        engine = SimVailTestEngine()
        report = engine.run_comparison()

        assert 0 <= report.overall_baseline_interception_rate <= 1
        assert 0 <= report.overall_no_baseline_interception_rate <= 1

        for pc in report.profile_comparisons:
            assert 0 <= pc.baseline_interception_rate <= 1
            assert 0 <= pc.no_baseline_interception_rate <= 1
            assert 0 <= pc.baseline_miss_rate <= 1
            assert 0 <= pc.no_baseline_miss_rate <= 1

    def test_comparison_is_reproducible(self):
        """对比实验结果可复现（固定随机种子）"""
        engine1 = SimVailTestEngine()
        report1 = engine1.run_comparison()

        engine2 = SimVailTestEngine()
        report2 = engine2.run_comparison()

        assert (
            report1.overall_baseline_interception_rate
            == report2.overall_baseline_interception_rate
        )
        assert (
            report1.overall_no_baseline_interception_rate
            == report2.overall_no_baseline_interception_rate
        )

    def test_improvement_calculation(self):
        """提升幅度计算正确"""
        engine = SimVailTestEngine()
        report = engine.run_comparison()

        expected = round(
            report.overall_baseline_interception_rate
            - report.overall_no_baseline_interception_rate,
            4,
        )
        assert report.overall_improvement == expected


# ============================================================
# 测试：对比报告渲染
# ============================================================

class TestComparisonReportRendering:
    """测试对比报告 Markdown 渲染"""

    def test_report_contains_header(self):
        """报告包含标题"""
        engine = SimVailTestEngine()
        report = engine.run_comparison()
        md = generate_comparison_report(report)

        assert "# SIM-VAIL 基线驱动审计对比报告" in md

    def test_report_contains_summary_table(self):
        """报告包含总体汇总表格"""
        engine = SimVailTestEngine()
        report = engine.run_comparison()
        md = generate_comparison_report(report)

        assert "## 总体汇总" in md
        assert "基线驱动" in md
        assert "无基线" in md
        assert "提升幅度" in md

    def test_report_contains_profile_sections(self):
        """报告包含各画像对比详情"""
        engine = SimVailTestEngine()
        report = engine.run_comparison()
        md = generate_comparison_report(report)

        assert "## 各画像对比详情" in md
        for pc in report.profile_comparisons:
            assert pc.profile_name in md

    def test_report_contains_experiment_notes(self):
        """报告包含实验说明"""
        engine = SimVailTestEngine()
        report = engine.run_comparison()
        md = generate_comparison_report(report)

        assert "## 实验说明" in md
        assert "boost=1.5x" in md


# ============================================================
# 测试：报告保存
# ============================================================

class TestSaveComparisonReport:
    """测试对比报告保存"""

    def test_save_creates_file(self, tmp_path):
        """保存报告创建文件"""
        engine = SimVailTestEngine()
        report = engine.run_comparison()

        path = save_comparison_report(report, output_dir=tmp_path)

        assert path.endswith("comparison_report.md")
        from pathlib import Path
        assert Path(path).exists()


# ============================================================
# 测试：FastAPI 路由
# ============================================================

class TestComparisonAPIRoute:
    """测试对比实验 API 路由"""

    def test_comparison_endpoint_returns_200(self):
        """POST /sim-vail/comparison 返回 200"""
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        response = client.post("/api/v1/sim-vail/comparison")

        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        assert "overall_baseline_interception_rate" in data
        assert "overall_no_baseline_interception_rate" in data
        assert "markdown_report" in data
        assert len(data["profile_comparisons"]) == 5


# ============================================================
# 运行测试
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
