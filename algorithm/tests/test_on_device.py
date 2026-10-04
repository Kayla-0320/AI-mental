"""
端侧推理模块单元测试

覆盖：
- 数据结构
- 模型构建
- 量化流程
- 基准测试
- 报告生成
"""
from __future__ import annotations

import numpy as np
import pytest

from privacy.on_device import (
    BenchmarkResult,
    QuantizationReport,
    _generate_test_data,
    _get_model_size_mb,
    _get_quantized_model_size_mb,
    _render_markdown,
    generate_report,
)


# ============================================================
# 数据结构测试
# ============================================================

class TestBenchmarkResult:
    def test_creation(self):
        """创建基准测试结果"""
        r = BenchmarkResult(
            model_type="original",
            model_size_mb=250.0,
            avg_latency_ms=100.0,
            p50_latency_ms=90.0,
            p95_latency_ms=150.0,
            accuracy=0.85,
            f1_macro=0.80,
            throughput=10.0,
        )
        assert r.model_type == "original"
        assert r.model_size_mb == 250.0
        assert r.accuracy == 0.85


class TestQuantizationReport:
    def test_creation(self):
        """创建量化报告"""
        orig = BenchmarkResult("original", 250, 100, 90, 150, 0.85, 0.80, 10)
        quant = BenchmarkResult("quantized", 65, 30, 25, 50, 0.83, 0.78, 33)
        report = QuantizationReport(
            base_model="distilbert",
            original=orig,
            quantized=quant,
            size_reduction_pct=74.0,
            latency_speedup=3.33,
            accuracy_drop_pct=2.35,
            meets_size_target=True,
            meets_latency_target=True,
            accuracy_warning=False,
        )
        assert report.meets_size_target is True
        assert report.accuracy_warning is False


# ============================================================
# 模型大小计算测试
# ============================================================

class TestModelSize:
    def test_quantized_smaller_than_original(self):
        """量化模型小于原始模型"""
        import torch
        import torch.nn as nn

        model = nn.Sequential(nn.Linear(100, 50), nn.Linear(50, 10))
        original_size = _get_model_size_mb(model)
        quantized_size = _get_quantized_model_size_mb(model)
        # 量化后应更小（int8 vs float32）
        assert quantized_size < original_size

    def test_model_size_positive(self):
        """模型大小为正值"""
        import torch.nn as nn
        model = nn.Linear(10, 5)
        size = _get_model_size_mb(model)
        assert size > 0


# ============================================================
# 报告生成测试
# ============================================================

class TestReportGeneration:
    @pytest.fixture
    def sample_results(self):
        """样本基准结果"""
        original = BenchmarkResult(
            model_type="original",
            model_size_mb=250.0,
            avg_latency_ms=100.0,
            p50_latency_ms=90.0,
            p95_latency_ms=150.0,
            accuracy=0.85,
            f1_macro=0.80,
            throughput=10.0,
        )
        quantized = BenchmarkResult(
            model_type="quantized",
            model_size_mb=65.0,
            avg_latency_ms=30.0,
            p50_latency_ms=25.0,
            p95_latency_ms=50.0,
            accuracy=0.83,
            f1_macro=0.78,
            throughput=33.3,
        )
        return original, quantized

    def test_generate_report_returns_data(self, sample_results, tmp_path):
        """生成报告返回 QuantizationReport"""
        orig, quant = sample_results
        save_path = str(tmp_path / "report.md")
        report = generate_report(orig, quant, save_path)
        assert isinstance(report, QuantizationReport)
        assert report.base_model is not None

    def test_generate_report_saves_markdown(self, sample_results, tmp_path):
        """报告保存为 Markdown"""
        orig, quant = sample_results
        save_path = str(tmp_path / "report.md")
        generate_report(orig, quant, save_path)
        content = open(save_path, encoding="utf-8").read()
        assert "# 端侧模型量化报告" in content
        assert "模型大小对比" in content
        assert "推理延迟对比" in content

    def test_report_detects_accuracy_warning(self, tmp_path):
        """准确率下降超过 3% 时触发警告"""
        orig = BenchmarkResult("original", 250, 100, 90, 150, 0.90, 0.85, 10)
        quant = BenchmarkResult("quantized", 65, 30, 25, 50, 0.85, 0.80, 33)
        save_path = str(tmp_path / "report.md")
        report = generate_report(orig, quant, save_path)
        assert report.accuracy_warning is True
        assert any("QAT" in r for r in report.recommendations)

    def test_report_no_accuracy_warning(self, tmp_path):
        """准确率下降在 3% 以内无警告"""
        orig = BenchmarkResult("original", 250, 100, 90, 150, 0.85, 0.80, 10)
        quant = BenchmarkResult("quantized", 65, 30, 25, 50, 0.84, 0.79, 33)
        save_path = str(tmp_path / "report.md")
        report = generate_report(orig, quant, save_path)
        assert report.accuracy_warning is False

    def test_report_size_target_check(self, tmp_path):
        """检查大小目标"""
        orig = BenchmarkResult("original", 250, 100, 90, 150, 0.85, 0.80, 10)
        quant = BenchmarkResult("quantized", 30, 30, 25, 50, 0.84, 0.79, 33)
        save_path = str(tmp_path / "report.md")
        report = generate_report(orig, quant, save_path)
        # 30MB > 25MB → 不满足
        assert report.meets_size_target is False

    def test_report_size_target_met(self, tmp_path):
        """大小目标满足"""
        orig = BenchmarkResult("original", 250, 100, 90, 150, 0.85, 0.80, 10)
        quant = BenchmarkResult("quantized", 20, 30, 25, 50, 0.84, 0.79, 33)
        save_path = str(tmp_path / "report.md")
        report = generate_report(orig, quant, save_path)
        # 20MB < 25MB → 满足
        assert report.meets_size_target is True

    def test_report_latency_target_check(self, tmp_path):
        """检查延迟目标"""
        orig = BenchmarkResult("original", 250, 100, 90, 150, 0.85, 0.80, 10)
        quant = BenchmarkResult("quantized", 20, 60, 55, 80, 0.84, 0.79, 16)
        save_path = str(tmp_path / "report.md")
        report = generate_report(orig, quant, save_path)
        # 60ms > 50ms → 不满足
        assert report.meets_latency_target is False

    def test_render_markdown_contains_sections(self):
        """Markdown 包含所有必要章节"""
        orig = BenchmarkResult("original", 250, 100, 90, 150, 0.85, 0.80, 10)
        quant = BenchmarkResult("quantized", 65, 30, 25, 50, 0.84, 0.79, 33)
        report = QuantizationReport(
            base_model="distilbert",
            original=orig, quantized=quant,
            size_reduction_pct=74.0, latency_speedup=3.33,
            accuracy_drop_pct=1.18,
            meets_size_target=True, meets_latency_target=True,
            accuracy_warning=False,
            recommendations=["所有目标均已满足"],
        )
        md = _render_markdown(report)
        assert "模型大小对比" in md
        assert "推理延迟对比" in md
        assert "准确率对比" in md
        assert "目标达成情况" in md
        assert "建议" in md
        assert "隐私保护说明" in md
