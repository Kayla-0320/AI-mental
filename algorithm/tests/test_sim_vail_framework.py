"""
SIM-VAIL 动态测试框架单元测试

覆盖：
- 5 个青少年脆弱性画像定义验证
- 模拟用户 Agent 消息生成
- 多轮对话测试引擎
- 审计记录与 Likert 评分
- JSON + Markdown 报告生成
- 报告文件保存
- 风险热力图生成
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim_vail import (
    VulnerabilityProfile,
    SimVailTestEngine,
    SimulatedUserAgent,
    MockSystemResponder,
    RiskState,
    DialogueIntent,
    TurnScript,
    get_profiles,
    OUTPUT_DIR,
)
from sim_vail.report import (
    generate_json_report,
    generate_markdown_report,
    save_reports,
    _build_heatmap,
    _collect_triggered_cases,
)


# ============================================================
# 画像定义验证
# ============================================================

class TestProfileDefinitions:
    """5 个青少年脆弱性画像验证"""

    def test_five_profiles_exist(self):
        """存在 5 个画像"""
        profiles = get_profiles()
        assert len(profiles) == 5

    def test_profile_ids_unique(self):
        """画像 ID 唯一"""
        profiles = get_profiles()
        ids = [p.profile_id for p in profiles]
        assert len(ids) == len(set(ids))

    def test_each_profile_has_turn_scripts(self):
        """每个画像都有对话脚本"""
        profiles = get_profiles()
        for p in profiles:
            assert len(p.turn_scripts) >= 5, (
                f"{p.name} 的对话脚本少于 5 轮"
            )

    def test_each_profile_has_dimensions(self):
        """每个画像都有心理维度"""
        profiles = get_profiles()
        for p in profiles:
            assert len(p.dimensions) >= 3, (
                f"{p.name} 的维度少于 3 个"
            )

    def test_each_profile_has_expected_crisis_turns(self):
        """每个画像都标记了预期危机轮次"""
        profiles = get_profiles()
        for p in profiles:
            assert len(p.expected_crisis_turns) > 0, (
                f"{p.name} 没有标记预期危机轮次"
            )

    def test_profile_has_clinical_basis(self):
        """每个画像都有临床依据"""
        profiles = get_profiles()
        for p in profiles:
            assert p.clinical_basis != ""
            assert len(p.clinical_basis) > 20

    def test_depression_profile_exists(self):
        """抑郁+低自尊画像存在"""
        profiles = get_profiles()
        ids = [p.profile_id for p in profiles]
        assert "depression_low_selfesteem" in ids

    def test_anxiety_profile_exists(self):
        """焦虑+学业压力画像存在"""
        profiles = get_profiles()
        ids = [p.profile_id for p in profiles]
        assert "anxiety_academic_pressure" in ids

    def test_bpd_profile_exists(self):
        """边缘性人格倾向+自伤史画像存在"""
        profiles = get_profiles()
        ids = [p.profile_id for p in profiles]
        assert "bpd_selfharm_history" in ids


# ============================================================
# 模拟用户 Agent
# ============================================================

class TestSimulatedUserAgent:
    """模拟用户 Agent 测试"""

    def test_agent_generates_messages(self):
        """Agent 能按序生成消息"""
        profile = get_profiles()[0]
        agent = SimulatedUserAgent(profile)

        msg1 = agent.generate_message()
        assert msg1 is not None
        assert msg1.turn_idx == 0

        msg2 = agent.generate_message()
        assert msg2 is not None
        assert msg2.turn_idx == 1

    def test_agent_returns_none_when_exhausted(self):
        """消息用完后返回 None"""
        profile = get_profiles()[0]
        agent = SimulatedUserAgent(profile)

        for _ in range(len(profile.turn_scripts)):
            agent.generate_message()

        assert agent.generate_message() is None

    def test_agent_reset(self):
        """Agent 重置后可重新生成"""
        profile = get_profiles()[0]
        agent = SimulatedUserAgent(profile)

        agent.generate_message()
        agent.generate_message()
        agent.reset()

        msg = agent.generate_message()
        assert msg is not None
        assert msg.turn_idx == 0

    def test_agent_system_prompt(self):
        """Agent 生成系统提示词"""
        profile = get_profiles()[0]
        agent = SimulatedUserAgent(profile)
        prompt = agent.get_system_prompt()
        assert profile.name in prompt


# ============================================================
# 模拟系统回复
# ============================================================

class TestMockSystemResponder:
    """模拟系统回复生成器"""

    def test_respond_returns_text(self):
        """回复返回非空文本"""
        responder = MockSystemResponder()
        response = responder.respond("你好", RiskState.NORMAL)
        assert len(response) > 0

    def test_crisis_response_differs(self):
        """危机回复与正常回复不同"""
        responder = MockSystemResponder()
        normal = responder.respond("今天不错", RiskState.NORMAL)
        crisis = responder.respond("不想活了", RiskState.CRISIS)
        assert normal != crisis


# ============================================================
# 测试引擎
# ============================================================

class TestSimVailEngine:
    """SIM-VAIL 测试引擎"""

    def test_run_single_profile(self):
        """运行单个画像测试"""
        engine = SimVailTestEngine()
        profile = get_profiles()[0]
        result = engine.run_single_profile(profile)

        assert result.profile_id == profile.profile_id
        assert result.total_turns == len(profile.turn_scripts)
        assert len(result.audit_records) > 0

    def test_run_all_profiles(self):
        """运行所有画像测试"""
        engine = SimVailTestEngine()
        report = engine.run_all()

        assert report.total_profiles == 5
        assert report.total_turns > 0
        assert len(report.profile_results) == 5

    def test_report_has_metrics(self):
        """报告包含核心指标"""
        engine = SimVailTestEngine()
        report = engine.run_all()

        assert 0 <= report.overall_interception_rate <= 1
        assert 0 <= report.overall_miss_rate <= 1
        assert 0 <= report.overall_false_alarm_rate <= 1
        assert 1 <= report.avg_likert_score <= 5

    def test_max_turns_configurable(self):
        """最大轮次可配置"""
        engine = SimVailTestEngine(max_turns=3)
        profile = get_profiles()[0]
        result = engine.run_single_profile(profile)

        assert result.total_turns == 3

    def test_audit_records_have_likert(self):
        """审计记录包含 Likert 评分"""
        engine = SimVailTestEngine()
        profile = get_profiles()[0]
        result = engine.run_single_profile(profile)

        for rec in result.audit_records:
            assert 1 <= rec.likert_score <= 5

    def test_with_safety_loop(self):
        """使用 SafetyLoop 闭环运行"""
        engine = SimVailTestEngine(use_safety_loop=True, max_turns=3)
        profile = get_profiles()[0]
        result = engine.run_single_profile(profile)

        assert result.total_turns == 3


# ============================================================
# 报告生成
# ============================================================

class TestReportGeneration:
    """报告生成测试"""

    @pytest.fixture
    def sample_report(self):
        engine = SimVailTestEngine()
        return engine.run_all()

    def test_json_report_valid(self, sample_report):
        """JSON 报告格式有效"""
        json_str = generate_json_report(sample_report)
        data = json.loads(json_str)

        assert "report_id" in data
        assert "summary" in data
        assert "profiles" in data
        assert len(data["profiles"]) == 5

    def test_json_report_has_metrics(self, sample_report):
        """JSON 报告包含指标"""
        data = json.loads(generate_json_report(sample_report))

        assert "interception_rate" in data["summary"]
        assert "miss_rate" in data["summary"]
        assert "false_alarm_rate" in data["summary"]

    def test_markdown_report_has_sections(self, sample_report):
        """Markdown 报告包含必要段落"""
        md = generate_markdown_report(sample_report)

        assert "# SIM-VAIL" in md
        assert "## 总体指标" in md
        assert "## 风险热力图" in md
        assert "## 各画像详细结果" in md
        assert "## 触发审计的案例" in md

    def test_markdown_report_has_heatmap(self, sample_report):
        """Markdown 报告包含热力图"""
        md = generate_markdown_report(sample_report)
        assert "```" in md

    def test_heatmap_contains_all_profiles(self, sample_report):
        """热力图包含所有画像"""
        heatmap_lines = _build_heatmap(sample_report)
        heatmap_text = "\n".join(heatmap_lines)

        for pr in sample_report.profile_results:
            assert pr.profile_name[:6] in heatmap_text

    def test_save_reports_creates_files(self, sample_report, tmp_path):
        """保存报告创建文件"""
        paths = save_reports(sample_report, output_dir=tmp_path)

        assert "json" in paths
        assert "markdown" in paths
        assert Path(paths["json"]).exists()
        assert Path(paths["markdown"]).exists()

    def test_saved_json_is_valid(self, sample_report, tmp_path):
        """保存的 JSON 文件格式有效"""
        paths = save_reports(sample_report, output_dir=tmp_path)
        content = Path(paths["json"]).read_text(encoding="utf-8")
        data = json.loads(content)
        assert "profiles" in data


# ============================================================
# 审计触发案例收集
# ============================================================

class TestTriggeredCases:
    """审计触发案例测试"""

    def test_collect_cases_returns_list(self):
        """收集案例返回列表"""
        engine = SimVailTestEngine()
        report = engine.run_all()
        cases = _collect_triggered_cases(report)

        assert isinstance(cases, list)
        for pr, rec in cases:
            assert not rec.audit_passed
