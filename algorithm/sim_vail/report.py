"""
SIM-VAIL 测试报告生成器

输出格式：
  - JSON: 结构化数据，供前端面板消费
  - Markdown: 人类可读报告，含风险热力图（ASCII）

报告内容：
  - 每个画像的拦截率、漏判率、误报率
  - 风险热力图（每轮对话的风险分布可视化）
  - 所有触发审计的案例，附具体对话内容
  - 总体汇总指标
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from sim_vail import (
    SimVailReport,
    ProfileTestResult,
    AuditRecord,
    ComparisonReport,
    OUTPUT_DIR,
)


# ============================================================
# JSON 报告
# ============================================================

def generate_json_report(report: SimVailReport) -> str:
    """生成 JSON 格式测试报告

    Args:
        report: SIM-VAIL 测试报告

    Returns:
        JSON 字符串
    """
    data = {
        "report_id": report.run_id,
        "timestamp": report.timestamp,
        "timestamp_human": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(report.timestamp)
        ),
        "summary": {
            "total_profiles": report.total_profiles,
            "total_turns": report.total_turns,
            "interception_rate": round(report.overall_interception_rate, 4),
            "miss_rate": round(report.overall_miss_rate, 4),
            "false_alarm_rate": round(report.overall_false_alarm_rate, 4),
            "avg_likert_score": round(report.avg_likert_score, 2),
        },
        "profiles": [],
    }

    for pr in report.profile_results:
        profile_data = {
            "profile_id": pr.profile_id,
            "profile_name": pr.profile_name,
            "total_turns": pr.total_turns,
            "intercepted": pr.interception_count,
            "missed": pr.miss_count,
            "false_alarms": pr.false_alarm_count,
            "avg_likert": round(pr.avg_likert, 2),
            "max_escalation": round(pr.max_escalation_reached, 2),
            "turns": [],
        }

        for rec in pr.audit_records:
            turn_data = {
                "turn_idx": rec.turn_idx,
                "user_message": rec.user_message,
                "system_response": rec.system_response,
                "audit_passed": rec.audit_passed,
                "triggered_axes": rec.triggered_axes,
                "severity": rec.severity,
                "action_taken": rec.action_taken,
                "risk_state": rec.risk_state,
                "likert_score": rec.likert_score,
            }
            profile_data["turns"].append(turn_data)

        data["profiles"].append(profile_data)

    return json.dumps(data, ensure_ascii=False, indent=2)


# ============================================================
# Markdown 报告
# ============================================================

def generate_markdown_report(report: SimVailReport) -> str:
    """生成 Markdown 格式测试报告

    Args:
        report: SIM-VAIL 测试报告

    Returns:
        Markdown 字符串
    """
    lines = [
        "# SIM-VAIL 动态测试报告",
        "",
        f"> 运行 ID: `{report.run_id}`",
        f"> 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}",
        f"> 画像数: {report.total_profiles} | 总轮次: {report.total_turns}",
        "",
        "---",
        "",
        "## 总体指标",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| 危机拦截率 | **{report.overall_interception_rate:.1%}** |",
        f"| 漏判率 | **{report.overall_miss_rate:.1%}** |",
        f"| 误报率 | **{report.overall_false_alarm_rate:.1%}** |",
        f"| 平均 Likert 评分 | **{report.avg_likert_score:.1f}** / 5.0 |",
        "",
        "---",
        "",
        "## 风险热力图",
        "",
        "每轮对话的风险分布（按画像 × 轮次矩阵展示）：",
        "",
    ]

    # 风险热力图（ASCII 矩阵）
    lines.extend(_build_heatmap(report))

    lines.extend([
        "",
        "---",
        "",
        "## 各画像详细结果",
        "",
    ])

    # 各画像详细结果
    for pr in report.profile_results:
        lines.extend(_build_profile_section(pr))

    # 触发审计的案例列表
    lines.extend([
        "---",
        "",
        "## 触发审计的案例",
        "",
    ])

    triggered_cases = _collect_triggered_cases(report)
    if triggered_cases:
        lines.extend([
            f"共 {len(triggered_cases)} 次审计触发：",
            "",
            "| # | 画像 | 轮次 | 用户消息 | 触发轴 | 严重程度 |",
            "|---|------|------|----------|--------|----------|",
        ])
        for i, (pr, rec) in enumerate(triggered_cases, 1):
            msg = rec.user_message[:25] + ("..." if len(rec.user_message) > 25 else "")
            axes = ", ".join(rec.triggered_axes) if rec.triggered_axes else "-"
            lines.append(
                f"| {i} | {pr.profile_name} | {rec.turn_idx} | "
                f"{msg} | {axes} | {rec.severity} |"
            )
    else:
        lines.append("无审计触发案例。")

    lines.extend([
        "",
        "---",
        "",
        "## 临床依据",
        "",
        "- SIM-VAIL: Llorens-de-Dios et al. (2024) Nature Medicine",
        "- C-SSRS: Columbia-Suicide Severity Rating Scale",
        "- PHQ-9: Patient Health Questionnaire-9 (Adolescent)",
        "- GAD-7: Generalized Anxiety Disorder-7",
        "- NICE CG185: Self-harm and suicide prevention",
        "- NICE CG26: Depression in children and young people",
        "",
        "> 本报告由自动化测试框架生成，模拟用户为 Prompt 工程实现，",
        "> 不涉及真实青少年参与。测试结果仅供系统评估参考。",
        "",
    ])

    return "\n".join(lines)


def _build_heatmap(report: SimVailReport) -> list[str]:
    """构建风险热力图（ASCII 矩阵）

    使用符号标注每轮对话的风险状态：
      · = normal (正常)
      ~ = at_risk (有风险)
      ! = crisis (危机)
      + = recovering (恢复中)
    """
    state_symbols = {
        "normal": "·",
        "at_risk": "~",
        "crisis": "!",
        "recovering": "+",
    }

    lines = ["```"]

    # 表头
    max_turns = max(
        (pr.total_turns for pr in report.profile_results), default=0
    )
    turn_header = "     | " + " ".join(f"T{i}" for i in range(max_turns))
    lines.append(turn_header)
    lines.append("     | " + "---" * max_turns)

    for pr in report.profile_results:
        name = pr.profile_name[:6].ljust(6)
        row = f"{name} | "
        for rec in pr.audit_records:
            symbol = state_symbols.get(rec.risk_state, "?")
            # 如果被审计拦截，用大写
            if not rec.audit_passed:
                symbol = symbol.upper()
            row += f" {symbol} "
        # 填充空位
        for _ in range(max_turns - pr.total_turns):
            row += "   "
        lines.append(row)

    lines.extend([
        "```",
        "",
        "图例: ·=正常 ~=风险 !=危机 +=恢复 | 大写=被审计拦截",
    ])

    return lines


def _build_profile_section(pr: ProfileTestResult) -> list[str]:
    """构建单个画像的详细结果段落"""
    lines = [
        f"### {pr.profile_name} (`{pr.profile_id}`)",
        "",
        f"- 总轮次: {pr.total_turns}",
        f"- 正确拦截: {pr.interception_count}",
        f"- 漏判: {pr.miss_count}",
        f"- 误报: {pr.false_alarm_count}",
        f"- 平均 Likert: {pr.avg_likert:.1f}",
        f"- 最高升级程度: {pr.max_escalation_reached:.2f}",
        "",
        "| 轮次 | 风险状态 | 审计 | 用户消息 | 系统回复 |",
        "|------|----------|------|----------|----------|",
    ]

    for rec in pr.audit_records:
        audit_mark = "PASS" if rec.audit_passed else f"FAIL({rec.severity})"
        user_msg = rec.user_message[:30] + ("..." if len(rec.user_message) > 30 else "")
        sys_msg = rec.system_response[:30] + ("..." if len(rec.system_response) > 30 else "")
        lines.append(
            f"| {rec.turn_idx} | {rec.risk_state} | {audit_mark} | "
            f"{user_msg} | {sys_msg} |"
        )

    lines.append("")
    return lines


def _collect_triggered_cases(
    report: SimVailReport,
) -> list[tuple[ProfileTestResult, AuditRecord]]:
    """收集所有触发审计的案例"""
    cases = []
    for pr in report.profile_results:
        for rec in pr.audit_records:
            if not rec.audit_passed:
                cases.append((pr, rec))
    return cases


# ============================================================
# 报告保存
# ============================================================

def save_reports(
    report: SimVailReport,
    output_dir: Optional[Path] = None,
) -> dict[str, str]:
    """保存 JSON + Markdown 双格式报告到文件

    Args:
        report: SIM-VAIL 测试报告
        output_dir: 输出目录（默认 sim_vail/outputs/）

    Returns:
        dict: {"json": json_path, "markdown": md_path}
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"simvail_{report.run_id}_{timestamp_str}"

    # JSON 报告
    json_content = generate_json_report(report)
    json_path = out_dir / f"{base_name}.json"
    json_path.write_text(json_content, encoding="utf-8")

    # Markdown 报告
    md_content = generate_markdown_report(report)
    md_path = out_dir / f"{base_name}.md"
    md_path.write_text(md_content, encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
    }


# ============================================================
# 对比实验报告
# ============================================================

def generate_comparison_report(report: ComparisonReport) -> str:
    """生成基线驱动对比实验 Markdown 报告

    报告内容：
      1. 每个画像在两种模式下的指标对比表格
      2. 汇总：基线驱动整体的提升幅度
      3. 基线驱动独有的拦截案例

    Args:
        report: 对比实验报告

    Returns:
        Markdown 字符串
    """
    lines = [
        "# SIM-VAIL 基线驱动审计对比报告",
        "",
        f"> 运行 ID: `{report.run_id}`",
        f"> 时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}",
        f"> 画像数: {len(report.profile_comparisons)}",
        "",
        "---",
        "",
        "## 总体汇总",
        "",
        "| 指标 | 基线驱动 | 无基线 | 提升幅度 |",
        "|------|----------|--------|----------|",
        f"| 危机拦截率 | **{report.overall_baseline_interception_rate:.1%}** | "
        f"**{report.overall_no_baseline_interception_rate:.1%}** | "
        f"{'+' if report.overall_improvement >= 0 else ''}{report.overall_improvement:.1%} |",
        "",
        f"基线驱动独有拦截案例总数: **{report.total_unique_interceptions}**",
        "",
        "---",
        "",
        "## 各画像对比详情",
        "",
    ]

    for pc in report.profile_comparisons:
        lines.extend(_build_comparison_profile_section(pc))

    # 基线驱动独有的拦截案例
    lines.extend([
        "---",
        "",
        "## 基线驱动独有的拦截案例",
        "",
        "以下案例在基线驱动模式下被成功拦截，但在无基线模式下漏判：",
        "",
    ])

    has_unique = False
    for pc in report.profile_comparisons:
        if pc.unique_interceptions:
            has_unique = True
            turns_str = ", ".join(f"T{t}" for t in pc.unique_interceptions)
            lines.append(f"- **{pc.profile_name}**: 轮次 [{turns_str}]")

    if not has_unique:
        lines.append("无独有拦截案例。")
        lines.append("")
        lines.append(
            "> 分析：MockSystemResponder 使用模板回复，"
            "审计器的五轴检测在两种模式下对相同输入的判定可能一致。"
            "基线感知的阈值调整主要在边界情况下产生差异，"
            "而模板回复的审计特征较为明显，导致两种模式结果趋同。"
            "在真实 LLM 回复场景下，基线驱动的预期提升会更显著。"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 实验说明",
        "",
        "- **基线驱动模式**: 根据画像心理维度生成模拟基线偏离，"
        "当偏离显著时审计器自动提高对应轴的敏感度（boost=1.5x）",
        "- **无基线模式**: 审计器使用默认阈值，不考虑个人基线",
        "- **可复现性**: 固定随机种子 (seed=42)，结果可复现",
        "",
        "> 本报告由自动化对比实验框架生成。",
        "> 模拟数据基于预定义画像脚本，不涉及真实青少年参与。",
        "",
    ])

    return "\n".join(lines)


def _build_comparison_profile_section(pc) -> list[str]:
    """构建单个画像的对比结果段落"""
    bl_intercept = f"{pc.baseline_interception_rate:.1%}"
    no_bl_intercept = f"{pc.no_baseline_interception_rate:.1%}"
    bl_miss = f"{pc.baseline_miss_rate:.1%}"
    no_bl_miss = f"{pc.no_baseline_miss_rate:.1%}"
    bl_fa = f"{pc.baseline_false_alarm_rate:.1%}"
    no_bl_fa = f"{pc.no_baseline_false_alarm_rate:.1%}"

    delta = pc.baseline_interception_rate - pc.no_baseline_interception_rate
    delta_str = f"{'+' if delta >= 0 else ''}{delta:.1%}"

    lines = [
        f"### {pc.profile_name} (`{pc.profile_id}`)",
        "",
        "| 指标 | 基线驱动 | 无基线 | 差异 |",
        "|------|----------|--------|------|",
        f"| 拦截率 | {bl_intercept} | {no_bl_intercept} | {delta_str} |",
        f"| 漏判率 | {bl_miss} | {no_bl_miss} | "
        f"{'+' if pc.baseline_miss_rate - pc.no_baseline_miss_rate >= 0 else ''}"
        f"{pc.baseline_miss_rate - pc.no_baseline_miss_rate:.1%} |",
        f"| 误报率 | {bl_fa} | {no_bl_fa} | "
        f"{'+' if pc.baseline_false_alarm_rate - pc.no_baseline_false_alarm_rate >= 0 else ''}"
        f"{pc.baseline_false_alarm_rate - pc.no_baseline_false_alarm_rate:.1%} |",
        "",
    ]

    if pc.unique_interceptions:
        turns_str = ", ".join(f"T{t}" for t in pc.unique_interceptions)
        lines.append(f"✅ 基线驱动独有拦截: 轮次 [{turns_str}]")
    else:
        lines.append("— 无独有拦截案例")

    lines.append("")
    return lines


def save_comparison_report(
    report: ComparisonReport,
    output_dir: Optional[Path] = None,
) -> str:
    """保存对比报告到文件

    Args:
        report: 对比实验报告
        output_dir: 输出目录（默认 sim_vail/outputs/）

    Returns:
        保存的文件路径
    """
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    md_content = generate_comparison_report(report)
    md_path = out_dir / "comparison_report.md"
    md_path.write_text(md_content, encoding="utf-8")

    return str(md_path)
