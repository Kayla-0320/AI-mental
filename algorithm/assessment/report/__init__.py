"""
评估报告生成模块 —— 整合各模块输出生成综合心理评估报告

报告内容：
    1. 基本信息摘要
    2. 多模态感知概览
    3. 量表映射结果（PHQ-9 / GAD-7 / PSS-10）
    4. 数字表型画像
    5. 纵向趋势分析
    6. 共病模式分析
    7. 风险评估与建议

输出格式：
    - Markdown 文本报告
    - 结构化 JSON 数据（供前端渲染）

核心接口：
    - generate_report(user_id, components) -> AssessmentReport
    - render_markdown(report) -> str
    - render_summary(report) -> str (简短版)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from shared.dataclasses import (
    EmotionResult,
    PhenotypeVector,
    RiskAssessment,
    RiskLevel,
)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class ReportComponent:
    """报告组件（各模块输出）

    Attributes:
        emotion_result: 感知层情绪结果
        scale_results: 量表映射结果 {量表名: ScaleResult}
        risk_result: 综合风险评估
        phenotype: 数字表型向量
        trend_results: 趋势分析 {指标名: TrendResult}
        comorbidity_result: 共病分析结果
        baseline_deviation: 基线偏离 {特征名: Z-score 信息}
    """
    emotion_result: Optional[EmotionResult] = None
    scale_results: dict = field(default_factory=dict)
    risk_result: Optional[object] = None
    phenotype: Optional[PhenotypeVector] = None
    trend_results: dict = field(default_factory=dict)
    comorbidity_result: Optional[object] = None
    baseline_deviation: dict = field(default_factory=dict)


@dataclass
class AssessmentReport:
    """综合评估报告

    Attributes:
        user_id: 用户 ID
        report_id: 报告唯一 ID
        generated_at: 生成时间
        components: 报告组件（各模块输出）
        overall_risk_level: 综合风险等级
        summary: 摘要文本
        sections: 各章节内容
    """
    user_id: str
    report_id: str = ""
    generated_at: float = 0.0
    components: Optional[ReportComponent] = None
    overall_risk_level: RiskLevel = RiskLevel.LOW
    summary: str = ""
    sections: dict[str, str] = field(default_factory=dict)


# ============================================================
# 报告章节生成
# ============================================================

def _gen_emotion_section(emotion: EmotionResult) -> str:
    """生成感知概览章节"""
    labels = ["快乐", "悲伤", "焦虑", "愤怒", "中性"]
    probs = emotion.text_emotion_probs

    lines = [
        "## 1. 多模态感知概览\n",
        f"**感知置信度**: {emotion.confidence:.1%}\n",
        "\n**情绪分布**:\n",
    ]
    for label, prob in zip(labels, probs):
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        lines.append(f"- {label}: {bar} {prob:.1%}")

    if emotion.audio_risk_prob is not None:
        lines.append(f"\n**语音风险概率**: {emotion.audio_risk_prob:.1%}")

    if emotion.evidence:
        lines.append("\n**分析证据**:")
        for e in emotion.evidence:
            lines.append(f"- {e}")

    return "\n".join(lines)


def _gen_scale_section(scale_results: dict) -> str:
    """生成量表映射章节"""
    if not scale_results:
        return "## 2. 临床量表映射\n\n暂无量表数据"

    lines = ["## 2. 临床量表映射\n"]

    for name, result in scale_results.items():
        lines.append(f"### {name}")
        lines.append(f"- **总分**: {result.total_score:.0f} / {result.max_score}")
        lines.append(f"- **严重程度**: {result.severity}")
        ci = result.confidence_interval
        lines.append(f"- **置信区间**: [{ci[0]:.1f}, {ci[1]:.1f}]")

        if hasattr(result, 'items') and result.items:
            lines.append(f"\n**条目详情**:")
            for item in result.items:
                score_bar = "●" * item.score + "○" * (3 - item.score)
                lines.append(f"  {item.item_number}. {item.item_text}: [{score_bar}] {item.score}/3")
        lines.append("")

    return "\n".join(lines)


def _gen_phenotype_section(phenotype: PhenotypeVector) -> str:
    """生成数字表型章节"""
    lines = [
        "## 3. 数字表型画像\n",
        f"**时间窗口**: {phenotype.time_window_days} 天\n",
    ]

    categories = [
        ("睡眠特征", phenotype.sleep_features),
        ("情感特征", phenotype.emotion_features),
        ("行为特征", phenotype.behavior_features),
        ("测评特征", phenotype.assessment_features),
        ("生理特征", phenotype.physiological_features),
    ]

    for cat_name, features in categories:
        if not features:
            continue
        lines.append(f"\n### {cat_name}")
        for f in features:
            if f.value is not None:
                change_mark = " ⚠️" if f.significant_change else ""
                lines.append(f"- {f.name}: **{f.value}** {f.unit} (置信度 {f.confidence:.0%}){change_mark}")
            else:
                lines.append(f"- {f.name}: *数据缺失*")

    missing = phenotype.missing_features()
    if missing:
        lines.append(f"\n> ⚠️ 缺失特征: {', '.join(missing)}")

    return "\n".join(lines)


def _gen_trend_section(trend_results: dict) -> str:
    """生成趋势分析章节"""
    if not trend_results:
        return "## 4. 纵向趋势分析\n\n暂无趋势数据"

    lines = ["## 4. 纵向趋势分析\n"]

    direction_labels = {
        "improving": "📈 改善",
        "stable": "➡️ 稳定",
        "worsening": "📉 恶化",
        "volatile": "📊 波动",
        "insufficient": "⏳ 数据不足",
    }

    for name, trend in trend_results.items():
        dir_label = direction_labels.get(trend.direction.value, trend.direction.value)
        lines.append(f"### {name}")
        lines.append(f"- **趋势**: {dir_label}")
        lines.append(f"- **斜率**: {trend.slope:.4f} /天")
        lines.append(f"- **R²**: {trend.r_squared:.2f}")
        lines.append(f"- **显著性**: {'是' if trend.is_significant else '否'}")

        if trend.change_points:
            lines.append(f"- **变化点**: {len(trend.change_points)} 个")
            for cp in trend.change_points[:3]:
                lines.append(f"  - 第 {cp.index} 个数据点: {cp.before_mean:.1f} → {cp.after_mean:.1f} "
                             f"(d={cp.effect_size:.2f}, {cp.severity.value})")
        lines.append("")

    return "\n".join(lines)


def _gen_risk_section(risk_result) -> str:
    """生成风险评估章节"""
    if risk_result is None:
        return "## 5. 综合风险评估\n\n暂无风险评估数据"

    level_labels = {
        RiskLevel.LOW: "🟢 低风险",
        RiskLevel.MEDIUM: "🟡 中风险",
        RiskLevel.HIGH: "🟠 高风险",
        RiskLevel.CRISIS: "🔴 危机",
    }

    lines = [
        "## 5. 综合风险评估\n",
        f"**风险等级**: {level_labels.get(risk_result.risk_level, str(risk_result.risk_level))}",
        f"**风险评分**: {risk_result.risk_score:.2f} / 1.00",
        f"**置信度**: {risk_result.confidence:.1%}",
        f"**输入信号**: {len(risk_result.signals)} 条\n",
    ]

    if hasattr(risk_result, 'phq9_range') and risk_result.phq9_range != (0.0, 0.0):
        lines.append(f"- PHQ-9 预估: {risk_result.phq9_range[0]:.0f}-{risk_result.phq9_range[1]:.0f}")
    if hasattr(risk_result, 'gad7_range') and risk_result.gad7_range != (0.0, 0.0):
        lines.append(f"- GAD-7 预估: {risk_result.gad7_range[0]:.0f}-{risk_result.gad7_range[1]:.0f}")

    if hasattr(risk_result, 'recommendations') and risk_result.recommendations:
        lines.append("\n### 临床建议")
        for rec in risk_result.recommendations:
            lines.append(f"- {rec}")

    return "\n".join(lines)


# ============================================================
# 核心接口
# ============================================================

def generate_report(
    user_id: str,
    components: ReportComponent,
) -> AssessmentReport:
    """生成综合评估报告

    整合各模块输出，生成结构化报告。

    Args:
        user_id: 用户 ID
        components: 各模块输出汇总

    Returns:
        AssessmentReport: 综合评估报告
    """
    import uuid
    report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"

    # 确定综合风险等级
    risk_level = RiskLevel.LOW
    if components.risk_result is not None:
        risk_level = components.risk_result.risk_level

    # 生成各章节
    sections = {}

    if components.emotion_result:
        sections["emotion"] = _gen_emotion_section(components.emotion_result)

    if components.scale_results:
        sections["scale"] = _gen_scale_section(components.scale_results)

    if components.phenotype:
        sections["phenotype"] = _gen_phenotype_section(components.phenotype)

    if components.trend_results:
        sections["trend"] = _gen_trend_section(components.trend_results)

    if components.risk_result:
        sections["risk"] = _gen_risk_section(components.risk_result)

    # 生成摘要
    summary = _generate_summary(user_id, risk_level, components)

    return AssessmentReport(
        user_id=user_id,
        report_id=report_id,
        generated_at=time.time(),
        components=components,
        overall_risk_level=risk_level,
        summary=summary,
        sections=sections,
    )


def _generate_summary(
    user_id: str,
    risk_level: RiskLevel,
    components: ReportComponent,
) -> str:
    """生成简短摘要"""
    level_cn = {
        RiskLevel.LOW: "低风险",
        RiskLevel.MEDIUM: "中风险",
        RiskLevel.HIGH: "高风险",
        RiskLevel.CRISIS: "危机",
    }

    parts = [f"用户 {user_id} 的综合心理评估结果为 **{level_cn[risk_level]}**。"]

    if components.emotion_result:
        probs = components.emotion_result.text_emotion_probs
        labels = ["快乐", "悲伤", "焦虑", "愤怒", "中性"]
        dominant = labels[probs.index(max(probs))]
        parts.append(f"主导情绪为「{dominant}」。")

    if components.scale_results:
        for name, result in components.scale_results.items():
            parts.append(f"{name} 预估 {result.total_score:.0f} 分（{result.severity}）。")

    if components.risk_result and hasattr(components.risk_result, 'recommendations'):
        top_recs = components.risk_result.recommendations[:2]
        if top_recs:
            parts.append("建议：" + "；".join(top_recs))

    return " ".join(parts)


def render_markdown(report: AssessmentReport) -> str:
    """渲染完整 Markdown 报告

    Args:
        report: 评估报告

    Returns:
        完整 Markdown 文本
    """
    lines = [
        f"# 心理评估报告",
        f"",
        f"**报告编号**: {report.report_id}",
        f"**用户 ID**: {report.user_id}",
        f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.generated_at))}",
        f"**综合风险等级**: {report.overall_risk_level.value}",
        f"",
        f"---",
        f"",
        f"## 摘要",
        f"",
        report.summary,
        f"",
        f"---",
        f"",
    ]

    # 按顺序输出各章节
    section_order = ["emotion", "scale", "phenotype", "trend", "risk"]
    for key in section_order:
        if key in report.sections:
            lines.append(report.sections[key])
            lines.append("")
            lines.append("---")
            lines.append("")

    lines.append("*本报告由 AI 系统自动生成，仅供临床参考，不能替代专业诊断。*")

    return "\n".join(lines)


def render_summary(report: AssessmentReport) -> str:
    """渲染简短摘要（用于咨询师端快速查看）

    Args:
        report: 评估报告

    Returns:
        简短摘要文本
    """
    level_emoji = {
        RiskLevel.LOW: "🟢",
        RiskLevel.MEDIUM: "🟡",
        RiskLevel.HIGH: "🟠",
        RiskLevel.CRISIS: "🔴",
    }

    return (
        f"{level_emoji.get(report.overall_risk_level, '⚪')} "
        f"**{report.overall_risk_level.value}** | "
        f"{report.summary}"
    )
