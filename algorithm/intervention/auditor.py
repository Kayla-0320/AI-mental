"""
五轴安全审计器 —— LLM 输出的安全审计中间件

五个审计维度：
    1. 危机升级延迟（阈值：延迟 > 2 轮对话）
    2. 妄想强化（阈值：置信度 > 0.7）
    3. 污名化与拒绝（触发词：诊断性标签）
    4. 谄媚倾向（阈值：同意率 > 90%）
    5. 轨迹漂移（阈值：偏离度 > 0.5）

每个轴返回 AuditResult（通过/不通过 + 理由）。
不通过时执行：重写回复 / 注入危机资源卡 / 升级至人工。

审计规则配置驱动，从 config/audit_rules.yaml 加载。
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from shared.dataclasses import (
    AuditAction,
    AuditAxis,
    AuditResult,
    AuditVerdict,
)


# ============================================================
# 配置加载
# ============================================================

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "audit_rules.yaml"


@functools.lru_cache(maxsize=1)
def load_audit_config(config_path: Optional[str] = None) -> dict:
    """加载审计规则配置

    Args:
        config_path: 配置文件路径（默认使用项目配置）

    Returns:
        配置字典
    """
    path = Path(config_path) if config_path else _CONFIG_PATH
    if not path.exists():
        # 返回默认配置
        return _get_default_config()

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_default_config() -> dict:
    """默认审计配置"""
    return {
        "crisis_delay": {
            "enabled": True,
            "max_turns_without_escalation": 2,
            "action_on_fail": "escalate_human",
        },
        "delusion_reinforcement": {
            "enabled": True,
            "confidence_threshold": 0.7,
            "action_on_fail": "rewrite",
            "keywords": ["被监控", "被下毒", "外星人", "超能力", "被跟踪"],
        },
        "stigma_rejection": {
            "enabled": True,
            "action_on_fail": "rewrite",
            "diagnostic_labels": ["你患有", "你得了", "你确诊", "你疯了"],
            "stigmatizing_terms": ["疯子", "神经病", "矫情", "太脆弱"],
        },
        "sycophancy": {
            "enabled": True,
            "agreement_rate_threshold": 0.9,
            "window_size": 10,
            "action_on_fail": "rewrite",
            "agreement_markers": ["你说得对", "完全正确", "我完全同意"],
        },
        "trajectory_drift": {
            "enabled": True,
            "deviation_threshold": 0.5,
            "action_on_fail": "rewrite",
            "topic_keywords": ["情绪", "感受", "想法", "压力", "焦虑"],
        },
        "crisis_resource_card": {
            "enabled": True,
            "hotlines": [
                {"name": "全国24小时心理援助热线", "number": "400-161-9995"},
            ],
            "message": "你的安全对我们很重要。",
        },
        "global": {
            "max_rewrite_attempts": 2,
            "enable_audit_log": True,
        },
    }


# ============================================================
# 审计上下文
# ============================================================

@dataclass
class AuditContext:
    """审计上下文 —— 包含审计所需的全部信息

    Attributes:
        llm_output: LLM 生成的回复文本
        dialog_state: 当前对话状态（来自状态机）
        turn_count: 当前对话轮次
        crisis_turns: 处于 CRISIS 状态的轮次数
        risk_level: 当前风险等级
        conversation_history: 对话历史（用于谄媚/漂移检测）
        llm_confidence: LLM 输出的置信度（用于妄想检测）
        current_topic: 当前治疗话题（用于漂移检测）
    """
    llm_output: str
    dialog_state: str = "INIT"
    turn_count: int = 0
    crisis_turns: int = 0
    risk_level: str = "low"
    conversation_history: list[dict] = field(default_factory=list)
    llm_confidence: float = 0.5
    current_topic: str = ""


# ============================================================
# 五轴审计器
# ============================================================

class SafetyAuditor:
    """五轴安全审计器

    对 LLM 输出进行五个维度的安全审计，
    任一轴不通过则触发相应处理动作。
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or load_audit_config()

    def audit(self, context: AuditContext) -> AuditVerdict:
        """执行五轴审计

        Args:
            context: 审计上下文

        Returns:
            AuditVerdict: 综合审计裁决
        """
        results: list[AuditResult] = []

        # 轴 1：危机升级延迟
        if self.config.get("crisis_delay", {}).get("enabled", True):
            results.append(self._audit_crisis_delay(context))

        # 轴 2：妄想强化
        if self.config.get("delusion_reinforcement", {}).get("enabled", True):
            results.append(self._audit_delusion_reinforcement(context))

        # 轴 3：污名化与拒绝
        if self.config.get("stigma_rejection", {}).get("enabled", True):
            results.append(self._audit_stigma_rejection(context))

        # 轴 4：谄媚倾向
        if self.config.get("sycophancy", {}).get("enabled", True):
            results.append(self._audit_sycophancy(context))

        # 轴 5：轨迹漂移
        if self.config.get("trajectory_drift", {}).get("enabled", True):
            results.append(self._audit_trajectory_drift(context))

        # 综合裁决
        all_passed = all(r.passed for r in results)
        final_action = self._determine_final_action(results)

        return AuditVerdict(
            passed=all_passed,
            results=results,
            final_action=final_action,
        )

    # --------------------------------------------------------
    # 轴 1：危机升级延迟
    # --------------------------------------------------------
    def _audit_crisis_delay(self, context: AuditContext) -> AuditResult:
        """审计危机升级延迟

        临床依据：NICE Guidelines (2011) 指出危机识别后应在首次接触时
                  即启动安全评估，延迟超过 2 轮对话可能导致错过干预窗口。

        规则：如果对话处于 CRISIS 状态且超过 max_turns 轮未升级 → 不通过
        """
        rule = self.config.get("crisis_delay", {})
        max_turns = rule.get("max_turns_without_escalation", 2)

        # 检查：CRISIS 状态 + 超过阈值轮次
        if context.dialog_state == "CRISIS" and context.crisis_turns > max_turns:
            return AuditResult(
                axis=AuditAxis.CRISIS_DELAY,
                passed=False,
                reason=f"危机状态已持续 {context.crisis_turns} 轮未升级（阈值: {max_turns}）",
                suggested_action=AuditAction.ESCALATE_HUMAN,
            )

        return AuditResult(
            axis=AuditAxis.CRISIS_DELAY,
            passed=True,
        )

    # --------------------------------------------------------
    # 轴 2：妄想强化
    # --------------------------------------------------------
    def _audit_delusion_reinforcement(self, context: AuditContext) -> AuditResult:
        """审计妄想强化

        临床依据：CBT for psychosis (CBTp) 研究表明，治疗师不应强化患者的
                  妄想信念。置信度 > 0.7 表示 LLM 过度认同患者的非理性信念。
                  来源：Morrison (2017) CBTp 治疗手册。

        规则：如果 LLM 输出包含妄想关键词 + 置信度 > 阈值 → 不通过
        """
        rule = self.config.get("delusion_reinforcement", {})
        threshold = rule.get("confidence_threshold", 0.7)
        keywords = rule.get("keywords", [])

        # 检查：LLM 输出是否包含妄想关键词
        contains_delusion_keyword = any(kw in context.llm_output for kw in keywords)

        # 检查：LLM 置信度是否过高（表示过度认同）
        if contains_delusion_keyword and context.llm_confidence > threshold:
            return AuditResult(
                axis=AuditAxis.DELUSION_REINFORCEMENT,
                passed=False,
                reason=f"LLM 输出包含妄想关键词且置信度过高 ({context.llm_confidence:.2f} > {threshold})",
                suggested_action=AuditAction.REWRITE,
            )

        return AuditResult(
            axis=AuditAxis.DELUSION_REINFORCEMENT,
            passed=True,
        )

    # --------------------------------------------------------
    # 轴 3：污名化与拒绝
    # --------------------------------------------------------
    def _audit_stigma_rejection(self, context: AuditContext) -> AuditResult:
        """审计污名化与拒绝

        临床依据：WHO 心理健康去污名化指南强调，不应使用诊断性标签定义个人。
                  "你是抑郁症" vs "你正在经历抑郁情绪"——前者固化身份，后者承认体验。
                  来源：WHO QualityRights (2019)。

        规则：如果 LLM 输出包含诊断性标签或污名化用语 → 不通过
        """
        rule = self.config.get("stigma_rejection", {})
        diagnostic_labels = rule.get("diagnostic_labels", [])
        stigmatizing_terms = rule.get("stigmatizing_terms", [])

        # 检查诊断性标签
        for label in diagnostic_labels:
            if label in context.llm_output:
                return AuditResult(
                    axis=AuditAxis.STIGMA_REJECTION,
                    passed=False,
                    reason=f"LLM 输出包含诊断性标签: '{label}'",
                    suggested_action=AuditAction.REWRITE,
                )

        # 检查污名化用语
        for term in stigmatizing_terms:
            if term in context.llm_output:
                return AuditResult(
                    axis=AuditAxis.STIGMA_REJECTION,
                    passed=False,
                    reason=f"LLM 输出包含污名化用语: '{term}'",
                    suggested_action=AuditAction.REWRITE,
                )

        return AuditResult(
            axis=AuditAxis.STIGMA_REJECTION,
            passed=True,
        )

    # --------------------------------------------------------
    # 轴 4：谄媚倾向
    # --------------------------------------------------------
    def _audit_sycophancy(self, context: AuditContext) -> AuditResult:
        """审计谄媚倾向

        临床依据：动机式访谈 (MI) 原则要求治疗师保持真诚，过度同意会破坏
                  治疗联盟。同意率 > 90% 表明 LLM 在谄媚而非真诚共情。
                  来源：Miller & Rollnick (2013) Motivational Interviewing。

        规则：在最近 window_size 轮中，如果同意标记出现率 > 阈值 → 不通过
        """
        rule = self.config.get("sycophancy", {})
        threshold = rule.get("agreement_rate_threshold", 0.9)
        window_size = rule.get("window_size", 10)
        markers = rule.get("agreement_markers", [])

        # 获取最近 N 轮对话
        recent_history = context.conversation_history[-window_size:]
        if not recent_history:
            return AuditResult(axis=AuditAxis.SYCOPHANCY, passed=True)

        # 统计同意标记出现次数
        agreement_count = 0
        for turn in recent_history:
            content = turn.get("content", "")
            if any(marker in content for marker in markers):
                agreement_count += 1

        # 计算同意率
        agreement_rate = agreement_count / len(recent_history) if recent_history else 0

        if agreement_rate > threshold:
            return AuditResult(
                axis=AuditAxis.SYCOPHANCY,
                passed=False,
                reason=f"同意率过高 ({agreement_rate:.1%} > {threshold:.1%})",
                suggested_action=AuditAction.REWRITE,
            )

        return AuditResult(
            axis=AuditAxis.SYCOPHANCY,
            passed=True,
        )

    # --------------------------------------------------------
    # 轴 5：轨迹漂移
    # --------------------------------------------------------
    def _audit_trajectory_drift(self, context: AuditContext) -> AuditResult:
        """审计轨迹漂移

        临床依据：心理咨询需要保持治疗方向。偏离度 > 0.5 表示对话已偏离
                  预设的治疗目标（如从 CBT 认知重构漂移到无关闲聊）。
                  来源：Lambert (2013) 心理治疗统一方案。

        规则：如果 LLM 输出与当前治疗话题的关键词重合度 < 阈值 → 不通过
        """
        rule = self.config.get("trajectory_drift", {})
        threshold = rule.get("deviation_threshold", 0.5)
        topic_keywords = rule.get("topic_keywords", [])

        if not topic_keywords or not context.current_topic:
            # 无话题信息时跳过此轴
            return AuditResult(axis=AuditAxis.TRAJECTORY_DRIFT, passed=True)

        # 计算 LLM 输出与话题关键词的重合度
        output_lower = context.llm_output.lower()
        keyword_matches = sum(1 for kw in topic_keywords if kw in output_lower)
        relevance_score = keyword_matches / len(topic_keywords) if topic_keywords else 0

        # 偏离度 = 1 - 相关度
        deviation = 1.0 - relevance_score

        if deviation > threshold:
            return AuditResult(
                axis=AuditAxis.TRAJECTORY_DRIFT,
                passed=False,
                reason=f"轨迹偏离度过高 ({deviation:.2f} > {threshold})",
                suggested_action=AuditAction.REWRITE,
            )

        return AuditResult(
            axis=AuditAxis.TRAJECTORY_DRIFT,
            passed=True,
        )

    # --------------------------------------------------------
    # 综合裁决
    # --------------------------------------------------------
    def _determine_final_action(self, results: list[AuditResult]) -> AuditAction:
        """确定最终处理动作（取最严重的）

        优先级：ESCALATE_HUMAN > INJECT_RESOURCE > REWRITE > PASS
        """
        action_priority = {
            AuditAction.PASS: 0,
            AuditAction.REWRITE: 1,
            AuditAction.INJECT_RESOURCE: 2,
            AuditAction.ESCALATE_HUMAN: 3,
        }

        max_priority = 0
        final_action = AuditAction.PASS

        for result in results:
            if not result.passed:
                priority = action_priority.get(result.suggested_action, 0)
                if priority > max_priority:
                    max_priority = priority
                    final_action = result.suggested_action

        return final_action


# ============================================================
# 危机资源卡生成
# ============================================================

def generate_crisis_resource_card(config: Optional[dict] = None) -> str:
    """生成危机资源卡文本

    Args:
        config: 审计配置（可选）

    Returns:
        危机资源卡文本
    """
    cfg = config or load_audit_config()
    card_config = cfg.get("crisis_resource_card", {})

    if not card_config.get("enabled", True):
        return ""

    lines = [card_config.get("message", "你的安全对我们很重要。")]
    lines.append("")

    for hotline in card_config.get("hotlines", []):
        lines.append(f"- {hotline['name']}: {hotline['number']}")

    return "\n".join(lines)


# ============================================================
# 便捷函数
# ============================================================

def audit_llm_output(
    llm_output: str,
    dialog_state: str = "INIT",
    turn_count: int = 0,
    crisis_turns: int = 0,
    risk_level: str = "low",
    conversation_history: Optional[list[dict]] = None,
    llm_confidence: float = 0.5,
    current_topic: str = "",
) -> AuditVerdict:
    """审计 LLM 输出的便捷函数

    Args:
        llm_output: LLM 生成的回复
        dialog_state: 当前对话状态
        turn_count: 当前轮次
        crisis_turns: CRISIS 状态持续轮次
        risk_level: 风险等级
        conversation_history: 对话历史
        llm_confidence: LLM 置信度
        current_topic: 当前治疗话题

    Returns:
        AuditVerdict: 综合审计裁决
    """
    context = AuditContext(
        llm_output=llm_output,
        dialog_state=dialog_state,
        turn_count=turn_count,
        crisis_turns=crisis_turns,
        risk_level=risk_level,
        conversation_history=conversation_history or [],
        llm_confidence=llm_confidence,
        current_topic=current_topic,
    )

    auditor = SafetyAuditor()
    return auditor.audit(context)
