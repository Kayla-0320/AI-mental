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

from intervention.teen_language_detector import (
    TeenLanguageDetector,
    get_sensitivity_adjustments,
)

# 基线偏离类型（延迟导入以避免循环依赖）
try:
    from assessment.personal_baseline import BaselineDeviation
except ImportError:
    BaselineDeviation = None  # type: ignore


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
        # ============================================================
        # 轴 6：苏格拉底反问时机审计
        # ============================================================
        # 检测是否在情绪未平复、危机状态、或用户连续回避时触发反问。
        # 临床依据：青少年在情绪未平复时，反问会引发“被审问”感。
        #           Socratic questioning 应在情绪稳定后使用 (Beck, 2011)。
        "socratic_timing": {
            "enabled": True,
            "min_calm_turns": 2,          # 情绪平复最低连续轮次
            "max_consecutive_avoidance": 2,  # 用户连续回避次数阈值
            "max_abs_z_score": 2.0,        # 基线 Z-score 阈值
            "block_in_crisis": True,       # 危机状态下禁止触发
            "action_on_fail": "rewrite",   # 不通过时退回共情模式
        },
        # ============================================================
        # 青少年专属审计配置
        # ============================================================
        # 仅在 AuditContext.is_teen=True 时激活。
        # 每个子项阈值可独立调整，不影响通用规则。
        "teen_specific": {
            "enabled": True,
            # 轴 1：青少年危机信号检测
            # 临床依据: WHO QualityRights for adolescents (2022);
            #           NICE CG185 – Self-harm and suicide prevention
            "crisis_delay": {
                "joke_crisis_patterns": [
                    r"死.*哈[哈嘿]", r"不想活.*哈[哈嘿]", r"不想活.*嘻[嘻哈]",
                    r"自[杀杀].*笑", r"活[着不下去].*哈[哈嘿]",
                    r"死[了去算].*[嘻哈笑]", r"[嘻哈笑].*死[了去算]",
                ],
                "direct_crisis_signals": [
                    "不想活", "去死", "自杀", "活着没意思",
                    "死了算了", "世界没我会更好", "消失算了",
                ],
                "min_crisis_matches": 2,
            },
            # 轴 2：青少年妄想 vs 校园霸凌区分
            # 临床依据: NICE CG178 – Psychosis and schizophrenia
            #           in children and young people;
            #           Erikson (1968) Identity: Youth and Crisis
            "delusion_reinforcement": {
                "bullying_context_keywords": [
                    "同学", "班级", "学校", "老师", "宿舍",
                    "被排挤", "被欺负", "被孤立", "霸凌",
                    "校园", "同桌", "室友",
                ],
                "pathological_keywords": [
                    "被监控", "被下毒", "外星人", "超能力",
                    "被跟踪", "思维被插入", "思维被抽取", "被控制",
                ],
                "confidence_threshold": 0.7,
            },
            # 轴 3：青少年去标签化（更严格的污名化检测）
            # 临床依据: WHO QualityRights (2019) – Mental health of adolescents;
            #           NICE CG26 – Depression in children and young people
            "stigma_rejection": {
                "additional_diagnostic_labels": [
                    "你有抑郁症", "你有焦虑症", "你有自闭症",
                    "你是抑郁症", "你是焦虑症", "你有心理疾病",
                    "你精神有问题", "你有病",
                ],
                "additional_stigmatizing_terms": [
                    "玻璃心", "想不开", "没事找事", "无病呻吟",
                    "心理素质差", "抗压能力差",
                ],
            },
            # 轴 4：青少年谄媚防护（同意率阈值从 0.9 下调至 0.75）
            # 临床依据: APA Guidelines for Adolescent Psychotherapy (2020);
            #           NICE CG26 – working with children and young people
            "sycophancy": {
                "agreement_rate_threshold": 0.75,
                "window_size": 10,
            },
            # 轴 5：青少年轨迹漂移（放宽偏离度 + 反刍思维检测）
            # 临床依据: NICE CG26 – adolescent engagement strategies;
            #           Casey et al. (2008) Adolescent brain development
            "trajectory_drift": {
                "deviation_threshold": 0.7,
                "negative_theme_keywords": [
                    "死", "自杀", "不想活", "没意义", "绝望",
                    "没人爱我", "废物", "失败", "痛苦", "崩溃",
                ],
                "negative_repeat_threshold": 3,
            },
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
    # --- 青少年专属审计扩展字段 ---
    user_input: str = ""       # 用户当前输入文本（用于青少年玩笑式危机检测等）
    is_teen: bool = False      # 用户是否为青少年（启用青少年专属检测逻辑）
    # --- 苏格拉底反问时机审计扩展字段 ---
    dialogue_mode: str = "EMPATHY"          # 当前对话模式（EMPATHY/SOCRATIC）
    consecutive_calm_turns: int = 0         # 连续无强负面情绪轮次
    consecutive_avoidance_count: int = 0    # 用户连续回避反问次数
    max_abs_z_score: float = 0.0            # 个人基线最大 Z-score 绝对值


# ============================================================
# 五轴审计器
# ============================================================

class SafetyAuditor:
    """五轴安全审计器

    对 LLM 输出进行五个维度的安全审计，
    任一轴不通过则触发相应处理动作。

    基线感知能力：
        当个人基线偏离显著时，自动提高对应审计轴的敏感度。
        通过 set_baseline_context() 设置基线偏离信息。
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or load_audit_config()
        # 青少年语用检测敏感度调整（每次 audit() 调用时更新）
        self._teen_sensitivity: dict = {}
        # 基线感知：个人基线偏离信息（通过 set_baseline_context 设置）
        self._baseline_deviation: Optional[object] = None  # BaselineDeviation
        # 基线感知：各轴的敏感度提升系数（每次 audit() 调用时计算）
        self._baseline_axis_boost: dict[str, float] = {}

    def set_baseline_context(self, deviation: object) -> None:
        """设置个人基线偏离上下文

        当基线偏离显著时，审计器会自动提高对应轴的敏感度。

        Args:
            deviation: BaselineDeviation 对象，包含显著偏离的维度列表
        """
        self._baseline_deviation = deviation

    def _compute_baseline_boost(self) -> dict[str, float]:
        """根据基线偏离计算各轴的敏感度提升系数

        临床依据：
            当个人基线显著偏离时，对应审计轴应更敏感。
            例如：心率显著升高 → 危机检测更敏感；
                  焦虑情绪显著升高 → 谄媚检测更敏感。

        Returns:
            各轴的 boost 系数字典 {axis_name: boost_factor}
        """
        boost: dict[str, float] = {}
        baseline_cfg = self.config.get("baseline_sensitivity", {})

        # 未启用或无基线数据时返回空字典
        if not baseline_cfg.get("enabled", True):
            return boost
        if self._baseline_deviation is None:
            return boost

        # 获取显著偏离的维度列表
        significant = getattr(self._baseline_deviation, "significant_deviations", [])
        if not significant:
            return boost

        # 获取维度到轴的映射
        mapping = baseline_cfg.get("modality_to_axis_mapping", {})
        axis_boost_cfg = baseline_cfg.get("axis_boost", {})

        # 收集每个轴需要提升的维度
        axes_to_boost: dict[str, list[str]] = {}
        for dim in significant:
            # 直接匹配
            if dim in mapping:
                axes = mapping[dim].split(",")
                for axis in axes:
                    axis = axis.strip()
                    if axis not in axes_to_boost:
                        axes_to_boost[axis] = []
                    axes_to_boost[axis].append(dim)
            # 情绪维度特殊处理（emotion_z_scores 中的索引）
            elif dim.startswith("emotion_"):
                if dim in mapping:
                    axes = mapping[dim].split(",")
                    for axis in axes:
                        axis = axis.strip()
                        if axis not in axes_to_boost:
                            axes_to_boost[axis] = []
                        axes_to_boost[axis].append(dim)

        # 计算每个轴的 boost 系数
        for axis, dims in axes_to_boost.items():
            if axis in axis_boost_cfg:
                boost[axis] = axis_boost_cfg[axis]

        return boost

    def get_baseline_adjustment_info(self) -> dict:
        """获取基线感知调整信息（用于日志记录）

        Returns:
            包含调整信息的字典：
            - baseline_adjusted: bool 是否有基线调整
            - adjusted_axes: list[str] 被调整的轴列表
            - baseline_evidence: list[str] 调整依据（显著偏离的维度）
        """
        if not self._baseline_axis_boost:
            return {
                "baseline_adjusted": False,
                "adjusted_axes": [],
                "baseline_evidence": [],
            }

        # 获取显著偏离的维度作为证据
        evidence = []
        if self._baseline_deviation is not None:
            evidence = getattr(self._baseline_deviation, "significant_deviations", [])

        return {
            "baseline_adjusted": True,
            "adjusted_axes": list(self._baseline_axis_boost.keys()),
            "baseline_evidence": evidence,
        }

    def audit(self, context: AuditContext) -> AuditVerdict:
        """执行五轴审计

        流程：
            1. [青少年预检] 运行语用特征检测，获取敏感度调整系数
            2. [基线感知] 计算各轴的敏感度提升系数
            3. 五轴审计（各轴根据敏感度调整动态修改阈值）
            4. 综合裁决

        Args:
            context: 审计上下文

        Returns:
            AuditVerdict: 综合审计裁决
        """
        results: list[AuditResult] = []

        # ---- 青少年语用预检 ----
        # 在五轴审计前运行语用检测器，识别掩饰性表达，
        # 动态提升对应审计轴的敏感度。
        # 临床依据：WHO Adolescent Mental Health Technical Brief (2022)
        #          青少年约 70% 的心理困扰以间接/掩饰形式表达。
        self._teen_sensitivity = {}
        if context.is_teen and context.user_input:
            self._teen_sensitivity = get_sensitivity_adjustments(
                context.user_input,
                context={
                    "conversation_history": context.conversation_history,
                    "risk_level": context.risk_level,
                },
            )

        # ---- 基线感知预检 ----
        # 根据个人基线偏离情况，计算各轴的敏感度提升系数。
        # 临床依据：个体常态的异常检测比群体阈值更精准。
        self._baseline_axis_boost = self._compute_baseline_boost()

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

        # 轴 6：苏格拉底反问时机
        if self.config.get("socratic_timing", {}).get("enabled", True):
            results.append(self._audit_socratic_timing(context))

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

        基线感知：当生理指标（心率、呼吸等）显著偏离个人常态时，
                  降低升级阈值（更敏感），因为生理唤醒可能提示危机状态。
        """
        rule = self.config.get("crisis_delay", {})
        max_turns = rule.get("max_turns_without_escalation", 2)

        # 青少年语用预检：若检测到掩饰性表达，降低升级阈值
        # （掩饰性表达意味着危机可能比表面更严重，需要更早升级）
        crisis_boost = self._teen_sensitivity.get("crisis_delay_boost", 1.0)
        if crisis_boost > 1.0 and max_turns > 1:
            max_turns = max(1, int(max_turns / crisis_boost))

        # 基线感知：生理指标显著偏离时，降低升级阈值
        baseline_boost = self._baseline_axis_boost.get("crisis_delay", 1.0)
        if baseline_boost > 1.0 and max_turns > 1:
            max_turns = max(1, int(max_turns / baseline_boost))

        # 检查：CRISIS 状态 + 超过阈值轮次
        if context.dialog_state == "CRISIS" and context.crisis_turns > max_turns:
            reason = f"危机状态已持续 {context.crisis_turns} 轮未升级（阈值: {max_turns}）"
            # 记录基线感知调整
            if baseline_boost > 1.0:
                reason += f" [基线感知: 生理指标偏离, boost={baseline_boost}]"
            return AuditResult(
                axis=AuditAxis.CRISIS_DELAY,
                passed=False,
                reason=reason,
                suggested_action=AuditAction.ESCALATE_HUMAN,
            )

        # 青少年专属检测（与通用规则并行）
        if context.is_teen and self.config.get("teen_specific", {}).get("enabled", True):
            teen_result = self._teen_crisis_delay_check(context)
            if not teen_result.passed:
                return teen_result

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

        基线感知：当认知相关维度显著偏离个人常态时，
                  降低置信度阈值（更敏感），因为认知异常可能加剧妄想强化。
        """
        rule = self.config.get("delusion_reinforcement", {})
        threshold = rule.get("confidence_threshold", 0.7)
        keywords = rule.get("keywords", [])

        # 基线感知：认知维度显著偏离时，降低置信度阈值
        baseline_boost = self._baseline_axis_boost.get("delusion_reinforcement", 1.0)
        effective_threshold = threshold
        if baseline_boost > 1.0:
            effective_threshold = threshold / baseline_boost

        # 检查：LLM 输出是否包含妄想关键词
        contains_delusion_keyword = any(kw in context.llm_output for kw in keywords)

        # 检查：LLM 置信度是否过高（表示过度认同）
        if contains_delusion_keyword and context.llm_confidence > effective_threshold:
            reason = f"LLM 输出包含妄想关键词且置信度过高 ({context.llm_confidence:.2f} > {effective_threshold:.2f})"
            if baseline_boost > 1.0:
                reason += f" [基线感知: 认知维度偏离, boost={baseline_boost}]"
            return AuditResult(
                axis=AuditAxis.DELUSION_REINFORCEMENT,
                passed=False,
                reason=reason,
                suggested_action=AuditAction.REWRITE,
            )

        # 青少年专属检测（与通用规则并行）
        if context.is_teen and self.config.get("teen_specific", {}).get("enabled", True):
            teen_result = self._teen_delusion_reinforcement_check(context)
            if not teen_result.passed:
                return teen_result

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

        # 青少年专属检测（与通用规则并行）
        if context.is_teen and self.config.get("teen_specific", {}).get("enabled", True):
            teen_result = self._teen_stigma_rejection_check(context)
            if not teen_result.passed:
                return teen_result

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

        基线感知：当情绪相关维度显著偏离个人常态时，
                  降低同意率阈值（更敏感），因为情绪异常时谄媚可能加剧依赖。
        """
        rule = self.config.get("sycophancy", {})
        threshold = rule.get("agreement_rate_threshold", 0.9)
        window_size = rule.get("window_size", 10)
        markers = rule.get("agreement_markers", [])

        # 基线感知：情绪维度显著偏离时，降低同意率阈值
        baseline_boost = self._baseline_axis_boost.get("sycophancy", 1.0)
        effective_threshold = threshold
        if baseline_boost > 1.0:
            effective_threshold = threshold / baseline_boost

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

        if agreement_rate > effective_threshold:
            reason = f"同意率过高 ({agreement_rate:.1%} > {effective_threshold:.1%})"
            if baseline_boost > 1.0:
                reason += f" [基线感知: 情绪维度偏离, boost={baseline_boost}]"
            return AuditResult(
                axis=AuditAxis.SYCOPHANCY,
                passed=False,
                reason=reason,
                suggested_action=AuditAction.REWRITE,
            )

        # 青少年专属检测（与通用规则并行）
        if context.is_teen and self.config.get("teen_specific", {}).get("enabled", True):
            teen_result = self._teen_sycophancy_check(context)
            if not teen_result.passed:
                return teen_result

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

        基线感知：当行为相关维度（打字速度、语速等）显著偏离个人常态时，
                  降低偏离度阈值（更敏感），因为行为异常可能提示轨迹漂移。
        """
        rule = self.config.get("trajectory_drift", {})
        threshold = rule.get("deviation_threshold", 0.5)
        topic_keywords = rule.get("topic_keywords", [])

        # 青少年语用预检：若检测到回避/掩饰性表达，降低偏离度阈值
        # （回避行为提示该话题有重大心理意义，需更敏感地捕捉偏移）
        drift_boost = self._teen_sensitivity.get("drift_boost", 1.0)
        if drift_boost > 1.0:
            threshold = threshold / drift_boost

        # 基线感知：行为维度显著偏离时，降低偏离度阈值
        baseline_boost = self._baseline_axis_boost.get("trajectory_drift", 1.0)
        if baseline_boost > 1.0:
            threshold = threshold / baseline_boost

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
            reason = f"轨迹偏离度过高 ({deviation:.2f} > {threshold:.2f})"
            if baseline_boost > 1.0:
                reason += f" [基线感知: 行为维度偏离, boost={baseline_boost}]"
            return AuditResult(
                axis=AuditAxis.TRAJECTORY_DRIFT,
                passed=False,
                reason=reason,
                suggested_action=AuditAction.REWRITE,
            )

        # 青少年专属检测（与通用规则并行）
        if context.is_teen and self.config.get("teen_specific", {}).get("enabled", True):
            teen_result = self._teen_trajectory_drift_check(context)
            if not teen_result.passed:
                return teen_result

        return AuditResult(
            axis=AuditAxis.TRAJECTORY_DRIFT,
            passed=True,
        )

    # --------------------------------------------------------
    # 轴 6：苏格拉底反问时机（分级降级版）
    # --------------------------------------------------------
    def _audit_socratic_timing(self, context: AuditContext) -> AuditResult:
        """审计苏格拉底反问时机（分级降级版）

        临床依据：Socratic questioning 应在情绪稳定后使用 (Beck, 2011)。
                  青少年在情绪未平复时，反问会引发“被审问”感，
                  导致抵触、烦躁、甚至中断对话。

        返回值扩展：passed + recommended_depth
            - recommended_depth=0: 退回共情模式（危机/情绪未平复/连续回避）
            - recommended_depth=1: SHALLOW 浅层反问（情绪刚平复）
            - recommended_depth=2: MEDIUM 中层反问（情绪稳定+基线正常）
            - recommended_depth=3: DEEP 深层反问（情绪稳定+连续探索）

        检测条件（任一触发则降级）：
            1. 危机状态下触发反问 → depth=0
            2. 情绪未平复（连续 calm turns < min_calm_turns）→ depth=0
            3. 用户连续回避次数达到阈值 → depth=0，冻结 3 轮
            4. 基线显著偏离（|Z| > max_abs_z_score）→ depth=1
        """
        rule = self.config.get("socratic_timing", {})
        min_calm = rule.get("min_calm_turns", 2)
        max_avoidance = rule.get("max_consecutive_avoidance", 2)
        max_z = rule.get("max_abs_z_score", 2.0)
        block_crisis = rule.get("block_in_crisis", True)

        # 只在 SOCRATIC 模式或 SOCRATIC_READY 状态下审计
        is_socratic_context = (
            context.dialogue_mode == "SOCRATIC"
            or context.dialog_state == "SOCRATIC_READY"
        )
        if not is_socratic_context:
            return AuditResult(axis=AuditAxis.SOCRATIC_TIMING, passed=True, recommended_depth=-1)

        # 检查 1：危机状态下 → 退回共情（depth=0）
        if block_crisis and context.risk_level in ("crisis", "high"):
            return AuditResult(
                axis=AuditAxis.SOCRATIC_TIMING,
                passed=False,
                reason=f"危机状态下禁止触发反问 (risk_level={context.risk_level})",
                suggested_action=AuditAction.REWRITE,
                recommended_depth=0,  # 退回共情
            )

        # 检查 2：情绪未平复 → 退回共情（depth=0）
        if context.consecutive_calm_turns < min_calm:
            return AuditResult(
                axis=AuditAxis.SOCRATIC_TIMING,
                passed=False,
                reason=(
                    f"情绪未平复时触发反问 "
                    f"(calm_turns={context.consecutive_calm_turns} < {min_calm})"
                ),
                suggested_action=AuditAction.REWRITE,
                recommended_depth=0,  # 退回共情
            )

        # 检查 3：用户连续回避 → 退回共情（depth=0），冻结 3 轮
        if context.consecutive_avoidance_count >= max_avoidance:
            return AuditResult(
                axis=AuditAxis.SOCRATIC_TIMING,
                passed=False,
                reason=(
                    f"用户连续回避 {context.consecutive_avoidance_count} 次后继续追问 "
                    f"(阈值={max_avoidance})，深度冻结 3 轮"
                ),
                suggested_action=AuditAction.REWRITE,
                recommended_depth=0,  # 退回共情，并触发冻结
            )

        # 检查 4：基线显著偏离 → 降级到 SHALLOW（depth=1）
        if abs(context.max_abs_z_score) > max_z:
            return AuditResult(
                axis=AuditAxis.SOCRATIC_TIMING,
                passed=False,
                reason=(
                    f"基线显著偏离时触发反问 "
                    f"(|Z|={abs(context.max_abs_z_score):.2f} > {max_z})，降级到 SHALLOW"
                ),
                suggested_action=AuditAction.REWRITE,
                recommended_depth=1,  # 降级到 SHALLOW
            )

        # 通过审计，根据条件推荐深度
        # depth=3: 情绪稳定 + 连续探索
        if context.consecutive_calm_turns >= 4:
            return AuditResult(
                axis=AuditAxis.SOCRATIC_TIMING,
                passed=True,
                recommended_depth=3,  # DEEP
            )
        # depth=2: 情绪稳定 + 基线正常
        if context.consecutive_calm_turns >= 3 and abs(context.max_abs_z_score) < 1.5:
            return AuditResult(
                axis=AuditAxis.SOCRATIC_TIMING,
                passed=True,
                recommended_depth=2,  # MEDIUM
            )
        # depth=1: 情绪刚平复
        return AuditResult(
            axis=AuditAxis.SOCRATIC_TIMING,
            passed=True,
            recommended_depth=1,  # SHALLOW
        )

    # ========================================================
    # 青少年专属检测逻辑（与通用规则并行运行）
    # ========================================================
    # 以下方法仅在 context.is_teen=True 时被调用，
    # 针对青少年发展心理学特征定制的检测规则。
    # 每个方法均在 evidence (reason) 中标注「青少年专属规则触发」。

    def _teen_crisis_delay_check(self, context: AuditContext) -> AuditResult:
        """青少年专属：危机升级延迟检测

        临床依据：
            青少年危机信号常以"玩笑式"表达出现（如"我死了算了哈哈"），
            通用检测器易漏判。研究表明，12-18 岁青少年中约 70% 的
            自杀意念表达采用间接或幽默包装的形式。
            来源：WHO QualityRights for adolescents (2022);
            NICE Guidelines CG185 – Self-harm and suicide prevention.

        规则：检测"玩笑标记 + 危机关键词"组合模式。
        """
        teen_cfg = self.config.get("teen_specific", {}).get("crisis_delay", {})
        joke_patterns = teen_cfg.get("joke_crisis_patterns", [
            r"死.*哈[哈嘿]", r"不想活.*嘻[嘻哈]", r"自[杀杀].*笑",
            r"活[着不下去].*哈[哈嘿]", r"死[了去算].*[嘻哈笑]",
            r"[嘻哈笑].*死[了去算]",
        ])
        direct_signals = teen_cfg.get("direct_crisis_signals", [
            "不想活", "去死", "自杀", "活着没意思", "死了算了",
            "世界没我会更好", "消失算了",
        ])
        min_matches = teen_cfg.get("min_crisis_matches", 2)

        text = context.user_input or context.llm_output
        matches = 0

        # 检测"玩笑+危机词"组合
        for pattern in joke_patterns:
            if re.search(pattern, text):
                matches += 1

        # 检测直接危机信号（即使没有玩笑包装）
        for signal in direct_signals:
            if signal in text:
                matches += 1

        if matches >= min_matches:
            return AuditResult(
                axis=AuditAxis.CRISIS_DELAY,
                passed=False,
                reason=(
                    f"[青少年专属规则触发] 检测到玩笑式危机信号表达 "
                    f"(匹配数: {matches}，阈值: {min_matches})。"
                    f"临床依据: WHO QualityRights for adolescents (2022); "
                    f"NICE CG185 – 青少年危机信号常以幽默包装出现"
                ),
                suggested_action=AuditAction.ESCALATE_HUMAN,
            )

        return AuditResult(axis=AuditAxis.CRISIS_DELAY, passed=True)

    def _teen_delusion_reinforcement_check(
        self, context: AuditContext
    ) -> AuditResult:
        """青少年专属：妄想强化检测

        临床依据：
            青少年正处于自我认同形成期（Erikson 心理社会发展理论的
            Identity vs Role Confusion 阶段），对"被针对""被排挤"的
            敏感度显著高于成人。需区分"真实的校园霸凌"与"病理性妄想"，
            避免将正常的社交困扰病理化。
            来源：NICE Guidelines CG178 – Psychosis and schizophrenia
            in children and young people;
            Erikson (1968) Identity: Youth and Crisis.

        规则：
            1. 若用户描述包含校园/社交语境（被排挤、被欺负等），
               不判定为妄想强化——可能是真实的霸凌经历。
            2. 仅在缺乏现实语境时，才对病理性妄想关键词触发检测。
        """
        teen_cfg = self.config.get("teen_specific", {}).get(
            "delusion_reinforcement", {}
        )
        bullying_context_keywords = teen_cfg.get("bullying_context_keywords", [
            "同学", "班级", "学校", "老师", "宿舍", "被排挤",
            "被欺负", "被孤立", "霸凌", "校园", "同桌", "室友",
        ])
        pathological_keywords = teen_cfg.get("pathological_keywords", [
            "被监控", "被下毒", "外星人", "超能力", "被跟踪",
            "思维被插入", "思维被抽取", "被控制",
        ])
        threshold = teen_cfg.get(
            "confidence_threshold",
            self.config.get("delusion_reinforcement", {}).get(
                "confidence_threshold", 0.7
            ),
        )

        text = context.user_input or context.llm_output

        # 步骤 1: 检测是否存在校园/霸凌现实语境
        has_bullying_context = any(
            kw in text for kw in bullying_context_keywords
        )

        # 步骤 2: 若有现实语境，不判定为妄想（避免过度病理化）
        if has_bullying_context:
            return AuditResult(axis=AuditAxis.DELUSION_REINFORCEMENT, passed=True)

        # 步骤 3: 无现实语境时，检测病理性妄想关键词
        contains_pathological = any(
            kw in context.llm_output for kw in pathological_keywords
        )

        if contains_pathological and context.llm_confidence > threshold:
            return AuditResult(
                axis=AuditAxis.DELUSION_REINFORCEMENT,
                passed=False,
                reason=(
                    f"[青少年专属规则触发] LLM 输出包含病理性妄想关键词 "
                    f"且置信度过高 ({context.llm_confidence:.2f} > {threshold})，"
                    f"且无校园/社交现实语境。"
                    f"临床依据: NICE CG178 – 青少年妄想需排除现实社交困扰"
                ),
                suggested_action=AuditAction.REWRITE,
            )

        return AuditResult(
            axis=AuditAxis.DELUSION_REINFORCEMENT, passed=True
        )

    def _teen_stigma_rejection_check(
        self, context: AuditContext
    ) -> AuditResult:
        """青少年专属：污名化与拒绝检测

        临床依据：
            青少年对心理标签极度敏感。直接告知"你有抑郁症"会导致
            青少年立即关闭对话并产生病耻感。污名化检测阈值应比成人
            更严格，触发后应替换为去标签化表达（如"你正在经历一些
            困难的情绪"）。
            来源：WHO QualityRights (2019) – Mental health of adolescents;
            NICE Guidelines CG26 – Depression in children and young people.

        规则：
            1. 扩展诊断标签列表，覆盖更多青少年常见的标签化表达。
            2. 触发后建议替换为去标签化表达。
        """
        teen_cfg = self.config.get("teen_specific", {}).get(
            "stigma_rejection", {}
        )
        extra_labels = teen_cfg.get("additional_diagnostic_labels", [
            "你有抑郁症", "你有焦虑症", "你有自闭症",
            "你是抑郁症", "你是焦虑症", "你有心理疾病",
            "你精神有问题", "你有病",
        ])
        extra_stigma = teen_cfg.get("additional_stigmatizing_terms", [
            "玻璃心", "想不开", "没事找事", "无病呻吟",
            "心理素质差", "抗压能力差",
        ])
        # 合并通用列表 + 青少年扩展列表
        all_labels = (
            self.config.get("stigma_rejection", {}).get("diagnostic_labels", [])
            + extra_labels
        )
        all_terms = (
            self.config.get("stigma_rejection", {}).get("stigmatizing_terms", [])
            + extra_stigma
        )

        # 检查诊断性标签（青少年扩展列表）
        for label in all_labels:
            if label in context.llm_output:
                return AuditResult(
                    axis=AuditAxis.STIGMA_REJECTION,
                    passed=False,
                    reason=(
                        f"[青少年专属规则触发] LLM 输出包含对青少年高度敏感的"
                        f"诊断性标签: '{label}'。"
                        f"建议替换为去标签化表达（如'你正在经历一些困难的情绪'）。"
                        f"临床依据: WHO QualityRights (2019); NICE CG26"
                    ),
                    suggested_action=AuditAction.REWRITE,
                )

        # 检查污名化用语（青少年扩展列表）
        for term in all_terms:
            if term in context.llm_output:
                return AuditResult(
                    axis=AuditAxis.STIGMA_REJECTION,
                    passed=False,
                    reason=(
                        f"[青少年专属规则触发] LLM 输出包含对青少年的"
                        f"污名化用语: '{term}'。"
                        f"临床依据: WHO QualityRights (2019) – 青少年去污名化"
                    ),
                    suggested_action=AuditAction.REWRITE,
                )

        return AuditResult(axis=AuditAxis.STIGMA_REJECTION, passed=True)

    def _teen_sycophancy_check(self, context: AuditContext) -> AuditResult:
        """青少年专属：谄媚倾向检测

        临床依据：
            青少年更易受 AI 无条件认同的影响，形成情感依赖或强化偏激认知。
            同意率阈值从成人 90% 下调至 75%，以防止青少年在"回音室效应"
            中固化非适应性信念。
            来源：APA Guidelines for Adolescent Psychotherapy (2020);
            NICE Guidelines CG26 – working with children and young people.

        规则：同意率阈值从 0.9 下调至 0.75（可通过配置调整）。
        """
        teen_cfg = self.config.get("teen_specific", {}).get("sycophancy", {})
        threshold = teen_cfg.get("agreement_rate_threshold", 0.75)
        window_size = teen_cfg.get(
            "window_size",
            self.config.get("sycophancy", {}).get("window_size", 10),
        )
        markers = self.config.get("sycophancy", {}).get("agreement_markers", [])

        recent_history = context.conversation_history[-window_size:]
        if not recent_history:
            return AuditResult(axis=AuditAxis.SYCOPHANCY, passed=True)

        agreement_count = 0
        for turn in recent_history:
            content = turn.get("content", "")
            if any(marker in content for marker in markers):
                agreement_count += 1

        agreement_rate = (
            agreement_count / len(recent_history) if recent_history else 0
        )

        if agreement_rate > threshold:
            return AuditResult(
                axis=AuditAxis.SYCOPHANCY,
                passed=False,
                reason=(
                    f"[青少年专属规则触发] 同意率过高 "
                    f"({agreement_rate:.1%} > {threshold:.1%})，"
                    f"可能导致青少年形成 AI 依赖或固化偏激认知。"
                    f"临床依据: APA Guidelines for Adolescent Psychotherapy (2020)"
                ),
                suggested_action=AuditAction.REWRITE,
            )

        return AuditResult(axis=AuditAxis.SYCOPHANCY, passed=True)

    def _teen_trajectory_drift_check(
        self, context: AuditContext
    ) -> AuditResult:
        """青少年专属：轨迹漂移检测

        临床依据：
            青少年对话节奏跳跃，思维发散属正常发展特征（前额叶皮层
            尚未完全发育），需区分"正常的思维跳跃"与"病理性轨迹漂移"。
            偏离度阈值适当放宽（从 0.5 上调至 0.7），但增加对
            "反复回到同一负面主题"（rumination）的检测——
            这可能是反刍思维的表现。
            来源：NICE Guidelines CG26 – adolescent engagement strategies;
            Casey et al. (2008) Adolescent brain development.

        规则：
            1. 偏离度阈值放宽至 0.7（青少年正常思维跳跃）。
            2. 新增"反复回到同一负面主题"检测（rumination 指标）。
        """
        teen_cfg = self.config.get("teen_specific", {}).get(
            "trajectory_drift", {}
        )
        relaxed_threshold = teen_cfg.get("deviation_threshold", 0.7)
        topic_keywords = self.config.get("trajectory_drift", {}).get(
            "topic_keywords", []
        )
        negative_theme_keywords = teen_cfg.get("negative_theme_keywords", [
            "死", "自杀", "不想活", "没意义", "绝望",
            "没人爱我", "废物", "失败", "痛苦", "崩溃",
        ])
        repeat_threshold = teen_cfg.get("negative_repeat_threshold", 3)

        # 规则 1: 放宽偏离度阈值
        if topic_keywords and context.current_topic:
            output_lower = context.llm_output.lower()
            keyword_matches = sum(
                1 for kw in topic_keywords if kw in output_lower
            )
            relevance_score = (
                keyword_matches / len(topic_keywords) if topic_keywords else 0
            )
            deviation = 1.0 - relevance_score

            # 仅在偏离度超过放宽后阈值时触发
            if deviation > relaxed_threshold:
                return AuditResult(
                    axis=AuditAxis.TRAJECTORY_DRIFT,
                    passed=False,
                    reason=(
                        f"[青少年专属规则触发] 轨迹偏离度过高 "
                        f"({deviation:.2f} > {relaxed_threshold})，"
                        f"已超出青少年正常思维跳跃范围。"
                        f"临床依据: NICE CG26; Casey et al. (2008)"
                    ),
                    suggested_action=AuditAction.REWRITE,
                )

        # 规则 2: 检测"反复回到同一负面主题"（rumination 指标）
        negative_count = 0
        for turn in context.conversation_history:
            content = turn.get("content", "")
            if any(neg in content for neg in negative_theme_keywords):
                negative_count += 1

        if negative_count >= repeat_threshold:
            return AuditResult(
                axis=AuditAxis.TRAJECTORY_DRIFT,
                passed=False,
                reason=(
                    f"[青少年专属规则触发] 对话中反复出现负面主题 "
                    f"(次数: {negative_count}，阈值: {repeat_threshold})，"
                    f"可能为反刍思维（rumination），需引导话题转换。"
                    f"临床依据: NICE CG26 – 青少年反刍思维检测"
                ),
                suggested_action=AuditAction.REWRITE,
            )

        return AuditResult(axis=AuditAxis.TRAJECTORY_DRIFT, passed=True)

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
    user_input: str = "",
    is_teen: bool = False,
    dialogue_mode: str = "EMPATHY",
    consecutive_calm_turns: int = 0,
    consecutive_avoidance_count: int = 0,
    max_abs_z_score: float = 0.0,
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
        user_input: 用户当前输入文本（青少年危机检测用）
        is_teen: 用户是否为青少年（启用青少年专属检测）
        dialogue_mode: 当前对话模式（EMPATHY/SOCRATIC）
        consecutive_calm_turns: 连续无强负面情绪轮次
        consecutive_avoidance_count: 用户连续回避反问次数
        max_abs_z_score: 个人基线最大 Z-score 绝对值

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
        user_input=user_input,
        is_teen=is_teen,
        dialogue_mode=dialogue_mode,
        consecutive_calm_turns=consecutive_calm_turns,
        consecutive_avoidance_count=consecutive_avoidance_count,
        max_abs_z_score=max_abs_z_score,
    )

    auditor = SafetyAuditor()
    return auditor.audit(context)
