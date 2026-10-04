"""
青少年语用特征检测器单元测试

覆盖：
- 五类掩饰性表达模式检测（反话/轻描淡写/玩笑化危机/试探性/回避性）
- 上下文感知判断（正面语境 vs 负面语境）
- 可解释性输出验证
- 与审计器集成验证（敏感度动态调整）
- 25 条青少年真实语料样本

临床依据：
    - WHO Adolescent Mental Health Technical Brief (2022)
    - Michelmore et al. (2022) Journal of Adolescent Health
    - NICE CG185 (2011) Self-harm and Suicide Prevention
    - King et al. (2021) Youth Suicide Prevention
    - Miller & Rollnick (2013) Motivational Interviewing
"""
from __future__ import annotations

import pytest

from intervention.teen_language_detector import (
    DisguisePatternType,
    SuggestedAction,
    TeenLanguageDetector,
    TeenLanguageResult,
    detect_teen_disguise,
    get_sensitivity_adjustments,
)


@pytest.fixture
def detector():
    return TeenLanguageDetector()


# ============================================================
# 青少年语料样本库（25 条，覆盖五类模式）
# ============================================================

# 每条语料格式：(text, context, expected_pattern, description)
TEEN_CORPUS = [
    # ---- 反话模式 (COUNTER) ----
    # 1. 负面上下文 + "我没事" → 应判定为反话
    (
        "我没事",
        {
            "conversation_history": [
                {"role": "user", "content": "今天被同学嘲笑了，很难过"},
            ],
            "risk_level": "medium",
        },
        DisguisePatternType.COUNTER,
        "负面上下文中的'我没事'——典型反话",
    ),
    # 2. 负面上下文 + "我很好"
    (
        "我真的很好，不用担心",
        {
            "conversation_history": [
                {"role": "user", "content": "爸妈又吵架了"},
            ],
            "risk_level": "medium",
        },
        DisguisePatternType.COUNTER,
        "家庭冲突后说'我很好'——反话掩饰",
    ),
    # 3. 正面上下文 + "我没事" → 不应判定为掩饰
    (
        "我没事呀，今天考试通过了超开心",
        {
            "conversation_history": [
                {"role": "user", "content": "考试通过了！太开心了"},
            ],
            "risk_level": "low",
        },
        None,
        "正面语境中的'我没事'——正常表达，非掩饰",
    ),
    # 4. 反话 + 否认情绪
    (
        "我才没有难过呢",
        {
            "conversation_history": [
                {"role": "user", "content": "好朋友转学了"},
            ],
            "risk_level": "low",
        },
        DisguisePatternType.COUNTER,
        "否认负面情绪——反话模式",
    ),
    # 5. 无上下文 + "我没事" → 低置信度
    (
        "我没事",
        {},
        None,
        "无上下文时'我没事'——信号不足不判定",
    ),

    # ---- 轻描淡写模式 (MINIMIZE) ----
    # 6. 危机上下文 + "还行"
    (
        "还行吧",
        {
            "conversation_history": [
                {"role": "user", "content": "每天都睡不着，想过消失"},
            ],
            "risk_level": "high",
        },
        DisguisePatternType.MINIMIZE,
        "危机信号后说'还行吧'——轻描淡写",
    ),
    # 7. 负面上下文 + "就那样"
    (
        "就那样吧，无所谓",
        {
            "conversation_history": [
                {"role": "user", "content": "被老师当众批评了"},
            ],
            "risk_level": "medium",
        },
        DisguisePatternType.MINIMIZE,
        "被批评后说'就那样'——淡化痛苦",
    ),
    # 8. 无危机上下文 + "还行" → 不判定
    (
        "还行，今天天气不错",
        {
            "risk_level": "low",
            "conversation_history": [],
        },
        None,
        "无危机上下文的'还行'——正常表达",
    ),

    # ---- 玩笑化危机模式 (JOKE_CRISIS) ----
    # 9. 危机词 + 哈哈
    (
        "死了算了哈哈",
        {"risk_level": "low"},
        DisguisePatternType.JOKE_CRISIS,
        "'死了算了哈哈'——典型玩笑化危机",
    ),
    # 10. 危机词 + 但不敢
    (
        "想跳楼但不敢哈哈",
        {"risk_level": "low"},
        DisguisePatternType.JOKE_CRISIS,
        "'想跳楼但不敢'——玩笑包装的危机意念",
    ),
    # 11. 危机词 + 开玩笑
    (
        "不想活了开玩笑的啦",
        {"risk_level": "low"},
        DisguisePatternType.JOKE_CRISIS,
        "'不想活了开玩笑的'——掩饰性危机表达",
    ),
    # 12. 纯危机词无幽默标记 → 不触发此模式
    (
        "我觉得活着没意思",
        {"risk_level": "high"},
        None,  # 不是玩笑化，而是直接危机信号（应由其他模块检测）
        "直接危机表达无幽默标记——不属于玩笑化模式",
    ),

    # ---- 试探性表达模式 (PROBE) ----
    # 13. 假设性消失
    (
        "如果我消失了会有人注意到吗",
        {"risk_level": "medium"},
        DisguisePatternType.PROBE,
        "'如果我消失了'——典型试探性表达",
    ),
    # 14. 试探性告别
    (
        "你会记得我吗",
        {"risk_level": "low"},
        DisguisePatternType.PROBE,
        "'你会记得我吗'——试探性告别",
    ),
    # 15. 间接求助
    (
        "有没有人在意我是否存在",
        {"risk_level": "medium"},
        DisguisePatternType.PROBE,
        "'有没有人在意我'——间接求助信号",
    ),
    # 16. 假设性死亡
    (
        "要是我不在了大家会怎样",
        {"risk_level": "medium"},
        DisguisePatternType.PROBE,
        "'要是我不在了'——假设性死亡试探",
    ),

    # ---- 回避性转移模式 (AVOID) ----
    # 17. 负面话题后回避
    (
        "算了没什么",
        {
            "conversation_history": [
                {"role": "user", "content": "我觉得很绝望"},
            ],
            "risk_level": "medium",
        },
        DisguisePatternType.AVOID,
        "表达绝望后突然说'算了没什么'——回避性转移",
    ),
    # 18. 负面话题后转移
    (
        "不说这个了",
        {
            "conversation_history": [
                {"role": "user", "content": "每天都想哭"},
            ],
            "risk_level": "medium",
        },
        DisguisePatternType.AVOID,
        "表达痛苦后说'不说这个了'——话题回避",
    ),
    # 19. 无负面上下文的回避 → 不判定
    (
        "算了没什么",
        {
            "conversation_history": [
                {"role": "user", "content": "今天中午吃了面条"},
            ],
            "risk_level": "low",
        },
        None,
        "中性话题后'算了没什么'——正常表达",
    ),
    # 20. 高风险下的回避
    (
        "不想说了",
        {"risk_level": "high"},
        DisguisePatternType.AVOID,
        "高风险下的'不想说了'——回避行为",
    ),

    # ---- 混合/边界样本 ----
    # 21. 玩笑化危机 + 高置信度
    (
        "哈哈死了算了真的",
        {
            "conversation_history": [
                {"role": "user", "content": "被全班孤立了"},
            ],
            "risk_level": "high",
        },
        DisguisePatternType.JOKE_CRISIS,
        "被孤立后'哈哈死了算了'——玩笑化危机（高危险）",
    ),
    # 22. 反话 + 高风险
    (
        "我好得很，一点都不难过",
        {
            "conversation_history": [
                {"role": "user", "content": "最好的朋友背叛了我"},
            ],
            "risk_level": "high",
        },
        DisguisePatternType.COUNTER,
        "背叛后'我好得很'——强烈反话",
    ),
    # 23. 正常表达（不应触发任何模式）
    (
        "今天心情还不错，和朋友去打球了",
        {"risk_level": "low"},
        None,
        "正常积极表达——不应触发任何掩饰检测",
    ),
    # 24. 轻描淡写 + 危机上下文
    (
        "一般般吧",
        {
            "conversation_history": [
                {"role": "user", "content": "割了手腕但是不深"},
            ],
            "risk_level": "crisis",
        },
        DisguisePatternType.MINIMIZE,
        "自伤后说'一般般'——极度轻描淡写（高危）",
    ),
    # 25. 试探 + 告别组合
    (
        "谢谢你陪我，以后还能见面吗",
        {"risk_level": "medium"},
        DisguisePatternType.PROBE,
        "'谢谢你陪我+以后还能见面吗'——试探性告别",
    ),
]


# ============================================================
# 语料库批量检测测试
# ============================================================

class TestTeenCorpusDetection:
    """使用 25 条青少年真实语料样本验证检测准确性"""

    @pytest.mark.parametrize(
        "text,context,expected_pattern,description",
        TEEN_CORPUS,
        ids=[t[3] for t in TEEN_CORPUS],
    )
    def test_corpus_detection(self, text, context, expected_pattern, description):
        """语料样本检测"""
        result = detect_teen_disguise(text, context)

        if expected_pattern is None:
            # 不应检测到掩饰性表达
            # （可能检测到但置信度低，或完全不检测）
            if result.detected:
                # 如果检测到了，置信度应该较低
                assert result.confidence < 0.5, (
                    f"'{text}' 不应被高置信度判定为掩饰: "
                    f"检测到 {result.pattern_type.value} (conf={result.confidence})"
                )
        else:
            # 应检测到指定模式
            assert result.detected, (
                f"'{text}' 应检测到 {expected_pattern.value}，但未检测到"
            )
            assert result.pattern_type == expected_pattern, (
                f"'{text}' 期望 {expected_pattern.value}，"
                f"实际 {result.pattern_type.value}"
            )
            assert result.confidence > 0.3, (
                f"'{text}' 置信度过低: {result.confidence}"
            )


# ============================================================
# 五类模式独立测试
# ============================================================

class TestCounterPattern:
    """反话模式测试"""

    def test_counter_with_negative_context(self, detector):
        """负面上下文中的反话应被检测"""
        result = detector.detect(
            "我没事",
            context={
                "conversation_history": [
                    {"role": "user", "content": "被同学嘲笑了很难过"},
                ],
                "risk_level": "medium",
            },
        )
        assert result.detected is True
        assert result.pattern_type == DisguisePatternType.COUNTER
        assert "反话" in result.explanation
        assert result.suggested_action == SuggestedAction.GENTLE_PROBE

    def test_counter_with_positive_context_not_detected(self, detector):
        """正面上下文中的'我没事'不应被判定为反话"""
        result = detector.detect(
            "我没事，今天考试通过了",
            context={
                "conversation_history": [
                    {"role": "user", "content": "考试通过了！开心"},
                ],
                "risk_level": "low",
            },
        )
        assert result.detected is False

    def test_counter_explanation_contains_evidence(self, detector):
        """反话检测结果应包含可解释证据"""
        result = detector.detect(
            "我没事",
            context={
                "conversation_history": [
                    {"role": "user", "content": "被欺负了"},
                ],
                "risk_level": "medium",
            },
        )
        assert result.detected is True
        assert "负面信号" in result.explanation
        assert "WHO" in result.explanation  # 应引用临床依据
        assert result.matched_text == "我没事"


class TestMinimizePattern:
    """轻描淡写模式测试"""

    def test_minimize_with_crisis_context(self, detector):
        """危机上下文中的轻描淡写应被检测"""
        result = detector.detect(
            "还行吧",
            context={
                "conversation_history": [
                    {"role": "user", "content": "每天都想消失"},
                ],
                "risk_level": "high",
            },
        )
        assert result.detected is True
        assert result.pattern_type == DisguisePatternType.MINIMIZE
        assert "淡化" in result.explanation

    def test_minimize_without_crisis_not_detected(self, detector):
        """无危机上下文的'还行'不判定"""
        result = detector.detect(
            "还行，今天天气不错",
            context={"risk_level": "low"},
        )
        assert result.detected is False


class TestJokeCrisisPattern:
    """玩笑化危机模式测试"""

    def test_joke_crisis_basic(self, detector):
        """基本玩笑化危机应被检测"""
        result = detector.detect("死了算了哈哈")
        assert result.detected is True
        assert result.pattern_type == DisguisePatternType.JOKE_CRISIS
        assert result.confidence >= 0.8
        assert result.suggested_action == SuggestedAction.SAFETY_CONFIRM

    def test_joke_crisis_with_casual_marker(self, detector):
        """轻描淡写包装（但不敢）也应被检测"""
        result = detector.detect("想跳楼但不敢")
        assert result.detected is True
        assert result.pattern_type == DisguisePatternType.JOKE_CRISIS

    def test_pure_crisis_not_joke(self, detector):
        """纯危机表达（无幽默标记）不触发玩笑化模式"""
        result = detector.detect("我觉得活着没意思")
        # 不应判定为玩笑化危机
        if result.detected:
            assert result.pattern_type != DisguisePatternType.JOKE_CRISIS


class TestProbePattern:
    """试探性表达模式测试"""

    def test_probe_disappearance(self, detector):
        """假设性消失应被检测"""
        result = detector.detect("如果我消失了会有人注意到吗")
        assert result.detected is True
        assert result.pattern_type == DisguisePatternType.PROBE
        assert result.confidence >= 0.7

    def test_probe_remember(self, detector):
        """'你会记得我吗'应被检测"""
        result = detector.detect("你会记得我吗")
        assert result.detected is True
        assert result.pattern_type == DisguisePatternType.PROBE

    def test_probe_explanation_cites_source(self, detector):
        """试探性表达检测应引用临床依据"""
        result = detector.detect("要是我不在了大家会怎样")
        assert result.detected is True
        assert "NICE CG185" in result.explanation or "King" in result.explanation


class TestAvoidPattern:
    """回避性转移模式测试"""

    def test_avoid_after_negative(self, detector):
        """负面话题后的回避应被检测"""
        result = detector.detect(
            "算了没什么",
            context={
                "conversation_history": [
                    {"role": "user", "content": "我觉得很绝望"},
                ],
                "risk_level": "medium",
            },
        )
        assert result.detected is True
        assert result.pattern_type == DisguisePatternType.AVOID
        assert "回避" in result.explanation

    def test_avoid_without_negative_not_detected(self, detector):
        """无负面上下文的回避不判定"""
        result = detector.detect(
            "算了没什么",
            context={
                "conversation_history": [
                    {"role": "user", "content": "今天吃了面条"},
                ],
                "risk_level": "low",
            },
        )
        assert result.detected is False

    def test_avoid_high_risk(self, detector):
        """高风险下的回避应被检测"""
        result = detector.detect(
            "不想说了",
            context={"risk_level": "high"},
        )
        assert result.detected is True
        assert result.pattern_type == DisguisePatternType.AVOID


# ============================================================
# 敏感度调整集成测试
# ============================================================

class TestSensitivityAdjustment:
    """验证语用检测对审计轴敏感度的动态调整"""

    def test_joke_crisis_boosts_crisis_delay(self):
        """玩笑化危机应提升危机延迟轴敏感度"""
        adj = get_sensitivity_adjustments("死了算了哈哈")
        assert adj["crisis_delay_boost"] > 1.0

    def test_counter_boosts_crisis_and_drift(self):
        """反话模式应提升危机延迟 + 漂移检测敏感度"""
        adj = get_sensitivity_adjustments(
            "我没事",
            context={
                "conversation_history": [
                    {"role": "user", "content": "被欺负了很难过"},
                ],
                "risk_level": "medium",
            },
        )
        assert adj["crisis_delay_boost"] > 1.0
        assert adj["drift_boost"] > 1.0

    def test_avoid_boosts_drift(self):
        """回避性转移应提升漂移检测敏感度"""
        adj = get_sensitivity_adjustments(
            "算了没什么",
            context={
                "conversation_history": [
                    {"role": "user", "content": "我很绝望"},
                ],
                "risk_level": "medium",
            },
        )
        assert adj["drift_boost"] > 1.0

    def test_no_disguise_no_boost(self):
        """无掩饰性表达时不应调整敏感度"""
        adj = get_sensitivity_adjustments("今天天气不错")
        assert adj["crisis_delay_boost"] == 1.0
        assert adj["drift_boost"] == 1.0

    def test_detected_patterns_in_adjustments(self):
        """调整结果应包含检测到的模式信息"""
        adj = get_sensitivity_adjustments("死了算了哈哈")
        assert len(adj["detected_patterns"]) > 0
        pattern = adj["detected_patterns"][0]
        assert "pattern_type" in pattern
        assert "confidence" in pattern
        assert "explanation" in pattern


# ============================================================
# 与审计器集成测试
# ============================================================

class TestAuditorIntegration:
    """验证语用检测与审计器的集成"""

    def test_auditor_runs_teen_precheck(self):
        """审计器在 is_teen=True 时应运行语用预检"""
        from intervention.auditor import AuditContext, SafetyAuditor

        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="我理解你的感受",
            user_input="死了算了哈哈",
            is_teen=True,
            dialog_state="CRISIS",
            crisis_turns=2,
        )
        verdict = auditor.audit(context)
        # 语用预检应提升危机延迟敏感度
        # 使得 max_turns 从 2 降低到 1
        # crisis_turns=2 > 1 → 触发
        assert verdict.passed is False

    def test_auditor_no_precheck_for_adult(self):
        """审计器在 is_teen=False 时不应运行语用预检"""
        from intervention.auditor import AuditContext, SafetyAuditor

        auditor = SafetyAuditor()
        context = AuditContext(
            llm_output="我理解你的感受",
            user_input="死了算了哈哈",
            is_teen=False,  # 成人
            dialog_state="CRISIS",
            crisis_turns=2,
        )
        verdict = auditor.audit(context)
        # 成人模式下 max_turns 保持 2
        # crisis_turns=2 不大于 2 → 通用规则通过
        # 但 _audit_crisis_delay 中 teen check 不运行（is_teen=False）
        assert verdict.passed is True

    def test_auditor_interface_unchanged(self):
        """审计器外部接口不应改变"""
        from intervention.auditor import AuditContext, SafetyAuditor

        auditor = SafetyAuditor()
        # 不传 user_input 和 is_teen 也应正常工作
        context = AuditContext(
            llm_output="测试回复",
            dialog_state="EXPLORE",
        )
        verdict = auditor.audit(context)
        assert verdict.passed is True


# ============================================================
# 检测报告生成
# ============================================================

class TestDetectionReport:
    """生成检测报告（使用 pytest -s 运行时输出）"""

    def test_generate_detection_report(self, detector):
        """生成并输出检测报告"""
        report_lines = [
            "=" * 70,
            "青少年语用特征检测报告",
            "=" * 70,
            "",
            f"{'序号':>4} | {'模式':^12} | {'置信度':>6} | {'建议动作':^14} | 文本",
            "-" * 70,
        ]

        for i, (text, context, expected, desc) in enumerate(TEEN_CORPUS, 1):
            result = detect_teen_disguise(text, context)
            pattern_str = (
                result.pattern_type.value if result.detected else "-"
            )
            conf_str = (
                f"{result.confidence:.2f}" if result.detected else "-"
            )
            action_str = (
                result.suggested_action.value if result.detected else "-"
            )
            match_mark = "[OK]" if (
                (expected is None and not result.detected)
                or (expected is not None and result.pattern_type == expected)
            ) else "[X]"

            report_lines.append(
                f"{i:>4} | {pattern_str:^12} | {conf_str:>6} | "
                f"{action_str:^14} | {match_mark} {text[:30]}"
            )

        report_lines.extend([
            "-" * 70,
            "",
            "检测说明：",
            "  [OK] = 检测结果与预期一致",
            "  [X]  = 检测结果与预期不一致",
            "",
            "临床依据：",
            "  - WHO Adolescent Mental Health Technical Brief (2022)",
            "  - Michelmore et al. (2022) Journal of Adolescent Health",
            "  - NICE CG185 (2011) Self-harm and Suicide Prevention",
            "  - King et al. (2021) Youth Suicide Prevention",
            "  - Miller & Rollnick (2013) Motivational Interviewing",
            "=" * 70,
        ])

        report = "\n".join(report_lines)
        print("\n" + report)

        # 验证报告生成成功
        assert len(report_lines) > 10
        assert "青少年语用特征检测报告" in report
