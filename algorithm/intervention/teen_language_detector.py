"""
青少年语用特征检测器 —— 识别青少年掩饰性心理困扰表达

五类掩饰性表达模式：
    1. 反话模式 (COUNTER)：
       "我没事""我好得很""不用担心我"
       → 当上下文含负面信号时判定为掩饰
    2. 轻描淡写 (MINIMIZE)：
       "就那样吧""还行""无所谓"
       → 当上下文含危机词时判定为掩饰
    3. 玩笑化危机 (JOKE_CRISIS)：
       "死了算了哈哈""想跳楼但不敢"
       → 危机词 + 幽默标记组合
    4. 试探性表达 (PROBE)：
       "如果我消失了会怎样""你会记得我吗"
       → 间接表达自杀/分离意念
    5. 回避性转移 (AVOID)：
       "不说这个了""算了没什么"
       → 话题突然从负面内容转移

临床依据：
    - 青少年约 70% 的心理困扰以间接/掩饰形式表达
      (Michelmore et al., 2022, Journal of Adolescent Health)
    - "反话式求助"是青少年特有的沟通模式
      (WHO, 2022, Adolescent Mental Health Technical Brief)
    - 掩饰性表达被通用检测器漏判是青少年危机干预失败的主因之一
      (NICE CG185, 2011, Self-harm and Suicide Prevention)

设计原则：
    - 不做纯关键词匹配，必须结合上下文判断
    - 每条检测结果附带可解释理由
    - 与 auditor.py 集成时不改变其外部接口
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# 掩饰性表达模式枚举
# ============================================================

class DisguisePatternType(str, Enum):
    """五类青少年掩饰性表达模式"""
    COUNTER = "counter"           # 反话模式
    MINIMIZE = "minimize"         # 轻描淡写
    JOKE_CRISIS = "joke_crisis"   # 玩笑化危机
    PROBE = "probe"               # 试探性表达
    AVOID = "avoid"               # 回避性转移


class SuggestedAction(str, Enum):
    """建议动作"""
    PASS = "pass"                         # 无需特殊处理
    GENTLE_PROBE = "gentle_probe"         # 温和探询（"你好像有些不对劲"）
    SAFETY_CONFIRM = "safety_confirm"     # 安全确认（"你现在安全吗？"）
    ESCALATE = "escalate"                 # 升级至人工


# ============================================================
# 检测结果数据类
# ============================================================

@dataclass
class TeenLanguageResult:
    """青少年语用检测结果

    Attributes:
        detected: 是否检测到掩饰性表达
        pattern_type: 检测到的掩饰模式类型（未检测到时为 None）
        confidence: 检测置信度 [0.0, 1.0]
        suggested_action: 建议动作
        explanation: 可解释的判定理由（面向咨询师/审计员）
        matched_text: 触发检测的具体文本片段
        context_signals: 上下文中支撑判定的信号列表
        sensitivity_boost: 建议对审计轴提升的敏感度系数
    """
    detected: bool = False
    pattern_type: Optional[DisguisePatternType] = None
    confidence: float = 0.0
    suggested_action: SuggestedAction = SuggestedAction.PASS
    explanation: str = ""
    matched_text: str = ""
    context_signals: list[str] = field(default_factory=list)
    sensitivity_boost: float = 1.0


# ============================================================
# 上下文信号词库
# ============================================================

# 负面情感信号词（用于判断"我没事"是否为反话）
_NEGATIVE_SIGNALS: list[str] = [
    # 情绪类
    "难过", "伤心", "哭", "崩溃", "绝望", "痛苦", "难受",
    "焦虑", "害怕", "恐惧", "孤独", "无助", "绝望",
    "生气", "愤怒", "烦", "累", "疲惫",
    # 危机类
    "死", "自杀", "不想活", "消失", "跳楼", "割腕",
    "没意义", "没意思", "活着没意思", "世界没我会更好",
    # 社交类
    "被欺负", "被排挤", "被孤立", "霸凌", "没人理",
    "没朋友", "没人爱我", "多余", "废物",
    # 行为类
    "失眠", "睡不着", "不想吃", "不想出门", "不想说话",
    "逃课", "退步", "成绩下降",
    # 伤害类
    "割腕", "手腕", "自伤", "受伤", "批评",
    "嘲笑", "辱骂", "打骂", "背叛", "抛弃",
]

# 正面语境信号词（用于排除"我没事"在正面语境下的正常表达）
_POSITIVE_SIGNALS: list[str] = [
    "开心", "高兴", "快乐", "棒", "太好了", "好极了",
    "兴奋", "期待", "喜欢", "感恩", "幸福", "满足",
    "通过", "成功", "进步", "改善", "好转",
]

# 幽默标记（用于玩笑化危机检测）
_HUMOR_MARKERS: list[str] = [
    "哈", "哈哈", "哈哈哈", "嘿嘿", "嘻嘻",
    "笑死", "lol", "lmao", "233", "666",
    " joking", "开玩笑", "搞笑",
]

# 话题回避标记（用于回避性转移检测）
_AVOIDANCE_MARKERS: list[str] = [
    "不说这个", "算了", "没什么", "没事了", "别问了",
    "不想说", "跳过", "换个话题", "不重要",
    "算了没什么", "没啥", "忘了",
]


# ============================================================
# 青少年语用检测器
# ============================================================

class TeenLanguageDetector:
    """青少年语用特征检测器

    结合文本模式匹配与上下文信号分析，
    识别青少年的掩饰性心理困扰表达。

    使用方式：
        detector = TeenLanguageDetector()
        result = detector.detect(
            text="我没事，不用担心",
            context={
                "recent_emotions": ["悲伤", "焦虑"],
                "risk_level": "medium",
                "conversation_history": [
                    {"role": "user", "content": "今天被同学嘲笑了"},
                ],
            },
        )
        if result.detected:
            print(result.explanation)
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}

    def detect(
        self,
        text: str,
        context: Optional[dict] = None,
    ) -> TeenLanguageResult:
        """检测文本中的掩饰性表达

        Args:
            text: 待检测文本（用户输入）
            context: 上下文信息字典，可包含：
                - recent_emotions: list[str] 近期情绪标签
                - risk_level: str 当前风险等级
                - conversation_history: list[dict] 对话历史
                - previous_topic: str 之前讨论的话题

        Returns:
            TeenLanguageResult: 检测结果
        """
        if context is None:
            context = {}

        # 提取上下文信号
        ctx_signals = self._extract_context_signals(text, context)

        # 依次检测五类模式（按严重程度排序，优先检测最危险的）
        detectors = [
            self._detect_joke_crisis,
            self._detect_probe,
            self._detect_counter,
            self._detect_minimize,
            self._detect_avoid,
        ]

        best_result = TeenLanguageResult()

        for detect_fn in detectors:
            result = detect_fn(text, ctx_signals, context)
            if result.detected and result.confidence > best_result.confidence:
                best_result = result

        return best_result

    # ========================================================
    # 上下文信号提取
    # ========================================================

    def _extract_context_signals(
        self, text: str, context: dict
    ) -> dict:
        """从上下文中提取辅助判定信号

        Returns:
            dict 包含：
                - has_negative_context: bool
                - has_positive_context: bool
                - has_recent_crisis: bool
                - negative_signals: list[str]
                - topic_shift_from_negative: bool
        """
        # 合并所有可分析文本
        all_context_text = text.lower()

        # 从对话历史提取
        history = context.get("conversation_history", [])
        recent_user_texts = []
        for turn in history[-6:]:  # 最近 6 轮
            if turn.get("role") == "user":
                content = turn.get("content", "")
                recent_user_texts.append(content)
                all_context_text += " " + content

        # 从近期情绪提取
        recent_emotions = context.get("recent_emotions", [])
        negative_emotions = {"悲伤", "焦虑", "愤怒", "恐惧"}
        has_negative_emotion = bool(
            negative_emotions & set(recent_emotions)
        )

        # 检测负面信号
        negative_signals = [
            sig for sig in _NEGATIVE_SIGNALS if sig in all_context_text
        ]
        has_negative_context = (
            len(negative_signals) > 0 or has_negative_emotion
        )

        # 检测正面信号
        positive_signals = [
            sig for sig in _POSITIVE_SIGNALS if sig in all_context_text
        ]
        has_positive_context = len(positive_signals) > 0

        # 检测近期危机信号
        crisis_keywords = [
            "死", "自杀", "不想活", "消失", "跳楼", "割腕",
            "没意义", "没意思",
        ]
        has_recent_crisis = any(
            kw in all_context_text for kw in crisis_keywords
        )

        # 检测话题是否从负面内容转移
        topic_shift = False
        if len(recent_user_texts) >= 2:
            prev_text = recent_user_texts[-1] if len(recent_user_texts) >= 1 else ""
            prev_has_negative = any(
                sig in prev_text for sig in _NEGATIVE_SIGNALS
            )
            current_has_avoidance = any(
                marker in text for marker in _AVOIDANCE_MARKERS
            )
            topic_shift = prev_has_negative and current_has_avoidance

        # 风险等级
        risk_level = context.get("risk_level", "low")

        return {
            "has_negative_context": has_negative_context,
            "has_positive_context": has_positive_context,
            "has_recent_crisis": has_recent_crisis,
            "negative_signals": negative_signals,
            "has_negative_emotion": has_negative_emotion,
            "topic_shift_from_negative": topic_shift,
            "risk_level": risk_level,
            "recent_user_texts": recent_user_texts,
        }

    # ========================================================
    # 模式 1：反话检测
    # ========================================================

    def _detect_counter(
        self, text: str, signals: dict, context: dict
    ) -> TeenLanguageResult:
        """反话模式检测

        临床依据：
            青少年在经历心理困扰时，常以"我没事"等反话表达，
            实际含义与字面相反。当上下文存在负面信号时，
            "我没事"极可能是求助信号。
            来源：WHO Adolescent Mental Health Technical Brief (2022)

        判定逻辑：
            1. 文本包含反话标记短语
            2. 且上下文存在负面信号（对话历史/情绪/风险等级）
            3. 且上下文不存在强正面信号（排除真正的"我很好"）
        """
        counter_patterns = [
            r"我没事", r"我[还都]好", r"我好得很",
            r"不用担心我", r"我[没没]怎样",
            r"我[挺很]好的", r"我真的[很挺]好",
            r"没[什么啥]事",
            r"我[一丁]点事[都没没]",
            r"我才没有(难过|伤心|不开心|焦虑)",
            r"我[真的完全]没[有]不开心",
            r"不用担心",
        ]

        matched = None
        for pattern in counter_patterns:
            m = re.search(pattern, text)
            if m:
                matched = m.group()
                break

        if not matched:
            return TeenLanguageResult()

        # 关键：上下文判断
        # 如果有强正面语境，"我没事"可能是真的
        if signals["has_positive_context"] and not signals["has_negative_context"]:
            return TeenLanguageResult()

        # 有负面上下文 → 判定为反话
        if signals["has_negative_context"]:
            neg_examples = signals["negative_signals"][:3]
            confidence = 0.7
            # 风险等级越高，置信度越高
            if signals["risk_level"] in ("medium", "high", "crisis"):
                confidence += 0.15
            # 有近期危机信号
            if signals["has_recent_crisis"]:
                confidence += 0.1
            confidence = min(confidence, 0.95)

            return TeenLanguageResult(
                detected=True,
                pattern_type=DisguisePatternType.COUNTER,
                confidence=round(confidence, 2),
                suggested_action=SuggestedAction.GENTLE_PROBE,
                explanation=(
                    f"检测到反话模式：'{matched}'。"
                    f"上下文存在负面信号（{', '.join(neg_examples)}），"
                    f"该表达可能为掩饰性求助。"
                    f"临床依据：WHO Adolescent Mental Health Technical Brief (2022) —— "
                    f"青少年约 70% 的心理困扰以间接/掩饰形式表达。"
                ),
                matched_text=matched,
                context_signals=[
                    f"负面信号词: {', '.join(neg_examples)}",
                    f"风险等级: {signals['risk_level']}",
                ],
                sensitivity_boost=1.3,
            )

        # 无明确上下文但有反话标记 → 低置信度提示
        return TeenLanguageResult(
            detected=True,
            pattern_type=DisguisePatternType.COUNTER,
            confidence=0.4,
            suggested_action=SuggestedAction.GENTLE_PROBE,
            explanation=(
                f"检测到可能的反话模式：'{matched}'。"
                f"上下文信号不足，置信度较低，建议关注。"
            ),
            matched_text=matched,
            context_signals=["无明确负面上下文"],
            sensitivity_boost=1.1,
        )

    # ========================================================
    # 模式 2：轻描淡写检测
    # ========================================================

    def _detect_minimize(
        self, text: str, signals: dict, context: dict
    ) -> TeenLanguageResult:
        """轻描淡写模式检测

        临床依据：
            青少年倾向于淡化自己的痛苦程度，使用"还行""就那样"
            等表达掩盖真实的心理困扰。当上下文存在危机词时，
            这种轻描淡写尤其危险——可能掩盖自杀意念。
            来源：Michelmore et al. (2022), Journal of Adolescent Health

        判定逻辑：
            1. 文本包含轻描淡写标记
            2. 且上下文存在危机词或高风险信号
        """
        minimize_patterns = [
            r"就那样[吧了]", r"就那样", r"还[行好]", r"无所谓",
            r"一[般点]吧", r"一般般", r"马马虎虎", r"凑合",
            r"就[这那]样", r"一[切切]都[好行]",
            r"不[太怎么]样", r"普普通通",
            r"就[是]那个样子",
        ]

        matched = None
        for pattern in minimize_patterns:
            m = re.search(pattern, text)
            if m:
                matched = m.group()
                break

        if not matched:
            return TeenLanguageResult()

        # 轻描淡写需要危机上下文才有意义
        # 如果上下文无危机信号，"还行"可能就是字面意思
        if not signals["has_negative_context"] and not signals["has_recent_crisis"]:
            return TeenLanguageResult()

        confidence = 0.6
        if signals["has_recent_crisis"]:
            confidence += 0.2
        if signals["risk_level"] in ("high", "crisis"):
            confidence += 0.1
        confidence = min(confidence, 0.95)

        neg_examples = signals["negative_signals"][:3]

        return TeenLanguageResult(
            detected=True,
            pattern_type=DisguisePatternType.MINIMIZE,
            confidence=round(confidence, 2),
            suggested_action=SuggestedAction.GENTLE_PROBE,
            explanation=(
                f"检测到轻描淡写模式：'{matched}'。"
                f"上下文存在危机信号（{', '.join(neg_examples) if neg_examples else '风险等级偏高'}），"
                f"该表达可能在淡化真实痛苦程度。"
                f"临床依据：Michelmore et al. (2022) —— "
                f"青少年常以'还行''无所谓'掩盖真实心理困扰。"
            ),
            matched_text=matched,
            context_signals=[
                f"危机/负面信号: {', '.join(neg_examples) if neg_examples else '风险等级偏高'}",
                f"风险等级: {signals['risk_level']}",
            ],
            sensitivity_boost=1.2,
        )

    # ========================================================
    # 模式 3：玩笑化危机检测
    # ========================================================

    def _detect_joke_cisis_impl(
        self, text: str, signals: dict, context: dict
    ) -> TeenLanguageResult:
        """玩笑化危机模式检测（内部实现）"""
        # 危机关键词
        crisis_keywords = [
            "死", "自杀", "不想活", "跳楼", "割腕",
            "消失", "死了算了", "活着没意思", "去死",
            "世界没我会更好", "没了我会",
        ]

        # 检测危机词
        found_crisis = [kw for kw in crisis_keywords if kw in text]
        if not found_crisis:
            return TeenLanguageResult()

        # 检测幽默标记
        found_humor = [m for m in _HUMOR_MARKERS if m in text]

        # 也检测"轻描淡写式"的非幽默包装（如"想跳楼但不敢"）
        casual_markers = [
            r"但不敢", r"只是想想", r"开玩笑[的啦]",
            r"随便说说", r"说说而已", r"想想而已",
        ]
        found_casual = [
            m for m in casual_markers if re.search(m, text)
        ]

        if not found_humor and not found_casual:
            return TeenLanguageResult()

        # 玩笑化危机 → 高危险信号，直接高置信度
        confidence = 0.85
        if signals["has_recent_crisis"]:
            confidence += 0.05
        if signals["risk_level"] in ("high", "crisis"):
            confidence += 0.05
        confidence = min(confidence, 0.98)

        matched_parts = found_crisis + (
            found_humor if found_humor else found_casual
        )

        return TeenLanguageResult(
            detected=True,
            pattern_type=DisguisePatternType.JOKE_CRISIS,
            confidence=round(confidence, 2),
            suggested_action=SuggestedAction.SAFETY_CONFIRM,
            explanation=(
                f"检测到玩笑化危机模式：文本包含危机词"
                f"（{', '.join(found_crisis)}）"
                f"{'与幽默标记' if found_humor else '与轻描淡写包装'}"
                f"（{', '.join(found_humor or found_casual)}）组合。"
                f"青少年常以玩笑方式表达真实危机意念，不可忽略。"
                f"临床依据：NICE CG185 (2011) —— "
                f"青少年危机信号常以幽默/间接方式出现，通用检测器易漏判。"
            ),
            matched_text=" + ".join(matched_parts),
            context_signals=[
                f"危机词: {', '.join(found_crisis)}",
                f"幽默/ casual 标记: {', '.join(found_humor or found_casual)}",
            ],
            sensitivity_boost=1.5,
        )

    # 别名保持调用一致
    _detect_joke_crisis = _detect_joke_cisis_impl

    # ========================================================
    # 模式 4：试探性表达检测
    # ========================================================

    def _detect_probe(
        self, text: str, signals: dict, context: dict
    ) -> TeenLanguageResult:
        """试探性表达模式检测

        临床依据：
            青少年在产生自杀意念时，常以假设性/试探性问题
            试探周围人的反应，如"如果我消失了会怎样"。
            这类表达是重要的预警信号，不应被当作"随便问问"。
            来源：NICE CG185 (2011);
            King et al. (2021) Youth Suicide Prevention.

        判定逻辑：
            1. 文本匹配试探性表达模式（假设性 + 消失/死亡/分离）
            2. 无需额外上下文——此类表达本身即为高危信号
        """
        probe_patterns = [
            # 假设性消失/死亡
            r"如果我消失[了的话]",
            r"要是我[不在没]了",
            r"如果[我].*死[了去]",
            r"如果.*没[有]我",
            r"你会[不]?记得我",
            r"有人会在意我[吗嘛]",
            r"没[有]?人会在意我[吗嘛]",
            r"如果.*从.*消失",
            r"要是.*不[在存]",
            r"如果我.*出[了事]",
            # 试探性告别
            r"你会想我[吗嘛]",
            r"以后.*还[能会]见[到到]",
            r"最后.*[一次].*",
            r"谢谢你.*陪[我]",
            # 间接求助
            r"有没有人.*在.*意我",
            r"我.*是.*不.*是.*多余[的]",
        ]

        matched = None
        for pattern in probe_patterns:
            m = re.search(pattern, text)
            if m:
                matched = m.group()
                break

        if not matched:
            return TeenLanguageResult()

        # 试探性表达本身就是高危信号
        confidence = 0.8
        if signals["has_negative_context"]:
            confidence += 0.1
        if signals["risk_level"] in ("high", "crisis"):
            confidence += 0.05
        confidence = min(confidence, 0.95)

        return TeenLanguageResult(
            detected=True,
            pattern_type=DisguisePatternType.PROBE,
            confidence=round(confidence, 2),
            suggested_action=SuggestedAction.SAFETY_CONFIRM,
            explanation=(
                f"检测到试探性表达模式：'{matched}'。"
                f"此类假设性/试探性表达可能是自杀意念的间接预警信号，"
                f"不应被当作'随便问问'。"
                f"临床依据：NICE CG185 (2011); "
                f"King et al. (2021) Youth Suicide Prevention —— "
                f"青少年常以假设性问题试探他人反应。"
            ),
            matched_text=matched,
            context_signals=[
                f"风险等级: {signals['risk_level']}",
            ] + ([f"负面上下文: 是"] if signals["has_negative_context"] else []),
            sensitivity_boost=1.5,
        )

    # ========================================================
    # 模式 5：回避性转移检测
    # ========================================================

    def _detect_avoid(
        self, text: str, signals: dict, context: dict
    ) -> TeenLanguageResult:
        """回避性转移模式检测

        临床依据：
            青少年在触及痛苦话题时，可能突然转移话题或
            否认之前的表达（"算了没什么""不说这个了"）。
            这种回避行为本身提示该话题对其有重大影响。
            来源：Motivated Interview (Miller & Rollnick, 2013) ——
            回避是"阻抗"的信号，治疗师应温和地回到该话题。

        判定逻辑：
            1. 文本包含回避标记
            2. 且对话历史中之前存在负面内容（话题从负面转移走）
        """
        matched = None
        for marker in _AVOIDANCE_MARKERS:
            if marker in text:
                matched = marker
                break

        if not matched:
            return TeenLanguageResult()

        # 需要上下文判断：之前是否在讨论负面话题
        # 情况 A: 对话历史中有负面内容 → 回避
        if signals["topic_shift_from_negative"]:
            return TeenLanguageResult(
                detected=True,
                pattern_type=DisguisePatternType.AVOID,
                confidence=0.75,
                suggested_action=SuggestedAction.GENTLE_PROBE,
                explanation=(
                    f"检测到回避性转移模式：'{matched}'。"
                    f"对话历史中存在负面内容，当前文本突然转移话题，"
                    f"提示该话题对来访者有重大心理意义。"
                    f"临床依据：Miller & Rollnick (2013) Motivational Interviewing —— "
                    f"回避是'阻抗'信号，应温和地回到该话题。"
                ),
                matched_text=matched,
                context_signals=[
                    "对话历史中存在负面内容",
                    "话题从负面内容转移",
                ],
                sensitivity_boost=1.2,
            )

        # 情况 B: 风险等级偏高 → 回避也可能有意义
        if signals["risk_level"] in ("medium", "high", "crisis"):
            return TeenLanguageResult(
                detected=True,
                pattern_type=DisguisePatternType.AVOID,
                confidence=0.55,
                suggested_action=SuggestedAction.GENTLE_PROBE,
                explanation=(
                    f"检测到可能的回避性转移：'{matched}'。"
                    f"当前风险等级为 {signals['risk_level']}，"
                    f"回避行为可能暗示未表达的困扰。"
                ),
                matched_text=matched,
                context_signals=[
                    f"风险等级: {signals['risk_level']}",
                ],
                sensitivity_boost=1.1,
            )

        # 情况 C: 无负面上下文且低风险 → 可能只是正常转移
        return TeenLanguageResult()


# ============================================================
# 便捷函数
# ============================================================

def detect_teen_disguise(
    text: str,
    context: Optional[dict] = None,
) -> TeenLanguageResult:
    """检测青少年掩饰性表达的便捷函数

    Args:
        text: 待检测文本
        context: 上下文信息

    Returns:
        TeenLanguageResult: 检测结果
    """
    detector = TeenLanguageDetector()
    return detector.detect(text, context)


# ============================================================
# 审计器集成：敏感度提升系数
# ============================================================

def get_sensitivity_adjustments(
    text: str,
    context: Optional[dict] = None,
) -> dict:
    """获取语用检测对五轴审计敏感度的调整建议

    供 auditor.py 在五轴审计前调用，
    根据检测结果动态调整各轴的检测阈值。

    Args:
        text: 用户输入文本
        context: 上下文信息

    Returns:
        dict 各轴敏感度提升系数：
            crisis_delay_boost: 危机延迟敏感度系数
            delusion_boost: 妄想强化敏感度系数
            stigma_boost: 污名化敏感度系数
            sycophancy_boost: 谄媚敏感度系数
            drift_boost: 漂移敏感度系数
            detected_patterns: list[dict] 检测到的模式列表
    """
    result = detect_teen_disguise(text, context)

    adjustments = {
        "crisis_delay_boost": 1.0,
        "delusion_boost": 1.0,
        "stigma_boost": 1.0,
        "sycophancy_boost": 1.0,
        "drift_boost": 1.0,
        "detected_patterns": [],
    }

    if not result.detected:
        return adjustments

    adjustments["detected_patterns"].append({
        "pattern_type": result.pattern_type.value,
        "confidence": result.confidence,
        "explanation": result.explanation,
        "matched_text": result.matched_text,
    })

    boost = result.sensitivity_boost

    # 根据模式类型提升对应审计轴
    if result.pattern_type == DisguisePatternType.JOKE_CRISIS:
        # 玩笑化危机 → 大幅提升危机延迟轴敏感度
        adjustments["crisis_delay_boost"] = boost
    elif result.pattern_type == DisguisePatternType.PROBE:
        # 试探性表达 → 提升危机延迟轴
        adjustments["crisis_delay_boost"] = boost
    elif result.pattern_type == DisguisePatternType.COUNTER:
        # 反话 → 提升危机延迟 + 漂移检测
        adjustments["crisis_delay_boost"] = boost
        adjustments["drift_boost"] = boost
    elif result.pattern_type == DisguisePatternType.MINIMIZE:
        # 轻描淡写 → 提升危机延迟 + 漂移检测
        adjustments["crisis_delay_boost"] = boost
        adjustments["drift_boost"] = result.sensitivity_boost
    elif result.pattern_type == DisguisePatternType.AVOID:
        # 回避性转移 → 提升漂移检测
        adjustments["drift_boost"] = boost

    return adjustments
