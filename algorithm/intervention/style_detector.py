"""
用户表达风格检测器 —— AI 回复节奏匹配

功能：
    1. 分析用户前 N 轮消息的表达特征
    2. 输出用户风格分类（INTROVERTED / EXTROVERTED / AGITATED / CALM）
    3. 为回复生成提供风格参数

风格分类依据：
    - INTROVERTED（内向型）：短消息、少标点、少提问、低情绪表达
    - EXTROVERTED（外向型）：长消息、多标点、多提问、高情绪表达
    - AGITATED（激动型）：极短/极快消息、高情绪强度、多感叹号
    - CALM（平稳型）：中等长度、正常标点、情绪稳定

设计依据：
    - 动机式访谈 (MI) 强调"跟随来访者节奏"
    - 青少年沟通风格差异大：内向者需要更多空间，外向者需要更多互动
    - 不匹配的节奏会让用户感到压迫（内向）或敷衍（外向）

约束：
    - 风格检测基于前 3 轮消息，实时更新
    - 不改变五轴审计逻辑
    - 不存储风格数据（仅当次对话使用）
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# 用户风格枚举
# ============================================================

class UserStyle(str, Enum):
    """用户表达风格分类

    四种风格对应不同的回复节奏：
    - INTROVERTED: 短回复、少提问、多陪伴
    - EXTROVERTED: 长回复、多提问、多互动
    - AGITATED: 极短回复、情绪反映为主、不提问
    - CALM: 正常节奏、可引入引导
    """
    INTROVERTED = "INTROVERTED"
    EXTROVERTED = "EXTROVERTED"
    AGITATED = "AGITATED"
    CALM = "CALM"


# ============================================================
# 风格特征数据
# ============================================================

@dataclass
class StyleFeatures:
    """用户表达风格特征

    从用户消息中提取的量化特征。

    Attributes:
        avg_message_length: 平均消息长度（字符数）
        avg_punctuation_per_sentence: 平均每句标点数
        avg_emotion_intensity: 平均情绪强度 (0-1)
        question_rate: 提问频率 (0-1)
        exclamation_rate: 感叹号使用率 (0-1)
        ellipsis_rate: 省略号使用率 (0-1)
        message_count: 分析的消息数量
    """
    avg_message_length: float = 0.0
    avg_punctuation_per_sentence: float = 0.0
    avg_emotion_intensity: float = 0.5
    question_rate: float = 0.0
    exclamation_rate: float = 0.0
    ellipsis_rate: float = 0.0
    message_count: int = 0


# ============================================================
# 风格检测器
# ============================================================

class StyleDetector:
    """用户表达风格检测器

    分析用户消息序列，输出风格分类。

    使用方式：
        detector = StyleDetector()
        style = detector.detect([
            {"role": "user", "content": "嗯"},
            {"role": "user", "content": "还行"},
            {"role": "user", "content": "不知道"},
        ])
        # -> UserStyle.INTROVERTED
    """

    # 风格分类阈值
    _SHORT_MESSAGE_THRESHOLD = 10       # 短消息阈值（字符数）
    _LONG_MESSAGE_THRESHOLD = 25        # 长消息阈值（字符数）
    _HIGH_EMOTION_THRESHOLD = 0.7       # 高情绪强度阈值
    _LOW_EMOTION_THRESHOLD = 0.3        # 低情绪强度阈值
    _HIGH_QUESTION_RATE = 0.3           # 高提问频率阈值
    _LOW_QUESTION_RATE = 0.1            # 低提问频率阈值
    _HIGH_EXCLAMATION_RATE = 0.3        # 高感叹号使用率阈值
    _HIGH_ELLIPSIS_RATE = 0.3           # 高省略号使用率阈值

    def detect(
        self,
        messages: list[dict],
        emotion_intensities: Optional[list[float]] = None,
    ) -> UserStyle:
        """检测用户表达风格

        分析用户前 3 轮消息的特征，输出风格分类。

        Args:
            messages: 对话消息列表 [{"role": "user", "content": "..."}]
            emotion_intensities: 每轮的情绪强度列表 (0-1)

        Returns:
            UserStyle: 检测到的用户风格
        """
        # 提取用户消息
        user_messages = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "user"
        ]

        if not user_messages:
            return UserStyle.CALM  # 默认平稳

        # 只分析前 3 轮
        user_messages = user_messages[:3]

        # 提取特征
        features = self._extract_features(user_messages, emotion_intensities)

        # 分类
        return self._classify(features)

    def _extract_features(
        self,
        messages: list[str],
        emotion_intensities: Optional[list[float]] = None,
    ) -> StyleFeatures:
        """提取风格特征

        Args:
            messages: 用户消息文本列表
            emotion_intensities: 情绪强度列表

        Returns:
            StyleFeatures: 提取的特征
        """
        if not messages:
            return StyleFeatures()

        lengths = [len(m) for m in messages]
        avg_length = sum(lengths) / len(lengths)

        # 标点统计
        total_punctuation = sum(self._count_punctuation(m) for m in messages)
        total_sentences = max(1, sum(self._count_sentences(m) for m in messages))
        avg_punct_per_sentence = total_punctuation / total_sentences

        # 情绪强度
        if emotion_intensities and len(emotion_intensities) >= len(messages):
            avg_emotion = sum(emotion_intensities[:len(messages)]) / len(messages)
        else:
            avg_emotion = 0.5  # 默认中等

        # 提问频率
        question_count = sum(1 for m in messages if self._is_question(m))
        question_rate = question_count / len(messages)

        # 感叹号使用率
        exclamation_count = sum(m.count("！") + m.count("!") for m in messages)
        total_chars = max(1, sum(len(m) for m in messages))
        exclamation_rate = exclamation_count / total_chars

        # 省略号使用率
        ellipsis_count = sum(m.count("...") + m.count("…") + m.count("。。") for m in messages)
        ellipsis_rate = ellipsis_count / len(messages)

        return StyleFeatures(
            avg_message_length=avg_length,
            avg_punctuation_per_sentence=avg_punct_per_sentence,
            avg_emotion_intensity=avg_emotion,
            question_rate=question_rate,
            exclamation_rate=exclamation_rate,
            ellipsis_rate=ellipsis_rate,
            message_count=len(messages),
        )

    def _classify(self, features: StyleFeatures) -> UserStyle:
        """根据特征分类风格

        分类优先级：
        1. AGITATED: 高情绪强度 + 高感叹号/短消息
        2. INTROVERTED: 短消息 + 低提问 + 高省略号
        3. EXTROVERTED: 长消息 + 高提问 + 高感叹号
        4. CALM: 其他

        Args:
            features: 风格特征

        Returns:
            UserStyle: 分类结果
        """
        # AGITATED: 高情绪 + 激动表达
        if (features.avg_emotion_intensity >= self._HIGH_EMOTION_THRESHOLD
                and (features.exclamation_rate >= self._HIGH_EXCLAMATION_RATE
                     or features.avg_message_length <= self._SHORT_MESSAGE_THRESHOLD)):
            return UserStyle.AGITATED

        # INTROVERTED: 短消息 + 少互动
        if (features.avg_message_length <= self._SHORT_MESSAGE_THRESHOLD
                and features.question_rate <= self._LOW_QUESTION_RATE
                and features.ellipsis_rate >= self._HIGH_ELLIPSIS_RATE):
            return UserStyle.INTROVERTED

        # 也判断为内向：短消息 + 低情绪 + 少提问
        if (features.avg_message_length <= self._SHORT_MESSAGE_THRESHOLD
                and features.avg_emotion_intensity <= self._LOW_EMOTION_THRESHOLD
                and features.question_rate <= self._LOW_QUESTION_RATE):
            return UserStyle.INTROVERTED

        # EXTROVERTED: 长消息 + 多互动
        if (features.avg_message_length >= self._LONG_MESSAGE_THRESHOLD
                and features.question_rate >= self._HIGH_QUESTION_RATE):
            return UserStyle.EXTROVERTED

        # 也判断为外向：长消息 + 高情绪
        if (features.avg_message_length >= self._LONG_MESSAGE_THRESHOLD
                and features.avg_emotion_intensity >= self._HIGH_EMOTION_THRESHOLD):
            return UserStyle.EXTROVERTED

        # 默认 CALM
        return UserStyle.CALM

    @staticmethod
    def _count_punctuation(text: str) -> int:
        """统计标点数量"""
        punctuation = set("，。！？、；：""''（）《》…—～·")
        return sum(1 for c in text if c in punctuation)

    @staticmethod
    def _count_sentences(text: str) -> int:
        """统计句子数"""
        if not text:
            return 1
        # 按句号、问号、感叹号分句
        sentences = re.split(r'[。！？!?]', text)
        return max(1, len([s for s in sentences if s.strip()]))

    @staticmethod
    def _is_question(text: str) -> bool:
        """判断是否为提问"""
        question_markers = ["？", "?", "吗", "呢", "么", "嘛", "怎么", "为什么", "什么", "哪"]
        return any(marker in text for marker in question_markers)


# ============================================================
# 风格 → 回复参数映射
# ============================================================

@dataclass
class ReplyStyleParams:
    """回复风格参数

    根据用户风格生成的回复参数。

    Attributes:
        max_reply_length: 最大回复长度（字符数）
        should_ask_question: 是否应该提问
        question_count: 建议提问数量
        tone: 回复语气（warm/energetic/calm/gentle）
        include_companionship: 是否包含陪伴表达
        include_guidance: 是否包含引导内容
    """
    max_reply_length: int = 50
    should_ask_question: bool = False
    question_count: int = 0
    tone: str = "calm"
    include_companionship: bool = True
    include_guidance: bool = False


def get_reply_style_params(style: UserStyle) -> ReplyStyleParams:
    """根据用户风格获取回复参数

    Args:
        style: 用户风格

    Returns:
        ReplyStyleParams: 回复风格参数
    """
    if style == UserStyle.INTROVERTED:
        return ReplyStyleParams(
            max_reply_length=30,
            should_ask_question=False,
            question_count=0,
            tone="gentle",
            include_companionship=True,
            include_guidance=False,
        )
    elif style == UserStyle.EXTROVERTED:
        return ReplyStyleParams(
            max_reply_length=80,
            should_ask_question=True,
            question_count=2,
            tone="energetic",
            include_companionship=True,
            include_guidance=True,
        )
    elif style == UserStyle.AGITATED:
        return ReplyStyleParams(
            max_reply_length=20,
            should_ask_question=False,
            question_count=0,
            tone="warm",
            include_companionship=True,
            include_guidance=False,
        )
    else:  # CALM
        return ReplyStyleParams(
            max_reply_length=50,
            should_ask_question=True,
            question_count=1,
            tone="calm",
            include_companionship=True,
            include_guidance=True,
        )
