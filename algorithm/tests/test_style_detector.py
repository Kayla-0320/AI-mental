"""
用户表达风格检测器单元测试

测试覆盖：
    1. 风格检测（INTROVERTED / EXTROVERTED / AGITATED / CALM）
    2. 特征提取（消息长度、标点、提问频率等）
    3. 回复风格参数映射
    4. 跨会话场景验证（内向/外向用户模拟）
"""
import sys
from pathlib import Path

import pytest

# 添加算法路径
algorithm_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(algorithm_path))

from intervention.style_detector import (
    StyleDetector,
    UserStyle,
    StyleFeatures,
    get_reply_style_params,
    ReplyStyleParams,
)


# ============================================================
# 1. 风格检测测试
# ============================================================

class TestStyleDetection:
    """测试风格检测"""

    def test_introverted_detection(self):
        """内向型检测：短消息 + 少提问 + 多省略号"""
        detector = StyleDetector()
        messages = [
            {"role": "user", "content": "嗯"},
            {"role": "user", "content": "还行"},
            {"role": "user", "content": "不知道..."},
        ]
        style = detector.detect(messages)
        assert style == UserStyle.INTROVERTED

    def test_extroverted_detection(self):
        """外向型检测：长消息 + 多提问"""
        detector = StyleDetector()
        messages = [
            {"role": "user", "content": "今天在学校发生了好多事情你想知道发生了什么吗我觉得特别有意思"},
            {"role": "user", "content": "后来我们又去了另一个地方你猜怎么着我觉得太有意思了你觉得呢"},
            {"role": "user", "content": "然后我们还讨论了一个问题你觉得应该怎么做我想听听你的想法"},
        ]
        style = detector.detect(messages)
        assert style == UserStyle.EXTROVERTED

    def test_agitated_detection(self):
        """激动型检测：高情绪 + 感叹号"""
        detector = StyleDetector()
        messages = [
            {"role": "user", "content": "我受不了了！！"},
            {"role": "user", "content": "太过分了！！！"},
            {"role": "user", "content": "我真的好烦！"},
        ]
        style = detector.detect(messages, emotion_intensities=[0.9, 0.9, 0.9])
        assert style == UserStyle.AGITATED

    def test_calm_detection(self):
        """平稳型检测：中等长度 + 正常表达"""
        detector = StyleDetector()
        messages = [
            {"role": "user", "content": "今天感觉还可以，没什么特别的。"},
            {"role": "user", "content": "就是正常上课，然后回了家。"},
            {"role": "user", "content": "晚上写了一会儿作业。"},
        ]
        style = detector.detect(messages)
        assert style == UserStyle.CALM

    def test_empty_messages_returns_calm(self):
        """空消息返回 CALM"""
        detector = StyleDetector()
        style = detector.detect([])
        assert style == UserStyle.CALM

    def test_only_analyzes_first_3_turns(self):
        """只分析前 3 轮"""
        detector = StyleDetector()
        messages = [
            {"role": "user", "content": "嗯..."},  # 内向
            {"role": "user", "content": "还行..."},
            {"role": "user", "content": "不知道..."},
            {"role": "user", "content": "今天在学校发生了好多事情你想知道发生了什么吗我觉得特别有意思"},  # 外向（不应被分析）
        ]
        style = detector.detect(messages)
        assert style == UserStyle.INTROVERTED


# ============================================================
# 2. 特征提取测试
# ============================================================

class TestFeatureExtraction:
    """测试特征提取"""

    def test_short_message_length(self):
        """短消息长度检测"""
        detector = StyleDetector()
        features = detector._extract_features(["嗯", "好", "行"])
        assert features.avg_message_length <= 10

    def test_long_message_length(self):
        """长消息长度检测"""
        detector = StyleDetector()
        long_msg = "今天在学校发生了很多事情我觉得特别有意思你想知道发生了什么吗后来我们又去了另一个地方"
        features = detector._extract_features([long_msg])
        assert features.avg_message_length >= 30

    def test_question_detection(self):
        """提问检测"""
        detector = StyleDetector()
        features = detector._extract_features(["你觉得呢？", "后来发生了什么？"])
        assert features.question_rate >= 0.5

    def test_no_question_detection(self):
        """无提问检测"""
        detector = StyleDetector()
        features = detector._extract_features(["嗯", "还行", "不知道"])
        assert features.question_rate == 0.0

    def test_exclamation_rate(self):
        """感叹号使用率"""
        detector = StyleDetector()
        features = detector._extract_features(["太好了！", "真的吗！"])
        assert features.exclamation_rate > 0

    def test_ellipsis_rate(self):
        """省略号使用率"""
        detector = StyleDetector()
        features = detector._extract_features(["嗯...", "不知道..."])
        assert features.ellipsis_rate >= 0.5


# ============================================================
# 3. 回复风格参数测试
# ============================================================

class TestReplyStyleParams:
    """测试回复风格参数"""

    def test_introverted_params(self):
        """内向型回复参数"""
        params = get_reply_style_params(UserStyle.INTROVERTED)
        assert params.max_reply_length == 30
        assert params.should_ask_question is False
        assert params.question_count == 0
        assert params.tone == "gentle"
        assert params.include_companionship is True
        assert params.include_guidance is False

    def test_extroverted_params(self):
        """外向型回复参数"""
        params = get_reply_style_params(UserStyle.EXTROVERTED)
        assert params.max_reply_length == 80
        assert params.should_ask_question is True
        assert params.question_count == 2
        assert params.tone == "energetic"
        assert params.include_guidance is True

    def test_agitated_params(self):
        """激动型回复参数"""
        params = get_reply_style_params(UserStyle.AGITATED)
        assert params.max_reply_length == 20
        assert params.should_ask_question is False
        assert params.question_count == 0
        assert params.tone == "warm"
        assert params.include_guidance is False

    def test_calm_params(self):
        """平稳型回复参数"""
        params = get_reply_style_params(UserStyle.CALM)
        assert params.max_reply_length == 50
        assert params.should_ask_question is True
        assert params.question_count == 1
        assert params.tone == "calm"
        assert params.include_guidance is True


# ============================================================
# 4. 跨会话场景验证
# ============================================================

class TestCrossSessionScenario:
    """跨会话场景验证"""

    def test_introverted_user_scenario(self):
        """内向用户场景：AI 回复变短、变少提问"""
        detector = StyleDetector()

        # 模拟内向用户连续 3 轮短消息
        messages = [
            {"role": "user", "content": "嗯"},
            {"role": "user", "content": "还行"},
            {"role": "user", "content": "不知道..."},
        ]
        style = detector.detect(messages)
        params = get_reply_style_params(style)

        assert style == UserStyle.INTROVERTED
        assert params.max_reply_length <= 30
        assert params.should_ask_question is False

    def test_extroverted_user_scenario(self):
        """外向用户场景：AI 回复变长、变多提问"""
        detector = StyleDetector()

        # 模拟外向用户连续 3 轮长消息
        messages = [
            {"role": "user", "content": "今天在学校发生了好多事情你想知道发生了什么吗我觉得特别有意思"},
            {"role": "user", "content": "后来我们又去了另一个地方你猜怎么着我觉得太有意思了你觉得呢"},
            {"role": "user", "content": "然后我们还讨论了一个问题你觉得应该怎么做我想听听你的想法"},
        ]
        style = detector.detect(messages)
        params = get_reply_style_params(style)

        assert style == UserStyle.EXTROVERTED
        assert params.max_reply_length >= 80
        assert params.should_ask_question is True
        assert params.question_count >= 2


# ============================================================
# 5. 边界条件测试
# ============================================================

class TestEdgeCases:
    """边界条件测试"""

    def test_single_message(self):
        """单条消息"""
        detector = StyleDetector()
        messages = [{"role": "user", "content": "嗯"}]
        style = detector.detect(messages)
        assert style in (UserStyle.INTROVERTED, UserStyle.CALM)

    def test_mixed_roles(self):
        """混合角色（只分析 user）"""
        detector = StyleDetector()
        messages = [
            {"role": "user", "content": "嗯"},
            {"role": "assistant", "content": "我在这里陪着你。"},
            {"role": "user", "content": "还行"},
            {"role": "assistant", "content": "慢慢来。"},
            {"role": "user", "content": "不知道..."},
        ]
        style = detector.detect(messages)
        assert style == UserStyle.INTROVERTED

    def test_with_emotion_intensities(self):
        """带情绪强度"""
        detector = StyleDetector()
        messages = [
            {"role": "user", "content": "我受不了了！！"},
            {"role": "user", "content": "太过分了！！！"},
        ]
        style = detector.detect(messages, emotion_intensities=[0.9, 0.9])
        assert style == UserStyle.AGITATED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
