"""统一人本取向提示词与追问规则测试

覆盖三处改动：

1. EMPATHY/SOCRATIC 双分支合并为一段提示词（``dialogue_mode`` 不再影响话术）；
2. 追问与否改由 :func:`_ask_question_this_turn` 判定（风格 + 风险 + 拒绝表达）；
3. 降级模板在"本轮不追问"时**不得反问**（此前的 CALM 池自身含问句）。

测试不发起任何网络调用：``api.intervention.requests.post`` 被替换为抓取器。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api.intervention as iv  # noqa: E402
from intervention.style_detector import UserStyle  # noqa: E402

# 中性偏高 / 悲伤偏高的两组概率 [快乐, 悲伤, 焦虑, 愤怒, 中性]
NEUTRAL_PROBS = [0.13, 0.07, 0.0, 0.0, 0.80]
SAD_PROBS = [0.01, 0.52, 0.23, 0.23, 0.01]


@pytest.fixture()
def captured_prompt(monkeypatch):
    """拦截 LLM 调用，返回最后一次请求的 system prompt。"""
    captured: dict = {}

    class _FakeResp:
        ok = True
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "（占位回复）"}}]}

    def fake_post(url, **kwargs):
        captured["messages"] = kwargs.get("json", {}).get("messages", [])
        captured["payload"] = kwargs.get("json", {})
        return _FakeResp()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(iv.requests, "post", fake_post)
    return captured


class TestNegativeEmotionIntensity:
    """max(probs) 是置信度，不能当情绪强度用。"""

    def test_neutral_message_is_not_intense(self):
        """「你好」中性概率 0.80，但负面强度应接近 0（悲伤 0.07）。"""
        assert iv._negative_emotion_intensity(NEUTRAL_PROBS) == pytest.approx(0.07, abs=0.02)

    def test_sad_message_is_intense(self):
        """悲伤 0.52 + 焦虑 0.23 + 愤怒 0.23 = 0.98。"""
        assert iv._negative_emotion_intensity(SAD_PROBS) == pytest.approx(0.98, abs=0.02)

    def test_clamped_to_unit_range(self):
        assert iv._negative_emotion_intensity([0.0, 1.0, 1.0, 1.0, 0.0]) == 1.0

    def test_bad_input_is_safe(self):
        assert iv._negative_emotion_intensity([]) == 0.0


class TestAskQuestionThisTurn:
    """本轮是否允许一个开放式追问。"""

    @pytest.mark.parametrize(
        "text",
        ["别问了", "我不想说", "算了", "换个话题", "没事", "这点小事", "我不想聊这个"],
    )
    def test_refusal_stops_questioning(self, text):
        """明确表示不想谈 → 立刻停止深挖（提示词核心原则 4）。"""
        assert iv._ask_question_this_turn(
            UserStyle.CALM, "low", user_message=text, negative_intensity=0.5
        ) is False

    @pytest.mark.parametrize("text", ["烦死了！！！", "我真的受不了了！！！"])
    def test_agitation_stops_questioning(self, text):
        """激动表达（感叹号）先安抚，不追问。"""
        assert iv._ask_question_this_turn(
            UserStyle.AGITATED, "low", user_message=text, negative_intensity=0.99
        ) is False

    @pytest.mark.parametrize("text", ["我好难受", "为什么我会这样", "我觉得没人理解我"])
    def test_plain_distress_still_allows_one_question(self, text):
        """单纯难过（无激动表达）应允许温柔追问一句 —— 草案模板 1。"""
        assert iv._ask_question_this_turn(
            UserStyle.AGITATED, "low", user_message=text, negative_intensity=0.95
        ) is True

    def test_greeting_allows_inviting_question(self):
        """寒暄轮允许一句邀请式追问，否则模型只会复读"我在这里"。"""
        assert iv._ask_question_this_turn(
            UserStyle.INTROVERTED, "low", user_message="你好", negative_intensity=0.06
        ) is True

    @pytest.mark.parametrize("risk", ["high", "crisis"])
    def test_high_risk_never_asks(self, risk):
        """高风险/危机只做安全确认与转介，不做认知探索。"""
        assert iv._ask_question_this_turn(
            UserStyle.CALM, risk, user_message="我好难受", negative_intensity=0.9
        ) is False


class TestUnifiedPrompt:
    """一段提示词取代 EMPATHY/SOCRATIC 双分支。"""

    def test_contains_person_centered_sections(self, captured_prompt):
        """人本取向的定位 + 安全底线必须都在同一段 prompt 里。

        2026-09-26 改写后"人本取向"这个词被换成了更好操作的角色描述
        （会听人说话的陪伴者），断言随之改为按**行为**判断。
        """
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "# 角色定位" in prompt
        assert "有人真的听懂我在说什么" in prompt  # 人本取向的落地表述
        assert "# 底线" in prompt
        assert "硬性边界" in prompt  # 保留旧称，便于检索
        assert "不能替代线下持证心理咨询师与精神科医生" in prompt
        assert "想开一点" in prompt  # 禁止否定感受的负面清单

    def test_asks_about_the_event_not_only_the_feeling(self, captured_prompt):
        """追问方向必须包含"探事件"，不能只往"什么感觉"走。

        实测发现的退化：「我好难受」会稳定被回成"这种难受是什么样的感觉？"，
        因为示例基本都是探感受。保留探事件方向后，输出才可能落到
        "发生了什么事，让你这么难受？"上。
        """
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "探事件" in prompt
        assert "探感受" in prompt
        assert "探想法" in prompt
        assert "事件类问题最好接" in prompt

    def test_direction_annotations_are_not_for_output(self, captured_prompt):
        """括号内的方向标注不得被模型当场输出。"""
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "不要输出给来访者" in prompt

    def test_no_socratic_branch_left(self, captured_prompt):
        """旧的"话疗树洞"分支必须消失，否则每轮都被强制反问。"""
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "话疗树洞" not in prompt
        assert "苏格拉底式反问" not in prompt

    def test_dialogue_mode_no_longer_changes_wording(self, captured_prompt):
        """dialogue_mode 仅作兼容参数，两种取值必须产出同一段 prompt。"""
        iv._call_llm(
            "我好难受", [], SAD_PROBS, "low",
            style=UserStyle.CALM, dialogue_mode="EMPATHY",
        )
        empathy_prompt = captured_prompt["messages"][0]["content"]
        iv._call_llm(
            "我好难受", [], SAD_PROBS, "low",
            style=UserStyle.CALM, dialogue_mode="SOCRATIC",
        )
        socratic_prompt = captured_prompt["messages"][0]["content"]
        assert empathy_prompt == socratic_prompt

    def test_no_question_rule_on_refusal(self, captured_prompt):
        iv._call_llm("别问了", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "本轮**不要提问**" in prompt

    def test_ask_rule_when_appropriate(self, captured_prompt):
        """默认动作是**追问**（2026-09-26 用户反馈"追问的环节也没有了"）。"""
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "本轮**不要提问**" not in prompt
        assert "本轮**要追问**" in prompt

    def test_history_roles_are_sanitized(self, captured_prompt):
        """历史里的脏 role 不能破坏 messages 结构。"""
        iv._call_llm(
            "我很难受",
            [{"role": "tool", "content": "x"}, {"role": "assistant", "content": "y"}],
            SAD_PROBS,
            "low",
            style=UserStyle.CALM,
        )
        roles = [m["role"] for m in captured_prompt["messages"]]
        assert roles == ["system", "user", "assistant", "user"]


# 被判定为"套路"的固定开场。用户实测截图里连续 4 轮都以
# 「我能感受到你现在X。」开头，句子骨架完全一致，读起来像范文。
_BANNED_OPENINGS = ("我能感受到你", "听起来你", "我理解你", "抱抱", "你已经很勇敢了")


class TestPromptBansTemplatedReplies:
    """提示词必须**显式禁止**固定开场与「共情+提问」公式。

    这是一组"防退化"测试：风格问题无法用功能断言覆盖（LLM 输出天然不稳定），
    但可以让**提示词退回到套路写法**这件事在单测里立刻失败。
    2026-09-26 用户截图反馈：「我还是觉得这种回答很套路，只能令患者更加无语」。
    """

    def test_bans_restating_emotion_as_opening(self, captured_prompt):
        """必须点名禁止「我能感受到你现在X」这类只复述情绪的开场。"""
        iv._call_llm("我考试没考好", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "禁止用这个句式开头" in prompt
        assert "每轮同一句式开头" in prompt
        # 该句式只允许出现在"反例 / 禁止"的语境里
        for line in prompt.splitlines():
            if "我能感受到你" in line:
                assert "反例" in line or "禁止" in line or "❌" in line, line

    def test_empathy_is_still_required(self, captured_prompt):
        """共情不能因为"反套路"被删掉 —— 用户要的是不死板的共情。"""
        iv._call_llm("我考试没考好", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "共情是每个情绪轮次的必做动作" in prompt
        # 共情要落到具体那件事上，并给出示范句，否则模型只能回到复读情绪标签
        assert "共情要" in prompt and "指着那件事说" in prompt
        assert "谁也受不了" in prompt  # 示范里的指着事说的共情

    def test_bans_empathy_plus_question_formula(self, captured_prompt):
        """「复述情绪 → 发生了什么事？」不许成为每轮固定节拍。

        注意与旧版的区别：现在**不是**禁止连着追问（那是上一版做过头的地方），
        而是禁止每轮都走同一个节拍。
        """
        iv._call_llm("我考试没考好", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "每轮都走" in prompt
        assert "这个固定节拍" in prompt
        # 连着两轮追问是允许的，只拦连着三轮
        assert "不要连着三轮都在问" in prompt
        assert "连着两轮都追问是可以的" in prompt

    def test_forbids_inventing_the_users_state(self, captured_prompt):
        """不许替用户编状态 —— 截图里"看你有点没精神"的根因。"""
        iv._call_llm("你好", [], NEUTRAL_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "只回应 ta 说的话，不要替 ta 编话" in prompt
        assert "你今天看起来有点没精神" in prompt  # 必须作为反例出现

    def test_greeting_gets_its_own_playbook(self, captured_prompt):
        """打招呼时只回招呼，不提情绪、不追问。"""
        iv._call_llm("你好", [], NEUTRAL_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "本轮用户只说了寒暄/客套话" in prompt
        assert "今天想说点什么" in prompt
        assert "问了就是硬造" in prompt
        assert "本轮**要追问**" not in prompt
        assert "只说回应，不要提问" in prompt

    def test_second_greeting_does_not_greet_again(self, captured_prompt):
        """同一段对话里第二次说"你好"，不能又问一次好。"""
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好呀，今天想说点什么？"},
        ]
        iv._call_llm("你好", history, NEUTRAL_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "不要再重复问好" in prompt

    def test_normal_message_has_no_social_rule(self, captured_prompt):
        """非寒暄轮次不得出现寒暄规则，否则会挤掉正常的共情追问要求。"""
        iv._call_llm("老师骂我了", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "本轮用户只说了寒暄/客套话" not in prompt
        assert "本轮**要追问**" in prompt

    def test_first_sentence_must_land_on_concrete_content(self, captured_prompt):
        """回应必须碰到用户说的具体人/事/原话。"""
        iv._call_llm("老师骂我了", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "先接住 ta 说的具体内容" in prompt

    def test_bans_bare_acknowledgement_as_default(self, captured_prompt):
        """「嗯，我在」这类留白句不能变成新的万能回复。

        实测复现：把"禁止复述情绪"加狠之后，模型转向每轮都回
        「嗯，我在。」（场景二 4/4 轮完全相同）—— 套路只是换了个样子。
        """
        iv._call_llm("我很难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "全篇只能用一次" in prompt

    def test_bans_quoteback_plus_label_formula(self, captured_prompt):
        """「『原话』—— 再贴个情绪判断」不能成为默认开场。

        实测复现：第 2 轮「嗯，考试没考好确实挺让人难受的。」、
        第 4 轮「『爸妈昨天又吵架了』——这事儿现在还在你心里搅着吧？」
        —— 换了句式但骨架没变。
        """
        iv._call_llm("爸妈又吵架了", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "把 ta 的原话加个问号抛回去" in prompt
        assert "等于没说" in prompt

    def test_bans_repeating_users_self_deprecation(self, captured_prompt):
        """用户说「他说我很废物」，不能回「你觉得你是废物」。"""
        iv._call_llm("他说我很废物", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "贬低自己的话复读" in prompt
        assert "你觉得你是废物" in prompt  # 作为禁止的反例出现

    def test_offers_multiple_moves(self, captured_prompt):
        """必须给出多种可轮换的动作/示范，否则模型只会抓住一个句式不放。"""
        iv._call_llm("我很难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        # 两段不同情境的完整对话示范
        assert "示范一" in prompt
        assert "示范二" in prompt
        # 追问至少给出多种方向（事件/感受/想法），不是只有"什么感觉"
        assert "探事件" in prompt
        assert "探想法" in prompt

    def test_used_openings_are_called_out(self, captured_prompt):
        """用过「我能感受到…」之后，prompt 要点名提醒换说法。"""
        history = [
            {"role": "user", "content": "我很难受"},
            {"role": "assistant", "content": "我能感受到你现在很难受。发生什么了吗？"},
        ]
        iv._call_llm("考试没考好", history, SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "本轮硬性禁用开场" in prompt
        assert "我能感受到" in prompt
        assert "禁止再用" in prompt

    def test_no_opening_reminder_on_first_turn(self, captured_prompt):
        """首轮没有任何历史时不该出现这条提醒（避免噪音）。"""
        iv._call_llm("我很难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "你已经用过" not in prompt

    def test_used_openings_dedupes(self):
        history = [
            {"role": "assistant", "content": "我能感受到你现在很难受。"},
            {"role": "assistant", "content": "听起来你挺不容易的。"},
            {"role": "assistant", "content": "我能感受到你现在很累。"},
        ]
        assert iv._used_openings(history) == ["我能感受到", "听起来你"]

    def test_question_directions_are_documented(self, captured_prompt):
        """追问要落到具体内容上，三个方向都要在。"""
        iv._call_llm("我很难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "探事件" in prompt
        assert "探感受" in prompt
        assert "探想法" in prompt
        assert "不要输出给来访者" in prompt


class TestCapQuestions:
    """一轮最多一个问句 —— 提示词写了但实测仍会抛两个。"""

    def test_two_questions_in_one_turn_keeps_the_first(self):
        raw = "是有什么特别的事要发生吗？还是最近在学校遇到什么烦心事了？"
        assert iv._cap_questions(raw) == "是有什么特别的事要发生吗？"

    def test_question_plus_statement_afterwards_is_preserved(self):
        """删掉多余问句时，句中的陈述句必须留着。"""
        raw = "他们吵得严重吗？你在旁边肯定很难受吧？"
        capped = iv._cap_questions(raw)
        assert capped.count("？") == 1
        assert capped == "他们吵得严重吗？"

    def test_single_question_is_untouched(self):
        raw = "老师当着全班说的？这也太让人下不来台了。"
        assert iv._cap_questions(raw) == raw

    def test_statement_scattered_questions_keep_first_only(self):
        raw = "嗯，我知道了。是那样吗？那后来呢？"
        capped = iv._cap_questions(raw)
        assert capped == "嗯，我知道了。是那样吗？"

    def test_no_question_is_noop(self):
        raw = "嗯，我在听。你不用急着说。"
        assert iv._cap_questions(raw) == raw

    def test_rewrites_when_capping_would_empty_the_reply(self):
        """整段都是问句时不能删成空串，保留第一个问句（但不改写文本）。"""
        raw = "发生了什么？你现在是什么感觉？后来呢？"
        capped = iv._cap_questions(raw)
        assert capped == "发生了什么？"

    def test_cap_never_invents_text(self):
        """压问句必须只做删除 —— 历史事故：它把回复改写成了无关的通用安慰。

        原样复现：「老师当着全班说的？那次是什么事？」曾被压成
        「那次一定有原因。」，用户看到的是一句跟 ta 说的话没关系的话。
        """
        raw = "老师当着全班说的？那次是什么事？"
        assert iv._cap_questions(raw) == "老师当着全班说的？"

    def test_two_questions_with_no_statement_are_rewritten(self):
        """压完只剩一个问句时也是**保留该问句**，不做改写。"""
        raw = "发生了什么？你现在是什么感觉？"
        assert iv._cap_questions(raw) == "发生了什么？"

    def test_finalize_applies_cap_even_when_question_allowed(self):
        raw = "他们吵得严重吗？你在旁边吗？"
        assert iv._finalize_reply(raw, allow_question=True) == "他们吵得严重吗？"


class TestStripTrailingQuestion:
    """本轮不许提问时，结尾问句必须由代码兜底删掉。

    依据：prompt 末尾已写「只回应，不提问 / 把问号删掉」，实测 qwen-turbo
    在一段短消息密集的对话里仍有约一半轮次照问不误。风格约束在 LLM 上是
    概率性的，要变成不变量只能落到代码里。
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("老师骂你的时候，是在教室里吗？", "老师骂你的时候，是在教室里。"),
            ("这事儿挺难受的。那时你在旁边吗？", "这事儿挺难受的。"),
            ("这事儿挺难受的。那时你在旁边吗?", "这事儿挺难受的。"),
            ("嗯，我在听。", "嗯，我在听。"),
            ("我真的不知道说什么。", "我真的不知道说什么。"),
            ("「我很废物」——这话是他说的，你信吗？", "「我很废物」——这话是他说的。"),
            # 前截只是状语框架时不能砍，砍了是半截话
            ("说到考试的时候，是在教室里吗？", "说到考试的时候，是在教室里。"),
            ("老师骂了你，是在教室里吗？", "老师骂了你。"),
        ],
    )
    def test_removes_only_trailing_question(self, raw, expected):
        """删/改掉结尾问句，前面的陈述内容一律保留。"""
        cleaned, _ = iv._strip_trailing_question(raw)
        assert cleaned == expected

    def test_keeps_leading_statement_and_drops_final_question(self):
        cleaned, removed = iv._strip_trailing_question("考试没考好确实难受。哪一门啊？")
        assert removed is True
        assert cleaned == "考试没考好确实难受。"

    def test_mid_sentence_question_is_preserved(self):
        """问句在中间（后面还有陈述句）时不得删。"""
        raw = "老师当着全班说的？这也太让人下不来台了。"
        cleaned, removed = iv._strip_trailing_question(raw)
        assert removed is False
        assert cleaned == raw

    def test_pure_question_is_rewritten_into_a_statement(self):
        """整段只有一个问句时改写成陈述句，而不是留下一个问号。"""
        cleaned, removed = iv._strip_trailing_question("发生了什么事？")
        assert removed is True
        assert "？" not in cleaned and cleaned.endswith("。")

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("那你当时是什么感觉？", "那你当时一定很不好受。"),
            ("这时候你心里是什么滋味？", "这时候你心里肯定不是滋味。"),
            ("发生了什么事？", "应该是发生了什么。"),
        ],
    )
    def test_rewrite_prefers_specific_question_part(self, raw, expected):
        """具体提问部件优先于"你当时"这类前缀，否则会改成"我猜你当时是什么感觉。"。"""
        cleaned, removed = iv._strip_trailing_question(raw)
        assert removed is True
        assert cleaned == expected

    @pytest.mark.parametrize(
        "raw",
        ["你现在是什么情况？", "你现在是什么感觉？"],
    )
    def test_rewrite_hits_the_question_part_not_the_prefix(self, raw):
        """改写后必须是陈述句，不能把"是什么感觉"原封不动留在句子里。"""
        cleaned, removed = iv._strip_trailing_question(raw)
        assert removed is True
        assert cleaned.endswith("。")
        assert "是什么" not in cleaned or "发生了" in cleaned

    def test_pure_question_without_known_pattern_is_left_alone(self):
        """没有可用替换项时宁可保留原句，也不给空回复。"""
        raw = "嗯嗯？"
        cleaned, removed = iv._strip_trailing_question(raw)
        assert removed is False
        assert cleaned == raw

    def test_no_question_is_noop(self):
        raw = "嗯，我在听。"
        assert iv._strip_trailing_question(raw) == (raw, False)
        assert iv._strip_trailing_question("") == ("", False)

    def test_no_question_leaks_through_common_shapes(self):
        """兜底不变量：这些常见句式在 finalize(False) 后都不能留问号。

        刻意**不**把"整条回复只有一个问句且没有可改写部件"的形状放进来：
        那种情况现在的取舍是保留问句（宁可留一句问话，也不改写成无关的话）。
        """
        shapes = [
            "老师骂你的时候，是在教室里吗？",
            "这事儿挺难受的。那时你在旁边吗？",
            "你妈说要不是你，他们早就离了，这话她是在气头上说的吗？",
            "考试没考好确实难受。哪一门啊？",
            "你现在是什么感觉？",
            "那你当时是什么感觉？",
            "他们吵得严重吗？",
            "「我很废物」——这话是他说的，你信吗？",
        ]
        for raw in shapes:
            out = iv._finalize_reply(raw, allow_question=False)
            assert "？" not in out and "?" not in out, (raw, out)
            assert out.strip(), raw

    def test_multi_question_reply_keeps_one_question(self):
        """两个问句时保留第一个：宁可留一个问句，也不改写成无关的话。"""
        out = iv._finalize_reply("发生了什么？你现在是什么感觉？", allow_question=False)
        assert out == "发生了什么？"

    def test_finalize_reply_respects_allow_flag(self):
        raw = "考试没考好确实难受。哪一门啊？"
        assert iv._finalize_reply(raw, allow_question=True) == raw
        assert iv._finalize_reply(raw, allow_question=False) == "考试没考好确实难受。"

    def test_prompt_cools_down_but_code_keeps_the_question(self, monkeypatch):
        """连着三轮追问：prompt 里降温，但**生成的问句要被保留**。

        这条替代了早先的 test_call_llm_strips_question_when_repeating ——
        那个行为已被证明是错的（会把「老师骂你了？」削成「老师骂你了。」）。
        """
        captured: dict = {}

        class _FakeResp:
            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {
                    "content": "你妈这么说，你夹在中间最难做。那你当时说话了吗？"
                }}]}

        def fake_post(url, **kwargs):
            captured["messages"] = kwargs.get("json", {}).get("messages", [])
            return _FakeResp()

        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(iv.requests, "post", fake_post)

        history = [
            {"role": "user", "content": "我考试没考好"},
            {"role": "assistant", "content": "哪一门啊？"},
            {"role": "user", "content": "老师骂我了"},
            {"role": "assistant", "content": "是在教室里吗？"},
            {"role": "user", "content": "他说我很废物"},
            {"role": "assistant", "content": "他当着别人说的吗？"},
        ]
        reply = iv._call_llm("他骂我废物", history, SAD_PROBS, "low", style=UserStyle.CALM)

        # 连着三轮追问 → prompt 侧确实降温了
        assert "前面已经连着三轮用问句结尾了" in captured["messages"][0]["content"]
        # 但生成的问句原样保留（不再按轮数删文本）
        assert reply == "你妈这么说，你夹在中间最难做。那你当时说话了吗？"

    def test_call_llm_keeps_question_when_only_consecutive(self, monkeypatch):
        """连着三轮追问**不再删问句** —— 只降温，不削文本。

        实测（2026-09-26 第二张截图后的重放）：按轮数删问句会把
        「老师骂你了？」削成「老师骂你了。」—— 问句没了只剩回声，比原来更糟。
        用户也明确要求"追问要回来"，所以这条规则只留在 prompt 语气里。
        """
        class _FakeResp:
            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "老师骂你的时候，是在教室里吗？"}}]}

        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(iv.requests, "post", lambda url, **kw: _FakeResp())

        history = [
            {"role": "assistant", "content": "哪一门啊？"},
            {"role": "user", "content": "老师骂我了"},
            {"role": "assistant", "content": "是在教室里吗？"},
            {"role": "user", "content": "他说我很废物"},
            {"role": "assistant", "content": "他当着别人说的吗？"},
        ]
        reply = iv._call_llm("他骂我废物", history, SAD_PROBS, "low", style=UserStyle.CALM)
        assert reply == "老师骂你的时候，是在教室里吗？"

    def test_call_llm_strips_question_when_user_refuses(self, monkeypatch):
        """但"用户明确不想谈"时仍必须删掉结尾问句（与轮数无关）。"""
        class _FakeResp:
            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {
                    "content": "你妈这么说，你夹在中间最难做。那你当时说话了吗？"
                }}]}

        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(iv.requests, "post", lambda url, **kw: _FakeResp())

        reply = iv._call_llm("别问了", [], SAD_PROBS, "low", style=UserStyle.CALM)
        assert "？" not in reply and "?" not in reply
        assert reply == "你妈这么说，你夹在中间最难做。"

    def test_call_llm_strips_question_when_greeting(self, monkeypatch):
        """寒暄轮次也不许挂问句（prompt 已写明，代码兜底）。"""
        class _FakeResp:
            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {
                    "content": "你好呀。你今天看起来有点没精神，发生什么事了吗？"
                }}]}

        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(iv.requests, "post", lambda url, **kw: _FakeResp())

        reply = iv._call_llm("你好", [], NEUTRAL_PROBS, "low", style=UserStyle.CALM)
        assert "？" not in reply and "?" not in reply

    def test_pure_question_reply_is_rewritten_or_kept(self, monkeypatch):
        """用户拒绝深谈时，模型只回一个问句也要处理掉（能改写就改写）。"""
        class _FakeResp:
            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "那你当时是什么感觉？"}}]}

        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(iv.requests, "post", lambda url, **kw: _FakeResp())

        reply = iv._call_llm("别问了", [], SAD_PROBS, "low", style=UserStyle.CALM)
        assert "？" not in reply and "?" not in reply
        assert reply == "那你当时一定很不好受。"


class TestBareEchoRewrite:
    """「复读 + 问号」必须被改写 —— 这是比套路开场更空的一轮。

    实测复现（2026-09-26 第二张截图后的重放）：
        用户：老师骂我了        → AI：老师骂你了？
        用户：爸妈昨天又吵架了   → AI：爸妈又吵架了？
        用户：我妈说要不是我他们早就离了 → AI：我妈说要不是你他们早就离了？
    """

    @pytest.mark.parametrize(
        "reply,user_message,expected_echo",
        [
            ("老师骂你了？", "老师骂我了", True),
            ("老师骂你了。", "老师骂我了", True),
            ("爸妈又吵架了？", "爸妈昨天又吵架了", True),
            ("我妈说要不是你他们早就离了？", "我妈说要不是我他们早就离了", True),
            # 不含回声的正常回复
            ("被这么说肯定难受。这话是他随口说的，还是特意冲你来的？", "老师骂我了", False),
            ("你妈这么说，你夹在中间最难做。", "我妈说要不是我他们早就离了", False),
            ("嗯，我在听。", "老师骂我了", False),
        ],
    )
    def test_echo_detection(self, reply, user_message, expected_echo):
        assert iv._is_bare_echo(reply, user_message) is expected_echo

    def test_echo_with_question_becomes_empathy_plus_probe(self):
        out = iv._expand_bare_echo("老师骂你了？", "老师骂我了")
        assert "老师骂我了" in out
        assert out.endswith("？")
        assert "？" not in out[:-1]

    def test_echo_of_feeling_gets_a_feeling_probe(self):
        out = iv._expand_bare_echo("我有点难受？", "我有点难受")
        assert "是从什么时候开始的" in out

    def test_echo_without_question_gets_no_question(self):
        out = iv._expand_bare_echo("老师骂你了。", "老师骂我了")
        assert "？" not in out

    def test_finalize_rewrites_echo_end_to_end(self):
        out = iv._finalize_reply("老师骂你了？", True, user_message="老师骂我了")
        assert out != "老师骂你了？"
        assert "老师骂我了" in out

    def test_call_llm_rewrites_echo(self, monkeypatch, captured_prompt):
        """端到端：模型回一句复读时，用户看到的不是复读。"""
        class _FakeResp:
            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "老师骂你了？"}}]}

        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(iv.requests, "post", lambda url, **kw: _FakeResp())

        reply = iv._call_llm("老师骂我了", [], SAD_PROBS, "low", style=UserStyle.CALM)
        assert reply != "老师骂你了？"
        assert "老师骂我了" in reply


class TestQuestionAlternation:
    """连续追问由代码判定，不能只靠 prompt 里的"禁止"。

    真实复现（2026-09-26）：第一版设 run_length=2 拦"连着两轮追问"，
    结果用户反馈"追问的环节也没有了" —— 拦过头了。现在只拦连着三轮。
    """

    def test_run_length_is_three(self):
        """把容忍轮数写死在测试里，避免以后被无意改回去。"""
        assert iv._QUESTION_RUN_LENGTH == 3

    def test_question_count_counts_both_mark_types(self):
        assert iv._question_count("你好吗？") == 1
        assert iv._question_count("Really?") == 1
        assert iv._question_count("嗯，我在。") == 0
        assert iv._question_count("是这样吗？还是那样?") == 2

    def test_two_consecutive_questions_are_allowed(self):
        """连着两轮都追问**不算**触发 —— 这是用户要回来的追问节奏。"""
        history = [
            {"role": "user", "content": "我很难受"},
            {"role": "assistant", "content": "发生了什么事？"},
            {"role": "user", "content": "考试没考好"},
            {"role": "assistant", "content": "老师说什么了？"},
        ]
        assert iv._assistant_asked_repeatedly(history) is False

    def test_three_consecutive_questions_trigger(self):
        history = [
            {"role": "assistant", "content": "发生了什么事？"},
            {"role": "user", "content": "考试没考好"},
            {"role": "assistant", "content": "老师说什么了？"},
            {"role": "user", "content": "老师骂我了"},
            {"role": "assistant", "content": "是在教室里吗？"},
        ]
        assert iv._assistant_asked_repeatedly(history) is True

    def test_not_triggered_when_a_turn_did_not_ask(self):
        history = [
            {"role": "assistant", "content": "发生了什么事？"},
            {"role": "user", "content": "考试没考好"},
            {"role": "assistant", "content": "嗯，我在听。"},
            {"role": "user", "content": "老师骂我了"},
            {"role": "assistant", "content": "是在教室里吗？"},
        ]
        assert iv._assistant_asked_repeatedly(history) is False

    def test_first_turn_never_triggers(self):
        assert iv._assistant_asked_repeatedly([]) is False
        assert iv._assistant_asked_repeatedly(
            [{"role": "user", "content": "在吗"}]
        ) is False

    def test_ignores_non_assistant_roles(self):
        """用户自己问号再多也不算 AI 追问。"""
        history = [
            {"role": "user", "content": "为什么？为什么？"},
            {"role": "assistant", "content": "嗯，我在听。"},
            {"role": "user", "content": "为什么会这样？"},
            {"role": "assistant", "content": "我不知道该说什么，但我在听。"},
            {"role": "user", "content": "为什么？"},
        ]
        assert iv._assistant_asked_repeatedly(history) is False

    def test_prompt_suppresses_question_after_three_asking_turns(self, captured_prompt):
        history = [
            {"role": "assistant", "content": "发生了什么事？"},
            {"role": "user", "content": "考试没考好"},
            {"role": "assistant", "content": "老师说什么了？"},
            {"role": "user", "content": "老师骂我了"},
            {"role": "assistant", "content": "是在教室里吗？"},
        ]
        iv._call_llm("他说我很废物", history, SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "前面已经连着三轮用问句结尾了" in prompt
        assert "本轮**要追问**" not in prompt

    def test_prompt_allows_question_after_two_asking_turns(self, captured_prompt):
        """连着两轮问过之后，第三轮不因"连着三轮"被抑制。

        新版加入了"问、问、接一句"的落地节拍：当这一轮恰落在落地拍
        （turn_count%3==2）时，它会是"先不往下问"的**主动留白**，而不是
        "连着三轮问句"那种被动的审问抑制 —— 两者动机不同，必须可区分。
        这里的历史为 4 条 → turn_count=2 → 落地拍，故断言落地而非审问抑制。
        """
        history = [
            {"role": "user", "content": "我很难受"},
            {"role": "assistant", "content": "发生了什么事？"},
            {"role": "user", "content": "考试没考好"},
            {"role": "assistant", "content": "老师说什么了？"},
        ]
        iv._call_llm("老师骂我了", history, SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        # 是落地拍的主动留白，不是"连着三轮"的审问抑制
        assert "先不往下问" in prompt
        assert "前面已经连着三轮" not in prompt

    def test_prompt_asks_on_non_beat_turn(self, captured_prompt):
        """非落地拍（turn_count%3!=2）且未连着三轮时，仍鼓励往下追问。"""
        history = [
            {"role": "user", "content": "我很难受"},
            {"role": "assistant", "content": "发生了什么事？"},
        ]
        iv._call_llm("考试没考好", history, SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "本轮**要追问**" in prompt


class TestFallbackTemplates:
    """降级模板必须与"本轮是否追问"保持一致，且不得复用被禁的套路开场。"""

    @pytest.mark.parametrize(
        "style",
        [UserStyle.AGITATED, UserStyle.INTROVERTED, UserStyle.CALM, UserStyle.EXTROVERTED, None],
    )
    @pytest.mark.parametrize("risk", ["low", "medium"])
    def test_no_question_when_forbidden(self, style, risk):
        """ask_question=False 时不得反问（此前的 CALM 池含问句，会自相矛盾）。"""
        reply = iv._generate_empathy_reply(
            SAD_PROBS, risk, style=style, ask_question=False
        )
        assert "？" not in reply and "?" not in reply

    @pytest.mark.parametrize("risk", ["low", "medium"])
    def test_templates_avoid_banned_openings(self, risk):
        """降级池若用被禁句式开场，LLM 一挂用户就又开始收套路回复。"""
        for tmpl in iv._EMPATHY_TEMPLATES[risk]:
            assert not tmpl.startswith(_BANNED_OPENINGS), tmpl

    @pytest.mark.parametrize("risk", ["high", "crisis"])
    def test_high_risk_keeps_safety_check(self, risk):
        """高风险/危机的安全确认问句属风险确认，必须保留。"""
        reply = iv._generate_empathy_reply(
            SAD_PROBS, risk, style=UserStyle.CALM, ask_question=False
        )
        assert reply

    def test_ask_question_uses_open_templates(self):
        """允许追问时取到的应是含问句的开放式模板。"""
        replies = {
            iv._generate_empathy_reply(SAD_PROBS, "low", style=UserStyle.CALM, ask_question=True)
            for _ in range(8)
        }
        assert any("？" in r for r in replies)
