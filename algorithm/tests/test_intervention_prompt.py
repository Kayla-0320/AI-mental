"""提示词契约与追问规则测试

覆盖三处：

1. **精简版提示词**（2026-10-05，用户反馈"每次回复都是一样的话术"后重写）：
   只保留 AI 身份、安全底线与「每轮最后一句必须是一个关心 ta 的问题」；
   不再注入示范对话、逐句禁用清单、情绪线索与节拍规则 —— 这些示范句本身就是
   模型最容易照抄的句子（实测连续四轮都以同一句示范开头）。
2. 追问与否仍由 :func:`_ask_question_this_turn` 判定（拒绝表达 / 激动 / 高风险）；
   但寒暄轮不再单独拦尾问句，高风险/危机反而放行（安全确认就是一句问话）。
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
    """精简版提示词：只留 AI 身份、话术总则与安全底线。"""

    def test_contains_identity_and_safety(self, captured_prompt):
        """身份 + 安全底线必须都在同一段 prompt 里。"""
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "AI 陪伴者" in prompt
        assert "# 底线" in prompt
        assert "不能替代线下持证的心理咨询师与精神科医生" in prompt
        assert "心理危机热线" in prompt  # 安全转介
        assert "想开一点" in prompt  # 禁止否定感受的负面清单

    def test_last_sentence_must_be_a_caring_question(self, captured_prompt):
        """本轮的核心契约：每一轮最后一句都是一个关心 ta 的问题。"""
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "每一轮回复的最后一句，必须是一个关心 ta 的问题" in prompt
        assert "只有最后一句是问句" in prompt

    def test_prompt_carries_no_copyable_example_dialogue(self, captured_prompt):
        """提示词里不许再出现可被整句照抄的示范台词与示范场景。

        回归：旧版写了「你一说这事我就听出来了，搁谁身上都难受。」等示范句，
        实测连续四轮回复都以它开头 —— 用户反馈"每次回复都是一样的话术"。
        """
        iv._call_llm("我现在有点难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        for copied in (
            "你一说这事我就听出来了",
            "搁谁身上都难受",
            "这话太重了",
            "我听着是",
            "示范一",
            "示范二",
            "被当着全班",
            "ta：",
        ):
            assert copied not in prompt, copied

    def test_prompt_is_compact(self, captured_prompt):
        """防止约束回涨：精简后的 prompt 必须明显短于旧版的 3200 字。"""
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert len(prompt) < 1600, len(prompt)

    def test_reply_length_is_capped_and_graded(self, captured_prompt):
        """长度要求是**分档的短回复**（2026-10-05 用户反馈"又有点长了"）。

        闲聊/寒暄 40 字以内，说到情绪或事情 100 字以内；定稿侧的硬截断必须
        与 prompt 同源，否则模型写长了也没人拦。
        """
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "短一点，别铺陈" in prompt
        assert "40 字以内" in prompt
        assert "绝不超过 100 字" in prompt
        assert "第一句必须是陈述" in prompt
        assert iv._MAX_REPLY_CHARS <= 120
        assert iv._MAX_SOCIAL_REPLY_CHARS <= 60

    def test_direction_annotations_are_not_for_output(self, captured_prompt):
        """括号内的说明不得被模型当场输出。"""
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
        assert "这一轮**不要提问**" in prompt

    def test_normal_turn_keeps_the_question_rule(self, captured_prompt):
        """默认动作是追问：普通情绪轮次不该出现"不要提问"。"""
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "这一轮**不要提问**" not in prompt
        assert "每一轮回复的最后一句，必须是一个关心 ta 的问题" in prompt

    def test_high_risk_keeps_the_safety_question(self, captured_prompt):
        """高风险/危机不放"不要提问"，反而明示结尾仍要落在关心 ta 的问句上。

        安全确认本身就是一句要问的话：按"高风险不追问"去删尾问句，
        会把「你现在安全吗？」一起删掉。
        """
        iv._call_llm("我不想活了", [], SAD_PROBS, "crisis", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "这一轮**不要提问**" not in prompt
        assert "先按上面「底线」做安全确认" in prompt

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


class TestLoosePromptContract:
    """"限制松一点"之后必须守住的几条。

    1. 不许再出现可照抄的示范句 —— 旧的「我听着是」「这话太重了」
       「搁谁身上都难受」正是模型每轮照搬的来源；
    2. 不许替 ta 编状态（截图里"看你有点没精神"的根因）；
    3. 反套路的要求只保留抽象一条（同一句话最多用一次），不做逐句点名。
    """

    def test_no_few_shot_dialogue_left(self, captured_prompt):
        """示范对话必须整段删除 —— 模型对 few-shot 句子的模仿是字面级的。"""
        iv._call_llm("我考试没考好", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "示范" not in prompt
        assert "✅" not in prompt and "❌" not in prompt
        assert "本轮硬性禁用" not in prompt

    def test_ban_is_abstract_not_a_phrase_list(self, captured_prompt):
        """别每轮说同一句只写成一条总则，不再列出被禁的具体句子。

        旧版把「我听着是」「这话太重了」等句子列成禁用清单，等于在 prompt 里
        又把这些句子念了一遍 —— 越禁越照抄。
        """
        history = [
            {"role": "user", "content": "我这次考试没考好"},
            {"role": "assistant", "content": "我听着是，这事儿现在还压着你呢？"},
        ]
        iv._call_llm("我还是觉得难受", history, SAD_PROBS, "medium", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "同一句话、同一个开场、同一句安慰，整段对话里最多用一次" in prompt
        assert "我听着是" not in prompt

    def test_forbids_inventing_the_users_state(self, captured_prompt):
        """不许替用户编状态 —— 截图里"看你有点没精神"的根因。"""
        iv._call_llm("你好", [], NEUTRAL_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "不许说 ta 没讲过的事" in prompt

    def test_greeting_still_gets_its_own_line(self, captured_prompt):
        """打招呼时只回招呼，不评价 ta 的状态，结尾仍是邀请 ta 开口的问题。"""
        iv._call_llm("你好", [], NEUTRAL_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "只说了寒暄" in prompt
        assert "不要评价 ta 的状态" in prompt
        assert "邀请 ta 开口的问题" in prompt

    def test_second_greeting_does_not_greet_again(self, captured_prompt):
        """同一段对话里第二次说"你好"，不能又问一次好。"""
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好呀，今天想说点什么？"},
        ]
        iv._call_llm("你好", history, NEUTRAL_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "不要再问一次好" in prompt

    def test_normal_message_has_no_social_line(self, captured_prompt):
        """非寒暄轮次不得出现寒暄规则，否则会挤掉正常的话术总则。"""
        iv._call_llm("老师骂我了", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "只说了寒暄" not in prompt

    def test_used_openings_dedupes(self):
        history = [
            {"role": "assistant", "content": "我能感受到你现在很难受。"},
            {"role": "assistant", "content": "听起来你挺不容易的。"},
            {"role": "assistant", "content": "我能感受到你现在很累。"},
        ]
        assert iv._used_openings(history) == ["我能感受到", "听起来你"]


class TestCapQuestions:
    """一轮最多一个问句 —— 提示词写了但实测仍会抛两个。

    ⚠️ 保留的是**最后**一个问句（2026-10-05 改）：新话术的契约是"最后一句
    是关心 ta 的问题"，保留第一个会把结尾整段砍掉，实测把模型真正想问的那句
    连同后面的内容一起删掉，只剩 9 个字的反问。
    """

    def test_two_questions_in_one_turn_keeps_the_last(self):
        raw = "是有什么特别的事要发生吗？还是最近在学校遇到什么烦心事了？"
        assert iv._cap_questions(raw) == "还是最近在学校遇到什么烦心事了？"

    def test_question_plus_statement_afterwards_is_preserved(self):
        """删掉多余问句时，句中的陈述句必须留着。"""
        raw = "他们吵得严重吗？那后来呢？你在旁边肯定挺难受的。"
        capped = iv._cap_questions(raw)
        assert capped.count("？") == 1
        assert capped == "那后来呢？你在旁边肯定挺难受的。"

    def test_single_question_is_untouched(self):
        raw = "老师当着全班说的？这也太让人下不来台了。"
        assert iv._cap_questions(raw) == raw

    def test_statement_scattered_questions_keep_last_only(self):
        raw = "嗯，我知道了。是那样吗？那后来呢？"
        capped = iv._cap_questions(raw)
        assert capped == "嗯，我知道了。那后来呢？"

    def test_no_question_is_noop(self):
        raw = "嗯，我在听。你不用急着说。"
        assert iv._cap_questions(raw) == raw

    def test_rewrites_when_capping_would_empty_the_reply(self):
        """整段都是问句时不能删成空串，保留最后一个问句（但不改写文本）。"""
        raw = "发生了什么？你现在是什么感觉？后来呢？"
        capped = iv._cap_questions(raw)
        assert capped == "后来呢？"

    def test_cap_never_invents_text(self):
        """压问句必须只做删除 —— 历史事故：它把回复改写成了无关的通用安慰。

        原样复现：「老师当着全班说的？那次是什么事？」曾被压成
        「那次一定有原因。」，用户看到的是一句跟 ta 说的话没关系的话。
        现在同样只做删除：留下的是原文里的最后一句。
        """
        raw = "老师当着全班说的？那次是什么事？"
        assert iv._cap_questions(raw) == "那次是什么事？"

    def test_two_questions_with_no_statement_are_capped(self):
        """压完只剩一个问句时也是**保留该问句**，不做改写。"""
        raw = "发生了什么？你现在是什么感觉？"
        assert iv._cap_questions(raw) == "你现在是什么感觉？"

    def test_finalize_applies_cap_even_when_question_allowed(self):
        raw = "他们吵得严重吗？你在旁边吗？"
        assert iv._finalize_reply(raw, allow_question=True) == "你在旁边吗？"


class TestQuestionLast:
    """契约：日常轮次的最后一句是"关心 ta 的问题"。"""

    def test_moves_the_only_question_to_the_end(self):
        """问句在中间时整句挪到末尾（切句以句末标点为准，"嗯，"跟着它那一句走）。"""
        raw = "嗯，最近有什么事让你觉得难受吗？我在这儿听着呢。"
        assert iv._ensure_question_last(raw) == "我在这儿听着呢。嗯，最近有什么事让你觉得难受吗？"

    def test_noop_when_question_is_already_last(self):
        raw = "嗯，我在听。最近发生什么了？"
        assert iv._ensure_question_last(raw) == raw

    def test_noop_when_several_questions(self):
        """问句不唯一时不猜 —— 先由 :func:`_cap_questions` 压到一个。"""
        raw = "是那样吗？那后来呢？"
        assert iv._ensure_question_last(raw) == raw

    def test_noop_without_question(self):
        raw = "嗯，我在听。你慢慢说。"
        assert iv._ensure_question_last(raw) == raw

    def test_split_merges_lone_closing_quote(self):
        """切句会切出孤立的收尾引号，必须先并回前一句。"""
        parts = iv._split_sentences_merged("他问我：“你还好吗？”")
        assert parts == ["他问我：“你还好吗？”"]

    def test_question_with_closing_quote_is_detected(self):
        assert iv._is_question_sentence("…哪里会不一样？”") is True
        assert iv._is_question_sentence("…哪里会不一样。") is False

    def test_ensure_question_last_does_not_move_quotes(self):
        """回归：引号曾被当成独立句子挪到句中间（「那……”考完之后…」）。"""
        raw = "那……考完之后，你有没有偷偷翻过课本，心里悄悄问自己：“如果再来一次，哪里会不一样？”"
        assert iv._ensure_question_last(raw) == raw

    def test_moves_quoted_question_to_the_end(self):
        raw = "“你还好吗？”我在旁边听着，心里也不好受。"
        assert iv._ensure_question_last(raw) == "我在旁边听着，心里也不好受。“你还好吗？”"

    def test_finalize_puts_the_caring_question_last(self):
        raw = "考试没考好确实挺让人难受的。你当时是什么心情呢？我在这儿陪着你。"
        out = iv._finalize_reply(
            raw, True, user_message="我考试没考好", risk_level="low",
        )
        assert out.endswith("？")
        assert "我在这儿陪着你" in out

    def test_trim_keeps_the_trailing_question(self):
        """压长度不能把结尾那句问话丢掉（它是这一轮的落点）。"""
        raw = "这是一句很长的承接话用来把长度顶上去。" * 10 + "你当时是什么心情呢？"
        out = iv._trim_reply(raw, [])
        assert out.endswith("？")
        assert len(out) <= iv._MAX_REPLY_CHARS + 20


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
        """两个问句时先压成一个（保留**最后**那个），本轮不许提问时再删掉。"""
        out = iv._finalize_reply("发生了什么？你现在是什么感觉？", allow_question=False)
        assert "？" not in out and "?" not in out
        assert out == "你现在一定很不好受。"

    def test_finalize_reply_respects_allow_flag(self):
        raw = "考试没考好确实难受。哪一门啊？"
        assert iv._finalize_reply(raw, allow_question=True) == raw
        assert iv._finalize_reply(raw, allow_question=False) == "考试没考好确实难受。"

    def test_consecutive_questions_are_never_stripped(self, monkeypatch):
        """连着三轮追问也**不删问句** —— 新话术本就要求每轮以问句结尾。

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

        # prompt 侧不再按轮数降温（旧版会注入"前面已经连着三轮用问句结尾了"）
        assert "前面已经连着三轮用问句结尾了" not in captured["messages"][0]["content"]
        # 生成的问句原样保留（不再按轮数删文本）
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

    def test_call_llm_keeps_question_on_greeting(self, monkeypatch):
        """寒暄轮次也放行尾问句 —— 新总则要求每轮以关心 ta 的问题收尾。

        旧版会因为 ``_is_social_only`` 把「今天想说点什么？」删成陈述句；
        现在只在"别问了 / 情绪太激动"这类**内容判据**上删尾问句。
        """
        class _FakeResp:
            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {
                    "content": "你好呀。今天想说点什么？"
                }}]}

        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(iv.requests, "post", lambda url, **kw: _FakeResp())

        reply = iv._call_llm("你好", [], NEUTRAL_PROBS, "low", style=UserStyle.CALM)
        assert reply == "你好呀。今天想说点什么？"

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

    def test_prompt_no_longer_varies_by_question_history(self, captured_prompt):
        """连问了几轮**不再**改变 prompt —— 旧版的隔轮节拍已删除。

        旧版会因为"连着三轮以问句结尾"注入降温规则，还会按 turn_count%3
        造一个"问、问、接一句"的落地拍；两处都是"每轮话术一个样"的来源。
        """
        history = [
            {"role": "assistant", "content": "发生了什么事？"},
            {"role": "user", "content": "考试没考好"},
            {"role": "assistant", "content": "老师说什么了？"},
            {"role": "user", "content": "老师骂我了"},
            {"role": "assistant", "content": "是在教室里吗？"},
        ]
        iv._call_llm("他说我很废物", history, SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        for gone in ("前面已经连着三轮用问句结尾了", "先不往下问", "本轮**要追问**"):
            assert gone not in prompt, gone
        assert "每一轮回复的最后一句，必须是一个关心 ta 的问题" in prompt


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


class TestClicheDedupAndLength:
    """跨轮陪伴套话去重 + 日常回复长度上限（2026-09-27 用户截图：连续两轮
    都出现"我会一直在这里陪着你"+"慢慢来"，且回复长到两大段）。"""

    def test_used_cliches_scans_full_body(self):
        """套话在句中/句尾也要被识别（区别于 _used_openings 只看句首）。"""
        hist = [{"role": "assistant", "content": "嗯。我会一直在这里陪着你，慢慢来。"}]
        used = iv._used_cliches(hist)
        assert any("我会一直" in c for c in used)
        assert "慢慢来" in used

    def test_trim_removes_repeated_cliche_sentences(self):
        """本轮回复里重复上轮说过的套话句应被删掉。"""
        hist = [{"role": "assistant", "content": "我会一直在这里陪着你。"}]
        new = "被这么说肯定难受。我会一直在这里陪着你。你愿意多说说吗？"
        out = iv._trim_reply(new, iv._used_cliches(hist))
        assert "我会一直在这里陪着你" not in out
        # 非套话内容必须保留
        assert "被这么说肯定难受" in out
        assert "你愿意多说说吗" in out

    def test_trim_enforces_length_cap(self):
        """低风险长回复被压到上限内，且不返回空。"""
        long = "这是一句比较长的话用来测试。" * 12
        out = iv._trim_reply(long, [])
        assert len(out) <= iv._MAX_REPLY_CHARS + 20
        assert out

    def test_crisis_reply_never_truncated(self):
        """危机/高风险含热线与安全确认，一个字都不能删。"""
        crisis = (
            "我非常担心你的安全。你现在是否处于危险中？"
            "请立即拨打24小时心理援助热线：400-161-9995。"
        )
        hist = [{"role": "assistant", "content": "我会一直陪着你，慢慢来。"}]
        out = iv._finalize_reply(
            crisis, True, user_message="我不想活了",
            conversation_history=hist, risk_level="crisis",
        )
        assert "400-161-9995" in out
        assert "安全" in out

    def test_low_risk_dedups_and_caps(self):
        """低风险走完整定稿：套话被删 + 长度受限。"""
        hist = [{"role": "assistant", "content": "我会一直在这里陪着你，慢慢来。"}]
        raw = (
            "听到你这么说我真的很心疼。我会一直在这里陪着你。"
            "我们可以慢慢来。你愿意和我说说具体是什么让你有压力吗？"
        )
        out = iv._finalize_reply(
            raw, True, user_message="我最近压力有点大",
            conversation_history=hist, risk_level="low",
        )
        assert "我会一直在这里陪着你" not in out
        assert len(out) <= iv._MAX_REPLY_CHARS + 20

    def test_social_turn_gets_the_shorter_cap(self):
        """闲聊轮次用 40 字上限（用户要求"一般的闲聊 40 字以内就够了"）。"""
        long = "你好呀，很高兴见到你。今天天气挺好的，你最近过得怎么样呢？" * 3
        out = iv._finalize_reply(long, True, user_message="你好", risk_level="low")
        assert len(out) <= iv._MAX_SOCIAL_REPLY_CHARS + 20
        assert out.endswith("？")

    def test_normal_turn_gets_the_normal_cap(self):
        """有内容的轮次用 100 字上限，且结尾仍是那句关心 ta 的问题。"""
        long = "我听到你说的这些了，这件事确实不容易。" * 8 + "你愿意多说说吗？"
        out = iv._finalize_reply(long, True, user_message="我最近压力很大", risk_level="low")
        assert len(out) <= iv._MAX_REPLY_CHARS + 20
        assert out.endswith("？")


class TestStreamingGateGranularity:
    """闸门按小句释放，让流式观感是"陆续长出"而非"整段蹦出"。"""

    def test_releases_multiple_clause_deltas(self):
        gate = iv._StreamingSafetyGate()
        released: list[str] = []
        for tok in ["你好，", "看到你", "发来的", "消息，", "我有点", "担心。"]:
            released += gate.feed(tok)
        released += gate.flush()
        assert len(released) >= 2

    def test_red_line_still_trips_at_clause(self):
        """小句切分不得放过红线词 —— 整词仍在某一片段里被拦下。"""
        gate = iv._StreamingSafetyGate()
        out: list[str] = []
        for tok in ["你", "有抑", "郁症", "，", "建议", "就医。"]:
            out += gate.feed(tok)
            if gate.tripped:
                break
        # 命中红线后不再放行含该词的内容
        joined = "".join(out)
        assert "你有抑郁症" not in joined


class TestSeverityToRisk:
    """量表中文分级名 → 风险贡献。回归此前用英文 moderate/severe 比较、
    除危机词外风险恒为 low 的 bug（见 api/intervention.py:_severity_to_risk）。"""

    def test_phq9_chinese_levels(self):
        assert iv._severity_to_risk("无抑郁") == "low"
        assert iv._severity_to_risk("轻度抑郁") == "medium"
        assert iv._severity_to_risk("中度抑郁") == "medium"
        assert iv._severity_to_risk("中重度抑郁") == "high"
        assert iv._severity_to_risk("重度抑郁") == "high"

    def test_gad7_and_pss10_levels(self):
        assert iv._severity_to_risk("轻度焦虑") == "medium"
        assert iv._severity_to_risk("中度焦虑") == "medium"
        assert iv._severity_to_risk("重度焦虑") == "high"
        assert iv._severity_to_risk("低压力") == "low"
        assert iv._severity_to_risk("中等压力") == "medium"
        assert iv._severity_to_risk("高压力") == "high"

    def test_empty_and_unknown_are_low(self):
        assert iv._severity_to_risk("") == "low"
        assert iv._severity_to_risk("未知") == "low"


class TestDominantEmotionCue:
    """问题①：把主导情绪映射成"只给模型看"的内部线索。

    背景（2026-09-27 用户反馈）："考试没考好，它没接住那份失落感。"
    根因不是模板代码 —— 回复由 qwen 生成、prompt 里早就禁了套路开场 ——
    而是 prompt **刻意不喂情绪标签**，模型只能从字面猜"该接哪一种情绪"。
    """

    def test_sad_maps_to_finer_grained_word(self):
        cue = iv._dominant_emotion_cue([0.01, 0.96, 0.0, 0.0, 0.03])
        assert "失落" in cue
        # 不能直接用分类名 —— 模型会把它当标签复述（"听起来你很悲伤"）
        assert "悲伤" not in cue

    def test_anxious_maps_to_anxiety_cue(self):
        cue = iv._dominant_emotion_cue([0.0, 0.1, 0.8, 0.0, 0.1])
        assert "悬" in cue or "不安" in cue

    def test_angry_maps_to_anger_cue(self):
        cue = iv._dominant_emotion_cue([0.0, 0.1, 0.1, 0.75, 0.05])
        assert "委屈" in cue or "火" in cue

    @pytest.mark.parametrize(
        "probs",
        [
            [0.13, 0.07, 0.0, 0.0, 0.80],   # 中性占主导（"你好"）
            [0.82, 0.05, 0.05, 0.03, 0.05],  # 快乐占主导
        ],
    )
    def test_non_negative_turn_has_no_cue(self, probs):
        """中性/快乐轮次**必须不给线索** —— 用户什么都没说时凭空共情最伤人。"""
        assert iv._dominant_emotion_cue(probs) == ""

    def test_below_threshold_has_no_cue(self):
        """主导情绪概率不够高时不猜（阈值 0.45）。"""
        assert iv._dominant_emotion_cue([0.2, 0.44, 0.16, 0.1, 0.1]) == ""

    def test_bad_input_is_safe(self):
        assert iv._dominant_emotion_cue([]) == ""
        assert iv._dominant_emotion_cue([None, "x"]) == ""  # type: ignore[list-item]

    def test_cue_is_no_longer_injected(self, captured_prompt):
        """线索行自 prompt 精简后不再注入（只留负性强度数值）。"""
        iv._call_llm("我这次考试没考好", [], [0.01, 0.96, 0.0, 0.0, 0.03], "medium")
        prompt = captured_prompt["messages"][0]["content"]
        assert "失落" not in prompt
        assert "最可能是在" not in prompt


class TestNarrativeThread:
    """问题③：连贯性锚点 —— "ta 这一段正在讲的那条线"。"""

    def test_prefers_previous_user_utterance(self):
        """锚点必须优先取**用户上一句原话**，而不是场景检索的近似场景。"""
        thread = iv._narrative_thread(
            "数学。我复习了一个月还是这样。",
            [
                {"role": "user", "content": "我这次考试没考好"},
                {"role": "assistant", "content": "是哪门课没考好？"},
                {"role": "user", "content": "数学。我复习了整整一个月，结果还是这样。"},
            ],
            scene_context="- 情境：考试失利后被家长责备\n  相近想法：「我完了」→ 灾难化",
        )
        assert "复习了整整一个月" in thread
        assert "家长责备" not in thread

    def test_falls_back_to_scene_when_no_history(self):
        thread = iv._narrative_thread(
            "我很难受", [], scene_context="- 情境：考试没考好\n  相近想法：「我完了」→ 灾难化"
        )
        assert "考试没考好" in thread

    def test_empty_when_nothing_available(self):
        assert iv._narrative_thread("我很难受", []) == ""
        assert iv._narrative_thread("我很难受", [], scene_context="") == ""

    def test_anchor_is_no_longer_injected(self, captured_prompt):
        """锚点自 prompt 精简后不再注入（连贯性交给"ta 说什么你就接什么"）。"""
        iv._call_llm(
            "数学。我复习了整整一个月，结果还是这样。",
            [
                {"role": "user", "content": "我这次考试没考好"},
                {"role": "assistant", "content": "考试没考好这种感觉确实挺让人难受的。"},
            ],
            SAD_PROBS,
            "medium",
        )
        prompt = captured_prompt["messages"][0]["content"]
        assert "# 必须扣住的那条线" not in prompt
        assert "禁止跳题" not in prompt


class TestDemoSentenceReuse:
    """照抄 prompt 示范句（「我听着是」）—— 实测 5 个场景里 3 个这么开头。

    ⚠️ 这里**刻意不做代码级删除**（试过、已回退）：「这话太重了。」本身就是合格的
    共情，删掉它会连共情一起删掉，回复退化成干巴巴的追问。所以只拦"跨轮复用"，
    由 :func:`_used_openings` 精确匹配实现。
    """

    def test_opening_detected_after_first_use(self):
        history = [
            {"role": "user", "content": "我这次考试没考好"},
            {"role": "assistant", "content": "我听着是，这事儿现在还压着你呢？"},
        ]
        assert "我听着是" in iv._used_openings(history)

    def test_first_use_is_not_banned(self):
        """第一次用不算复用 —— 不能第一轮就把它禁掉。"""
        assert iv._used_openings([]) == []
        assert iv._used_openings([{"role": "assistant", "content": "我在。"}]) == []

    def test_reuse_list_is_no_longer_injected(self, captured_prompt):
        """"已用过的开场"不再注入 prompt（清单本身就是照抄源）。"""
        iv._call_llm(
            "我还是觉得难受",
            [
                {"role": "user", "content": "我这次考试没考好"},
                {"role": "assistant", "content": "我听着是，这事儿现在还压着你呢？"},
            ],
            SAD_PROBS,
            "medium",
        )
        prompt = captured_prompt["messages"][0]["content"]
        assert "# 本轮硬性禁用" not in prompt
        assert "我听着是" not in prompt

    def test_demo_constant_records_copy_prone_sentences(self):
        """常量是"已在示范里出现、最容易被照抄"的句子清单（记录 + 测试用）。"""
        assert "我听着是" in iv._DEMO_SENTENCES
        assert "这话太重了" in iv._DEMO_SENTENCES
