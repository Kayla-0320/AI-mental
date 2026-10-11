"""提示词契约与追问规则测试

覆盖三处：

1. **统一提示词**（2026-10-05 首次重写；**2026-10-11 用户两次指定改版**）：
   只保留 AI 身份、风格要求、安全底线与「**提问是选项，不是义务**」的优先级规则；
   不再注入示范对话、逐句禁用清单、情绪线索与节拍规则
   —— 这些示范句本身就是模型最容易照抄的句子（实测连续四轮都以同一句示范开头）。
   ⚠️ 旧版的硬契约是"**每轮**最后一句必须是一个关心 ta 的问题"，实测让每轮都以
   「你觉得呢？」收尾，用户反馈"听起来像机械地执行话术"，2026-10-11 已删除；
   现在只在"这一轮确实要问"时要求问句落在最后一句（口语轮次的交接信号）。
   ⚠️ 2026-10-11 第二次改版：改用用户给的《角色设定：温暖的心理陪伴伙伴》七条原则
   （自然真诚 / 简洁适度 / 先倾听再引导 / 适时提问 / 尊重 / 适合语音 / 保持边界），
   删掉上一版的"按情况回应""说之前先想"两节（用户："提示词我觉得可以少点"），
   并把"敷衍式短回复"写成抽象判据（截图里 ta 只说半句「我说。」→ 回复「我在听。」）。
2. 追问与否仍由 :func:`_ask_question_this_turn` 判定（拒绝表达 / 激动 / 高风险）；
   它现在是**否决权**而不是"必须问"的义务，所以寒暄轮不再单独拦尾问句，
   高风险/危机反而放行（安全确认就是一句问话）。
3. 降级模板在"本轮不追问"时**不得反问**（此前的 CALM 池自身含问句）。
4. 收束/结束语（2026-10-11："当用户说我感觉好些了，就可以不用再问问题了，
   可以说一些结束对话的语言"）：命中好转/结束意图时本轮不追问，prompt 换成
   收束指令（承接 → 复述 → 小提醒 → 留门），并与寒暄规则、「不要提问」互斥。
5. **`speak` 计划的完整性契约**（2026-10-11，用户报告"回复和语音内容不匹配、
   语音输出不全"）：客户端按 `speak.segments` 送合成，所以"分段拼接必须等于
   定稿文本"是文字与语音同源的前提。这里把它测成不变量。

测试不发起任何网络调用：``api.intervention.requests.post`` 被替换为抓取器。
"""
from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api.intervention as iv  # noqa: E402
from intervention.style_detector import UserStyle  # noqa: E402


def _no_space(text: str) -> str:
    """比对用归一化：去掉所有空白。

    ⚠️ 必须归一化再比：`_split_speak_units` 会 `strip()` 每个片段并丢弃纯空白片段，
    所以带空格/换行的回复 `''.join(segments)` **本来就不等于**原文 ——
    那不是契约违约，直接判违约会把正常计划全部误伤。
    """
    return re.sub(r"\s+", "", text or "")

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

    def test_question_is_optional_with_priority_rule(self, captured_prompt):
        """本版核心契约（2026-10-11）：提问是选项，优先级 倾听 > 建议 > 引导性提问。

        ⚠️ 反向断言同样重要：旧版那句"每一轮回复的最后一句，必须是一个关心 ta
        的问题"**必须消失** —— 它正是"每轮都用『你觉得呢？』收尾"的来源。
        """
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "提问是选项，不是义务" in prompt
        assert "ta 只想倾诉" in prompt
        assert "才提一个引导性问题" in prompt
        assert "每一轮回复的最后一句，必须是一个关心 ta 的问题" not in prompt

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
        """防止约束回涨：提示词长度必须控制住（旧版曾膨胀到 3200 字）。

        ⚠️ 上限三次调整的记录（每次都以**实测基线 + 一点余量**为准）：
          · 1600 → 2400（2026-10-11 第一版统一提示词，实测 2287 字）；
          · 2400 → 2000（2026-10-11 用户第二次改版，见 TestPersonaDraft）——
            用户原话"提示词我觉得可以少点"，改用七条原则版后实测 **1829 字**
            （草稿本体 1778 + 针对"半句输入"补的 51 字）。
        留余量的意义：这条断言是**防回涨**的闸门，不是精确断言；再加约束就会顶到它。
        """
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert len(prompt) < 2000, len(prompt)

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

    def test_normal_turn_does_not_forbid_question(self, captured_prompt):
        """普通情绪轮次不该出现"不要提问"，但也不再强制问。

        ``_ask_question_this_turn`` 是否决权：返回 True 只代表"这轮允许问"，
        提示词里的默认动作是"ta 只想倾诉就先接住，可以一个问题都不问"。
        """
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "这一轮**不要提问**" not in prompt
        assert "提问是选项，不是义务" in prompt
        assert "把问话放在最后一句" in prompt

    def test_high_risk_keeps_the_safety_question(self, captured_prompt):
        """高风险/危机不放"不要提问"，反而明示结尾仍要落在关心 ta 的问句上。

        安全确认本身就是一句要问的话：按"高风险不追问"去删尾问句，
        会把「你现在安全吗？」一起删掉。
        """
        iv._call_llm("我不想活了", [], SAD_PROBS, "crisis", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "这一轮**不要提问**" not in prompt
        assert "先按上面「底线」做安全确认" in prompt

    def test_duplicate_trailing_user_turn_is_dropped(self, captured_prompt):
        """回归（2026-10-11）：历史最后一条 == 本轮发言时，**不得**把它喂两遍。

        真机症状（用户截图）："为什么显示我说了两遍？我每个句子都只说了一遍" ——
        AI 连着三轮回「听你轻轻说了两遍」「你连着问了两遍」「你说了两遍『难受』」。
        根因在 server 侧（先落库再取历史，于是把本轮也取了回来），而算法侧又会
        追加本轮发言 —— 模型确实看到了两遍，它没说错，是输入错了。
        """
        iv._call_llm(
            "你好。", [{"role": "user", "content": "你好。"}],
            NEUTRAL_PROBS, "low", style=UserStyle.CALM,
        )
        user_turns = [m["content"] for m in captured_prompt["messages"] if m["role"] == "user"]
        assert user_turns.count("你好。") == 1, user_turns

    def test_duplicate_dedupe_ignores_whitespace(self, captured_prompt):
        """历史里带尾随空白的同一条发言，同样算重复（服务端文本会原样落库）。"""
        iv._call_llm(
            "你好。", [{"role": "user", "content": " 你好。\n"}],
            NEUTRAL_PROBS, "low", style=UserStyle.CALM,
        )
        user_turns = [m["content"] for m in captured_prompt["messages"] if m["role"] == "user"]
        assert len(user_turns) == 1, user_turns

    def test_repeat_across_turns_is_kept(self, captured_prompt):
        """**隔轮**重复同一句话是真实表达，不能被去重删掉（只比紧邻的前一条）。"""
        history = [
            {"role": "user", "content": "你好。"},
            {"role": "assistant", "content": "你好呀。还想说点什么？"},
        ]
        iv._call_llm("你好。", history, NEUTRAL_PROBS, "low", style=UserStyle.CALM)
        user_turns = [m["content"] for m in captured_prompt["messages"] if m["role"] == "user"]
        assert user_turns.count("你好。") == 2, user_turns

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
        """打招呼时只回招呼、不评价 ta 的状态；想邀请 ta 开口就问一句（不是必须）。"""
        iv._call_llm("你好", [], NEUTRAL_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "只说了寒暄" in prompt
        assert "不要评价 ta 的状态" in prompt
        assert "想邀请 ta 开口" in prompt

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


class TestPersonaDraft:
    """用户草稿版提示词的契约（2026-10-11 第二次改版）。

    用户把整段提示词换成了自己写的《角色设定：温暖的心理陪伴伙伴》（七条对话原则
    + 对话目标），并说"提示词我觉得可以少点"。这一版**以草稿为准**，只补三块
    机器与安全上必须的东西（身份/格式长度/不许做+底线），本类把这三块都钉住。

    为什么"少点"不是审美偏好：上一版 2287 字里塞了"分档提问优先级 + 按情况回应 +
    说之前先想 + 禁用清单"，约束条数一多，模型为了同时满足它们会把回复压到最保险的
    形状上 —— 真机截图（ta 只说半句「我说。」）回复「我在听。」就是这种压窄的结果。
    """

    # 草稿里的七条原则标题，一个字都不许改（改了就说明有人又"顺手重写"了草稿）
    PRINCIPLE_TITLES = (
        "**自然真诚**",
        "**简洁适度**",
        "**先倾听，再引导**",
        "**适时提问**",
        "**尊重 ta**",
        "**适合语音交流**",
        "**保持合理边界**",
    )

    def test_persona_is_the_users_draft(self, captured_prompt):
        """草稿本体必须在 prompt 里（标题 + 对话目标）。"""
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "# 角色设定：温暖的心理陪伴伙伴" in prompt
        for title in self.PRINCIPLE_TITLES:
            assert title in prompt, title
        assert "# 对话目标" in prompt
        assert "灵活选择倾听、安慰、共情、提问或建议" in prompt

    def test_content_free_short_reply_is_forbidden_abstractly(self, captured_prompt):
        """真机回归（2026-10-11 截图）：ta 说「我说。」→ 回复「我在听。」

        一句表态、什么内容都没接住，用户反馈"这最后一句回复明显不对"。
        判据写成**抽象的**（"只表个态、什么内容都没接住 = 没回应"），并针对"只剩半句"
        给出可执行的下一步（先复述听出来的方向，再请 ta 接着说）；
        ⚠️ 反面示范**不许**写进 prompt：把「我在听」念一遍就是提示模型说它一次
        （2026-10-05 的事故就是 prompt 里的示范句被整句照抄）。
        """
        iv._call_llm("我说。", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "只表个态、什么内容都没接住的短回复，等于没回应" in prompt
        assert "明显没说完" in prompt
        assert "再请 ta 接着说" in prompt
        assert "细节不许补" in prompt          # 接话头不等于替 ta 编细节
        assert "我在听" not in prompt

    def test_verbose_sections_are_gone(self, captured_prompt):
        """上一版那两节被删掉了 —— 这条断言是"少点"的闸门。

        "按情况回应"（七种情境逐条给打法）与"说之前先想"（思考步骤）都是**额外约束**：
        它们与七条原则重复，却在模型眼里是"必须逐条满足的清单"，正是回复被压窄的来源。
        """
        iv._call_llm("我好难受", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "# 按情况回应" not in prompt
        assert "说之前先想" not in prompt

    def test_safety_and_identity_survive_the_rewrite(self, captured_prompt):
        """改草稿**不能**把身份与安全红线改掉（AGENTS.md：安全规则不得被顺手改）。"""
        iv._call_llm("我不想活了", [], SAD_PROBS, "crisis", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "AI 陪伴者" in prompt
        assert "不能替代线下持证的心理咨询师与精神科医生" in prompt
        assert "# 底线" in prompt
        assert "心理危机热线" in prompt
        assert "不许诊断、不许提药物" in prompt


class TestCapQuestions:
    """一轮最多一个问句 —— 提示词写了但实测仍会抛两个。

    ⚠️ 保留的是**最后**一个问句（2026-10-05 改）：定稿侧要求"要问就问在最后
    一句"（口语轮次的交接信号），保留第一个会把结尾整段砍掉，实测把模型真正想
    问的那句连同后面的内容一起删掉，只剩 9 个字的反问。
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
    """契约：**要问**就问在最后一句（本轮没有问句时原样返回，不补问句）。"""

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
        """寒暄轮次放行尾问句 —— 邀请 ta 开口是**允许**的，不再被删成陈述句。

        提示词把提问写成选项（"想邀请 ta 开口就轻轻问一句，也可以只应一声"），
        代码侧只在"别问了 / 情绪太激动"这类**内容判据**上删尾问句。
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
        assert "提问是选项，不是义务" in prompt


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
        """有内容的轮次用 100 字上限，且结尾那句问话被留在末尾。"""
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
        """线索行自 prompt 精简后不再注入（只留负性强度数值）。

        ⚠️ 断言用线索行的**完整形状**（``失落 —— <说明>``，见
        :func:`_dominant_emotion_cue`）而不是光秃秃的"失落"二字：2026-10-11 的那版
        提示词曾在"按情况回应"一节里正常地列举了"委屈、焦虑、孤独、失落"这些情绪名，
        按词断言会误报成"线索又被注回来了"（那一节已随第二次改版删除，但判据照旧
        —— 情绪名出现在 prompt 里属于正常，只有"线索行"才算注入）。
        """
        iv._call_llm("我这次考试没考好", [], [0.01, 0.96, 0.0, 0.0, 0.03], "medium")
        prompt = captured_prompt["messages"][0]["content"]
        assert "失落 ——" not in prompt
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


class TestClosingIntent:
    """收束 / 结束语判据（2026-10-11 需求："我感觉好些了" → 可以收尾了）。

    只测纯函数 :func:`_closing_intent`：命中面（好转 / 想结束）、不该命中的面
    （否定表达、说的是别人、转折句里还压着负性），以及安全优先（高风险/危机
    一律 None）。prompt 侧的接线见 :class:`TestClosingPrompt`。
    """

    @pytest.mark.parametrize(
        "text",
        [
            "我感觉好些了", "我好多了", "好一点了", "没那么难受了",
            "轻松多了", "心里舒服多了", "想通了", "我明白了", "我知道怎么做了",
        ],
    )
    def test_improvement_detected(self, text):
        assert iv._closing_intent(text, "low") == "improved"

    @pytest.mark.parametrize(
        "text",
        [
            "我先不聊了", "先这样吧", "下次再聊", "我要去忙了", "我去学习了",
            "拜拜", "再见", "谢谢你，我好多了",
        ],
    )
    def test_ending_detected(self, text):
        assert iv._closing_intent(text, "low") == "ending"

    @pytest.mark.parametrize(
        "text",
        [
            "我不好", "我不好受", "我不太好", "我没好", "我没好起来",
            "还是老样子", "我不知道该怎么办",
            "你好", "谢谢", "好的", "我好难受", "好",
        ],
    )
    def test_negation_and_small_talk_do_not_hit(self, text):
        """`好` 单独出现不算；"我不好 / 我不好受 / 我没好"必须不命中。"""
        assert iv._closing_intent(text, "low") is None

    @pytest.mark.parametrize("risk", ["high", "crisis"])
    def test_high_risk_never_closes(self, risk):
        """"我好多了"在危机后恰恰是最需要核实的一句 —— 不许据此收尾。"""
        assert iv._closing_intent("我好多了", risk) is None
        assert iv._closing_intent("先这样吧", risk) is None

    def test_crisis_word_in_same_sentence_blocks_improvement(self):
        """"我好多了，但有时候还是想死" 绝不能判成好转。"""
        assert iv._closing_intent("我好多了，但有时候还是想死", "low") is None

    def test_improvement_about_someone_else_is_not_ta(self):
        """不带主语的"好多了"若在说别人，不算本轮好转。"""
        assert iv._closing_intent("我朋友好多了，可我还是这样", "low") is None
        assert iv._closing_intent("她好多了", "low") is None

    def test_contrast_clause_with_residual_distress_blocks_closing(self):
        """好转被同一句的转折句削弱时不收尾（宁可不收，也别提前关门）。"""
        assert iv._closing_intent("我好多了，但还是有点担心", "low") is None

    def test_first_person_improvement_after_address_still_hits(self):
        """主语守卫只拦"说的是别人"，"妈妈，我好多了"仍是 ta 自己的好转。"""
        assert iv._closing_intent("妈妈，我好多了", "low") == "improved"

    def test_empty_message_is_none(self):
        assert iv._closing_intent("", "low") is None
        assert iv._closing_intent("   ", "low") is None

    def test_ask_question_is_disabled_on_closing_turn(self):
        """收束轮不再追问（两个调用点共用这一处判定）。"""
        assert iv._ask_question_this_turn(
            UserStyle.CALM, "low", "我感觉好些了"
        ) is False

    def test_ask_question_still_true_for_plain_distress(self):
        """防误伤：普通情绪轮该问还是要问。"""
        assert iv._ask_question_this_turn(
            UserStyle.CALM, "low", "我今天被老师批评了，很难受",
            negative_intensity=0.8,
        ) is True


class TestQuotableUserLines:
    """收束复述的白名单：**只能**把用户原话交给模型（防幻觉的最后一环）。

    2026-10-11 真机回归：空历史时模型编出了「刚才你提到，试着把心里的话说出来」。
    提示词禁令压不住，改成"把可复述的原话列出来"这种结构性约束（见
    :func:`_quotable_user_lines` 的 docstring）。
    """

    def test_keeps_only_user_lines(self):
        hist = [
            {"role": "user", "content": "我复习了一个月还是考砸了"},
            {"role": "assistant", "content": "复习一个月还是这个结果，挺难接受的。"},
            {"role": "system", "content": "内部说明"},
        ]
        assert iv._quotable_user_lines(hist) == ["我复习了一个月还是考砸了"]

    def test_drops_blank_and_bare_acknowledgements(self):
        """「嗯 / 好」这类应声复述回去像学舌，也不算"有用的一两点"。"""
        hist = [
            {"role": "user", "content": "   "},
            {"role": "user", "content": "嗯"},
            {"role": "user", "content": "好"},
            {"role": "user", "content": "我今天特别累，什么都不想干"},
        ]
        assert iv._quotable_user_lines(hist) == ["我今天特别累，什么都不想干"]

    def test_keeps_only_the_most_recent_lines(self):
        hist = [{"role": "user", "content": f"第{i}句话说说看"} for i in range(5)]
        assert iv._quotable_user_lines(hist) == [
            "第2句话说说看", "第3句话说说看", "第4句话说说看",
        ]

    def test_dedupes_and_normalizes_whitespace(self):
        """完全相同的原话只列一次；换行/多余空格归一成一个空格（不断词）。

        ⚠️ 归一后**不同**的文本不算重复：「我今天\\n很累」会变成「我今天 很累」，
        和「我今天很累」是两条 —— 去重按归一后的字面相等，不做模糊匹配。
        """
        hist = [
            {"role": "user", "content": "我今天很累"},
            {"role": "user", "content": " 我今天很累 "},
            {"role": "user", "content": "我今天\n很累"},
            {"role": "user", "content": "我今天很烦"},
        ]
        assert iv._quotable_user_lines(hist) == [
            "我今天很累", "我今天 很累", "我今天很烦",
        ]

    def test_bad_input_is_safe(self):
        assert iv._quotable_user_lines([]) == []
        assert iv._quotable_user_lines(None) == []          # type: ignore[arg-type]
        assert iv._quotable_user_lines(["x"]) == []         # type: ignore[list-item]
        assert iv._quotable_user_lines([{"role": "user"}]) == []


class TestClosingPrompt:
    """收束轮与高危轮的 prompt 契约（走 :func:`_call_llm` 的真实接线）。"""

    # 收束指令的关键短语。断言"在不在"比断言整段话术更稳：
    # 话术会随用户反馈改措辞，而"这一轮要收束、要留门"是契约。
    CLOSING_MARKERS = ("收束轮", "留门", "结尾不要出现问号")
    # 断言禁语本身也在 prompt 里（以"绝不许说"的形式），守住安全红线。
    FORBIDDEN_ASSERTIONS = ("你已经完全好了", "你没事了", "你治好了")

    def test_improved_turn_switches_to_closing(self, captured_prompt):
        iv._call_llm("我感觉好些了", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        for marker in self.CLOSING_MARKERS:
            assert marker in prompt
        assert "说自己比刚才好一些了" in prompt
        # 禁语清单要在 prompt 里（否则模型会把"听你这么说"写成疗效承诺）
        assert "绝不许" in prompt
        for phrase in self.FORBIDDEN_ASSERTIONS:
            assert phrase in prompt

    def test_ending_turn_switches_to_closing(self, captured_prompt):
        iv._call_llm("我先不聊了", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        for marker in self.CLOSING_MARKERS:
            assert marker in prompt
        assert "想先聊到这里" in prompt

    def test_closing_forbids_reciting_things_ta_never_said(self, captured_prompt):
        """真机回归（2026-10-11）：收束第 2 步的"复述"是幻觉源。

        实测（空历史 + 用户只说"我感觉好些了，谢谢你陪我聊"）模型编出了
        「刚才你说试着把心里的话说出来，整个人就轻松了一点；还提到今天早饭
        吃得很慢，也没催自己」—— ta 一个字都没说过。收束话术**必须**写明
        只能复述真说过的话、想不起来就跳过这一步，否则它正好绕过"不许编"。
        """
        iv._call_llm("我感觉好些了", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "只能复述 ta 真的说过的话" in prompt
        assert "直接跳过这一步" in prompt
        assert "刚才你提到" not in prompt   # 反面示范本身也不许写进 prompt

    def test_closing_recital_is_whitelisted_to_the_users_own_words(self, captured_prompt):
        """有历史时，可复述的原话被逐条列出来 —— 模型没有"凭空写"的余地。"""
        history = [
            {"role": "user", "content": "我复习了一个月还是考砸了"},
            {"role": "assistant", "content": "复习一个月还是这个结果，挺难接受的。"},
            {"role": "user", "content": "嗯"},
        ]
        iv._call_llm("我感觉好些了", history, SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "只从下面这几句 ta **自己说过**的原话里" in prompt
        assert "我复习了一个月还是考砸了" in prompt
        assert "只能" in prompt

    def test_closing_is_mutually_exclusive_with_social_and_no_question(self, captured_prompt):
        """收束 > 寒暄/不提问：三段本轮规则最多只能生效一条。

        收束轮要求"结尾不要出现问号"，寒暄轮则建议"想邀请 ta 开口就问一句"
        —— 两段并存时模型会挑更显眼的那条，收束就静默失效了。
        """
        iv._call_llm("我感觉好些了", [], SAD_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "收束轮" in prompt
        assert "想邀请 ta 开口" not in prompt            # 寒暄规则已让位
        assert "这一轮**不要提问**" not in prompt       # 通用"不要提问"也已让位

    def test_plain_distress_turn_has_no_closing(self, captured_prompt):
        """防误伤：普通情绪轮不得出现收束指令。"""
        iv._call_llm(
            "我今天被老师批评了，很难受", [], SAD_PROBS, "low", style=UserStyle.CALM
        )
        prompt = captured_prompt["messages"][0]["content"]
        assert "收束轮" not in prompt
        assert "留门" not in prompt

    def test_high_risk_turn_keeps_safety_confirmation(self, captured_prompt):
        """高风险轮即便 ta 说"我好多了"，也必须继续走安全确认。"""
        iv._call_llm("我好多了", [], SAD_PROBS, "high", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "收束轮" not in prompt
        assert "先按上面「底线」做安全确认" in prompt

    def test_greeting_turn_keeps_its_own_rule(self, captured_prompt):
        """回归：收束逻辑不得把普通寒暄轮的打法改掉。"""
        iv._call_llm("你好", [], NEUTRAL_PROBS, "low", style=UserStyle.CALM)
        prompt = captured_prompt["messages"][0]["content"]
        assert "想邀请 ta 开口" in prompt
        assert "收束轮" not in prompt

class TestSpeakPlanContract:
    """`speak` 计划的完整性契约：**分段拼接必须等于定稿文本**（按空白归一化）。

    为什么把它当成不变量：客户端是**按 `speak.segments` 送合成**的，字幕也以
    `speak.text` 为准。两者一旦对不上，"听到的"和"看到的"从源头就分叉了 ——
    2026-10-11 用户报告的"回复和语音内容完全不匹配"正是这一类。
    """

    CASES = (
        "听起来今天在学校挺不好受的。老师批评你的时候，哪句话最扎心？",
        "嗯， 我在。今天想从哪儿说起？",          # 句中带空格（服务端会 strip）
        "可以打 400 161 9995 找值班老师。",        # 号码：客户端有意不合成
        "你的 CBT 练习做得不错，继续这样。",        # 英文缩写（术语替换发生在更下游）
        "第一句。第二句！第三句？第四句；第五句",
        "这是一句没有标点的很长的话" * 8,          # 超长无标点
        "……",                                   # 纯标点
        "a b c d e. f g h?",
    )

    @pytest.mark.parametrize("text", CASES)
    def test_split_speak_units_is_lossless(self, text):
        segs = iv._split_speak_units(text)
        if not _no_space(text):
            assert segs == []
            return
        assert _no_space("".join(segs)) == _no_space(text), (
            f"分段丢了字或改了字：{segs!r}"
        )

    @pytest.mark.parametrize("text", CASES)
    def test_plan_segments_match_planned_text(self, text):
        if not _no_space(text):
            return
        planned, segs = iv._plan_streamed_reply(
            text, True, user_message="我今天有点累",
            conversation_history=[], risk_level="low",
        )
        if segs is None:
            # 定稿改动了文本时**刻意不给计划**（宁可不给，也不给对不上的计划）
            assert _no_space(planned) != _no_space(text)
            return
        assert _no_space("".join(segs)) == _no_space(planned)

    def test_plan_refuses_when_finalize_rewrites_text(self, monkeypatch):
        """定稿改动了文本时必须返回 segments=None（宁可不给计划，也不给对不上的计划）。

        契约原文：*"`segments` 为 None 表示计划不可信（定稿改动了文本），
        消费端应退回按 delta 自行切分 —— 宁可不给计划，也不能给出与 text 对不上的计划。"*

        这里把 `_finalize_reply` 换成"一定会改写"的桩，测的是**分支本身**；
        定稿具体会改什么由其它测试覆盖。刻意**不用**"两个问句"做样例：
        问句是在 `_cap_questions` 阶段压掉的，那时计划还没生成，属于正常路径。
        """
        monkeypatch.setattr(
            iv, "_finalize_reply",
            lambda text, *a, **k: text + "（审计改写）",
        )
        planned, segs = iv._plan_streamed_reply(
            "你好呀。", True, user_message="你好",
            conversation_history=[], risk_level="low",
        )
        assert planned.endswith("（审计改写）")
        assert segs is None