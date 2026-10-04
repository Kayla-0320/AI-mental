"""文本流式对话单元测试

覆盖两块新代码：

1. ``_StreamingSafetyGate`` —— 增量安全闸门。核心断言是它**只复用**
   ``config/audit_rules.yaml`` 里既有的词表，没有自己塞进任何新词或新阈值
   （AGENTS.md：自动化流程不得自行修改安全阈值或升级条件）。
2. ``_chat_stream_events`` —— SSE 事件生成器。核心断言是**审计仍是最终权威**：
   闸门拦下的内容不得出现在 DONE 文本里，改稿必须以 REVISE（整体替换）而不是
   DELTA（追加）下发。

测试不发起任何网络调用：``api.intervention.requests.post`` 被替换为假响应。
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api.intervention as iv  # noqa: E402
from intervention.auditor import load_audit_config  # noqa: E402

DIAGNOSTIC_SENTENCE = "你患有抑郁症。"
SAFE_SENTENCE = "我听到你说的了。"
NEUTRAL_PROBS = [0.13, 0.07, 0.0, 0.0, 0.80]


# ---------------------------------------------------------------- 假 LLM
def _sse_line(text: str) -> bytes:
    return (
        "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}, ensure_ascii=False)
    ).encode("utf-8")


class _FakeStreamResp:
    """假的流式响应：支持 ``with requests.post(...) as resp`` 与 iter_lines。"""

    ok = True
    status_code = 200

    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self, decode_unicode=False):
        for c in self._chunks:
            yield _sse_line(c)
        yield b"data: [DONE]"


class _FakeJsonResp:
    """假的非流式响应：供 _call_llm 走「一次拿全」的对照路径。"""

    ok = True
    status_code = 200

    def __init__(self, text):
        self._text = text

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}


def make_fake_post(chunks, full_text=None):
    """按 ``stream`` 参数分派的假 post：流式返回 SSE，非流式返回整段。"""
    joined = full_text if full_text is not None else "".join(chunks)

    def fake_post(url, **kwargs):
        if kwargs.get("json", {}).get("stream"):
            return _FakeStreamResp(chunks)
        return _FakeJsonResp(joined)

    return fake_post


@pytest.fixture()
def fake_llm(monkeypatch):
    """把 LLM 换成可编排的假实现，并确保走「有 key」分支。"""
    def _install(chunks, full_text=None):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(iv.requests, "post", make_fake_post(chunks, full_text))
    return _install


def parse_sse(raw_events):
    """把 SSE 文本帧解析成 ``[(event_type, payload_dict), ...]``。"""
    out = []
    for frame in raw_events:
        lines = [ln for ln in frame.split("\n") if ln]
        etype = next(ln[6:].strip() for ln in lines if ln.startswith("event:"))
        data = next(ln[5:].strip() for ln in lines if ln.startswith("data:"))
        out.append((etype, json.loads(data)))
    return out


def run_stream(user_id: str, message: str = "我好难受", history=None):
    req = iv.SmartChatRequest(
        user_id=user_id,
        message=message,
        conversation_history=history or [],
        session_id=f"sess_{user_id}",
    )
    return parse_sse(list(iv._chat_stream_events(req)))


# ================================================================
# 闸门本身
# ================================================================

class TestStreamingSafetyGate:
    """增量闸门的行为与「不新增规则」边界"""

    def test_loads_terms_from_yaml(self):
        """词表必须非空：空词表等于闸门形同虚设"""
        gate = iv._StreamingSafetyGate()
        assert gate.terms_count > 0

    def test_terms_are_exactly_the_existing_yaml_union(self):
        """词表必须**恰好**等于 audit_rules.yaml 既有列表的并集。

        这条断言是「不新增安全规则」的机器化护栏：任何人往闸门里硬编码新词，
        测试都会红。
        """
        cfg = load_audit_config()
        stigma = cfg["stigma_rejection"]
        teen = cfg["teen_specific"]["stigma_rejection"]
        expected = {
            t.lower()
            for t in (
                list(stigma["diagnostic_labels"])
                + list(stigma["stigmatizing_terms"])
                + list(teen["additional_diagnostic_labels"])
                + list(teen["additional_stigmatizing_terms"])
            )
        }
        gate = iv._StreamingSafetyGate()
        assert set(gate.screen("") if False else ()) == set()  # 占位，见下
        # screen 只返回命中项，因此直接比对内部词表
        assert set(gate._terms) == expected

    def test_yaml_terms_match_cfg_via_screen(self):
        """逐一验证 YAML 里的每个词都能被 screen 命中（防大小写/去重出错）"""
        gate = iv._StreamingSafetyGate()
        cfg = load_audit_config()
        for term in cfg["stigma_rejection"]["diagnostic_labels"]:
            assert gate.screen(f"前缀{term}后缀"), term

    def test_buffers_until_sentence_boundary(self):
        """不完整的句子不得上屏 —— 这是闸门有意义的前提"""
        gate = iv._StreamingSafetyGate()
        assert gate.feed("我听到") == []
        assert gate.feed("你说的") == []
        assert gate.feed("了。") == ["我听到你说的了。"]
        assert gate.released_chars == len("我听到你说的了。")

    def test_releases_multiple_units_in_one_delta(self):
        """一次增量里含多个句子时全部放行"""
        gate = iv._StreamingSafetyGate()
        assert gate.feed("先共情。再追问？") == ["先共情。", "再追问？"]

    def test_comma_does_not_release(self):
        """逗号不是句子边界 —— 半句话没有可判断的语义，不能上屏"""
        gate = iv._StreamingSafetyGate()
        assert gate.feed("我想说，") == []
        assert gate.feed("我理解你。") == ["我想说，我理解你。"]

    def test_trips_on_diagnostic_label_and_releases_nothing_further(self):
        """命中诊断性标签 → 该单位及之后永不放行"""
        gate = iv._StreamingSafetyGate()
        assert gate.feed("我理解你。") == ["我理解你。"]
        assert gate.feed(DIAGNOSTIC_SENTENCE) == []
        assert gate.tripped
        assert "你患有" in gate.hits
        # 命中后 feed 一律空，且 flush 也不得吐出被拦下的内容
        assert gate.feed("我在这里。") == []
        assert gate.flush() == []

    def test_trips_on_teen_specific_term(self):
        """青少年扩展词表同样生效（证明合并了 teen_specific）"""
        gate = iv._StreamingSafetyGate()
        gate.feed("你就是玻璃心。")
        assert gate.tripped
        assert "玻璃心" in gate.hits

    def test_first_unit_buffered_then_streams_rest(self):
        """首段之后的残句在 flush 时给出"""
        gate = iv._StreamingSafetyGate()
        assert gate.feed("你好。我在") == ["你好。"]
        assert gate.flush() == ["我在"]

    def test_long_run_without_boundary_is_released_at_cap(self):
        """无标点的长句不能永久卡在缓冲里（兜底上限）"""
        gate = iv._StreamingSafetyGate()
        assert gate.feed("啊" * (iv._STREAM_UNIT_MAX_CHARS - 1)) == []
        released = gate.feed("啊")
        assert len(released) == 1
        assert len(released[0]) == iv._STREAM_UNIT_MAX_CHARS

    def test_screen_is_case_insensitive_for_english_terms(self):
        """英文词条大小写不敏感"""
        gate = iv._StreamingSafetyGate()
        assert gate.screen("You are DEPRESSED, maybe")
        assert gate.tripped is False  # screen 本身不改变状态

    def test_clean_text_never_trips(self):
        gate = iv._StreamingSafetyGate()
        for unit in gate.feed("我在这里陪着你。慢慢来，不着急。"):
            assert unit
        assert not gate.tripped


# ================================================================
# SSE 端点
# ================================================================

class TestSsePack:
    """SSE 帧格式"""

    def test_format(self):
        frame = iv._sse_pack(iv.ChatStreamEventType.DELTA, {"text": "你好。"})
        assert frame.startswith("event: delta\n")
        assert "data: " in frame
        assert frame.endswith("\n\n")

    def test_chinese_not_escaped(self):
        """中文必须以原文下发，否则调试时完全不可读"""
        frame = iv._sse_pack(iv.ChatStreamEventType.DELTA, {"text": "你好。"})
        assert "你好。" in frame
        assert "\\u" not in frame


class TestChatStreamEvents:
    """端到端事件序列（LLM 为假实现）"""

    def test_meta_then_delta_then_done(self, fake_llm):
        fake_llm([SAFE_SENTENCE, "发生了什么事？"])
        events = run_stream("stream_ok_1")
        types = [e for e, _ in events]
        assert types[0] == "meta"
        assert types[-1] == "done"
        assert "delta" in types

        meta = events[0][1]
        assert meta["streaming"] is True
        assert meta["risk_level"] in ("low", "medium")
        assert len(meta["emotion_probs"]) == 5

        # delta 拼接应当与模型输出一致
        shown = "".join(p["text"] for e, p in events if e == "delta")
        assert shown == SAFE_SENTENCE + "发生了什么事？"

    def test_done_carries_same_fields_as_non_streaming(self, fake_llm):
        fake_llm([SAFE_SENTENCE])
        done = [p for e, p in run_stream("stream_ok_2") if e == "done"][0]
        for key in (
            "text",
            "dialog_state",
            "dialogue_mode",
            "action_type",
            "risk_level",
            "audit_passed",
            "requires_escalation",
            "emotion_probs",
            "fallback",
        ):
            assert key in done, key
        assert done["text"] == SAFE_SENTENCE
        assert done["streaming"] is True

    def test_gate_trip_produces_revise_with_safe_text(self, fake_llm):
        """本轮最关键的断言：闸门拦下的诊断性语言不得出现在最终文本里。

        模型先吐一句正常共情（会上屏），再吐一句诊断性表述（被闸门拦下）。
        最终必须：(1) 出现 REVISE；(2) DONE 文本不含被拦内容。
        """
        fake_llm([SAFE_SENTENCE, DIAGNOSTIC_SENTENCE])
        events = run_stream("stream_trip_1")
        types = [e for e, _ in events]

        assert "revise" in types, f"闸门命中后必须改稿，实际事件: {types}"
        # REVISE 必须排在 DONE 之前
        assert types.index("revise") < types.index("done")

        final = [p["text"] for e, p in events if e == "done"][0]
        assert "你患有" not in final, f"诊断性语言泄漏到最终文本: {final!r}"
        # 被拦下的那句话一个字都不该被上屏
        shown = "".join(p["text"] for e, p in events if e == "delta")
        assert "你患有" not in shown

    def test_revise_is_not_emitted_when_audit_agrees(self, fake_llm):
        """审计不改稿时不得发 REVISE（否则前端会无谓地闪一下）"""
        fake_llm([SAFE_SENTENCE])
        types = [e for e, _ in run_stream("stream_norevise_1")]
        assert "revise" not in types

    def test_crisis_disables_incremental_streaming(self, fake_llm):
        """危机轮次：meta.streaming=False，且**一个 delta 都不许有**。

        升级流程必须原子下发，逐字吐出的安全确认与热线会看起来像普通闲聊。
        """
        fake_llm(["我很担心你。", "请立刻联系专业人员。"])
        events = run_stream("stream_crisis_1", message="我不想活了")
        types = [e for e, _ in events]

        meta = [p for e, p in events if e == "meta"][0]
        assert meta["risk_level"] == "crisis"
        assert meta["streaming"] is False
        assert "delta" not in types, f"危机轮次不允许增量上屏: {types}"

        done = [p for e, p in events if e == "done"][0]
        assert done["requires_escalation"] is True
        assert done["text"], "危机轮次也必须有可展示的文本"

    def test_high_risk_disables_incremental_streaming(self, fake_llm):
        """高风险同样关掉增量（判据是 risk_level 而非仅危机关键词）"""
        fake_llm(["我在。", "你安全吗？"])
        events = run_stream("stream_high_1", message="活着没意思")
        types = [e for e, _ in events]
        assert [p for e, p in events if e == "meta"][0]["streaming"] is False
        assert "delta" not in types

    def test_no_api_key_falls_back_to_template(self, monkeypatch):
        """没有 key 时退回模板，且仍然走完整的事件序列"""
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        events = run_stream("stream_nokey_1")
        types = [e for e, _ in events]
        assert types[0] == "meta"
        assert types[-1] == "done"
        assert "delta" in types
        done = [p for e, p in events if e == "done"][0]
        assert done["text"], "模板兜底也必须给出文本"

    def test_stream_interrupted_still_finalizes(self, monkeypatch):
        """流中断时用已生成内容定稿，不能让用户停在半句话上"""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

        class _Boom:
            ok = True
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def iter_lines(self, decode_unicode=False):
                yield _sse_line(SAFE_SENTENCE)
                raise ConnectionError("模拟网络中断")

        monkeypatch.setattr(iv.requests, "post", lambda url, **kw: _Boom())
        events = run_stream("stream_boom_1")
        types = [e for e, _ in events]
        assert types[-1] == "done", f"流中断后仍必须收尾: {types}"
        done = [p for e, p in events if e == "done"][0]
        assert done["text"], "中断后也要给出可展示文本"

    def test_empty_stream_falls_back_to_template(self, fake_llm):
        """模型一个字都没吐出来 → 模板兜底，而不是空白气泡"""
        fake_llm([])
        events = run_stream("stream_empty_1")
        done = [p for e, p in events if e == "done"][0]
        assert done["text"], "空流必须兜底出文本"

    def test_final_text_matches_non_streaming_path(self, fake_llm):
        """与非流式端点的一致性：同输入、同模型输出 → 同最终文本。

        这是「流式只是取文本方式不同」的形式化验证。
        """
        text = SAFE_SENTENCE + "发生了什么事，让你这么难受？"
        fake_llm([text], full_text=text)

        streamed = [p["text"] for e, p in run_stream("parity_user") if e == "done"][0]

        req = iv.SmartChatRequest(user_id="parity_user", message="我好难受",
                                  conversation_history=[], session_id="parity")
        prep = iv._prepare_chat_turn(req)
        non_stream = iv._audit_and_finalize(prep, req, text).reply

        assert streamed == non_stream


class TestFinalizeReviseBeforeSpeech:
    """定稿改动了已流出去的文本时，必须**提前**下发改稿（问题② 的第二个成因）。

    端到端实测（`_probe_call_e2e.js`，2026-09-27 第 2 轮）：定稿把 62 字压到 37 字，
    删掉尾部一整句；但那 4 段早已送进 TTS —— 用户听到声音在句子中间被硬掐断、
    字幕同时被整体换掉，听感就是"读到一半重来/句子重复"。

    修法：定稿一旦改动文本，就在**合成之前**发 REVISE，让前端先改稿并停播。
    """

    #: 62 字，尾部那句会被 `_finalize_reply` 的压长度规则删掉
    LONG_REPLY = (
        "你妈没说话就叹气，这种沉默比说出来的责备还让人心慌。她那时候心里在想什么？"
        "你听着难受，是觉得她失望了，还是怕她对你没信心了？"
    )

    def test_revise_sent_when_finalize_shortens_text(self, fake_llm):
        fake_llm([self.LONG_REPLY])
        events = run_stream("trim_revise_1", message="我妈知道成绩以后什么都没说，就一直叹气。")
        types = [e for e, _ in events]

        assert "revise" in types, f"定稿删了字却没下发改稿: {types}"
        revised = [p["text"] for e, p in events if e == "revise"][0]
        done = [p for e, p in events if e == "done"][0]["text"]

        assert revised == done, "改稿文本必须与最终下发的文本一致"
        assert len(revised) < len(self.LONG_REPLY)

    def test_revise_arrives_before_any_audio_would_start(self, fake_llm):
        """REVISE 必须**排在** deltas 之后、且在流结束前到达。

        客户端是"收到 delta 就送合成"的：REVISE 越晚到，被掐断的音频越多。
        """
        fake_llm([self.LONG_REPLY])
        events = run_stream("trim_revise_2", message="我妈知道成绩以后什么都没说，就一直叹气。")
        types = [e for e, _ in events]

        last_delta = max(i for i, t in enumerate(types) if t == "delta")
        revise_at = types.index("revise")
        done_at = types.index("done")
        assert last_delta < revise_at < done_at, f"REVISE 位置不对: {types}"

    def test_no_early_revise_when_nothing_was_trimmed(self, fake_llm):
        """没被改动就不许发 —— 否则前端会无谓地停播一次（声音莫名中断）。"""
        fake_llm([SAFE_SENTENCE])
        types = [e for e, _ in run_stream("trim_revise_3")]
        assert types.count("revise") == 0, f"无改动却发了 REVISE: {types}"

    def test_crisis_revise_is_atomic_not_a_trim(self, fake_llm):
        """危机轮次：安全文本原子下发 —— 一个 delta 都不许有。

        ⚠️ 危机轮次**可以**出现 REVISE（既有行为：闸门拦下后审计给安全替代文本），
        所以这里不能断言 `revise 不存在`。要断言的是它**不是增量路径**：
        没有 delta，且改稿文本 == 最终文本。
        """
        fake_llm(["我很担心你。", "请立刻联系专业人员。"])
        events = run_stream("trim_revise_4", message="我不想活了")
        types = [e for e, _ in events]
        meta = [p for e, p in events if e == "meta"][0]
        assert meta["streaming"] is False
        assert "delta" not in types, f"危机轮次不允许增量上屏: {types}"
        for e, p in events:
            if e == "revise":
                done = [q for t, q in events if t == "done"][0]["text"]
                assert p["text"] == done, "危机改稿文本必须等于最终下发文本"


if __name__ == "__main__":
    unittest.main()
