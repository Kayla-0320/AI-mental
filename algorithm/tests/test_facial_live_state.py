"""面部实时状态 → prompt 的契约、闸门与**安全边界**回归测试。

背景（2026-10-11）：用户要求"面部识别结果作为临时提示词进入 AI 问答与电话板块"。
本轮实现的口径是**安全版**：

* 客户端把本机三分类表情读数打成一份实时状态摘要（`client/src/services/facialLiveState.ts`），
  随对话/通话请求上行，算法层经 :func:`api.intervention._build_facial_context` 转成一段
  **低权重内部参考**注入 system prompt；
* 它**不参与** `risk_level`、融合权重、量表分与任何审计判定 —— 这是本文件重点守的边界。

测试不发起任何网络调用：需要 LLM 的地方一律替换 ``api.intervention.requests.post``。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import api.intervention as iv  # noqa: E402

# 一份"合法"的客户端摘要（字段与 client/src/services/facialLiveState.ts 的 FacialLiveState 一致）
GOOD_STATE = {
    "version": iv._FACIAL_LIVE_STATE_VERSION,
    "sampledAt": 1_700_000_000_000,
    "ageMs": 320,
    "present": True,
    "probabilities": {"happy": 0.12, "neutral": 0.71, "sad": 0.17},
    "dominant": "neutral",
    "confidence": 0.71,
    "offMapProbability": 0.0,
    "sampleHz": 9.4,
    "expressionIntensity": 0.3,
    "provider": "wasm",
}

# 一张"看起来很悲伤"的脸 —— 用它来证明它**不会**抬高风险等级
SAD_STATE = {
    **GOOD_STATE,
    "probabilities": {"happy": 0.02, "neutral": 0.08, "sad": 0.90},
    "dominant": "sad",
    "confidence": 0.90,
    "expressionIntensity": 0.85,
}

# 中性偏高的文本情绪 [快乐, 悲伤, 焦虑, 愤怒, 中性]
NEUTRAL_PROBS = [0.13, 0.07, 0.0, 0.0, 0.80]


def _with(**over):
    d = dict(GOOD_STATE)
    d.update(over)
    return d


class TestFacialContextGates:
    """任何一条前置条件不满足都必须返回空串（"宁可漏报不误报"）。"""

    def test_valid_state_produces_guardrailed_text(self):
        text = iv._build_facial_context(GOOD_STATE)
        assert text
        # 只给比例，不给"ta 很悲伤"这类结论
        assert "0.12" in text and "0.71" in text and "0.17" in text
        # 必须带使用禁令与"不是风险信号"声明
        assert "不许" in text
        assert "风险信号" in text
        # 不得出现任何诊断性词汇（AGENTS.md 禁止诊断性语言）
        for banned in ("抑郁症", "焦虑症", "诊断", "确诊"):
            assert banned not in text

    def test_no_diagnosis_label_is_emitted(self):
        """即便读数是 sad 占 0.90，文本里也不许出现"悲伤"作为对 ta 的断言。

        允许出现的只有比例与"开心 / 中性 / 悲伤"的顺序图例。
        """
        text = iv._build_facial_context(SAD_STATE)
        assert text
        assert "0.90" in text
        # 图例本身只出现一次（顺序说明），不得出现"ta 很悲伤"这类表述
        assert "ta 很悲伤" not in text
        assert "你很悲伤" not in text

    @pytest.mark.parametrize("payload", [None, "x", 123, [], {"foo": "bar"}])
    def test_non_dict_or_empty_payload_is_ignored(self, payload):
        assert iv._build_facial_context(payload) == ""

    def test_version_mismatch_is_ignored(self):
        assert iv._build_facial_context(_with(version=999)) == ""
        assert iv._build_facial_context(_with(version="1")) == ""

    def test_missing_version_is_ignored(self):
        assert iv._build_facial_context({k: v for k, v in GOOD_STATE.items() if k != "version"}) == ""

    def test_stale_reading_is_ignored(self):
        """人脸离开画面后旧读数不能被当成"此刻的表情"。"""
        assert iv._build_facial_context(_with(ageMs=iv._FACIAL_MAX_AGE_MS + 1)) == ""

    def test_age_boundary_is_inclusive(self):
        assert iv._build_facial_context(_with(ageMs=iv._FACIAL_MAX_AGE_MS)) != ""

    @pytest.mark.parametrize("bad_age", [-1, "320", None, True, float("nan")])
    def test_invalid_age_is_ignored(self, bad_age):
        assert iv._build_facial_context(_with(ageMs=bad_age)) == ""

    def test_absent_face_is_ignored(self):
        assert iv._build_facial_context(_with(present=False)) == ""
        assert iv._build_facial_context({k: v for k, v in GOOD_STATE.items() if k != "present"}) == ""

    def test_missing_or_partial_probabilities_are_ignored(self):
        assert iv._build_facial_context(_with(probabilities=None)) == ""
        assert iv._build_facial_context(_with(probabilities={"happy": 0.5, "neutral": 0.5})) == ""
        assert iv._build_facial_context(
            _with(probabilities={"happy": 0.5, "neutral": 0.5, "sad": "0.0"})
        ) == ""

    def test_out_of_range_probability_is_ignored(self):
        assert iv._build_facial_context(
            _with(probabilities={"happy": 1.5, "neutral": 0.0, "sad": 0.0})
        ) == ""
        assert iv._build_facial_context(
            _with(probabilities={"happy": -0.1, "neutral": 0.6, "sad": 0.5})
        ) == ""

    def test_all_zero_probabilities_are_ignored(self):
        assert iv._build_facial_context(
            _with(probabilities={"happy": 0.0, "neutral": 0.0, "sad": 0.0})
        ) == ""


class TestPromptInjection:
    """有读数才出现那一段；没读数时**整段不出现**（不留占位句让模型去猜）。"""

    BASE = dict(
        user_message="今天有点累",
        conversation_history=[],
        emotion_probs=NEUTRAL_PROBS,
        risk_level="low",
    )

    def test_no_facial_section_without_reading(self):
        msgs = iv._build_llm_messages(**self.BASE, facial_context="")
        assert "镜头里读到的表情" not in msgs[0]["content"]

    def test_facial_section_injected_with_reading(self):
        ctx = iv._build_facial_context(GOOD_STATE)
        msgs = iv._build_llm_messages(**self.BASE, facial_context=ctx)
        system = msgs[0]["content"]
        assert "镜头里读到的表情" in system
        assert "一个字都不要说给 ta 听" in system
        assert "0.17" in system

    def test_injection_keeps_message_shape(self):
        ctx = iv._build_facial_context(GOOD_STATE)
        msgs = iv._build_llm_messages(**self.BASE, facial_context=ctx)
        assert msgs[0]["role"] == "system"
        assert msgs[-1] == {"role": "user", "content": "今天有点累"}

    def test_empty_context_is_identical_to_default(self):
        """`facial_context` 缺省与显式空串必须产出完全一样的 messages。"""
        a = iv._build_llm_messages(**self.BASE)
        b = iv._build_llm_messages(**self.BASE, facial_context="")
        assert a == b


class TestFacialNeverChangesRisk:
    """**核心安全边界**：面部读数不得改变风险等级、证据链与请求契约。"""

    def test_request_contract_accepts_facial_features(self):
        req = iv.SmartChatRequest(
            user_id="u1",
            message="我没事",
            facial_features=GOOD_STATE,
        )
        assert req.facial_features == GOOD_STATE
        # 缺省必须是 None（文字聊天 / 没开摄像头时不带该字段）
        assert iv.SmartChatRequest(user_id="u1", message="我没事").facial_features is None

    def test_prepare_turn_risk_unchanged_by_sad_face(self):
        """同一句中性文本，带上"悲伤 90%"的人脸读数后风险等级必须**完全一致**。

        ⚠️ 两次调用用**不同的 user_id**：TAMER 的 RQI/CUSUM 是**按会话累积**的
        （见 `_get_tamer` 的会话级注册表），同一个 user_id 连调两次会让第二次
        变成"第二轮"，证据链自然不同 —— 那是 TAMER 的预期行为，与本测试要守的
        "面部不进风险"无关。
        """
        msg = "今天过得还行"

        without = iv._prepare_chat_turn(
            iv.SmartChatRequest(user_id="risk_guard_without", message=msg, conversation_history=[])
        )
        with_sad = iv._prepare_chat_turn(
            iv.SmartChatRequest(
                user_id="risk_guard_with", message=msg, conversation_history=[],
                facial_features=SAD_STATE,
            )
        )

        assert without.risk_level == with_sad.risk_level, (
            "面部读数改变了风险等级 —— 违反'面部不进风险'的边界"
        )
        assert without.crisis_detected == with_sad.crisis_detected
        assert without.ask_question == with_sad.ask_question
        # 有新鲜读数时，证据链里只应多出"面部实时状态"这一条（且明确写了不参与风险）
        facial_evidence = [
            e for e in with_sad.emotion_result.evidence if "面部实时状态" in e
        ]
        assert len(facial_evidence) == 1
        assert "不参与风险判定" in facial_evidence[0]
        assert not any("面部实时状态" in e for e in without.emotion_result.evidence)

    def test_prepare_turn_ignores_malformed_facial_features(self):
        """坏数据既不能注入，也不能让前处理抛错。"""
        base = dict(user_id="risk_guard_bad", message="今天过得还行", conversation_history=[])
        prep = iv._prepare_chat_turn(
            iv.SmartChatRequest(**base, facial_features={"version": 999, "present": True})
        )
        assert prep.facial_context == ""
        assert not any("面部实时状态" in e for e in prep.emotion_result.evidence)

    def test_no_facial_features_yields_empty_context(self):
        prep = iv._prepare_chat_turn(
            iv.SmartChatRequest(user_id="no_face", message="今天过得还行", conversation_history=[])
        )
        assert prep.facial_context == ""

    def test_fresh_reading_reaches_prompt_end_to_end(self, monkeypatch):
        """端到端：合法读数 → `_prepare_chat_turn` → system prompt 里出现那一段。"""
        captured: dict = {}

        class _FakeResp:
            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "嗯，我在听。今天发生了什么？"}}]}

        def fake_post(url, **kwargs):
            captured["messages"] = kwargs.get("json", {}).get("messages", [])
            return _FakeResp()

        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(iv.requests, "post", fake_post)

        req = iv.SmartChatRequest(
            user_id="e2e_face",
            message="今天过得还行",
            conversation_history=[],
            facial_features=GOOD_STATE,
        )
        prep = iv._prepare_chat_turn(req)
        iv._call_llm(
            req.message,
            req.conversation_history,
            prep.emotion_result.text_emotion_probs,
            prep.risk_level,
            memory_context=prep.memory_context,
            tamer_result=prep.tamer_result,
            facial_context=prep.facial_context,
        )
        system = captured["messages"][0]["content"]
        assert "镜头里读到的表情" in system
        assert prep.risk_level in system


class TestHttpContract:
    """走真实 FastAPI 路由的契约验证 —— 客户端字段名与"不进风险"都在这一层再确认一次。"""

    @staticmethod
    def _capture(monkeypatch):
        captured: dict = {}

        class _FakeResp:
            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "我在听。今天发生了什么？"}}]}

        def fake_post(url, **kwargs):
            captured["messages"] = kwargs.get("json", {}).get("messages", [])
            return _FakeResp()

        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
        monkeypatch.setattr(iv.requests, "post", fake_post)
        return captured

    def _post(self, payload):
        from fastapi.testclient import TestClient
        from main import app

        return TestClient(app).post("/api/v1/intervention/smart-chat", json=payload)

    def test_http_accepts_facial_features_and_injects_section(self, monkeypatch):
        captured = self._capture(monkeypatch)
        resp = self._post({
            "user_id": "http_face_with",
            "message": "今天过得还行",
            "conversation_history": [],
            "facial_features": GOOD_STATE,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert any("面部实时状态" in e for e in body["evidence"])
        assert "镜头里读到的表情" in captured["messages"][0]["content"]

    def test_http_without_facial_features_has_no_section(self, monkeypatch):
        captured = self._capture(monkeypatch)
        resp = self._post({
            "user_id": "http_face_without",
            "message": "今天过得还行",
            "conversation_history": [],
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert not any("面部实时状态" in e for e in body["evidence"])
        assert "镜头里读到的表情" not in captured["messages"][0]["content"]

    def test_http_risk_level_identical_with_and_without_face(self, monkeypatch):
        """**安全边界的 HTTP 级复验**：同一句话，带不带人脸读数，风险等级必须一致。"""
        msg = "今天过得还行"

        self._capture(monkeypatch)
        a = self._post({"user_id": "http_risk_a", "message": msg, "conversation_history": []}).json()
        b = self._post({
            "user_id": "http_risk_b", "message": msg, "conversation_history": [],
            "facial_features": SAD_STATE,
        }).json()

        assert a["risk_level"] == b["risk_level"]
        assert a["requires_escalation"] == b["requires_escalation"]
