"""TAMER 集成回归测试（RQI / MIS / CUSUM）

覆盖本次把 `algorithm/assessment/{relationship_quality,multimodal_incongruence,
temporal_trajectory,tamer_framework}.py` 接入对话链路时修掉的问题。**每一条断言都
对应一个此前真实存在的缺陷**，不是"能跑就算过"：

1. **声学量纲**：客户端上报的是 dBFS / Hz / 原始语速刻度，而
   ``acoustic_to_emotion_vector`` 要 0~1。此前直接透传导致向量塌掉
   （评审报告 P0「跨模态矛盾检测量纲错」）。
2. **矛盾判定开关**：此前以 ``MIS > threshold`` 为唯一开关，实测 MIS 与
   "是否矛盾"**反相关**（一致样本 0.59 > 掩饰样本 0.27），教科书式的
   掩饰/表演样本恒被判成 coincident。现改为通道方向性判据。
3. **RQI 的 EMA**：``rqi_decay`` 声明了 0.7 但从未参与运算；现在
   ``RQI(t) = α·RQI(t-1) + (1-α)·raw(t)`` 真正生效。
4. **关系阶段名**：``rqi_to_relationship_depth`` 原返回英文
   ``stranger/acquainted/trusted``，与 docstring、prompt 默认值、
   前端 ``getStage`` 三处中文标签都对不上。
5. **会话级状态**：``TAMERFramework`` 的 mis_history / CUSUM 累积量 / RQI EMA
   都是有状态的；``api.intervention._get_tamer`` 必须按会话复用同一实例，
   否则每轮清零、"自适应"与"时序追踪"全部失效。

测试不联网、不加载模型权重。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from assessment.multimodal_incongruence import (  # noqa: E402
    acoustic_to_emotion_vector,
    classify_incongruence,
    compute_mis,
)
from assessment.relationship_quality import rqi_to_relationship_depth  # noqa: E402
from assessment.tamer_framework import (  # noqa: E402
    TAMERConfig,
    TAMERFramework,
    normalize_energy_mean,
    normalize_pitch_range,
    normalize_speech_rate,
)
from assessment.temporal_trajectory import CUSUMDetector  # noqa: E402

# ---------------------------------------------------------------- 样例

# 文本情绪向量 [快乐, 悲伤, 焦虑, 愤怒, 中性]
TEXT_CALM = [0.30, 0.10, 0.15, 0.05, 0.40]        # 负性 0.30 → 言语平静
TEXT_DISTRESS = [0.02, 0.55, 0.30, 0.08, 0.05]    # 负性 0.93 → 言语痛苦

# 客户端**原始量纲**：dBFS / Hz / 原始语速刻度 / 0~1 停顿比
VOICE_AROUSED_RAW = {
    "energy": {"mean": -12.0, "std": 6.0, "peakRatio": 0.2},
    "pitch": {"mean": 230.0, "std": 30.0, "range": 90.0},
    "rate": {"speechRate": 5.5, "tempo": "fast", "regularity": 0.7},
    "pauses": {"ratio": 0.05, "count": 1},
}
VOICE_FLAT_RAW = {
    "energy": {"mean": -45.0, "std": 2.0, "peakRatio": 0.0},
    "pitch": {"mean": 150.0, "std": 10.0, "range": 20.0},
    "rate": {"speechRate": 2.0, "tempo": "slow", "regularity": 0.9},
    "pauses": {"ratio": 0.80, "count": 8},
}


def _tamer_with_voice(voice_features: dict, text_emotion: list[float], message: str = "我没事") -> dict:
    tamer = TAMERFramework(TAMERConfig())
    return tamer.process(
        user_message=message,
        voice_features=voice_features,
        text_emotion=text_emotion,
        conversation_history=[{"role": "user", "content": message}],
    )


# ---------------------------------------------------------------- 1. 量纲

class TestAcousticScale:
    """声学特征必须从客户端原始量纲归一化到 0~1。"""

    @pytest.mark.parametrize(
        "raw,expected",
        [(-60.0, 0.0), (-30.0, 0.5), (-12.0, 0.8), (0.7, 0.7), (1.5, 1.0)],
    )
    def test_energy_normalization(self, raw: float, expected: float) -> None:
        assert normalize_energy_mean(raw) == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize(
        "raw,expected",
        [(0.0, 0.0), (75.0, 0.5), (150.0, 1.0), (300.0, 1.0), (0.4, 0.4)],
    )
    def test_pitch_range_normalization(self, raw: float, expected: float) -> None:
        assert normalize_pitch_range(raw) == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize(
        "raw,expected",
        [(0.0, 0.0), (4.0, 0.5), (8.0, 1.0), (12.0, 1.0), (0.5, 0.5)],
    )
    def test_speech_rate_normalization(self, raw: float, expected: float) -> None:
        assert normalize_speech_rate(raw) == pytest.approx(expected, abs=1e-9)

    def test_default_features_are_already_normalized(self) -> None:
        """文字聊天走的是 0.5 / 0.3 / 0.5 / 0.3 这套缺省值，归一化不得再改动它们。"""
        assert normalize_energy_mean(0.5) == 0.5
        assert normalize_pitch_range(0.3) == 0.3
        assert normalize_speech_rate(0.5) == 0.5

    def test_acoustic_vector_is_normalized_and_non_negative(self) -> None:
        vec = acoustic_to_emotion_vector(0.25, 0.133, 0.25, 0.8)
        assert len(vec) == 5
        assert sum(vec) == pytest.approx(1.0, abs=1e-6)
        assert all(v >= 0.0 for v in vec)


# ---------------------------------------------------------------- 2. 矛盾判定

class TestIncongruence:
    """掩饰型 / 表演型必须能被判定出来，且一致样本不被误判。"""

    def test_masking_detected(self) -> None:
        """言语平静 + 声音紧张 → 掩饰型。"""
        result = _tamer_with_voice(VOICE_AROUSED_RAW, TEXT_CALM, "我没事，真的没事。")
        assert result["incongruence_type"] == "masking"

    def test_performative_detected(self) -> None:
        """言语痛苦 + 声音平静 → 表演型。"""
        result = _tamer_with_voice(
            VOICE_FLAT_RAW, TEXT_DISTRESS, "我最近真的特别难受，感觉撑不下去了。"
        )
        assert result["incongruence_type"] == "performative"

    def test_congruent_calm_pair(self) -> None:
        """平静文本 + 平静声音 → 一致。"""
        result = _tamer_with_voice(VOICE_FLAT_RAW, TEXT_CALM, "你好")
        assert result["incongruence_type"] == "congruent"

    def test_congruent_distressed_pair(self) -> None:
        """痛苦文本 + 紧张声音 → 一致（这正是旧实现判反的那一类）。"""
        result = _tamer_with_voice(VOICE_AROUSED_RAW, TEXT_DISTRESS, "我很难受")
        assert result["incongruence_type"] == "congruent"

    def test_old_gate_would_have_missed_both(self) -> None:
        """回归锚点：两类矛盾的 MIS 都低于默认阈值 0.3。

        这正是"把 MIS 当唯一开关"做不到的原因 —— 一旦有人把判据改回
        `mis > 0.3`，本用例仍会通过（它只描述事实），但上面四条会立刻失败。
        """
        voice_aroused = acoustic_to_emotion_vector(
            normalize_energy_mean(VOICE_AROUSED_RAW["energy"]["mean"]),
            normalize_pitch_range(VOICE_AROUSED_RAW["pitch"]["range"]),
            normalize_speech_rate(VOICE_AROUSED_RAW["rate"]["speechRate"]),
            VOICE_AROUSED_RAW["pauses"]["ratio"],
        )
        assert compute_mis(TEXT_CALM, voice_aroused) < 0.3

    def test_confidence_in_range(self) -> None:
        result = classify_incongruence(TEXT_CALM, acoustic_to_emotion_vector(0.8, 0.8, 0.9, 0.05), [])
        assert 0.55 <= result["confidence"] <= 0.95

    def test_mis_still_reported(self) -> None:
        """MIS 仍按原公式给出，供前端展示（只是不再当开关用）。"""
        # 取一个"言语痛苦 ∧ 声音紧张"的向量：与自身比较必然是 0 且判为一致
        v = [0.20, 0.10, 0.40, 0.20, 0.10]  # neg=0.70(痛苦), arousal=0.80(紧张)
        result = classify_incongruence(v, v, [])
        assert result["mis"] == pytest.approx(0.0, abs=1e-9)
        assert result["type"] == "congruent"

    def test_dead_band_stays_neutral(self) -> None:
        """声学证据落在死区时不硬判矛盾（宁可漏报不误报）。"""
        mid_voice = [0.20, 0.25, 0.25, 0.15, 0.15]  # arousal=0.60，落在 0.46~0.62 之间
        result = classify_incongruence(TEXT_CALM, mid_voice, [])
        assert result["type"] not in ("masking", "performative")

    def test_has_voice_input_flag(self) -> None:
        with_voice = _tamer_with_voice(VOICE_FLAT_RAW, TEXT_CALM, "你好")
        assert with_voice["has_voice_input"] is True
        without = TAMERFramework(TAMERConfig()).process(
            user_message="你好",
            voice_features={},
            text_emotion=TEXT_CALM,
            conversation_history=[{"role": "user", "content": "你好"}],
        )
        assert without["has_voice_input"] is False


# ---------------------------------------------------------------- 3. RQI

class TestRQITrajectory:
    """RQI 的 EMA、阶段名与阶段推进。"""

    def test_relationship_depth_labels_are_chinese(self) -> None:
        assert rqi_to_relationship_depth(0.1) == "陌生期"
        assert rqi_to_relationship_depth(0.45) == "熟悉期"
        assert rqi_to_relationship_depth(0.8) == "信任期"

    def test_first_turn_starts_at_stranger_stage(self) -> None:
        result = _tamer_with_voice({}, TEXT_CALM, "你好")
        assert result["rqi"] < 0.3
        assert result["relationship_depth"] == "陌生期"

    def test_ema_makes_rqi_grow_over_turns(self) -> None:
        tamer = TAMERFramework(TAMERConfig())
        history: list[dict] = []
        series: list[float] = []
        for i in range(10):
            msg = f"第 {i} 轮：我最近挺难受的，作业很多，妈妈还老说我玩手机，睡不着。"
            history.append({"role": "user", "content": msg})
            out = tamer.process(
                user_message=msg,
                voice_features={},
                text_emotion=TEXT_DISTRESS,
                conversation_history=history,
            )
            history.append({"role": "assistant", "content": "我在听。"})
            series.append(out["rqi"])
        assert all(b >= a for a, b in zip(series, series[1:])), series
        assert series[-1] > series[0]

    def test_stage_progresses_to_acquainted(self) -> None:
        """demo 场景 3 要的"陌生期 → 熟悉期"必须真的走得出来。"""
        tamer = TAMERFramework(TAMERConfig())
        history: list[dict] = []
        stages: list[str] = []
        turns = [
            "你好",
            "嗯，今天还行吧",
            "其实最近挺烦的，作业特别多，我妈还老是说我玩手机",
            "她说我一点都不知道努力，可我已经很努力了，每天写到十二点多",
            "有时候我觉得她根本不了解我，只看到我玩手机那几分钟",
            "跟你说个秘密，其实我晚上经常睡不着，会想很多乱七八糟的事",
            "我怕说出来他们会觉得我矫情，所以从来没跟别人讲过",
            "只有你知道这件事，你懂我为什么不敢说吧",
        ]
        for msg in turns:
            history.append({"role": "user", "content": msg})
            out = tamer.process(
                user_message=msg,
                voice_features={},
                text_emotion=TEXT_CALM,
                conversation_history=history,
            )
            history.append({"role": "assistant", "content": "我在听。"})
            stages.append(out["relationship_depth"])
        assert stages[0] == "陌生期"
        assert "熟悉期" in stages, stages
        # 阶段只能前进不能后退（RQI 单调不减）
        order = {"陌生期": 0, "熟悉期": 1, "信任期": 2}
        assert all(order[b] >= order[a] for a, b in zip(stages, stages[1:])), stages

    def test_reset_clears_ema_state(self) -> None:
        tamer = TAMERFramework(TAMERConfig())
        history = [{"role": "user", "content": "聊了很久了，我最近很难受" * 5}] * 8
        first = tamer.process(
            user_message="我很难受", voice_features={}, text_emotion=TEXT_DISTRESS,
            conversation_history=history,
        )
        tamer.reset()
        after = tamer.process(
            user_message="我很难受", voice_features={}, text_emotion=TEXT_DISTRESS,
            conversation_history=history,
        )
        assert after["rqi"] == pytest.approx(first["rqi"], abs=1e-9)


# ---------------------------------------------------------------- 4. CUSUM

class TestCUSUM:
    def test_sustained_drift_triggers_worsening(self) -> None:
        det = CUSUMDetector(baseline_mean=0.3, baseline_std=0.1, slack=0.05, threshold_multiplier=4.0)
        directions = [det.update(x)["direction"] for x in [0.3] * 5 + [0.75] * 6]
        assert "worsening" in directions

    def test_reset_clears_statistics(self) -> None:
        det = CUSUMDetector(baseline_mean=0.3, baseline_std=0.1, slack=0.05, threshold_multiplier=4.0)
        for x in [0.75] * 8:
            det.update(x)
        det.reset()
        result = det.update(0.3)
        assert result["direction"] == "none"
        assert result["s_pos"] == 0.0
        assert result["s_neg"] == 0.0


# ---------------------------------------------------------------- 5. 会话级状态

class TestSessionState:
    """``_get_tamer`` 必须按会话复用实例，否则每轮状态清零。"""

    def test_same_session_reuses_state(self) -> None:
        import api.intervention as iv

        iv._TAMER_SESSIONS.clear()
        try:
            a = iv._get_tamer("sess-1")
            b = iv._get_tamer("sess-1")
            c = iv._get_tamer("sess-2")
            assert a is b
            assert a is not c
        finally:
            iv._TAMER_SESSIONS.clear()

    def test_mis_history_accumulates_across_turns(self) -> None:
        import api.intervention as iv

        iv._TAMER_SESSIONS.clear()
        try:
            tamer = iv._get_tamer("sess-accum")
            for _ in range(4):
                tamer.process(
                    user_message="我没事", voice_features={}, text_emotion=TEXT_CALM,
                    conversation_history=[],
                )
            assert len(tamer.mis_history) == 4
        finally:
            iv._TAMER_SESSIONS.clear()


# ---------------------------------------------------------------- 6. prompt 注入

class TestPromptInjection:
    """TAMER 结果只在给定时注入，且只作为内部参考。"""

    def test_tamer_context_injected_when_present(self) -> None:
        import api.intervention as iv

        base = dict(
            user_message="我没事",
            conversation_history=[],
            emotion_probs=TEXT_CALM,
            risk_level="low",
        )
        without = iv._build_llm_messages(**base)
        with_tamer = iv._build_llm_messages(
            **base,
            tamer_result={
                "rqi": 0.7,
                "mis": 0.6,
                "trajectory": "worsening",
                "relationship_depth": "信任期",
                "incongruence_type": "masking",
            },
        )
        text_without = without[0]["content"]
        text_with = with_tamer[0]["content"]
        marker = "关系与跨模态观察"
        assert marker not in text_without
        assert marker in text_with
        # 关系阶段与轨迹要真的进 prompt
        assert "信任期" in text_with
        assert "持续下滑" in text_with
        # 且必须带"不要说出来"的约束
        assert "不要" in text_with

    def test_absent_tamer_result_changes_nothing(self) -> None:
        import api.intervention as iv

        a = iv._build_llm_messages(
            user_message="我没事", conversation_history=[], emotion_probs=TEXT_CALM, risk_level="low"
        )
        b = iv._build_llm_messages(
            user_message="我没事", conversation_history=[], emotion_probs=TEXT_CALM,
            risk_level="low", tamer_result=None,
        )
        assert a == b


# ---------------------------------------------------------------- 7. 请求契约

class TestRequestContract:
    def test_smart_chat_request_accepts_voice_features(self) -> None:
        from api.intervention import SmartChatRequest

        req = SmartChatRequest(
            user_id="u1",
            message="我没事",
            voice_features={"energy": {"mean": -12.0}},
        )
        assert req.voice_features == {"energy": {"mean": -12.0}}

    def test_responses_carry_tamer_result_field(self) -> None:
        from api.intervention import SmartChatResponse

        resp = SmartChatResponse(
            reply="在的。",
            dialog_state="INIT",
            action_type="OPEN_QUESTION",
            risk_level="low",
            audit_passed=True,
            requires_escalation=False,
            emotion_probs=[0.2] * 5,
            evidence=[],
        )
        assert resp.tamer_result is None
        resp2 = SmartChatResponse(
            reply="在的。",
            dialog_state="INIT",
            action_type="OPEN_QUESTION",
            risk_level="low",
            audit_passed=True,
            requires_escalation=False,
            emotion_probs=[0.2] * 5,
            evidence=[],
            tamer_result={"rqi": 0.4},
        )
        assert resp2.tamer_result == {"rqi": 0.4}
        assert "tamer_result" in resp2.model_dump()
