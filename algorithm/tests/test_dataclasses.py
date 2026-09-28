"""
dataclasses 模块单元测试

验证所有数据类的实例化、字段类型和约束。
"""
import time
import unittest
from dataclasses import fields

from shared.dataclasses import (
    AsrEngine,
    AsrEventType,
    AsrResult,
    ChatStreamEvent,
    ChatStreamEventType,
    CrisisAlert,
    EmotionResult,
    EvidenceItem,
    FeatureSummary,
    RiskAssessment,
    RiskLevel,
    StreamingAsrUpdate,
    TtsEngine,
    TtsSynthesisResult,
)


class TestRiskLevel(unittest.TestCase):
    """RiskLevel 枚举测试"""

    def test_enum_values(self):
        """枚举值必须为 low / medium / high / crisis"""
        self.assertEqual(RiskLevel.LOW, "low")
        self.assertEqual(RiskLevel.MEDIUM, "medium")
        self.assertEqual(RiskLevel.HIGH, "high")
        self.assertEqual(RiskLevel.CRISIS, "crisis")

    def test_enum_count(self):
        """枚举成员数量必须为 4"""
        self.assertEqual(len(RiskLevel), 4)

    def test_enum_is_str(self):
        """枚举值必须是 str 子类"""
        for member in RiskLevel:
            self.assertIsInstance(member, str)


class TestEvidenceItem(unittest.TestCase):
    """EvidenceItem 数据类测试"""

    def setUp(self):
        self.item = EvidenceItem(
            source="perception.text",
            description="检测到焦虑关键词",
            weight=0.85,
        )

    def test_field_types(self):
        """字段类型必须为 str, str, float"""
        self.assertIsInstance(self.item.source, str)
        self.assertIsInstance(self.item.description, str)
        self.assertIsInstance(self.item.weight, float)

    def test_field_count(self):
        """字段数量必须为 3"""
        self.assertEqual(len(fields(EvidenceItem)), 3)


class TestEmotionResult(unittest.TestCase):
    """EmotionResult 数据类测试"""

    def setUp(self):
        self.result = EmotionResult(
            text_emotion_probs=[0.1, 0.3, 0.4, 0.1, 0.1],
            audio_risk_prob=0.6,
            confidence=0.88,
            timestamp=time.time(),
            evidence=["文本含焦虑关键词", "语速偏快"],
        )

    def test_text_emotion_probs_length(self):
        """text_emotion_probs 必须为 5 维"""
        self.assertEqual(len(self.result.text_emotion_probs), 5)

    def test_text_emotion_probs_type(self):
        """text_emotion_probs 元素必须为 float"""
        for p in self.result.text_emotion_probs:
            self.assertIsInstance(p, float)

    def test_audio_risk_prob_optional(self):
        """audio_risk_prob 允许为 None"""
        result_no_audio = EmotionResult(
            text_emotion_probs=[0.2, 0.2, 0.2, 0.2, 0.2],
            audio_risk_prob=None,
            confidence=0.7,
            timestamp=time.time(),
            evidence=[],
        )
        self.assertIsNone(result_no_audio.audio_risk_prob)

    def test_audio_risk_prob_type(self):
        """audio_risk_prob 必须为 float 或 None"""
        self.assertIsInstance(self.result.audio_risk_prob, (float, type(None)))

    def test_confidence_type(self):
        """confidence 必须为 float"""
        self.assertIsInstance(self.result.confidence, float)

    def test_timestamp_type(self):
        """timestamp 必须为 float"""
        self.assertIsInstance(self.result.timestamp, float)

    def test_evidence_type(self):
        """evidence 必须为 list[str]"""
        self.assertIsInstance(self.result.evidence, list)
        for e in self.result.evidence:
            self.assertIsInstance(e, str)

    def test_field_count(self):
        """字段数量必须为 6（含安全字段 crisis_keywords）"""
        self.assertEqual(len(fields(EmotionResult)), 6)

    # ---- crisis_keywords：安全相关字段 ----

    def test_crisis_keywords_defaults_to_empty_list(self):
        """crisis_keywords 可省略，默认为空列表（向后兼容）"""
        result = EmotionResult(
            text_emotion_probs=[0.2, 0.2, 0.2, 0.2, 0.2],
            audio_risk_prob=None,
            confidence=0.5,
            timestamp=time.time(),
            evidence=[],
        )
        self.assertEqual(result.crisis_keywords, [])

    def test_crisis_keywords_not_shared_between_instances(self):
        """默认值必须是每个实例独立的列表，不能共享同一个可变对象"""
        a = EmotionResult([0.2] * 5, None, 0.5, time.time(), [])
        b = EmotionResult([0.2] * 5, None, 0.5, time.time(), [])
        a.crisis_keywords.append("自杀")
        self.assertEqual(b.crisis_keywords, [])

    def test_crisis_keywords_accepts_hits(self):
        """命中危机词时可结构化携带，供 escalation 层触发人工复核"""
        result = EmotionResult(
            text_emotion_probs=[0.1, 0.4, 0.3, 0.1, 0.1],
            audio_risk_prob=None,
            confidence=0.9,
            timestamp=time.time(),
            evidence=["[危机筛查] 命中 1 个危机关键词"],
            crisis_keywords=["自杀"],
        )
        self.assertEqual(result.crisis_keywords, ["自杀"])


class TestRiskAssessment(unittest.TestCase):
    """RiskAssessment 数据类测试"""

    def setUp(self):
        self.assessment = RiskAssessment(
            phq9_estimated=(10.0, 15.0),
            gad7_estimated=(8.0, 12.0),
            risk_level=RiskLevel.MEDIUM,
            confidence=0.82,
            evidence=[
                EvidenceItem("assessment.scale", "PHQ-9 预估中高分", 0.9),
                EvidenceItem("perception.behavior", "近期活跃度下降", 0.6),
            ],
        )

    def test_phq9_estimated_type(self):
        """phq9_estimated 必须为 tuple[float, float]"""
        self.assertIsInstance(self.assessment.phq9_estimated, tuple)
        self.assertEqual(len(self.assessment.phq9_estimated), 2)
        for v in self.assessment.phq9_estimated:
            self.assertIsInstance(v, float)

    def test_gad7_estimated_type(self):
        """gad7_estimated 必须为 tuple[float, float]"""
        self.assertIsInstance(self.assessment.gad7_estimated, tuple)
        self.assertEqual(len(self.assessment.gad7_estimated), 2)

    def test_risk_level_enum(self):
        """risk_level 必须为 RiskLevel 枚举"""
        self.assertIsInstance(self.assessment.risk_level, RiskLevel)

    def test_risk_level_all_values(self):
        """risk_level 支持所有枚举值"""
        for level in RiskLevel:
            a = RiskAssessment(
                phq9_estimated=(0.0, 5.0),
                gad7_estimated=(0.0, 5.0),
                risk_level=level,
                confidence=0.5,
                evidence=[],
            )
            self.assertEqual(a.risk_level, level)

    def test_confidence_type(self):
        """confidence 必须为 float"""
        self.assertIsInstance(self.assessment.confidence, float)

    def test_evidence_type(self):
        """evidence 必须为 list[EvidenceItem]"""
        self.assertIsInstance(self.assessment.evidence, list)
        for e in self.assessment.evidence:
            self.assertIsInstance(e, EvidenceItem)

    def test_field_count(self):
        """字段数量必须为 5"""
        self.assertEqual(len(fields(RiskAssessment)), 5)


class TestCrisisAlert(unittest.TestCase):
    """CrisisAlert 数据类测试"""

    def setUp(self):
        self.alert = CrisisAlert(
            user_id="user_abc123",
            risk_score=0.92,
            trigger_evidence=["检测到自伤关键词", "情绪急剧恶化"],
            timestamp=time.time(),
            recommended_action="立即通知监护人并推送危机热线",
        )

    def test_user_id_type(self):
        """user_id 必须为 str"""
        self.assertIsInstance(self.alert.user_id, str)

    def test_risk_score_type(self):
        """risk_score 必须为 float"""
        self.assertIsInstance(self.alert.risk_score, float)

    def test_trigger_evidence_type(self):
        """trigger_evidence 必须为 list[str]"""
        self.assertIsInstance(self.alert.trigger_evidence, list)
        for e in self.alert.trigger_evidence:
            self.assertIsInstance(e, str)

    def test_timestamp_type(self):
        """timestamp 必须为 float"""
        self.assertIsInstance(self.alert.timestamp, float)

    def test_recommended_action_type(self):
        """recommended_action 必须为 str"""
        self.assertIsInstance(self.alert.recommended_action, str)

    def test_field_count(self):
        """字段数量必须为 5"""
        self.assertEqual(len(fields(CrisisAlert)), 5)


class TestFeatureSummary(unittest.TestCase):
    """FeatureSummary 数据类测试

    注意：此数据类确保不包含原始文本或可识别信息。
    """

    def setUp(self):
        self.summary = FeatureSummary(
            text_emotion_probs=[0.15, 0.25, 0.35, 0.10, 0.15],
            audio_risk_prob=None,
            confidence=0.91,
            session_id="session_xyz789",
            timestamp=time.time(),
        )

    def test_text_emotion_probs_type(self):
        """text_emotion_probs 必须为 list[float]"""
        self.assertIsInstance(self.summary.text_emotion_probs, list)
        for p in self.summary.text_emotion_probs:
            self.assertIsInstance(p, float)

    def test_audio_risk_prob_optional(self):
        """audio_risk_prob 允许为 None"""
        self.assertIsNone(self.summary.audio_risk_prob)

    def test_audio_risk_prob_with_value(self):
        """audio_risk_prob 也可为 float"""
        s = FeatureSummary(
            text_emotion_probs=[0.2, 0.2, 0.2, 0.2, 0.2],
            audio_risk_prob=0.45,
            confidence=0.8,
            session_id="s1",
            timestamp=time.time(),
        )
        self.assertIsInstance(s.audio_risk_prob, float)

    def test_confidence_type(self):
        """confidence 必须为 float"""
        self.assertIsInstance(self.summary.confidence, float)

    def test_session_id_type(self):
        """session_id 必须为 str"""
        self.assertIsInstance(self.summary.session_id, str)

    def test_timestamp_type(self):
        """timestamp 必须为 float"""
        self.assertIsInstance(self.summary.timestamp, float)

    def test_no_raw_text_fields(self):
        """确保数据类不包含原始文本或可识别信息字段"""
        field_names = {f.name for f in fields(FeatureSummary)}
        # 不应包含任何可能携带原始信息的字段
        sensitive_names = {"text", "content", "message", "user_name", "nickname"}
        self.assertTrue(field_names.isdisjoint(sensitive_names))

    def test_field_count(self):
        """字段数量必须为 5"""
        self.assertEqual(len(fields(FeatureSummary)), 5)


class TestAsrResult(unittest.TestCase):
    """AsrResult / AsrEngine 数据类测试（本地语音转文字的跨模块接口）"""

    def test_engine_enum_values(self):
        """引擎枚举必须为 paraformer / unavailable"""
        self.assertEqual(AsrEngine.PARAFORMER, "sherpa-onnx-paraformer")
        self.assertEqual(AsrEngine.UNAVAILABLE, "unavailable")

    def test_engine_has_no_browser_option(self):
        """不得提供浏览器 Web Speech API 选项（那条路径会把音频送到厂商云）"""
        for member in AsrEngine:
            self.assertNotIn("web", member.value.lower())
            self.assertNotIn("speech-api", member.value.lower())

    def test_engine_is_str(self):
        """枚举值必须是 str 子类"""
        for member in AsrEngine:
            self.assertIsInstance(member, str)

    def test_text_type(self):
        """text 必须为 str"""
        self.assertIsInstance(AsrResult(text="你好").text, str)

    def test_defaults(self):
        """默认值：引擎 paraformer、时长为 0、无错误"""
        r = AsrResult(text="你好")
        self.assertIs(r.engine, AsrEngine.PARAFORMER)
        self.assertEqual(r.duration_ms, 0.0)
        self.assertEqual(r.latency_ms, 0.0)
        self.assertEqual(r.error, "")

    def test_field_count(self):
        """字段数量必须为 5"""
        self.assertEqual(len(fields(AsrResult)), 5)

    def test_is_empty(self):
        """空白文本视为空"""
        self.assertTrue(AsrResult(text="").is_empty)
        self.assertTrue(AsrResult(text="   ").is_empty)
        self.assertFalse(AsrResult(text="你好").is_empty)

    def test_ok_true_on_success(self):
        """正常识别（哪怕没识别出文本）也算成功"""
        self.assertTrue(AsrResult(text="你好").ok)
        self.assertTrue(AsrResult(text="").ok)

    def test_ok_false_on_error(self):
        """带错误信息时不算成功"""
        self.assertFalse(AsrResult(text="", error="模型缺失").ok)

    def test_ok_false_when_unavailable(self):
        """引擎不可用时不算成功"""
        self.assertFalse(AsrResult(text="", engine=AsrEngine.UNAVAILABLE).ok)

    def test_result_is_constructible_positionally(self):
        """必须支持位置参数构造（与其它数据类保持一致）"""
        r = AsrResult("识别文本", AsrEngine.PARAFORMER, 1234.0, 45.6, "")
        self.assertEqual(r.text, "识别文本")
        self.assertEqual(r.duration_ms, 1234.0)


class TestStreamingAsrUpdate(unittest.TestCase):
    """StreamingAsrUpdate / AsrEventType 测试（流式识别的跨模块接口）"""

    def test_event_type_values(self):
        """事件类型必须为 partial / final"""
        self.assertEqual(AsrEventType.PARTIAL, "partial")
        self.assertEqual(AsrEventType.FINAL, "final")

    def test_event_type_count(self):
        """事件类型只有两种"""
        self.assertEqual(len(AsrEventType), 2)

    def test_event_type_is_str(self):
        """枚举值必须是 str 子类"""
        for member in AsrEventType:
            self.assertIsInstance(member, str)

    def test_defaults(self):
        """默认：流式 zipformer 引擎、时长与耗时均为 0、无 reason"""
        u = StreamingAsrUpdate(event=AsrEventType.PARTIAL, text="你好")
        self.assertIs(u.engine, AsrEngine.ZIPFORMER_STREAMING)
        self.assertEqual(u.elapsed_ms, 0.0)
        self.assertEqual(u.latency_ms, 0.0)
        self.assertEqual(u.reason, "")

    def test_field_count(self):
        """字段数量必须为 7（含 refined 纠错标记）"""
        self.assertEqual(len(fields(StreamingAsrUpdate)), 7)

    def test_refined_defaults_false(self):
        """默认未纠错"""
        self.assertFalse(StreamingAsrUpdate(AsrEventType.FINAL, "你好").refined)

    def test_refined_flag(self):
        """定稿若经离线大模型纠错，必须能标记出来且引擎随之改变"""
        u = StreamingAsrUpdate(
            AsrEventType.FINAL,
            "你好。",
            reason="endpoint",
            engine=AsrEngine.PARAFORMER,
            refined=True,
        )
        self.assertTrue(u.refined)
        self.assertIs(u.engine, AsrEngine.PARAFORMER)

    def test_is_final(self):
        """is_final 只在 FINAL 事件为真"""
        self.assertFalse(StreamingAsrUpdate(AsrEventType.PARTIAL, "你").is_final)
        self.assertTrue(StreamingAsrUpdate(AsrEventType.FINAL, "你").is_final)

    def test_is_empty(self):
        """空白文本视为空"""
        self.assertTrue(StreamingAsrUpdate(AsrEventType.PARTIAL, "").is_empty)
        self.assertTrue(StreamingAsrUpdate(AsrEventType.PARTIAL, "  ").is_empty)
        self.assertFalse(StreamingAsrUpdate(AsrEventType.PARTIAL, "你").is_empty)

    def test_final_reason(self):
        """定稿必须能区分是端点检测还是用户主动结束"""
        self.assertEqual(
            StreamingAsrUpdate(AsrEventType.FINAL, "你", reason="endpoint").reason, "endpoint"
        )
        self.assertEqual(
            StreamingAsrUpdate(AsrEventType.FINAL, "你", reason="finalize").reason, "finalize"
        )

    def test_engine_has_streaming_value(self):
        """引擎枚举必须含流式 zipformer，且仍不含任何浏览器方案"""
        self.assertEqual(AsrEngine.ZIPFORMER_STREAMING, "sherpa-onnx-streaming-zipformer")
        for member in AsrEngine:
            self.assertNotIn("web", member.value.lower())


class TestChatStreamEventType(unittest.TestCase):
    """ChatStreamEventType 枚举测试 —— 跨语言契约，取值不得随意改动"""

    def test_enum_values(self):
        """六个事件取值必须与 Node 中转层、浏览器消费端一致"""
        self.assertEqual(ChatStreamEventType.META, "meta")
        self.assertEqual(ChatStreamEventType.SPEAK, "speak")
        self.assertEqual(ChatStreamEventType.DELTA, "delta")
        self.assertEqual(ChatStreamEventType.REVISE, "revise")
        self.assertEqual(ChatStreamEventType.DONE, "done")
        self.assertEqual(ChatStreamEventType.ERROR, "error")

    def test_exactly_six_members(self):
        """事件类型数量固定；新增事件必须同步改三端，这里刻意做成会失败的护栏

        2026-09-27 由 5 → 6：新增 ``SPEAK``（定稿合成计划）。
        同步位置（缺一处就会静默丢事件）：
            · ``algorithm/api/intervention.py``    产出端
            · ``server/src/services/algorithm-bridge.ts``  白名单 + 类型
            · ``client/src/services/chatStream.ts``        解析与回调
        """
        self.assertEqual(len(list(ChatStreamEventType)), 6)

    def test_is_str_enum(self):
        """必须能被 json.dumps 直接序列化（SSE 载荷依赖这一点）"""
        self.assertIsInstance(ChatStreamEventType.DELTA, str)
        self.assertEqual(ChatStreamEventType.DELTA, "delta")


class TestChatStreamEvent(unittest.TestCase):
    """ChatStreamEvent 数据类测试"""

    def test_defaults_are_inert(self):
        """默认值必须是「什么都没说」的中性态，不能默认已通过审计"""
        e = ChatStreamEvent(ChatStreamEventType.META)
        self.assertEqual(e.text, "")
        self.assertEqual(e.risk_level, "")
        self.assertEqual(e.emotion_probs, [])
        self.assertFalse(e.streaming)
        self.assertTrue(e.audit_passed)
        self.assertFalse(e.requires_escalation)
        self.assertFalse(e.fallback)
        self.assertEqual(e.error, "")

    def test_is_terminal(self):
        """只有 DONE / ERROR 是终结事件"""
        self.assertTrue(ChatStreamEvent(ChatStreamEventType.DONE).is_terminal)
        self.assertTrue(ChatStreamEvent(ChatStreamEventType.ERROR).is_terminal)
        for t in (
            ChatStreamEventType.META,
            ChatStreamEventType.DELTA,
            ChatStreamEventType.REVISE,
        ):
            self.assertFalse(ChatStreamEvent(t).is_terminal, t)

    def test_replaces_bubble_only_for_revise(self):
        """只有 REVISE 是整体替换语义；DELTA 必须是追加语义。

        这条是流式路径安全性的关键：把权威改稿当增量拼上去，会得到
        「不安全原文 + 安全改稿」这种最坏结果。
        """
        self.assertTrue(ChatStreamEvent(ChatStreamEventType.REVISE, "改稿").replaces_bubble)
        for t in (
            ChatStreamEventType.META,
            ChatStreamEventType.DELTA,
            ChatStreamEventType.DONE,
            ChatStreamEventType.ERROR,
        ):
            self.assertFalse(ChatStreamEvent(t).replaces_bubble, t)

    def test_emotion_probs_not_shared_between_instances(self):
        """可变默认值不得在实例间共享"""
        a = ChatStreamEvent(ChatStreamEventType.META)
        b = ChatStreamEvent(ChatStreamEventType.META)
        a.emotion_probs.append(0.5)
        self.assertEqual(b.emotion_probs, [])

    def test_done_payload_carries_audit_verdict(self):
        """DONE 必须能带上与非流式端点同构的审计结论"""
        e = ChatStreamEvent(
            ChatStreamEventType.DONE,
            text="我在这里。",
            risk_level="high",
            dialogue_mode="EMPATHY",
            emotion_probs=[0.1, 0.6, 0.2, 0.05, 0.05],
            audit_passed=False,
            requires_escalation=True,
        )
        self.assertEqual(e.text, "我在这里。")
        self.assertFalse(e.audit_passed)
        self.assertTrue(e.requires_escalation)

    def test_field_names_are_the_wire_contract(self):
        """字段名会被序列化进 SSE 载荷，改名等于破坏协议"""
        names = {f.name for f in fields(ChatStreamEvent)}
        self.assertEqual(
            names,
            {
                "event",
                "text",
                "risk_level",
                "dialogue_mode",
                "emotion_probs",
                "streaming",
                "audit_passed",
                "requires_escalation",
                "fallback",
                "error",
            },
        )


class TestTtsEngine(unittest.TestCase):
    """TtsEngine 枚举测试"""

    def test_enum_values(self):
        """枚举值必须为 sherpa-onnx-vits / qwen3-tts / cosyvoice3 / unavailable"""
        self.assertEqual(TtsEngine.VITS, "sherpa-onnx-vits")
        self.assertEqual(TtsEngine.QWEN3, "qwen3-tts")
        self.assertEqual(TtsEngine.COSYVOICE, "cosyvoice3")
        self.assertEqual(TtsEngine.UNAVAILABLE, "unavailable")

    def test_enum_is_str(self):
        """枚举值必须是 str 子类（要直接进 JSON）"""
        for member in TtsEngine:
            self.assertIsInstance(member, str)

    def test_no_browser_speech_synthesis_option(self):
        """刻意不提供浏览器 speechSynthesis

        它的音频不经过 Web Audio，拿不到 AudioNode，因此既不能作为回声消除的
        参考信号，也做不了文本级自回声过滤（docs/voice_call_plan.md §6.3）。
        这条断言是防止有人"顺手加一个省事的引擎"。
        """
        # 3 个真引擎（vits / qwen3 / cosyvoice）+ 1 个 UNAVAILABLE
        self.assertEqual(len(TtsEngine), 4)
        self.assertNotIn("speech-synthesis", {e.value for e in TtsEngine})


class TestTtsSynthesisResult(unittest.TestCase):
    """TtsSynthesisResult 数据类测试"""

    def test_field_names_are_the_wire_contract(self):
        """字段名会进 WS 的 end 帧，改名等于破坏协议"""
        names = {f.name for f in fields(TtsSynthesisResult)}
        self.assertEqual(
            names,
            {
                "seq",
                "engine",
                "spoken_text",
                "sample_rate",
                "audio_ms",
                "first_chunk_ms",
                "synth_ms",
                "cancelled",
                "error",
            },
        )

    def test_defaults(self):
        """默认值必须是"未合成"的诚实状态，不能假装成功"""
        r = TtsSynthesisResult(seq=1)
        self.assertEqual(r.engine, TtsEngine.VITS)
        self.assertEqual(r.audio_ms, 0.0)
        self.assertEqual(r.error, "")
        self.assertFalse(r.cancelled)

    def test_rtf(self):
        """RTF = 合成耗时 / 音频时长"""
        r = TtsSynthesisResult(seq=1, audio_ms=4000.0, synth_ms=600.0)
        self.assertAlmostEqual(r.rtf, 0.15, places=6)

    def test_rtf_zero_audio_is_zero_not_inf(self):
        """没产出音频时 RTF 必须返回 0.0（表示"无从判断"），不能除零

        不能返回 inf 或抛异常：调用方会拿它跟 0.4 比较，inf 会被误判成"很慢"，
        而真实语义是"这一句根本没合成出东西"。
        """
        self.assertEqual(TtsSynthesisResult(seq=1, audio_ms=0.0, synth_ms=100.0).rtf, 0.0)

    def test_vits_callback_fires_once_so_first_chunk_equals_synth(self):
        """实测：VITS 每句只回调一次，所以 first_chunk_ms 与 synth_ms 相等

        这条断言把 docs/tts_benchmark.md §1 发现一钉在类型层：
        任何人若开始假设"首块比整句快"，都必须先推翻这条测试。
        """
        r = TtsSynthesisResult(seq=1, audio_ms=4000.0, first_chunk_ms=443.4, synth_ms=443.4)
        self.assertEqual(r.first_chunk_ms, r.synth_ms)

    def test_ok_semantics(self):
        """可用引擎 + 无错误 = 成功；被取消也算成功（取消是正常路径，不是故障）"""
        self.assertTrue(TtsSynthesisResult(seq=1).ok)
        self.assertTrue(TtsSynthesisResult(seq=1, cancelled=True).ok)
        self.assertFalse(TtsSynthesisResult(seq=1, error="模型缺失").ok)
        self.assertFalse(TtsSynthesisResult(seq=1, engine=TtsEngine.UNAVAILABLE).ok)

    def test_is_silent_exposes_the_truncation_failure_mode(self):
        """`is_silent` 存在的意义：让"合成成功但没声音"能被上游发现

        这正是 max_num_sentences=1 那类静默失效的形状 —— 引擎正常返回、
        error 为空，只是音频是空的或短得离谱。没有这个字段，上游只能看到
        "成功"，无从察觉内容被丢了。
        """
        self.assertTrue(TtsSynthesisResult(seq=1, audio_ms=0.0).is_silent)
        self.assertFalse(TtsSynthesisResult(seq=1, audio_ms=4101.0).is_silent)


if __name__ == "__main__":
    unittest.main()
