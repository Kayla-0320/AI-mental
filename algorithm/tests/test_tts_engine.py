# -*- coding: utf-8 -*-
"""本地 TTS 引擎单元测试

分两类：
    1. **纯逻辑**（不需要模型）—— 术语替换、线程数钳制、入参校验、取消语义。
       这些在没下模型的机器上也要能跑。
    2. **需要真模型** —— 合成、以及那条**截断回归测试**。
       模型不在时整体 skip，而不是让 CI 红。

关于那条截断测试（``test_multi_sentence_is_not_truncated``）：
    它是 ``max_num_sentences=1`` 那个坑的唯一可靠护栏。配错时
    「4 句文本」与「1 句文本」的音频时长几乎相等（实测 4258 ms vs 4218 ms），
    断言立刻失败。详见 docs/tts_benchmark.md §1 发现三。
"""
from __future__ import annotations

import inspect
import os
import unittest
from pathlib import Path

import numpy as np

from shared.dataclasses import TtsEngine
from tts.engine import (
    MAX_TEXT_CHARS,
    VitsTtsEngine,
    _find_onnx,
    _usable,
    default_num_threads,
    describe_model_dir,
    resolve_tts_model_dir,
)
from tts.pronunciation import PRONOUNCE_FIX, prepare_for_speech, unpronounceable_spans

#: 模型是否就绪 —— 决定第二类测试跑不跑
_MODEL_DIR, _RESOLVE_ERROR = resolve_tts_model_dir()
_HAS_MODEL = _MODEL_DIR is not None


# ============================================================
# 一、纯逻辑（无需模型）
# ============================================================

class TestPrepareForSpeech(unittest.TestCase):
    """术语替换：把念不出来的拉丁文换成可发音的中文"""

    def test_replaces_known_term(self):
        """CBT 必须被换成中文 —— 词表不含 ASCII 字母，原样送会被静默跳过"""
        self.assertEqual(prepare_for_speech("我们试试 CBT 里的方法"), "我们试试 认知行为疗法 里的方法")

    def test_case_insensitive(self):
        """大小写不敏感（LLM 可能输出 cbt）"""
        self.assertEqual(prepare_for_speech("cbt"), "认知行为疗法")
        self.assertEqual(prepare_for_speech("Cbt"), "认知行为疗法")

    def test_matches_adjacent_to_chinese(self):
        """中文紧邻时也必须替换

        这是本模块最容易写错的一处：Python 的 `\\b` 以 `\\w` 为界，而 `\\w`
        **包含中文**，所以 `试试CBT里` 里 CBT 两侧都不构成词边界，
        用 `\\bCBT\\b` 会漏匹配 → 词被静默吞掉。
        """
        self.assertEqual(prepare_for_speech("试试CBT里"), "试试认知行为疗法里")

    def test_does_not_replace_inside_a_longer_word(self):
        """`CBTs` 不是 `CBT` —— 不能误伤"""
        self.assertEqual(prepare_for_speech("CBTs"), "CBTs")

    def test_longest_key_wins(self):
        """长键优先，避免短键吃掉长键的一部分

        表里同时有 ACT 和 MBCT；若先替换 ACT，MBCT 会被拆成 "M接纳承诺疗法"。
        """
        self.assertEqual(prepare_for_speech("MBCT"), "正念认知疗法")

    def test_keeps_numbers_untouched(self):
        """数字交给 number.fst / phone.fst，本表不碰"""
        self.assertEqual(prepare_for_speech("拨打 400-161-9995"), "拨打 400-161-9995")

    def test_empty_input(self):
        """空输入返回空串，不抛异常"""
        self.assertEqual(prepare_for_speech(""), "")
        self.assertEqual(prepare_for_speech(None), "")  # type: ignore[arg-type]

    def test_text_without_latin_is_unchanged(self):
        """纯中文原样返回"""
        s = "听起来这段时间你过得挺不容易的。"
        self.assertEqual(prepare_for_speech(s), s)


class TestUnpronounceableSpans(unittest.TestCase):
    """未覆盖的拉丁文必须可被观测 —— 丢内容不能是无声的"""

    def test_detects_unmapped_word(self):
        """表里没有的词要被报出来"""
        self.assertEqual(unpronounceable_spans("这是一个 HELLOWORLD 测试"), ["HELLOWORLD"])

    def test_mapped_words_are_not_reported(self):
        """表里有的词不算"会丢" """
        self.assertEqual(unpronounceable_spans("CBT 和 DBT"), [])

    def test_no_latin_at_all(self):
        """没有拉丁文时返回空列表"""
        self.assertEqual(unpronounceable_spans("你好，我在听。"), [])

    def test_every_mapped_key_is_pure_ascii_letters(self):
        """表的键必须是纯 ASCII 字母数字 —— 否则正则永远匹配不到它"""
        for key in PRONOUNCE_FIX:
            self.assertRegex(key, r"^[A-Za-z][A-Za-z0-9]*$", key)


class TestThreadConfig(unittest.TestCase):
    """线程数配置 —— 实测它是 TTS 最重要的性能杠杆"""

    def test_default_num_threads_is_clamped(self):
        """必须落在 [2, 8]

        实测 1 线程 RTF 0.568（达不到 go 判据），所以下限是 2 而不是 1；
        上限 8 是因为之后收益递减、且要与流式 ASR 争 CPU。
        """
        n = default_num_threads()
        self.assertGreaterEqual(n, 2)
        self.assertLessEqual(n, 8)

    def test_leaves_headroom_for_asr(self):
        """给同机的流式 ASR 留核：不应等于 CPU 总数"""
        import os
        cores = os.cpu_count() or 4
        if cores > 3:
            self.assertLess(default_num_threads(), cores)


class TestModelDirResolution(unittest.TestCase):
    """模型定位：**换模型必须真的生效，坏了必须报错**

    这一组测试守的是一个真实踩过的 bug：
        `TTS_MODEL_DIR` 指向 `vits-zh-hf-theresa`（onnx 叫 `theresa.onnx`），
        而 `_usable()` 当时写死要 `vits-zh-hf-fanchen-C.onnx` →
        判它"不可用" → **无声回退到默认模型**。
        调用方只看到"换模型没生效"，日志里一个字的抱怨都没有。
    """

    def test_unusable_dir_is_detected(self):
        """不含必需文件的目录必须被判为不可用"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(_usable(Path(tmp)), "空目录不应被判为可用模型")

    def test_missing_dir_is_not_usable(self):
        """不存在的路径不能通过校验"""
        self.assertFalse(_usable(Path("/definitely/not/here")))

    def test_onnx_name_is_discovered_not_hardcoded(self):
        """onnx 文件名必须**按目录内容识别**，不能写死

        各模型的 onnx 名不同（`theresa.onnx` / `vits-zh-hf-fanchen-C.onnx` /
        `model.onnx`）。写死名字等于把"换模型"这条路的门焊死。
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            fake = d / "some-other-name.onnx"
            fake.write_bytes(b"x" * 128)
            found = _find_onnx(d)
            self.assertIsNotNone(found, "只要目录里有 onnx，就应能被识别（不论文件名）")
            self.assertEqual(found.name, "some-other-name.onnx")

    def test_prefers_non_quantized_onnx(self):
        """同时存在 int8 与完整版时，优先完整版（音质更好）"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "vits-aishell3.int8.onnx").write_bytes(b"x" * 64)
            (d / "vits-aishell3.onnx").write_bytes(b"x" * 4096)
            found = _find_onnx(d)
            self.assertIsNotNone(found)
            self.assertNotIn(".int8.", found.name)

    def test_explicit_bad_path_fails_instead_of_falling_back(self):
        """⚠️ 显式指定却不可用 → 必须失败，**不得**回退到别的模型

        这是本组最重要的一条：静默回退会让"换模型没生效"变成一个查不出来的问题。
        """
        resolved, reason = resolve_tts_model_dir("/definitely/not/here")
        self.assertIsNone(resolved, "显式指定的坏路径不得回退到默认模型")
        self.assertIn("不可用", reason)
        self.assertIn("显式传入的路径", reason)

    def test_env_dir_wins_over_default(self):
        """TTS_MODEL_DIR 指向的可用模型必须被采纳（而不是默认模型）

        用真实的第二个模型做验证；只装了一个模型时跳过（不算失败）。
        """
        import os
        import tempfile

        root = Path(__file__).resolve().parents[1] / "models" / "tts"
        others = [d for d in sorted(root.iterdir()) if _usable(d) and d.name != "vits-zh-hf-fanchen-C"] \
            if root.is_dir() else []
        if not others:
            self.skipTest("本机只装了一个 TTS 模型，无法验证'换模型生效'")

        target = others[0]
        old = os.environ.get("TTS_MODEL_DIR")
        os.environ["TTS_MODEL_DIR"] = str(target)
        try:
            resolved, reason = resolve_tts_model_dir()
        finally:
            if old is None:
                os.environ.pop("TTS_MODEL_DIR", None)
            else:
                os.environ["TTS_MODEL_DIR"] = old

        self.assertEqual(reason, "")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.name, target.name, "TTS_MODEL_DIR 必须被采纳")

    def test_env_bad_path_reports_which_source(self):
        """环境变量指向坏路径时，错误信息要指明**是哪个来源**坏的"""
        import os

        old = os.environ.get("TTS_MODEL_DIR")
        os.environ["TTS_MODEL_DIR"] = "/definitely/not/here"
        try:
            resolved, reason = resolve_tts_model_dir()
        finally:
            if old is None:
                os.environ.pop("TTS_MODEL_DIR", None)
            else:
                os.environ["TTS_MODEL_DIR"] = old

        self.assertIsNone(resolved)
        self.assertIn("TTS_MODEL_DIR", reason)

    def test_describe_model_dir_names_what_is_missing(self):
        """失败说明必须列出缺了什么，而不是只说一句「不可用」"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "lexicon.txt").write_text("x", encoding="utf-8")
            desc = describe_model_dir(d)
            self.assertIn("*.onnx", desc)
            self.assertIn("tokens.txt", desc)


class TestPrepareForSpeechIntegration(unittest.TestCase):
    """替换表与引擎的衔接：speak 的文本必须是替换后的"""

    @unittest.skipUnless(_HAS_MODEL, "未找到 TTS 模型")
    def test_spoken_text_is_post_replacement(self):
        """`spoken_text` 必须是替换后的文本

        前端做文本级自回声过滤要拿这个值 —— 若它回的是原文（含 CBT），
        而实际念的是"认知行为疗法"，回声过滤就会失配、把 AI 自己的声音
        当成用户插话。
        """
        engine = VitsTtsEngine.instance()
        if not engine.load():
            self.skipTest(f"引擎不可用：{engine.load_error}")
        prepared = prepare_for_speech("我们试试 CBT 里的方法")
        result = engine.synthesize(prepared, seq=1, on_chunk=lambda _s: 0)
        self.assertNotIn("CBT", result.spoken_text)
        self.assertIn("认知行为疗法", result.spoken_text)


# ============================================================
# 二、需要真模型
# ============================================================

@unittest.skipUnless(_HAS_MODEL, f"未找到 TTS 模型（{_MODEL_DIR}）—— 跳过需要真模型的测试")
class TestSynthesize(unittest.TestCase):
    """真实合成。跑之前请确认模型已下载

    python algorithm/tools/download_asr_model.py --preset tts-zh-fanchen-C
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = VitsTtsEngine.instance()
        if not cls.engine.load():
            raise unittest.SkipTest(f"TTS 引擎加载失败：{cls.engine.load_error}")

    def _synth(self, text: str, seq: int = 0):
        """合成一次，返回 (收到的二进制块列表, 结果对象)"""
        chunks: list[bytes] = []

        def _collect(samples: np.ndarray) -> int:
            from api.tts import float_to_pcm16  # noqa: PLC0415

            chunks.append(float_to_pcm16(samples))
            return 0

        result = self.engine.synthesize(text, seq=seq, on_chunk=_collect)
        return chunks, result

    def test_load_reports_real_config(self):
        """加载后必须报出真实配置，而不是默认值占位"""
        self.assertTrue(self.engine.available)
        self.assertGreater(self.engine.sample_rate, 0)
        self.assertGreater(self.engine.num_speakers, 0)

    def test_synthesize_produces_audio(self):
        """能合成出非空音频，且 RTF 达标（< 0.4）"""
        _, result = self._synth("我在听，你慢慢说。")
        self.assertEqual(result.engine, TtsEngine.VITS)
        self.assertEqual(result.error, "")
        self.assertFalse(result.is_silent, "合成结果为静音 —— 模型或文本有问题")
        self.assertLess(result.rtf, 0.4, f"RTF {result.rtf:.3f} 未达标")

    def test_callback_fires_exactly_once(self):
        """实测：VITS 对每一句只回调一次（这决定了 WS 一帧一帧的协议）"""
        chunks, _ = self._synth("我在听。")
        self.assertEqual(len(chunks), 1)

    def test_multi_sentence_is_not_truncated(self):
        """⚠️ 核心回归测试：多句文本不得被截断

        `max_num_sentences=1` 时，generate() 只合成第一句、其余**静默丢弃**：
        66 字的 4 句文本只产出 4258 ms 音频，与 16 字单句的 4218 ms 几乎一样。
        没有异常、没有日志、没有报错 —— 这正是本项目最怕的失效类型。

        断言用"4 句 ≥ 2.5 × 1 句"：配错时比值 ≈ 1.0，配对时 ≈ 3.9，
        中间留了近 3 倍余量，不会被 VITS 的时长随机性（±20%）触发假警报。
        """
        one = "听起来这段时间你过得挺不容易的。"
        four = (
            "听起来这段时间你过得挺不容易的。"
            "晚上睡不着的时候，脑子里会反复想些什么呢？"
            "是白天发生的事，还是对以后的担心？"
            "你可以慢慢讲，我不着急。"
        )
        _, r_one = self._synth(one)
        _, r_four = self._synth(four)
        ratio = r_four.audio_ms / r_one.audio_ms
        self.assertGreater(
            ratio, 2.5,
            f"多句文本疑似被截断：4 句 {r_four.audio_ms:.0f} ms vs "
            f"1 句 {r_one.audio_ms:.0f} ms（比值 {ratio:.2f}）。"
            f"请检查 OfflineTtsConfig.max_num_sentences 是否为 -1。",
        )

    def test_empty_text_is_a_normal_empty_result(self):
        """空文本是正常情况（闸门可能只放行了标点），不是错误"""
        _, result = self._synth("   ", seq=7)
        self.assertEqual(result.error, "")
        self.assertEqual(result.seq, 7)
        self.assertEqual(result.audio_ms, 0.0)

    def test_overlong_text_is_rejected_not_truncated(self):
        """超长文本必须**报错**，不能悄悄截断

        静默截断是本项目最怕的失效类型；这里宁可让上游知道它违反了切句约定。
        """
        _, result = self._synth("啊" * (MAX_TEXT_CHARS + 1), seq=1)
        self.assertIn("过长", result.error)
        self.assertEqual(result.engine, TtsEngine.UNAVAILABLE)

    def test_cancel_stops_delivery(self):
        """回调返回非 0 → 音频不再交付，结果被标记为 cancelled"""
        delivered: list[int] = []

        def _cancel(_samples: np.ndarray) -> int:
            delivered.append(1)
            return 1  # 立刻取消

        result = self.engine.synthesize("听起来这段时间你过得挺不容易的。", seq=3, on_chunk=_cancel)
        self.assertTrue(result.cancelled)
        self.assertTrue(result.ok, "取消是正常路径，不是故障")
        self.assertEqual(result.audio_ms, 0.0, "取消后不应再计入已交付音频")

    def test_seq_is_echoed_back(self):
        """seq 必须原样带回 —— 客户端靠它对号入座"""
        _, result = self._synth("我在听。", seq=42)
        self.assertEqual(result.seq, 42)


class TestStatus(unittest.TestCase):
    """状态自述 —— 现场排障靠它"""

    def test_status_always_reports_max_num_sentences(self):
        """`max_num_sentences` 必须恒为 -1 且出现在状态里

        它是截断坑的护栏值。把它报出来，是为了让"现场到底用的是哪个配置"
        一眼可见，而不是只能去读代码。
        """
        info = VitsTtsEngine.instance().status()
        self.assertEqual(info["max_num_sentences"], -1)

    def test_status_includes_troubleshooting_fields(self):
        """线程数与模型目录必须报出 —— 它们是最常见的两个根因"""
        info = VitsTtsEngine.instance().status()
        for key in ("available", "engine", "model_dir", "num_threads", "error"):
            self.assertIn(key, info)


# ============================================================
# 二、音色/语速配置（纯逻辑，无需模型）
#
# 背景（2026-09-27 用户反馈）："声音的 AI 味道太浓了，像机器一个字一个字读。"
# `say` 控制帧本来就支持 voice/sid/speed，但**调用方从来不发、服务端也没有配置入口**
# —— 结果是"想换个声音"只能改代码。这一组测试守住新加的配置面：
# `TTS_SID` / `TTS_SPEED` 环境变量，以及它们必须随任务入队（不能被闭包覆盖）。
# ============================================================

class TestVoicePreferences(unittest.TestCase):
    """`_voice_preferences()` 的解析与兜底。"""

    def setUp(self):
        from api import tts as tts_api
        self.api = tts_api
        self._saved = {k: os.environ.get(k) for k in ("TTS_SID", "TTS_SPEED")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_defaults(self):
        """没配就是 sid=0 / speed=1.0（与改动前的行为一致）。"""
        self.assertEqual(self.api._voice_preferences(), (0, 1.0))

    def test_reads_env(self):
        os.environ["TTS_SID"] = "148"
        os.environ["TTS_SPEED"] = "1.12"
        self.assertEqual(self.api._voice_preferences(), (148, 1.12))

    def test_bad_values_fall_back(self):
        """非法值必须回退默认，而不是抛异常把整个 TTS 会话打死。"""
        os.environ["TTS_SID"] = "abc"
        os.environ["TTS_SPEED"] = "fast"
        self.assertEqual(self.api._voice_preferences(), (0, 1.0))

    def test_speed_is_clamped(self):
        """语速夹到 [0.5, 2.0]：越界值多半是误配，直接放过会让听感失控。"""
        os.environ["TTS_SPEED"] = "9"
        self.assertEqual(self.api._voice_preferences()[1], 2.0)
        os.environ["TTS_SPEED"] = "0.01"
        self.assertEqual(self.api._voice_preferences()[1], 0.5)


class TestVoiceParamsReachTheEngine(unittest.TestCase):
    """sid/speed 必须**随任务入队**并传给引擎。

    这一条是真实的坑：`worker()` 是长驻协程、`say` 是逐帧到达的，
    把 sid/speed 放在闭包里而不是队列元组里，后一条 `say` 会覆盖前一条的参数 ——
    表现为"时好时坏"，极难排查。
    """

    def test_say_kwargs_carry_sid_and_speed(self):
        import inspect
        from api import tts as tts_api
        src = inspect.getsource(tts_api.tts_stream)
        self.assertIn("jobs.put((seq, prepared, voice, sid, speed))", src,
                      "sid/speed 必须随任务入队，不能只放闭包")
        self.assertIn("seq, text, voice, sid, speed = item", src,
                      "worker 必须从队列元组里取 sid/speed")
        # VITS 分支真的把它们传给了引擎
        self.assertIn('synth_kwargs = {"seq": seq, "sid": sid, "speed": speed',
                      src, "VITS 分支必须把 sid/speed 传给 synthesize")

    def test_engine_accepts_sid_and_speed(self):
        """引擎签名必须收 sid/speed —— 否则传了也是静默丢弃。"""
        sig = inspect.signature(VitsTtsEngine.synthesize)
        self.assertIn("sid", sig.parameters)
        self.assertIn("speed", sig.parameters)


if __name__ == "__main__":
    unittest.main()
