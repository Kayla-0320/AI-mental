# -*- coding: utf-8 -*-
"""CosyVoice3 引擎（微服务客户端）单元测试

全部**纯逻辑**：把 HTTP 层打桩，不需要模型、不需要 GPU、不需要真的起微服务。
这很重要 —— 远程引擎最容易出的问题是"连不上/返回异常时行为不对"，
而那类路径恰恰是没模型时也必须被测到的。

覆盖四组：
    1. 配置解析（COSYVOICE_API_URL / COSYVOICE_SPEED / TTS_SPEED 的优先级与钳制）
    2. PCM 反序列化（int16 → float32 的边界：空、奇数长度、满量程）
    3. 探测语义（available / load_error 的三类失败要能分开报）
    4. 合成语义（请求体、回调契约、取消、错误不上抛、采样率以响应头为准）
"""
from __future__ import annotations

import io
import json
import os
import unittest
import urllib.error
from unittest import mock

import numpy as np

from shared.dataclasses import TtsEngine
from tts.cosyvoice_engine import (
    DEFAULT_BASE_URL,
    MAX_TEXT_CHARS,
    CosyVoiceTtsEngine,
    _pcm16_to_float32,
    default_base_url,
    default_speed,
)


def _pcm_bytes(samples: list[float]) -> bytes:
    """float32 样本 → int16 小端字节（与服务端 float_to_pcm16 同口径）"""
    arr = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


class _FakeResponse:
    """最小可用的 urlopen 返回值：支持 context manager / read / headers

    ⚠️ 刻意**不用 io.BytesIO**：真实 HTTP 分块流是"按到达逐块交付"，
    而 BytesIO.read(n) 会把小于 n 的多块合并成一次返回 —— 那样测不出
    "引擎是否逐块交付给 on_chunk"这个关键契约（第一版就打桩错了这里）。
    所以这里每块只交付一块，模拟网络按块到达。
    """

    def __init__(self, blocks: list[bytes], headers: dict[str, str] | None = None) -> None:
        self._blocks = list(blocks)
        self._idx = 0
        self.headers = headers or {}

    def read(self, n: int = -1) -> bytes:
        if n < 0:  # read() → 一次性给完剩余
            rest = b"".join(self._blocks[self._idx:])
            self._idx = len(self._blocks)
            return rest
        if self._idx >= len(self._blocks):
            return b""
        # 单块大于 n 时按 n 切；否则整块交付（与真实分块流一致）
        block = self._blocks[self._idx]
        if len(block) <= n:
            self._idx += 1
            return block
        self._blocks[self._idx] = block[n:]
        return block[:n]

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _health_payload(**over: object) -> bytes:
    base = {
        "available": True, "engine": "cosyvoice3", "model_dir": "/m",
        "model_name": "Fun-CosyVoice3-0.5B", "sample_rate": 24000,
        "speed": 1.2, "prompt": "prompt_qwen3_voice.wav", "error": "",
    }
    base.update(over)
    return json.dumps(base).encode("utf-8")


def _reset_singleton() -> None:
    """单例会跨用例存活，测试里必须显式清掉，否则顺序敏感"""
    CosyVoiceTtsEngine._instance = None


# ============================================================
# 一、配置解析
# ============================================================

class TestConfigResolution(unittest.TestCase):
    """环境变量优先级与钳制"""

    def tearDown(self) -> None:
        for key in ("COSYVOICE_API_URL", "COSYVOICE_SPEED", "TTS_SPEED"):
            os.environ.pop(key, None)
        _reset_singleton()

    def test_base_url_default(self):
        """无环境变量时用默认地址"""
        os.environ.pop("COSYVOICE_API_URL", None)
        self.assertEqual(default_base_url(), DEFAULT_BASE_URL)

    def test_base_url_from_env_and_trailing_slash_stripped(self):
        """环境变量覆盖，且末尾斜杠要去掉（否则拼出 //tts/stream）"""
        os.environ["COSYVOICE_API_URL"] = "http://10.0.0.5:9000/"
        self.assertEqual(default_base_url(), "http://10.0.0.5:9000")

    def test_speed_default_is_1_2(self):
        """没配任何环境变量时是实测调好的 1.2，不是原速"""
        os.environ.pop("COSYVOICE_SPEED", None)
        os.environ.pop("TTS_SPEED", None)
        self.assertAlmostEqual(default_speed(), 1.2)

    def test_cosyvoice_speed_wins_over_tts_speed(self):
        """COSYVOICE_SPEED 优先于 TTS_SPEED

        理由：TTS_SPEED 是给 VITS 的简单插值，语义与 CosyVoice 的
        「逐块保音高加速」不同，不能混用。
        """
        os.environ["TTS_SPEED"] = "1.0"
        os.environ["COSYVOICE_SPEED"] = "1.35"
        self.assertAlmostEqual(default_speed(), 1.35)

    def test_tts_speed_used_when_cosyvoice_speed_absent(self):
        """只有 TTS_SPEED 时也认它（部署时少配一个变量）"""
        os.environ.pop("COSYVOICE_SPEED", None)
        os.environ["TTS_SPEED"] = "1.4"
        self.assertAlmostEqual(default_speed(), 1.4)

    def test_speed_is_clamped(self):
        """超出 [0.5, 2.0] 要钳住 —— 极端语速会毁掉音质且很难听出原因"""
        os.environ["COSYVOICE_SPEED"] = "9.0"
        self.assertAlmostEqual(default_speed(), 2.0)
        os.environ["COSYVOICE_SPEED"] = "0.01"
        self.assertAlmostEqual(default_speed(), 0.5)

    def test_bad_speed_value_falls_back_not_crash(self):
        """非数字不能崩，要么忽略要么用默认值"""
        os.environ["COSYVOICE_SPEED"] = "fast"
        os.environ.pop("TTS_SPEED", None)
        self.assertAlmostEqual(default_speed(), 1.2)


# ============================================================
# 二、PCM 反序列化
# ============================================================

class TestPcmDecoding(unittest.TestCase):
    """int16 小端 → float32 [-1,1]"""

    def test_roundtrip_matches_source(self):
        """round-trip 误差应在 int16 量化精度内"""
        src = np.array([0.0, 0.5, -0.5, 0.999, -0.999], dtype=np.float32)
        out = _pcm16_to_float32(_pcm_bytes(list(src)))
        self.assertEqual(len(out), len(src))
        np.testing.assert_allclose(out, src, atol=1e-4)

    def test_empty_bytes_gives_empty_float32(self):
        """空块返回空数组，而不是抛异常（服务端可能发空块）"""
        out = _pcm16_to_float32(b"")
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(len(out), 0)

    def test_odd_length_is_truncated_not_crash(self):
        """奇数字节要截断 —— 直接 frombuffer 会抛 ValueError"""
        out = _pcm16_to_float32(b"\x01\x02\x03")
        self.assertEqual(len(out), 1)

    def test_full_scale_does_not_wrap(self):
        """满量程样本不能回绕成反相噪声（-32768 → -1.0，不是 +1.0）"""
        out = _pcm16_to_float32(np.array([-32768], dtype="<i2").tobytes())
        self.assertLess(out[0], 0)
        self.assertAlmostEqual(out[0], -1.0, places=3)


# ============================================================
# 三、探测语义
# ============================================================

class TestHealthProbe(unittest.TestCase):
    """available / load_error 的三类失败必须能分开报"""

    def setUp(self) -> None:
        _reset_singleton()
        self.engine = CosyVoiceTtsEngine(base_url="http://127.0.0.1:8002")

    def tearDown(self) -> None:
        _reset_singleton()

    def test_available_on_healthy_service(self):
        """健康时 available=True 且采纳服务自述的采样率"""
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeResponse([_health_payload(sample_rate=24000)])):
            self.assertTrue(self.engine.available)
            self.assertEqual(self.engine.sample_rate, 24000)
            self.assertEqual(self.engine.engine, TtsEngine.COSYVOICE)
            self.assertEqual(self.engine.load_error, "")
            self.assertEqual(self.engine.num_speakers, 1)
            # 计算不在本进程，线程数必须是 0，不能乱报
            self.assertEqual(self.engine.num_threads, 0)

    def test_connection_failure_explains_how_to_start_service(self):
        """连不上时错误信息必须告诉人"去起微服务"

        本项目最难的失效是"服务起来了但某能力静默失效"，所以错误必须可操作。
        """
        with mock.patch("urllib.request.urlopen",
                        side_effect=urllib.error.URLError("Connection refused")):
            self.assertFalse(self.engine.available)
            self.assertIn("连不上", self.engine.load_error)
            self.assertIn("uvicorn", self.engine.load_error)
            self.assertEqual(self.engine.engine, TtsEngine.UNAVAILABLE)

    def test_service_online_but_model_unavailable_is_reported_separately(self):
        """服务在线但模型坏了 —— 与"连不上"是两类问题，不能混成一句"""
        payload = _health_payload(available=False, error="找不到模型目录：/x")
        with mock.patch("urllib.request.urlopen", return_value=_FakeResponse([payload])):
            self.assertFalse(self.engine.available)
            self.assertIn("模型不可用", self.engine.load_error)
            self.assertIn("找不到模型目录", self.engine.load_error)

    def test_http_error_is_reported(self):
        """HTTP 4xx/5xx 也要变成状态，不能抛给请求处理链"""
        err = urllib.error.HTTPError("http://x/health", 503, "Service Unavailable", {}, None)
        with mock.patch("urllib.request.urlopen", side_effect=err):
            self.assertFalse(self.engine.available)
            self.assertIn("503", self.engine.load_error)

    def test_probe_is_not_repeated_after_success(self):
        """成功探测后不重复探测（省掉每次合成前的额外往返）"""
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeResponse([_health_payload()])) as m:
            self.engine.load()
            self.engine.load()
            self.engine.load()
            self.assertEqual(m.call_count, 1)

    def test_status_dict_has_the_keys_api_layer_reads(self):
        """status() 的键必须覆盖 api/tts.py 读的那几个（缺一个就是 KeyError）"""
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeResponse([_health_payload()])):
            info = self.engine.status()
        for key in ("available", "engine", "model_dir", "model_name", "sample_rate",
                    "num_speakers", "num_threads", "max_num_sentences",
                    "voice_prompt", "speed", "error"):
            self.assertIn(key, info, f"status() 缺键：{key}")


# ============================================================
# 四、合成语义
# ============================================================

class TestSynthesizeStream(unittest.TestCase):
    """请求体、回调契约、取消、错误处理"""

    def setUp(self) -> None:
        _reset_singleton()
        self.engine = CosyVoiceTtsEngine(base_url="http://127.0.0.1:8002")
        # 预置成"已探测成功"，聚焦合成路径
        with mock.patch("urllib.request.urlopen",
                        return_value=_FakeResponse([_health_payload()])):
            self.engine.load()

    def tearDown(self) -> None:
        _reset_singleton()

    def _patch_stream(self, blocks: list[bytes], headers: dict[str, str] | None = None):
        return mock.patch("urllib.request.urlopen",
                          return_value=_FakeResponse(blocks, headers))

    def test_chunks_are_delivered_in_order_and_counted(self):
        """分块要按序交付，且 audio_ms 按实际交付样本数算"""
        blocks = [_pcm_bytes([0.1] * 100), _pcm_bytes([0.2] * 50)]
        seen: list[int] = []

        with self._patch_stream(blocks, {"X-Sample-Rate": "24000"}):
            result = self.engine.synthesize_stream(
                "你好。", seq=7, on_chunk=lambda a: (seen.append(len(a)), 0)[1])

        self.assertEqual(seen, [100, 50])
        self.assertEqual(result.engine, TtsEngine.COSYVOICE)
        self.assertEqual(result.seq, 7)
        self.assertEqual(result.sample_rate, 24000)
        self.assertAlmostEqual(result.audio_ms, 150 / 24000 * 1000, places=3)
        self.assertFalse(result.cancelled)
        self.assertTrue(result.ok)
        self.assertEqual(result.spoken_text, "你好。")

    def test_callback_returning_nonzero_cancels(self):
        """回调返回非 0 → 停止并标 cancelled（打断语义，与 VITS/Qwen3 一致）"""
        blocks = [_pcm_bytes([0.1] * 100), _pcm_bytes([0.2] * 100)]
        with self._patch_stream(blocks):
            result = self.engine.synthesize_stream(
                "你好。", seq=1, on_chunk=lambda a: 1)
        self.assertTrue(result.cancelled)
        # 取消算成功：用户打断是正常路径，不是故障
        self.assertTrue(result.ok)
        self.assertEqual(result.audio_ms, 0.0)

    def test_sample_rate_comes_from_response_header(self):
        """采样率以响应头为准，不能写死 24000"""
        with self._patch_stream([_pcm_bytes([0.1] * 10)], {"X-Sample-Rate": "16000"}):
            result = self.engine.synthesize_stream("你好。", seq=0, on_chunk=lambda a: 0)
        self.assertEqual(result.sample_rate, 16000)
        self.assertAlmostEqual(result.audio_ms, 10 / 16000 * 1000, places=3)

    def test_speed_1_0_means_unspecified_not_original_speed(self):
        """speed=1.0 表示"用服务侧默认"，不能把 1.2 覆盖回原速

        这是本项目最怕的失效类型："改了没生效"。算法侧 TTS_SPEED 的默认值
        就是 1.0，若原样透传，服务侧调好的 1.2× 会被无声取消。
        """
        sent: list[dict] = []

        def _capture(request, timeout=None):  # type: ignore[no-untyped-def]
            sent.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse([_pcm_bytes([0.1] * 10)])

        with mock.patch("urllib.request.urlopen", side_effect=_capture):
            self.engine.synthesize_stream("你好。", seq=0, speed=1.0, on_chunk=lambda a: 0)
        self.assertAlmostEqual(sent[0]["speed"], 1.2)

    def test_explicit_speed_is_passed_through(self):
        """显式传的语速要透传（用户真的想改时得生效）"""
        sent: list[dict] = []

        def _capture(request, timeout=None):  # type: ignore[no-untyped-def]
            sent.append(json.loads(request.data.decode("utf-8")))
            return _FakeResponse([_pcm_bytes([0.1] * 10)])

        with mock.patch("urllib.request.urlopen", side_effect=_capture):
            self.engine.synthesize_stream("你好。", seq=0, speed=1.45, on_chunk=lambda a: 0)
        self.assertAlmostEqual(sent[0]["speed"], 1.45)

    def test_empty_text_returns_silent_result_without_http(self):
        """空文本是正常情况（闸门只放行标点），不该发 HTTP 也不该报错"""
        with mock.patch("urllib.request.urlopen") as m:
            result = self.engine.synthesize_stream("   ", seq=3, on_chunk=lambda a: 0)
            m.assert_not_called()
        self.assertEqual(result.audio_ms, 0.0)
        self.assertEqual(result.error, "")
        self.assertTrue(result.is_silent)

    def test_overlong_text_is_rejected_not_truncated(self):
        """超长文本要显式失败 —— 静默截断是本项目最怕的失效"""
        with mock.patch("urllib.request.urlopen") as m:
            result = self.engine.synthesize_stream(
                "字" * (MAX_TEXT_CHARS + 1), seq=0, on_chunk=lambda a: 0)
            m.assert_not_called()
        self.assertFalse(result.ok)
        self.assertIn("过长", result.error)

    def test_http_error_becomes_result_not_exception(self):
        """HTTP 失败要转成 result.error，绝不能抛穿到 WS 处理链"""
        err = urllib.error.HTTPError(
            "http://x/tts/stream", 500, "Internal Server Error", {},
            io.BytesIO(b'{"detail":"\\u5408\\u6210\\u5931\\u8d25"}'))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            result = self.engine.synthesize_stream("你好。", seq=2, on_chunk=lambda a: 0)
        self.assertFalse(result.ok)
        self.assertEqual(result.engine, TtsEngine.UNAVAILABLE)
        self.assertIn("500", result.error)
        self.assertEqual(result.seq, 2)

    def test_transport_error_becomes_result_not_exception(self):
        """连接中断（微服务半路挂掉）也要变成 result.error"""
        with mock.patch("urllib.request.urlopen",
                        side_effect=ConnectionResetError("connection reset")):
            result = self.engine.synthesize_stream("你好。", seq=0, on_chunk=lambda a: 0)
        self.assertFalse(result.ok)
        self.assertIn("合成失败", result.error)

    def test_silent_success_is_flagged_as_silent(self):
        """服务返回 200 但一个字节都没有 —— 必须能被 is_silent 发现"""
        with self._patch_stream([]):
            result = self.engine.synthesize_stream("你好。", seq=0, on_chunk=lambda a: 0)
        self.assertTrue(result.ok)
        self.assertTrue(result.is_silent)


if __name__ == "__main__":
    unittest.main()
