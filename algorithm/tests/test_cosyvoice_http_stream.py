"""用真实 HTTP 分块响应验证首块交付，不用会掩盖 read(n) 攒包问题的替身。"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import numpy as np
import pytest

from tts.cosyvoice_engine import CosyVoiceTtsEngine


@pytest.fixture
def pcm_server() -> Iterator[tuple[str, dict]]:
    """第二块等到第一块被客户端消费后再发送，避免靠耗时阈值判断流式交付。"""
    state = {"chunks": [], "consumed": threading.Event(), "streamed": False}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Sample-Rate", "24000")
            self.end_headers()
            for index, chunk in enumerate(state["chunks"]):
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
                if index == 0:
                    state["streamed"] = state["consumed"].wait(1)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_small_first_http_chunk_is_delivered_before_next_chunk(pcm_server: tuple[str, dict]) -> None:
    """首块不足8192字节时也要立即交付，不能等下一次模型输出填满读取缓冲。"""
    url, state = pcm_server
    state["chunks"] = [np.full(100, 123, dtype="<i2").tobytes(), np.full(50, -456, dtype="<i2").tobytes()]
    engine = CosyVoiceTtsEngine(base_url=url)
    engine._checked = True
    received: list[np.ndarray] = []

    def consume(samples: np.ndarray) -> int:
        received.append(samples)
        state["consumed"].set()
        return 0

    result = engine.synthesize_stream("测试连续播放。", on_chunk=consume)
    assert result.ok, result.error
    assert state["streamed"], "第一块被read(n)扣住，直到服务端放弃等待才交付"
    assert sum(len(chunk) for chunk in received) == 150
    assert result.audio_ms == pytest.approx(150 / 24)


def test_pcm_samples_survive_odd_http_boundaries(pcm_server: tuple[str, dict]) -> None:
    """HTTP边界可以落在int16的两个字节之间，尾字节要拼到下一块，不能逐块丢。"""
    url, state = pcm_server
    original = np.array([123, -456, 789, -1234, 32767, -32768], dtype="<i2")
    raw = original.tobytes()
    state["chunks"] = [raw[:3], raw[3:7], raw[7:]]
    engine = CosyVoiceTtsEngine(base_url=url)
    engine._checked = True
    received: list[np.ndarray] = []

    def consume(samples: np.ndarray) -> int:
        received.append(samples)
        state["consumed"].set()
        return 0

    result = engine.synthesize_stream("测试边界。", on_chunk=consume)
    assert result.ok, result.error
    assert state["streamed"]
    np.testing.assert_array_equal(np.concatenate(received), original.astype(np.float32) / 32768)
