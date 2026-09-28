# -*- coding: utf-8 -*-
"""语音识别 API —— 本地 sherpa-onnx

接口：
    - POST   /asr/transcribe : 上传一段 16bit PCM WAV（base64）→ 返回识别文本（离线整段）
    - WS     /asr/ws         : 持续推 PCM 分片 → 回推中间结果与定稿（流式，边说边出字）
    - GET    /asr/status     : 引擎可用性（离线 + 流式）与模型路径
    - GET    /asr/model-dir  : 自动解析到的模型目录，便于排查模型放错位置

与浏览器 Web Speech API 的关键差别：
    音频只在本机「浏览器 → 本服务」之间流转，不经过任何外部云服务。
    前端配套：`client/src/services/localAsr.ts`（离线）与
    `client/src/services/streamingAsr.ts`（流式）。

端点声明为**同步** `def`：识别是 CPU 密集操作，交给 FastAPI 的线程池执行，
避免阻塞事件循环（实测离线 RTF ≈ 0.008、流式每 100ms 块 ≈ 1.6 ms）。
"""
from __future__ import annotations

import base64
import binascii
import json
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from asr.engine import ParaformerRecognizer, decode_wav, resolve_model_dir
from asr.punctuation import PunctuationRestorer
from asr.streaming import StreamingRecognizer
from shared.dataclasses import AsrEngine, StreamingAsrUpdate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/asr", tags=["语音识别"])

#: base64 长度上限 —— 约 100 秒 16kHz 单声道 16bit 音频，防止异常大请求打爆内存
MAX_AUDIO_B64_CHARS = 4_000_000


class TranscribeRequest(BaseModel):
    """语音识别请求"""
    audio_base64: str = Field(..., description="16bit PCM WAV（或裸 PCM）的 base64 编码")
    sample_rate: int = Field(
        default=16000,
        ge=8000,
        le=48000,
        description="采样率；非 16000 时由识别器内部重采样",
    )


class TranscribeResponse(BaseModel):
    """语音识别响应"""
    text: str = Field(..., description="识别文本，可能为空字符串")
    engine: str = Field(..., description="实际使用的引擎")
    duration_ms: float = Field(default=0.0, description="输入音频时长（毫秒）")
    latency_ms: float = Field(default=0.0, description="本次识别耗时（毫秒）")
    error: str = Field(default="", description="失败原因；非空表示本次识别失败")


class StatusResponse(BaseModel):
    """引擎状态"""
    available: bool
    engine: str
    model_dir: str | None = None
    model_name: str | None = None
    sample_rate: int = 16000
    error: str = ""
    # 流式引擎（/asr/ws）的独立状态：它与离线引擎用不同模型，可能一个有一个没有
    streaming_available: bool = False
    streaming_engine: str = AsrEngine.UNAVAILABLE.value
    streaming_model_name: str | None = None
    streaming_error: str = ""
    # 标点模型：只有定稿才会用到，缺失不影响识别
    punctuation_available: bool = False
    punctuation_error: str = ""


def _update_payload(u: StreamingAsrUpdate) -> dict[str, object]:
    """把流式更新转成 WebSocket 的 JSON 载荷"""
    return {
        "type": u.event.value,  # partial | final
        "text": u.text,
        "elapsed_ms": round(u.elapsed_ms, 1),
        "latency_ms": round(u.latency_ms, 1),
        "reason": u.reason,
        "engine": u.engine.value,
        # 定稿文本是否经过离线大模型复识别纠错（标点是定稿时无条件尝试的）
        "refined": u.refined,
    }


@router.websocket("/ws")
async def asr_stream(websocket: WebSocket) -> None:
    """流式语音识别 WebSocket。

    客户端 → 服务端：
        文本帧   = JSON 控制指令，**必须在送音频之前先发一条 hello**：
            {"type":"hello","mode":"utterance"}  整段录入：停顿不定稿，只有
                                                 finalize 才定稿（聊天页语音输入）
            {"type":"hello","mode":"streaming"}  端点模式：停顿即定稿（通话页）
            缺省（没发 hello）= streaming，与旧客户端兼容
        二进制帧 = 16kHz 单声道 **int16 小端** PCM，建议每帧 100ms（3200 字节）
        文本帧   = JSON 控制指令
            {"type":"finalize"}  用户主动结束，冲刷残余并定稿
            {"type":"reset"}     丢弃当前段

    服务端 → 客户端（均为 JSON 文本帧）：
        {"type":"status", available, engine, error, mode} 连接建立后立即下发
        {"type":"partial", text, elapsed_ms, latency_ms} 中间结果，会被覆盖，只用于上屏
        {"type":"final",   text, elapsed_ms, latency_ms, reason}
                                                        定稿；reason=endpoint|finalize
        {"type":"error", message}

    为什么把 partial / final 分开：中间结果不稳定，下游的文本情感、认知扭曲、
    语音语义分析必须只吃定稿，否则同一句话会被反复分析。
    """
    await websocket.accept()

    recognizer = StreamingRecognizer.instance()
    if not recognizer.available:
        await websocket.send_json(
            {
                "type": "status",
                "available": False,
                "engine": AsrEngine.UNAVAILABLE.value,
                "error": recognizer.load_error,
            }
        )
        # 1011 = internal error：明确告诉前端"服务端能力缺失"，而不是让它以为连上了
        await websocket.close(code=1011)
        return

    # ── 首帧：决定"说话停顿要不要自动定稿" ──────────────────────────────────
    # 必须在任何音频之前确定：模式决定端点检测怎么建、要不要走流式模型。
    finalize_on_endpoint = True
    first_audio: bytes | None = None
    try:
        first = await websocket.receive()
    except WebSocketDisconnect:
        return
    if first.get("type") == "websocket.disconnect":
        return
    raw_first_audio = first.get("bytes")
    if raw_first_audio:
        # 旧客户端不发 hello 就直接送音频：按端点模式处理，且这一帧不能吞掉
        logger.info("[ASR:stream] 未收到 hello，按端点模式处理（旧客户端）")
        first_audio = raw_first_audio
    first_text = first.get("text")
    if first_text:
        try:
            hello = json.loads(first_text)
        except json.JSONDecodeError:
            await websocket.send_json({"type": "error", "message": "首帧不是合法 JSON"})
            await websocket.close(code=1003)
            return
        mode = hello.get("mode")
        if mode == "utterance":
            finalize_on_endpoint = False
        elif mode not in (None, "streaming"):
            await websocket.send_json({"type": "error", "message": f"未知模式：{mode}"})
            await websocket.close(code=1003)
            return

    session = recognizer.new_session(finalize_on_endpoint=finalize_on_endpoint)
    if session is None:  # 竞态兜底：available 之后加载失败
        await websocket.send_json(
            {
                "type": "status",
                "available": False,
                "engine": AsrEngine.UNAVAILABLE.value,
                "error": recognizer.load_error or "流式会话创建失败",
            }
        )
        await websocket.close(code=1011)
        return

    await websocket.send_json(
        {
            "type": "status",
            "available": True,
            "engine": recognizer.engine.value,
            "error": "",
            "mode": "streaming" if finalize_on_endpoint else "utterance",
        }
    )

    async def _feed(data: bytes) -> None:
        """喂一块音频，把产生的更新推给客户端"""
        # 解码是 CPU 密集操作，丢到线程池，别卡住事件循环
        updates = await run_in_threadpool(session.accept_pcm16, data)
        for u in updates:
            await websocket.send_json(_update_payload(u))

    try:
        if first_audio:
            await _feed(first_audio)

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data:
                await _feed(data)
                continue

            text = message.get("text")
            if not text:
                continue

            try:
                ctrl = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "控制帧不是合法 JSON"})
                continue

            action = ctrl.get("type")
            if action == "finalize":
                u = await run_in_threadpool(session.finalize)
                if u is not None:
                    await websocket.send_json(_update_payload(u))
                else:
                    # 本段没有内容也要回一条 final，前端才知道"结束了"。
                    # engine 取会话自己的流式引擎：`recognizer.engine.value` 在
                    # 端点检测关闭时可能报出与本研究路径不符的标识。
                    await websocket.send_json(
                        {"type": "final", "text": "", "elapsed_ms": 0, "latency_ms": 0,
                         "reason": "finalize", "engine": session.engine.value}
                    )
            elif action == "reset":
                session.reset()
            else:
                await websocket.send_json(
                    {"type": "error", "message": f"未知控制指令：{action}"}
                )
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 —— 单条连接出错不应影响服务
        logger.exception("[ASR:stream] WebSocket 处理异常")
        try:
            await websocket.close(code=1011)
        except Exception:  # noqa: BLE001
            pass


@router.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    """把一段音频转成文字（本地推理，音频不出本机）。"""
    if len(req.audio_base64) > MAX_AUDIO_B64_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"音频过大（base64 长度 {len(req.audio_base64)} > {MAX_AUDIO_B64_CHARS}）",
        )

    try:
        raw = base64.b64decode(req.audio_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"audio_base64 解码失败：{exc}") from exc

    if not raw:
        raise HTTPException(status_code=400, detail="音频为空")

    try:
        samples, sample_rate = decode_wav(raw)
    except Exception as exc:  # noqa: BLE001 —— 格式问题一律 400，附上原因
        raise HTTPException(status_code=400, detail=f"音频解析失败：{exc}") from exc

    recognizer = ParaformerRecognizer.instance()
    result = recognizer.transcribe(samples, sample_rate)

    # 引擎不可用属于「服务未就绪」而非请求错误：返回 200 + error 字段，
    # 让前端能如实告知用户，而不是拿到一个看似成功的空字符串。
    return TranscribeResponse(
        text=result.text,
        engine=result.engine.value,
        duration_ms=round(result.duration_ms, 1),
        latency_ms=round(result.latency_ms, 1),
        error=result.error,
    )


@router.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    """查询本地 ASR 各环节状态（流式 / 离线 / 标点各自的可用性与失败原因）。"""
    offline = ParaformerRecognizer.instance()
    streaming = StreamingRecognizer.instance()
    punct = PunctuationRestorer.instance()
    oi = offline.status()
    si = streaming.status()
    pi = punct.status()
    return StatusResponse(
        available=bool(oi["available"]),
        engine=str(oi["engine"]),
        model_dir=oi["model_dir"],  # type: ignore[arg-type]
        model_name=oi["model_name"],  # type: ignore[arg-type]
        sample_rate=int(oi["sample_rate"]),  # type: ignore[arg-type]
        error=str(oi["error"]),
        streaming_available=bool(si["available"]),
        streaming_engine=str(si["engine"]),
        streaming_model_name=si["model_name"],  # type: ignore[arg-type]
        streaming_error=str(si["error"]),
        punctuation_available=bool(pi["available"]),
        punctuation_error=str(pi["error"]),
    )


@router.get("/model-dir")
def model_dir() -> dict[str, str | None]:
    """返回自动解析到的模型目录，便于排查「模型放错位置」这类问题。"""
    d = resolve_model_dir()
    return {"model_dir": str(d) if d else None, "expected_engine": AsrEngine.PARAFORMER.value}
