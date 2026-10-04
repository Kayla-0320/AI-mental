# -*- coding: utf-8 -*-
"""语音合成 API —— 本地 sherpa-onnx VITS

接口：
    - WS   /tts/ws      : 逐句送文本 → 回推 16 kHz 单声道 int16 PCM（陪伴通话的出声端）
    - GET  /tts/status  : 引擎可用性、采样率、说话人数、线程数、模型目录

与 ``/asr/ws`` 的分工：那条是**音频上行**（用户说话），这条是**音频下行**（AI 说话）。
两者共用同一套约定：文本帧是 JSON 控制，二进制帧是裸 PCM。

帧协议
------
客户端 → 服务端（文本帧 JSON）::

    {"type":"say","seq":1,"text":"听起来这段时间你挺不容易的。"}
    {"type":"say","seq":1,"text":"...","voice":"温柔、共情的成熟女性"}  # 仅 qwen3 生效
    {"type":"cancel","seq":1}      取消该 seq（清空待合成队列 + 丢弃在途结果）
    {"type":"close"}               结束本连接的合成会话

服务端 → 客户端::

    {"type":"status","available":true,"engine":"qwen3-tts",
     "sample_rate":24000,"num_speakers":1,"num_threads":0,
     "voice_prompt":"温和、共情、成熟女性…","error":""}
    {"type":"start","seq":1,"sample_rate":24000}
    二进制帧（1 个）= int16 小端、单声道 PCM，即整句音频
    {"type":"end","seq":1,"spoken_text":"...","audio_ms":4101.1,
     "first_chunk_ms":443.4,"synth_ms":443.4,"rtf":0.108,"cancelled":false}
    {"type":"cancelled","seq":1}
    {"type":"error","seq":1,"message":"……"}

**为什么用 `start` / `end` 文本帧把二进制帧包起来**：二进制帧自身没有边界语义，
客户端无法知道"这一帧属于哪一句"。

⚠️ **实测（docs/tts_benchmark.md §1 发现一）：VITS 对每一句只回调一次**，
携带整句合成完的音频。所以一个 `say` 恒定对应**一个**二进制帧，
不是"多块流"。客户端仍应写成"累积到 `end` 为止"的循环（将来换真流式 TTS 时不用改），
但**不要**依赖"块数 > 1 才能工作"。

**为什么合成要串行**：``OfflineTts`` 的并发安全性**未验证**
（``asr/punctuation.py:161`` 有"会话不保证并发安全 → 加锁串行化"的先例）。
这里用一个**模块级单工作线程池**，把所有连接的合成全部排队串行执行。
代价是一路长句会短暂挡住另一路的合成；收益是彻底避开一个没验证过的并发假设。
实测 RTF 0.11–0.14（合成比播放快 7 倍），排队在实践中不会成为瓶颈。
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from shared.dataclasses import TtsEngine
from tts import create_engine, float_to_pcm16
from tts.pronunciation import prepare_for_speech

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tts", tags=["语音合成"])

#: 合成线程池 —— **全局单工作线程**，保证 `OfflineTts` 永不被并发调用。
#: 不要改成 `run_in_threadpool`：它走 anyio 的共享限流器，会并发。
_TTS_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-synth")

#: 收尾时等待在途合成结束的上限（秒）。超过就放弃，避免客户端断开后连接悬挂。
_DRAIN_TIMEOUT_S = 8.0


def _voice_preferences() -> tuple[int, float]:
    """从环境变量读**默认音色**与**默认语速**（换成"改配置就能生效"，不用改代码）。

    背景：`say` 控制帧本来就支持 ``voice`` 字段，但调用方**从来不发**，服务端也没有
    任何配置入口 —— 结果是"想换个声音"只能改代码。这里补上配置面：

        ``TTS_SID``     VITS 说话人编号（默认 0）。`fanchen-C` 有 187 个，
                        可用 `_probe_vits_speakers.py` 扫描挑选。
        ``TTS_SPEED``   语速倍率（默认 1.0）。**只对 VITS 生效** —— Qwen3-TTS
                        当前版本未暴露语速参数，传了也会被忽略并留日志。

    ⚠️ 越界的 sid 不会在这里报错：各模型对越界的行为不一致（可能落到某个默认音色、
    也可能报错）。真正的兜底是引擎返回的 error，会照常回给前端。

    Returns:
        tuple[int, float]: ``(说话人编号, 语速)``；无法解析时用安全默认值。
    """
    try:
        sid = int(os.environ.get("TTS_SID", "0") or 0)
    except (TypeError, ValueError):
        logger.warning("[TTS] TTS_SID 不是整数，回退 0")
        sid = 0
    try:
        speed = float(os.environ.get("TTS_SPEED", "1.0") or 1.0)
    except (TypeError, ValueError):
        logger.warning("[TTS] TTS_SPEED 不是数字，回退 1.0")
        speed = 1.0
    if not (0.5 <= speed <= 2.0):
        logger.warning("[TTS] TTS_SPEED=%.2f 超出合理范围 [0.5, 2.0]，已夹紧", speed)
        speed = max(0.5, min(2.0, speed))
    return sid, speed


class TtsStatusResponse(BaseModel):
    """合成引擎状态"""
    available: bool = Field(..., description="模型是否已加载且可用")
    engine: str = Field(default=TtsEngine.UNAVAILABLE.value, description="实际使用的引擎")
    model_dir: str | None = Field(default=None, description="加载到的模型目录")
    model_name: str | None = Field(default=None, description="模型名")
    sample_rate: int = Field(default=0, description="输出采样率（Hz）；0 表示未加载")
    num_speakers: int = Field(default=0, description="说话人数量（可挑选音色）")
    num_threads: int = Field(default=0, description="合成线程数（现场核对性能配置用）")
    max_num_sentences: int = Field(
        default=-1,
        description="文本切分上限。**恒为 -1** —— 设成 1 会静默丢弃第一句之后的所有内容",
    )
    voice_prompt: str | None = Field(
        default=None,
        description="Qwen3-TTS 的 voice design 描述（仅 qwen3 引擎返回）",
    )
    speed: float | None = Field(
        default=None,
        description=(
            "引擎的默认语速倍率。VITS 下来自 TTS_SPEED；"
            "cosyvoice 下来自微服务自述（默认 1.2，逐块保音高加速）。"
            "换语速最怕的就是「改了没生效还看不出来」，所以必须如实上报"
        ),
    )
    error: str = Field(default="", description="失败原因；非空表示不可用")


async def _send_json(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    """发一条 JSON 文本帧；连接已断时返回 False 而不是抛异常。

    为什么吞掉异常：客户端挂断（关标签页 / 切路由）是**正常路径**。
    让它把合成 worker 炸掉，会连带丢掉本已算好的音频与日志。
    """
    try:
        await websocket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


async def _send_bytes(websocket: WebSocket, data: bytes) -> bool:
    """发一块二进制 PCM；连接已断时返回 False"""
    try:
        await websocket.send_bytes(data)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


@router.websocket("/ws")
async def tts_stream(websocket: WebSocket) -> None:
    """流式语音合成的 WebSocket 入口。

    两个协程分工：
        * ``receiver`` —— 只负责读控制帧、入队，**从不阻塞在合成上**
          （否则 `cancel` 在合成期间收不到，打断就失效了）；
        * ``worker``   —— 串行取任务、丢线程池合成、把 PCM 发回去。

    详见模块 docstring 的帧协议。
    """
    await websocket.accept()

    engine = create_engine()
    if not engine.available:
        await _send_json(websocket, {
            "type": "status",
            "available": False,
            "engine": TtsEngine.UNAVAILABLE.value,
            "error": engine.load_error or "TTS 引擎不可用",
        })
        # 1011 = internal error：明确告诉前端"服务端能力缺失"，而不是让它以为连上了
        with contextlib.suppress(Exception):
            await websocket.close(code=1011)
        return

    default_sid, default_speed = _voice_preferences()
    await _send_json(websocket, {
        "type": "status",
        "available": True,
        "engine": engine.engine.value,
        "sample_rate": engine.sample_rate,
        "num_speakers": engine.num_speakers,
        "num_threads": engine.num_threads,
        "voice_prompt": getattr(engine, "voice_prompt", None),
        # 让"当前到底用的哪个音色/语速"可见 —— 换音色最怕的就是改了没生效还看不出来
        "sid": default_sid if engine.engine is not TtsEngine.QWEN3 else None,
        "speed": default_speed,
        "error": "",
    })

    loop = asyncio.get_running_loop()
    #: 待合成任务：``(seq, text, voice, sid, speed)``。
    #: sid/speed 必须随任务一起走 —— 只放在闭包里会被后一条 say 覆盖。
    jobs: asyncio.Queue[tuple[int, str, str | None, int, float] | None] = asyncio.Queue()
    #: 已被取消的 seq。用 set 而不是给任务打标记：cancel 可能在任务还没出队时到达。
    cancelled: set[int] = set()

    async def worker() -> None:
        """串行合成循环（唯一写二进制帧的地方）"""
        while True:
            item = await jobs.get()
            if item is None:
                return
            seq, text, voice, sid, speed = item

            if seq in cancelled:
                await _send_json(websocket, {"type": "cancelled", "seq": seq})
                continue

            if not await _send_json(websocket, {
                "type": "start", "seq": seq, "sample_rate": engine.sample_rate,
            }):
                return

            # Qwen3 引擎用流式合成：回调在合成中途多次触发，需要用 Queue
            # + call_soon_threadsafe 实时推送，而不是等合成完再一次性发。
            # VITS 引擎仍用旧逻辑（回调只触发一次）。
            is_streaming = hasattr(engine, "synthesize_stream")
            if is_streaming:
                # 用 Queue 做合成线程与 WebSocket 发送协程的桥梁。
                #
                # ⚠️ 这里**不要再写** `loop = asyncio.get_running_loop()`：那会让 `loop`
                # 在 worker() 里变成局部名，于是下面 VITS 分支（本 if 之外）读它时抛
                # `UnboundLocalError: cannot access local variable 'loop'` —— 表现为
                # TTS 一路返回 `{"type":"error","seq":N,"message":"合成服务内部错误"}`、
                # 一个音频字节都不发，而服务日志之外看不出异常（2026-09-27 实测复现）。
                # 外层会话初始化时已经取过一次，直接复用。
                chunk_q: asyncio.Queue[bytes | None] = asyncio.Queue()

                def _on_chunk_stream(samples: np.ndarray) -> int:
                    if seq in cancelled:
                        return 1
                    pcm = float_to_pcm16(samples)
                    loop.call_soon_threadsafe(chunk_q.put_nowait, pcm)
                    return 0

                synth_kwargs = {"seq": seq, "sid": sid, "speed": speed, "on_chunk": _on_chunk_stream}
                if voice is not None and hasattr(engine, "voice_prompt"):
                    synth_kwargs["voice"] = voice

                # 启动合成任务（不 await，让它跑在后台）
                synth_task = loop.run_in_executor(
                    _TTS_POOL,
                    functools.partial(engine.synthesize_stream, text, **synth_kwargs),
                )

                # 边合成边发送：从 Queue 取块并推送，直到收到 None（合成结束）
                try:
                    while True:
                        try:
                            chunk = await asyncio.wait_for(chunk_q.get(), timeout=0.1)
                        except asyncio.TimeoutError:
                            # 检查合成任务是否已完成
                            if synth_task.done():
                                break
                            continue
                        if chunk is None:
                            break
                        # 发送二进制帧
                        try:
                            await websocket.send_bytes(chunk)
                        except Exception:
                            # 客户端断开，取消合成
                            cancelled.add(seq)
                            synth_task.cancel()
                            return
                finally:
                    # 确保合成任务完成
                    if not synth_task.done():
                        with contextlib.suppress(Exception):
                            await synth_task

                # 获取合成结果
                try:
                    result = await synth_task
                except Exception:  # noqa: BLE001
                    logger.exception("[TTS:ws] 流式合成任务异常（seq=%s）", seq)
                    await _send_json(websocket, {
                        "type": "error", "seq": seq, "message": "合成服务内部错误",
                    })
                    continue

                if seq in cancelled or result.cancelled:
                    await _send_json(websocket, {"type": "cancelled", "seq": seq})
                    continue

                if result.error:
                    await _send_json(websocket, {
                        "type": "error", "seq": seq, "message": result.error,
                    })
                    continue

                # 发送 end 帧
                await _send_json(websocket, {
                    "type": "end",
                    "seq": seq,
                    "spoken_text": result.spoken_text,
                    "sample_rate": result.sample_rate,
                    "audio_ms": round(result.audio_ms, 1),
                    "first_chunk_ms": round(result.first_chunk_ms, 1),
                    "synth_ms": round(result.synth_ms, 1),
                    "rtf": round(result.synth_ms / result.audio_ms if result.audio_ms > 0 else 0.0, 3),
                    "cancelled": False,
                })

            else:
                # VITS 引擎：旧逻辑（回调只触发一次，合成完再发）
                chunks: list[bytes] = []

                def _on_chunk(samples: np.ndarray) -> int:
                    if seq in cancelled:
                        return 1
                    chunks.append(float_to_pcm16(samples))
                    return 0

                # VITS 分支：sid（音色）与 speed（语速）由此生效 ——
                # 这两个值来自 TTS_SID / TTS_SPEED（见 _voice_preferences），
                # 也可被 `say` 控制帧逐句覆盖。
                synth_kwargs = {"seq": seq, "sid": sid, "speed": speed, "on_chunk": _on_chunk}
                if voice is not None and hasattr(engine, "voice_prompt"):
                    synth_kwargs["voice"] = voice

                try:
                    result = await loop.run_in_executor(
                        _TTS_POOL,
                        functools.partial(engine.synthesize, text, **synth_kwargs),
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("[TTS:ws] 合成任务异常（seq=%s）", seq)
                    await _send_json(websocket, {
                        "type": "error", "seq": seq, "message": "合成服务内部错误",
                    })
                    continue

                if seq in cancelled or result.cancelled:
                    await _send_json(websocket, {"type": "cancelled", "seq": seq})
                    continue

                if result.error:
                    await _send_json(websocket, {
                        "type": "error", "seq": seq, "message": result.error,
                    })
                    continue

                if result.is_silent:
                    # 合成"成功"但没产出音频 —— 留痕后照常回 end，让前端知道这一句没声音，
                    # 而不是让它一直等（静默失效必须可见）。
                    logger.warning("[TTS:ws] seq=%s 合成结果为空音频：%s", seq, text[:40])

                for buf in chunks:
                    if not await _send_bytes(websocket, buf):
                        return

                await _send_json(websocket, {
                    "type": "end",
                    "seq": seq,
                    # spoken_text 是**替换术语之后**真正被念出来的文本。
                    # 前端做文本级自回声过滤必须用这个值，不能用它自己发上来的原文。
                    "spoken_text": result.spoken_text,
                    "sample_rate": result.sample_rate,
                    "audio_ms": round(result.audio_ms, 1),
                    "first_chunk_ms": round(result.first_chunk_ms, 1),
                    "synth_ms": round(result.synth_ms, 1),
                    "rtf": round(result.rtf, 3),
                    "cancelled": False,
                })

    async def receiver() -> None:
        """只读控制帧，绝不阻塞在合成上"""
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                return

            text = message.get("text")
            if not text:
                continue

            try:
                ctrl = json.loads(text)
            except json.JSONDecodeError:
                await _send_json(websocket, {"type": "error", "message": "控制帧不是合法 JSON"})
                continue
            if not isinstance(ctrl, dict):
                await _send_json(websocket, {"type": "error", "message": "控制帧必须是 JSON 对象"})
                continue

            action = ctrl.get("type")

            if action == "say":
                seq = int(ctrl.get("seq") or 0)
                prepared = prepare_for_speech(str(ctrl.get("text") or ""))
                # voice 是 Qwen3 的 voice design 临时覆盖（空字符串视为未指定）；
                # VITS 引擎会忽略这个字段。前端不需要改 —— 不传就是 None。
                voice_raw = ctrl.get("voice")
                voice = str(voice_raw).strip() if voice_raw else None
                # sid / speed：调用方可逐句覆盖（不传就用 TTS_SID / TTS_SPEED 的默认值）
                try:
                    sid = int(ctrl.get("sid", default_sid))
                except (TypeError, ValueError):
                    sid = default_sid
                try:
                    speed = float(ctrl.get("speed", default_speed))
                except (TypeError, ValueError):
                    speed = default_speed
                if not prepared.strip():
                    # 空文本是正常的（闸门可能只放行了标点）：立刻回 end，别让前端干等
                    await _send_json(websocket, {
                        "type": "end", "seq": seq, "spoken_text": "",
                        "sample_rate": engine.sample_rate, "audio_ms": 0.0,
                        "first_chunk_ms": 0.0, "synth_ms": 0.0, "rtf": 0.0,
                        "cancelled": False,
                    })
                    continue
                await jobs.put((seq, prepared, voice, sid, speed))

            elif action == "cancel":
                seq = int(ctrl.get("seq") or 0)
                cancelled.add(seq)
                # 清空**还没开始合成**的同一个 seq —— 否则取消后它照样会被合成出来
                cancelled_items: list[tuple[int, str, str | None, int, float] | None] = []
                while not jobs.empty():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        cancelled_items.append(jobs.get_nowait())
                for item in cancelled_items:
                    if item is not None and item[0] != seq:
                        jobs.put_nowait(item)

            elif action == "close":
                return

            else:
                await _send_json(websocket, {
                    "type": "error", "message": f"未知控制指令：{action}",
                })

    recv_task = asyncio.create_task(receiver(), name="tts-receiver")
    work_task = asyncio.create_task(worker(), name="tts-worker")

    try:
        await recv_task
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 —— 单条连接出错不应影响服务
        logger.exception("[TTS:ws] 接收循环异常")
    finally:
        # 让 worker 把队列里剩下的做完再退出；超时就放弃（客户端已经走了）
        with contextlib.suppress(Exception):
            await jobs.put(None)
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(work_task, timeout=_DRAIN_TIMEOUT_S)
        for task in (recv_task, work_task):
            if not task.done():
                task.cancel()
        if not work_task.done():
            with contextlib.suppress(Exception):
                await work_task


@router.get("/status", response_model=TtsStatusResponse)
def status() -> TtsStatusResponse:
    """查询本地 TTS 状态。

    **服务的自述状态才是事实** —— 本项目已经因为"服务起来了但某能力静默失效"
    坏过一次（8001 用错解释器：``/health`` 照常 200，只有 ``/asr/ws`` 404）。
    所以这里把 ``num_threads`` / ``model_dir`` / ``num_speakers`` 一并报出：
    现场出问题时，"用了几个线程"和"加载的是哪个目录"是最常见的两个根因。
    """
    info = create_engine().status()
    return TtsStatusResponse(
        available=bool(info["available"]),
        engine=str(info["engine"]),
        model_dir=info["model_dir"],  # type: ignore[arg-type]
        model_name=info["model_name"],  # type: ignore[arg-type]
        sample_rate=int(info["sample_rate"]),  # type: ignore[arg-type]
        num_speakers=int(info["num_speakers"]),  # type: ignore[arg-type]
        num_threads=int(info["num_threads"]),  # type: ignore[arg-type]
        max_num_sentences=int(info["max_num_sentences"]),  # type: ignore[arg-type]
        voice_prompt=info.get("voice_prompt"),  # type: ignore[arg-type]
        # Qwen3 从不报 speed（它没这个参数），这里用 get 保持三条引擎都安全
        speed=info.get("speed"),  # type: ignore[arg-type]
        error=str(info["error"]),
    )
