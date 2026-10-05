# -*- coding: utf-8 -*-
"""CosyVoice3 语音合成微服务（独立进程 / 独立 venv）

## 为什么单独成一个服务

CosyVoice3 依赖 ``torch==2.7.0+cu128`` + ``numpy<2`` + ``transformers==4.51.3``，
与算法服务的 ``torch==2.11`` + ``numpy==2.5`` **直接冲突**，放不进同一个解释器。
所以做成独立进程，算法侧通过 HTTP/WS 调用（见 ``algorithm/tts/cosyvoice_engine.py``）。
这与项目既有的"ASR 与 TTS 分开上报"是同一条原则：依赖与失败模式独立。

## 音色

零样本克隆，参考音 = ``prompt_qwen3_voice.wav``（仓库内，262 KB）。
它是从 Qwen3-TTS 生成的一段中文里裁出的前两句（0–5.6 s / 28 字），
句界由能量包络测得。换音色只需替换这个 wav + 同步 :data:`PROMPT_TEXT`。

## 为什么默认开 1.2 倍速，以及它为什么不是简单设 speed

CosyVoice 的 ``speed`` 参数在**流式**调用下只作用于最后一块收尾音频 ——
``cosyvoice/cli/model.py`` 里有::

    if speed != 1.0:
        assert token_offset == 0 and finalize is True, 'speed change only support non-stream inference mode'

流式下 ``token_offset`` 已经不是 0，speed 几乎不生效（实测 1.45 只快 4%）。
而在**非流式**下虽然能正常提速到 1.2×，却必须等整段合成完才出声（实测首块 8.39 s）。

所以这里的做法是：**流式出声 + 逐块保音高加速**，而不是攒到最后。

⚠️ **伸缩算子绝不能用 librosa 的相位声码器 —— 它就是"金属味"的来源**
（2026-10-05，听感判决 + 同批样本受控对照双重定罪）。

`librosa.effects.time_stretch` 是 STFT 相位声码器：按固定帧率做相位推进，
产生 phasiness（相位涂抹），听感就是**金属/塑料感**。
受控对照（`_probe_metal_controlled.py`：同一批模型样本，输入逐比特相同，
唯一变量是伸缩算子；CPPS 是嗓音学里测"粗哑/金属"的首选，越低越糙）::

    条件                    CPPS     HNR(dB)   46.9Hz 调制   接缝 click
    不伸缩（模型裸输出）      4.651    6.82      5.7 dB        —
    相位声码器 1.2（旧）      3.952    5.33      8.6 dB       99.92%（有）
    WSOLA 1.2（现用）        4.793    6.55      7.8 dB       97%（无）

相位声码器把 CPPS 砍掉 **15%**；WSOLA 把它**恢复到与不伸缩同一水平**。
46.9 Hz 正是 `time_stretch` 默认 `hop=512 @24k` 的帧率，那根窄带嗡声
就是电子感的来源。用户听感判决：**"不伸缩的 wav 没有金属音，1.2x 就有"**
—— 与上表完全一致。

WSOLA（波形相似度叠加）走时域：为每一帧找一个"波形最像上一帧自然延续"的位置
再做交叠相加，**完全不做相位推进**，所以结构上不可能产生帧率嗡声。
逐块调用也不掉指标（CPPS 4.793 vs 整段一次 4.853），接缝无 click。

耗时与相位声码器同级：**7.28 s 音频 26 ms → RTF 0.0035**，完全追得上播放。
参数扫描（20/30/40 ms 帧 × ±5/10/15 ms 搜索）中 30 ms/±10 ms 最优。

实测（RTX 5060 Laptop 8 GB，33 字文本）::

    流式 speed=1.0        首块 3.19s  音频 11.80s  4.41 字/秒
    流式 speed=1.2        首块 2.06s  音频 11.44s  4.55 字/秒（speed 没真正生效）
    非流式 speed=1.2      首块 8.39s  音频  9.80s  5.31 字/秒（快但等太久）
    流式+逐块加速         首块 3.46s  音频  9.83s  5.29 字/秒  ← 本服务的默认行为

回滚开关：设 ``COSYVOICE_SPEED_ALGO=librosa`` 可退回旧相位声码器，**仅供对照**。

## 帧协议

对外只提供 **HTTP 流式**（``POST /tts/stream``）与健康检查，不再提供 WebSocket：

    POST /tts/stream
      body: {"text": "你复习了一个月还考得比上次差，", "speed": 1.2}
      响应: Content-Type: audio/pcm
            X-Sample-Rate: 24000
            X-Speed: 1.2
            body = int16 小端单声道 PCM，**分块流式**（Transfer-Encoding: chunked）

为什么用 HTTP 而不是 WS：算法侧是**同步**调用（跑在线程池里），
HTTP 分块流用 ``urllib`` 就能边收边处理，不需要额外维护一条 asyncio 事件循环
再往同步线程里桥接。而且本项目既有的 WS 契约（``/tts/ws``）本来就是
"客户端发文本、服务端回 PCM"，方向是单向的，不需要双工。

## 启动

    cd "AI-mental-main (2)/AI-mental-main/AI-mental-main/algorithm/tts/cosyvoice_service"

    # 模型权重与运行环境（见 README.md）
    set COSYVOICE_ROOT=D:\\Develop\\AIC\\_tts_candidates\\cosyvoice
    %COSYVOICE_ROOT%\\.venv\\Scripts\\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8002
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

# ── 配置（全部可用环境变量覆盖）─────────────────────────────────────────────
#: CosyVoice 仓库与模型权重所在根目录。默认取仓库外层的实验目录，
#: 生产部署时应显式设置成权重实际落盘位置。
DEFAULT_ROOT = Path(
    os.environ.get("COSYVOICE_ROOT", r"D:\Develop\AIC\_tts_candidates\cosyvoice")
)
#: 模型目录名（Fun-CosyVoice3-0.5B）
MODEL_DIRNAME = os.environ.get("COSYVOICE_MODEL", "Fun-CosyVoice3-0.5B")
#: 参考音：仓库内自带，保证"音色"这一交付物可复现（**用户提供的真人录音**）。
#:
#: ⚠️ **参考音用"真人录音"还是"合成语音"，是音色质量的第一决定因素。**
#: 实测（2026-10-05，同一目标文本 / 同一 seed / hop=25 / 不伸缩，指标为频谱平坦度，
#: 越低越不像噪声、越干净）::
#:
#:     合成参考音（Qwen3-TTS 生成，5.6 s）      0.03133   ← 金属味明显（旧默认）
#:     真人录音（用户提供，6.5 s）              0.01130   −64%   ← 现用
#:     真人录音（用户提供，20.3 s）             0.01355   −57%
#:     真人录音（用户提供，27.0 s）             0.01311   −58%
#:     真人录音（CosyVoice 官方 demo）          0.01198   −62%
#:
#: 原因：合成参考音里本来就有声码器伪影，零样本克隆会**把它放大**（实测约 3 倍：
#: 参考 0.54% → 输出 1.60%）；真人录音不会放大（0.12% → 0.14%）。
#: 所以听到"很浓的机器人金属味"时，**先怀疑参考音是不是合成的，而不是去调模型参数**。
#:
#: ⚠️ **参考音不是越长越好**。取 6.5 s 而不是 20–27 s，是因为 RTF 会随长度明显恶化
#: （prompt speech token 全部进 LLM 上下文，越长逐 token 越慢）。实测：
#:
#:     参考音    首块     RTF      音质(flat)
#:     6.50 s   1.39 s   0.830    0.01130   ← 三项同时最好
#:     12.20 s  1.75 s   0.840    0.02008
#:     18.20 s  1.92 s   0.975    0.01876
#:     20.30 s  2.14 s   1.111    0.01355
#:     27.00 s  2.03 s   1.620    0.01311   ← RTF 1.62，播放会断粮
#:
#: RTF < 1 才追得上播放。6.5 s 这档 RTF 0.83，比旧的合成参考音（1.27）**还快**。
#: 音质不随长度改善（flatness 非单调），所以没有理由取更长的参考音。
#:
#: ⚠️ 录音授权：使用真人录音做声音克隆**必须取得说话人本人授权**。
#: 换参考音时同时更新 :data:`PROMPT_TEXT`，且**必须逐字一致**。
#:
#: ⚠️ **参考音必须降噪，否则底噪会被一起克隆成"背景杂音"**（2026-10-05 实测）：
#: 用户原始录音 SNR 仅 17.2 dB、底噪 −30 dBFS，且噪声集中在 500 Hz 以下
#: （房间隆隆声/桌面震动，**不是**工频哼声 —— 50 Hz 突出度只有 +1.0 dB）。
#: 用录音自带的静音段估计噪声谱做**谱减**（over-subtraction 1.2、谱底 8%，
#: 见 ``_probe_denoise_ref.py``），SNR 17.2 → 23.3 dB，而克隆输出的
#: **句内音节间隙电平从 −33.5 dB 降到 −44.9 dB（−11 dB）**，flatness 再降 34%。
#:
#:     参考音            句内最低帧 P5   flatness   RTF
#:     原始（未降噪）       −33.5 dB     0.01130   0.834
#:     谱减降噪             −44.9 dB     0.00743   0.717   ← 现用
#:     谱减 + 70 Hz 高通    −41.4 dB     0.00932   0.710   （高通反而更差，不用）
#:
#: 判据不是"参考音听起来干不干净"，而是**合成输出的句内间隙电平** ——
#: 往句间插人工静音会把"最低帧"测成数字静音（−240 dB），掩盖真实底噪。
PROMPT_WAV = Path(os.environ.get(
    "COSYVOICE_PROMPT_WAV", str(HERE / "prompt_user_recording.wav")))
#: 参考音对应的文本。**必须与参考音实际内容逐字一致**，否则克隆质量显著下降。
#:
#: 当前参考音 = ``prompt_user_recording.wav``（**6.50 s**，用户提供的真人录音），
#: 内容是下面这 2 行（参考音就是这段录音的开头，切在第 2 行后的停顿处）：
#:
#:     长江东流浪花淘尽英雄
#:     岳阳楼的风掠过三国江面
#:
#: ⚠️ 源录音全长 36 s（另外还有 5 行），但我们**只取前 6.5 s** —— 理由见
#: :data:`PROMPT_WAV` 的长度取舍实测。若将来要改回更长的参考音，
#: 必须同步改这里，并重新量 RTF。
#:
#: 前缀 ``You are a helpful assistant.<|endofprompt|>`` 是 CosyVoice3 的硬性要求 ——
#: ``llm.py`` 里有 ``assert 151646 in prompt_text``，缺了这个标记直接抛异常。
PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>"
    "长江东流浪花淘尽英雄"
    "岳阳楼的风掠过三国江面"
)
#: 零样本说话人缓存 id。用 ``add_zero_shot_spk`` 把参考音的语音 token 与说话人
#: embedding **预先算一次并缓存**，之后每次合成都直接复用。
#:
#: ⚠️ 不缓存极贵：``frontend.py`` 在 ``zero_shot_spk_id == ''`` 时**每次合成**都要
#: 对参考音重跑 ``_extract_speech_feat`` / ``_extract_speech_token``(ONNX) /
#: ``_extract_spk_embedding``(ONNX) 三遍。参考音 5.6 s -> 27 s 后实测
#: RTF 1.27 -> **3.07**（通话不可用）；缓存后归零。
ZERO_SHOT_SPK_ID = os.environ.get("COSYVOICE_SPK_ID", "aic_companion")
#: 默认语速倍率。1.2 是实测"语速够用且音色不受损"的档位。
DEFAULT_SPEED = float(os.environ.get("COSYVOICE_SPEED", "1.2"))
#: 单次合成的文本上限（与算法侧 MAX_TEXT_CHARS 对齐）
MAX_TEXT_CHARS = 200
#: CosyVoice 流式产出的初始块粒度（token 数）。默认取 **模型自己的 25**。
#:
#: ⚠️ **这个参数是首块延迟与音质之间的直接取舍**，两边都实测过。
#:
#: 压低它能让第一句更快出声，代价是**声码器每次拿到的上下文变短**：
#:
#:     固定种子实测（同一句话、同一 seed，音频总长都是 3.08 s，唯一变量是 hop）::
#:
#:         hop   首块     块数   最小块长    边界跳变倍数
#:          8    1.74 s    4     0.32 s      0.743
#:         10    1.62 s    4     0.24 s      1.446     ← 曾经采用
#:         15    2.03 s    3     0.84 s      6.119
#:         25    2.13 s    2     1.24 s      0.091     ← 现在（模型默认）
#:
#: 关键那一列是**最小块长**：hop=10 时声码器第一块只有 0.24 s，hop=25 是 1.24 s，
#: 差 5 倍。HiFi-GAN 上下文越短，边界相位越难接上 —— 现场听到的就是
#: "金属味 + 不连续"。而代价只有 **+0.5 s** 首块延迟，这笔账明显划算。
#:
#: （"边界跳变倍数"这一列不单调，因为它还取决于边界恰好落在语音还是静音上，
#: 只能当参考，不能单看。最小块长才是稳定的判据。）
#:
#: 历史教训：早期版本只看"首块 3.20 → 1.58 s"就选了 hop=10，没有量音质代价，
#: 结果用户听到的是金属味的断续语音 —— **延迟指标好看不等于体验好**。
#:
#: 需要更快的首块时可以设 COSYVOICE_FIRST_HOP=10 换回来，但要知道这是在拿音质换。
STREAM_FIRST_HOP = int(os.environ.get("COSYVOICE_FIRST_HOP", "25"))

#: 相位声码器/重采样依赖 librosa；延迟导入，缺依赖时报错更清楚
_librosa: Any = None


def _load_librosa() -> Any:
    """延迟导入 librosa（只在真正需要加速时导入）"""
    global _librosa
    if _librosa is None:
        import librosa  # noqa: PLC0415

        _librosa = librosa
    return _librosa


def pitch_safe_resample(x: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """保音高地把音频时长变为 ``1/rate``。

    做法：先用相位声码器把时长压缩到 ``1/rate``（音高不变，这正是 time_stretch
    的语义），再用高质量重采样把长度精确校正到目标值。

    Args:
        x: float32 单声道波形。
        sr: 采样率（仅用于校验，函数内不重采样到别的采样率）。
        rate: 速度倍率，>1 表示变快。

    Returns:
        np.ndarray: 加速后的 float32 波形；``rate`` 接近 1 时原样返回。
    """
    if abs(rate - 1.0) < 1e-3 or len(x) < 2:
        return np.asarray(x, dtype=np.float32)

    librosa = _load_librosa()
    stretched = librosa.effects.time_stretch(np.asarray(x, dtype=np.float32), rate=rate)
    target_len = int(round(len(x) / rate))
    if target_len > 1 and len(stretched) > 1 and len(stretched) != target_len:
        # soxr_hq 是 librosa 里质量最高、速度也够的选择
        stretched = librosa.resample(
            stretched, orig_sr=len(stretched), target_sr=target_len, res_type="soxr_hq"
        )
    return np.asarray(stretched, dtype=np.float32)


#: 变速算子的实现：``wsola``（默认）或 ``librosa``（旧相位声码器，**仅用于对照**）。
#:
#: ⚠️ 不要退回 librosa —— 它是 STFT 相位声码器，会给语音加上"金属味"
#: （听感已判定 + CPPS 掉 15%，见模块 docstring 的对照表）。
SPEED_ALGO = os.environ.get("COSYVOICE_SPEED_ALGO", "wsola").strip().lower()
#: WSOLA 帧长 / 相似度搜索半径（毫秒）。30/±10 是参数扫描的最优档。
WSOLA_FRAME_MS = float(os.environ.get("COSYVOICE_WSOLA_FRAME_MS", "30"))
WSOLA_SEARCH_MS = float(os.environ.get("COSYVOICE_WSOLA_SEARCH_MS", "10"))


def wsola_time_stretch(
    x: np.ndarray,
    sr: int,
    rate: float,
    frame_ms: float = WSOLA_FRAME_MS,
    search_ms: float = WSOLA_SEARCH_MS,
) -> np.ndarray:
    """WSOLA（波形相似度叠加）保音高变速 —— 替代相位声码器，消除"金属味"。

    原理：把输入切成互相交叠的短帧，输出帧按 ``hs = frame/2`` 紧密排列；
    分析位置按 ``ha = hs*rate`` 前进。每取一帧时，在目标位置附近 ``±search_ms``
    内找一个**与上一帧的"自然延续"最相似**（归一化互相关最大）的位置，
    再送入交叠相加。全程只做波形搬运，**不做相位推进** ——
    因此不会产生相位声码器那种固定帧率的相位涂抹（听感=金属/电子）。

    Args:
        x: float32 单声道波形。
        sr: 采样率（Hz）。
        rate: 速度倍率，>1 表示变快。
        frame_ms: 分析帧长（毫秒）。
        search_ms: 相似度搜索半径（毫秒）。

    Returns:
        np.ndarray: 变速后的 float32 波形，长度对齐到 ``round(len(x)/rate)``；
        输入过短或 ``rate≈1`` 时原样返回（调用方按块调用，短块不该被拉伸）。
    """
    if abs(rate - 1.0) < 1e-3 or len(x) < 2048:
        return np.asarray(x, dtype=np.float32)

    sig = np.asarray(x, dtype=np.float32)
    n = int(sr * frame_ms / 1000.0)
    n -= n % 2                                   # 保证偶数，便于半窗对齐
    if n < 64 or len(sig) < n:
        return sig
    hs = n // 2                                  # 合成跳距（50% 交叠）
    ha = int(round(hs * rate))                   # 分析跳距
    if ha <= 0:
        return sig
    delta = int(sr * search_ms / 1000.0)
    tlen = hs                                    # 模板长度 = 合成跳距
    win = np.hanning(n).astype(np.float32)
    target = int(round(len(sig) / rate))
    if target < 2:
        return sig

    # 第一遍：为每一帧挑一个"最像自然延续"的输入位置
    picks: list[tuple[int, int]] = []
    k = 0
    prev = -1
    while k * hs < target:
        ai = int(round(k * ha))
        if ai + n > len(sig):                    # 末尾几帧贴着尾巴取，铺满 target
            ai = len(sig) - n
        if ai < 0:
            break
        if prev < 0 or delta <= 0:
            idx = ai
        else:
            lo = max(0, ai - delta)
            hi = min(len(sig) - n, ai + delta)
            tmpl = sig[prev + hs: prev + hs + tlen]
            if hi < lo or len(tmpl) < tlen:
                idx = ai
            else:
                seg = sig[lo: hi + n]
                corr = np.correlate(seg, tmpl, "valid")
                energy = np.correlate(seg.astype(np.float64) ** 2,
                                      np.ones(tlen), "valid")
                if len(corr) == 0:
                    idx = ai
                else:
                    ncc = corr / np.sqrt(
                        energy * float(np.dot(tmpl, tmpl)) + 1e-12)
                    idx = lo + int(np.argmax(ncc))
                    if idx + n > len(sig):
                        idx = len(sig) - n
        picks.append((k * hs, idx))
        prev = idx
        k += 1

    if not picks:
        return sig

    # 第二遍：交叠相加。**首帧不淡入、末帧不淡出** ——
    # 否则逐块调用时每块两端各留半个窗长的掉音（实测接缝凹口 16~28 dB）。
    out = np.zeros(target + n, dtype=np.float64)
    wsum = np.zeros_like(out)
    last = len(picks) - 1
    for i, (o, idx) in enumerate(picks):
        if o + n > len(out):
            break
        w = win
        if i == 0 or i == last:
            w = win.copy()
            if i == 0:
                w[:hs] = 1.0
            if i == last:
                w[hs:] = 1.0
        out[o:o + n] += sig[idx:idx + n] * w
        wsum[o:o + n] += w

    valid = wsum > 1e-6
    out[valid] /= wsum[valid]
    y = out[:target]
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return np.asarray(y, dtype=np.float32)


def speed_up(x: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """按 :data:`SPEED_ALGO` 选择变速算子（生产入口）。

    Args:
        x: float32 单声道波形。
        sr: 采样率（Hz）。
        rate: 速度倍率。

    Returns:
        np.ndarray: 变速后的 float32 波形。
    """
    if SPEED_ALGO == "librosa":
        return pitch_safe_resample(x, sr, rate)
    return wsola_time_stretch(x, sr, rate)


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 [-1,1] → int16 小端字节。

    先 clip 再乘，避免超范围样本在 astype 时回绕成刺耳噪声。
    与前端 ``ttsPlayer.ts`` 的 ``floatToInt16`` 互为逆运算。
    """
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class CosyVoice3Engine:
    """CosyVoice3 单例（懒加载、失败原因记在 ``load_error``，不抛异常）

    进程内只允许一个实例：模型占显存约 1.65 GiB，且 ``AutoModel`` 的
    ``tts()`` 内部用 ``self.lock`` 保护会话字典，多实例只会重复占显存。
    """

    _instance: Optional["CosyVoice3Engine"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Any = None
        self._load_error: str = ""
        self._sample_rate: int = 24000
        self._synth_lock = threading.Lock()
        #: 只在进程内第一次合成时打一次细粒度计时（避免刷日志）
        self._timed_once: bool = False
        #: 参考音特征缓存 id；空串表示"每次合成都重算参考音"（慢，但正确）
        self._spk_id: str = ""

    @classmethod
    def instance(cls) -> "CosyVoice3Engine":
        """取全局单例（双检锁）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def available(self) -> bool:
        """是否已加载可用（会触发懒加载）"""
        return self.load()

    @property
    def load_error(self) -> str:
        """最近一次加载失败原因；空串表示无失败"""
        return self._load_error

    @property
    def sample_rate(self) -> int:
        """输出采样率（Hz）"""
        return self._sample_rate

    @property
    def model_dir(self) -> Optional[str]:
        """实际加载到的模型目录"""
        return str(DEFAULT_ROOT / "pretrained_models" / MODEL_DIRNAME)

    def load(self) -> bool:
        """懒加载模型与参考音特征；失败原因写进 :attr:`load_error`。

        三类失败要分开报，因为它们对应的处置完全不同：
            路径不对（改 COSYVOICE_ROOT）／参考音缺失（补 wav）／模型加载异常（看日志）。
        """
        if self._model is not None:
            return True

        with self._lock:
            if self._model is not None:
                return True

            model_dir = DEFAULT_ROOT / "pretrained_models" / MODEL_DIRNAME
            repo_dir = DEFAULT_ROOT / "CosyVoice"
            if not (repo_dir / "cosyvoice" / "cli" / "cosyvoice.py").is_file():
                self._load_error = (
                    f"找不到 CosyVoice 仓库：{repo_dir}。"
                    f"请设置 COSYVOICE_ROOT 指向含 CosyVoice/ 与 pretrained_models/ 的目录"
                )
                logger.warning("[TTS:cosyvoice] %s", self._load_error)
                return False
            if not model_dir.is_dir():
                self._load_error = f"找不到模型目录：{model_dir}"
                logger.warning("[TTS:cosyvoice] %s", self._load_error)
                return False
            if not PROMPT_WAV.is_file():
                self._load_error = f"找不到参考音：{PROMPT_WAV}"
                logger.warning("[TTS:cosyvoice] %s", self._load_error)
                return False

            # 仓库与 Matcha-TTS 都要进 sys.path（后者是 flow matching 的依赖）
            for p in (str(repo_dir), str(repo_dir / "third_party" / "Matcha-TTS")):
                if p not in sys.path:
                    sys.path.insert(0, p)

            # 加载前预检可提交内存：不够就明确拒绝，别把整机一起拖垮（详见函数 docstring）
            # 阈值与 scripts/run-tts.mjs 的 MIN_COMMIT_MB 保持一致，避免两处判断不同步。
            #
            # ⚠️ 注意检查点位置：本检查在 `import torch` **之后**（torch+onnxruntime
            #    光导入就要吃 ~1.4 GB，整个进程起来约 4-6 GB），所以这里读到的余量
            #    比"启动前系统余量"低好几 GB。标定依据：
            #      观测到真实崩溃时 本处读数 = 10997 MB
            #      成功加载时       本处读数 ≈ 12000 MB 以上
            #    取 12 GB 作为下限。
            headroom = _commit_headroom_mb()
            min_headroom = int(os.environ.get("COSYVOICE_MIN_COMMIT_MB", "12288") or 12288)
            if headroom is not None and headroom < min_headroom:
                self._load_error = (
                    f"可提交内存余量不足：仅 {headroom} MB，加载约需 {min_headroom} MB。"
                    f"请先关掉占用内存的程序（Docker/WSL、浏览器、聊天软件）再重试；"
                    f"根治办法是把系统页面文件改成『系统托管』——"
                    f"当前被固定为 4 GB，os error 1455「页面文件太小」就是这个原因。"
                )
                logger.warning("[TTS:cosyvoice] %s", self._load_error)
                return False
            logger.info(
                "[TTS:cosyvoice] 加载前可提交内存余量 %s MB（下限 %d MB）",
                headroom if headroom is not None else "未知", min_headroom,
            )

            try:
                import torch  # noqa: PLC0415
                from cosyvoice.cli.cosyvoice import AutoModel  # noqa: PLC0415

                _install_frugal_onnx_loader()
                _install_half_loader()
                self._model = AutoModel(model_dir=str(model_dir), fp16=True)
                self._sample_rate = int(self._model.sample_rate)
                self._spk_id = self._cache_prompt_features()
            except Exception as exc:  # noqa: BLE001 —— 加载失败必须转成状态
                self._model = None
                self._load_error = f"CosyVoice3 加载失败：{type(exc).__name__}: {exc}"
                logger.exception("[TTS:cosyvoice] %s", self._load_error)
                return False

            self._load_error = ""
            logger.info(
                "[TTS:cosyvoice] 已加载 %s（采样率 %d，参考音 %s，默认语速 %.2f）",
                MODEL_DIRNAME, self._sample_rate, PROMPT_WAV.name, DEFAULT_SPEED,
            )
            self._warmup()
            return True

    def _cache_prompt_features(self) -> str:
        """把参考音的语音 token / 说话人 embedding **预算一次并缓存**，返回 spk_id。

        为什么必须缓存（2026-10-05 实测，代价很大）：
        ``frontend.py`` 的 ``frontend_zero_shot`` 在 ``zero_shot_spk_id == ''`` 时，
        **每次合成**都要对参考音重跑三遍昂贵处理::

            _extract_speech_feat    读盘 + 特征
            _extract_speech_token   ONNX 语音 tokenizer 跑完整段参考音
            _extract_spk_embedding  campplus ONNX 跑完整段参考音

        参考音从 5.6 s 换成 27 s（真人录音，音质明显更干净）之后，实测 RTF
        从 1.27 恶化到 **3.07** —— 合成比播放慢 3 倍，通话直接不可用。
        缓存后这部分**归零**（``zero_shot_spk_id`` 非空时只做一次 dict 拷贝）。

        注意：缓存省掉的是"参考音的预处理"；参考音的 speech token 仍会进 LLM
        上下文，所以参考音越长、上下文越长、逐 token 生成仍会略慢 —— 这是
        "音质 vs 延迟"的另一半取舍，见 :data:`PROMPT_WAV` 的说明。

        Returns:
            str: 可用的 spk_id；**失败返回空串**。空串会让引擎退回"每次重算"的
            老路（慢但正确）—— 缓存失败不该让整个服务不可用。
        """
        try:
            ok = self._model.add_zero_shot_spk(
                PROMPT_TEXT, str(PROMPT_WAV), ZERO_SHOT_SPK_ID)
            if ok:
                logger.info(
                    "[TTS:cosyvoice] 参考音特征已缓存（spk_id=%r）："
                    "每次合成不再重跑参考音的 ONNX 预处理",
                    ZERO_SHOT_SPK_ID,
                )
                return ZERO_SHOT_SPK_ID
            logger.warning("[TTS:cosyvoice] add_zero_shot_spk 返回假值，退回不缓存")
        except Exception:  # noqa: BLE001 —— 缓存失败降级，不能让服务起不来
            logger.exception(
                "[TTS:cosyvoice] 参考音特征缓存失败，退回每次重算（RTF 会明显变差）")
        return ""

    def _warmup(self) -> None:
        """预热两个"首次调用"的一次性开销。**不做这一步，每个进程的第一句都会慢一倍。**

        实测（RTX 5060 Laptop 8 GB，首块内部拆解）::

            未预热：model_ms=2698  stretch_ms=1692   → 首块交付 4.39 s
            已预热：期望回到模型自身水位 ~1.58 s

        两笔开销互相独立，都要预热：

        1. 伸缩算子 —— 相位声码器（``COSYVOICE_SPEED_ALGO=librosa``）第一次调用要初始化
           librosa 的 STFT/相位声码缓存，实测一个 5760 样本的块要 1692 ms
           （之后的同类调用只需几毫秒）。现用的 WSOLA 没有这类一次性开销，
           但保留预热以便两种实现都可切换。
        2. 第一次流式合成 —— 要建 CUDA 上下文之外的各种惰性结构、暖热 kernel。
           用一句极短文本跑一遍即可。

        预热失败**不影响可用性**（只是慢一点），所以吞掉异常只记日志。
        """
        import numpy as np  # noqa: PLC0415

        # 1) 伸缩算子
        try:
            t = time.perf_counter()
            probe = np.zeros(self._sample_rate // 4, dtype=np.float32)
            probe[::100] = 0.05  # 非零，避免算子走空分支
            speed_up(probe, self._sample_rate, DEFAULT_SPEED)
            logger.info("[TTS:cosyvoice] 伸缩算子预热完成 %.0f ms（%s）",
                        (time.perf_counter() - t) * 1000, SPEED_ALGO)
        except Exception:  # noqa: BLE001 —— 预热失败只影响首句速度，不影响可用性
            logger.exception("[TTS:cosyvoice] 伸缩算子预热失败（不影响可用性）")

        # 2) 首次流式合成：用最短文本，不落盘、不交付
        try:
            t = time.perf_counter()
            inner = self._model.model
            saved_hop = inner.token_hop_len
            inner.token_hop_len = STREAM_FIRST_HOP
            try:
                for _item in self._model.inference_zero_shot(
                        "你好。", PROMPT_TEXT, str(PROMPT_WAV),
                        zero_shot_spk_id=self._spk_id,
                        stream=True, speed=1.0, text_frontend=False):
                    break  # 拿到第一块就够，不必跑完整句
            finally:
                inner.token_hop_len = saved_hop
            logger.info("[TTS:cosyvoice] 首次合成预热完成 %.0f ms", (time.perf_counter() - t) * 1000)
        except Exception:  # noqa: BLE001
            logger.exception("[TTS:cosyvoice] 首次合成预热失败（不影响可用性）")

    def synthesize_chunks(
        self,
        text: str,
        speed: float = DEFAULT_SPEED,
    ) -> Iterator[np.ndarray]:
        """流式合成，逐块产出**已加速**的 float32 波形。

        实现要点（都是实测换来的，见模块 docstring 的数据表）：

        1. **把 ``token_hop_len`` 临时降到 :data:`STREAM_FIRST_HOP`**，让模型吐小块。
           默认 25 的产出是 1.24/2.0/4.0/4.0 秒的大块，首块要 3.20 s 才能交付。
        2. **逐块伸缩，不攒块**。攒块会把"等模型产出一整块"的时间直接加到首块延迟上
           （实测攒到 2 s → 首块 6.0 s；逐块 → 1.58 s）。
           相位声码对小块的算子开销确实占比更高，但它只有几毫秒，远小于攒块的代价。
        3. 用完**必须把 ``token_hop_len`` 还原** —— 它是引擎单例上的状态，
           不还原会污染后续请求（且下一个请求的 hop 会被越改越小）。

        Args:
            text: 待合成文本（调用方保证已过安全闸门与术语替换）。
            speed: 语速倍率，保音高加速（不是 CosyVoice 的 speed 参数）。

        Yields:
            np.ndarray: float32 单声道波形块，范围 [-1, 1]。

        Raises:
            RuntimeError: 引擎未加载或文本非法（由调用方转成 HTTP 错误）。
        """
        if not self.load():
            raise RuntimeError(self._load_error or "CosyVoice3 不可用")

        stripped = (text or "").strip()
        if not stripped:
            return
        if len(stripped) > MAX_TEXT_CHARS:
            raise RuntimeError(f"文本过长（{len(stripped)} > {MAX_TEXT_CHARS} 字），调用方应先切句")

        # 串行化：CosyVoice 内部虽有 lock，但显存与线程状态仍以串行为稳
        with self._synth_lock:
            inner = self._model.model
            saved_hop = inner.token_hop_len
            inner.token_hop_len = STREAM_FIRST_HOP
            t_start = time.perf_counter()
            try:
                for item in self._model.inference_zero_shot(
                        stripped, PROMPT_TEXT, str(PROMPT_WAV),
                        zero_shot_spk_id=self._spk_id,
                        stream=True, speed=1.0, text_frontend=False):
                    t_model = time.perf_counter()
                    raw = item["tts_speech"].detach().cpu().numpy().reshape(-1)
                    t_stretch = time.perf_counter()
                    # 逐块立刻伸缩：不攒，攒块的时间会直接变成首块延迟
                    fast = speed_up(raw, self._sample_rate, speed)
                    # 只在进程内第一次合成时打一次首块拆解 —— 现场排查"首块为什么慢"
                    # 时，能一眼分清是模型慢还是伸缩慢（这两者的处置完全不同）
                    if not self._timed_once:
                        self._timed_once = True
                        logger.info(
                            "cvsvc|first_block|model_ms=%.0f|stretch_ms=%.0f|samples=%d",
                            (t_model - t_start) * 1000.0,
                            (time.perf_counter() - t_stretch) * 1000.0,
                            len(raw),
                        )
                    yield fast
            finally:
                # 引擎是单例，hop 必须还原，否则污染后续请求
                inner.token_hop_len = saved_hop

    def status(self) -> dict[str, Any]:
        """引擎状态字典（供 ``/health`` 如实上报）"""
        ok = self.load()
        return {
            "available": ok,
            "engine": "cosyvoice3",
            "model_dir": self.model_dir if ok else None,
            "model_name": MODEL_DIRNAME,
            "sample_rate": self._sample_rate,
            "speed": DEFAULT_SPEED,
            "speed_algo": SPEED_ALGO,
            "prompt": PROMPT_WAV.name,
            "error": self._load_error,
        }


def _commit_headroom_mb() -> Optional[int]:
    """当前**可提交内存**余量（MB）。非 Windows 或读取失败时返回 ``None``。

    背景（2026-10-05 实测，整机级事故）：本机页面文件被固定成 4 GB、非系统托管，
    提交上限 = 31.4 GB 物理 + 4 GB 页面文件 ≈ 35.4 GB。模型加载峰值一旦超过余量，
    后果**不止是这次加载失败** —— 失败后进程还会残留数 GB 提交（实测 7.3 GB），
    把整机推向分配失败：ONNX 报 ``bad allocation``、错误码 ``1455``
    （``ERROR_COMMITMENT_LIMIT``「页面文件太小」），同一分钟内 ``dwm.exe``、
    ``msedge.exe`` 等无关进程一起崩。现场表现就是"服务一启动就崩"，
    而且**没有任何 Python 回溯**（原生分配失败，不是异常）。

    所以加载前必须先看余量：不够就明确拒绝并给出可执行的处置，
    而不是硬上把整机一起拖垮。
    """
    try:
        import ctypes  # noqa: PLC0415

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        # ullAvailPageFile 即"剩余可提交量"（已含物理内存部分）
        return int(status.ullAvailPageFile / (1024 * 1024))
    except Exception:  # noqa: BLE001 —— 预检本身不该成为新的失败点
        return None


def _install_frugal_onnx_loader() -> None:
    """让 ONNX 会话不再预留大块内存池（BFC arena），并摘掉不可用的 CUDA provider。

    背景（2026-10-05 实测）：``cosyvoice/cli/frontend.py`` 直接构造
    ``SessionOptions``，默认开启 CPU BFC arena，按 2 的幂**成倍预留**。
    对 ``speech_tokenizer_v3.onnx``（924 MB 权重）实测提交内存开销：

        默认 BFC arena    +1990 MB
        关掉 BFC arena    +909 MB     ← 白占的 1.08 GB

    在这台提交上限只有 ~35.4 GB（31.4 GB 物理 + 固定 4 GB 页面文件）、
    余量常年不足 8 GB 的机器上，这 1 GB 就是"能加载"和
    「os error 1455 / ONNX bad allocation / 整机连 dwm.exe 一起崩」的分界线。

    ``frontend.py`` 还会按 ``torch.cuda.is_available()`` 请求
    ``CUDAExecutionProvider``，但本机 onnxruntime 只有 Azure/CPU provider ——
    请求不存在的 provider 只会打一条 warning 再回落，徒增不确定性，这里直接摘掉。

    幂等：重复调用只生效一次。
    """
    import onnxruntime as ort  # noqa: PLC0415

    if getattr(ort.InferenceSession, "_frugal_installed", False):
        return

    _original_init = ort.InferenceSession.__init__
    _available = set(ort.get_available_providers())

    def _frugal_init(  # type: ignore[no-untyped-def]
        self, path_or_bytes, sess_options=None, providers=None,
        provider_options=None, **kwargs,
    ):
        if sess_options is None:
            sess_options = ort.SessionOptions()
        # 关掉成倍预留的内存池：按需分配，峰值只算真正用到的
        sess_options.enable_cpu_mem_arena = False
        if providers:
            kept = [p for p in providers if p in _available]
            providers = kept or ["CPUExecutionProvider"]
        _original_init(
            self, path_or_bytes, sess_options=sess_options, providers=providers,
            provider_options=provider_options, **kwargs,
        )

    ort.InferenceSession.__init__ = _frugal_init
    ort.InferenceSession._frugal_installed = True  # type: ignore[attr-defined]
    logger.info(
        "[TTS:cosyvoice] ONNX 已启用精简加载（关闭 BFC arena；可用 provider=%s）",
        sorted(_available),
    )


def _install_half_loader() -> None:
    """让 CosyVoice3 以 fp16 上卡，且修掉一个会让整段静默假死的 dtype 坑。

    两个必须一起做的修改（依据 docs 里的实测记录）：

    1. **半精度加载**：``fp16=True`` 只开 autocast，**不省显存**。
       默认路径是 fp32 整份 ``.to("cuda")``（llm 1.9 G + flow 1.3 G），
       8 GB 卡上会 OOM。这里把 ``CosyVoiceModel.load`` 换成
       "CPU 读盘 → 转半精度 → 上卡"，显存占用降到 1.65 GiB。

    2. **spk 投影层 dtype 钉住**：``flow.py`` 里 ``F.normalize`` 在 autocast 下
       会产出 fp32，而 ``flow_matching.py`` 用 ``spks.dtype`` 分配 ``spks_in``
       并做 ``spks_in[0] = spks``，半精度下抛::

           RuntimeError: expand(torch.cuda.HalfTensor{[0, 80]}, size=[80])

       ⚠️ 这个异常发生在 ``llm_job`` **子线程**里，会被 threading 吞掉，
       主循环永远等不到 ``llm_end_dict`` —— 表现为"整机 CPU/GPU 都 0% 的假死"，
       没有任何 Python 回溯。所以必须给输出 80 维的 Linear 挂 forward hook，
       把输出 dtype 钉回权重 dtype。
    """
    import torch  # noqa: PLC0415
    from cosyvoice.cli import model as model_mod  # noqa: PLC0415

    if getattr(model_mod.CosyVoiceModel, "_half_loader_installed", False):
        return

    def _pin_spk_dtype(module: torch.nn.Module) -> None:
        for sub in module.modules():
            if isinstance(sub, torch.nn.Linear) and sub.out_features == 80:
                sub.register_forward_hook(lambda m, _i, o: o.to(m.weight.dtype))

    def _read_state(path: str) -> dict:
        """读盘。优先 ``mmap`` —— fp32 源权重走文件映射，而不是整份进匿名内存。

        为什么必须这样（2026-10-05 实测，代价很大）：整份读入会把**提交内存**
        吃穿。本机是 31 GB RAM + 固定 4 GB 页面文件（提交上限 ~35.4 GB），
        加载失败时进程仍残留 7.3 GB 提交，下一次尝试直接撞
        ``OSError: 页面文件太小 (os error 1455)``；严重时连 ``dwm.exe`` 都被带崩，
        表现为"一启动就崩"且没有任何 Python 回溯。

        ``mmap=True`` 让 fp32 源页可被系统随时回收，峰值只算真正新增的那份
        （llm.pt 1931 MB + flow.pt 1268 MB 不再各占一份匿名内存）。
        """
        try:
            return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except Exception:  # noqa: BLE001 —— 旧格式权重文件不支持 mmap，回退整份读
            logger.warning("[TTS:cosyvoice] %s 不支持 mmap，回退整份读取", path)
            return torch.load(path, map_location="cpu", weights_only=True)

    def _half_state(path: str) -> dict:
        return {k: (v.half() if torch.is_floating_point(v) else v)
                for k, v in _read_state(path).items()}

    def _assign(module: torch.nn.Module, path: str) -> None:
        """把一个权重文件**接管**进模块，而不是拷进去。

        ``assign=True`` 是这里的关键：默认行为是"把 state_dict 逐张拷进模块已有的
        参数"，于是同一份权重在装载瞬间存在**两份**（临时字典 + 模块参数）。
        实测（2026-10-05，提交内存分步计量）：llm 1225 MB + flow 634 MB 这一下
        就是 **~1.9 GB 的瞬时峰值**，而本机提交余量常常只有几 GB —— 正是它把加载
        推过临界点，引发 ``os error 1455`` 或原生崩溃。

        assign 之后模块直接持有这些张量，不再有第二份；临时字典 ``del`` 掉即可。

        ⚠️ 只对 llm / flow 用。hift 不能这么干：它的权重在旧路径里是"半精度拷进
        fp32 模块"（``load_state_dict`` 会把半精度上转成 fp32），最终是 fp32；
        用 assign 会让 hift 变成半精度，istft 波形可能失真。
        """
        state = _half_state(path)
        module.load_state_dict(state, strict=True, assign=True)
        del state
        gc.collect()

    def load(self, llm_model, flow_model, hift_model):  # type: ignore[no-untyped-def]
        dev = torch.device("cuda")

        # 先降精度：assign 只接管 state_dict 里有的参数，模块自带、不在权重文件里的
        # 缓冲区要靠这一步转半精度（旧代码是用加载后的 `.to(dev).half()` 做同一件事）。
        self.llm.half()
        _assign(self.llm, llm_model)
        self.llm.to(dev).eval()
        torch.cuda.empty_cache()

        self.flow.half()
        _assign(self.flow, flow_model)
        self.flow.to(dev).eval()
        _pin_spk_dtype(self.flow.spk_embed_affine_layer)
        torch.cuda.empty_cache()

        # ⚠️ hift 必须保持 fp32，且**不能**走 _half_state。
        #
        # hift 是声码器（HiFi-GAN），直接决定音色 —— 早期版本这里图省内存用了
        # _half_state，等于把声码器权重量化到 fp16 精度后再上转 fp32。
        # 实测代价（同一句话 A/B）：HF>6kHz 占比 0.79% → 1.06%（+34%）、
        # 频谱平坦度 0.02278 → 0.02664、谱质心 584 → 606 Hz —— 全部指向
        # "金属味/噪声化"，而省下的只有 ~40 MB。不值得。
        #
        # mmap 读入 fp32，直接拷进 fp32 模块（这也是上游 CosyVoice 的原始路径）。
        hift_state = {k.replace("generator.", ""): v for k, v in _read_state(hift_model).items()}
        # hift 的 istft 等缓冲区同样必须留 fp32，否则波形失真
        hift_state = {k: (v.float() if "stft" in k or "window" in k else v)
                      for k, v in hift_state.items()}
        self.hift.load_state_dict(hift_state, strict=True)
        del hift_state
        gc.collect()
        self.hift.to(dev).eval()
        torch.cuda.empty_cache()

    model_mod.CosyVoiceModel.load = load
    model_mod.CosyVoiceModel._half_loader_installed = True  # type: ignore[attr-defined]


app = FastAPI(title="CosyVoice3 TTS 微服务", version="1.0.0")


@app.get("/health")
def health() -> dict[str, Any]:
    """引擎可用性与关键配置。**服务的自述状态才是事实** —— 本项目的既有原则。"""
    return CosyVoice3Engine.instance().status()


class TtsStreamRequest(BaseModel):
    """``POST /tts/stream`` 的请求体"""
    text: str = Field(..., description="待合成文本（调用方应先切句并做术语替换）")
    speed: float = Field(
        default=DEFAULT_SPEED,
        description="语速倍率，逐块保音高加速。默认取 COSYVOICE_SPEED（1.2）",
    )


@app.post("/tts/stream")
async def tts_stream(req: TtsStreamRequest) -> StreamingResponse:
    """流式合成：边合成边以 int16 PCM 分块回推。

    与算法侧的 ``/tts/ws`` 契约对应：那边把 PCM 转成二进制帧发给浏览器，
    这边只负责"文本进、PCM 出"，把加速（1.2×）留在服务内完成。

    ⚠️ 这里刻意用 **async 生成器 + asyncio.Queue**，而不是同步生成器：
    同步生成器会被 Starlette 放进线程池逐块迭代，实测出现"攒成一批才发出"
    （58 块里 55 块在 0 ms 内到达）—— 首块延迟被放大到 5.2 s。
    改成合成跑在线程里、块经 asyncio 队列逐个 yield，才是真的逐块流式。

    ⚠️ 合成失败**在第一个字节之前**会被转成 503，让调用方立刻看到原因；
    一旦开始回推就只能中断流（HTTP 语义所限），所以把失败检查前置。
    """
    engine = CosyVoice3Engine.instance()
    if not engine.available:
        raise HTTPException(status_code=503, detail=engine.load_error or "CosyVoice3 不可用")

    speed = float(req.speed)
    if not (0.5 <= speed <= 2.0):
        speed = max(0.5, min(2.0, speed))

    t0 = time.perf_counter()
    queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stats: dict[str, float] = {"first_ms": 0.0, "delivered": 0.0, "chunks": 0}

    def _produce() -> None:
        """在线程里跑合成，把 PCM 块投进 asyncio 队列（None 作为结束哨兵）"""
        try:
            for wav in engine.synthesize_chunks(req.text, speed=speed):
                if not stats["first_ms"]:
                    stats["first_ms"] = (time.perf_counter() - t0) * 1000.0
                stats["delivered"] += len(wav)
                stats["chunks"] += 1
                loop.call_soon_threadsafe(queue.put_nowait, float_to_pcm16(wav))
        except Exception:  # noqa: BLE001 —— 流已开始，只能记日志并终止
            logger.exception("[TTS:cosyvoice] 流式合成中断（已推 %d 块）", stats["chunks"])
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def pcm_iter() -> AsyncIterator[bytes]:
        loop.run_in_executor(None, _produce)
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
        synth_ms = (time.perf_counter() - t0) * 1000.0
        audio_ms = stats["delivered"] / engine.sample_rate * 1000.0 if engine.sample_rate else 0.0
        # 结构化一行，便于直接 grep "cvsvc|" 看现场（本项目的进度条输出很难读）
        logger.info(
            "cvsvc|done|first_ms=%.0f|synth_ms=%.0f|audio_ms=%.0f|rtf=%.3f|chunks=%d|speed=%.2f",
            stats["first_ms"], synth_ms, audio_ms,
            (synth_ms / audio_ms) if audio_ms else 0.0, stats["chunks"], speed,
        )

    return StreamingResponse(
        pcm_iter(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(engine.sample_rate),
            "X-Speed": f"{speed:.2f}",
            # 关掉中间层缓冲，保证"边合成边到达"（nginx 之类需要同样配置）
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )
