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

所以这里的做法是：**流式出声 + 逐块保音高加速**。
每一块音频一到达就立刻做相位声码器时间伸缩（先重采样抬音高、再伸缩降回来，
净效果是只改速度不改音高），而不是攒到最后。

实测（RTX 5060 Laptop 8 GB，33 字文本）::

    流式 speed=1.0        首块 3.19s  音频 11.80s  4.41 字/秒
    流式 speed=1.2        首块 2.06s  音频 11.44s  4.55 字/秒（speed 没真正生效）
    非流式 speed=1.2      首块 8.39s  音频  9.80s  5.31 字/秒（快但等太久）
    流式+逐块加速         首块 3.46s  音频  9.83s  5.29 字/秒  ← 本服务的默认行为

伸缩算子耗时只占音频时长的 0.7%（4 s 音频约 15 ms），完全追得上播放。
音色与语调经客观验证未变：半音 SD 4.72 vs 基准 4.69、半音跨度 14.91 vs 14.74。

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
#: 参考音：仓库内自带，保证"音色"这一交付物可复现
PROMPT_WAV = Path(os.environ.get("COSYVOICE_PROMPT_WAV", str(HERE / "prompt_qwen3_voice.wav")))
#: 参考音对应的文本。**必须与参考音实际内容逐字一致**，否则克隆质量显著下降。
#: 内容是下面这句（参考音 = 这句话的 0–5.6 s 录音）：
#:     你复习了一个月还考得比上次差，这种感觉确实挺让人难受的。
#: 前缀 ``You are a helpful assistant.<|endofprompt|>`` 是 CosyVoice3 的硬性要求 ——
#: ``llm.py`` 里有 ``assert 151646 in prompt_text``，缺了这个标记直接抛异常。
PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>"
    "你复习了一个月还考得比上次差，这种感觉确实挺让人难受的。"
)
#: 默认语速倍率。1.2 是实测"语速够用且音色不受损"的档位。
DEFAULT_SPEED = float(os.environ.get("COSYVOICE_SPEED", "1.2"))
#: 单次合成的文本上限（与算法侧 MAX_TEXT_CHARS 对齐）
MAX_TEXT_CHARS = 200
#: CosyVoice 流式产出的初始块粒度（token 数）。
#:
#: ⚠️ **这是本服务首块延迟最重要的一个参数**，不要把默认值 25 当成"够用"。
#: 它决定模型产出的音频块有多大，而块太大有两个后果：
#:   1. 首块要等更久才凑够；
#:   2. 收敛慢 —— 默认 25 下模型产出是 1.24/2.0/**4.0/4.0**/0.56 秒，
#:      第三块要等到 6.05 s 才开始生成。
#:
#: 实测（RTX 5060 Laptop 8 GB，33 字文本，逐块保音高伸缩到 1.2×）::
#:     hop=25（默认）  交付首块 3.20 s  音频 9.83 s  5.29 字/秒  F0 218.2
#:     hop=10          交付首块 1.58 s  音频 9.53 s  5.45 字/秒  F0 218.2  ← 采用
#:     hop=5           交付首块 1.43 s  音频 9.80 s  5.31 字/秒  F0 218.2
#:     hop=3           崩（flow 的 pre_lookahead 越界，ZeroDivisionError）
#:
#: 取 10 而不是 5：首块只差 0.15 s，但语速更接近目标、RTF 更低。
STREAM_FIRST_HOP = int(os.environ.get("COSYVOICE_FIRST_HOP", "10"))

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

            try:
                import torch  # noqa: PLC0415
                from cosyvoice.cli.cosyvoice import AutoModel  # noqa: PLC0415

                _install_half_loader()
                self._model = AutoModel(model_dir=str(model_dir), fp16=True)
                self._sample_rate = int(self._model.sample_rate)
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

    def _warmup(self) -> None:
        """预热两个"首次调用"的一次性开销。**不做这一步，每个进程的第一句都会慢一倍。**

        实测（RTX 5060 Laptop 8 GB，首块内部拆解）::

            未预热：model_ms=2698  stretch_ms=1692   → 首块交付 4.39 s
            已预热：期望回到模型自身水位 ~1.58 s

        两笔开销互相独立，都要预热：

        1. ``pitch_safe_resample`` —— 第一次调用要初始化 librosa 的 STFT/相位声码缓存，
           实测一个 5760 样本的块要 1692 ms（之后的同类调用只需几毫秒）。
        2. 第一次流式合成 —— 要建 CUDA 上下文之外的各种惰性结构、暖热 kernel。
           用一句极短文本跑一遍即可。

        预热失败**不影响可用性**（只是慢一点），所以吞掉异常只记日志。
        """
        import numpy as np  # noqa: PLC0415

        # 1) 伸缩算子
        try:
            t = time.perf_counter()
            probe = np.zeros(self._sample_rate // 4, dtype=np.float32)
            probe[::100] = 0.05  # 非零，避免相位声码器走空分支
            pitch_safe_resample(probe, self._sample_rate, DEFAULT_SPEED)
            logger.info("[TTS:cosyvoice] 伸缩算子预热完成 %.0f ms", (time.perf_counter() - t) * 1000)
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
                        stream=True, speed=1.0, text_frontend=False):
                    t_model = time.perf_counter()
                    raw = item["tts_speech"].detach().cpu().numpy().reshape(-1)
                    t_stretch = time.perf_counter()
                    # 逐块立刻伸缩：不攒，攒块的时间会直接变成首块延迟
                    fast = pitch_safe_resample(raw, self._sample_rate, speed)
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
            "prompt": PROMPT_WAV.name,
            "error": self._load_error,
        }


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

    def load(self, llm_model, flow_model, hift_model):  # type: ignore[no-untyped-def]
        dev = torch.device("cuda")

        def _half(path: str) -> dict:
            state = torch.load(path, map_location="cpu", weights_only=True)
            return {k: (v.half() if torch.is_floating_point(v) else v) for k, v in state.items()}

        self.llm.load_state_dict(_half(llm_model), strict=True)
        self.llm.to(dev).half().eval()
        torch.cuda.empty_cache()
        self.flow.load_state_dict(_half(flow_model), strict=True)
        self.flow.to(dev).half().eval()
        _pin_spk_dtype(self.flow.spk_embed_affine_layer)
        torch.cuda.empty_cache()
        hift_state = {k.replace("generator.", ""): v for k, v in _half(hift_model).items()}
        # hift 的 istft 等缓冲区必须留 fp32，否则波形失真
        hift_state = {k: (v.float() if "stft" in k or "window" in k else v)
                      for k, v in hift_state.items()}
        self.hift.load_state_dict(hift_state, strict=True)
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
