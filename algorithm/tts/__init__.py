# -*- coding: utf-8 -*-
"""本地语音合成（TTS）—— 陪伴通话的出声端

模块划分：
    ``engine``            sherpa-onnx VITS 单例（懒加载、失败原因如实上报）
    ``qwen3_engine``      Qwen3-TTS 单例（对话级韵律，voice design + 可选参考音频）
    ``cosyvoice_engine``  CosyVoice3 **微服务客户端**（零样本克隆，HTTP 流式）
    ``pronunciation``     送合成前的术语替换（词表不含拉丁字母，缩写会被吞掉）
    ``cosyvoice_service`` CosyVoice3 服务本体（**独立 venv/进程**，不是本包依赖）

引擎切换：
    通过环境变量 ``TTS_BACKEND`` 选择：
        * ``vits``（默认）—— 本地离线，零 GPU 占用，适合兜底与联调
        * ``qwen3``        —— Qwen3-TTS VoiceDesign 1.7B，对话级韵律，需 GPU 显存
        * ``cosyvoice``    —— CosyVoice3 零样本克隆，跑在独立微服务里（默认 8002）
    也可在 ``say`` 控制帧里用 ``voice`` 字段临时覆盖 voice prompt（仅 qwen3）。

    三个引擎**对外接口同构**（``available`` / ``load_error`` / ``engine`` /
    ``sample_rate`` / ``status()`` / ``synthesize_stream()``），
    所以 ``api/tts.py`` 不区分它是本地还是远程。

音频通道见 ``api/tts.py`` 的 ``/api/v1/tts/ws``。

为什么单独成一个包而不是塞进 ``asr/``：两者的模型、依赖与失败模式都独立 ——
ASR 缺模型不影响出声，反之亦然。``/tts/status`` 与 ``/asr/status`` 分开上报，
现场才能一眼看出是哪一端坏了。
"""
from tts.engine import (
    MAX_TEXT_CHARS,
    MODEL_NAME,
    VitsTtsEngine,
    default_num_threads,
    resolve_tts_model_dir,
)
from tts.pronunciation import prepare_for_speech, unpronounceable_spans
from tts.qwen3_engine import (
    Qwen3TtsEngine,
    create_engine,
    float_to_pcm16,
)

__all__ = [
    "VitsTtsEngine",
    "Qwen3TtsEngine",
    "create_engine",
    "float_to_pcm16",
    "resolve_tts_model_dir",
    "default_num_threads",
    "MODEL_NAME",
    "MAX_TEXT_CHARS",
    "prepare_for_speech",
    "unpronounceable_spans",
]
