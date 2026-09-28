# -*- coding: utf-8 -*-
"""本地语音识别模块 —— sherpa-onnx（中文，CPU）

三段流水线，全部在本机 CPU 上完成：
    streaming.py     流式 zipformer（25.8 MB）—— 边说边出字，~700ms 出首条中间结果
    engine.py        离线 Paraformer（78 MB）—— 定稿时复识别纠错，更准
    punctuation.py   标点 CT-Transformer（294 MB）—— 定稿时补标点

对外暴露三个进程级单例；跨模块的数据结构统一定义在 `shared/dataclasses.py`。
"""
from asr.engine import (
    MODEL_NAME,
    REQUIRED_FILES,
    SAMPLE_RATE,
    ParaformerRecognizer,
    decode_wav,
    resolve_model_dir,
)
from asr.punctuation import (
    PUNCT_MODEL_NAME,
    PunctuationRestorer,
    resolve_punct_model_dir,
)
from asr.streaming import (
    STREAMING_MODEL_NAME,
    STREAMING_REQUIRED_FILES,
    StreamingRecognizer,
    StreamingSession,
    resolve_streaming_model_dir,
)

__all__ = [
    # 离线整段
    "ParaformerRecognizer",
    "decode_wav",
    "resolve_model_dir",
    "MODEL_NAME",
    "REQUIRED_FILES",
    "SAMPLE_RATE",
    # 流式
    "StreamingRecognizer",
    "StreamingSession",
    "resolve_streaming_model_dir",
    "STREAMING_MODEL_NAME",
    "STREAMING_REQUIRED_FILES",
    # 标点
    "PunctuationRestorer",
    "resolve_punct_model_dir",
    "PUNCT_MODEL_NAME",
]
