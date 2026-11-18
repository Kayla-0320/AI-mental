"""生产路径音频质量门控回归测试。

覆盖两件事：
1. 近静音输入必须弃权（audio_risk_prob=None），不能给"0 风险"这种假证据；
2. 干净信号必须保持原流程不变（门控不能把正常音频也拦掉）。
"""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from perception.perception_service import PerceptionService


def _write_wav(path: Path, samples: list[float], sample_rate: int = 16000) -> None:
    """把 [-1,1] 浮点样本写成 16bit 单声道 WAV（不引入额外依赖）。"""
    frames = b"".join(
        struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767)) for value in samples
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)


@pytest.fixture()
def service() -> PerceptionService:
    return PerceptionService()


def test_near_silent_audio_abstains(tmp_path: Path, service: PerceptionService) -> None:
    """全零音频应判为质量不足并弃权，而不是返回 0 风险。"""
    path = tmp_path / "silence.wav"
    _write_wav(path, [0.0] * 16000)

    result = service.analyze_audio(str(path))

    assert result.audio_risk_prob is None
    assert result.confidence == pytest.approx(0.2)
    assert any("语音质量不足" in item for item in result.evidence)


def test_clean_tone_keeps_original_pipeline(tmp_path: Path, service: PerceptionService) -> None:
    """幅度正常的信号不应被质量门控拦下，原流程照旧。"""
    path = tmp_path / "tone.wav"
    _write_wav(path, [0.3 * math.sin(2 * math.pi * 220 * i / 16000) for i in range(16000)])

    result = service.analyze_audio(str(path))

    assert result.audio_risk_prob is not None
    assert not any("语音质量不足" in item for item in result.evidence)


def test_missing_file_still_raises(tmp_path: Path, service: PerceptionService) -> None:
    """质量门控不得改变"文件不存在就抛错"的既有契约。"""
    with pytest.raises(FileNotFoundError):
        service.analyze_audio(str(tmp_path / "absent.wav"))
