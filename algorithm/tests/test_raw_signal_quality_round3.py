"""原始信号质量的受控故障和类型边界测试，不作为真人质量检出率。"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from data_quality.raw_signals import assess_audio_signal, assess_av_alignment, assess_visual_signal


def tone() -> np.ndarray:
    """生成固定非削波波形供信号契约测试。"""
    return (0.2 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.float32)


def textured_frames() -> np.ndarray:
    """固定种子的纹理帧不包含情绪或身份真值。"""
    return np.random.default_rng(42).integers(10, 245, (8, 128, 128, 3), dtype=np.uint8)


def test_normalized_audio_and_integer_pcm_have_consistent_rms() -> None:
    expected = assess_audio_signal(tone(), 16000)
    actual = assess_audio_signal((tone() * 32767).astype(np.int16), 16000)
    assert expected.usable and actual.usable
    assert actual.metrics["rms"] == pytest.approx(expected.metrics["rms"], abs=1e-4)


def test_silence_is_not_a_valid_emotion_observation() -> None:
    result = assess_audio_signal(np.zeros(16000, dtype=float), 16000)
    assert not result.usable
    assert result.quality_score == 0
    assert "near_silent_audio" in result.reason_codes


def test_clipped_audio_is_downweighted() -> None:
    clean = assess_audio_signal(tone(), 16000)
    damaged = assess_audio_signal(np.clip(tone() * 20, -1, 1), 16000)
    assert "clipping_proxy" in damaged.reason_codes
    assert damaged.quality_score < clean.quality_score


def test_white_noise_proxy_is_explicit_not_a_true_snr() -> None:
    noise = np.random.default_rng(42).normal(0, 0.15, 16000)
    result = assess_audio_signal(noise, 16000)
    assert "noise_like_spectrum_proxy" in result.reason_codes
    assert "spectral_flatness_not_true_snr_or_speech_detection" in result.limitations


@pytest.mark.parametrize("samples,rate", [(np.ones(10), True), (np.ones(10), 0), (np.ones(10) * 2, 16000), (np.ones((2, 3, 4)), 16000), (np.ones(4, dtype=bool), 16000)])
def test_invalid_audio_input_is_rejected(samples: np.ndarray, rate: int) -> None:
    with pytest.raises(ValueError):
        assess_audio_signal(samples, rate)


def test_missing_or_nonfinite_audio_is_unusable() -> None:
    assert not assess_audio_signal(np.array([]), 16000).usable
    assert not assess_audio_signal(np.array([np.nan]), 16000).usable


def test_black_frames_have_exposure_and_blur_proxies() -> None:
    result = assess_visual_signal(np.zeros((8, 64, 64, 3), dtype=np.uint8))
    assert {"underexposure_proxy", "low_contrast_proxy", "blur_proxy"} <= set(result.reason_codes)
    assert result.quality_score < 1


def test_real_temporal_metadata_is_checked() -> None:
    result = assess_visual_signal(textured_frames(), timestamps=list(np.arange(8, dtype=float)), face_present=[True] * 8)
    assert result.usable
    assert "frame_timing_unverified" not in result.reason_codes
    broken = assess_visual_signal(textured_frames(), timestamps=[0.0] * 8)
    assert not broken.usable
    assert "non_monotonic_frame_timestamps" in broken.reason_codes


def test_absent_face_metadata_is_not_fabricated_as_face_presence() -> None:
    unknown = assess_visual_signal(textured_frames())
    assert "target_face_presence_unverified" in unknown.reason_codes
    known_absent = assess_visual_signal(textured_frames(), face_present=[False] * 8)
    assert not known_absent.usable


def test_gaussian_blur_changes_observable_sharpness() -> None:
    clean = textured_frames()
    blurred = np.stack([cv2.GaussianBlur(frame, (31, 31), 8) for frame in clean])
    assert assess_visual_signal(blurred).metrics["laplacian_variance_128px"] < assess_visual_signal(clean).metrics["laplacian_variance_128px"]


def test_repeated_frames_are_proxy_not_identity_or_truthfulness_claim() -> None:
    repeated = np.repeat(textured_frames()[:1], 8, axis=0)
    result = assess_visual_signal(repeated)
    assert "repeated_frames_proxy" in result.reason_codes
    assert "repeated_frames_can_be_legitimate_static_scene" in result.limitations


@pytest.mark.parametrize("kwargs", [{"timestamps": [True] * 8}, {"timestamps": ["0"] * 8}, {"timestamps": [float("nan")] * 8}, {"face_present": [1] * 8}, {"face_present": [True]}])
def test_bad_frame_metadata_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        assess_visual_signal(textured_frames(), **kwargs)


def test_container_consistency_does_not_claim_lip_sync() -> None:
    result = assess_av_alignment(1.0, 1.0)
    assert result.usable
    assert "av_start_offset_unverified" in result.reason_codes
    assert "container_metadata_consistency_only_not_lip_sync_ground_truth" in result.limitations


def test_delay_and_duration_faults_are_observable() -> None:
    assert not assess_av_alignment(1.0, 3.0).usable
    assert not assess_av_alignment(1.0, 1.0, audio_start=1.0, video_start=0.0).usable


@pytest.mark.parametrize("args,kwargs", [((True, 1.0), {}), ((float("nan"), 1.0), {}), ((1.0, 1.0), {"audio_start": "1", "video_start": 0.0}), ((1.0, 1.0), {"relative_duration_tolerance": 1.1})])
def test_bad_alignment_metadata_is_rejected(args: tuple[float, float], kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        assess_av_alignment(*args, **kwargs)
