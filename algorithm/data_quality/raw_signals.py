"""从原始信号计算可观察的质量代理，不用情绪一致性判断观测好坏。"""
from __future__ import annotations

import math
from numbers import Real

import cv2
import numpy as np

from shared import RawSignalQuality


def _positive_real(value: object, name: str, *, allow_zero: bool = False) -> float:
    """验证配置和时间量，拒绝布尔、字符串及非有限值。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} 必须为有限数值")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0 or (converted == 0 and not allow_zero):
        raise ValueError(f"{name} 必须为合法非负数值")
    return converted


def assess_audio_signal(samples: np.ndarray, sample_rate: int) -> RawSignalQuality:
    """测量归一化PCM的静音、削波和频谱平坦度；不声称得到真实SNR或语音金标。"""
    if type(sample_rate) is not int or not 1000 <= sample_rate <= 192000:
        raise ValueError("sample_rate 必须为1000至192000的整数")
    values = np.asarray(samples)
    if values.ndim not in (1, 2) or values.dtype.kind not in "fiu":
        raise ValueError("audio 必须为一维mono或二维[time,channels]的数值PCM")
    if values.ndim == 2 and not 1 <= values.shape[1] <= 8:
        raise ValueError("audio声道数量必须在1至8之间")
    channels = values.shape[1] if values.ndim == 2 else 1
    limits = ["heuristic_observable_proxies_not_human_quality_gold", "spectral_flatness_not_true_snr_or_speech_detection"]
    if not values.size:
        return RawSignalQuality("audio", False, 0.0, {"duration_seconds": 0.0}, ["missing_audio"], limits)
    if not np.isfinite(values).all():
        return RawSignalQuality("audio", False, 0.0, {"non_finite_fraction": float((~np.isfinite(values)).mean())}, ["non_finite_audio"], limits)
    if values.dtype.kind in "iu":
        if values.dtype.kind == "u":
            info = np.iinfo(values.dtype)
            values = (values.astype(np.float64) - (info.max + 1) / 2) / ((info.max + 1) / 2)
        else:
            values = values.astype(np.float64) / float(max(abs(np.iinfo(values.dtype).min), np.iinfo(values.dtype).max))
    else:
        values = values.astype(np.float64)
    if np.max(np.abs(values)) > 1.000001:
        raise ValueError("float audio必须是[-1,1]归一化PCM；不能将特征向量当原始波形")
    mono = values.mean(axis=1) if values.ndim == 2 else values
    rms = float(np.sqrt(np.mean(values ** 2)))
    clipping = float((np.abs(values) >= 0.999).mean())
    dc = float(abs(mono.mean()))
    frame_size = max(1, int(sample_rate * 0.02))
    frame_rms = [float(np.sqrt(np.mean(mono[start:start + frame_size] ** 2))) for start in range(0, len(mono), frame_size)]
    silence = float(np.mean(np.asarray(frame_rms) < 1e-4))
    # 代理统计使用确定性、等距子样本上限，避免无限长度FFT占用。
    segment = mono[: min(len(mono), sample_rate * 2)]
    if len(segment) >= 64 and np.std(segment) > 1e-8:
        power = np.abs(np.fft.rfft((segment - segment.mean()) * np.hanning(len(segment)))) ** 2
        power = power[1:] + 1e-18
        flatness = float(np.exp(np.log(power).mean()) / power.mean())
    else:
        flatness = 0.0
    metrics = {"duration_seconds": len(mono) / sample_rate, "channels": float(channels),
               "rms": rms, "peak_absolute": float(np.max(np.abs(values))), "clipping_fraction": clipping,
               "silence_frame_fraction": silence, "absolute_dc_offset": dc, "spectral_flatness_proxy": flatness}
    reasons: list[str] = []
    usable = rms >= 1e-4
    score = 1.0
    if not usable:
        reasons.append("near_silent_audio")
        score = 0.0
    if clipping > 0.01:
        reasons.append("clipping_proxy")
        score *= max(0.05, 1.0 - clipping)
    if dc > 0.1:
        reasons.append("dc_offset_proxy")
        score *= 0.5
    if flatness > 0.45:
        reasons.append("noise_like_spectrum_proxy")
        score *= 0.5
    if len(mono) / sample_rate < 0.1:
        reasons.append("short_audio_segment")
        score *= 0.5
    return RawSignalQuality("audio", usable, float(np.clip(score, 0, 1)), metrics, reasons, limits)


def assess_visual_signal(
    frames: np.ndarray, *, timestamps: list[float] | None = None,
    face_present: list[bool] | None = None,
) -> RawSignalQuality:
    """测量RGB视频的曝光、对比度、清晰度及重复帧；人脸存在仅消费显式外部证据。"""
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] != 3 or values.dtype.kind not in "fiu":
        raise ValueError("frames必须为[N,H,W,3] RGB数值数组")
    if not 1 <= values.shape[1] <= 4096 or not 1 <= values.shape[2] <= 4096 or len(values) > 1024:
        raise ValueError("frames的数量和分辨率超出有界研究范围")
    limits = ["heuristic_observable_proxies_not_human_quality_gold", "repeated_frames_can_be_legitimate_static_scene", "not_emotion_or_truthfulness_measure"]
    if not len(values):
        return RawSignalQuality("visual", False, 0.0, {"frame_count": 0.0}, ["missing_video_frames"], limits)
    if not np.isfinite(values).all():
        return RawSignalQuality("visual", False, 0.0, {"non_finite_fraction": float((~np.isfinite(values)).mean())}, ["non_finite_frames"], limits)
    if values.min() < 0 or values.max() > 255:
        raise ValueError("frames强度必须为0至255；不能将VLM特征当原始RGB")
    reasons: list[str] = []
    usable, score = True, 1.0
    if timestamps is None:
        reasons.append("frame_timing_unverified")
    else:
        if len(timestamps) != len(values):
            raise ValueError("timestamps必须与frames逐一对应")
        if any(isinstance(t, bool) or not isinstance(t, Real) or not math.isfinite(t) or t < 0 for t in timestamps):
            raise ValueError("timestamps必须为有限非负数")
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            usable, score = False, 0.0
            reasons.append("non_monotonic_frame_timestamps")
    luminance, contrast, sharpness, fingerprints = [], [], [], []
    for frame in values:
        resized = cv2.resize(frame.astype(np.uint8), (128, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        luminance.append(float(gray.mean()))
        contrast.append(float(gray.std()))
        sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        fingerprints.append(gray.tobytes())
    dark_fraction = float((np.asarray(luminance) < 10).mean())
    bright_fraction = float((np.asarray(luminance) > 245).mean())
    low_contrast = float((np.asarray(contrast) < 5).mean())
    blur_fraction = float((np.asarray(sharpness) < 20).mean())
    repeated_fraction = 1.0 - len(set(fingerprints)) / len(values)
    metrics = {"frame_count": float(len(values)), "luminance_mean": float(np.mean(luminance)),
               "contrast_mean": float(np.mean(contrast)), "laplacian_variance_128px": float(np.mean(sharpness)),
               "dark_frame_fraction": dark_fraction, "overexposed_frame_fraction": bright_fraction,
               "low_contrast_frame_fraction": low_contrast, "blur_proxy_fraction": blur_fraction,
               "exact_repeated_frame_fraction": repeated_fraction}
    for fraction, code in ((dark_fraction, "underexposure_proxy"), (bright_fraction, "overexposure_proxy"),
                           (low_contrast, "low_contrast_proxy"), (blur_fraction, "blur_proxy")):
        if fraction > 0.5:
            reasons.append(code)
            score *= 0.5
    if repeated_fraction > 0.8:
        reasons.append("repeated_frames_proxy")
        score *= 0.8
    if face_present is None:
        reasons.append("target_face_presence_unverified")
    else:
        if len(face_present) != len(values) or any(type(flag) is not bool for flag in face_present):
            raise ValueError("face_present必须为逐帧严格bool列表")
        metrics["confirmed_face_fraction"] = float(np.mean(face_present))
        if not any(face_present):
            usable, score = False, 0.0
            reasons.append("no_confirmed_target_face")
    return RawSignalQuality("visual", usable, float(np.clip(score, 0, 1)), metrics, reasons, limits)


def assess_av_alignment(
    audio_duration: float, video_duration: float, *, audio_start: float | None = None,
    video_start: float | None = None, maximum_offset_seconds: float = 0.25,
    relative_duration_tolerance: float = 0.1,
) -> RawSignalQuality:
    """核验容器时间元数据；它不等价于口型与语音内容已经同步。"""
    ad = _positive_real(audio_duration, "audio_duration")
    vd = _positive_real(video_duration, "video_duration")
    maximum = _positive_real(maximum_offset_seconds, "maximum_offset_seconds", allow_zero=True)
    relative = _positive_real(relative_duration_tolerance, "relative_duration_tolerance", allow_zero=True)
    if relative > 1:
        raise ValueError("relative_duration_tolerance必须在[0,1]")
    delta = abs(ad - vd)
    metrics = {"audio_duration_seconds": ad, "video_duration_seconds": vd, "absolute_duration_difference": delta}
    reasons: list[str] = []
    usable = delta <= max(maximum, relative * max(ad, vd))
    if not usable:
        reasons.append("container_duration_mismatch")
    if audio_start is None and video_start is None:
        reasons.append("av_start_offset_unverified")
    elif audio_start is None or video_start is None:
        usable = False
        reasons.append("incomplete_av_start_metadata")
    else:
        ast = _positive_real(audio_start, "audio_start", allow_zero=True)
        vst = _positive_real(video_start, "video_start", allow_zero=True)
        metrics["absolute_start_offset_seconds"] = abs(ast - vst)
        if abs(ast - vst) > maximum:
            usable = False
            reasons.append("av_start_offset_exceeded")
    return RawSignalQuality("alignment", usable, 1.0 if usable else 0.0, metrics, reasons,
                            ["container_metadata_consistency_only_not_lip_sync_ground_truth"])
