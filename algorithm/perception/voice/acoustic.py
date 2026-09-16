"""
语音声学特征提取模块

从语音信号中提取与心理健康相关的声学标志物：
    1. 基频 (F0 / pitch) —— 反映情绪唤醒度
    2. 能量 (Energy / intensity) —— 反映情绪强度
    3. 语速 (Speech rate) —— 反映焦虑/抑郁状态
    4. 共振峰 (Formants F1/F2) —— 反映发音质量（可选）
    5. 语音质量 (Jitter/Shimmer) —— 反映情绪稳定性（可选）

核心接口：
    - extract_acoustic_features(wav_path) -> AcousticFeatures
    - extract_from_buffer(audio_data, sample_rate) -> AcousticFeatures
    - features_to_emotion_risk(features) -> float (风险概率估计)

依赖：
    - librosa: 音频加载和特征提取
    - numpy: 数值计算
    - 可选：parselmouth (Praat 绑定) 用于 Jitter/Shimmer

临床依据：
    - Scherer (2003) —— 语音情感计算的声学标志物综述
    - Cummins et al. (2015) —— 语音作为抑郁 biomarker
    - Jiang et al. (2020) —— 青少年语音情感识别
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ============================================================
# 数据结构
# ============================================================

@dataclass
class AcousticFeatures:
    """声学特征向量

    Attributes:
        f0_mean: 基频均值 (Hz) —— 情绪唤醒度指标
        f0_std: 基频标准差 (Hz) —— 情绪波动性
        f0_range: 基频范围 (Hz) —— 情绪表达力
        energy_mean: 能量均值 (dB) —— 情绪强度
        energy_std: 能量标准差 (dB) —— 情绪稳定性
        speech_rate: 语速 (音节/秒) —— 焦虑/抑郁指标
        pause_ratio: 停顿比例 [0, 1] —— 犹豫/思考指标
        duration: 语音时长 (秒)
        jitter: 基频微扰 —— 情绪稳定性（可选）
        shimmer: 振幅微扰 —— 情绪稳定性（可选）
        formant_f1_mean: 第一共振峰均值 (Hz)（可选）
        formant_f2_mean: 第二共振峰均值 (Hz)（可选）
        sample_rate: 采样率 (Hz)
        valid: 特征是否有效提取
    """
    f0_mean: float = 0.0
    f0_std: float = 0.0
    f0_range: float = 0.0
    energy_mean: float = 0.0
    energy_std: float = 0.0
    speech_rate: float = 0.0
    pause_ratio: float = 0.0
    duration: float = 0.0
    jitter: float = 0.0
    shimmer: float = 0.0
    formant_f1_mean: float = 0.0
    formant_f2_mean: float = 0.0
    sample_rate: int = 16000
    valid: bool = False


# ============================================================
# 音频加载
# ============================================================

def _load_audio(wav_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """加载音频文件

    优先使用 librosa，回退到 scipy。

    Args:
        wav_path: WAV 音频文件路径
        target_sr: 目标采样率

    Returns:
        (audio_array, sample_rate)

    Raises:
        FileNotFoundError: 文件不存在
        RuntimeError: 加载失败
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"音频文件不存在: {wav_path}")

    try:
        import librosa
        audio, sr = librosa.load(wav_path, sr=target_sr, mono=True)
        return audio, sr
    except ImportError:
        pass

    # 回退到 scipy
    try:
        from scipy.io import wavfile
        sr, audio = wavfile.read(wav_path)
        # 转为 mono float
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32) / 32768.0
        # 重采样（简化：直接截断/插值）
        if sr != target_sr:
            ratio = target_sr / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio), new_len),
                np.arange(len(audio)),
                audio,
            )
        return audio, target_sr
    except ImportError:
        raise RuntimeError("需要安装 librosa 或 scipy 来加载音频")


# ============================================================
# 特征提取
# ============================================================

def extract_acoustic_features(
    wav_path: str,
    target_sr: int = 16000,
) -> AcousticFeatures:
    """从 WAV 文件提取声学特征

    Args:
        wav_path: WAV 音频文件路径
        target_sr: 目标采样率

    Returns:
        AcousticFeatures: 声学特征向量
    """
    audio, sr = _load_audio(wav_path, target_sr)
    return _extract_features_from_audio(audio, sr)


def extract_from_buffer(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
) -> AcousticFeatures:
    """从音频缓冲区提取声学特征

    Args:
        audio_data: 音频数据 (float32, mono)
        sample_rate: 采样率

    Returns:
        AcousticFeatures: 声学特征向量
    """
    return _extract_features_from_audio(audio_data, sample_rate)


def _extract_features_from_audio(
    audio: np.ndarray,
    sr: int,
) -> AcousticFeatures:
    """核心特征提取逻辑

    Args:
        audio: 音频数据
        sr: 采样率

    Returns:
        AcousticFeatures
    """
    features = AcousticFeatures(sample_rate=sr)

    if len(audio) < sr * 0.5:  # 至少 0.5 秒
        return features

    try:
        import librosa

        # --------------------------------------------------
        # 1. 基频 (F0) —— 使用 librosa.pyin
        # --------------------------------------------------
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),  # ~65 Hz
            fmax=librosa.note_to_hz("C6"),  # ~1047 Hz
            sr=sr,
        )

        # 过滤无效 F0（NaN）
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            features.f0_mean = round(float(np.mean(valid_f0)), 2)
            features.f0_std = round(float(np.std(valid_f0)), 2)
            features.f0_range = round(float(np.max(valid_f0) - np.min(valid_f0)), 2)

        # --------------------------------------------------
        # 2. 能量 (RMS Energy)
        # --------------------------------------------------
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        # 转换为 dB
        rms_db = librosa.amplitude_to_db(rms + 1e-10)
        features.energy_mean = round(float(np.mean(rms_db)), 2)
        features.energy_std = round(float(np.std(rms_db)), 2)

        # --------------------------------------------------
        # 3. 语速估计 —— 基于过零率和能量包络
        # --------------------------------------------------
        # 简化语速估计：使用短时能量包络的峰值率
        hop_length = 512
        frame_duration = hop_length / sr  # 每帧时长（秒）
        energy_envelope = librosa.feature.rms(
            y=audio, frame_length=2048, hop_length=hop_length
        )[0]

        # 检测能量峰值（代表音节/字）
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(
            energy_envelope,
            height=np.mean(energy_envelope) * 0.5,
            distance=int(0.1 / frame_duration),  # 最小间隔 100ms
        )

        duration_seconds = len(audio) / sr
        features.duration = round(duration_seconds, 2)

        if duration_seconds > 0:
            # 语速 = 峰值数 / 时长（近似音节/秒）
            features.speech_rate = round(len(peaks) / duration_seconds, 2)

        # --------------------------------------------------
        # 4. 停顿比例 —— 低能量帧占比
        # --------------------------------------------------
        energy_threshold = np.mean(energy_envelope) * 0.3
        silent_frames = np.sum(energy_envelope < energy_threshold)
        total_frames = len(energy_envelope)
        features.pause_ratio = round(float(silent_frames / total_frames), 3) if total_frames > 0 else 0.0

        # --------------------------------------------------
        # 5. 共振峰 (Formants) —— 可选，需要 LPC
        # --------------------------------------------------
        try:
            f1, f2 = _estimate_formants(audio, sr)
            features.formant_f1_mean = round(f1, 1)
            features.formant_f2_mean = round(f2, 1)
        except Exception:
            pass  # 共振峰提取失败不影响其他特征

        features.valid = True

    except ImportError:
        # librosa 不可用时，使用纯 numpy 简化提取
        features = _extract_features_numpy(audio, sr)

    return features


def _estimate_formants(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """估计前两个共振峰 (F1, F2)

    使用简化的 LPC (线性预测编码) 方法。

    Args:
        audio: 音频数据
        sr: 采样率

    Returns:
        (F1_mean, F2_mean) 单位 Hz
    """
    from scipy.signal import lpc as scipy_lpc
    from scipy.signal import tf2zpk

    # 分帧处理
    frame_length = 400  # 25ms @ 16kHz
    hop_length = 160    # 10ms
    n_frames = (len(audio) - frame_length) // hop_length

    if n_frames < 1:
        return 500.0, 1500.0  # 默认值

    f1_values = []
    f2_values = []

    for i in range(min(n_frames, 50)):  # 最多取 50 帧
        start = i * hop_length
        frame = audio[start:start + frame_length]

        # 加汉明窗
        window = np.hamming(len(frame))
        frame = frame * window

        # LPC 分析（12 阶）
        try:
            lpc_order = 12
            # 使用自相关法计算 LPC 系数
            R = np.correlate(frame, frame, mode="full")
            R = R[len(frame) - 1:]  # 取正延迟部分

            # Levinson-Durbin 递推
            a = np.zeros(lpc_order + 1)
            a[0] = 1.0
            E = R[0]

            for k in range(lpc_order):
                if E < 1e-10:
                    break
                lam = -np.sum(a[1:k + 1] * R[k::-1]) / E
                a_new = a.copy()
                a_new[k + 1] = lam
                for j in range(1, k + 1):
                    a_new[j] = a[j] + lam * a[k + 1 - j]
                a = a_new
                E = E * (1 - lam ** 2)

            # 从 LPC 系数计算共振峰
            # 求根
            roots = np.roots(a)
            # 只保留单位圆内的根
            roots = roots[np.abs(roots) < 1.0]

            # 计算频率
            angles = np.angle(roots)
            freqs = np.abs(angles) * sr / (2 * np.pi)

            # 只保留 0-5000 Hz 范围内的共振峰
            freqs = freqs[(freqs > 50) & (freqs < 5000)]
            freqs = np.sort(freqs)

            if len(freqs) >= 2:
                f1_values.append(freqs[0])
                f2_values.append(freqs[1])
        except Exception:
            continue

    if f1_values and f2_values:
        return float(np.mean(f1_values)), float(np.mean(f2_values))

    return 500.0, 1500.0  # 默认值


def _extract_features_numpy(
    audio: np.ndarray,
    sr: int,
) -> AcousticFeatures:
    """纯 numpy 简化特征提取（无 librosa 依赖）

    当 librosa 不可用时使用，提取基础声学特征。

    Args:
        audio: 音频数据
        sr: 采样率

    Returns:
        AcousticFeatures
    """
    features = AcousticFeatures(sample_rate=sr)
    duration = len(audio) / sr

    if duration < 0.5:
        return features

    # 基频估计：使用自相关法
    # 取中间段音频（避免开头/结尾静音）
    mid_start = int(len(audio) * 0.2)
    mid_end = int(len(audio) * 0.8)
    mid_audio = audio[mid_start:mid_end]

    # 自相关法估计基频
    min_lag = int(sr / 500)   # 500 Hz 上限
    max_lag = int(sr / 65)    # 65 Hz 下限
    autocorr = np.correlate(mid_audio, mid_audio, mode="full")
    autocorr = autocorr[len(mid_audio) - 1:]  # 正延迟

    if len(autocorr) > max_lag:
        search_region = autocorr[min_lag:max_lag]
        if len(search_region) > 0:
            best_lag = np.argmax(search_region) + min_lag
            if best_lag > 0:
                features.f0_mean = round(sr / best_lag, 2)

    # 能量
    rms = np.sqrt(np.mean(audio ** 2))
    features.energy_mean = round(20 * np.log10(rms + 1e-10), 2)

    # 语速：基于过零率
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio)))) / 2
    features.speech_rate = round(zero_crossings / duration / 100, 2)  # 归一化

    # 停顿比例
    frame_size = 512
    n_frames = len(audio) // frame_size
    if n_frames > 0:
        frame_energies = np.array([
            np.mean(audio[i * frame_size:(i + 1) * frame_size] ** 2)
            for i in range(n_frames)
        ])
        threshold = np.mean(frame_energies) * 0.3
        silent = np.sum(frame_energies < threshold)
        features.pause_ratio = round(silent / n_frames, 3)

    features.duration = round(duration, 2)
    features.valid = True

    return features


# ============================================================
# 声学特征 → 情绪风险概率
# ============================================================

def features_to_emotion_risk(
    features: AcousticFeatures,
    weights: Optional[dict] = None,
) -> float:
    """将声学特征映射为情绪风险概率

    基于文献中的经验规则：
    - 低基频 + 低能量 → 抑郁倾向
    - 高基频 + 高语速 + 高能量 → 焦虑倾向
    - 高停顿比例 → 犹豫/不确定

    Args:
        features: 声学特征
        weights: 自定义权重（可选）

    Returns:
        情绪风险概率 [0.0, 1.0]
    """
    if not features.valid:
        return 0.0

    # 默认权重
    if weights is None:
        weights = {
            "low_f0": 0.15,       # 低基频（抑郁指标）
            "low_energy": 0.15,   # 低能量（抑郁指标）
            "high_speech_rate": 0.20,  # 高语速（焦虑指标）
            "high_pause_ratio": 0.20,  # 高停顿（犹豫指标）
            "low_f0_range": 0.15,    # 低基频范围（情感平淡）
            "high_energy_var": 0.15,  # 高能量波动（不稳定）
        }

    risk_score = 0.0

    # 1. 低基频（< 120 Hz 可能表示抑郁）
    if 80 < features.f0_mean < 120:
        risk_score += weights["low_f0"]

    # 2. 低能量（< -30 dB 可能表示低动力）
    if features.energy_mean < -30:
        risk_score += weights["low_energy"]

    # 3. 高语速（> 5 音节/秒可能表示焦虑）
    if features.speech_rate > 5.0:
        risk_score += weights["high_speech_rate"]

    # 4. 高停顿比例（> 0.4 可能表示犹豫/思维迟缓）
    if features.pause_ratio > 0.4:
        risk_score += weights["high_pause_ratio"]

    # 5. 低基频范围（< 50 Hz 可能表示情感平淡）
    if features.f0_range < 50 and features.f0_mean > 0:
        risk_score += weights["low_f0_range"]

    # 6. 高能量波动（> 10 dB 可能表示情绪不稳定）
    if features.energy_std > 10:
        risk_score += weights["high_energy_var"]

    return round(min(1.0, max(0.0, risk_score)), 4)


# ============================================================
# 特征摘要（供跨模块传递）
# ============================================================

def features_to_summary(features: AcousticFeatures) -> dict:
    """将声学特征转换为摘要字典（供跨模块传递）

    仅包含统计特征，不包含原始音频数据。

    Args:
        features: 声学特征

    Returns:
        特征摘要字典
    """
    return {
        "f0_mean": features.f0_mean,
        "f0_std": features.f0_std,
        "f0_range": features.f0_range,
        "energy_mean": features.energy_mean,
        "energy_std": features.energy_std,
        "speech_rate": features.speech_rate,
        "pause_ratio": features.pause_ratio,
        "duration": features.duration,
        "risk_prob": features_to_emotion_risk(features),
        "valid": features.valid,
    }
