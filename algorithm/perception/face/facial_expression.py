"""
面部微表情检测模块 —— 基于面部动作单元 (AU) 的情绪识别

支持两种输入源：
1. MediaPipe FaceMesh 468 关键点 + blendshapes（前端推荐）
2. 传统 68 点 landmarks（回退方案）

提取面部特征：
- 面部动作编码系统 (FACS) 的 Action Units
- 面部关键点几何特征
- 表情强度与持续时间
- 微表情瞬态检测（< 500ms）

情绪映射规则：
- AU1+AU4+AU15 → 悲伤
- AU1+AU2+AU26 → 恐惧
- AU4+AU5+AU7 → 愤怒
- AU6+AU12 → 快乐
- AU1+AU4+AU7+AU15 → 痛苦

使用方式：
    from perception.face.facial_expression import (
        FacialFeatures,
        extract_facial_features,
        extract_from_mediapipe,
        features_to_emotion_risk,
    )
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ============================================================
# 数据结构
# ============================================================

@dataclass
class FacialFeatures:
    """面部特征数据类"""
    # Action Units 强度 (0-1)
    au1_inner_brow_raiser: float = 0.0
    au2_outer_brow_raiser: float = 0.0
    au4_brow_lowerer: float = 0.0
    au5_upper_lid_raiser: float = 0.0
    au6_cheek_raiser: float = 0.0
    au7_lid_tightener: float = 0.0
    au9_nose_wrinkler: float = 0.0
    au10_upper_lip_raiser: float = 0.0
    au12_lip_corner_puller: float = 0.0
    au15_lip_corner_depressor: float = 0.0
    au17_chin_raiser: float = 0.0
    au20_lip_stretcher: float = 0.0
    au26_jaw_drop: float = 0.0

    # 几何特征
    brow_distance_ratio: float = 0.0  # 眉毛距离/脸宽
    mouth_width_ratio: float = 0.0  # 嘴宽/脸宽
    eye_aspect_ratio: float = 0.0  # 眼睛纵横比（EAR）
    mouth_open_ratio: float = 0.0  # 张嘴程度

    # 微表情特征
    micro_expression_count: int = 0  # 微表情次数
    micro_expression_duration_avg: float = 0.0  # 平均微表情持续时间 (ms)
    expression_intensity_mean: float = 0.0  # 平均表情强度
    expression_intensity_max: float = 0.0  # 最大表情强度

    # 头部姿态
    head_pitch: float = 0.0  # 俯仰角
    head_yaw: float = 0.0  # 偏航角
    head_roll: float = 0.0  # 翻滚角

    # 元数据
    frame_count: int = 0
    fps: float = 30.0
    detection_confidence: float = 0.0


@dataclass
class MicroExpressionEvent:
    """微表情事件"""
    start_frame: int = 0
    end_frame: int = 0
    duration_ms: float = 0.0
    primary_au: str = ""
    intensity: float = 0.0
    mapped_emotion: str = ""


# ============================================================
# AU 到情绪映射规则
# ============================================================

AU_EMOTION_RULES: dict[str, dict[str, Any]] = {
    "sadness": {
        "aus": ["au1_inner_brow_raiser", "au4_brow_lowerer", "au15_lip_corner_depressor"],
        "weight": 0.8,
        "description": "内眉上扬 + 眉毛下压 + 嘴角下拉",
    },
    "fear": {
        "aus": ["au1_inner_brow_raiser", "au2_outer_brow_raiser", "au26_jaw_drop"],
        "weight": 0.7,
        "description": "眉毛上扬 + 下巴下垂",
    },
    "anger": {
        "aus": ["au4_brow_lowerer", "au5_upper_lid_raiser", "au7_lid_tightener"],
        "weight": 0.75,
        "description": "眉毛下压 + 眼睑收紧",
    },
    "happiness": {
        "aus": ["au6_cheek_raiser", "au12_lip_corner_puller"],
        "weight": 0.85,
        "description": "脸颊上扬 + 嘴角上提",
    },
    "distress": {
        "aus": ["au1_inner_brow_raiser", "au4_brow_lowerer", "au7_lid_tightener", "au15_lip_corner_depressor"],
        "weight": 0.9,
        "description": "眉毛紧锁 + 眼睑收紧 + 嘴角下拉",
    },
    "surprise": {
        "aus": ["au1_inner_brow_raiser", "au2_outer_brow_raiser", "au5_upper_lid_raiser", "au26_jaw_drop"],
        "weight": 0.6,
        "description": "眉毛上扬 + 眼睛睁大 + 下巴下垂",
    },
    "disgust": {
        "aus": ["au9_nose_wrinkler", "au10_upper_lip_raiser"],
        "weight": 0.7,
        "description": "鼻子皱起 + 上唇上扬",
    },
}


# ============================================================
# 特征提取
# ============================================================

def _compute_eye_aspect_ratio(landmarks: np.ndarray, eye_indices: tuple[int, ...]) -> float:
    """计算眼睛纵横比 (EAR)

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    正常睁眼约 0.25-0.35，闭眼 < 0.15

    Args:
        landmarks: 68 点面部关键点 (68, 2)
        eye_indices: 眼睛关键点索引
    """
    if landmarks.shape[0] < 68:
        return 0.3

    pts = [landmarks[i] for i in eye_indices]
    if len(pts) < 6:
        return 0.3

    vertical1 = np.linalg.norm(pts[1] - pts[5])
    vertical2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])

    if horizontal < 1e-6:
        return 0.3

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return float(ear)


def _compute_mouth_ratio(landmarks: np.ndarray) -> tuple[float, float]:
    """计算嘴巴纵横比

    Returns:
        (mouth_width_ratio, mouth_open_ratio)
    """
    if landmarks.shape[0] < 68:
        return 0.3, 0.1

    # 嘴角: 48, 54
    mouth_width = np.linalg.norm(landmarks[48] - landmarks[54])
    # 上下唇: 51, 57
    mouth_height = np.linalg.norm(landmarks[51] - landmarks[57])
    # 脸宽: 0, 16
    face_width = np.linalg.norm(landmarks[0] - landmarks[16])

    if face_width < 1e-6:
        return 0.3, 0.1

    width_ratio = mouth_width / face_width
    open_ratio = mouth_height / max(mouth_width, 1e-6)

    return float(width_ratio), float(open_ratio)


def _compute_brow_distance(landmarks: np.ndarray) -> float:
    """计算眉毛距离比"""
    if landmarks.shape[0] < 68:
        return 0.15

    # 眉毛: 19-26, 眼睛上: 37-46
    brow_y_mean = np.mean(landmarks[19:27, 1])
    eye_y_mean = np.mean(landmarks[37:47, 1])
    face_height = np.linalg.norm(landmarks[27] - landmarks[8])

    if face_height < 1e-6:
        return 0.15

    return float(abs(brow_y_mean - eye_y_mean) / face_height)


def _estimate_head_pose(landmarks: np.ndarray) -> tuple[float, float, float]:
    """粗略估计头部姿态

    Returns:
        (pitch, yaw, roll) 角度
    """
    if landmarks.shape[0] < 68:
        return 0.0, 0.0, 0.0

    # 简化估计：基于关键点对称性
    # 偏航角：基于鼻子两侧对称性
    nose_center = landmarks[30]
    left_eye_center = np.mean(landmarks[36:42], axis=0)
    right_eye_center = np.mean(landmarks[42:48], axis=0)

    face_center = (left_eye_center + right_eye_center) / 2
    face_width = np.linalg.norm(left_eye_center - right_eye_center)

    if face_width < 1e-6:
        return 0.0, 0.0, 0.0

    yaw = (nose_center[0] - face_center[0]) / face_width * 45  # 度
    pitch = (landmarks[8, 1] - landmarks[27, 1]) / face_width * 30
    roll = np.arctan2(
        right_eye_center[1] - left_eye_center[1],
        right_eye_center[0] - left_eye_center[0],
    ) * 180 / np.pi

    return float(np.clip(pitch, -45, 45)), float(np.clip(yaw, -60, 60)), float(np.clip(roll, -30, 30))


def extract_facial_features(
    landmarks_sequence: list[np.ndarray],
    au_intensities: list[dict[str, float]] | None = None,
    fps: float = 30.0,
) -> FacialFeatures:
    """从面部关键点序列提取面部特征

    Args:
        landmarks_sequence: 每帧的面部关键点列表，每个形状 (68, 2)
        au_intensities: 每帧的 AU 强度字典列表（可选，无则从几何估计）
        fps: 帧率

    Returns:
        FacialFeatures
    """
    if not landmarks_sequence:
        return FacialFeatures()

    num_frames = len(landmarks_sequence)

    # 几何特征（取平均）
    ear_values = []
    mouth_widths = []
    mouth_opens = []
    brow_dists = []
    pitches, yaws, rolls = [], [], []

    for lm in landmarks_sequence:
        ear = _compute_eye_aspect_ratio(lm, (36, 37, 38, 39, 40, 41))
        ear_values.append(ear)

        mw, mo = _compute_mouth_ratio(lm)
        mouth_widths.append(mw)
        mouth_opens.append(mo)

        brow_dists.append(_compute_brow_distance(lm))
        p, y, r = _estimate_head_pose(lm)
        pitches.append(p)
        yaws.append(y)
        rolls.append(r)

    # AU 强度（取平均和最大）
    au_names = [
        "au1_inner_brow_raiser", "au2_outer_brow_raiser", "au4_brow_lowerer",
        "au5_upper_lid_raiser", "au6_cheek_raiser", "au7_lid_tightener",
        "au9_nose_wrinkler", "au10_upper_lip_raiser", "au12_lip_corner_puller",
        "au15_lip_corner_depressor", "au17_chin_raiser", "au20_lip_stretcher",
        "au26_jaw_drop",
    ]

    if au_intensities:
        au_means = {}
        au_maxs = {}
        for au_name in au_names:
            values = [frame.get(au_name, 0.0) for frame in au_intensities]
            au_means[au_name] = np.mean(values) if values else 0.0
            au_maxs[au_name] = max(values) if values else 0.0
    else:
        # 从几何特征近似估计 AU
        au_means = _estimate_aus_from_geometry(
            np.mean(ear_values),
            np.mean(mouth_widths),
            np.mean(mouth_opens),
            np.mean(brow_dists),
        )
        au_maxs = {k: min(v * 1.5, 1.0) for k, v in au_means.items()}

    # 微表情检测：AU 强度突变
    micro_events = _detect_micro_expressions(au_intensities or [], fps)

    features = FacialFeatures(
        au1_inner_brow_raiser=au_means.get("au1_inner_brow_raiser", 0.0),
        au2_outer_brow_raiser=au_means.get("au2_outer_brow_raiser", 0.0),
        au4_brow_lowerer=au_means.get("au4_brow_lowerer", 0.0),
        au5_upper_lid_raiser=au_means.get("au5_upper_lid_raiser", 0.0),
        au6_cheek_raiser=au_means.get("au6_cheek_raiser", 0.0),
        au7_lid_tightener=au_means.get("au7_lid_tightener", 0.0),
        au9_nose_wrinkler=au_means.get("au9_nose_wrinkler", 0.0),
        au10_upper_lip_raiser=au_means.get("au10_upper_lip_raiser", 0.0),
        au12_lip_corner_puller=au_means.get("au12_lip_corner_puller", 0.0),
        au15_lip_corner_depressor=au_means.get("au15_lip_corner_depressor", 0.0),
        au17_chin_raiser=au_means.get("au17_chin_raiser", 0.0),
        au20_lip_stretcher=au_means.get("au20_lip_stretcher", 0.0),
        au26_jaw_drop=au_means.get("au26_jaw_drop", 0.0),
        brow_distance_ratio=np.mean(brow_dists),
        mouth_width_ratio=np.mean(mouth_widths),
        eye_aspect_ratio=np.mean(ear_values),
        mouth_open_ratio=np.mean(mouth_opens),
        micro_expression_count=len(micro_events),
        micro_expression_duration_avg=(
            np.mean([e.duration_ms for e in micro_events]) if micro_events else 0.0
        ),
        expression_intensity_mean=np.mean(list(au_maxs.values())) if au_maxs else 0.0,
        expression_intensity_max=max(au_maxs.values()) if au_maxs else 0.0,
        head_pitch=np.mean(pitches),
        head_yaw=np.mean(yaws),
        head_roll=np.mean(rolls),
        frame_count=num_frames,
        fps=fps,
        detection_confidence=0.85 if num_frames > 10 else 0.5,
    )

    return features


def _estimate_aus_from_geometry(
    ear: float,
    mouth_width: float,
    mouth_open: float,
    brow_dist: float,
) -> dict[str, float]:
    """从几何特征近似估计 AU 强度"""
    aus: dict[str, float] = {}

    # AU1 (内眉上扬): 眉毛距离大
    aus["au1_inner_brow_raiser"] = np.clip(brow_dist / 0.2, 0, 1)
    # AU2 (外眉上扬)
    aus["au2_outer_brow_raiser"] = np.clip(brow_dist / 0.22, 0, 1)
    # AU4 (眉毛下压): 眉毛距离小
    aus["au4_brow_lowerer"] = np.clip(1.0 - brow_dist / 0.15, 0, 1)
    # AU5 (上眼睑抬起): EAR 大
    aus["au5_upper_lid_raiser"] = np.clip((ear - 0.2) / 0.15, 0, 1)
    # AU6 (脸颊抬起): EAR 稍大 + 嘴宽
    aus["au6_cheek_raiser"] = np.clip((ear - 0.15) / 0.2 + mouth_width, 0, 1) * 0.5
    # AU7 (眼睑收紧): EAR 小
    aus["au7_lid_tightener"] = np.clip((0.25 - ear) / 0.15, 0, 1)
    # AU9 (鼻子皱起)
    aus["au9_nose_wrinkler"] = 0.0
    # AU10 (上唇上扬)
    aus["au10_upper_lip_raiser"] = np.clip(mouth_open * 2, 0, 1) * 0.5
    # AU12 (嘴角上提): 嘴宽大
    aus["au12_lip_corner_puller"] = np.clip((mouth_width - 0.3) / 0.2, 0, 1)
    # AU15 (嘴角下拉): 嘴宽小
    aus["au15_lip_corner_depressor"] = np.clip((0.3 - mouth_width) / 0.15, 0, 1)
    # AU17 (下巴抬起)
    aus["au17_chin_raiser"] = np.clip(mouth_open, 0, 1) * 0.3
    # AU20 (嘴唇拉伸): 嘴宽大
    aus["au20_lip_stretcher"] = np.clip((mouth_width - 0.35) / 0.15, 0, 1)
    # AU26 (下巴下垂): 张嘴大
    aus["au26_jaw_drop"] = np.clip((mouth_open - 0.3) / 0.4, 0, 1)

    return aus


def _detect_micro_expressions(
    au_sequence: list[dict[str, float]],
    fps: float,
    threshold: float = 0.4,
) -> list[MicroExpressionEvent]:
    """检测微表情事件

    微表情定义：持续时间 < 500ms 的 AU 强度突变

    Args:
        au_sequence: 每帧的 AU 强度
        fps: 帧率
        threshold: AU 变化阈值
    """
    if len(au_sequence) < 3:
        return []

    events: list[MicroExpressionEvent] = []
    au_keys = [k for k in au_sequence[0].keys() if k.startswith("au")]

    for au_key in au_keys:
        values = [frame.get(au_key, 0.0) for frame in au_sequence]

        # 检测突变：相邻帧差值超过阈值
        for i in range(1, len(values) - 1):
            diff = abs(values[i] - values[i - 1])
            if diff > threshold:
                # 找到峰值
                peak_frame = i
                # 向前向后搜索结束帧
                start = i - 1
                end = i + 1
                while end < len(values) and values[end] > values[i - 1] + threshold * 0.3:
                    end += 1

                duration_frames = end - start
                duration_ms = duration_frames / fps * 1000

                # 微表情：持续时间 < 500ms
                if duration_ms < 500:
                    # 映射到情绪
                    mapped = _au_to_emotion(au_key)
                    events.append(MicroExpressionEvent(
                        start_frame=start,
                        end_frame=end,
                        duration_ms=duration_ms,
                        primary_au=au_key,
                        intensity=values[peak_frame],
                        mapped_emotion=mapped,
                    ))

    return events


def _au_to_emotion(au_key: str) -> str:
    """将单个 AU 映射到最可能的情绪"""
    au_emotion_map = {
        "au1_inner_brow_raiser": "sadness",
        "au2_outer_brow_raiser": "surprise",
        "au4_brow_lowerer": "anger",
        "au5_upper_lid_raiser": "surprise",
        "au6_cheek_raiser": "happiness",
        "au7_lid_tightener": "anger",
        "au9_nose_wrinkler": "disgust",
        "au10_upper_lip_raiser": "disgust",
        "au12_lip_corner_puller": "happiness",
        "au15_lip_corner_depressor": "sadness",
        "au17_chin_raiser": "distress",
        "au20_lip_stretcher": "fear",
        "au26_jaw_drop": "fear",
    }
    return au_emotion_map.get(au_key, "neutral")


# ============================================================
# 情绪风险映射
# ============================================================

def features_to_emotion_risk(features: FacialFeatures) -> dict[str, Any]:
    """将面部特征映射为情绪风险结果

    Args:
        features: 面部特征

    Returns:
        dict with emotion_probs, risk_score, risk_label, confidence, metadata
    """
    # 计算每种情绪的得分
    emotion_scores: dict[str, float] = {}

    for emotion_name, rule in AU_EMOTION_RULES.items():
        au_values = []
        for au_field in rule["aus"]:
            au_value = getattr(features, au_field, 0.0)
            au_values.append(au_value)

        if au_values:
            # 情绪得分 = AU 强度均值 * 规则权重
            score = np.mean(au_values) * rule["weight"]
            emotion_scores[emotion_name] = float(np.clip(score, 0, 1))
        else:
            emotion_scores[emotion_name] = 0.0

    # 归一化为概率分布
    total = sum(emotion_scores.values())
    if total > 0:
        emotion_probs = {k: v / total for k, v in emotion_scores.items()}
    else:
        emotion_probs = {"neutral": 1.0}
        for name in AU_EMOTION_RULES:
            emotion_probs[name] = 0.0

    # 负面风险 = sadness + fear + anger + distress 的概率
    negative_risk = sum(
        emotion_probs.get(name, 0.0)
        for name in ["sadness", "fear", "anger", "distress"]
    )

    # 微表情加成：频繁的微表情增加风险
    micro_bonus = min(features.micro_expression_count * 0.05, 0.15)
    negative_risk = min(negative_risk + micro_bonus, 1.0)

    # 情绪标签
    if negative_risk > 0.6:
        label = "high_risk"
    elif negative_risk > 0.3:
        label = "moderate_risk"
    else:
        label = "low_risk"

    return {
        "emotion_probs": emotion_probs,
        "risk_score": negative_risk,
        "risk_label": label,
        "confidence": features.detection_confidence,
        "metadata": {
            "source": "facial_expression",
            "micro_expression_count": features.micro_expression_count,
            "expression_intensity": features.expression_intensity_max,
            "head_pitch": features.head_pitch,
            "head_yaw": features.head_yaw,
            "frame_count": features.frame_count,
        },
    }


# ============================================================
# MediaPipe FaceMesh 468 点支持
# ============================================================

# MediaPipe FaceMesh → FACS AU 映射（基于 blendshapes 名称）
# MediaPipe 的 blendshapes 直接对应 FACS AU
MEDIAPIPE_BLENDSHAPE_TO_AU: dict[str, str] = {
    "browInnerUp": "au1_inner_brow_raiser",
    "browDownLeft": "au4_brow_lowerer",
    "browDownRight": "au4_brow_lowerer",
    "eyeBlinkLeft": "au7_lid_tightener",
    "eyeBlinkRight": "au7_lid_tightener",
    "eyeWideLeft": "au5_upper_lid_raiser",
    "eyeWideRight": "au5_upper_lid_raiser",
    "cheekPuff": "au9_nose_wrinkler",
    "cheekSquintLeft": "au6_cheek_raiser",
    "cheekSquintRight": "au6_cheek_raiser",
    "jawOpen": "au26_jaw_drop",
    "jawForward": "au26_jaw_drop",
    "mouthSmileLeft": "au12_lip_corner_puller",
    "mouthSmileRight": "au12_lip_corner_puller",
    "mouthFrownLeft": "au15_lip_corner_depressor",
    "mouthFrownRight": "au15_lip_corner_depressor",
    "mouthDimplerLeft": "au12_lip_corner_puller",
    "mouthDimplerRight": "au12_lip_corner_puller",
    "mouthStretchLeft": "au20_lip_stretcher",
    "mouthStretchRight": "au20_lip_stretcher",
    "mouthPucker": "au20_lip_stretcher",
    "mouthPressLeft": "au4_brow_lowerer",
    "mouthPressRight": "au4_brow_lowerer",
    "mouthShrugLower": "au17_chin_raiser",
    "mouthShrugUpper": "au10_upper_lip_raiser",
    "noseSneerLeft": "au9_nose_wrinkler",
    "noseSneerRight": "au9_nose_wrinkler",
    "mouthRollLower": "au15_lip_corner_depressor",
    "mouthRollUpper": "au10_upper_lip_raiser",
}

# MediaPipe FaceMesh 468 点中的关键索引（与 68 点对应区域）
MEDIAPIPE_LANDMARK_GROUPS: dict[str, list[int]] = {
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    "left_brow": [70, 63, 105, 66, 107],
    "right_brow": [336, 296, 334, 293, 300],
    "outer_lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317],
    "inner_lips": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317],
    "face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "nose_tip": [1, 2, 98, 327],
}


def extract_from_mediapipe(
    blendshapes_sequence: list[dict[str, float]],
    landmarks_468_sequence: list[np.ndarray] | None = None,
    fps: float = 30.0,
) -> FacialFeatures:
    """从 MediaPipe FaceMesh 数据提取面部特征

    支持前端 MediaPipe FaceLandmarker 输出的 468 关键点 + blendshapes。
    blendshapes 直接映射到 FACS AU，比 68 点几何估计更精确。

    Args:
        blendshapes_sequence: 每帧的 blendshapes 字典列表
            例: [{"browInnerUp": 0.3, "mouthSmileLeft": 0.8, ...}, ...]
        landmarks_468_sequence: 每帧的 468 点坐标列表（可选，用于几何特征）
            每个形状 (468, 3) 或 (468, 2)
        fps: 帧率

    Returns:
        FacialFeatures
    """
    if not blendshapes_sequence:
        return FacialFeatures()

    num_frames = len(blendshapes_sequence)

    # 1. 从 blendshapes 提取 AU 强度（主要来源）
    au_frame_values: dict[str, list[float]] = {au: [] for au in set(MEDIAPIPE_BLENDSHAPE_TO_AU.values())}

    for frame_bs in blendshapes_sequence:
        # 同一 AU 可能对应多个 blendshape（左右脸），取最大值
        au_frame_max: dict[str, float] = {}
        for bs_name, au_name in MEDIAPIPE_BLENDSHAPE_TO_AU.items():
            bs_value = frame_bs.get(bs_name, 0.0)
            if au_name not in au_frame_max or bs_value > au_frame_max[au_name]:
                au_frame_max[au_name] = bs_value

        for au_name, value in au_frame_max.items():
            au_frame_values[au_name].append(value)

    # 计算 AU 均值和最大值
    au_means = {au: float(np.mean(vals)) if vals else 0.0 for au, vals in au_frame_values.items()}
    au_maxs = {au: float(max(vals)) if vals else 0.0 for au, vals in au_frame_values.items()}

    # 2. 从 468 点提取几何特征（如果有）
    ear_values = []
    mouth_widths = []
    mouth_opens = []
    brow_dists = []

    if landmarks_468_sequence:
        for lm468 in landmarks_468_sequence:
            # 从 468 点中提取 EAR
            left_eye = [lm468[i][:2] for i in MEDIAPIPE_LANDMARK_GROUPS["left_eye"] if i < len(lm468)]
            right_eye = [lm468[i][:2] for i in MEDIAPIPE_LANDMARK_GROUPS["right_eye"] if i < len(lm468)]

            if len(left_eye) >= 6:
                ear_l = _compute_ear_from_points(left_eye)
                ear_values.append(ear_l)
            if len(right_eye) >= 6:
                ear_r = _compute_ear_from_points(right_eye)
                ear_values.append(ear_r)

            # 嘴宽/张嘴比
            outer_lips = [lm468[i][:2] for i in MEDIAPIPE_LANDMARK_GROUPS["outer_lips"] if i < len(lm468)]
            if len(outer_lips) >= 4:
                mw, mo = _compute_mouth_from_points(outer_lips)
                mouth_widths.append(mw)
                mouth_opens.append(mo)

            # 眉毛距离
            left_brow = [lm468[i][:2] for i in MEDIAPIPE_LANDMARK_GROUPS["left_brow"] if i < len(lm468)]
            if left_brow and len(left_eye) >= 2:
                brow_y = np.mean([p[1] for p in left_brow])
                eye_y = np.mean([p[1] for p in left_eye[:4]])
                face_h = abs(lm468[10][1] - lm468[152][1]) if 152 < len(lm468) else 1.0
                brow_dists.append(abs(brow_y - eye_y) / max(face_h, 1e-6))

    # 3. 微表情检测
    au_dicts = []
    for frame_bs in blendshapes_sequence:
        au_dict: dict[str, float] = {}
        for bs_name, au_name in MEDIAPIPE_BLENDSHAPE_TO_AU.items():
            val = frame_bs.get(bs_name, 0.0)
            if au_name not in au_dict or val > au_dict[au_name]:
                au_dict[au_name] = val
        au_dicts.append(au_dict)

    micro_events = _detect_micro_expressions(au_dicts, fps)

    # 4. 构建 FacialFeatures
    features = FacialFeatures(
        au1_inner_brow_raiser=au_means.get("au1_inner_brow_raiser", 0.0),
        au2_outer_brow_raiser=au_means.get("au2_outer_brow_raiser", 0.0),
        au4_brow_lowerer=au_means.get("au4_brow_lowerer", 0.0),
        au5_upper_lid_raiser=au_means.get("au5_upper_lid_raiser", 0.0),
        au6_cheek_raiser=au_means.get("au6_cheek_raiser", 0.0),
        au7_lid_tightener=au_means.get("au7_lid_tightener", 0.0),
        au9_nose_wrinkler=au_means.get("au9_nose_wrinkler", 0.0),
        au10_upper_lip_raiser=au_means.get("au10_upper_lip_raiser", 0.0),
        au12_lip_corner_puller=au_means.get("au12_lip_corner_puller", 0.0),
        au15_lip_corner_depressor=au_means.get("au15_lip_corner_depressor", 0.0),
        au17_chin_raiser=au_means.get("au17_chin_raiser", 0.0),
        au20_lip_stretcher=au_means.get("au20_lip_stretcher", 0.0),
        au26_jaw_drop=au_means.get("au26_jaw_drop", 0.0),
        brow_distance_ratio=np.mean(brow_dists) if brow_dists else 0.15,
        mouth_width_ratio=np.mean(mouth_widths) if mouth_widths else 0.3,
        eye_aspect_ratio=np.mean(ear_values) if ear_values else 0.28,
        mouth_open_ratio=np.mean(mouth_opens) if mouth_opens else 0.1,
        micro_expression_count=len(micro_events),
        micro_expression_duration_avg=(
            np.mean([e.duration_ms for e in micro_events]) if micro_events else 0.0
        ),
        expression_intensity_mean=np.mean(list(au_maxs.values())) if au_maxs else 0.0,
        expression_intensity_max=max(au_maxs.values()) if au_maxs else 0.0,
        frame_count=num_frames,
        fps=fps,
        detection_confidence=0.95 if num_frames > 10 else 0.7,  # MediaPipe 置信度更高
    )

    return features


def _compute_ear_from_points(eye_points: list) -> float:
    """从任意数量的眼部关键点计算 EAR"""
    pts = [np.array(p[:2], dtype=float) for p in eye_points]
    if len(pts) < 4:
        return 0.28

    # 简化 EAR：垂直距离均值 / 水平距离
    n = len(pts)
    horizontal = np.linalg.norm(pts[0] - pts[n // 2])
    if horizontal < 1e-6:
        return 0.28

    vertical_sum = 0.0
    count = 0
    for i in range(1, n // 2):
        j = n - i
        if j < n:
            vertical_sum += np.linalg.norm(pts[i] - pts[j])
            count += 1

    ear = vertical_sum / max(count, 1) / horizontal
    return float(np.clip(ear, 0, 0.5))


def _compute_mouth_from_points(lip_points: list) -> tuple[float, float]:
    """从唇部关键点计算嘴宽比和张嘴比"""
    pts = [np.array(p[:2], dtype=float) for p in lip_points]
    if len(pts) < 4:
        return 0.3, 0.1

    n = len(pts)
    mouth_width = np.linalg.norm(pts[0] - pts[n // 2])
    mouth_height = np.linalg.norm(pts[n // 4] - pts[3 * n // 4])
    face_width = mouth_width * 2.5  # 估算脸宽

    width_ratio = mouth_width / max(face_width, 1e-6)
    open_ratio = mouth_height / max(mouth_width, 1e-6)

    return float(width_ratio), float(open_ratio)


def features_to_summary(features: FacialFeatures) -> dict[str, Any]:
    """生成面部特征摘要

    Returns:
        包含主要表情、AU 摘要、风险等级的字典
    """
    result = features_to_emotion_risk(features)

    # 找出最显著的情绪
    probs = result["emotion_probs"]
    dominant_emotion = max(probs, key=probs.get) if probs else "neutral"

    return {
        "dominant_emotion": dominant_emotion,
        "dominant_probability": probs.get(dominant_emotion, 0.0),
        "risk_score": result["risk_score"],
        "risk_label": result["risk_label"],
        "micro_expressions": features.micro_expression_count,
        "expression_intensity": features.expression_intensity_max,
        "eye_aspect_ratio": features.eye_aspect_ratio,
        "head_pose": {
            "pitch": features.head_pitch,
            "yaw": features.head_yaw,
            "roll": features.head_roll,
        },
        "confidence": result["confidence"],
    }
