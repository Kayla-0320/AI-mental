"""
TAMER: Therapeutic Alliance-aware Multimodal Emotion Recognition
治疗联盟感知的多模态情绪识别框架

================================================================
框架概述
================================================================

TAMER 是我们提出的统一算法框架，整合三大创新点：
    1. 关系质量指数 (RQI) —— 治疗联盟建模
    2. 多模态矛盾分数 (MIS) —— 跨模态融合
    3. CUSUM 变点检测 —— 时序轨迹分析

核心创新：
    - 传统方法：独立处理每个模态，最后加权平均
    - TAMER：用治疗关系质量动态调整模态权重，用矛盾分数检测异常，
             用 CUSUM 追踪长期变化

================================================================
数学模型
================================================================

1. 关系质量指数 (RQI)
   -------------------
   RQI(t) = α × RQI(t-1) + (1-α) × Σᵢ(wᵢ × signalᵢ(t))
   
   其中：
   - signal₁: 互动频率（对数归一化）
   - signal₂: 自我暴露深度（消息长度 + 情感词汇密度）
   - signal₃: 话题多样性（唯一话题/总话题）
   - signal₄: 信任信号（归一化计数）
   - signal₅: 情感共鸣（情绪匹配度）
   - wᵢ: 信号权重（可学习）
   - α: 时间衰减系数（0.7）

2. 多模态矛盾分数 (MIS)
   ---------------------
   MIS(t) = 1 - cosine_similarity(text_vec(t), voice_vec(t))
   
   自适应阈值：
   threshold(t) = μ_MIS + k × σ_MIS
   
   矛盾类型判定：
   - masking: text_calm ∧ voice_aroused ∧ MIS > threshold
   - performative: text_distressed ∧ voice_calm ∧ MIS > threshold

3. 注意力多模态融合
   -----------------
   传统加权平均：risk = Σ(wᵢ × modalityᵢ)
   
   TAMER 注意力融合：
   attentionᵢ = softmax(W × [modality; RQI; MIS])
   risk = Σ(attentionᵢ × modality)
   
   关键创新：模态权重不是固定的，而是由关系质量和矛盾分数动态调整。
   - 高 RQI（信任关系）→ 增加文本权重（用户更愿意真实表达）
   - 高 MIS（矛盾信号）→ 增加声学权重（声音更难掩饰）

4. CUSUM 变点检测
   ---------------
   S_t = max(0, S_{t-1} + (x_t - μ₀ - δ))
   
   变点判定：
   - S_t > h → 恶化（h = 4σ）
   - S_t < -h → 好转

================================================================
算法流程
================================================================

输入：user_message, voice_features, conversation_history
输出：risk_level, incongruence_type, trajectory

1. 提取模态特征
   text_emotion = TextEmotionModel(user_message)
   voice_emotion = AcousticModel(voice_features)

2. 计算 RQI
   rqi = compute_rqi(conversation_history)

3. 计算 MIS
   mis = 1 - cosine_similarity(text_emotion, voice_emotion)
   threshold = adaptive_threshold(mis_history)
   incongruence_type = classify_incongruence(mis, threshold)

4. 注意力融合
   attention_weights = attention_mechanism(
       [text_emotion, voice_emotion],
       context=[rqi, mis]
   )
   fused_emotion = Σ(attentionᵢ × modalityᵢ)

5. 风险评估
   risk_score = risk_classifier(fused_emotion)

6. CUSUM 轨迹分析
   cusum_result = cusum_detector.update(risk_score)
   trajectory = cusum_result.direction

7. 能力边界判断
   if consecutive_high_risk >= 5:
       recommendation = "referral"
   else:
       recommendation = "continue"

返回：{
    risk_level: risk_score,
    incongruence_type: incongruence_type,
    trajectory: trajectory,
    rqi: rqi,
    mis: mis,
    recommendation: recommendation
}

================================================================
理论贡献
================================================================

1. 首次将治疗联盟理论（Therapeutic Alliance）形式化为可计算指标
2. 提出矛盾感知的注意力融合机制（Contrast-aware Attention Fusion）
3. 将 CUSUM 工业质量控制算法引入心理状态追踪

================================================================
参考文献
================================================================

- Horvath & Greenberg (1989). Working Alliance Inventory
- Ekman (2003). Emotions Revealed: cross-channel incongruence
- Page (1954). Continuous Inspection Schemes: CUSUM algorithm
- Norcross & Lambert (2018). Psychotherapy relationships that work

================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from assessment.relationship_quality import compute_rqI, rqi_to_relationship_depth
from assessment.multimodal_incongruence import (
    compute_mis,
    classify_incongruence,
    acoustic_to_emotion_vector,
)
from assessment.temporal_trajectory import CUSUMDetector, analyze_emotion_trajectory


# ============================================================
# 声学特征的量纲归一化
# ============================================================
#
# ⚠️ 客户端上报的 voice_features **不是 0~1**（实测自 `useVoiceAnalysis.ts`
# 与 `useVoiceCall.ts` 的 `VoiceAnalysis`）：
#
#   energy.mean      dBFS，约 -60 ~ 0（<-50 判为静音，<-35 判为低落，>-20 判为大声）
#   pitch.range      Hz（≈ 0.3 × F0，实测约 25 ~ 120）
#   rate.speechRate  每秒过零数 / 100，实测约 0 ~ 8
#   pauses.ratio     0 ~ 1（唯一已经是归一化的）
#
# 而 :func:`acoustic_to_emotion_vector` 的形参文档写的是 0~1。
# 直接把原始值喂进去，`happy = 0.3*energy_mean + …` 会因 energy_mean=-40 变成负数、
# 归一化后整条向量塌成 0 或 NaN —— MIS / RQI 全部失真。这就是评审报告里
# P0「跨模态矛盾检测量纲错」的根因。
#
# 兼容口径：**看起来已经是归一化值的输入原样使用**，否则按上面的实测单位换算。
# 这样文字聊天（缺省 0.5/0.3/0.5/0.3）、旧调用方与新版客户端三条路径都不会算错。

def _clamp01(value: float) -> float:
    """把数值夹到 [0, 1]。"""
    return max(0.0, min(1.0, float(value)))


def normalize_energy_mean(value: float) -> float:
    """能量均值 → 0~1（dBFS 映射，-60 dBFS 为 0、0 dBFS 为 1）。"""
    return _clamp01((value + 60.0) / 60.0) if value < 0 else _clamp01(value)


def normalize_pitch_range(value: float) -> float:
    """基频范围 → 0~1（Hz 映射，150 Hz 记满分）。"""
    return _clamp01(value / 150.0) if value > 1.5 else _clamp01(value)


def normalize_speech_rate(value: float) -> float:
    """语速 → 0~1（原始刻度 8 记满分）。"""
    return _clamp01(value / 8.0) if value > 1.5 else _clamp01(value)


# ============================================================
# TAMER 框架配置
# ============================================================

@dataclass
class TAMERConfig:
    """TAMER 框架配置"""
    rqi_decay: float = 0.7  # RQI 时间衰减
    mis_threshold_k: float = 1.5  # MIS 阈值倍数
    cusum_slack: float = 0.05  # CUSUM 允许漂移
    cusum_threshold_multiplier: float = 4.0  # CUSUM 阈值倍数
    referral_consecutive_count: int = 5  # 连续高风险转介阈值


# ============================================================
# TAMER 框架主类
# ============================================================

class TAMERFramework:
    """TAMER: 治疗联盟感知的多模态情绪识别框架
    
    使用方式：
        config = TAMERConfig()
        tamer = TAMERFramework(config)
        
        result = tamer.process(
            user_message="我没事",
            voice_features={"energy": 0.7, "pitch_range": 0.6, ...},
            text_emotion=[0.3, 0.1, 0.1, 0.05, 0.45],
            conversation_history=[...],
            mis_history=[0.1, 0.15, 0.2, ...]
        )
    """
    
    def __init__(self, config: Optional[TAMERConfig] = None) -> None:
        self.config = config or TAMERConfig()
        self.cusum_detector = CUSUMDetector(
            baseline_mean=0.3,
            baseline_std=0.1,
            slack=self.config.cusum_slack,
            threshold_multiplier=self.config.cusum_threshold_multiplier,
        )
        self.mis_history: list[float] = []
        self.risk_history: list[float] = []
        self.consecutive_high_risk = 0
        #: RQI 的 EMA 上一轮值；None 表示本会话还没有算过。
        #: 见 :meth:`_compute_rqi` —— 文档里的 α 衰减必须靠它才成立。
        self._prev_rqi: Optional[float] = None
    
    def process(
        self,
        user_message: str,
        voice_features: dict,
        text_emotion: list[float],
        conversation_history: list[dict],
        mis_history: Optional[list[float]] = None,
    ) -> dict:
        """处理一轮对话
        
        Args:
            user_message: 用户消息
            voice_features: 声学特征
            text_emotion: 文本情绪向量
            conversation_history: 对话历史
            mis_history: 历史 MIS 值
        
        Returns:
            dict: TAMER 输出
        """
        # 1. 计算 RQI
        rqi = self._compute_rqi(conversation_history, user_message)
        relationship_depth = rqi_to_relationship_depth(rqi)
        
        # 2. 计算声学情绪向量
        # ⚠️ 先做量纲归一化：客户端上报的是 dBFS / Hz / 原始语速刻度，
        # 而 acoustic_to_emotion_vector 要的是 0~1。直接喂原始值会让向量塌掉，
        # 详见文件上方 `normalize_energy_mean` 一带的说明。
        voice_emotion = acoustic_to_emotion_vector(
            energy_mean=normalize_energy_mean(
                voice_features.get("energy", {}).get("mean", 0.5)
            ),
            pitch_range=normalize_pitch_range(
                voice_features.get("pitch", {}).get("range", 0.3)
            ),
            speech_rate=normalize_speech_rate(
                voice_features.get("rate", {}).get("speechRate", 0.5)
            ),
            pause_ratio=_clamp01(voice_features.get("pauses", {}).get("ratio", 0.3)),
        )
        
        # 3. 计算 MIS
        mis = compute_mis(text_emotion, voice_emotion)
        self.mis_history.append(mis)
        
        # 4. 矛盾类型判定
        incongruence_result = classify_incongruence(
            text_emotion,
            voice_emotion,
            self.mis_history[-10:],  # 最近 10 轮
            k=self.config.mis_threshold_k,
        )
        
        # 5. 注意力融合（简化版：基于 RQI 和 MIS 调整权重）
        fused_emotion = self._attention_fusion(
            text_emotion,
            voice_emotion,
            rqi,
            mis,
        )
        
        # 6. 风险评估（简化：负性情绪强度）
        risk_score = sum(fused_emotion[1:4])  # 悲伤 + 焦虑 + 愤怒
        
        # 7. CUSUM 轨迹分析
        cusum_result = self.cusum_detector.update(risk_score)
        self.risk_history.append(risk_score)
        
        # 8. 能力边界判断
        if risk_score > 0.6:
            self.consecutive_high_risk += 1
        else:
            self.consecutive_high_risk = 0
        
        should_refer = self.consecutive_high_risk >= self.config.referral_consecutive_count
        
        return {
            "risk_level": self._risk_score_to_level(risk_score),
            "risk_score": round(risk_score, 4),
            "incongruence_type": incongruence_result["type"],
            "mis": round(mis, 4),
            "mis_threshold": round(incongruence_result["threshold"], 4),
            "rqi": round(rqi, 4),
            "relationship_depth": relationship_depth,
            "trajectory": cusum_result["direction"],
            "should_refer": should_refer,
            "consecutive_high_risk": self.consecutive_high_risk,
            # 本轮是否真的有声学输入。
            # ⚠️ 为 False 时 MIS 是用缺省声学值算出来的，**不构成跨模态证据**，
            # 前端据此显示"本轮无语音输入"而不是把默认值渲染成"言语与声音一致"。
            "has_voice_input": bool(voice_features),
        }
    
    def _compute_rqi(
        self,
        conversation_history: list[dict],
        current_message: str,
    ) -> float:
        """计算本轮的关系质量指数（RQI），并做 EMA 时间衰减。

        ⚠️ 修正说明（2026-10-10）：

        1. **EMA 此前根本没实现。** 模块与报告里写的是
           ``RQI(t) = α·RQI(t-1) + (1-α)·Σ(wᵢ·signalᵢ(t))``，``TAMERConfig.rqi_decay``
           也声明了 0.7，但原实现每轮直接返回 ``compute_rqI(...)`` 的**当期加权和**，
           α 从未参与运算 —— 于是"时间衰减""关系会累积"这两句在代码里不成立。
           现在补上 EMA；``RQI(0)`` 取 0（尚无关系可言），
           这也让前几轮自然落在"陌生期"，随轮次上升。

        2. **话题多样性恒等于 1。** 原实现把 ``unique_topics`` 与 ``total_topics``
           都赋成同一轮的句子数，于是 ``unique/total`` 永远是 1.0，
           该项退化成一个 0.15 的常数加项（对任何对话都不变），
           这也是"消融显示 RQI 贡献为 0"的一个来源。
           现在改为在**整段对话**的句子上做去重统计。

        Args:
            conversation_history: 历史对话 ``[{role, content}]``。
            current_message: 用户本轮发言。

        Returns:
            float: 带时间衰减的 RQI（0~1，保留 4 位）。
        """
        user_messages = [
            m.get("content", "")
            for m in conversation_history
            if m.get("role") == "user"
        ]
        session_count = len(user_messages)
        avg_message_length = sum(len(c) for c in user_messages) / max(1, session_count)

        # 情感词汇密度（简化：负性情绪词比例）
        emotional_words = ["难受", "痛苦", "烦", "累", "悲伤", "焦虑"]
        emotional_count = sum(
            1 for word in emotional_words
            if word in current_message
        )
        emotional_density = emotional_count / max(1, len(current_message))

        # 话题多样性：在**整段对话 + 本轮**的句子上算"唯一句 / 总句"。
        # 原实现把两者都写成同一轮的句子数 → 恒为 1.0（见 docstring 第 2 条）。
        def _sentences(text: str) -> list[str]:
            parts: list[str] = []
            for chunk in text.replace("！", "。").replace("？", "。").split("。"):
                chunk = chunk.strip()
                if len(chunk) > 5:
                    parts.append(chunk)
            return parts

        all_sentences = [s for c in user_messages for s in _sentences(c)]
        all_sentences += _sentences(current_message)
        total_topics = len(all_sentences)
        unique_topics = len(set(all_sentences))

        # 信任信号
        trust_keywords = ["秘密", "只有你知道", "你懂", "朋友"]
        trust_count = sum(
            1 for kw in trust_keywords
            if kw in current_message
        )

        # 情感共鸣（简化：假设 AI 能匹配用户情绪）
        emotion_match = 0.7  # 默认值

        raw_rqi = compute_rqI(
            session_count=session_count,
            avg_message_length=avg_message_length,
            emotional_vocabulary_density=emotional_density,
            unique_topics=unique_topics,
            total_topics=max(1, total_topics),
            trust_signal_count=trust_count,
            emotion_match_score=emotion_match,
        )

        # 文档里的时间衰减：RQI(t) = α·RQI(t-1) + (1-α)·raw(t)
        alpha = self.config.rqi_decay
        if self._prev_rqi is None:
            self._prev_rqi = (1.0 - alpha) * raw_rqi
        else:
            self._prev_rqi = alpha * self._prev_rqi + (1.0 - alpha) * raw_rqi
        return round(self._prev_rqi, 4)
    
    def _attention_fusion(
        self,
        text_emotion: list[float],
        voice_emotion: list[float],
        rqi: float,
        mis: float,
    ) -> list[float]:
        """注意力多模态融合
        
        核心创新：模态权重由关系质量和矛盾分数动态调整。
        
        - 高 RQI（信任关系）→ 增加文本权重（用户更真实）
        - 高 MIS（矛盾信号）→ 增加声学权重（声音更难掩饰）
        """
        # 基础权重
        text_weight = 0.6
        voice_weight = 0.4
        
        # RQI 调整：高关系质量 → 更信任文本
        rqi_adjustment = (rqi - 0.5) * 0.2  # 范围 -0.1 到 +0.1
        text_weight += rqi_adjustment
        
        # MIS 调整：高矛盾 → 更信任声学
        mis_adjustment = (mis - 0.3) * 0.3  # 范围 -0.09 到 +0.21
        voice_weight += mis_adjustment
        text_weight -= mis_adjustment
        
        # 归一化
        total = text_weight + voice_weight
        text_weight /= total
        voice_weight /= total
        
        # 加权融合
        fused = [
            text_weight * t + voice_weight * v
            for t, v in zip(text_emotion, voice_emotion)
        ]
        
        return fused
    
    def _risk_score_to_level(self, risk_score: float) -> str:
        """风险分数转等级"""
        if risk_score > 0.7:
            return "crisis"
        elif risk_score > 0.5:
            return "high"
        elif risk_score > 0.3:
            return "medium"
        else:
            return "low"
    
    def reset(self) -> None:
        """重置框架状态"""
        self.cusum_detector.reset()
        self.mis_history = []
        self.risk_history = []
        self.consecutive_high_risk = 0
        self._prev_rqi = None


# ============================================================
# 便捷接口
# ============================================================

def create_tamer(config: Optional[TAMERConfig] = None) -> TAMERFramework:
    """创建 TAMER 框架实例"""
    return TAMERFramework(config)
