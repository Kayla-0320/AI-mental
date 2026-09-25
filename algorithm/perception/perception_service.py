"""
感知层统一服务接口

所有下游模块（评估、干预、画像）通过此接口获取情感数据。
多模态融合调用 fusion/fusion.py 中的 fuse_multimodal() 统一接口，
支持 weighted_average / stacking / dynamic 三种融合策略。

v2: 接入真实文本模型 + 面部/行为特征融合
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Optional

from shared.dataclasses import EmotionResult
from perception.fusion.fusion import fuse_multimodal, StackingFusion
from perception.fusion.hierarchical_fusion import (
    HierarchicalFusionEngine,
    ModalityResult as HUAFModalityResult,
    risk_score_to_probs as huaf_risk_to_probs,
)
from perception.age_config import (
    AgeGroup,
    AgeProfile,
    get_age_profile,
    get_age_profile_by_age,
)

# 新增 7 模态导入
from perception.circadian.circadian_rhythm import CircadianFeatures, analyze_circadian_risk
from perception.cognitive.cognitive_distortion import CognitiveDistortionFeatures, analyze_cognitive_distortions
from perception.physiological.rppg_hrv import HRVFeatures, analyze_hrv_risk
from perception.physiological.breathing import BreathingFeatures, analyze_breathing_risk
from perception.behavioral_activation.behavioral_activation import BehavioralActivationFeatures, analyze_behavioral_activation_risk
from perception.eye.eye_tracking import EyeMovementFeatures, analyze_eye_movement_risk
from perception.voice_semantics.deep_semantics import VoiceSemanticsFeatures, analyze_voice_semantics_risk
from assessment.digital_twin import get_digital_twin_manager

logger = logging.getLogger(__name__)


# ============================================================
# 模型标签映射
# text_emotion 模型输出: [anxiety(0), depression(1), anger(2), neutral(3), positive(4)]
# EmotionResult 期望:     [快乐(0),    悲伤(1),      焦虑(2),  愤怒(3),   中性(4)]
# ============================================================

def _remap_probs(model_probs: list[float]) -> list[float]:
    """将模型输出概率映射为 EmotionResult 格式

    Args:
        model_probs: 模型输出 [anxiety, depression, anger, neutral, positive]

    Returns:
        EmotionResult 格式 [快乐, 悲伤, 焦虑, 愤怒, 中性]
    """
    anxiety, depression, anger, neutral, positive = model_probs
    return [positive, depression, anxiety, anger, neutral]


# ============================================================
# 文本情感引擎：学习模型优先，规则引擎降级
# ============================================================
# 主路径：RoBERTa 微调模型（在 MMPsy/EATD/PsyQA 等数据集上训练）
# 降级路径：jieba + NRC 情感词典（基于心理学理论的规则引擎）
# ============================================================

# 语义说明：_text_model_available 表示「文本情感引擎可调用」，
# **不是**「RoBERTa 学习模型已加载」。词典降级方案始终可用，因此该标志通常为 True；
# 真正用的是哪个引擎看 _text_engine。危机关键词筛查不依赖该标志
# （见 _screen_crisis_keywords），以免引擎选择影响安全路径。
_text_model_available = False
_text_engine = "none"  # "roberta" | "lexicon" | "none"


@functools.lru_cache(maxsize=1)
def _try_load_text_model() -> bool:
    """尝试加载文本情感模型，返回是否成功

    引擎优先级（学习模型优先，规则引擎降级）：
    1. [主路径] RoBERTa 微调模型 —— 在 MMPsy/EATD/PsyQA 数据集上训练
    2. [降级] jieba + NRC 情感词典 —— 基于心理学理论的规则引擎
    """
    global _text_model_available, _text_engine
    try:
        from perception.text_emotion.predict import _check_trained_model
        if _check_trained_model():
            _text_model_available = True
            _text_engine = "roberta"
            logger.info("文本情感引擎: RoBERTa (本地训练模型)")
        else:
            # RoBERTa 不可用，但词典方案始终可用
            # （本标志含义为「文本引擎可调用」，非「学习模型已加载」）
            _text_model_available = True
            _text_engine = "lexicon"
            logger.info("文本情感引擎: jieba + 情感词典 (轻量方案)")
    except Exception as e:
        logger.warning(f"文本情感引擎加载异常: {e}")
        _text_model_available = False
        _text_engine = "none"
    return _text_model_available


def _real_text_predict(text: str) -> tuple[list[float], str, str]:
    """调用文本情感模型

    Args:
        text: 待分析文本

    Returns:
        (EmotionResult格式的5维概率, 引擎名称, 匹配词列表)
    """
    from perception.text_emotion.predict import predict
    result = predict(text)
    model_probs = result["probs"]  # [anxiety, depression, anger, neutral, positive]
    engine = result.get("engine", "roberta")
    matched = result.get("matched", [])
    return _remap_probs(model_probs), engine, matched


# ============================================================
# 危机关键词筛查（安全优先，独立于情感引擎）
# ============================================================
# ⚠️ 比赛 Demo 为模拟实现，生产环境需接入真实危机干预流程
#
# 背景：危机词表原先只写在 _rule_based_text_analysis 内部，而该分支在
# _text_model_available 恒为 True 时不可达，导致任何直接调用 analyze_text()
# 的入口（如 POST /perception/analyze）都拿不到危机信号。
# 现将其提为模块级常量，并在 analyze_text() 的所有返回路径上无条件筛查。
_DEFAULT_CRISIS_KEYWORDS: tuple[str, ...] = (
    # ── 直接自杀/自伤意图 ──
    '自杀', '自残', '自伤', '不想活', '想死', '去死', '跳楼', '割腕',
    '活不下去', '死了算了', '不想活了', '不想醒来',
    '服毒', '上吊', 'overdose', '吃药自杀',
    '划手', '划手臂', '伤害自己', '烧自己', '刺自己', '掐自己',
    # ── 间接危机表达（绝望/无价值/告别）──
    '活着没意思', '活着没什么意义', '没有活下去的理由',
    '活着是一种负担', '活着好累', '不想存在',
    '消失就好了', '消失好了', '如果我不在了',
    '没人会在意', '世界没有我会更好', '没有我会更好',
    '我是废物', '没人要我', '我是个负担',
    '了结', '解脱', '一了百了', '了断',
    '结束生命', '结束一切', '结束这一切',
    '撑不住了', '撑不下去', '不想努力了', '放弃一切',
    # ── 告别/后事安排暗示 ──
    '交代后事', '最后的告别', '写遗书',
    '把我的东西', '留给你们', '对不起大家',
    '请原谅我', '我不值得', '我配不上',
    '你们会过得更好', '没有我的日子',
    # ── 隐晦/哲学化危机表达 ──
    '不想面对', '看不到希望', '看不到出路',
    '一切都没意义', '一切都没有意义', '好累不想动',
    '好想逃', '无尽的黑暗', '深渊',
    '存在没有意义', '存在的荒谬', '虚无',
    '精神内耗到极限', '倦怠到极限',
    '人间不值得', '生而为人我很抱歉',
    '活着像行尸走肉', '灵魂已经死了',
    '心已经空了', '什么都感觉不到了',
    # ── 青少年常见危机表达 ──
    '不想上学', '累了', '没有意义',
    '活着干嘛', '有什么用',
    '反正没人在乎', '多我一个不多',
    '我就是个笑话', '谁都不会心疼',
)


def _screen_crisis_keywords(
    text: str,
    age_profile: Optional[AgeProfile] = None,
) -> tuple[int, list[str]]:
    """危机关键词筛查（不依赖情感引擎，任何路径都应调用）

    Args:
        text: 待筛查文本
        age_profile: 年龄画像；提供时使用该年龄段的专属危机词库

    Returns:
        (命中个数, 命中的关键词列表)
    """
    if age_profile is not None:
        keywords = age_profile.text.crisis_keywords
    else:
        keywords = _DEFAULT_CRISIS_KEYWORDS
    text_lower = text.lower()
    matched = [w for w in keywords if w in text_lower]
    return len(matched), matched


# ============================================================
# 面部/行为特征提取（可选模块）
# ============================================================

def _extract_facial_risk(
    facial_features: Optional[dict] = None,
    age_profile: Optional[AgeProfile] = None,
) -> Optional[float]:
    """从面部特征提取情绪风险分（支持年龄校准）

    Args:
        facial_features: 面部特征字典（来自 facial_expression.py）
        age_profile: 年龄画像（可选）

    Returns:
        风险概率 [0, 1]，或 None（无数据时）
    """
    if not facial_features:
        return None
    try:
        from perception.face.facial_expression import FacialFeatures, features_to_emotion_risk
        # 将 dict 转换为 FacialFeatures dataclass
        if isinstance(facial_features, dict):
            ff = FacialFeatures(**{k: v for k, v in facial_features.items() if hasattr(FacialFeatures, k)})
        else:
            ff = facial_features
        result = features_to_emotion_risk(ff, age_profile=age_profile)
        return result.get("risk_score", None)
    except Exception as e:
        logger.warning(f"面部特征提取失败: {e}")
        return None


def _extract_behavior_risk(
    behavior_features: Optional[dict] = None,
    age_profile: Optional[AgeProfile] = None,
) -> Optional[float]:
    """从行为特征提取情绪风险分（支持年龄校准）

    Args:
        behavior_features: 行为特征字典（来自 behavior_pattern.py）
        age_profile: 年龄画像（可选）

    Returns:
        风险概率 [0, 1]，或 None（无数据时）
    """
    if not behavior_features:
        return None
    try:
        from perception.behavior.behavior_pattern import BehaviorFeatures, behavior_to_emotion_risk
        # 将 dict 转换为 BehaviorFeatures dataclass
        if isinstance(behavior_features, dict):
            bf = BehaviorFeatures(**{k: v for k, v in behavior_features.items() if hasattr(BehaviorFeatures, k)})
        else:
            bf = behavior_features
        result = behavior_to_emotion_risk(bf, age_profile=age_profile)
        return result.get("risk_score", None)
    except Exception as e:
        logger.warning(f"行为特征提取失败: {e}")
        return None


def _acoustic_to_emotion_probs(acoustic, age_profile: Optional["AgeProfile"] = None) -> list[float]:
    """将声学特征映射为5维情绪概率分布（支持年龄校准）

    基于 Scherer (2003) 和 Cummins et al. (2015) 的经验规则，
    结合 Jiang et al. (2020) 的青少年年龄校准。
    - 低基频 + 低能量 → 悲伤
    - 高基频 + 高语速 → 焦虑
    - 高能量 + 高 F0 范围 → 愤怒
    - 平稳基频 + 中等能量 → 中性
    - 中等 F0 + 变化丰富 → 快乐

    Args:
        acoustic: AcousticFeatures 实例
        age_profile: 年龄画像（可选），用于校准阈值

    Returns:
        [快乐, 悲伤, 焦虑, 愤怒, 中性] 概率分布
    """
    probs = [0.0] * 5  # [快乐, 悲伤, 焦虑, 愤怒, 中性]

    # 年龄校准后的阈值
    if age_profile is not None:
        vp = age_profile.voice
        f0_anx_thr = vp.f0_anxiety_threshold
        f0_dep_thr = vp.f0_depression_threshold
        sr_anx_thr = vp.speech_rate_anxiety_threshold
        sr_dep_lo = vp.speech_rate_normal_range[0]
        pause_thr = vp.pause_depression_threshold
    else:
        f0_anx_thr = 250.0
        f0_dep_thr = 150.0
        sr_anx_thr = 5.0
        sr_dep_lo = 2.5
        pause_thr = 0.3

    # 基频指标
    f0_norm = min(1.0, max(0.0, (acoustic.f0_mean - 100) / 200))  # 归一化到 [0,1]

    # 低基频 + 低能量 → 悲伤（使用年龄校准阈值）
    if acoustic.f0_mean < f0_dep_thr and acoustic.energy_mean < 50:
        probs[1] += 0.4

    # 高基频 + 高语速 → 焦虑（使用年龄校准阈值）
    if acoustic.f0_mean > f0_anx_thr or acoustic.speech_rate > sr_anx_thr:
        probs[2] += 0.35

    # 高能量 + 大 F0 范围 → 愤怒
    if acoustic.energy_mean > 65 and acoustic.f0_range > 100:
        probs[3] += 0.35

    # 高停顿 → 悲伤/焦虑（使用年龄校准阈值）
    if acoustic.pause_ratio > pause_thr:
        probs[1] += 0.15
        probs[2] += 0.1

    # 语速慢 → 悲伤（使用年龄校准阈值）
    if acoustic.speech_rate < sr_dep_lo:
        probs[1] += 0.2

    # 能量波动大 → 愤怒/焦虑
    if acoustic.energy_std > 10:
        probs[2] += 0.1
        probs[3] += 0.1

    # F0 范围大 + 中等语速 → 快乐
    if acoustic.f0_range > 80 and sr_dep_lo < acoustic.speech_rate < sr_anx_thr:
        probs[0] += 0.3

    # 如果所有指标都正常 → 中性
    if max(probs) < 0.1:
        probs[4] = 0.6
    else:
        probs[4] = max(0.05, 1.0 - sum(probs))

    # 归一化
    s = sum(probs)
    if s > 0:
        probs = [p / s for p in probs]
    else:
        probs = [0.2, 0.2, 0.2, 0.2, 0.2]

    return probs


# ============================================================
# 融合策略配置
# ============================================================
# 默认使用 HUAF-TC（Hierarchical Uncertainty-Aware Fusion with
# Temporal Consistency）层级融合算法——本系统的核心原创算法。
# 当 HUAF-TC 不可用时，自动降级到传统融合策略。

DEFAULT_FUSION_STRATEGY = "hierarchical"
TEXT_WEIGHT = 0.6
AUDIO_WEIGHT = 0.4


# ============================================================
# PerceptionService v2
# ============================================================

class PerceptionService:
    """感知层统一服务 —— 基于 HUAF-TC 原创算法的 11 模态层级融合

    核心算法：HUAF-TC（Hierarchical Uncertainty-Aware Fusion with
    Temporal Consistency），将 11 种模态按认知层级组织为三层：
      L1 基础感知层（文本/语音/面部）→ 熵驱动不确定性加权
      L2 认知解释层（昼夜节律/认知扭曲/行为）→ 年龄自适应门控
      L3 生理背景层（HRV/呼吸/行为激活/眼动/语音语义）→ 时序一致性平滑

    每个模态优先使用训练好的学习模型；当学习模型不可用时，
    自动降级到基于心理学理论的规则引擎（如 NRC 词典、FACS 规则、
    Scherer 声学标志物等），保证系统始终可用。

    支持年龄差异化校准：传入 age_group 或 age 后，
    各模态的关键词库、阈值、融合权重会按年龄段自动调整。
    """

    def __init__(
        self,
        fusion_strategy: str = DEFAULT_FUSION_STRATEGY,
        stacking_model: Optional[StackingFusion] = None,
        age_group: Optional[AgeGroup] = None,
        age: Optional[int] = None,
    ) -> None:
        self._fusion_strategy = fusion_strategy
        self._stacking_model = stacking_model

        # 年龄画像：优先使用显式 age_group，其次从 age 推断
        if age_group is not None:
            self._age_profile: Optional[AgeProfile] = get_age_profile(age_group)
        elif age is not None:
            self._age_profile = get_age_profile_by_age(age)
        else:
            self._age_profile = None

        # 尝试加载真实文本模型
        _try_load_text_model()

        # HUAF-TC 层级融合引擎（默认启用，核心原创算法）
        self._huaf_engine: Optional[HierarchicalFusionEngine] = None
        if fusion_strategy == "hierarchical":
            self._huaf_engine = HierarchicalFusionEngine()
            logger.info("融合引擎: HUAF-TC (Hierarchical Uncertainty-Aware Fusion)")
        elif fusion_strategy != "hierarchical":
            logger.warning(
                f"融合引擎: {fusion_strategy} (降级模式，"
                f"建议使用 fusion_strategy='hierarchical' 启用 HUAF-TC 原创算法)"
            )

        # 心理数字孪生（个性化因果模型 + 轨迹预测 + 干预优化）
        self._twin_mgr = get_digital_twin_manager()

    # ----------------------------------------------------------
    # 文本情感分析
    # ----------------------------------------------------------

    def analyze_text(self, text: str, age_profile: Optional[AgeProfile] = None) -> EmotionResult:
        """文本情感分析

        优先使用 RoBERTa 模型，不可用时使用 jieba + 情感词典，
        最终兜底降级为关键词规则引擎。
        当提供 age_profile 时，使用年龄专属关键词库。

        Args:
            text: 待分析的文本内容
            age_profile: 年龄画像（可选，覆盖实例配置）

        Returns:
            EmotionResult: 5维情绪概率分布 + 证据
        """
        profile = age_profile or self._age_profile
        if _text_model_available:
            try:
                probs, engine, matched = _real_text_predict(text)
                # 找到主导情绪
                emotion_labels = ['快乐', '悲伤', '焦虑', '愤怒', '中性']
                dominant_idx = max(range(5), key=lambda i: probs[i])
                dominant = emotion_labels[dominant_idx]
                confidence = probs[dominant_idx]

                # 根据引擎类型生成不同的 evidence
                if engine == "roberta":
                    engine_tag = "RoBERTa"
                elif engine == "lexicon":
                    engine_tag = "jieba+词典"
                else:
                    engine_tag = "ML模型"

                evidence_parts = [f"文本情感分析（{engine_tag}）：主导情绪={dominant}，置信度={confidence:.2%}"]
                if matched:
                    evidence_parts.append(f"匹配词: {', '.join(matched[:5])}")
                if profile is not None:
                    evidence_parts.append(f"年龄校准: {profile.age_group.value}")

                return self._with_crisis_screen(
                    EmotionResult(
                        text_emotion_probs=probs,
                        audio_risk_prob=None,
                        confidence=round(confidence, 4),
                        timestamp=time.time(),
                        evidence=evidence_parts,
                    ),
                    text,
                    profile,
                )
            except Exception as e:
                logger.warning(f"文本模型推理失败，降级到规则引擎: {e}")

        # 降级：基于关键词的规则引擎（支持年龄差异化）
        # 注意：此分支在正常「无 RoBERTa 模型」时不会被走到 —— 那种情况由
        # text_emotion.predict 内部的词典引擎承担；本分支是 predict() 抛异常后的最后兜底。
        return self._with_crisis_screen(
            self._rule_based_text_analysis(text, profile), text, profile
        )

    def _rule_based_text_analysis(self, text: str, age_profile: Optional[AgeProfile] = None) -> EmotionResult:
        """[降级路径] 基于关键词的文本分析 —— 当 RoBERTa 模型不可用时自动启用

        这是规则引擎降级方案，基于 C-SSRS 危机关键词、NRC 情感词典
        和年龄专属俚语库进行文本分析。当训练好的 RoBERTa 模型可用时，
        系统会自动使用学习模型替代此规则引擎。

        当提供 age_profile 时，使用该年龄段的专属关键词库。
        """
        # 选择关键词库：有年龄画像则用年龄专属词库，否则用默认通用词库
        if age_profile is not None:
            tp = age_profile.text
            crisis_keywords = tp.crisis_keywords
            anxiety_keywords = tp.anxiety_keywords
            depression_keywords = tp.depression_keywords
            anger_keywords = tp.anger_keywords
            # 额外加入俚语作为情绪信号（俚语多表示情绪波动）
            slang_bonus = len([w for w in tp.slang_keywords if w in text.lower()]) * 0.02
        else:
            # 综合危机关键词库（覆盖直接/间接/隐晦/青少年危机表达）
            # 综合危机关键词库见模块级常量 _DEFAULT_CRISIS_KEYWORDS
            crisis_keywords = _DEFAULT_CRISIS_KEYWORDS
            anxiety_keywords = [
                # ── 基础情绪词 ──
                '焦虑', '紧张', '担心', '害怕', '恐惧', '不安',
                '惶恐', '惊慌', '心慌', '忐忑', '心神不宁',
                # ── 躯体化症状 ──
                '心跳', '心跳好快', '手抖', '头疼', '肚子疼', '出汗',
                '喘不过气', '胸闷', '呼吸困难', '头晕',
                '心悸', '冒冷汗', '手脚发麻', '胃疼', '恶心想吐',
                '喉咙堵', '浑身发抖', '腿软',
                # ── 睡眠问题 ──
                '失眠', '睡不着', '做噩梦', '早醒',
                '半夜惊醒', '翻来覆去睡不着', '入睡困难',
                # ── 压力表达 ──
                '压力好大', '压力大', '压力山大', '喘不过气',
                '要疯了', '要崩溃了', '受不了',
                '快撑不住', '精神紧绷', '神经紧绷',
                # ── 场景触发 ──
                '考试', '面试', '成绩', '挂科', '毕不了业',
                '社交恐惧', '害怕见人',
                '上台', '演讲', '汇报', '答辩',
                '被点名', '被提问', '被叫起来',
                # ── 强迫/反复确认 ──
                '反复检查', '总担心', '控制不住地想',
                '万一', '要是...怎么办',
            ]
            depression_keywords = [
                # ── 基础情绪词 ──
                '低落', '难过', '悲伤', '疲惫', '哭泣', '哭了', '想哭',
                '心酸', '心酸酸的', '泪目',
                # ── 无意义/空虚 ──
                '没意思', '没用', '没有意义', '空虚', '无聊',
                '迷茫', '无趣', '无动力', '提不起劲',
                '什么都没意义', '做什么都没意义', '干什么都没劲',
                # ── 绝望/黑暗 ──
                '绝望', '黑暗', '灰色', '痛苦', '崩溃',
                '撑不下去', '撑不住',
                '看不到未来', '没有未来', '前途暗淡',
                # ── 孤独/无助 ──
                '孤独', '无助', '孤单', '一个人', '没人理我', '没人理解',
                '被抛弃', '被遗忘', '被忽略', '透明人',
                '没有朋友', '孤零零', '形单影只',
                # ── 疲惫/麻木 ──
                '好累', '心累', '身心疲惫', '累', '麻木',
                '不开心', '郁闷', '丧', 'emo',
                '精疲力竭', '筋疲力尽', '心力交瘁',
                '感觉不到快乐', '笑不出来',
                # ── 自我否定 ──
                '我不行', '我太差', '我好笨', '我什么都做不好',
                '都是我的错', '怪我自己', '怪我不好',
                '不够好', '差劲', '一无是处',
                # ── 行为信号 ──
                '不想动', '不想说话', '不想出门', '不想吃饭',
                '行尸走肉', '封闭', '把自己关起来',
                '整天躺着', '哪都不想去', '谁都不想见',
                '暴饮暴食', '吃很多', '睡不着',
            ]
            anger_keywords = [
                # ── 基础情绪词 ──
                '生气', '愤怒', '气死', '恼火', '恼怒',
                '火大', '来气', '窝火', '憋屈',
                # ── 不满/不公 ──
                '过分', '受够了', '不公平', '凭什么',
                '太离谱', '欺人太甚', '欺人太甚',
                '看不上', '看不惯', '忍不了',
                # ── 厌恶/排斥 ──
                '恨', '讨厌', '恶心', '烦', '烦死',
                '厌恶', '反感', '嫌弃', '作呕',
                # ── 爆发表达 ──
                '暴躁', '爆发', '忍无可忍', '窒息',
                '气死了', '烦死了', '好气', '好烦',
                '怒火中烧', '咬牙切齿', '肺都要气炸',
                '要爆炸了', '心态崩了', '炸了',
                # ── 攻击/对抗 ──
                '不想理你', '滚', '闭嘴', '别烦我',
                '少来', '别装了', '你懂什么',
                '站着说话不腰疼', '说风凉话',
                # ── 抱怨/指责 ──
                '总是这样', '每次都这样', '从来不考虑',
                '太自私', '太恶心', '真够可以的',
            ]
            slang_bonus = 0.0

        text_lower = text.lower()
        crisis_count = sum(1 for w in crisis_keywords if w in text_lower)
        anxiety_count = sum(1 for w in anxiety_keywords if w in text_lower)
        depression_count = sum(1 for w in depression_keywords if w in text_lower)
        anger_count = sum(1 for w in anger_keywords if w in text_lower)

        # 改进的概率计算公式：
        # 使用 count/(count+1) 替代 count/total，确保单个关键词也能产生足够信号
        # 危机关键词权重最高（1.0），因为危机检测是安全第一
        crisis_signal = crisis_count / (crisis_count + 1) * 1.0 if crisis_count > 0 else 0.0
        depression_signal = depression_count / (depression_count + 1) * 0.8 if depression_count > 0 else 0.0
        anxiety_signal = anxiety_count / (anxiety_count + 1) * 0.8 if anxiety_count > 0 else 0.0
        anger_signal = anger_count / (anger_count + 1) * 0.8 if anger_count > 0 else 0.0

        # [快乐, 悲伤, 焦虑, 愤怒, 中性]
        # 危机信号平分到悲伤和焦虑（危机既可能是绝望也可能是恐慌）
        probs = [
            0.05 + slang_bonus * 0.5,
            depression_signal + crisis_signal * 0.5,
            anxiety_signal + crisis_signal * 0.5 + slang_bonus,
            anger_signal,
            max(0.05, 1.0 - crisis_signal - depression_signal - anxiety_signal - anger_signal - 0.05 - slang_bonus * 0.5),
        ]
        # 归一化
        s = sum(probs)
        probs = [p / s for p in probs]

        emotion_labels = ['快乐', '悲伤', '焦虑', '愤怒', '中性']
        dominant_idx = max(range(5), key=lambda i: probs[i])
        dominant = emotion_labels[dominant_idx]

        evidence_parts = []
        if crisis_count > 0:
            evidence_parts.append(f"检测到危机关键词({crisis_count}个)")
        if anxiety_count > 0:
            evidence_parts.append(f"焦虑相关词({anxiety_count}个)")
        if depression_count > 0:
            evidence_parts.append(f"抑郁相关词({depression_count}个)")
        if anger_count > 0:
            evidence_parts.append(f"愤怒相关词({anger_count}个)")
        if age_profile is not None:
            evidence_parts.append(f"年龄校准: {age_profile.age_group.value}")
        if not evidence_parts:
            evidence_parts.append("未检测到明显情绪关键词")

        age_tag = f"（{age_profile.age_group.value}）" if age_profile else ""
        return EmotionResult(
            text_emotion_probs=probs,
            audio_risk_prob=None,
            confidence=round(probs[dominant_idx], 4),
            timestamp=time.time(),
            evidence=[f"文本情感分析（规则引擎降级{age_tag}）：主导情绪={dominant}，" + "；".join(evidence_parts)],
            crisis_keywords=[w for w in crisis_keywords if w in text_lower],
        )

    def _with_crisis_screen(
        self,
        result: EmotionResult,
        text: str,
        age_profile: Optional[AgeProfile] = None,
    ) -> EmotionResult:
        """在情感引擎输出之上叠加危机关键词筛查（幂等）

        危机筛查必须独立于情感引擎：无论走学习模型、词典还是规则降级，
        只要字面命中危机词表，就要在结果里留下结构化痕迹，
        供下游 escalation 层触发人工复核。

        幂等性：若 result.crisis_keywords 已有内容（规则引擎路径已筛查过），
        直接返回，避免重复叠加概率。

        Args:
            result: 情感引擎产出
            text: 原始输入文本
            age_profile: 年龄画像（可选）

        Returns:
            EmotionResult: 带危机标记的结果（未命中时原样返回）
        """
        if result.crisis_keywords:
            return result

        count, matched = _screen_crisis_keywords(text, age_profile)
        if count == 0:
            return result

        # 危机信号平分到「悲伤」(idx=1) 与「焦虑」(idx=2) 通道后重新归一化。
        # 不做硬覆盖：保留情感引擎的相对判断，只抬高这两个通道，
        # 真正的升级决策交给 escalation 层，感知层不越权下结论。
        crisis_signal = count / (count + 1)
        probs = list(result.text_emotion_probs)
        probs[1] = probs[1] + crisis_signal * 0.5
        probs[2] = probs[2] + crisis_signal * 0.5
        total = sum(probs)
        probs = [p / total for p in probs]

        evidence = list(result.evidence)
        evidence.append(
            f"[危机筛查] 命中 {count} 个危机关键词：{', '.join(matched[:5])}"
            f"{'…' if len(matched) > 5 else ''}；已抬高悲伤/焦虑通道并标记需人工复核"
        )

        return EmotionResult(
            text_emotion_probs=probs,
            audio_risk_prob=result.audio_risk_prob,
            confidence=result.confidence,
            timestamp=result.timestamp,
            evidence=evidence,
            crisis_keywords=matched,
        )

    # ----------------------------------------------------------
    # 语音情感分析
    # ----------------------------------------------------------

    def analyze_audio(self, wav_path: str, age_profile: Optional[AgeProfile] = None) -> EmotionResult:
        """语音情感分析 —— 使用真实 librosa 声学特征提取

        提取 F0/能量/语速/停顿等声学标志物，
        通过经验规则映射为情绪风险概率。
        当提供 age_profile 时，使用年龄专属基频基线和阈值。

        Args:
            wav_path: WAV 音频文件路径
            age_profile: 年龄画像（可选，覆盖实例配置）

        Returns:
            EmotionResult: 语音风险概率 + 声学证据

        Raises:
            FileNotFoundError: 音频文件不存在
        """
        profile = age_profile or self._age_profile
        import os
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"音频文件不存在: {wav_path}")

        try:
            from perception.voice.acoustic import extract_acoustic_features, features_to_emotion_risk
            # 提取真实声学特征
            acoustic = extract_acoustic_features(wav_path)

            if acoustic.valid:
                # 将声学特征映射为风险概率（传入年龄画像校准）
                risk_prob = features_to_emotion_risk(acoustic)

                # 根据声学特征生成证据
                evidence_parts = []
                # 年龄校准后的基频判定
                f0_anx_thr = profile.voice.f0_anxiety_threshold if profile else 250.0
                f0_dep_thr = profile.voice.f0_depression_threshold if profile else 150.0
                if acoustic.f0_mean < f0_dep_thr:
                    evidence_parts.append(f"低基频({acoustic.f0_mean:.0f}Hz，抑郁指标)")
                elif acoustic.f0_mean > f0_anx_thr:
                    evidence_parts.append(f"高基频({acoustic.f0_mean:.0f}Hz，焦虑指标)")
                sr_anx_thr = profile.voice.speech_rate_anxiety_threshold if profile else 5.0
                sr_dep_lo = profile.voice.speech_rate_normal_range[0] if profile else 2.5
                if acoustic.speech_rate > sr_anx_thr:
                    evidence_parts.append(f"快语速({acoustic.speech_rate:.1f}音节/s，焦虑指标)")
                elif acoustic.speech_rate < sr_dep_lo:
                    evidence_parts.append(f"慢语速({acoustic.speech_rate:.1f}音节/s，抑郁指标)")
                pause_thr = profile.voice.pause_depression_threshold if profile else 0.3
                if acoustic.pause_ratio > pause_thr:
                    evidence_parts.append(f"高停顿比({acoustic.pause_ratio:.0%}，犹豫指标)")
                if acoustic.energy_std > 10:
                    evidence_parts.append(f"高能量波动({acoustic.energy_std:.1f}dB，不稳定指标)")

                if not evidence_parts:
                    evidence_parts.append(f"声学特征正常(F0={acoustic.f0_mean:.0f}Hz, 语速={acoustic.speech_rate:.1f})")

                # 根据声学特征推断情绪倾向（传入年龄画像校准）
                emotion_probs = _acoustic_to_emotion_probs(acoustic, profile)

                age_tag = f"（{profile.age_group.value}）" if profile else ""
                return EmotionResult(
                    text_emotion_probs=emotion_probs,
                    audio_risk_prob=round(risk_prob, 4),
                    confidence=round(min(0.95, 0.5 + acoustic.duration * 0.05), 4),
                    timestamp=time.time(),
                    evidence=[f"语音声学分析（librosa{age_tag}）：风险={risk_prob:.2f}，" + "；".join(evidence_parts)],
                )
            else:
                # 特征提取失败（音频太短或无语音）
                return EmotionResult(
                    text_emotion_probs=[0.20, 0.20, 0.20, 0.20, 0.20],
                    audio_risk_prob=0.0,
                    confidence=0.3,
                    timestamp=time.time(),
                    evidence=["语音分析：音频过短或无有效语音段"],
                )
        except ImportError:
            logger.warning("librosa 未安装，语音分析降级")
            return EmotionResult(
                text_emotion_probs=[0.20, 0.20, 0.20, 0.20, 0.20],
                audio_risk_prob=0.0,
                confidence=0.3,
                timestamp=time.time(),
                evidence=["语音分析降级：librosa 未安装"],
            )
        except Exception as e:
            logger.warning(f"语音分析异常: {e}")
            return EmotionResult(
                text_emotion_probs=[0.20, 0.20, 0.20, 0.20, 0.20],
                audio_risk_prob=0.0,
                confidence=0.3,
                timestamp=time.time(),
                evidence=[f"语音分析异常: {e}"],
            )

    # ----------------------------------------------------------
    # 数字孪生辅助方法
    # ----------------------------------------------------------

    def _build_twin_observation(
        self,
        new_risks: dict[str, Optional[float]],
        text_result: EmotionResult,
    ) -> dict[str, float]:
        """将 11 模态风险分转换为数字孪生观测向量"""
        obs: dict[str, float] = {}
        # 文本情绪：用焦虑+悲伤概率作为风险指标
        probs = text_result.text_emotion_probs
        obs["text_emotion"] = round(max(0.0, min(1.0, probs[2] * 0.6 + probs[1] * 0.4)), 4)
        # 其他模态风险分
        dim_map = {
            "circadian": "circadian",
            "cognitive": "cognitive",
            "hrv": "hrv",
            "breathing": "breathing",
            "behavioral_act": "behavioral_act",
            "eye": "eye",
            "voice_semantics": "voice_semantics",
        }
        for risk_key, dim_key in dim_map.items():
            risk_val = new_risks.get(risk_key)
            if risk_val is not None:
                obs[dim_key] = round(float(risk_val), 4)
        return obs

    def get_digital_twin_prediction(self, user_id: str, horizon: int = 12) -> dict:
        """获取数字孪生预测（轨迹 + 干预决策）

        用于 API 端点调用，返回完整的预测结果。
        """
        traj = self._twin_mgr.predict_trajectory(user_id, horizon)
        decision = self._twin_mgr.get_intervention_decision(user_id, horizon)
        summary = self._twin_mgr.get_summary(user_id)

        return {
            "user_id": user_id,
            "trajectory": {
                "points": [
                    {
                        "time_step": p.time_step,
                        "predicted": p.predicted,
                        "uncertainty": round(p.uncertainty, 4),
                        "crisis_probability": round(p.crisis_probability, 4),
                    }
                    for p in traj.points
                ],
                "max_crisis_risk": round(traj.get_crisis_risk(), 4),
                "deterioration_time": traj.get_deterioration_time(),
            },
            "intervention": {
                "should_intervene": decision.should_intervene,
                "urgency": round(decision.urgency, 4),
                "optimal_delay": decision.optimal_delay,
                "recommended_dimensions": decision.recommended_dimensions,
                "expected_benefit": round(decision.expected_benefit, 4),
                "reasoning": decision.reasoning,
                "chain_to_break": decision.chain_to_break,
            },
            "twin_summary": summary,
        }

    # ----------------------------------------------------------
    # 多模态融合（支持4种模态）
    # ----------------------------------------------------------

    def analyze_multimodal(
        self,
        text: str,
        wav_path: Optional[str] = None,
        facial_features: Optional[dict] = None,
        behavior_features: Optional[dict] = None,
        age_profile: Optional[AgeProfile] = None,
        # 新增 7 模态特征
        circadian_features: Optional[dict] = None,
        cognitive_features: Optional[dict] = None,
        hrv_features: Optional[dict] = None,
        breathing_features: Optional[dict] = None,
        behavioral_activation_features: Optional[dict] = None,
        eye_features: Optional[dict] = None,
        voice_semantics_features: Optional[dict] = None,
        # 数字孪生用户 ID
        user_id: Optional[str] = None,
    ) -> EmotionResult:
        """多模态情感分析（11 模态融合）

        原始 4 模态：文本、语音、面部、行为
        新增 7 模态：昼夜节律、认知扭曲、HRV(rPPG)、呼吸模式、
                    行为激活、眼动模式、语音深层语义

        支持年龄差异化：当提供 age_profile 时，使用年龄专属融合权重。

        Args:
            text: 待分析的文本内容
            wav_path: WAV 音频文件路径（可选）
            facial_features: 面部特征字典（可选）
            behavior_features: 行为特征字典（可选）
            age_profile: 年龄画像（可选，覆盖实例配置）
            circadian_features: 昼夜节律特征（可选）
            cognitive_features: 认知扭曲特征（可选）
            hrv_features: HRV 特征（可选）
            breathing_features: 呼吸模式特征（可选）
            behavioral_activation_features: 行为激活特征（可选）
            eye_features: 眼动模式特征（可选）
            voice_semantics_features: 语音深层语义特征（可选）

        Returns:
            EmotionResult: 融合后的情绪分析结果
        """
        profile = age_profile or self._age_profile

        # 1. 文本分析（始终执行，传入年龄画像）
        text_result = self.analyze_text(text, age_profile=profile)

        # 2. 语音分析（可选，传入年龄画像）
        audio_result: Optional[EmotionResult] = None
        if wav_path is not None:
            try:
                audio_result = self.analyze_audio(wav_path, age_profile=profile)
            except Exception:
                audio_result = None

        # 3. 原始 4 模态风险分
        facial_risk = _extract_facial_risk(facial_features, profile)
        behavior_risk = _extract_behavior_risk(behavior_features, profile)

        # 4. 新增 7 模态风险分
        new_risks: dict[str, Optional[float]] = {}
        new_evidence: dict[str, list[str]] = {}

        # 4a. 昼夜节律
        if circadian_features:
            try:
                cf = CircadianFeatures(**{k: v for k, v in circadian_features.items() if hasattr(CircadianFeatures, k)})
                cr = analyze_circadian_risk(cf, profile)
                new_risks["circadian"] = cr["risk_score"]
                new_evidence["circadian"] = cr["evidence"]
            except Exception as e:
                logger.warning(f"昼夜节律分析失败: {e}")

        # 4b. 认知扭曲
        if cognitive_features:
            try:
                cof = CognitiveDistortionFeatures(**{k: v for k, v in cognitive_features.items() if hasattr(CognitiveDistortionFeatures, k)})
                cor = analyze_cognitive_distortions(cof, profile)
                new_risks["cognitive"] = cor["risk_score"]
                new_evidence["cognitive"] = cor["evidence"]
            except Exception as e:
                logger.warning(f"认知扭曲分析失败: {e}")
        elif text:  # 没有显式传入时，从文本自动分析
            try:
                cof = CognitiveDistortionFeatures(text=text)
                cor = analyze_cognitive_distortions(cof, profile)
                new_risks["cognitive"] = cor["risk_score"]
                new_evidence["cognitive"] = cor["evidence"]
            except Exception:
                pass

        # 4c. HRV
        if hrv_features:
            try:
                hf = HRVFeatures(**{k: v for k, v in hrv_features.items() if hasattr(HRVFeatures, k)})
                hr = analyze_hrv_risk(hf, profile)
                new_risks["hrv"] = hr["risk_score"]
                new_evidence["hrv"] = hr["evidence"]
            except Exception as e:
                logger.warning(f"HRV分析失败: {e}")

        # 4d. 呼吸模式
        if breathing_features:
            try:
                bf = BreathingFeatures(**{k: v for k, v in breathing_features.items() if hasattr(BreathingFeatures, k)})
                br = analyze_breathing_risk(bf, profile)
                new_risks["breathing"] = br["risk_score"]
                new_evidence["breathing"] = br["evidence"]
            except Exception as e:
                logger.warning(f"呼吸模式分析失败: {e}")

        # 4e. 行为激活
        if behavioral_activation_features:
            try:
                baf = BehavioralActivationFeatures(**{k: v for k, v in behavioral_activation_features.items() if hasattr(BehavioralActivationFeatures, k)})
                bar = analyze_behavioral_activation_risk(baf, profile)
                new_risks["behavioral_act"] = bar["risk_score"]
                new_evidence["behavioral_act"] = bar["evidence"]
            except Exception as e:
                logger.warning(f"行为激活分析失败: {e}")

        # 4f. 眼动模式
        if eye_features:
            try:
                ef = EyeMovementFeatures(**{k: v for k, v in eye_features.items() if hasattr(EyeMovementFeatures, k)})
                er = analyze_eye_movement_risk(ef, profile)
                new_risks["eye"] = er["risk_score"]
                new_evidence["eye"] = er["evidence"]
            except Exception as e:
                logger.warning(f"眼动分析失败: {e}")

        # 4g. 语音深层语义
        if voice_semantics_features:
            try:
                vsf = VoiceSemanticsFeatures(**{k: v for k, v in voice_semantics_features.items() if hasattr(VoiceSemanticsFeatures, k)})
                vsr = analyze_voice_semantics_risk(vsf, profile)
                new_risks["voice_semantics"] = vsr["risk_score"]
                new_evidence["voice_semantics"] = vsr["evidence"]
            except Exception as e:
                logger.warning(f"语音语义分析失败: {e}")

        # 5. 融合
        # 5a. HUAF-TC 层级融合（新算法：11 模态层级融合）
        if self._fusion_strategy == "hierarchical" and self._huaf_engine:
            huaf_mods: list[HUAFModalityResult] = []
            # L1 基础感知层
            huaf_mods.append(HUAFModalityResult(
                name="text", probs=list(text_result.text_emotion_probs),
                confidence=text_result.confidence, evidence=list(text_result.evidence),
            ))
            if audio_result:
                huaf_mods.append(HUAFModalityResult(
                    name="voice", probs=list(audio_result.text_emotion_probs),
                    confidence=audio_result.confidence, evidence=list(audio_result.evidence),
                ))
            if facial_risk is not None:
                huaf_mods.append(HUAFModalityResult(
                    name="face", probs=huaf_risk_to_probs(facial_risk),
                    risk_score=facial_risk, confidence=0.7,
                ))
            # L2 认知解释层
            for mod_name, risk_val in [
                ("circadian", new_risks.get("circadian")),
                ("cognitive", new_risks.get("cognitive")),
                ("behavior", behavior_risk),
            ]:
                if risk_val is not None:
                    huaf_mods.append(HUAFModalityResult(
                        name=mod_name, probs=huaf_risk_to_probs(risk_val),
                        risk_score=risk_val, confidence=0.6,
                        evidence=new_evidence.get(mod_name, []),
                    ))
            # L3 生理背景层
            for mod_name, risk_val in [
                ("hrv", new_risks.get("hrv")),
                ("breathing", new_risks.get("breathing")),
                ("behavioral_act", new_risks.get("behavioral_act")),
                ("eye", new_risks.get("eye")),
                ("voice_semantics", new_risks.get("voice_semantics")),
            ]:
                if risk_val is not None:
                    huaf_mods.append(HUAFModalityResult(
                        name=mod_name, probs=huaf_risk_to_probs(risk_val),
                        risk_score=risk_val, confidence=0.6,
                        evidence=new_evidence.get(mod_name, []),
                    ))
            age_group_str = profile.age_group.value if profile else None
            huaf_result = self._huaf_engine.fuse(huaf_mods, age_group=age_group_str)
            extra_evidence = list(huaf_result.evidence)
            if profile is not None:
                extra_evidence.append(f"年龄校准: {profile.age_group.value}")

            # 更新心理数字孪生（将 11 模态风险分输入因果模型）
            twin_obs = self._build_twin_observation(new_risks, text_result)
            if twin_obs and user_id:
                self._twin_mgr.update_observation(user_id, twin_obs)
                extra_evidence.append("[数字孪生] 状态已更新")

            return EmotionResult(
                text_emotion_probs=huaf_result.fused_probs,
                audio_risk_prob=audio_result.audio_risk_prob if audio_result else None,
                confidence=huaf_result.confidence,
                timestamp=huaf_result.timestamp,
                evidence=extra_evidence,
            )

        # 5b. 原始融合策略（weighted_average / stacking / dynamic）
        fused = fuse_multimodal(
            text_result=text_result,
            audio_result=audio_result,
            strategy=self._fusion_strategy,
            stacking_model=self._stacking_model,
            age_profile=profile,
        )

        # 6. 构建证据列表
        extra_evidence = list(fused.evidence)
        if profile is not None:
            extra_evidence.append(f"年龄校准: {profile.age_group.value}")
        if facial_risk is not None:
            extra_evidence.append(f"面部微表情风险分: {facial_risk:.2f}")
        if behavior_risk is not None:
            extra_evidence.append(f"行为模式风险分: {behavior_risk:.2f}")
        for mod_name, ev_list in new_evidence.items():
            for ev in ev_list[:2]:  # 每个模态最多取 2 条证据
                extra_evidence.append(f"[{mod_name}] {ev}")

        # 7. 综合 11 模态风险调整
        adjusted_probs = list(fused.text_emotion_probs)

        # 原始模态调整
        if facial_risk is not None and facial_risk > 0.6:
            adjusted_probs[2] = min(0.95, adjusted_probs[2] + facial_risk * 0.1)
        if behavior_risk is not None and behavior_risk > 0.6:
            adjusted_probs[1] = min(0.95, adjusted_probs[1] + behavior_risk * 0.05)
            adjusted_probs[2] = min(0.95, adjusted_probs[2] + behavior_risk * 0.05)

        # 新模态风险注入（使用年龄专属权重）
        if profile is not None:
            fp = profile.fusion
            new_mod_weights = {
                "circadian": fp.circadian_weight,
                "cognitive": fp.cognitive_weight,
                "hrv": fp.hrv_weight,
                "breathing": fp.breathing_weight,
                "behavioral_act": fp.behavioral_act_weight,
                "eye": fp.eye_weight,
                "voice_semantics": fp.voice_semantics_weight,
            }
        else:
            new_mod_weights = {
                "circadian": 0.06, "cognitive": 0.08, "hrv": 0.05,
                "breathing": 0.05, "behavioral_act": 0.04,
                "eye": 0.03, "voice_semantics": 0.05,
            }

        for mod_name, risk in new_risks.items():
            if risk is not None and risk > 0.1:
                weight = new_mod_weights.get(mod_name, 0.03)
                # 风险注入：焦虑(2)和悲伤(1)维度
                adjusted_probs[2] = min(0.95, adjusted_probs[2] + risk * weight * 0.5)
                adjusted_probs[1] = min(0.95, adjusted_probs[1] + risk * weight * 0.3)

        # 重新归一化
        s = sum(adjusted_probs)
        adjusted_probs = [p / s for p in adjusted_probs]

        # 8. 置信度提升（更多模态 → 更高置信度）
        active_new_mod_count = sum(1 for v in new_risks.values() if v is not None and v > 0)
        confidence_boost = min(0.1, active_new_mod_count * 0.01)
        new_confidence = min(0.99, fused.confidence + confidence_boost)

        # 更新心理数字孪生
        twin_obs = self._build_twin_observation(new_risks, text_result)
        if twin_obs and user_id:
            self._twin_mgr.update_observation(user_id, twin_obs)
            extra_evidence.append("[数字孪生] 状态已更新")

        return EmotionResult(
            text_emotion_probs=adjusted_probs,
            audio_risk_prob=fused.audio_risk_prob,
            confidence=round(new_confidence, 4),
            timestamp=fused.timestamp,
            evidence=extra_evidence,
        )


# ============================================================
# 全局单例
# ============================================================

_perception_service: Optional[PerceptionService] = None


def get_perception_service(
    fusion_strategy: str = DEFAULT_FUSION_STRATEGY,
    age_group: Optional[AgeGroup] = None,
    age: Optional[int] = None,
) -> PerceptionService:
    """获取 PerceptionService 全局单例

    注意：全局单例仅创建一次，后续调用的 age_group/age 参数不会生效。
    如需不同年龄画像的独立实例，请直接构造 PerceptionService。
    """
    global _perception_service
    if _perception_service is None:
        _perception_service = PerceptionService(
            fusion_strategy=fusion_strategy,
            age_group=age_group,
            age=age,
        )
    return _perception_service
