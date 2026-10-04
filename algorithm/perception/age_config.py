"""
年龄分层配置中心 —— 青少年多模态情绪感知的年龄差异化校准

基于发展心理学研究（Erikson, 1968; Casey et al., 2008），
青少年在认知、情绪表达、语言习惯上存在显著的年龄差异。
本模块为三个年龄阶段提供差异化的感知参数配置。

年龄分组：
- EARLY_ADOLESCENT (12-15岁, 初中): 情绪波动大, 俚语丰富, 掩饰能力弱
- MID_ADOLESCENT   (15-18岁, 高中): 抽象思维发展, 开始掩饰, 俚语复杂化
- YOUNG_ADULT      (18-22岁, 大学): 表达成熟, 掩饰能力强, 微表情细微

临床依据：
- Jiang et al. (2020) —— 青少年语音情感识别的年龄差异
- Ekman & Friesen (1971) —— 面部动作编码系统跨年龄适用性
- Nolen-Hoeksema (2012) —— 青少年情绪调节的发展差异

使用方式：
    from perception.age_config import AgeGroup, get_age_profile, infer_age_group

    profile = get_age_profile(AgeGroup.EARLY_ADOLESCENT)
    # 或从实际年龄推断
    group = infer_age_group(14)  # -> AgeGroup.EARLY_ADOLESCENT
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# 年龄分组枚举
# ============================================================

class AgeGroup(str, Enum):
    """青少年年龄分组

    基于 WHO 青少年定义 (10-19岁) 及发展阶段理论划分。
    """
    EARLY_ADOLESCENT = "early_adolescent"   # 12-15岁（初中）
    MID_ADOLESCENT = "mid_adolescent"       # 15-18岁（高中）
    YOUNG_ADULT = "young_adult"             # 18-22岁（大学）


# ============================================================
# 文本模态参数
# ============================================================

@dataclass
class TextAgeParams:
    """文本模态的年龄差异化参数"""

    # 年龄段特有网络俚语/缩写（情绪信号词）
    # 来源：中国互联网青少年用语习惯调查 + 社交平台语料统计
    slang_keywords: list[str] = field(default_factory=list)
    # 年龄段特有危机表达（自杀/自伤相关关键词差异）
    # 来源：C-SSRS (Posner et al., 2011) 青少年适配版
    crisis_keywords: list[str] = field(default_factory=list)
    # 焦虑相关关键词（年龄特异性表达）
    anxiety_keywords: list[str] = field(default_factory=list)
    # 抑郁相关关键词（年龄特异性表达）
    depression_keywords: list[str] = field(default_factory=list)
    # 愤怒相关关键词（年龄特异性表达）
    anger_keywords: list[str] = field(default_factory=list)

    # 表达正式度权重 (0-1)：越低越口语化/非正式
    # 初中=0.2（大量口语/缩写），高中=0.5（混合），大学=0.8（较正式）
    expression_formality: float = 0.5

    # emoji 情感权重：初中生更依赖 emoji 表达情绪
    emoji_sensitivity: float = 0.10

    # 文本长度基线（字/消息）：年龄越小文本越短
    message_length_baseline: float = 30.0


# ============================================================
# 语音模态参数
# ============================================================

@dataclass
class VoiceAgeParams:
    """语音模态的年龄差异化参数

    临床依据：
    - Jiang et al. (2020): 青少年基频随年龄下降
    - Scherer (2003): 声学标志物的年龄校准必要性
    """

    # 基频基线 (Hz)：随年龄下降（声带发育）
    # 初中≈220Hz, 高中≈190Hz, 大学≈170Hz
    f0_baseline: float = 190.0

    # 焦虑判定高基频阈值 (Hz)：初中生正常基频更高
    f0_anxiety_threshold: float = 270.0

    # 抑郁判定低基频阈值 (Hz)：相对于基线的偏移
    f0_depression_threshold: float = 120.0

    # 正常语速范围 (音节/秒)
    speech_rate_normal_range: tuple[float, float] = (3.0, 5.0)

    # 抑郁判定停顿比阈值：初中生停顿容忍度更高
    pause_depression_threshold: float = 0.30

    # 语速焦虑阈值 (音节/秒)
    speech_rate_anxiety_threshold: float = 5.0


# ============================================================
# 面部模态参数
# ============================================================

@dataclass
class FaceAgeParams:
    """面部模态的年龄差异化参数

    临床依据：
    - Ekman (2003): 情绪掩饰能力随年龄增长
    - Williams et al. (2009): 青少年微表情发展差异
    """

    # AU 强度基线 (0-1)：掩饰能力递增 → 基线递减
    # 初中=0.6（表达直接），高中=0.5，大学=0.4（更含蓄）
    au_intensity_baseline: float = 0.5

    # 微表情检测灵敏度阈值 (0-1)：大学需更灵敏（掩饰更多）
    # 初中=0.30，高中=0.25，大学=0.20
    micro_expression_threshold: float = 0.25

    # 掩饰检测权重 (0-1)：AU 强度与表情一致性检测
    # 初中=0.1（少掩饰），高中=0.3，大学=0.5（多掩饰）
    masking_detection_weight: float = 0.3

    # 微表情风险加成系数：频繁微表情 → 增加风险分
    micro_expression_risk_weight: float = 0.05


# ============================================================
# 行为模态参数
# ============================================================

@dataclass
class BehaviorAgeParams:
    """行为模态的年龄差异化参数

    临床依据：
    - Owens et al. (2014): 青少年睡眠模式与年龄的关系
    - Twenge (2017): 数字原住民的行为模式年龄差异
    """

    # 深夜风险阈值 (小时, 24h制)：年龄越大容忍越晚
    # 初中=23:00, 高中=0:00, 大学=1:00
    late_night_hour: int = 0

    # 打字速度基线 (字/分钟)：年龄越小打字越快（手机游戏训练）
    # 初中=45, 高中=40, 大学=35
    typing_speed_baseline: float = 40.0

    # 会话模式权重分配 (频率变化 vs 时段规律性)
    # 初中偏重频率变化 (0.6, 0.4)，大学偏重规律性 (0.3, 0.7)
    session_frequency_weight: float = 0.5
    session_regularity_weight: float = 0.5

    # 社交参与度低风险阈值：年龄越小社交需求越高
    social_engagement_low_threshold: float = 0.2

    # 行为变化敏感度：初中生行为变化更剧烈，阈值更高
    behavior_change_threshold: float = 0.4


# ============================================================
# 昼夜节律参数 (Circadian Rhythm)
# ============================================================

@dataclass
class CircadianAgeParams:
    """昼夜节律模态的年龄差异化参数

    临床依据：
    - Walker & Harvey (2019): 青少年睡眠模式年龄差异
    - DSM-5: 睡眠紊乱是抑郁核心诊断标准
    """

    # 理想就寝时间 (小时, 24h制)
    # 初中=21.5, 高中=22.5, 大学=23.5
    ideal_bedtime: float = 22.5

    # 理想起床时间 (小时, 24h制)
    # 初中=6.5, 高中=6.5, 大学=7.5
    ideal_waketime: float = 6.5

    # 深夜活跃风险起始时间 (小时)
    # 初中=22:00, 高中=23:00, 大学=0:00
    late_activity_risk_hour: int = 23

    # 作息不规律惩罚系数 (0-1)
    irregularity_penalty_weight: float = 0.3

    # 最小睡眠时长需求 (小时)
    # 初中=9.0, 高中=8.5, 大学=8.0
    min_sleep_hours: float = 8.5


# ============================================================
# 认知扭曲参数 (Cognitive Distortion)
# ============================================================

@dataclass
class CognitiveDistortionAgeParams:
    """认知扭曲深度分析的年龄差异化参数

    临床依据：
    - Beck (1979): CBT 认知扭曲理论
    - Garber & Hollon (1991): 青少年认知扭曲发展特征
    """

    # 灾难化思维关键词（年龄特异性表达）
    catastrophizing_keywords: list[str] = field(default_factory=list)
    # 非黑即白关键词
    black_and_white_keywords: list[str] = field(default_factory=list)
    # 过度概括关键词
    overgeneralization_keywords: list[str] = field(default_factory=list)
    # 自我归咎关键词
    self_blame_keywords: list[str] = field(default_factory=list)
    # 无望感关键词
    hopelessness_keywords: list[str] = field(default_factory=list)
    # 读心术/预言家关键词
    mind_reading_keywords: list[str] = field(default_factory=list)
    # 应该陈述关键词
    should_statements_keywords: list[str] = field(default_factory=list)

    # 认知扭曲基础风险阈值 (0-1)
    # 初中生表达直接→易检测, 阈值低; 大学生隐晦→需更灵敏
    base_risk_threshold: float = 0.3

    # 每类扭曲的最大风险贡献 (0-1)
    max_risk_per_category: float = 0.15


# ============================================================
# 心率变异性参数 (HRV via rPPG)
# ============================================================

@dataclass
class HRVAgeParams:
    """心率变异性模态的年龄差异化参数

    临床依据：
    - Thayer & Sternberg (2006): HRV 与情绪调节
    - Marcovitch et al. (2010): 青少年 HRV 发展常模
    """

    # 静息心率基线 (bpm)：随年龄递减
    # 初中=85, 高中=75, 大学=70
    resting_hr_baseline: float = 75.0

    # 焦虑心率阈值 (bpm)
    # 初中=110, 高中=100, 大学=95
    anxiety_hr_threshold: float = 100.0

    # HRV RMSSD 正常基线 (ms)：青少年普遍较高
    # 初中=55, 高中=45, 大学=40
    hrv_rmssd_baseline: float = 45.0

    # 低 HRV 风险阈值 (ms)：低于此值 → 自主神经失调
    # 初中=25, 高中=20, 大学=18
    hrv_low_risk_threshold: float = 20.0

    # 心率变异性风险权重 (0-1)
    hr_variability_risk_weight: float = 0.3


# ============================================================
# 呼吸模式参数 (Breathing Pattern)
# ============================================================

@dataclass
class BreathingAgeParams:
    """呼吸模式模态的年龄差异化参数

    临床依据：
    - Ley (1987): 呼吸模式与焦虑
    - Mezzasalma & Abelson (2001): 青少年过度换气
    """

    # 正常呼吸频率基线 (次/分钟)：随年龄递减
    # 初中=18, 高中=16, 大学=15
    normal_rate_baseline: float = 16.0

    # 焦虑呼吸频率阈值 (次/分钟)
    # 初中=24, 高中=22, 大学=20
    anxiety_rate_threshold: float = 22.0

    # 抑郁呼吸频率阈值 (次/分钟)
    # 初中=11, 高中=10, 大学=9
    depression_rate_threshold: float = 10.0

    # 呼吸规律性正常范围 (CV)
    regularity_normal_cv: float = 0.25

    # 叹气风险阈值 (次/分钟)：频繁叹气 → 焦虑/抑郁
    sigh_risk_threshold: float = 3.0


# ============================================================
# 行为激活参数 (Behavioral Activation)
# ============================================================

@dataclass
class BehavioralActivationAgeParams:
    """行为激活水平模态的年龄差异化参数

    临床依据：
    - Martell et al. (2001): 行为激活理论 — 抑郁的核心是正性强化减少
    - Masi et al. (2011): 青少年社交行为与心理健康
    """

    # 每日最低互动次数阈值
    # 初中=15, 高中=10, 大学=8
    min_daily_interactions: int = 10

    # 功能模块探索率基线 (0-1)：好奇心指标
    # 初中=0.6, 高中=0.5, 大学=0.4
    exploration_rate_baseline: float = 0.5

    # 社交退缩风险阈值 (0-1)：低于此值 → 退缩风险
    social_withdrawal_threshold: float = 0.2

    # 活动时段集中度阈值 (0-1)：活动过于集中 → 刻板行为
    activity_concentration_threshold: float = 0.7

    # 行为变化检测窗口 (天)
    change_detection_window: int = 7

    # 退缩风险权重 (0-1)
    withdrawal_risk_weight: float = 0.3


# ============================================================
# 眼动模式参数 (Eye Movement)
# ============================================================

@dataclass
class EyeMovementAgeParams:
    """眼动模式模态的年龄差异化参数

    临床依据：
    - Armstrong & Olatunji (2012): 焦虑青少年的负面注意偏向
    - Peckham et al. (2010): 抑郁注视模式
    """

    # 正常眨眼频率基线 (次/分钟)
    # 初中=18, 高中=16, 大学=15
    blink_rate_baseline: float = 16.0

    # 低眨眼风险阈值 (次/分钟)：过低 → 注意力过度集中/解离
    low_blink_threshold: float = 8.0

    # 高眨眼风险阈值 (次/分钟)：过高 → 焦虑/干眼
    high_blink_threshold: float = 30.0

    # 向下注视风险阈值 (比例, 0-1)
    # 持续向下注视 → 回避/抑郁
    downward_gaze_risk_threshold: float = 0.5

    # 注意力分散阈值 (0-1)：注视频繁转移 → 焦虑
    attention_scatter_threshold: float = 0.6

    # 眼动风险权重 (0-1)
    eye_risk_weight: float = 0.2


# ============================================================
# 语音深层语义参数 (Voice Deep Semantics)
# ============================================================

@dataclass
class VoiceSemanticsAgeParams:
    """语音深层语义模态的年龄差异化参数

    临床依据：
    - Rude et al. (2004): 第一人称代词与抑郁
    - Stirman & Pennebaker (2001): 语言标记与自杀风险
    """

    # 第一人称单数风险阈值 (比例, 0-1)
    # 过度使用"我" → 自我关注过度 → 抑郁风险
    first_person_singular_threshold: float = 0.15

    # 第一人称复数风险阈值 (比例)
    # 少用"我们" → 社交疏离
    first_person_plural_threshold: float = 0.02

    # 绝对化表达风险阈值 (比例)
    absolutist_expression_threshold: float = 0.05

    # 自我指涉总密度风险阈值 (0-1)
    self_referential_density_threshold: float = 0.25

    # 消极情感词比例阈值 (0-1)
    negative_affect_ratio_threshold: float = 0.20

    # 语义风险权重 (0-1)
    semantic_risk_weight: float = 0.3


# ============================================================
# 融合策略参数
# ============================================================

@dataclass
class FusionAgeParams:
    """融合策略的年龄差异化参数

    设计原理：
    - 初中生面部掩饰弱 → 面部权重大 (0.20)
    - 大学生掩饰强 → 面部权重低 (0.08)，文本权重高 (0.40)
    - 11 模态总权重 = 1.0
    """

    # 11 模态权重 (总和 = 1.0)
    text_weight: float = 0.30
    audio_weight: float = 0.12
    face_weight: float = 0.12
    behavior_weight: float = 0.10
    # 新增 7 模态
    circadian_weight: float = 0.06
    cognitive_weight: float = 0.08
    hrv_weight: float = 0.05
    breathing_weight: float = 0.05
    behavioral_act_weight: float = 0.04
    eye_weight: float = 0.03
    voice_semantics_weight: float = 0.05

    # 风险阈值微调偏移量
    # 初中=-0.05（更敏感，减少漏报），大学=+0.05（减少误报）
    risk_threshold_offset: float = 0.0


# ============================================================
# 综合年龄画像
# ============================================================

@dataclass
class AgeProfile:
    """青少年年龄画像 —— 聚合所有模态的年龄差异化参数

    11 模态：文本、语音、面部、行为 + 昼夜节律、认知扭曲、
    HRV(rPPG)、呼吸模式、行为激活、眼动模式、语音深层语义
    """

    age_group: AgeGroup = AgeGroup.MID_ADOLESCENT
    age_range: tuple[int, int] = (15, 18)
    description: str = ""

    # 原始 4 模态
    text: TextAgeParams = field(default_factory=TextAgeParams)
    voice: VoiceAgeParams = field(default_factory=VoiceAgeParams)
    face: FaceAgeParams = field(default_factory=FaceAgeParams)
    behavior: BehaviorAgeParams = field(default_factory=BehaviorAgeParams)
    fusion: FusionAgeParams = field(default_factory=FusionAgeParams)

    # 新增 7 模态
    circadian: CircadianAgeParams = field(default_factory=CircadianAgeParams)
    cognitive: CognitiveDistortionAgeParams = field(default_factory=CognitiveDistortionAgeParams)
    hrv: HRVAgeParams = field(default_factory=HRVAgeParams)
    breathing: BreathingAgeParams = field(default_factory=BreathingAgeParams)
    behavioral_activation: BehavioralActivationAgeParams = field(default_factory=BehavioralActivationAgeParams)
    eye: EyeMovementAgeParams = field(default_factory=EyeMovementAgeParams)
    voice_semantics: VoiceSemanticsAgeParams = field(default_factory=VoiceSemanticsAgeParams)


# ============================================================
# 各年龄段配置实例
# ============================================================

def _build_early_adolescent_profile() -> AgeProfile:
    """初中阶段 (12-15岁) 年龄画像

    特征：情绪波动大、网络俚语多、表情掩饰能力弱、声调偏高
    """
    return AgeProfile(
        age_group=AgeGroup.EARLY_ADOLESCENT,
        age_range=(12, 15),
        description="初中阶段 (12-15岁)：情绪波动大，表达直接，依赖网络俚语和 emoji",
        text=TextAgeParams(
            slang_keywords=[
                # 初中生高频网络用语（情绪信号词）
                "yyds", "绝绝子", "awsl", "xswl", "破防了",
                "emo了", "裂开", "蚌埠住了", "栓Q", "芭比Q",
                "无语子", "好家伙", "真的会谢", "我直接", "DNA动了",
                "6到飞起", "上头", "下头", "塌房", "摆烂",
                "尊嘟假嘟", "i人", "e人", "city不city",
            ],
            crisis_keywords=[
                # 初中生危机表达（更直接、更口语化）
                "不想活", "去死", "跳楼", "割腕", "自杀",
                "活不下去", "死了算了", "不想上学", "消失",
                "想死", "自残", "划手", "不想醒来",
                "没人要我", "我是废物", "活着没意思",
                "不想活了", "自伤", "划手臂",
                "撑不住了", "撑不下去", "好累不想动",
                "没人会在意", "没有意义", "累了",
                # 新增：更口语化/直接的表达
                "不想读书了", "不想念了", "读书没用",
                "爸妈对不起", "都是我的错",
                "你们不要我了", "反正没人关心我",
                "活着干嘛", "有什么用",
                "我什么都不是", "我就是个累赘",
                "不想回来", "不想回家",
                "把我丢掉算了", "多我一个不多",
            ],
            anxiety_keywords=[
                "好紧张", "害怕", "要考试了", "考砸了", "完了完了",
                "心跳好快", "睡不着", "好焦虑", "怎么办", "要疯了",
                "压力好大", "作业好多", "来不及了",
                "手抖", "肚子疼", "头疼", "喘不过气",
                "不敢去学校", "社交恐惧", "做噩梦",
                # 新增：初中生典型焦虑场景
                "被老师叫", "被点名", "家长会被叫家长",
                "考不好怎么办", "考不上高中", "分班",
                "同学笑话我", "被欺负", "被孤立",
                "不敢举手", "上课不敢回答问题",
            ],
            depression_keywords=[
                "好无聊", "没意思", "不想动", "好累", "不开心",
                "哭了", "难过", "好烦", "灰色", "一个人",
                "没人理我", "好孤单", "想哭", "心情差",
                "心累", "郁闷", "迷茫", "不想说话",
                "不想吃饭", "没动力", "提不起劲",
                # 新增：初中生抑郁信号
                "不想去学校", "不想见同学",
                "整天玩手机", "什么都不想做",
                "好没劲", "没朋友", "被排挤",
                "心情很糟", "烦心事好多", "偷偷哭",
            ],
            anger_keywords=[
                "气死了", "烦死了", "讨厌", "不公平", "凭什么",
                "好过分", "受够了", "气死我了", "好气", "好烦啊",
                "不想理你", "滚", "闭嘴",
                "别烦我", "暴躁", "恼火", "恶心",
                # 新增：初中生愤怒表达
                "太过分了", "欺负人", "以大欺小",
                "你们都不懂", "大人最烦了",
                "管得着吗", "少管闲事", "讨厌死了",
            ],
            expression_formality=0.2,
            emoji_sensitivity=0.15,
            message_length_baseline=25.0,
        ),
        voice=VoiceAgeParams(
            f0_baseline=220.0,
            f0_anxiety_threshold=300.0,
            f0_depression_threshold=140.0,
            speech_rate_normal_range=(3.0, 5.5),
            pause_depression_threshold=0.35,
            speech_rate_anxiety_threshold=5.5,
        ),
        face=FaceAgeParams(
            au_intensity_baseline=0.6,
            micro_expression_threshold=0.30,
            masking_detection_weight=0.1,
            micro_expression_risk_weight=0.04,
        ),
        behavior=BehaviorAgeParams(
            late_night_hour=23,
            typing_speed_baseline=45.0,
            session_frequency_weight=0.6,
            session_regularity_weight=0.4,
            social_engagement_low_threshold=0.25,
            behavior_change_threshold=0.35,
        ),
        circadian=CircadianAgeParams(
            ideal_bedtime=21.5, ideal_waketime=6.5,
            late_activity_risk_hour=22, irregularity_penalty_weight=0.35,
            min_sleep_hours=9.0,
        ),
        cognitive=CognitiveDistortionAgeParams(
            catastrophizing_keywords=["完了", "完蛋", "全毁了", "天塌了", "太可怕了"],
            black_and_white_keywords=["总是", "从不", "永远", "绝对", "一定", "完全"],
            overgeneralization_keywords=["每次都", "从来不", "所有人", "没有人"],
            self_blame_keywords=["都是我的错", "怪我", "我不行", "我太差", "废物", "垃圾"],
            hopelessness_keywords=["没有希望", "不会好的", "没救了", "不可能", "没办法"],
            mind_reading_keywords=["他们肯定觉得", "别人都笑我", "大家都知道", "看不起我"],
            should_statements_keywords=["应该", "必须", "一定要", "不应该", "不可以"],
            base_risk_threshold=0.25, max_risk_per_category=0.15,
        ),
        hrv=HRVAgeParams(
            resting_hr_baseline=85.0, anxiety_hr_threshold=110.0,
            hrv_rmssd_baseline=55.0, hrv_low_risk_threshold=25.0,
            hr_variability_risk_weight=0.3,
        ),
        breathing=BreathingAgeParams(
            normal_rate_baseline=18.0, anxiety_rate_threshold=24.0,
            depression_rate_threshold=11.0, regularity_normal_cv=0.25,
            sigh_risk_threshold=3.0,
        ),
        behavioral_activation=BehavioralActivationAgeParams(
            min_daily_interactions=15, exploration_rate_baseline=0.6,
            social_withdrawal_threshold=0.25, activity_concentration_threshold=0.7,
            change_detection_window=7, withdrawal_risk_weight=0.3,
        ),
        eye=EyeMovementAgeParams(
            blink_rate_baseline=18.0, low_blink_threshold=8.0,
            high_blink_threshold=30.0, downward_gaze_risk_threshold=0.5,
            attention_scatter_threshold=0.6, eye_risk_weight=0.2,
        ),
        voice_semantics=VoiceSemanticsAgeParams(
            first_person_singular_threshold=0.15,
            first_person_plural_threshold=0.02,
            absolutist_expression_threshold=0.05,
            self_referential_density_threshold=0.25,
            negative_affect_ratio_threshold=0.20,
            semantic_risk_weight=0.3,
        ),
        fusion=FusionAgeParams(
            text_weight=0.25,
            audio_weight=0.12,
            face_weight=0.20,
            behavior_weight=0.10,
            circadian_weight=0.06,
            cognitive_weight=0.08,
            hrv_weight=0.05,
            breathing_weight=0.04,
            behavioral_act_weight=0.03,
            eye_weight=0.02,
            voice_semantics_weight=0.05,
            risk_threshold_offset=-0.05,
        ),
    )


def _build_mid_adolescent_profile() -> AgeProfile:
    """高中阶段 (15-18岁) 年龄画像

    特征：抽象思维发展、开始情绪掩饰、俚语复杂化、声调趋稳
    """
    return AgeProfile(
        age_group=AgeGroup.MID_ADOLESCENT,
        age_range=(15, 18),
        description="高中阶段 (15-18岁)：抽象思维发展，开始掩饰情绪，表达更复杂",
        text=TextAgeParams(
            slang_keywords=[
                # 高中生高频网络用语（更抽象、更隐喻）
                "emo", "内卷", "躺平", "破防", "精神状态",
                "发疯文学", "摆烂", "社恐", "i人e人", "电子榨菜",
                "精神状态不太好", "发疯中", "已黑化", "在逃公主",
                "纯爱战士", "大冤种", "显眼包", "嘴替", "CPU",
                "抽象", "上价值", "格局打开", "人生建议",
            ],
            crisis_keywords=[
                # 高中生危机表达（更隐晦、文学化）
                "消失好了", "不想存在", "累了", "没有意义",
                "活着好累", "如果我不在了", "解脱", "看不到希望",
                "一切都没意义", "不想面对", "世界没有我会更好",
                "撑不住了", "好想逃", "无尽的黑暗", "深渊",
                "自残", "割腕", "不想活", "想死",
                "自伤", "划手", "不想活了", "死了算了",
                "活着没意思", "没人要我", "我是废物",
                "了结", "一了百了", "结束这一切",
                "撑不下去", "好累不想动", "看不到出路",
                # 新增：高中生隐晦/文艺化表达
                "人间不值得", "生而为人我很抱歉",
                "活着像行尸走肉", "灵魂已经死了",
                "心已经空了", "什么都感觉不到了",
                "不想努力了", "放弃一切",
                "对不起爸妈", "让你们失望了",
                "反正没人在乎", "多我一个不多",
                "我就是个笑话", "谁都不会心疼",
                "活着干嘛", "有什么用",
            ],
            anxiety_keywords=[
                "焦虑", "紧张", "高考", "排名", "考不好",
                "压力山大", "失眠", "担心", "害怕失败", "喘不过气",
                "内卷", "竞争", "跟不上", "来不及", "崩溃",
                "手抖", "头疼", "心跳好快", "做噩梦",
                "社交恐惧", "害怕见人", "要崩溃了",
                # 新增：高中生典型焦虑场景
                "模考", "月考", "期中考", "期末考",
                "考不上大学", "落榜", "复读",
                "被比较", "别人家的孩子",
                "父母期望", "对不起父母",
                "同学关系", "被孤立", "被排挤",
                "来不及复习", "考砸了怎么办",
            ],
            depression_keywords=[
                "低落", "emo", "丧", "空虚", "没动力",
                "没意思", "疲惫", "迷茫", "孤独", "无趣",
                "不想说话", "社恐", "封闭", "灰色", "行尸走肉",
                "心累", "难过", "想哭", "郁闷",
                "提不起劲", "不想动", "麻木", "没人理解",
                # 新增：高中生抑郁信号
                "不想上学", "不想去教室",
                "上课走神", "什么都听不进去",
                "感觉被抛弃", "没人懂我",
                "笑不出来", "感觉不到快乐",
                "整天躺着", "哪都不想去",
                "看不到未来", "没有未来",
            ],
            anger_keywords=[
                "烦", "暴躁", "崩溃", "不公", "凭什么",
                "受够了", "压力", "窒息", "忍耐", "爆发",
                "气死", "无语", "恶心", "讨厌",
                "烦死", "好气", "忍无可忍", "别烦我",
                # 新增：高中生愤怒表达
                "太压抑", "管得太多了",
                "你们根本不懂", "只会逼我学习",
                "分数比人重要", "太不公平了",
                "被误解", "被冤枉", "有苦说不出",
            ],
            expression_formality=0.5,
            emoji_sensitivity=0.10,
            message_length_baseline=35.0,
        ),
        voice=VoiceAgeParams(
            f0_baseline=190.0,
            f0_anxiety_threshold=270.0,
            f0_depression_threshold=120.0,
            speech_rate_normal_range=(3.0, 5.0),
            pause_depression_threshold=0.30,
            speech_rate_anxiety_threshold=5.0,
        ),
        face=FaceAgeParams(
            au_intensity_baseline=0.5,
            micro_expression_threshold=0.25,
            masking_detection_weight=0.3,
            micro_expression_risk_weight=0.05,
        ),
        behavior=BehaviorAgeParams(
            late_night_hour=0,
            typing_speed_baseline=40.0,
            session_frequency_weight=0.5,
            session_regularity_weight=0.5,
            social_engagement_low_threshold=0.2,
            behavior_change_threshold=0.4,
        ),
        circadian=CircadianAgeParams(
            ideal_bedtime=22.5, ideal_waketime=6.5,
            late_activity_risk_hour=23, irregularity_penalty_weight=0.30,
            min_sleep_hours=8.5,
        ),
        cognitive=CognitiveDistortionAgeParams(
            catastrophizing_keywords=["完了", "完蛋", "全毁了", "一切都完了", "太可怕了"],
            black_and_white_keywords=["总是", "从不", "永远不", "绝对", "完全", "根本"],
            overgeneralization_keywords=["每次都", "从来不", "所有人", "没有人", "到处"],
            self_blame_keywords=["都是我的错", "怪我", "我不配", "我不行", "废物", "没用"],
            hopelessness_keywords=["没有希望", "不会好的", "没救了", "不可能", "做不到"],
            mind_reading_keywords=["他们肯定觉得", "别人都", "大家都知道", "看不起"],
            should_statements_keywords=["应该", "必须", "一定要", "不应该"],
            base_risk_threshold=0.30, max_risk_per_category=0.15,
        ),
        hrv=HRVAgeParams(
            resting_hr_baseline=75.0, anxiety_hr_threshold=100.0,
            hrv_rmssd_baseline=45.0, hrv_low_risk_threshold=20.0,
            hr_variability_risk_weight=0.3,
        ),
        breathing=BreathingAgeParams(
            normal_rate_baseline=16.0, anxiety_rate_threshold=22.0,
            depression_rate_threshold=10.0, regularity_normal_cv=0.25,
            sigh_risk_threshold=3.0,
        ),
        behavioral_activation=BehavioralActivationAgeParams(
            min_daily_interactions=10, exploration_rate_baseline=0.5,
            social_withdrawal_threshold=0.2, activity_concentration_threshold=0.7,
            change_detection_window=7, withdrawal_risk_weight=0.3,
        ),
        eye=EyeMovementAgeParams(
            blink_rate_baseline=16.0, low_blink_threshold=8.0,
            high_blink_threshold=30.0, downward_gaze_risk_threshold=0.5,
            attention_scatter_threshold=0.6, eye_risk_weight=0.2,
        ),
        voice_semantics=VoiceSemanticsAgeParams(
            first_person_singular_threshold=0.15,
            first_person_plural_threshold=0.02,
            absolutist_expression_threshold=0.05,
            self_referential_density_threshold=0.25,
            negative_affect_ratio_threshold=0.20,
            semantic_risk_weight=0.3,
        ),
        fusion=FusionAgeParams(
            text_weight=0.30,
            audio_weight=0.12,
            face_weight=0.12,
            behavior_weight=0.10,
            circadian_weight=0.06,
            cognitive_weight=0.08,
            hrv_weight=0.05,
            breathing_weight=0.05,
            behavioral_act_weight=0.04,
            eye_weight=0.03,
            voice_semantics_weight=0.05,
            risk_threshold_offset=0.0,
        ),
    )


def _build_young_adult_profile() -> AgeProfile:
    """大学阶段 (18-22岁) 年龄画像

    特征：表达成熟、掩饰能力强、微表情更细微、声调稳定
    """
    return AgeProfile(
        age_group=AgeGroup.YOUNG_ADULT,
        age_range=(18, 22),
        description="大学阶段 (18-22岁)：表达成熟，善于掩饰，需要更精细的检测",
        text=TextAgeParams(
            slang_keywords=[
                # 大学生高频网络用语（更成熟、更反讽）
                "精神内耗", "PUA", "情绪价值", "边界感", "松弛感",
                "赛博朋克", "数字游民", "润", "开摆", "已读乱回",
                "精神状态堪忧", "确诊为", "人生是旷野", "发疯",
                "班味", "牛马", "打工人", "工具人", "社会性死亡",
                "情绪稳定", "内核强大", "课题分离", "自我疗愈",
            ],
            crisis_keywords=[
                # 大学生危机表达（更隐晦、哲学化、学术化）
                "存在没有意义", "虚无", "解脱", "结束这一切",
                "活着是一种负担", "如果世界没有我", "消失", "倦怠",
                "精神内耗到极限", "撑不下去", "看不到出路",
                "深渊", "无尽的疲惫", "存在的荒谬", "了结",
                "自杀", "自残", "不想活", "了断",
                "自伤", "划手", "想死", "死了算了",
                "活着没意思", "没人要我", "我是废物",
                "一了百了", "结束一切", "撑不住了",
                "不想面对", "看不到希望", "好累不想动",
                "活着好累", "不想存在", "没有意义",
                # 新增：大学生隐晦/哲学化表达
                "人间不值得", "生而为人我很抱歉",
                "活着像行尸走肉", "灵魂已经死了",
                "心已经空了", "什么都感觉不到了",
                "不想努力了", "放弃一切",
                "请原谅我", "我不值得", "我配不上",
                "你们会过得更好", "没有我的日子",
                "反正没人在乎", "多我一个不多",
                "我就是个笑话", "谁都不会心疼",
                "活着干嘛", "有什么用",
            ],
            anxiety_keywords=[
                "焦虑", "内耗", "deadline", "绩点", "考研",
                "就业", "迷茫", "peer pressure", "同辈压力",
                "失眠", "压力", "喘不过气", "崩溃边缘", "burnout",
                "手抖", "头疼", "心跳好快", "胸闷",
                "社交恐惧", "害怕失败", "要崩溃了", "做噩梦",
                # 新增：大学生典型焦虑场景
                "秋招", "春招", "简历", "面试",
                "保研", "推免", "论文", "开题", "答辩",
                "实习", "找不到工作", "失业",
                "被卷", "被比较", "同龄人",
                "经济压力", "房租", "负债",
            ],
            depression_keywords=[
                "低落", "空虚", "倦怠", "无力感", "意义感缺失",
                "抑郁", "丧", "emo", "疲惫", "孤独",
                "麻木", "行尸走肉", "提不起劲", "灰色",
                "心累", "难过", "想哭", "郁闷",
                "没人理解", "不想说话", "不想动", "无动力",
                # 新增：大学生抑郁信号
                "感觉不到快乐", "笑不出来",
                "整天躺着", "哪都不想去",
                "看不到未来", "没有未来",
                "感觉被抛弃", "没人懂我",
                "社交回避", "不想见人",
                "意义感缺失", "价值感缺失",
            ],
            anger_keywords=[
                "愤怒", "不公", "体制", "压迫", "窒息",
                "受够了", "崩溃", "暴躁", "忍无可忍", "恶心",
                "无语", "过分", "荒谬",
                "烦死", "好气", "凭什么", "别烦我",
                # 新增：大学生愤怒表达
                "内卷", "躺平", "摆烂",
                "社会不公", "阶层固化",
                "被剥削", "被压榨", "工具人",
                "你们根本不懂", "站着说话不腰疼",
                "太自私", "太恶心", "真够可以的",
            ],
            expression_formality=0.8,
            emoji_sensitivity=0.05,
            message_length_baseline=45.0,
        ),
        voice=VoiceAgeParams(
            f0_baseline=170.0,
            f0_anxiety_threshold=250.0,
            f0_depression_threshold=110.0,
            speech_rate_normal_range=(2.8, 4.8),
            pause_depression_threshold=0.28,
            speech_rate_anxiety_threshold=4.8,
        ),
        face=FaceAgeParams(
            au_intensity_baseline=0.4,
            micro_expression_threshold=0.20,
            masking_detection_weight=0.5,
            micro_expression_risk_weight=0.06,
        ),
        behavior=BehaviorAgeParams(
            late_night_hour=1,
            typing_speed_baseline=35.0,
            session_frequency_weight=0.3,
            session_regularity_weight=0.7,
            social_engagement_low_threshold=0.15,
            behavior_change_threshold=0.45,
        ),
        circadian=CircadianAgeParams(
            ideal_bedtime=23.5, ideal_waketime=7.5,
            late_activity_risk_hour=0, irregularity_penalty_weight=0.25,
            min_sleep_hours=8.0,
        ),
        cognitive=CognitiveDistortionAgeParams(
            catastrophizing_keywords=["全完了", "一切都毁了", "无法挽回", "灾难"],
            black_and_white_keywords=["总是", "从不", "绝对", "完全", "根本", "所有"],
            overgeneralization_keywords=["每次", "从来", "所有人", "没有人", "到处"],
            self_blame_keywords=["都是我的错", "我不配", "我不行", "太差", "废物", "没用"],
            hopelessness_keywords=["没有希望", "不会好", "没救", "不可能", "做不到", "无望"],
            mind_reading_keywords=["他们觉得", "别人都", "大家都知道", "看不起", "嘲笑"],
            should_statements_keywords=["应该", "必须", "一定要", "不应该", "不可以", "理应"],
            base_risk_threshold=0.35, max_risk_per_category=0.15,
        ),
        hrv=HRVAgeParams(
            resting_hr_baseline=70.0, anxiety_hr_threshold=95.0,
            hrv_rmssd_baseline=40.0, hrv_low_risk_threshold=18.0,
            hr_variability_risk_weight=0.3,
        ),
        breathing=BreathingAgeParams(
            normal_rate_baseline=15.0, anxiety_rate_threshold=20.0,
            depression_rate_threshold=9.0, regularity_normal_cv=0.25,
            sigh_risk_threshold=3.0,
        ),
        behavioral_activation=BehavioralActivationAgeParams(
            min_daily_interactions=8, exploration_rate_baseline=0.4,
            social_withdrawal_threshold=0.15, activity_concentration_threshold=0.7,
            change_detection_window=7, withdrawal_risk_weight=0.3,
        ),
        eye=EyeMovementAgeParams(
            blink_rate_baseline=15.0, low_blink_threshold=8.0,
            high_blink_threshold=30.0, downward_gaze_risk_threshold=0.5,
            attention_scatter_threshold=0.6, eye_risk_weight=0.2,
        ),
        voice_semantics=VoiceSemanticsAgeParams(
            first_person_singular_threshold=0.12,
            first_person_plural_threshold=0.02,
            absolutist_expression_threshold=0.04,
            self_referential_density_threshold=0.20,
            negative_affect_ratio_threshold=0.18,
            semantic_risk_weight=0.3,
        ),
        fusion=FusionAgeParams(
            text_weight=0.35,
            audio_weight=0.12,
            face_weight=0.08,
            behavior_weight=0.10,
            circadian_weight=0.06,
            cognitive_weight=0.08,
            hrv_weight=0.05,
            breathing_weight=0.05,
            behavioral_act_weight=0.04,
            eye_weight=0.02,
            voice_semantics_weight=0.05,
            risk_threshold_offset=0.05,
        ),
    )


# ============================================================
# 工厂函数
# ============================================================

_PROFILES: dict[AgeGroup, AgeProfile] = {}


def _ensure_profiles() -> None:
    """延迟初始化所有画像（避免模块加载时的循环依赖）"""
    if not _PROFILES:
        _PROFILES[AgeGroup.EARLY_ADOLESCENT] = _build_early_adolescent_profile()
        _PROFILES[AgeGroup.MID_ADOLESCENT] = _build_mid_adolescent_profile()
        _PROFILES[AgeGroup.YOUNG_ADULT] = _build_young_adult_profile()


def get_age_profile(age_group: AgeGroup) -> AgeProfile:
    """获取指定年龄分组的完整画像

    Args:
        age_group: 年龄分组枚举

    Returns:
        AgeProfile 实例（包含所有模态的差异化参数）
    """
    _ensure_profiles()
    return _PROFILES[age_group]


def infer_age_group(age: Optional[int]) -> Optional[AgeGroup]:
    """从实际年龄推断年龄分组

    Args:
        age: 用户年龄（岁），None 时返回 None（使用默认配置）

    Returns:
        AgeGroup 枚举值，或 None（年龄不在范围内时）
    """
    if age is None:
        return None
    if 12 <= age <= 15:
        return AgeGroup.EARLY_ADOLESCENT
    elif 16 <= age <= 18:
        return AgeGroup.MID_ADOLESCENT
    elif 19 <= age <= 22:
        return AgeGroup.YOUNG_ADULT
    elif age < 12:
        # 低于范围 → 使用最敏感的分组
        return AgeGroup.EARLY_ADOLESCENT
    else:
        # 高于范围 → 使用最成熟的分组
        return AgeGroup.YOUNG_ADULT


def get_age_profile_by_age(age: Optional[int]) -> Optional[AgeProfile]:
    """从实际年龄获取完整画像（便捷函数）

    Args:
        age: 用户年龄（岁），None 时返回 None

    Returns:
        AgeProfile 实例，或 None
    """
    group = infer_age_group(age)
    if group is None:
        return None
    return get_age_profile(group)


# ============================================================
# 所有年龄组列表（供 API/测试使用）
# ============================================================

def get_all_age_groups() -> list[dict]:
    """获取所有年龄分组的摘要信息

    Returns:
        包含分组名称、年龄范围、描述的字典列表
    """
    _ensure_profiles()
    return [
        {
            "group": profile.age_group.value,
            "age_range": profile.age_range,
            "description": profile.description,
        }
        for profile in [
            _PROFILES[AgeGroup.EARLY_ADOLESCENT],
            _PROFILES[AgeGroup.MID_ADOLESCENT],
            _PROFILES[AgeGroup.YOUNG_ADULT],
        ]
    ]
