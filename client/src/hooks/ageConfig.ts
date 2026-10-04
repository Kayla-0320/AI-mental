/**
 * 年龄分层配置 —— 前端多模态感知的年龄差异化校准
 *
 * 与后端 algorithm/perception/age_config.py 对齐。
 * 三个年龄阶段：初中(12-15)、高中(15-18)、大学(18-22)
 *
 * 临床依据：
 * - Jiang et al. (2020) —— 青少年语音情感识别的年龄差异
 * - Ekman & Friesen (1978) —— FACS 跨年龄适用性
 * - Nolen-Hoeksema (2012) —— 青少年情绪调节的发展差异
 */

// ===== 年龄分组（与后端 AgeGroup 枚举对齐） =====
export type AgeGroup = 'early_adolescent' | 'mid_adolescent' | 'young_adult';

// ===== 性别分组（用于生理指标校准） =====
export type Gender = 'male' | 'female' | 'other';

/**
 * 性别修正因子 —— 基于生理学研究的小幅校准
 * 参考：
 * - Koenig et al. (2015) —— HRV 的性别差异 meta 分析
 * - Boissoneault et al. (2019) —— 眨眼频率的性别差异
 * - Randler & Scholz (2020) —— 昼夜节律的性别差异
 */
interface GenderModifiers {
  /** 心率基线修正 (bpm) —— 女性静息心率偏高 +3 */
  hrBaselineOffset: number;
  /** HRV RMSSD 修正 (%) —— 女性 HRV 略高 +10% */
  hrvModifier: number;
  /** 眨眼频率基线修正 (次/分) —— 女性眨眼频率偏高 +2 */
  blinkRateOffset: number;
  /** 深夜风险小时修正 —— 女性就寝时间略早 -0.5h */
  circadianOffset: number;
  /** 呼吸频率基线修正 (次/分) —— 女性呼吸频率偏高 +1 */
  breathingRateOffset: number;
}

const GENDER_MODIFIERS: Record<Gender, GenderModifiers> = {
  male:   { hrBaselineOffset: -2, hrvModifier: 0.95, blinkRateOffset: -1, circadianOffset: 0.5, breathingRateOffset: -1 },
  female: { hrBaselineOffset: 3, hrvModifier: 1.10, blinkRateOffset: 2, circadianOffset: -0.5, breathingRateOffset: 1 },
  other:  { hrBaselineOffset: 0, hrvModifier: 1.00, blinkRateOffset: 0, circadianOffset: 0, breathingRateOffset: 0 },
};

export function getGenderModifiers(gender: Gender | null | undefined): GenderModifiers {
  return gender ? GENDER_MODIFIERS[gender] : GENDER_MODIFIERS.other;
}

/** 从实际年龄推断年龄分组 */
export function inferAgeGroup(age: number | null | undefined): AgeGroup | null {
  if (age == null) return null;
  if (age <= 15) return 'early_adolescent';
  if (age <= 18) return 'mid_adolescent';
  if (age <= 22) return 'young_adult';
  if (age < 12) return 'early_adolescent';
  return 'young_adult';
}

/** 从出生日期计算年龄 */
export function ageFromBirthDate(birthDate: string | null | undefined): number | null {
  if (!birthDate) return null;
  const birth = new Date(birthDate);
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  const m = today.getMonth() - birth.getMonth();
  if (m < 0 || (m === 0 && today.getDate() < birth.getDate())) age--;
  return age > 0 && age < 100 ? age : null;
}

// ============================================================
// 文本模态 —— 年龄差异化词典与阈值
// ============================================================

interface TextAgeConfig {
  /** 网络俚语（情绪信号词） */
  slangKeywords: string[];
  /** 危机关键词 */
  crisisHigh: string[];
  crisisMedium: string[];
  crisisLow: string[];
  /** emoji 情感权重 */
  emojiSensitivity: number;
}

const TEXT_CONFIGS: Record<AgeGroup, TextAgeConfig> = {
  early_adolescent: {
    slangKeywords: [
      'yyds', '绝绝子', 'awsl', 'xswl', '破防了', 'emo了', '裂开',
      '蚌埠住了', '栓Q', '芭比Q', '无语子', '好家伙', '真的会谢',
      'DNA动了', '6到飞起', '上头', '下头', '塌房', '摆烂',
      '尊嘟假嘟', 'i人', 'e人', 'city不city',
    ],
    crisisHigh: [
      '不想活', '去死', '跳楼', '割腕', '自杀', '自残', '自伤',
      '活不下去', '死了算了', '想死', '不想醒来',
    ],
    crisisMedium: [
      '消失', '不想上学', '没人要我', '我是废物', '活着没意思',
      '消失好了', '累了', '不想存在',
    ],
    crisisLow: [
      '好无聊', '没意思', '不想动', '好累', '不开心', '好烦',
      '好孤单', '心情差',
    ],
    emojiSensitivity: 0.15,
  },
  mid_adolescent: {
    slangKeywords: [
      'emo', '内卷', '躺平', '破防', '精神状态', '发疯文学', '摆烂',
      '社恐', 'i人e人', '电子榨菜', '发疯中', '已黑化', '纯爱战士',
      '大冤种', '显眼包', '嘴替', 'CPU', '抽象', '上价值',
      '格局打开', '人生建议',
    ],
    crisisHigh: [
      '自杀', '自残', '割腕', '不想活', '想死', '去死',
      '撑不住了', '自伤',
    ],
    crisisMedium: [
      '消失好了', '不想存在', '没有意义', '活着好累', '如果我不在了',
      '解脱', '看不到希望', '一切都没意义', '无尽的黑暗', '深渊',
    ],
    crisisLow: [
      '累了', '好想逃', 'emo', '丧', '没动力', '没意思',
      '迷茫', '孤独', '不想说话',
    ],
    emojiSensitivity: 0.10,
  },
  young_adult: {
    slangKeywords: [
      '精神内耗', 'PUA', '情绪价值', '边界感', '松弛感', '润', '开摆',
      '已读乱回', '发疯', '班味', '牛马', '打工人', '工具人',
      '社会性死亡', '情绪稳定', '内核强大', '课题分离', '自我疗愈',
      '确诊为', '人生是旷野',
    ],
    crisisHigh: [
      '自杀', '自残', '不想活', '想死', '了断', '了结',
      '结束这一切', '撑不下去',
    ],
    crisisMedium: [
      '存在没有意义', '虚无', '解脱', '活着是一种负担',
      '如果世界没有我', '精神内耗到极限', '看不到出路', '深渊',
      '无尽的疲惫', '存在的荒谬',
    ],
    crisisLow: [
      '倦怠', '空虚', '无力感', '意义感缺失', '麻木',
      '提不起劲', '疲惫', '孤独', 'burnout',
    ],
    emojiSensitivity: 0.05,
  },
};

// ============================================================
// 语音模态 —— 年龄差异化阈值
// ============================================================

interface VoiceAgeConfig {
  /** 基频基线 (Hz) */
  f0Baseline: number;
  /** 焦虑判定高基频阈值 */
  f0AnxietyThreshold: number;
  /** 抑郁判定低基频阈值 */
  f0DepressionThreshold: number;
  /** 语速焦虑阈值 */
  speechRateAnxietyThreshold: number;
  /** 正常语速下限 */
  speechRateNormalLow: number;
  /** 停顿抑郁阈值 */
  pauseDepressionThreshold: number;
}

const VOICE_CONFIGS: Record<AgeGroup, VoiceAgeConfig> = {
  early_adolescent: {
    f0Baseline: 220, f0AnxietyThreshold: 300, f0DepressionThreshold: 140,
    speechRateAnxietyThreshold: 5.5, speechRateNormalLow: 3.0, pauseDepressionThreshold: 0.35,
  },
  mid_adolescent: {
    f0Baseline: 190, f0AnxietyThreshold: 270, f0DepressionThreshold: 120,
    speechRateAnxietyThreshold: 5.0, speechRateNormalLow: 3.0, pauseDepressionThreshold: 0.30,
  },
  young_adult: {
    f0Baseline: 170, f0AnxietyThreshold: 250, f0DepressionThreshold: 110,
    speechRateAnxietyThreshold: 4.8, speechRateNormalLow: 2.8, pauseDepressionThreshold: 0.28,
  },
};

// ============================================================
// 面部模态 —— 年龄差异化阈值
// ============================================================

interface FaceAgeConfig {
  /** AU 强度基线 (掩饰能力递增 → 基线递减) */
  auIntensityBaseline: number;
  /** 微表情检测灵敏度阈值 */
  microExpressionThreshold: number;
  /** 掩饰检测权重 */
  maskingDetectionWeight: number;
}

const FACE_CONFIGS: Record<AgeGroup, FaceAgeConfig> = {
  early_adolescent: { auIntensityBaseline: 0.6, microExpressionThreshold: 0.30, maskingDetectionWeight: 0.1 },
  mid_adolescent:   { auIntensityBaseline: 0.5, microExpressionThreshold: 0.25, maskingDetectionWeight: 0.3 },
  young_adult:      { auIntensityBaseline: 0.4, microExpressionThreshold: 0.20, maskingDetectionWeight: 0.5 },
};

// ============================================================
// 行为/键盘模态 —— 年龄差异化阈值
// ============================================================

interface KeyboardAgeConfig {
  /** 打字速度基线 (字/分钟) */
  typingSpeedBaseline: number;
  /** 深夜风险阈值 (小时) */
  lateNightHour: number;
}

const KEYBOARD_CONFIGS: Record<AgeGroup, KeyboardAgeConfig> = {
  early_adolescent: { typingSpeedBaseline: 45, lateNightHour: 23 },
  mid_adolescent:   { typingSpeedBaseline: 40, lateNightHour: 0 },
  young_adult:      { typingSpeedBaseline: 35, lateNightHour: 1 },
};

// ============================================================
// 融合权重 —— 年龄差异化 (11 模态)
// ============================================================

interface FusionAgeConfig {
  text: number; voice: number; facial: number; keyboard: number;
  circadian: number; cognitive: number; hrv: number; breathing: number;
  behavioralAct: number; eye: number; voiceSemantics: number;
  /** 风险阈值偏移 (初中=-0.05更敏感, 大学=+0.05减少误报) */
  riskThresholdOffset: number;
}

const FUSION_CONFIGS: Record<AgeGroup, FusionAgeConfig> = {
  early_adolescent: {
    text: 0.25, voice: 0.12, facial: 0.20, keyboard: 0.10,
    circadian: 0.06, cognitive: 0.08, hrv: 0.05, breathing: 0.04,
    behavioralAct: 0.03, eye: 0.02, voiceSemantics: 0.05,
    riskThresholdOffset: -0.05,
  },
  mid_adolescent: {
    text: 0.30, voice: 0.12, facial: 0.12, keyboard: 0.10,
    circadian: 0.06, cognitive: 0.08, hrv: 0.05, breathing: 0.05,
    behavioralAct: 0.04, eye: 0.03, voiceSemantics: 0.05,
    riskThresholdOffset: 0.0,
  },
  young_adult: {
    text: 0.35, voice: 0.12, facial: 0.08, keyboard: 0.10,
    circadian: 0.06, cognitive: 0.08, hrv: 0.05, breathing: 0.05,
    behavioralAct: 0.04, eye: 0.02, voiceSemantics: 0.05,
    riskThresholdOffset: 0.05,
  },
};

// ============================================================
// 昼夜节律 —— 年龄差异化
// ============================================================

interface CircadianAgeConfig {
  /** 理想就寝时间 */
  idealBedtime: number;
  /** 深夜风险起始小时 */
  lateActivityRiskHour: number;
  /** 最小睡眠需求 (小时) */
  minSleepHours: number;
}

const CIRCADIAN_CONFIGS: Record<AgeGroup, CircadianAgeConfig> = {
  early_adolescent: { idealBedtime: 21.5, lateActivityRiskHour: 22, minSleepHours: 9.0 },
  mid_adolescent:   { idealBedtime: 22.5, lateActivityRiskHour: 23, minSleepHours: 8.5 },
  young_adult:      { idealBedtime: 23.5, lateActivityRiskHour: 0, minSleepHours: 8.0 },
};

// ============================================================
// 认知扭曲 —— 年龄差异化关键词
// ============================================================

interface CognitiveDistortionAgeConfig {
  catastrophizing: string[];
  blackAndWhite: string[];
  selfBlame: string[];
  hopelessness: string[];
  mindReading: string[];
  shouldStatements: string[];
}

const COGNITIVE_CONFIGS: Record<AgeGroup, CognitiveDistortionAgeConfig> = {
  early_adolescent: {
    catastrophizing: ['完了', '完蛋', '全毁了', '天塌了', '太可怕了'],
    blackAndWhite: ['总是', '从不', '永远', '绝对', '一定', '完全'],
    selfBlame: ['都是我的错', '怪我', '我不行', '我太差', '废物', '垃圾'],
    hopelessness: ['没有希望', '不会好的', '没救了', '不可能', '没办法'],
    mindReading: ['他们肯定觉得', '别人都笑我', '大家都知道', '看不起我'],
    shouldStatements: ['应该', '必须', '一定要', '不应该', '不可以'],
  },
  mid_adolescent: {
    catastrophizing: ['完了', '完蛋', '全毁了', '一切都完了', '太可怕了'],
    blackAndWhite: ['总是', '从不', '永远不', '绝对', '完全', '根本'],
    selfBlame: ['都是我的错', '怪我', '我不配', '我不行', '废物', '没用'],
    hopelessness: ['没有希望', '不会好的', '没救了', '不可能', '做不到'],
    mindReading: ['他们肯定觉得', '别人都', '大家都知道', '看不起'],
    shouldStatements: ['应该', '必须', '一定要', '不应该'],
  },
  young_adult: {
    catastrophizing: ['全完了', '一切都毁了', '无法挽回', '灾难'],
    blackAndWhite: ['总是', '从不', '绝对', '完全', '根本', '所有'],
    selfBlame: ['都是我的错', '我不配', '我不行', '太差', '废物', '没用'],
    hopelessness: ['没有希望', '不会好', '没救', '不可能', '做不到', '无望'],
    mindReading: ['他们觉得', '别人都', '大家都知道', '看不起', '嘲笑'],
    shouldStatements: ['应该', '必须', '一定要', '不应该', '不可以', '理应'],
  },
};

// ============================================================
// HRV/呼吸/眼动/行为激活/语音语义 —— 年龄差异化阈值
// ============================================================

interface HRVAgeConfig {
  restingHRBaseline: number;
  anxietyHRThreshold: number;
  hrvLowRiskThreshold: number;
}

const HRV_CONFIGS: Record<AgeGroup, HRVAgeConfig> = {
  early_adolescent: { restingHRBaseline: 85, anxietyHRThreshold: 110, hrvLowRiskThreshold: 25 },
  mid_adolescent:   { restingHRBaseline: 75, anxietyHRThreshold: 100, hrvLowRiskThreshold: 20 },
  young_adult:      { restingHRBaseline: 70, anxietyHRThreshold: 95, hrvLowRiskThreshold: 18 },
};

interface BreathingAgeConfig {
  normalRateBaseline: number;
  anxietyRateThreshold: number;
  depressionRateThreshold: number;
}

const BREATHING_CONFIGS: Record<AgeGroup, BreathingAgeConfig> = {
  early_adolescent: { normalRateBaseline: 18, anxietyRateThreshold: 24, depressionRateThreshold: 11 },
  mid_adolescent:   { normalRateBaseline: 16, anxietyRateThreshold: 22, depressionRateThreshold: 10 },
  young_adult:      { normalRateBaseline: 15, anxietyRateThreshold: 20, depressionRateThreshold: 9 },
};

interface BehavioralActivationAgeConfig {
  minDailyInteractions: number;
  explorationRateBaseline: number;
  socialWithdrawalThreshold: number;
}

const BEHAVIORAL_ACT_CONFIGS: Record<AgeGroup, BehavioralActivationAgeConfig> = {
  early_adolescent: { minDailyInteractions: 15, explorationRateBaseline: 0.6, socialWithdrawalThreshold: 0.25 },
  mid_adolescent:   { minDailyInteractions: 10, explorationRateBaseline: 0.5, socialWithdrawalThreshold: 0.2 },
  young_adult:      { minDailyInteractions: 8, explorationRateBaseline: 0.4, socialWithdrawalThreshold: 0.15 },
};

interface EyeMovementAgeConfig {
  blinkRateBaseline: number;
  lowBlinkThreshold: number;
  highBlinkThreshold: number;
}

const EYE_CONFIGS: Record<AgeGroup, EyeMovementAgeConfig> = {
  early_adolescent: { blinkRateBaseline: 18, lowBlinkThreshold: 8, highBlinkThreshold: 30 },
  mid_adolescent:   { blinkRateBaseline: 16, lowBlinkThreshold: 8, highBlinkThreshold: 30 },
  young_adult:      { blinkRateBaseline: 15, lowBlinkThreshold: 8, highBlinkThreshold: 30 },
};

// ============================================================
// 统一获取接口
// ============================================================

export function getTextConfig(ageGroup: AgeGroup | null): TextAgeConfig {
  return ageGroup ? TEXT_CONFIGS[ageGroup] : TEXT_CONFIGS.mid_adolescent;
}
export function getVoiceConfig(ageGroup: AgeGroup | null): VoiceAgeConfig {
  return ageGroup ? VOICE_CONFIGS[ageGroup] : VOICE_CONFIGS.mid_adolescent;
}
export function getFaceConfig(ageGroup: AgeGroup | null): FaceAgeConfig {
  return ageGroup ? FACE_CONFIGS[ageGroup] : FACE_CONFIGS.mid_adolescent;
}
export function getKeyboardConfig(ageGroup: AgeGroup | null): KeyboardAgeConfig {
  return ageGroup ? KEYBOARD_CONFIGS[ageGroup] : KEYBOARD_CONFIGS.mid_adolescent;
}
export function getFusionConfig(ageGroup: AgeGroup | null): FusionAgeConfig {
  return ageGroup ? FUSION_CONFIGS[ageGroup] : FUSION_CONFIGS.mid_adolescent;
}
export function getCircadianConfig(ageGroup: AgeGroup | null): CircadianAgeConfig {
  return ageGroup ? CIRCADIAN_CONFIGS[ageGroup] : CIRCADIAN_CONFIGS.mid_adolescent;
}
export function getCognitiveConfig(ageGroup: AgeGroup | null): CognitiveDistortionAgeConfig {
  return ageGroup ? COGNITIVE_CONFIGS[ageGroup] : COGNITIVE_CONFIGS.mid_adolescent;
}
export function getHRVConfig(ageGroup: AgeGroup | null): HRVAgeConfig {
  return ageGroup ? HRV_CONFIGS[ageGroup] : HRV_CONFIGS.mid_adolescent;
}
export function getBreathingConfig(ageGroup: AgeGroup | null): BreathingAgeConfig {
  return ageGroup ? BREATHING_CONFIGS[ageGroup] : BREATHING_CONFIGS.mid_adolescent;
}
export function getBehavioralActConfig(ageGroup: AgeGroup | null): BehavioralActivationAgeConfig {
  return ageGroup ? BEHAVIORAL_ACT_CONFIGS[ageGroup] : BEHAVIORAL_ACT_CONFIGS.mid_adolescent;
}
export function getEyeConfig(ageGroup: AgeGroup | null): EyeMovementAgeConfig {
  return ageGroup ? EYE_CONFIGS[ageGroup] : EYE_CONFIGS.mid_adolescent;
}

/** 年龄分组标签（中文） */
export const AGE_GROUP_LABELS: Record<AgeGroup, string> = {
  early_adolescent: '初中 (12-15岁)',
  mid_adolescent: '高中 (15-18岁)',
  young_adult: '大学 (18-22岁)',
};

// ============================================================
// 个人基线偏移 → 融合权重自适应调整
// ============================================================

/**
 * 根据各模态的个人基线 z-score 调整融合权重。
 *
 * 原理：偏离个人基线越大的模态，其信号越有诊断价值，应提升权重；
 *       接近基线的模态信息量低，可降低权重以减少噪声。
 *
 * @param baseWeights  年龄基准融合权重（归一化前）
 * @param zScores      各模态当前值的 z-score（key 与权重 key 对齐）
 * @returns 调整后的权重（未归一化）
 */
export function adjustWeightsByBaseline(
  baseWeights: Record<string, number>,
  zScores: Record<string, number>,
): Record<string, number> {
  const adjusted = { ...baseWeights };
  for (const key of Object.keys(adjusted)) {
    const z = Math.abs(zScores[key] ?? 0);
    if (z > 2) {
      // 偏离 > 2σ → 该模态信号显著，提升权重 30%
      adjusted[key] *= 1.3;
    } else if (z < 0.5) {
      // 偏离 < 0.5σ → 该模态接近常态，降低权重 30%
      adjusted[key] *= 0.7;
    }
    // 0.5 ≤ z ≤ 2 → 不调整
  }
  return adjusted;
}
