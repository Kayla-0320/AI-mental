/**
 * 多模态情感感知类型定义
 *
 * 理论基础：
 * - 键盘动力学： keystroke dynamics (Biometric identification, 2011)
 * - 文本情感：   NRC Emotion Lexicon (Mohammad & Turney, 2013)
 * - 声学特征：   Scherer (2003), Cummins et al. (2015)
 * - 面部表情：   FACS (Ekman & Friesen, 1978), AU 动作单元系统
 * - 微表情：     Ekman (2001), 持续时间 < 500ms 的无意识表情
 */

// ===== 键盘动力学 (Keystroke Dynamics) =====
export interface KeyboardDynamics {
  // 打字节奏
  typingSpeed: number;            // 字/分钟
  avgInterval: number;            // 平均键间间隔 ms
  intervalVariance: number;       // 键间间隔变异系数 (CV)
  rhythmScore: number;            // 节奏规律性 0-1

  // 停顿分析
  pauseCount: number;             // 长停顿次数 (>1.5s, 认知负荷指标)
  pauseRatio: number;             // 停顿占比 0-1
  avgPauseDuration: number;       // 平均停顿时长 ms

  // 修正行为
  deletionRate: number;           // 删除率 0-1
  correctionRate: number;         // 修正率 (删除/总按键) 0-1
  backspaceBurst: number;         // 连续删除爆发次数 (焦虑指标)

  // 错误与压力
  errorRate: number;              // 估计错误率 0-1
  pressureIndex: number;          // 按键压力指数 0-1 (基于速度变化)

  // 综合
  anxietyIndex: number;           // 综合焦虑指数 0-100
  sampleSize: number;             // 样本按键数
  isCollecting: boolean;
}

// ===== 文本情感分析 (Text Emotion Analysis) =====
export interface TextAnalysis {
  // 八维情绪分布：[快乐, 悲伤, 焦虑, 愤怒, 恐惧, 厌恶, 惊讶, 中性]
  emotionDistribution: number[];
  dominantEmotion: string;
  dominantConfidence: number;

  // 危机信号检测
  crisisMarkers: string[];        // 检测到的危机关键词
  crisisLevel: 'none' | 'low' | 'medium' | 'high';

  // 认知标记 (CBT 认知扭曲)
  cognitiveMarkers: {
    catastrophizing: number;      // 灾难化思维 0-1
    blackAndWhite: number;        // 非黑即白 0-1
    overgeneralization: number;   // 过度概括 0-1
    selfBlame: number;            // 自我归咎 0-1
    hopelessness: number;         // 无望感 0-1
  };

  // 语言特征
  sentiment: number;              // 情感效价 -1(消极) ~ +1(积极)
  emotionalIntensity: number;     // 情绪强度 0-1
  firstPersonRate: number;        // 第一人称密度 (自我关注指标)
  negationRate: number;           // 否定词密度 0-1
  absoluteWords: number;          // 绝对化用词次数 (总是/从不/一定)
  questionRate: number;           // 疑问句比例 (不确定指标)

  // 书写行为
  avgMessageLength: number;       // 平均消息长度
  writingPressure: number;        // 书写压力 0-1 (消息长度×速度)
  topicCoherence: number;         // 话题连贯性 0-1

  // 元数据
  totalCharsAnalyzed: number;
  lastAnalyzedText: string;
  timestamp: number;
}

// ===== 声音声学分析 (Voice Acoustic Analysis) =====
export interface VoiceAnalysis {
  // 基频特征 (F0 / Pitch) — 情绪唤醒度指标
  pitch: {
    mean: number;                 // 基频均值 Hz
    std: number;                  // 基频标准差 (情绪波动性)
    range: number;                // 基频范围 Hz (情绪表达力)
    min: number;
    max: number;
  };

  // 能量特征 (Energy / Intensity) — 情绪强度
  energy: {
    mean: number;                 // 能量均值 dB
    std: number;                  // 能量波动 (情绪稳定性)
    peakRatio: number;            // 峰值占比 0-1
  };

  // 语速特征 (Speech Rate) — 焦虑/抑郁指标
  rate: {
    speechRate: number;           // 语速 音节/秒
    tempo: 'slow' | 'normal' | 'fast';
    regularity: number;           // 语速规律性 0-1
  };

  // 停顿特征 (Pause) — 犹豫/思维迟缓指标
  pauses: {
    ratio: number;                // 停顿比 0-1
    count: number;                // 停顿次数
    avgDuration: number;          // 平均停顿 ms
    longPauseCount: number;       // 长停顿 (>2s) 次数
  };

  // 频谱特征 (Spectral) — 情绪色彩
  spectral: {
    centroid: number;             // 频谱质心 Hz (明亮度)
    rolloff: number;              // 频谱滚降点 Hz
    flatness: number;             // 频谱平坦度 0-1 (噪声vs谐波)
  };

  // 情绪推断
  emotionProbs: number[];         // [快乐, 悲伤, 焦虑, 愤怒, 中性]
  riskScore: number;              // 风险评分 0-1
  detectedEmotion: string;

  // 语音识别 (Speech Recognition)
  speechText: string;             // 最近识别的语音文本
  speechTextHistory: string[];    // 识别历史（最近 5 条）
  isSpeechRecognizing: boolean;   // 是否正在识别语音

  // 本地语音识别状态（sherpa-onnx，音频不出本机）
  // 刻意如实上报：引擎不可用时必须能看出来，不允许静默降级成空文本
  asrEngine: string;              // 'sherpa-onnx-streaming-zipformer' | 'unavailable' | ''
  asrError: string;               // 非空表示识别不可用及具体原因
  asrLatencyMs: number;           // 最近一次解码耗时（毫秒）
  // 流式识别的中间结果：会被后续结果覆盖，只用于"边说边出字"上屏。
  // 下游的文本情感/认知扭曲/语音语义分析只吃 speechText（定稿），不吃这个字段。
  asrPartialText: string;
  // 最近一次定稿的文本 + 递增序号。
  // 序号是必需的：连续两段说出完全相同的话时 speechText 不变，
  // 只靠文本变化判断会漏掉第二段。
  asrFinalText: string;
  asrFinalSeq: number;
  // 最近一次定稿是否经过离线大模型复识别纠错
  asrRefined: boolean;

  // 状态
  isRecording: boolean;
  duration: number;               // 已录制时长 秒
  timestamp: number;
}

// ===== 面部微表情分析 (Facial Micro-expression Analysis) =====
export interface FacialAnalysis {
  // 面部动作单元 (Action Units, FACS)
  // 基于 Ekman & Friesen (1978) 面部动作编码系统
  actionUnits: {
    // --- 眉毛区域 ---
    au1_browRaise: number;        // 内眉上扬 (悲伤指标)
    au2_browOuterRaise: number;   // 外眉上扬 (惊讶指标)
    au4_browLower: number;        // 眉毛下压 (愤怒/专注指标)

    // --- 眼睛区域 ---
    au5_eyeOpen: number;          // 上眼睑抬起 (惊讶/恐惧)
    au7_eyeTighten: number;       // 眼睑收紧 (愤怒/痛苦)
    eyeAspect: number;            // 眼睛纵横比 EAR (疲劳/注意力)

    // --- 鼻子/脸颊区域 ---
    au9_noseWrinkle: number;      // 鼻子皱起 (厌恶指标)
    au6_cheekRaise: number;       // 脸颊上扬 (真笑指标, Duchenne marker)

    // --- 嘴巴区域 ---
    au10_lipRaise: number;        // 上唇上扬 (厌恶/轻蔑)
    au12_lipCorner: number;       // 嘴角上提 (快乐指标)
    au15_lipCornerDrop: number;   // 嘴角下拉 (悲伤指标)
    au17_chinRaise: number;       // 下巴抬起 (痛苦指标)
    au20_lipStretch: number;      // 嘴唇拉伸 (恐惧指标)
    au26_jawDrop: number;         // 下巴下垂 (惊讶/恐惧)
    mouthAspect: number;          // 张嘴程度

    // --- 头部姿态 ---
    headPitch: number;            // 俯仰角 (低头=回避)
    headYaw: number;              // 偏航角 (转头=回避)
    headRoll: number;             // 翻滚角 (歪头=好奇/困惑)
  };

  // 几何特征
  geometry: {
    browDistance: number;          // 眉毛-眼睛距离比
    mouthWidth: number;            // 嘴宽/脸宽比
    eyeAspect: number;             // 眼睛纵横比 (EAR)
    mouthOpen: number;             // 张嘴程度
  };

  // 微表情事件
  microExpressions: {
    count: number;                 // 微表情次数 (< 500ms)
    recentEmotions: string[];      // 最近检测到的微表情情绪
  };

  // 综合表情
  dominantExpression: string;      // 主导表情
  expressionIntensity: number;     // 表情强度 0-1
  emotionMapping: {               // AU → 情绪映射
    happiness: number;
    sadness: number;
    anger: number;
    fear: number;
    surprise: number;
    disgust: number;
    distress: number;
  };

  // 状态
  /** 采集循环是否在跑（**会话级**状态；不等于"画面里有人"）。 */
  isDetecting: boolean;
  /**
   * 本帧是否检测到人脸（**帧级**状态）。
   *
   * 这是整个视觉链路里最诚实、最抗噪的一个信号：
   * 低头看手机、侧脸、走动、说话带动下半脸都不影响它，
   * 而它的产品含义是真实的 —— "对方还在听吗"。
   * 通话场景里如果需要视觉反馈，优先用这个，而不是微表情。
   */
  facePresent: boolean;
  /**
   * 检测置信度 0-1，**必须来自本帧真实测得的信号质量**；拿不到时为 `null`。
   *
   * 改造前这里是 `min(0.95, 0.7 + frameCount * 0.005)` —— 一个**帧计数常量**：
   * 只要摄像头开着满 50 帧（5 秒）就恒为 0.95，画面全黑、没有人在镜头里也一样。
   * 它会一路传进融合权重并改写落库的 `patientProfile.riskLevel`。
   *
   * 现在：由「AU 是否来自模型输出 × 人脸尺度 × 头部姿态」三者相乘得到；
   * `null` 表示本次没有可信的置信度，**下游必须把该模态剔除**，而不是拿常数顶上。
   */
  confidence: number | null;
  frameCount: number;
  timestamp: number;
  /** 设备不可用 / 流中断的原因（中文，可直接上屏）；空串表示正常。 */
  unavailableReason: string;
}

// ===== 融合后的综合情绪状态 =====
export interface ComprehensiveEmotionState {
  // 融合情绪结果
  emotionProbs: number[];          // [快乐, 悲伤, 焦虑, 愤怒, 中性]
  confidence: number;
  dominantEmotion: string;
  riskLevel: 'low' | 'medium' | 'high' | 'crisis';

  // 各模态详细数据
  keyboard: KeyboardDynamics;
  text: TextAnalysis;
  voice: VoiceAnalysis;
  facial: FacialAnalysis;
  // 新增 7 模态
  circadian: CircadianAnalysis;
  cognitiveDistortion: CognitiveDistortionAnalysis;
  hrv: HRVAnalysis;
  breathing: BreathingAnalysis;
  behavioralActivation: BehavioralActivationAnalysis;
  eyeMovement: EyeMovementAnalysis;
  voiceSemantics: VoiceSemanticsAnalysis;

  // 融合元数据
  fusionWeights: {
    text: number;
    voice: number;
    facial: number;
    keyboard: number;
    circadian: number;
    cognitive: number;
    hrv: number;
    breathing: number;
    behavioralAct: number;
    eye: number;
    voiceSemantics: number;
  };
  activeModalities: string[];      // 当前活跃的模态列表
  lastUpdated: number;

  // 数据可信度说明：哪些模态本设备根本不具备采集能力
  // 用于界面与报告如实告知，禁止对不可用模态编造数值
  unavailableModalities: string[];
  availabilityNote: string;

  // 温暖治愈信息
  caringMessage: string;
  evidence: string[];
  narrativeAnalysis: string;       // 叙事性多模态分析长文本
}

// ===== 新增 7 模态类型定义 =====

// ===== 昼夜节律分析 =====
// 注意：本模块只记录「App 内活动时间」，**不检测睡眠**。
// 睡眠时长/质量必须由用户手动记录，禁止由活动时间或情绪分推算。
export interface CircadianAnalysis {
  riskScore: number;               // 深夜活动风险 0-1（活动时间推导，非睡眠检测）
  lateNightRisk: number;           // 深夜活动风险 0-1
  regularityScore: number;         // 活动时间规律性 0-1
  lastActivityHour: number;        // 最近活动小时 (0-23)
  activitySampleCount: number;     // 已采集的活动样本数（样本不足时规律性不可信）
  isActive: boolean;
  timestamp: number;
}

// ===== 认知扭曲深度分析 =====
export interface CognitiveDistortionAnalysis {
  riskScore: number;               // 认知扭曲风险 0-1
  categories: {
    catastrophizing: number;       // 灾难化 0-1
    blackAndWhite: number;         // 非黑即白 0-1
    overgeneralization: number;    // 过度概括 0-1
    selfBlame: number;             // 自我归咎 0-1
    hopelessness: number;          // 无望感 0-1
    mindReading: number;           // 读心术 0-1
    shouldStatements: number;      // 应该陈述 0-1
  };
  totalDistortionCount: number;
  timestamp: number;
}

// ===== rPPG 心率变异性分析 =====
export interface HRVAnalysis {
  riskScore: number;               // HRV 风险 0-1
  heartRate: number;               // 心率 bpm
  hrvRmssd: number;                // HRV RMSSD ms
  signalQuality: number;           // 信号质量 0-1
  isMeasuring: boolean;
  timestamp: number;
}

// ===== 呼吸模式分析 =====
export interface BreathingAnalysis {
  riskScore: number;               // 呼吸风险 0-1
  breathingRate: number;           // 呼吸频率 次/分
  regularityCV: number;            // 规律性变异系数
  sighCount: number;               // 叹气次数
  signalQuality: number;           // 信号质量 0-1
  isMeasuring: boolean;
  timestamp: number;
}

// ===== 行为激活水平分析 =====
export interface BehavioralActivationAnalysis {
  riskScore: number;               // 行为退缩风险 0-1
  dailyInteractionCount: number;   // 今日互动次数
  explorationRate: number;         // 功能探索率 0-1
  socialWithdrawalScore: number;   // 社交退缩分 0-1
  trendDirection: 'improving' | 'stable' | 'declining';
  isActive: boolean;
  timestamp: number;
}

// ===== 眼动模式分析 =====
// ⚠️ 本平台未接入虹膜注视跟踪：downwardGazeRatio / attentionScatter 均由头部姿态推算，
//    分别是「低头帧占比」和「头部偏航波动」，属注视方向的代理指标。字段名沿用历史接口。
export interface EyeMovementAnalysis {
  riskScore: number;               // 眼动风险 0-1
  blinkRate: number;               // 眨眼频率 次/分（60 秒滚动窗口）
  downwardGazeRatio: number;       // 低头帧占比 0-1（头部俯仰代理）
  attentionScatter: number;        // 头部偏航波动 0-1（头部姿态代理）
  signalQuality: number;           // 信号质量 0-1（来源基准分 × 面部帧覆盖率）
  isMeasuring: boolean;
  timestamp: number;
  /** 眨眼来源：blendshape=MediaPipe 眼睑闭合度，ear=关键点纵横比降级 */
  blinkMethod?: 'blendshape' | 'ear' | 'none';
  /** 平均眨眼时长 ms */
  avgBlinkDuration?: number;
  /** 长眨眼次数（>400ms） */
  longBlinkCount?: number;
  /** 60 秒窗口内计入的眨眼次数 */
  blinkSampleCount?: number;
  /** 标记：注视相关字段来自头部姿态代理，非眼球注视 */
  poseProxy?: boolean;
  /**
   * 眨眼通道的实测帧率（自适应档位 30/15/10，跑不动会自动降档）。
   * 它是「闭眼时长精度」的唯一决定因素，所以必须可见。
   */
  blinkChannelFps?: number;
  /**
   * 本次测量的闭眼时长量化粒度（ms）= 1000 / blinkChannelFps。
   * 30fps → 33ms；10fps → 100ms。界面/证据里应据此说明时长精度，
   * 而不是让读者以为时长是毫秒级精确的。
   */
  blinkDurationQuantumMs?: number;
  /** 状态机实测的采样间隔中位数（ms） */
  blinkSampleIntervalMs?: number;
  /** 时长由亚采样插值得到的比例 0–1（插值默认关闭，故通常为 0） */
  blinkInterpolatedShare?: number;
  /**
   * 「每眼睁闭眼」第二判定通道（浏览器内 ONNX，Apache-2.0，46KB）的状态。
   * ⚠️ 这些字段**只上报、不参与 riskScore**：模型训练域是近红外眼图，
   * 在本平台 RGB 画面上存在域偏移，必须先标定阈值。
   */
  eyeStateModelStatus?: string;
  /** 两眼中较小的闭眼概率（与 min(eyeBlinkLeft, eyeBlinkRight) 口径对齐） */
  eyeStateModelClosedBoth?: number;
  eyeStateModelClosedLeft?: number;
  eyeStateModelClosedRight?: number;
  /** 单次推理耗时 ms */
  eyeStateModelLatencyMs?: number;
  /** 累计参与比较的帧数 */
  eyeStateModelSamples?: number;
  /** 与 blendshape 判定的一致率 0–1（评估第二通道是否可信的入口指标） */
  eyeStateModelAgreement?: number;
}

// ===== 语音深层语义分析 =====
export interface VoiceSemanticsAnalysis {
  riskScore: number;               // 语义风险 0-1
  firstPersonSingularRatio: number; // 第一人称单数密度
  firstPersonPluralRatio: number;   // 第一人称复数密度
  absolutistRatio: number;         // 绝对化表达密度
  selfReferentialDensity: number;  // 自我指涉密度
  negativeAffectRatio: number;     // 消极情感词比例
  timestamp: number;
}

// ===== 默认值 =====
export const defaultKeyboard: KeyboardDynamics = {
  typingSpeed: 0, avgInterval: 0, intervalVariance: 0, rhythmScore: 0,
  pauseCount: 0, pauseRatio: 0, avgPauseDuration: 0,
  deletionRate: 0, correctionRate: 0, backspaceBurst: 0,
  errorRate: 0, pressureIndex: 0, anxietyIndex: 0,
  sampleSize: 0, isCollecting: false,
};

export const defaultText: TextAnalysis = {
  emotionDistribution: [0, 0, 0, 0, 0, 0, 0, 1],
  dominantEmotion: '中性', dominantConfidence: 0,
  crisisMarkers: [], crisisLevel: 'none' as const,
  cognitiveMarkers: { catastrophizing: 0, blackAndWhite: 0, overgeneralization: 0, selfBlame: 0, hopelessness: 0 },
  sentiment: 0, emotionalIntensity: 0, firstPersonRate: 0,
  negationRate: 0, absoluteWords: 0, questionRate: 0,
  avgMessageLength: 0, writingPressure: 0, topicCoherence: 0,
  totalCharsAnalyzed: 0, lastAnalyzedText: '', timestamp: 0,
};

export const defaultVoice: VoiceAnalysis = {
  pitch: { mean: 0, std: 0, range: 0, min: 0, max: 0 },
  energy: { mean: 0, std: 0, peakRatio: 0 },
  rate: { speechRate: 0, tempo: 'normal' as const, regularity: 0 },
  pauses: { ratio: 0, count: 0, avgDuration: 0, longPauseCount: 0 },
  spectral: { centroid: 0, rolloff: 0, flatness: 0 },
  emotionProbs: [0.2, 0.2, 0.2, 0.2, 0.2],
  riskScore: 0, detectedEmotion: '中性',
  speechText: '', speechTextHistory: [], isSpeechRecognizing: false,
  asrEngine: '', asrError: '', asrLatencyMs: 0,
  asrPartialText: '', asrFinalText: '', asrFinalSeq: 0, asrRefined: false,
  isRecording: false, duration: 0, timestamp: 0,
};

export const defaultFacial: FacialAnalysis = {
  actionUnits: {
    au1_browRaise: 0, au2_browOuterRaise: 0, au4_browLower: 0,
    au5_eyeOpen: 0, au7_eyeTighten: 0, eyeAspect: 0.3,
    au9_noseWrinkle: 0, au6_cheekRaise: 0,
    au10_lipRaise: 0, au12_lipCorner: 0, au15_lipCornerDrop: 0,
    au17_chinRaise: 0, au20_lipStretch: 0, au26_jawDrop: 0,
    mouthAspect: 0, headPitch: 0, headYaw: 0, headRoll: 0,
  },
  geometry: { browDistance: 0, mouthWidth: 0, eyeAspect: 0, mouthOpen: 0 },
  microExpressions: { count: 0, recentEmotions: [] },
  dominantExpression: '平静', expressionIntensity: 0,
  emotionMapping: { happiness: 0, sadness: 0, anger: 0, fear: 0, surprise: 0, disgust: 0, distress: 0 },
  isDetecting: false, facePresent: false, confidence: null, frameCount: 0, timestamp: 0,
  unavailableReason: '',
};

// ===== 新增 7 模态默认值 =====
export const defaultCircadian: CircadianAnalysis = {
  riskScore: 0, lateNightRisk: 0, regularityScore: 0.5,
  lastActivityHour: 12, activitySampleCount: 0, isActive: false, timestamp: 0,
};

export const defaultCognitiveDistortion: CognitiveDistortionAnalysis = {
  riskScore: 0,
  categories: { catastrophizing: 0, blackAndWhite: 0, overgeneralization: 0, selfBlame: 0, hopelessness: 0, mindReading: 0, shouldStatements: 0 },
  totalDistortionCount: 0, timestamp: 0,
};

export const defaultHRV: HRVAnalysis = {
  riskScore: 0, heartRate: 0, hrvRmssd: 0, signalQuality: 0,
  isMeasuring: false, timestamp: 0,
};

export const defaultBreathing: BreathingAnalysis = {
  riskScore: 0, breathingRate: 0, regularityCV: 0, sighCount: 0,
  signalQuality: 0, isMeasuring: false, timestamp: 0,
};

export const defaultBehavioralActivation: BehavioralActivationAnalysis = {
  riskScore: 0, dailyInteractionCount: 0, explorationRate: 0,
  socialWithdrawalScore: 0, trendDirection: 'stable' as const,
  isActive: false, timestamp: 0,
};

export const defaultEyeMovement: EyeMovementAnalysis = {
  riskScore: 0, blinkRate: 0, downwardGazeRatio: 0, attentionScatter: 0,
  signalQuality: 0, isMeasuring: false, timestamp: 0,
};

export const defaultVoiceSemantics: VoiceSemanticsAnalysis = {
  riskScore: 0, firstPersonSingularRatio: 0, firstPersonPluralRatio: 0,
  absolutistRatio: 0, selfReferentialDensity: 0, negativeAffectRatio: 0,
  timestamp: 0,
};
