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
    avgDuration: number;           // 平均持续 ms
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
  isDetecting: boolean;
  confidence: number;              // 检测置信度 0-1
  frameCount: number;
  timestamp: number;
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

  // 融合元数据
  fusionWeights: {
    text: number;
    voice: number;
    facial: number;
    keyboard: number;
  };
  activeModalities: string[];      // 当前活跃的模态列表
  lastUpdated: number;

  // 温暖治愈信息
  caringMessage: string;
  evidence: string[];
  narrativeAnalysis: string;       // 叙事性多模态分析长文本
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
  microExpressions: { count: 0, avgDuration: 0, recentEmotions: [] },
  dominantExpression: '平静', expressionIntensity: 0,
  emotionMapping: { happiness: 0, sadness: 0, anger: 0, fear: 0, surprise: 0, disgust: 0, distress: 0 },
  isDetecting: false, confidence: 0, frameCount: 0, timestamp: 0,
};
