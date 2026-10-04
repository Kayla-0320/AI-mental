/**
 * 多模态情感感知上下文
 *
 * 整合模态感知（当前真实可用 9 项）：
 * 1. 键盘动力学  2. 文本语义分析  3. 声音声学  4. 面部微表情
 * 5. 昼夜活动    6. 认知扭曲      7. 行为激活  8. 眨眼/头姿
 * 9. 语音深层语义
 *
 * ⚠️ rPPG 心率、呼吸模式、睡眠三项本设备无可靠采集能力，
 * 一律不产出数值、不参与融合、不进入报告（见 hooks/perceptionCapabilities.ts）。
 *
 * 所有模态均按初中/高中/大学三阶段做年龄差异化校准。
 * 融合策略：基于置信度的动态加权 (Dynamic Confidence Weighting)
 * 参考：fusion.py 中的 fuse_dynamic_weighting
 */
import React, { createContext, useContext, useState, useCallback, useRef, useEffect, ReactNode } from 'react';
import { flushSync } from 'react-dom';
import { useLocation } from 'react-router-dom';
import { useKeyboardDynamics } from '../hooks/useKeyboardDynamics';
import { useTextAnalysis } from '../hooks/useTextAnalysis';
import { useVoiceAnalysis } from '../hooks/useVoiceAnalysis';
import { useFacialAnalysis, type FacialFrameSignal } from '../hooks/useFacialAnalysis';
import { useEyeStateModel, isBlendshapeClosed, type EyeStateModelReport } from '../hooks/useEyeStateModel';
import type { EyePatchPixels } from '../hooks/eyePatchCanvas';
import { useCircadianRhythm } from '../hooks/useCircadianRhythm';
import { useCognitiveDistortion } from '../hooks/useCognitiveDistortion';
import { useRPPG } from '../hooks/useRPPG';
import { useBreathing } from '../hooks/useBreathing';
import {
  RPPG_AVAILABLE,
  BREATHING_AVAILABLE,
  EYE_STATE_MODEL_AVAILABLE,
} from '../hooks/perceptionCapabilities';
import { useBehavioralActivation } from '../hooks/useBehavioralActivation';
import { useEyeTracking } from '../hooks/useEyeTracking';
import { useVoiceSemantics } from '../hooks/useVoiceSemantics';
import type { ComprehensiveEmotionState, KeyboardDynamics, TextAnalysis, VoiceAnalysis, FacialAnalysis } from '../types/multimodal.types';
import { defaultKeyboard, defaultText, defaultVoice, defaultFacial,
  defaultCircadian, defaultCognitiveDistortion, defaultHRV, defaultBreathing,
  defaultBehavioralActivation, defaultEyeMovement, defaultVoiceSemantics } from '../types/multimodal.types';
import { profileApi } from '../services';
import { useInterventionFeedback, type InterventionHistory } from '../hooks/useInterventionFeedback';
import { usePersonalBaseline, type BaselineDeviation, type PersonalBaseline } from '../hooks/usePersonalBaseline';
import { useAdaptiveWeights, type AdaptiveWeightState } from '../hooks/useAdaptiveWeights';
import { useMoodTrajectory, type MoodTrajectory as MoodTrajectoryType } from '../hooks/useMoodTrajectory';
import type { AgeGroup } from '../hooks/ageConfig';
import type { Gender } from '../hooks/ageConfig';
import { inferAgeGroup, ageFromBirthDate, getFusionConfig, getGenderModifiers, adjustWeightsByBaseline } from '../hooks/ageConfig';
import { useAuthStore } from '../store/authStore';

// 温暖治愈消息生成
const CARING_MESSAGES = [
  '你现在的感受很重要，谢谢你让自己被看见 💛',
  '无论什么情绪，都是正常的，你做得很好 🌿',
  '深呼吸，慢慢来，这里很安全 🌸',
  '你不需要完美，真实的你就很好 ✨',
  '每一种情绪都有它的意义，谢谢你愿意感受 🌈',
  '此刻的你，值得被温柔对待 💜',
  '累了就休息一下，不着急 🍃',
  '你的感受很重要，我一直在这里陪着你 🤗',
];

// ===== 当前不具备采集能力的模态（诚实披露，禁止编造数值） =====
const UNAVAILABLE_MODALITIES: string[] = [
  ...(RPPG_AVAILABLE ? [] : ['心率/HRV（rPPG）']),
  ...(BREATHING_AVAILABLE ? [] : ['呼吸频率']),
  '睡眠（无夜间传感器）',
];

const AVAILABILITY_NOTE =
  '本报告仅使用真实采集到的信号：键盘动力学、文字语义、语音声学、面部微表情、'
  + '语音语义、认知扭曲、昼夜活动记录、行为互动、眨眼（眼睑闭合度）。'
  + '心率、呼吸频率与睡眠本设备无法自动检测，因此不给出任何数值，也不纳入焦虑指数计算。';

// ===== 叙事性多模态分析生成器 =====
// 将真实可采集的模态数据转化为温暖的心理咨询师式观察报告
function generateNarrativeAnalysis(state: ComprehensiveEmotionState): string {
  const { keyboard, text, voice, facial, circadian, cognitiveDistortion, behavioralActivation, eyeMovement, voiceSemantics, dominantEmotion, emotionProbs } = state;
  const paragraphs: string[] = [];

  // --- 开场：整体情绪氛围 ---
  const emotionLabels = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];
  const top2 = emotionProbs.map((p, i) => ({ label: emotionLabels[i], prob: p }))
    .sort((a, b) => b.prob - a.prob).slice(0, 2);

  if (dominantEmotion && dominantEmotion !== '等待感知') {
    const moodDesc: Record<string, string> = {
      '快乐': '你的整体状态散发着温暖的光',
      '悲伤': '我们感受到你心里有一片淡淡的云',
      '焦虑': '我们注意到你似乎有些不安',
      '愤怒': '我们感受到你内心有一股力量在涌动',
      '中性': '你现在的状态比较平静',
    };
    paragraphs.push(`从我们捕捉到的信号来看，${moodDesc[dominantEmotion] || '你在经历一些情绪波动'}。${top2[0].label}是你此刻最主要的情绪（${Math.round(top2[0].prob * 100)}%），${top2[1].prob > 0.1 ? `同时也夹杂着一些${top2[1].label}（${Math.round(top2[1].prob * 100)}%）` : '情绪比较单一集中'}。`);
  }

  // --- 键盘动力学：打字行为描述 ---
  if (keyboard.isCollecting && keyboard.sampleSize > 3) {
    const kbParts: string[] = [];
    if (keyboard.typingSpeed > 60) {
      kbParts.push('你打字的速度很快');
    } else if (keyboard.typingSpeed > 0 && keyboard.typingSpeed < 20) {
      kbParts.push('你打字的速度比较慢，似乎在认真斟酌每一个字');
    } else if (keyboard.typingSpeed > 0) {
      kbParts.push('你的打字节奏比较平稳');
    }
    if (keyboard.pauseCount > 2) {
      kbParts.push(`中间有${keyboard.pauseCount}次较长的停顿（超过1.5秒），像是在思考要不要继续说下去`);
    }
    if (keyboard.backspaceBurst > 0) {
      kbParts.push(`有${keyboard.backspaceBurst}次连续删除的行为，可能有些话写了又删，犹豫着要不要表达`);
    }
    if (keyboard.rhythmScore < 0.5 && keyboard.sampleSize > 10) {
      kbParts.push('打字节奏不太规律，时快时慢');
    }
    if (keyboard.pressureIndex > 0.6) {
      kbParts.push('按键的力度感比较强');
    }
    if (kbParts.length > 0) {
      paragraphs.push(`从你的打字方式来看，${kbParts.join('，')}。${keyboard.anxietyIndex > 50 ? '这些信号让我们有些担心你的状态' : keyboard.anxietyIndex > 30 ? '你可能正在经历一些内心的波动' : '你的打字状态整体比较放松'}。`);
    }
  }

  // --- 文字语义：内容分析 ---
  if (text.timestamp > 0 && text.totalCharsAnalyzed > 0) {
    const textParts: string[] = [];
    if (text.dominantEmotion && text.dominantEmotion !== '中性') {
      textParts.push(`你写下的文字里，${text.dominantEmotion}的情绪比较明显`);
    }
    if (text.sentiment < -0.3) {
      textParts.push('整体偏向消极的情感色彩');
    } else if (text.sentiment > 0.3) {
      textParts.push('整体带着积极的情感色彩');
    }
    if (text.cognitiveMarkers.catastrophizing > 0.3) {
      textParts.push('我们注意到你有一些「灾难化」的思维倾向——把事情往最坏的方向想');
    }
    if (text.cognitiveMarkers.selfBlame > 0.3) {
      textParts.push('你的文字里有自我责备的痕迹，但请记住，很多事不是你的错');
    }
    if (text.cognitiveMarkers.blackAndWhite > 0.3) {
      textParts.push('你似乎习惯用「总是」「从不」这样的词，世界其实没有那么多绝对');
    }
    if (text.cognitiveMarkers.hopelessness > 0.3) {
      textParts.push('我们感受到你文字中的无力感，但请相信，情况是可以改变的');
    }
    if (text.firstPersonRate > 0.5) {
      textParts.push('你更多地关注着自己的感受，这是自我觉察的表现');
    }
    if (text.crisisLevel !== 'none') {
      textParts.push(`我们检测到了${text.crisisLevel === 'high' ? '较为严重的' : '一些'}危机信号，这让我们很关心你`);
    }
    if (textParts.length > 0) {
      paragraphs.push(`从你写下的内容来看，${textParts.join('；')}。你已经表达了${text.totalCharsAnalyzed}个字，每一句都是你内心真实的流露。`);
    }
  }

  // --- 声音声学：语音特征描述 ---
  if (voice.isRecording && voice.duration > 2) {
    const voiceParts: string[] = [];
    if (voice.pitch.mean > 250) {
      voiceParts.push('你的声音基频偏高');
    } else if (voice.pitch.mean > 0 && voice.pitch.mean < 130) {
      voiceParts.push('你的声音比较低沉');
    }
    if (voice.rate.tempo === 'fast') {
      voiceParts.push('语速偏快，像是有许多话急着要说');
    } else if (voice.rate.tempo === 'slow') {
      voiceParts.push('语速比较缓慢，像是在慢慢整理思绪');
    }
    if (voice.pauses.ratio > 0.3) {
      voiceParts.push('说话中有较多的停顿');
    }
    if (voice.energy.std > 10) {
      voiceParts.push('声音的起伏比较大，情绪波动明显');
    }
    if (voiceParts.length > 0) {
      paragraphs.push(`从你的声音来看，${voiceParts.join('，')}。${voice.detectedEmotion !== '中性' ? `声音中传递出的情绪倾向是${voice.detectedEmotion}` : '声音整体比较平稳'}。`);
    }
    // 添加语音识别内容
    if (voice.speechText) {
      paragraphs.push(`你刚才说到：“${voice.speechText}”。从你表达的内容中，我们能感受到你的情绪和想法。`);
    }
  }

  // --- 面部微表情：表情描述 ---
  // `facePresent` 是硬门槛：没人脸就不该生成任何关于表情的描述
  // （否则人离开画面后会继续拿上一帧的读数说话）。
  if (facial.isDetecting && facial.facePresent && facial.frameCount >= 2) {
    const faceParts: string[] = [];
    if (facial.dominantExpression && facial.dominantExpression !== '平静') {
      const exprDesc: Record<string, string> = {
        happiness: '你的嘴角有上扬的迹象，脸颊肌肉在运动——这是一个真诚的微笑信号',
        sadness: '我们注意到你的眉毛内侧有上扬的趋势，嘴角微微下拉，这是悲伤时典型的面部信号',
        anger: '你的眉毛区域有明显的下压和收紧，眼睑也在用力，这通常与愤怒或不满有关',
        fear: '你的眉毛上扬、眼睛睁大，下巴有下垂的趋势，这些信号让我们感受到你可能有些害怕或紧张',
        surprise: '你的眉毛上扬、眼睛睁大、下巴微张，看起来像是遇到了什么意外的事',
        disgust: '你的鼻子区域有皱起的迹象，上唇微微上扬，这通常是不适或厌恶的信号',
        distress: '你的面部多个区域同时呈现紧张状态，眉毛、眼睑和嘴角都在传递不安',
      };
      faceParts.push(exprDesc[facial.dominantExpression] || `你的主导表情是${facial.dominantExpression}`);
    }
    if (facial.microExpressions.count > 3) {
      // 只说**次数**。改造前这里还断言"持续时间不到半秒"，但那个时长是一个
      // 写死的常量（`avgDuration = 200`），不是测出来的 —— 我们拿不出这个结论。
      faceParts.push(`我们还捕捉到了${facial.microExpressions.count}次一闪而过的表情变化，那些你自己可能都没意识到`);
    }
    if (facial.expressionIntensity > 0.5) {
      faceParts.push('表情强度比较高，情绪表达很充分');
    }
    if (faceParts.length > 0) {
      paragraphs.push(`从你的面部表情来看，${faceParts.join('；')}。`);
    }
  }

  // --- 新增 7 模态描述 ---
  // 昼夜活动（只描述 App 内活动时间，不推断睡眠）
  if (circadian.isActive && circadian.riskScore > 0.2) {
    if (circadian.lateNightRisk > 0.3) {
      paragraphs.push(`我们注意到你最近在深夜（最近一次约${Math.round(circadian.lastActivityHour)}点）还在使用这个平台。深夜时段情绪更容易被放大，如果那段时间你正觉得难受，试着把想说的话留到白天再说，或者找信任的人聊聊。`);
    }
    if (circadian.activitySampleCount >= 10 && circadian.regularityScore < 0.4) {
      paragraphs.push('你使用平台的时间比较分散，时早时晚。相对固定的作息节奏有助于情绪稳定——这一点我们只能从你的使用时间观察到，真实的睡眠情况还是以你自己的记录和感受为准。');
    }
  }

  // 认知扭曲
  if (cognitiveDistortion.totalDistortionCount > 0) {
    const cogParts: string[] = [];
    if (cognitiveDistortion.categories.catastrophizing > 0.1) cogParts.push('「灾难化」——倾向于把事情往最坏的方向想');
    if (cognitiveDistortion.categories.selfBlame > 0.1) cogParts.push('「自我归咎」——习惯性地把问题归到自己身上');
    if (cognitiveDistortion.categories.hopelessness > 0.1) cogParts.push('「无望感」——觉得事情不会好起来');
    if (cognitiveDistortion.categories.blackAndWhite > 0.1) cogParts.push('「非黑即白」——用绝对的方式看待事物');
    if (cognitiveDistortion.categories.mindReading > 0.1) cogParts.push('「读心术」——总觉得别人在负面评价自己');
    if (cogParts.length > 0) {
      paragraphs.push(`我们从你的表达中察觉到一些思维模式：${cogParts.join('；')}。这些都是很常见的思维习惯，认识到它们就是改变的第一步。`);
    }
  }

  // ⚠️ 心率 / 呼吸 / 睡眠：本平台不具备可信采集能力（见 perceptionCapabilities.ts），
  // 因此报告里不生成任何相关段落，也不做任何数值推算。
  // 原实现会输出"心率偏高（180bpm）""呼吸频率偏快（xx次/分）"这类凭空数字，已移除。

  // 行为激活（只能观察到平台内互动，不能替代真实社交状况）
  if (behavioralActivation.isActive) {
    if (behavioralActivation.trendDirection === 'declining') {
      paragraphs.push('我们观察到你这几天在这个平台上的互动比之前少了一些。这可能只是最近比较忙，但如果你同时也觉得不太想和人说话，试着从一件很小的社交开始，比如给朋友发一条消息。');
    } else if (behavioralActivation.socialWithdrawalScore > 0.5) {
      paragraphs.push('今天你在平台里的互动次数偏少。这本身说明不了什么，但如果这种"不想打开、不想说话"的感觉持续存在，与他人保持连接会是很有帮助的保护因素。');
    }
  }

  // 头部姿态（⚠️ 不是眼球注视：本平台未接入虹膜跟踪，此处是头部俯仰代理指标）
  if (eyeMovement.isMeasuring && eyeMovement.signalQuality > 0.3) {
    if (eyeMovement.downwardGazeRatio > 0.5) {
      paragraphs.push(`摄像头画面中你较多处于低头姿态（占 ${Math.round(eyeMovement.downwardGazeRatio * 100)}% 的面部帧），这通常与低落情绪或回避有关。`);
    }
    if (eyeMovement.blinkRate > 25) {
      paragraphs.push(`眨眼频率偏高（约 ${eyeMovement.blinkRate} 次/分），可能是眼睛疲劳或紧张的信号。`);
    }
  }

  // 语音深层语义
  if (voiceSemantics.timestamp > 0 && voiceSemantics.riskScore > 0.2) {
    const vsParts: string[] = [];
    if (voiceSemantics.firstPersonSingularRatio > 0.15) vsParts.push('你较多地使用「我」来描述经历');
    if (voiceSemantics.absolutistRatio > 0.04) vsParts.push('有一些绝对化的表达（如「总是」「永远」）');
    if (voiceSemantics.negativeAffectRatio > 0.6) vsParts.push('消极情感词的比例偏高');
    if (vsParts.length > 0) {
      paragraphs.push(`从你说话的方式来看，${vsParts.join('，')}。语言是我们内心的镜子，觉察到这些模式就是力量的体现。`);
    }
  }

  // --- 总结：温暖收尾 ---
  if (paragraphs.length > 0) {
    const anxietyProb = emotionProbs[2] || 0;
    if (anxietyProb > 0.5) {
      paragraphs.push('综合所有信号，我们感受到你现在可能承受着不少压力。请记住，感到焦虑是完全正常的反应，你不需要独自面对这一切。试着做几次深呼吸，或者找信任的人聊聊，都会对你有帮助。');
    } else if (anxietyProb > 0.3) {
      paragraphs.push('综合来看，你可能正在经历一些轻微的情绪波动。这很正常，每个人的情绪都会有起有落。照顾好自己的身体和心情，适当休息，一切都会好起来的。');
    } else {
      paragraphs.push('综合所有信号，你现在的整体状态还不错。继续保持对自己的觉察，这是心理健康很重要的一步。如果任何时候你想聊聊，我们都在这里。');
    }
  }

  // --- 数据来源说明：如实告知哪些指标并没有被测量 ---
  if (UNAVAILABLE_MODALITIES.length > 0) {
    paragraphs.push(`（说明：以上分析仅基于真实采集到的文字、语音、键盘、面部、互动与使用时间信号。`
      + `以下指标未在本设备上采集，报告中不含它们的任何数值：${UNAVAILABLE_MODALITIES.join('、')}。）`);
  }

  return paragraphs.join('\n\n');
}

interface AnxietyContextType {
  comprehensiveState: ComprehensiveEmotionState;
  cameraEnabled: boolean;
  /**
   * 最近一次媒体启动失败的原因（中文，可直接上屏）；空串表示没有错误。
   *
   * 存在的理由：改造前摄像头启动失败会被 `useFacialAnalysis.start()` 静默吞掉，
   * 于是"用户拒了摄像头权限"的表现是界面显示"摄像头已开启"、实际一个数据都没有。
   * 失败必须能被看见。
   */
  mediaNotice: string;
  /**
   * 摄像头 + 麦克风总开关。
   * @returns 是否真的打开了（关闭操作返回 true）。失败时返回 false 并设置 `mediaNotice`。
   */
  toggleCamera: () => Promise<boolean>;
  modalityStatus: {
    keyboardStarted: boolean;
    voiceStarted: boolean;
    facialStarted: boolean;
    textListenerRegistered: boolean;
  };
  updateCounter: number;
  keyboardMetrics: KeyboardDynamics;
  textMetrics: TextAnalysis;
  voiceMetrics: VoiceAnalysis;
  facialMetrics: FacialAnalysis;
  circadianMetrics: typeof defaultCircadian;
  cognitiveMetrics: typeof defaultCognitiveDistortion;
  hrvMetrics: typeof defaultHRV;
  breathingMetrics: typeof defaultBreathing;
  behavioralMetrics: typeof defaultBehavioralActivation;
  eyeMetrics: typeof defaultEyeMovement;
  voiceSemanticsMetrics: typeof defaultVoiceSemantics;
  analyzeText: (text: string) => void;
  // 纯语音输入（只开麦克风，不开摄像头）—— 供聊天页"语音录入"用。
  // 与 toggleCamera 分开是刻意的：聊天时不该强制打开摄像头。
  voiceInputActive: boolean;
  startVoiceInput: () => Promise<void>;
  /**
   * 结束语音输入。
   *
   * 返回 **最后一段定稿文本**：撤销端点检测 + 离线复识别要 ~600ms，
   * 调用方必须 await 才能拿到"点结束前刚说的那句"，不能只靠订阅 state。
   */
  stopVoiceInput: () => Promise<string>;
  /** 语音定稿序号：每次有新的一段定稿就自增，供页面区分"新的一段" */
  voiceFinalSeq: number;
  /** 最近一段定稿的文本（与 `voiceFinalSeq` 配套使用） */
  voiceFinalText: string;
  reportToServer: (context?: string) => Promise<void>;
  realityTask: any;
  showTaskModal: boolean;
  setShowTaskModal: (v: boolean) => void;
  fetchRealityTask: () => Promise<void>;
  reset: () => void;
  // 个人基线
  baseline: PersonalBaseline;
  baselineDeviation: BaselineDeviation;
  getZScoreForValue: (modality: keyof Omit<PersonalBaseline, 'emotionBaseline' | 'createdAt' | 'lastUpdated' | 'sampleCount'>, value: number) => number;
  // 自适应权重
  adaptiveWeights: AdaptiveWeightState;
  // 情绪轨迹
  moodTrajectory: MoodTrajectoryType;
  getWeeklyReport: () => string;
  // 干预历史
  interventionHistory: InterventionHistory;
  getRecommendedIntervention: () => string;
}

const defaultComprehensive: ComprehensiveEmotionState = {
  emotionProbs: [0.2, 0.2, 0.2, 0.2, 0.2],
  confidence: 0,
  dominantEmotion: '等待感知',
  riskLevel: 'low',
  keyboard: { ...defaultKeyboard },
  text: { ...defaultText },
  voice: { ...defaultVoice },
  facial: { ...defaultFacial },
  circadian: { ...defaultCircadian },
  cognitiveDistortion: { ...defaultCognitiveDistortion },
  hrv: { ...defaultHRV },
  breathing: { ...defaultBreathing },
  behavioralActivation: { ...defaultBehavioralActivation },
  eyeMovement: { ...defaultEyeMovement },
  voiceSemantics: { ...defaultVoiceSemantics },
  fusionWeights: { text: 0, voice: 0, facial: 0, keyboard: 0, circadian: 0, cognitive: 0, hrv: 0, breathing: 0, behavioralAct: 0, eye: 0, voiceSemantics: 0 },
  activeModalities: [],
  lastUpdated: 0,
  unavailableModalities: UNAVAILABLE_MODALITIES,
  availabilityNote: AVAILABILITY_NOTE,
  caringMessage: CARING_MESSAGES[0],
  evidence: [],
  narrativeAnalysis: '',
};

const AnxietyContext = createContext<AnxietyContextType | null>(null);

export function AnxietyProvider({ children }: { children: ReactNode }) {
  // 年龄分组：从用户信息推断
  const user = useAuthStore(s => s.user);
  const userAge = (user as any)?.age ?? null;
  const birthDate = (user as any)?.birthDate ?? null;
  const ageGroup: AgeGroup | null = inferAgeGroup(userAge ?? ageFromBirthDate(birthDate));
  const userGender: Gender | null = ((user as any)?.gender as Gender) ?? null;
  const genderModifiers = getGenderModifiers(userGender);

  // 硬件信号回调：面部帧 → rPPG + 眨眼/头姿（复用同一帧的 MediaPipe 推理结果）
  const onFrameSignalRef = useRef<(signal: FacialFrameSignal) => void>();
  onFrameSignalRef.current = (signal) => {
    rppg.updateFromFrame(signal.greenMean, Date.now());
    eyeTracking.updateFromFacialFrame(signal);
    // 把同帧的 blendshape 判定留给第二判定通道做一致率比较（不参与风险）
    eyeStateRef.current?.observeBlendshape(isBlendshapeClosed(signal.blinkScore));
  };
  const onFrameSignal = useCallback((signal: FacialFrameSignal) => {
    onFrameSignalRef.current?.(signal);
  }, []);

  // 硬件信号回调：眼 patch → 浏览器内 ONNX 每眼二次判定（**只上报，不进风险**）
  // 用 ref 转发，避免 useFacialAnalysis 与 useEyeStateModel 的创建顺序互相牵制
  const onEyePatchRef = useRef<((patches: EyePatchPixels) => void) | null>(null);
  const onEyePatchFrame = useCallback((patches: EyePatchPixels) => {
    onEyePatchRef.current?.(patches);
  }, []);
  const onEyeStateReportRef = useRef<((r: EyeStateModelReport) => void) | null>(null);

  // 硬件信号回调：音频能量 → 呼吸
  const onAudioEnergyRef = useRef<(rms: number) => void>();
  onAudioEnergyRef.current = (rms) => {
    breathingHook.updateFromAudioEnergy(rms);
  };
  const onAudioEnergy = useCallback((rms: number) => {
    onAudioEnergyRef.current?.(rms);
  }, []);

  // 语音定稿回调：把"定稿那一刻"转发给订阅页（聊天页靠它把最后一句落进输入框）。
  // 只靠订阅 voiceMetrics 会漏 —— 点"结束"之后 isRecording 先变 false，
  // 而最后一段定稿还要等撤销端点检测 + 离线复识别（约 600ms）才到。
  const [voiceFinalSeq, setVoiceFinalSeq] = useState(0);
  const voiceFinalRef = useRef<{ text: string; seq: number }>({ text: '', seq: 0 });
  const handleVoiceFinal = useCallback((text: string, seq: number) => {
    voiceFinalRef.current = { text, seq };
    setVoiceFinalSeq(seq);
  }, []);

  // 将年龄分组传递给各 Hook
  const keyboard = useKeyboardDynamics(ageGroup);
  const textAnalysis = useTextAnalysis(ageGroup);
  // 聊天页的语音录入走**整段录入**模式：说话停顿不定稿，用户点"结束"才把整段
  // 音频交给离线大模型识别。原先的"停顿几秒自动截断定稿"会让每一段都缺上下文
  // （实测 12s、句间停顿 1.2s 的音频：端点切段后"重点呢"→"重点来"、"动荡"→"动量"，
  // 整段识别则全对），所以那次截断既不准、也不是用户想要的交互。
  // 通话页用的是另一个 hook 实例（`useVoiceCall`），仍保持"停顿即定稿"。
  const voice = useVoiceAnalysis(ageGroup, onAudioEnergy, handleVoiceFinal, null, 'utterance');
  const facial = useFacialAnalysis(ageGroup, onFrameSignal, onEyePatchFrame);
  // 新增 7 模态 Hook（均传入 ageGroup 做年龄差异化校准）
  const circadian = useCircadianRhythm(ageGroup);
  const cognitiveDistortion = useCognitiveDistortion(ageGroup);
  const rppg = useRPPG(ageGroup);
  const breathingHook = useBreathing(ageGroup);
  const behavioralActivation = useBehavioralActivation(ageGroup);
  const eyeTracking = useEyeTracking(ageGroup);

  // ── 每眼「睁闭眼」第二判定通道（默认关闭，见 perceptionCapabilities 的说明）──
  // 默认关闭时不加载任何 ONNX 资源、不产生任何推理；开启后也只上报不进风险。
  const eyeState = useEyeStateModel(EYE_STATE_MODEL_AVAILABLE, (r) => onEyeStateReportRef.current?.(r));
  const eyeStateRef = useRef<ReturnType<typeof useEyeStateModel> | null>(null);
  eyeStateRef.current = eyeState;
  onEyePatchRef.current = eyeState.onEyePatches;
  onEyeStateReportRef.current = (r) => {
    eyeTracking.updateFromEyeStateModel({ ...r, status: eyeState.status });
  };
  useEffect(() => {
    eyeTracking.setEyeStateStatus(eyeState.status);
    // 只依赖 status：eyeTracking 的 setter 是稳定引用
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [eyeState.status]);
  const voiceSemantics = useVoiceSemantics(ageGroup);
  // 干预效果闭环
  const interventionFeedback = useInterventionFeedback();
  // 个人基线校准
  const personalBaseline = usePersonalBaseline();
  // 模态权重自适应
  const adaptiveWeightsHook = useAdaptiveWeights();
  // 情绪时间轨迹
  const moodTrajectoryHook = useMoodTrajectory();

  // 用 ref 稳定 analyzeText 引用，避免事件监听器反复重建
  const analyzeTextRef = useRef(textAnalysis.analyzeText);
  analyzeTextRef.current = textAnalysis.analyzeText;

  // 直接管理指标状态（不依赖 hook 的内部状态）
  const [keyboardMetrics, setKeyboardMetrics] = useState(keyboard.metrics);
  const [textMetrics, setTextMetrics] = useState(textAnalysis.metrics);
  const [voiceMetrics, setVoiceMetrics] = useState(voice.metrics);
  const [facialMetrics, setFacialMetrics] = useState(facial.metrics);
  // 新增 7 模态指标状态
  const [circadianMetrics, setCircadianMetrics] = useState(circadian.metrics);
  const [cognitiveMetrics, setCognitiveMetrics] = useState(cognitiveDistortion.metrics);
  const [hrvMetrics, setHrvMetrics] = useState(rppg.metrics);
  const [breathingMetrics, setBreathingMetrics] = useState(breathingHook.metrics);
  const [behavioralMetrics, setBehavioralMetrics] = useState(behavioralActivation.metrics);
  const [eyeMetrics, setEyeMetrics] = useState(eyeTracking.metrics);
  const [voiceSemanticsMetrics, setVoiceSemanticsMetrics] = useState(voiceSemantics.metrics);

  // 同步 hook 的 metrics 到本地状态（使用 JSON.stringify 强制检测变化）
  useEffect(() => {
    setKeyboardMetrics(keyboard.metrics);
  }, [JSON.stringify(keyboard.metrics)]);
  useEffect(() => {
    setTextMetrics(textAnalysis.metrics);
  }, [JSON.stringify(textAnalysis.metrics)]);
  useEffect(() => {
    setVoiceMetrics(voice.metrics);
  }, [JSON.stringify(voice.metrics)]);
  useEffect(() => {
    setFacialMetrics(facial.metrics);
  }, [JSON.stringify(facial.metrics)]);
  useEffect(() => { setCircadianMetrics(circadian.metrics); }, [JSON.stringify(circadian.metrics)]);
  useEffect(() => { setCognitiveMetrics(cognitiveDistortion.metrics); }, [JSON.stringify(cognitiveDistortion.metrics)]);
  useEffect(() => { setHrvMetrics(rppg.metrics); }, [JSON.stringify(rppg.metrics)]);
  useEffect(() => { setBreathingMetrics(breathingHook.metrics); }, [JSON.stringify(breathingHook.metrics)]);
  useEffect(() => { setBehavioralMetrics(behavioralActivation.metrics); }, [JSON.stringify(behavioralActivation.metrics)]);
  useEffect(() => { setEyeMetrics(eyeTracking.metrics); }, [JSON.stringify(eyeTracking.metrics)]);
  useEffect(() => { setVoiceSemanticsMetrics(voiceSemantics.metrics); }, [JSON.stringify(voiceSemantics.metrics)]);

  // 语音识别文本 → 送入文本分析 + 认知扭曲 + 语音语义
  useEffect(() => {
    if (voiceMetrics.speechText && voiceMetrics.speechText.trim().length > 0) {
      console.log('[多模态] 语音识别文本:', voiceMetrics.speechText);
      analyzeTextRef.current(voiceMetrics.speechText);
      cognitiveDistortion.analyzeText(voiceMetrics.speechText);
      voiceSemantics.analyzeSpeechText(voiceMetrics.speechText);
    }
  }, [voiceMetrics.speechText]); // eslint-disable-line react-hooks/exhaustive-deps

  const [comprehensiveState, setComprehensiveState] = useState<ComprehensiveEmotionState>({ ...defaultComprehensive });
  const [cameraEnabled, setCameraEnabled] = useState(false);
  /** 最近一次媒体启动失败的原因（中文，可上屏）。空串 = 没有错误。 */
  const [mediaNotice, setMediaNotice] = useState('');
  const [realityTask, setRealityTask] = useState<any>(null);
  const [showTaskModal, setShowTaskModal] = useState(false);
  const [modalityStatus, setModalityStatus] = useState({
    keyboardStarted: false,
    voiceStarted: false,
    facialStarted: false,
    textListenerRegistered: false,
  });
  const [updateCounter, setUpdateCounter] = useState(0); // 用于验证状态更新是否工作
  const lastReportRef = useRef(0);
  const hasTriggeredRef = useRef(false);
  const fusionIntervalRef = useRef(0);
  /**
   * 媒体启动的重入锁。
   *
   * 摄像头/麦克风的所有启动入口（手动事件、开关按钮）共用这一把锁：
   * `cameraEnabled` 这个 state 要等 `await` 之后才更新，挡不住"await 期间又点一次"，
   * 而那一次会真的再开一路设备 —— 这是改造前 4 次点击泄漏出 5 路麦克风的直接原因。
   */
  const startingMediaRef = useRef(false);

  // 自动启动键盘分析（组件完全挂载后）
  useEffect(() => {
    console.log('[多模态] AnxietyProvider 已挂载，准备启动键盘分析');
    const timer = setTimeout(() => {
      console.log('[多模态] 调用 keyboard.start()...');
      keyboard.start();
      console.log('[多模态] keyboard.start() 已调用，当前 metrics:', keyboard.metrics);
    }, 500); // 延迟 500ms 确保 hook 完全初始化
    return () => {
      clearTimeout(timer);
      keyboard.stop();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // 监听手动启动事件
  useEffect(() => {
    /**
     * 处理「启动多模态感知」事件。
     *
     * 改造要点（这三条以前都是错的）：
     * 1. **重入守卫**：以前完全没有守卫，每次事件都真的再开一路摄像头 + 麦克风
     *    （实测连点 4 次 → 5 条 live 麦克风轨道）。
     * 2. **不提前撒谎**：以前先乐观地把 `isRecording / isDetecting` 置 true 再去启动，
     *    启动失败时界面已经显示"采集正常"了。现在只在**成功之后**才置位。
     * 3. **await**：`voice.start() / facial.start()` 是 async 的，以前不 await，
     *    于是那两层 `try/catch` 根本抓不到它们的失败（会变成 unhandled rejection）。
     */
    const handleStart = () => {
      if (startingMediaRef.current) return;
      if (voice.metrics.isRecording || facial.metrics.isDetecting) return;
      startingMediaRef.current = true;
      void (async () => {
        try {
          try {
            keyboard.start();
            setKeyboardMetrics(prev => ({ ...prev, isCollecting: true }));
          } catch (err) {
            console.error('[多模态] 键盘启动失败:', err);
          }

          // 麦克风和摄像头**分别**记录结果：只给麦克风权限是很常见的情况，
          // 不该因为摄像头失败就把已经开好的麦克风也当成失败。
          const [voiceOk, facialOk] = await Promise.all([
            voice.start().then(
              () => true,
              (err) => { console.error('[多模态] 声音启动失败:', err); return false; },
            ),
            facial.start().then(
              () => true,
              (err) => { console.error('[多模态] 面部启动失败:', err); return false; },
            ),
          ]);

          setVoiceMetrics(prev => ({ ...prev, isRecording: voiceOk }));
          setFacialMetrics(prev => ({ ...prev, isDetecting: facialOk }));
          setTextMetrics(prev => ({ ...prev, timestamp: Date.now() }));
          setModalityStatus({
            keyboardStarted: true,
            voiceStarted: voiceOk,
            facialStarted: facialOk,
            textListenerRegistered: true,
          });
          // 只有真的有设备启动成功才置总开关。以前无条件置 true，
          // 于是"摄像头权限被拒"时界面依然显示已开启、而一个数据都没有。
          setCameraEnabled(voiceOk || facialOk);
          setComprehensiveState(prev => ({ ...prev, lastUpdated: Date.now() }));
          setUpdateCounter(prev => prev + 1);
          console.log(`[多模态] 启动结果 voice=${voiceOk} facial=${facialOk}`);
        } finally {
          startingMediaRef.current = false;
        }
      })();
    };
    window.addEventListener('start-multimodal-analysis', handleStart);
    return () => window.removeEventListener('start-multimodal-analysis', handleStart);
  }, [keyboard, voice, facial]);

  // 全局文本采集：拦截所有 input/textarea 的输入（仅注册一次）
  useEffect(() => {
    let debounceTimer: number | null = null;
    let isComposing = false; // IME 组合输入标志

    const handleInput = (e: Event) => {
      // IME 组合输入期间不触发分析
      if (isComposing) return;

      const target = e.target as HTMLInputElement | HTMLTextAreaElement;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {
        const text = target.value;
        behavioralActivation.recordInteraction();
        circadian.recordActivity();
        if (text.length > 2) {
          // 防抖：停止打字 500ms 后才分析，避免每次按键都触发重渲染
          if (debounceTimer) clearTimeout(debounceTimer);
          debounceTimer = window.setTimeout(() => {
            console.log('[多模态] 文本采集:', text.slice(0, 50));
            analyzeTextRef.current(text);
            (window as any).__recentUserText = text;
          }, 500);
        }
      }
    };

    // IME 组合输入开始/结束事件
    const handleCompositionStart = () => { isComposing = true; };
    const handleCompositionEnd = (e: Event) => {
      isComposing = false;
      // 组合输入结束后，手动触发一次分析
      const target = e.target as HTMLInputElement | HTMLTextAreaElement;
      if (target && target.value.length > 2) {
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = window.setTimeout(() => {
          analyzeTextRef.current(target.value);
          (window as any).__recentUserText = target.value;
        }, 100);
      }
    };

    document.addEventListener('input', handleInput, true);
    document.addEventListener('compositionstart', handleCompositionStart, true);
    document.addEventListener('compositionend', handleCompositionEnd, true);
    console.log('[多模态] 文本采集监听器已注册（含 IME 支持）');
    return () => {
      if (debounceTimer) clearTimeout(debounceTimer);
      document.removeEventListener('input', handleInput, true);
      document.removeEventListener('compositionstart', handleCompositionStart, true);
      document.removeEventListener('compositionend', handleCompositionEnd, true);
      console.log('[多模态] 文本采集监听器已移除');
    };
  }, []); // 空依赖，仅注册一次

  // 四模态融合 (每 3 秒执行一次，使用年龄专属权重)
  const performFusion = useCallback(() => {
    const activeModalities: string[] = [];
    const weights: Record<string, number> = {
      text: 0, voice: 0, facial: 0, keyboard: 0,
      circadian: 0, cognitive: 0, hrv: 0, breathing: 0,
      behavioralAct: 0, eye: 0, voiceSemantics: 0,
    };

    // 年龄专属融合权重（11 模态）
    const fusionConfig = getFusionConfig(ageGroup);
  
    // 确定活跃模态和置信度（使用年龄专属权重）
    if (textMetrics.timestamp > 0) {
      activeModalities.push('text');
      weights.text = Math.min(fusionConfig.text, textMetrics.dominantConfidence * 0.5 + fusionConfig.text * 0.6);
    }
    if (voiceMetrics.isRecording) {
      activeModalities.push('voice');
      weights.voice = fusionConfig.voice;
    }
    // ⚠️ 面部模态的可信度必须来自**真实信号质量**。
    //    `confidence === null` 表示"本帧没有可信读数"（没检测到人脸 / MediaPipe 不可用 /
    //    脸太小 / 侧脸太偏），此时**不能**把该模态算进融合。
    //    改造前这里用的是帧计数常量 `min(0.95, 0.7 + frameCount*0.005)` ——
    //    人已经离开画面，系统还在按 0.95 的置信度读他的情绪。
    if (facialMetrics.isDetecting && facialMetrics.confidence !== null) {
      activeModalities.push('facial');
      weights.facial = Math.min(fusionConfig.facial, Math.max(fusionConfig.facial * 0.5, facialMetrics.confidence * fusionConfig.facial));
    }
    if (keyboardMetrics.isCollecting) {
      activeModalities.push('keyboard');
      weights.keyboard = fusionConfig.keyboard;
    }
    // 新增 7 模态活跃判定
    // ⚠️ 本设备不具备心率/呼吸采集能力，这两个模态永不参与融合，
    // 否则噪声数据会污染焦虑指数（详见 hooks/perceptionCapabilities.ts）
    if (circadianMetrics.isActive) { activeModalities.push('circadian'); weights.circadian = fusionConfig.circadian; }
    if (cognitiveMetrics.riskScore > 0 || cognitiveMetrics.timestamp > 0) { activeModalities.push('cognitive'); weights.cognitive = fusionConfig.cognitive; }
    if (RPPG_AVAILABLE && hrvMetrics.isMeasuring) { activeModalities.push('hrv'); weights.hrv = fusionConfig.hrv; }
    if (BREATHING_AVAILABLE && breathingMetrics.isMeasuring) { activeModalities.push('breathing'); weights.breathing = fusionConfig.breathing; }
    if (behavioralMetrics.isActive) { activeModalities.push('behavioralAct'); weights.behavioralAct = fusionConfig.behavioralAct; }
    if (eyeMetrics.isMeasuring) { activeModalities.push('eye'); weights.eye = fusionConfig.eye; }
    if (voiceSemanticsMetrics.timestamp > 0) { activeModalities.push('voiceSemantics'); weights.voiceSemantics = fusionConfig.voiceSemantics; }
  
    // 如果没有活跃模态
    if (activeModalities.length === 0) {
      console.log('[多模态融合] 无活跃模态', {
        textTs: textMetrics.timestamp,
        textChars: textMetrics.totalCharsAnalyzed,
        voiceRec: voiceMetrics.isRecording,
        voiceDur: voiceMetrics.duration,
        faceDet: facialMetrics.isDetecting,
        faceConf: facialMetrics.confidence,
        kbCollect: keyboardMetrics.isCollecting,
        kbSample: keyboardMetrics.sampleSize,
      });
      return;
    }
    console.log('[多模态融合] 活跃模态:', activeModalities, '权重:', weights);

    // === 模态降级策略：基于信号质量动态调整权重 ===
    const signalQualities: Record<string, number> = {
      text: 1.0,
      voice: 1.0,
      facial: facialMetrics.confidence ?? 0,
      keyboard: 1.0,
      circadian: 1.0,
      cognitive: 1.0,
      hrv: hrvMetrics.signalQuality,
      breathing: breathingMetrics.signalQuality,
      behavioralAct: 1.0,
      eye: eyeMetrics.signalQuality,
      voiceSemantics: 1.0,
    };
    // 信号质量 < 0.3 时降低权重，避免低质量数据污染融合
    for (const key of Object.keys(weights)) {
      const sq = signalQualities[key] ?? 1.0;
      if (sq < 0.3) weights[key] *= sq; // 质量越低，权重衰减越多
    }

    // === 个人基线偏移 → 权重自适应调整 ===
    if (personalBaseline.isCalibrated()) {
      const baselineZScores: Record<string, number> = {
        hrv: personalBaseline.getZScoreForValue('heartRate', hrvMetrics.heartRate || 0),
        breathing: personalBaseline.getZScoreForValue('breathingRate', breathingMetrics.breathingRate || 0),
        keyboard: personalBaseline.getZScoreForValue('typingSpeed', keyboardMetrics.typingSpeed || 0),
        eye: personalBaseline.getZScoreForValue('blinkRate', eyeMetrics.blinkRate || 0),
        voice: personalBaseline.getZScoreForValue('voiceF0', voiceMetrics.pitch?.mean || 0),
      };
      const adjusted = adjustWeightsByBaseline(weights, baselineZScores);
      Object.assign(weights, adjusted);
    }

    // === 自适应权重系数应用 ===
    const adaptiveCoefficients = adaptiveWeightsHook.getAllCoefficients();
    if (Object.keys(adaptiveCoefficients).length > 0) {
      for (const key of Object.keys(weights)) {
        const coeff = adaptiveCoefficients[key] ?? 1.0;
        weights[key] *= coeff;
      }
    }

    // 归一化权重
    const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0);
    if (totalWeight === 0) return;
    const normalizedWeights = {
      text: weights.text / totalWeight,
      voice: weights.voice / totalWeight,
      facial: weights.facial / totalWeight,
      keyboard: weights.keyboard / totalWeight,
      circadian: weights.circadian / totalWeight,
      cognitive: weights.cognitive / totalWeight,
      hrv: weights.hrv / totalWeight,
      breathing: weights.breathing / totalWeight,
      behavioralAct: weights.behavioralAct / totalWeight,
      eye: weights.eye / totalWeight,
      voiceSemantics: weights.voiceSemantics / totalWeight,
    };

    // 文本情绪 (8维→5维: [快乐, 悲伤, 焦虑, 愤怒, 中性])
    const textEmotion = textMetrics.emotionDistribution;
    const textProbs5 = [
      textEmotion[0], // 快乐
      textEmotion[1], // 悲伤
      textEmotion[2], // 焦虑
      textEmotion[3], // 愤怒
      textEmotion[7], // 中性
    ];

    // 语音情绪
    const voiceProbs5 = voiceMetrics.emotionProbs;

    // 面部情绪 → 5维
    const faceMap = facialMetrics.emotionMapping;
    const facialProbs5 = [
      faceMap.happiness,
      faceMap.sadness,
      faceMap.fear * 0.5 + faceMap.distress * 0.5, // 焦虑 = 恐惧+痛苦
      faceMap.anger,
      Math.max(0, 1 - Object.values(faceMap).reduce((s, v) => s + v, 0) / 3), // 中性
    ];

    // 键盘焦虑 → 5维
    const kbAnxiety = keyboardMetrics.anxietyIndex / 100;
    const keyboardProbs5 = [0.1, kbAnxiety * 0.3, kbAnxiety, 0.1, Math.max(0.1, 1 - kbAnxiety * 0.5)];

    // 新增 7 模态 → 5维情绪映射 [快乐, 悲伤, 焦虑, 愤怒, 中性]
    const circProbs5 = [0.05, circadianMetrics.riskScore * 0.4, circadianMetrics.riskScore * 0.5, 0.05, Math.max(0.1, 1 - circadianMetrics.riskScore * 0.5)];
    const cogProbs5 = [0.05, cognitiveMetrics.riskScore * 0.45, cognitiveMetrics.riskScore * 0.35, cognitiveMetrics.riskScore * 0.1, Math.max(0.1, 1 - cognitiveMetrics.riskScore * 0.4)];
    const hrvProbs5 = [0.1, hrvMetrics.riskScore * 0.3, hrvMetrics.riskScore * 0.5, 0.05, Math.max(0.1, 1 - hrvMetrics.riskScore * 0.4)];
    const breathProbs5 = [0.1, breathingMetrics.riskScore * 0.3, breathingMetrics.riskScore * 0.5, 0.05, Math.max(0.1, 1 - breathingMetrics.riskScore * 0.4)];
    const behProbs5 = [0.1, behavioralMetrics.riskScore * 0.5, behavioralMetrics.riskScore * 0.2, 0.05, Math.max(0.1, 1 - behavioralMetrics.riskScore * 0.4)];
    const eyeProbs5 = [0.1, eyeMetrics.riskScore * 0.2, eyeMetrics.riskScore * 0.5, 0.05, Math.max(0.1, 1 - eyeMetrics.riskScore * 0.3)];
    const vsProbs5 = [0.05, voiceSemanticsMetrics.riskScore * 0.4, voiceSemanticsMetrics.riskScore * 0.35, voiceSemanticsMetrics.riskScore * 0.1, Math.max(0.1, 1 - voiceSemanticsMetrics.riskScore * 0.4)];

    // 加权融合（11 模态）
    const fusedProbs = [0, 0, 0, 0, 0];
    if (weights.text > 0) {
      fusedProbs.forEach((_, i) => { fusedProbs[i] += textProbs5[i] * normalizedWeights.text; });
    }
    if (weights.voice > 0) {
      fusedProbs.forEach((_, i) => { fusedProbs[i] += voiceProbs5[i] * normalizedWeights.voice; });
    }
    if (weights.facial > 0) {
      fusedProbs.forEach((_, i) => { fusedProbs[i] += facialProbs5[i] * normalizedWeights.facial; });
    }
    if (weights.keyboard > 0) {
      fusedProbs.forEach((_, i) => { fusedProbs[i] += keyboardProbs5[i] * normalizedWeights.keyboard; });
    }
    if (weights.circadian > 0) {
      fusedProbs.forEach((_, i) => { fusedProbs[i] += circProbs5[i] * normalizedWeights.circadian; });
    }
    if (weights.cognitive > 0) {
      fusedProbs.forEach((_, i) => { fusedProbs[i] += cogProbs5[i] * normalizedWeights.cognitive; });
    }
    if (weights.hrv > 0) {
      fusedProbs.forEach((_, i) => { fusedProbs[i] += hrvProbs5[i] * normalizedWeights.hrv; });
    }
    if (weights.breathing > 0) {
      fusedProbs.forEach((_, i) => { fusedProbs[i] += breathProbs5[i] * normalizedWeights.breathing; });
    }
    if (weights.behavioralAct > 0) {
      fusedProbs.forEach((_, i) => { fusedProbs[i] += behProbs5[i] * normalizedWeights.behavioralAct; });
    }
    if (weights.eye > 0) {
      fusedProbs.forEach((_, i) => { fusedProbs[i] += eyeProbs5[i] * normalizedWeights.eye; });
    }
    if (weights.voiceSemantics > 0) {
      fusedProbs.forEach((_, i) => { fusedProbs[i] += vsProbs5[i] * normalizedWeights.voiceSemantics; });
    }

    // 归一化
    const probSum = fusedProbs.reduce((a, b) => a + b, 0) || 1;
    const normalizedProbs = fusedProbs.map(p => Math.round((p / probSum) * 100) / 100);

    // 主导情绪
    const labels = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];
    const dominantIdx = normalizedProbs.indexOf(Math.max(...normalizedProbs));
    const dominantEmotion = labels[dominantIdx];
    const confidence = normalizedProbs[dominantIdx];

    // 风险等级（使用年龄校准偏移）
    const anxietyProb = normalizedProbs[2];
    const riskOffset = fusionConfig.riskThresholdOffset;
    const adjustedAnxiety = Math.max(0, Math.min(1, anxietyProb - riskOffset));
    const riskLevel = adjustedAnxiety > 0.7 ? 'crisis' : adjustedAnxiety > 0.5 ? 'high' : adjustedAnxiety > 0.3 ? 'medium' : 'low';

    // 生成证据
    const evidence: string[] = [];
    if (activeModalities.includes('text')) {
      evidence.push(`文本分析：主导情绪=${textMetrics.dominantEmotion}，情感效价=${textMetrics.sentiment}`);
      if (textMetrics.crisisLevel !== 'none') {
        evidence.push(`⚠️ 危机信号：${textMetrics.crisisLevel}级 (${textMetrics.crisisMarkers.join('、')})`);
      }
    }
    if (activeModalities.includes('voice')) {
      const voiceEvidence = `声音分析：基频=${voiceMetrics.pitch.mean}Hz，语速=${voiceMetrics.rate.speechRate}音节/s`;
      if (voiceMetrics.speechText) {
        evidence.push(`${voiceEvidence}，语音内容：“${voiceMetrics.speechText.slice(0, 50)}${voiceMetrics.speechText.length > 50 ? '...' : ''}”`);
      } else {
        evidence.push(voiceEvidence);
      }
    }
    if (activeModalities.includes('facial')) {
      evidence.push(`面部分析：${facialMetrics.dominantExpression}，强度=${facialMetrics.expressionIntensity}，微表情${facialMetrics.microExpressions.count}次`);
    }
    if (activeModalities.includes('keyboard')) {
      evidence.push(`键盘动力学：${keyboardMetrics.typingSpeed}字/分，焦虑指数=${keyboardMetrics.anxietyIndex}`);
    }
    if (activeModalities.includes('circadian')) {
      evidence.push(`昼夜活动：深夜活动风险=${circadianMetrics.lateNightRisk}，最近活动${circadianMetrics.lastActivityHour}时，样本${circadianMetrics.activitySampleCount}条`);
    }
    if (activeModalities.includes('cognitive')) {
      evidence.push(`认知扭曲：风险=${cognitiveMetrics.riskScore}，${cognitiveMetrics.totalDistortionCount}类扭曲`);
    }
    // 心率/呼吸不具备采集能力：不写入 evidence，避免下游把噪声当成生理指标
    if (activeModalities.includes('behavioralAct')) {
      evidence.push(`行为激活：互动=${behavioralMetrics.dailyInteractionCount}次，趋势=${behavioralMetrics.trendDirection}`);
    }
    if (activeModalities.includes('eye')) {
      const blinkSrc = eyeMetrics.blinkMethod === 'ear' ? '（EAR 降级）' : '';
      evidence.push(`眨眼${blinkSrc}：${eyeMetrics.blinkRate}次/分，低头帧占比=${Math.round(eyeMetrics.downwardGazeRatio * 100)}%（头部姿态代理）`);
    }
    if (activeModalities.includes('voiceSemantics')) {
      evidence.push(`语音语义：风险=${voiceSemanticsMetrics.riskScore}，消极词比=${Math.round(voiceSemanticsMetrics.negativeAffectRatio * 100)}%`);
    }

    // 温暖消息
    const caringMessage = CARING_MESSAGES[Math.floor(Date.now() / 30000) % CARING_MESSAGES.length];

    // 生成叙事性分析长文本
    const narrativeState: ComprehensiveEmotionState = {
      emotionProbs: normalizedProbs,
      confidence: Math.round(confidence * 100) / 100,
      dominantEmotion,
      riskLevel,
      keyboard: keyboardMetrics,
      text: textMetrics,
      voice: voiceMetrics,
      facial: facialMetrics,
      circadian: circadianMetrics,
      cognitiveDistortion: cognitiveMetrics,
      hrv: hrvMetrics,
      breathing: breathingMetrics,
      behavioralActivation: behavioralMetrics,
      eyeMovement: eyeMetrics,
      voiceSemantics: voiceSemanticsMetrics,
      fusionWeights: normalizedWeights,
      activeModalities,
      lastUpdated: Date.now(),
      unavailableModalities: UNAVAILABLE_MODALITIES,
      availabilityNote: AVAILABILITY_NOTE,
      caringMessage,
      evidence,
      narrativeAnalysis: '',
    };
    const narrativeAnalysis = generateNarrativeAnalysis(narrativeState);

    setComprehensiveState({
      ...narrativeState,
      narrativeAnalysis,
    });

    // === 个人基线更新 ===
    // 只对真实可采集的指标建立基线；心率/呼吸无采集能力，不写入基线
    personalBaseline.updateBaseline({
      typingSpeed: keyboardMetrics.typingSpeed || undefined,
      blinkRate: eyeMetrics.blinkRate || undefined,
      voiceF0: voiceMetrics.pitch?.mean || undefined,
      speechRate: voiceMetrics.rate?.speechRate || undefined,
      emotionProbs: normalizedProbs,
    });

    // === 自适应权重记录融合结果 ===
    const modalityRiskScores: Record<string, number> = {};
    if (activeModalities.includes('text')) modalityRiskScores.text = textMetrics.timestamp > 0 ? (textMetrics.emotionDistribution[2] || 0) : 0;
    if (activeModalities.includes('voice')) modalityRiskScores.voice = voiceProbs5[2];
    if (activeModalities.includes('facial')) modalityRiskScores.facial = facialProbs5[2];
    if (activeModalities.includes('keyboard')) modalityRiskScores.keyboard = keyboardMetrics.anxietyIndex / 100;
    if (activeModalities.includes('circadian')) modalityRiskScores.circadian = circadianMetrics.riskScore;
    if (activeModalities.includes('cognitive')) modalityRiskScores.cognitive = cognitiveMetrics.riskScore;
    if (activeModalities.includes('hrv')) modalityRiskScores.hrv = hrvMetrics.riskScore;
    if (activeModalities.includes('breathing')) modalityRiskScores.breathing = breathingMetrics.riskScore;
    if (activeModalities.includes('behavioralAct')) modalityRiskScores.behavioralAct = behavioralMetrics.riskScore;
    if (activeModalities.includes('eye')) modalityRiskScores.eye = eyeMetrics.riskScore;
    if (activeModalities.includes('voiceSemantics')) modalityRiskScores.voiceSemantics = voiceSemanticsMetrics.riskScore;
    adaptiveWeightsHook.recordFusion(modalityRiskScores, anxietyProb);

    // === 情绪轨迹记录 ===
    moodTrajectoryHook.recordPoint(
      anxietyProb,
      anxietyProb,
      dominantEmotion,
      activeModalities.length,
    );
  }, [keyboardMetrics, textMetrics, voiceMetrics, facialMetrics, circadianMetrics, cognitiveMetrics, hrvMetrics, breathingMetrics, behavioralMetrics, eyeMetrics, voiceSemanticsMetrics, ageGroup]);

  // 用 ref 存储最新的 performFusion，避免 setInterval 闭包问题
  const performFusionRef = useRef(performFusion);
  performFusionRef.current = performFusion;

  // 定期融合（使用 ref 确保总是调用最新的 performFusion）
  useEffect(() => {
    const intervalId = window.setInterval(() => {
      // 先更新行为激活指标（基于累积的交互数据）
      behavioralActivation.updateMetrics();
      performFusionRef.current();
    }, 3000);
    return () => clearInterval(intervalId);
  }, []); // 空依赖，只设置一次

  // 页面导航时记录模块访问（行为激活 - 探索率）
  const location = useLocation();
  useEffect(() => {
    if (location.pathname) {
      const moduleName = location.pathname.replace('/', '') || 'home';
      behavioralActivation.recordModuleVisit(moduleName);
    }
  }, [location.pathname]);

  // 摄像头自动重启：如果开关是开的但面部检测停止了，尝试重启
  useEffect(() => {
    if (!cameraEnabled) return;
    // `facePresent` 不作为重启条件：人暂时离开画面是正常情况，
    // 用"检测不到"去触发重启会让摄像头反复开关。
    if (facial.metrics.isDetecting) return; // 还在检测，不需要重启
    if (facial.metrics.frameCount === 0) return; // 还没启动过，不需要重启

    // 摄像头中途停止了，尝试重启
    console.log('[多模态] 摄像头中途停止，尝试自动重启...');
    const restartTimer = setTimeout(async () => {
      try {
        await facial.start();
        console.log('[多模态] 摄像头重启成功');
      } catch (err) {
        console.log('[多模态] 摄像头重启失败:', err);
      }
    }, 2000);

    return () => clearTimeout(restartTimer);
  }, [cameraEnabled, facial.metrics.isDetecting, facial.metrics.frameCount, facial.start]);

  // 统一摄像头开关（键盘已自动启动，只需控制 voice 和 facial）
  const toggleCamera = useCallback(async (): Promise<boolean> => {
    // 重入锁：`cameraEnabled` 要等 await 之后才更新，挡不住"await 期间又点一次"。
    // 改造前正是这个窗口让 4 次点击开出 5 路麦克风。
    if (startingMediaRef.current) return false;
    startingMediaRef.current = true;
    try {
      if (cameraEnabled) {
        console.log('[多模态] 关闭摄像头/麦克风...');
        // 等语音会话把最后一段冲刷完；否则残余音频的定稿会在拆图之后才回来，
        // 既可能丢字，也会与重新开启的下一段混在一起。
        await voice.stop();
        facial.stop();
        // ⚠️ 采集停止后，帧/音频驱动的模态必须显式重置。
        // 它们的 isMeasuring 是在 updateFromFrame()/updateFromAudioEnergy() 里置 true 的，
        // 停止采集后不会再有任何更新把它置回 false，于是会以「冻结的最后一帧读数」
        // 继续留在活跃模态里参与多模态融合（表现为关掉摄像头后仍显示「N 个模态采集中」）。
        eyeTracking.reset();    // 眨眼 / 眼动（帧驱动）
        rppg.reset();           // 心率 / HRV（帧驱动）
        breathingHook.reset();  // 呼吸（音频驱动）
        setCameraEnabled(false);
        setMediaNotice('');
        return true;
      }

      console.log('[多模态] 开启摄像头/麦克风...');
      // 分别记录结果：只给麦克风权限是常见情况，不该因为摄像头失败就丢掉麦克风。
      const [voiceOk, facialOk] = await Promise.all([
        voice.start().then(
          () => true,
          (err: unknown) => { console.error('[多模态] 麦克风启动失败:', err); return false; },
        ),
        facial.start().then(
          () => true,
          (err: unknown) => { console.error('[多模态] 摄像头启动失败:', err); return false; },
        ),
      ]);

      if (voiceOk || facialOk) {
        setCameraEnabled(true);
      }
      // 失败原因直接上屏。摄像头单独失败时给出明确提示，而不是让界面
      // 停在"看起来一切正常"的状态（改造前 start() 把异常吞了，界面显示已开启）。
      if (!voiceOk && !facialOk) {
        setMediaNotice('麦克风和摄像头都无法启动，请检查浏览器权限设置');
      } else if (!facialOk) {
        setMediaNotice('摄像头不可用（可能没有授权或被其他程序占用）· 已仅启用麦克风');
      } else if (!voiceOk) {
        setMediaNotice('麦克风不可用（可能没有授权）· 已仅启用摄像头');
      } else {
        setMediaNotice('');
      }
      console.log(`[多模态] 启动结果 voice=${voiceOk} facial=${facialOk}`);
      return voiceOk || facialOk;
    } finally {
      startingMediaRef.current = false;
    }
  }, [cameraEnabled, voice, facial, eyeTracking, rppg, breathingHook]);

  // 纯语音输入：只启动麦克风（不含摄像头），用于聊天页的"语音录入"。
  // 如果多模态开关已经启动了录音，这里直接复用，避免同时开两路麦克风。
  const startVoiceInput = useCallback(async () => {
    if (voice.metrics.isRecording) return;
    try {
      await voice.start();
      console.log('[多模态] 语音输入已启动（仅麦克风）');
    } catch (err) {
      console.error('[多模态] 语音输入启动失败:', err);
    }
  }, [voice]);

  const stopVoiceInput = useCallback(async (): Promise<string> => {
    // 内部会等最后一段定稿（撤销端点检测 + 离线复识别）落地再把文本交出来
    const finalText = await voice.stop();
    console.log('[多模态] 语音输入已停止，最后一段:', finalText || '(无)');
    return finalText;
  }, [voice]);

  // 全局文本分析入口
  const analyzeTextGlobal = useCallback((text: string) => {
    textAnalysis.analyzeText(text);
    (window as any).__recentUserText = text;
  }, [textAnalysis]);

  // 焦虑超标时自动获取任务
  const anxietyIndex = comprehensiveState.emotionProbs[2] * 100;

  const fetchRealityTask = useCallback(async () => {
    try {
      const res = await profileApi.generateRealityTask({
        anxietyLevel: anxietyIndex,
        context: comprehensiveState.evidence.join('；'),
      }) as any;
      setRealityTask(res.data);
      setShowTaskModal(true);
    } catch {}
  }, [anxietyIndex, comprehensiveState.evidence]);

  useEffect(() => {
    if (anxietyIndex > 60 && !hasTriggeredRef.current && cameraEnabled) {
      hasTriggeredRef.current = true;
      fetchRealityTask();
    }
    if (anxietyIndex <= 30) hasTriggeredRef.current = false;
  }, [anxietyIndex, cameraEnabled, fetchRealityTask]);

  const reportToServer = useCallback(async (context?: string) => {
    const now = Date.now();
    if (now - lastReportRef.current < 10000) return;
    lastReportRef.current = now;
    try {
      await fetch('/api/profile/anxiety-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
        },
        body: JSON.stringify({
          anxietyIndex,
          level: comprehensiveState.riskLevel,
          context: context || comprehensiveState.evidence.join('；'),
          // 完整多模态数据
          emotionProbs: comprehensiveState.emotionProbs,
          dominantEmotion: comprehensiveState.dominantEmotion,
          confidence: comprehensiveState.confidence,
          fusionWeights: comprehensiveState.fusionWeights,
          evidence: comprehensiveState.evidence,
          multimodal: {
            text: textAnalysis.metrics.timestamp > 0,
            voice: voice.metrics.isRecording,
            facial: facial.metrics.isDetecting,
            keyboard: keyboard.metrics.isCollecting,
            circadian: circadian.metrics.isActive,
            cognitive: cognitiveDistortion.metrics.riskScore > 0,
            hrv: rppg.metrics.isMeasuring,
            breathing: breathingHook.metrics.isMeasuring,
            behavioralAct: behavioralActivation.metrics.isActive,
            eye: eyeTracking.metrics.isMeasuring,
            voiceSemantics: voiceSemantics.metrics.timestamp > 0,
          },
          keyboard: keyboard.metrics,
          text: textAnalysis.metrics,
          voice: voice.metrics,
          facial: facial.metrics,
          circadian: circadian.metrics,
          cognitiveDistortion: cognitiveDistortion.metrics,
          hrv: rppg.metrics,
          breathing: breathingHook.metrics,
          behavioralActivation: behavioralActivation.metrics,
          eyeMovement: eyeTracking.metrics,
          voiceSemantics: voiceSemantics.metrics,
        }),
      });
    } catch {}
  }, [anxietyIndex, comprehensiveState, textAnalysis.metrics, voice.metrics, facial.metrics, keyboard.metrics, circadian.metrics, cognitiveDistortion.metrics, rppg.metrics, breathingHook.metrics, behavioralActivation.metrics, eyeTracking.metrics, voiceSemantics.metrics]);

  // 定期上报
  useEffect(() => {
    if (!cameraEnabled) return;
    const interval = setInterval(() => reportToServer(), 15000);
    return () => clearInterval(interval);
  }, [cameraEnabled, reportToServer]);

  const reset = useCallback(() => {
    keyboard.reset();
    textAnalysis.reset();
    voice.reset();
    facial.reset();
    circadian.reset();
    cognitiveDistortion.reset();
    rppg.reset();
    breathingHook.reset();
    behavioralActivation.reset();
    eyeTracking.reset();
    voiceSemantics.reset();
    interventionFeedback.reset();
    personalBaseline.reset();
    adaptiveWeightsHook.reset();
    moodTrajectoryHook.reset();
    setCameraEnabled(false);
    setComprehensiveState({ ...defaultComprehensive });
    hasTriggeredRef.current = false;
    setRealityTask(null);
  }, [keyboard, textAnalysis, voice, facial, circadian, cognitiveDistortion, rppg, breathingHook, behavioralActivation, eyeTracking, voiceSemantics]);

  const value: AnxietyContextType = {
    comprehensiveState,
    cameraEnabled,
    modalityStatus,
    updateCounter,
    keyboardMetrics,
    textMetrics,
    voiceMetrics,
    facialMetrics,
    circadianMetrics,
    cognitiveMetrics,
    hrvMetrics,
    breathingMetrics,
    behavioralMetrics,
    eyeMetrics,
    voiceSemanticsMetrics,
    mediaNotice,
    toggleCamera,
    analyzeText: analyzeTextGlobal,
    voiceInputActive: voiceMetrics.isRecording,
    startVoiceInput,
    stopVoiceInput,
    voiceFinalSeq,
    voiceFinalText: voiceFinalRef.current.text,
    reportToServer,
    realityTask,
    showTaskModal,
    setShowTaskModal,
    fetchRealityTask,
    reset,
    baseline: personalBaseline.baseline,
    baselineDeviation: personalBaseline.getBaselineDeviation(),
    getZScoreForValue: personalBaseline.getZScoreForValue,
    adaptiveWeights: adaptiveWeightsHook.state,
    moodTrajectory: moodTrajectoryHook.trajectory,
    getWeeklyReport: moodTrajectoryHook.getWeeklyReport,
    interventionHistory: interventionFeedback.getHistory(),
    getRecommendedIntervention: interventionFeedback.getRecommendedType,
  };

  return (
    <AnxietyContext.Provider value={value}>
      {children}
    </AnxietyContext.Provider>
  );
}

export function useAnxiety() {
  const ctx = useContext(AnxietyContext);
  if (!ctx) throw new Error('useAnxiety must be used within AnxietyProvider');
  return ctx;
}
