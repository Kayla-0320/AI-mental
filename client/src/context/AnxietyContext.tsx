/**
 * 多模态情感感知上下文
 *
 * 整合四大感知模态：
 * 1. 键盘动力学 — 全局键盘事件采集
 * 2. 文本语义分析 — 全局文本输入采集（所有页面）
 * 3. 声音声学分析 — 摄像头开启时同步采集
 * 4. 面部微表情分析 — 摄像头开启时同步采集
 *
 * 融合策略：基于置信度的动态加权 (Dynamic Confidence Weighting)
 * 参考：fusion.py 中的 fuse_dynamic_weighting
 */
import React, { createContext, useContext, useState, useCallback, useRef, useEffect, ReactNode } from 'react';
import { flushSync } from 'react-dom';
import { useKeyboardDynamics } from '../hooks/useKeyboardDynamics';
import { useTextAnalysis } from '../hooks/useTextAnalysis';
import { useVoiceAnalysis } from '../hooks/useVoiceAnalysis';
import { useFacialAnalysis } from '../hooks/useFacialAnalysis';
import type { ComprehensiveEmotionState, KeyboardDynamics, TextAnalysis, VoiceAnalysis, FacialAnalysis } from '../types/multimodal.types';
import { defaultKeyboard, defaultText, defaultVoice, defaultFacial } from '../types/multimodal.types';
import { profileApi } from '../services';

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

// ===== 叙事性多模态分析生成器 =====
// 将四模态数据转化为温暖的心理咨询师式观察报告
function generateNarrativeAnalysis(state: ComprehensiveEmotionState): string {
  const { keyboard, text, voice, facial, dominantEmotion, emotionProbs } = state;
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
  if (facial.isDetecting && facial.frameCount >= 2) {
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
      faceParts.push(`我们还捕捉到了${facial.microExpressions.count}次微表情——那些持续时间不到半秒的、你自己可能都没意识到的表情变化`);
    }
    if (facial.expressionIntensity > 0.5) {
      faceParts.push('表情强度比较高，情绪表达很充分');
    }
    if (faceParts.length > 0) {
      paragraphs.push(`从你的面部表情来看，${faceParts.join('；')}。`);
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

  return paragraphs.join('\n\n');
}

interface AnxietyContextType {
  comprehensiveState: ComprehensiveEmotionState;
  cameraEnabled: boolean; // 感知开关状态（独立状态，不依赖 activeModalities）
  modalityStatus: {
    keyboardStarted: boolean;
    voiceStarted: boolean;
    facialStarted: boolean;
    textListenerRegistered: boolean;
  };
  updateCounter: number; // 用于验证状态更新是否工作
  keyboardMetrics: KeyboardDynamics;
  textMetrics: TextAnalysis;
  voiceMetrics: VoiceAnalysis;
  facialMetrics: FacialAnalysis;
  toggleCamera: () => void;
  analyzeText: (text: string) => void; // 全局文本分析入口
  reportToServer: (context?: string) => Promise<void>;
  realityTask: any;
  showTaskModal: boolean;
  setShowTaskModal: (v: boolean) => void;
  fetchRealityTask: () => Promise<void>;
  reset: () => void;
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
  fusionWeights: { text: 0.4, voice: 0.2, facial: 0.2, keyboard: 0.2 },
  activeModalities: [],
  lastUpdated: 0,
  caringMessage: CARING_MESSAGES[0],
  evidence: [],
  narrativeAnalysis: '',
};

const AnxietyContext = createContext<AnxietyContextType | null>(null);

export function AnxietyProvider({ children }: { children: ReactNode }) {
  const keyboard = useKeyboardDynamics();
  const textAnalysis = useTextAnalysis();
  const voice = useVoiceAnalysis();
  const facial = useFacialAnalysis();

  // 用 ref 稳定 analyzeText 引用，避免事件监听器反复重建
  const analyzeTextRef = useRef(textAnalysis.analyzeText);
  analyzeTextRef.current = textAnalysis.analyzeText;

  // 直接管理指标状态（不依赖 hook 的内部状态）
  const [keyboardMetrics, setKeyboardMetrics] = useState(keyboard.metrics);
  const [textMetrics, setTextMetrics] = useState(textAnalysis.metrics);
  const [voiceMetrics, setVoiceMetrics] = useState(voice.metrics);
  const [facialMetrics, setFacialMetrics] = useState(facial.metrics);

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

  // 语音识别文本 → 送入文本分析
  useEffect(() => {
    if (voiceMetrics.speechText && voiceMetrics.speechText.trim().length > 0) {
      console.log('[多模态] 语音识别文本:', voiceMetrics.speechText);
      // 将语音识别的文本送入文本分析
      analyzeTextRef.current(voiceMetrics.speechText);
    }
  }, [voiceMetrics.speechText]);

  const [comprehensiveState, setComprehensiveState] = useState<ComprehensiveEmotionState>({ ...defaultComprehensive });
  const [cameraEnabled, setCameraEnabled] = useState(false);
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
    const handleStart = () => {
      console.log('[多模态] 收到手动启动事件');
      
      // 直接更新本地指标状态（不依赖 hook 的内部状态）
      setKeyboardMetrics(prev => ({ ...prev, isCollecting: true }));
      setVoiceMetrics(prev => ({ ...prev, isRecording: true }));
      setFacialMetrics(prev => ({ ...prev, isDetecting: true }));
      setTextMetrics(prev => ({ ...prev, timestamp: Date.now() }));
      
      setModalityStatus({
        keyboardStarted: true,
        voiceStarted: true,
        facialStarted: true,
        textListenerRegistered: true,
      });
      
      // 调用 hook 的 start()（用于启动实际的采集逻辑）
      try {
        keyboard.start();
        console.log('[多模态] 键盘启动完成');
      } catch (err) {
        console.error('[多模态] 键盘启动失败:', err);
      }
      try {
        voice.start();
        console.log('[多模态] 声音启动完成');
      } catch (err) {
        console.error('[多模态] 声音启动失败:', err);
      }
      try {
        facial.start();
        console.log('[多模态] 面部启动完成');
      } catch (err) {
        console.error('[多模态] 面部启动失败:', err);
      }
      
      setCameraEnabled(true);
      setComprehensiveState(prev => ({ ...prev, lastUpdated: Date.now() }));
      setUpdateCounter(prev => prev + 1); // 递增计数器，验证状态更新
      console.log('[多模态] 所有模态已启动，本地状态已更新，updateCounter:', updateCounter + 1);
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

  // 四模态融合 (每 3 秒执行一次)
  const performFusion = useCallback(() => {
    const activeModalities: string[] = [];
    const weights: Record<string, number> = { text: 0, voice: 0, facial: 0, keyboard: 0 };
  
    // 确定活跃模态和置信度（简化条件：只要启动就激活）
    if (textMetrics.timestamp > 0) {
      activeModalities.push('text');
      weights.text = Math.min(0.5, textMetrics.dominantConfidence * 0.5 + 0.2);
    }
    if (voiceMetrics.isRecording) {
      activeModalities.push('voice');
      weights.voice = Math.min(0.4, 0.3);
    }
    if (facialMetrics.isDetecting) {
      activeModalities.push('facial');
      weights.facial = Math.min(0.4, Math.max(0.15, facialMetrics.confidence * 0.3));
    }
    if (keyboardMetrics.isCollecting) {
      activeModalities.push('keyboard');
      weights.keyboard = 0.2;
    }
  
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

    // 归一化权重
    const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0);
    if (totalWeight === 0) return;
    const normalizedWeights = {
      text: weights.text / totalWeight,
      voice: weights.voice / totalWeight,
      facial: weights.facial / totalWeight,
      keyboard: weights.keyboard / totalWeight,
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

    // 加权融合
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

    // 归一化
    const probSum = fusedProbs.reduce((a, b) => a + b, 0) || 1;
    const normalizedProbs = fusedProbs.map(p => Math.round((p / probSum) * 100) / 100);

    // 主导情绪
    const labels = ['快乐', '悲伤', '焦虑', '愤怒', '中性'];
    const dominantIdx = normalizedProbs.indexOf(Math.max(...normalizedProbs));
    const dominantEmotion = labels[dominantIdx];
    const confidence = normalizedProbs[dominantIdx];

    // 风险等级
    const anxietyProb = normalizedProbs[2];
    const riskLevel = anxietyProb > 0.7 ? 'crisis' : anxietyProb > 0.5 ? 'high' : anxietyProb > 0.3 ? 'medium' : 'low';

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
      fusionWeights: normalizedWeights,
      activeModalities,
      lastUpdated: Date.now(),
      caringMessage,
      evidence,
      narrativeAnalysis: '',
    };
    const narrativeAnalysis = generateNarrativeAnalysis(narrativeState);

    setComprehensiveState({
      ...narrativeState,
      narrativeAnalysis,
    });
  }, [keyboardMetrics, textMetrics, voiceMetrics, facialMetrics]);

  // 用 ref 存储最新的 performFusion，避免 setInterval 闭包问题
  const performFusionRef = useRef(performFusion);
  performFusionRef.current = performFusion;

  // 定期融合（使用 ref 确保总是调用最新的 performFusion）
  useEffect(() => {
    const intervalId = window.setInterval(() => {
      performFusionRef.current();
    }, 3000);
    return () => clearInterval(intervalId);
  }, []); // 空依赖，只设置一次

  // 摄像头自动重启：如果开关是开的但面部检测停止了，尝试重启
  useEffect(() => {
    if (!cameraEnabled) return;
    if (facial.metrics.isDetecting) return; // 还在检测，不需要重启
    if (facial.metrics.frameCount === 0) return; // 还没启动过，不需要重启

    // 摄像头中途停止了，尝试重启
    console.log('[多模态] 摄像头中途停止，尝试自动重启...');
    const restartTimer = setTimeout(async () => {
      try {
        await facial.start();
        console.log('[多模态] 摄像头重启成功');
      } catch {
        console.log('[多模态] 摄像头重启失败');
      }
    }, 2000);

    return () => clearTimeout(restartTimer);
  }, [cameraEnabled, facial.metrics.isDetecting, facial.metrics.frameCount, facial.start]);

  // 统一摄像头开关（键盘已自动启动，只需控制 voice 和 facial）
  const toggleCamera = useCallback(async () => {
    console.log('[多模态] toggleCamera 被调用，当前 cameraEnabled:', cameraEnabled);
    if (cameraEnabled) {
      console.log('[多模态] 关闭摄像头/麦克风...');
      voice.stop();
      facial.stop();
      setCameraEnabled(false);
    } else {
      console.log('[多模态] 开启摄像头/麦克风...');
      try {
        await Promise.all([voice.start(), facial.start()]);
        console.log('[多模态] 摄像头/麦克风启动成功');
        setCameraEnabled(true);
      } catch (err) {
        console.error('[多模态] 摄像头/麦克风启动失败:', err);
      }
    }
  }, [cameraEnabled, voice, facial]);

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
      await fetch('/api/profile/profile/anxiety-report', {
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
          },
          keyboard: keyboard.metrics,
          text: textAnalysis.metrics,
          voice: voice.metrics,
          facial: facial.metrics,
        }),
      });
    } catch {}
  }, [anxietyIndex, comprehensiveState, textAnalysis.metrics, voice.metrics, facial.metrics, keyboard.metrics]);

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
    setCameraEnabled(false);
    setComprehensiveState({ ...defaultComprehensive });
    hasTriggeredRef.current = false;
    setRealityTask(null);
  }, [keyboard, textAnalysis, voice, facial]);

  const value: AnxietyContextType = {
    comprehensiveState,
    cameraEnabled,
    modalityStatus,
    updateCounter,
    keyboardMetrics,
    textMetrics,
    voiceMetrics,
    facialMetrics,
    toggleCamera,
    analyzeText: analyzeTextGlobal,
    reportToServer,
    realityTask,
    showTaskModal,
    setShowTaskModal,
    fetchRealityTask,
    reset,
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
