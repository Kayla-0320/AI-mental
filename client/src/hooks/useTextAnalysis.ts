/**
 * 文本语义深度分析 Hook
 *
 * 理论依据：
 * - NRC Emotion Lexicon (Mohammad & Turney, 2013) — 八维情绪词汇映射
 * - CBT 认知扭曲理论 (Beck, 1979) — 灾难化/非黑即白/过度概括等
 * - LIWC (Pennebaker et al., 2015) — 语言特征与心理健康
 * - 危机评估：Columbia Suicide Severity Rating Scale (C-SSRS) 关键词
 *
 * 分析维度：
 * - 八维情绪分布 (快乐/悲伤/焦虑/愤怒/恐惧/厌恶/惊讶/中性)
 * - 危机信号检测 (自伤/自杀/绝望等关键词)
 * - 认知扭曲标记 (灾难化/非黑即白/过度概括/自我归咎/无望感)
 * - 语言特征 (情感效价/情绪强度/第一人称密度/否定词/绝对化用词)
 * - 书写行为 (消息长度/书写压力/话题连贯性)
 *
 * 采集范围：用户在平台上输入的所有文本
 */
import { useState, useCallback, useRef } from 'react';
import type { TextAnalysis } from '../types/multimodal.types';
import { defaultText } from '../types/multimodal.types';

// ===== 情绪词典 (基于 NRC Emotion Lexicon 中文适配) =====
const EMOTION_LEXICON: Record<string, number[]> = {
  // [快乐, 悲伤, 焦虑, 愤怒, 恐惧, 厌恶, 惊讶, 中性]
  '开心': [1,0,0,0,0,0,0,0], '高兴': [1,0,0,0,0,0,0,0], '快乐': [1,0,0,0,0,0,0,0],
  '幸福': [1,0,0,0,0,0,0,0], '棒': [1,0,0,0,0,0,0,0], '好': [0.5,0,0,0,0,0,0,0.5],
  '喜欢': [1,0,0,0,0,0,0,0], '爱': [1,0,0,0,0,0,0,0], '感恩': [1,0,0,0,0,0,0,0],
  '难过': [0,1,0,0,0,0,0,0], '伤心': [0,1,0,0,0,0,0,0], '哭': [0,1,0,0,0,0,0,0],
  '悲伤': [0,1,0,0,0,0,0,0], '痛苦': [0,1,0.3,0,0,0,0,0], '失落': [0,1,0.2,0,0,0,0,0],
  '孤独': [0,1,0.2,0,0,0,0,0], '寂寞': [0,1,0.2,0,0,0,0,0], '空虚': [0,1,0.2,0,0,0,0,0],
  '焦虑': [0,0,1,0,0,0,0,0], '紧张': [0,0,1,0,0,0,0,0], '担心': [0,0,1,0,0,0,0,0],
  '害怕': [0,0,0.5,0,1,0,0,0], '恐惧': [0,0,0.3,0,1,0,0,0], '不安': [0,0,1,0,0.5,0,0,0],
  '压力': [0,0,1,0,0,0,0,0], '烦躁': [0,0,0.5,0.5,0,0,0,0], '失眠': [0,0,0.5,0,0,0,0,0],
  '生气': [0,0,0,1,0,0,0,0], '愤怒': [0,0,0,1,0,0,0,0], '气死': [0,0,0,1,0,0,0,0],
  '烦': [0,0,0.3,0.5,0,0,0,0], '讨厌': [0,0,0,0.5,0,0.5,0,0], '恨': [0,0,0,1,0,0,0,0],
  '吓': [0,0,0.3,0,1,0,0,0], '恐怖': [0,0,0,0,1,0,0,0], '惊': [0,0,0,0,0.5,0,1,0],
  '恶心': [0,0,0,0,0,1,0,0], '呕': [0,0,0,0,0,1,0,0],
  '意外': [0,0,0,0,0,0,1,0], '没想到': [0,0,0,0,0,0,1,0],
};

// ===== 危机关键词 (C-SSRS  adapted) =====
const CRISIS_HIGH = ['自杀', '自残', '自伤', '不想活', '想死', '去死', '跳楼', '割腕', '结束生命', '活着没意义', '不如死了'];
const CRISIS_MEDIUM = ['活着没意思', '没有意义', '没有人会在意', '消失了也好', '世界没有我', '拖累', '负担'];
const CRISIS_LOW = ['好累', '好烦', '受不了', '撑不下去', '快崩溃', '要疯了'];

// ===== 认知扭曲标记 (CBT) =====
const COGNITIVE_PATTERNS = {
  catastrophizing: ['完了', '完蛋', '全毁了', '一切都完了', '最坏的', '太可怕了', '天塌了'],
  blackAndWhite: ['总是', '从不', '永远不', '绝对', '一定', '完全', '根本', '所有', '每次'],
  overgeneralization: ['每次都', '从来不', '所有人', '没有人', '到处', '总是这样'],
  selfBlame: ['都是我的错', '怪我', '我不配', '我不行', '我太差', '没用的', '废物', '垃圾'],
  hopelessness: ['没有希望', '不会好的', '没救了', '永远', '不可能', '做不到', '没办法'],
};

// ===== 否定词 & 绝对化用词 =====
const NEGATION_WORDS = ['不', '没', '别', '无', '非', '未', '莫', '勿', '难以', '无法', '不能', '不会'];
const ABSOLUTE_WORDS = ['总是', '从不', '永远', '绝对', '一定', '所有', '全部', '完全', '根本', '每次', '所有'];
const FIRST_PERSON = ['我', '我的', '自己', '本人', '俺', '咱'];

export function useTextAnalysis() {
  const [metrics, setMetrics] = useState<TextAnalysis>({ ...defaultText });
  const allTextsRef = useRef<string[]>([]);
  const messageLengthsRef = useRef<number[]>([]);

  const analyzeText = useCallback((text: string) => {
    if (!text.trim()) return;

    console.log('[文本分析] 开始分析, 长度:', text.length);
    const textLower = text.toLowerCase();
    allTextsRef.current.push(text);
    messageLengthsRef.current.push(text.length);

    // === 1. 八维情绪分布 ===
    const emotionScores = [0, 0, 0, 0, 0, 0, 0, 0.5]; // 8维，中性有基础分
    let matchCount = 0;
    for (const [word, scores] of Object.entries(EMOTION_LEXICON)) {
      if (textLower.includes(word)) {
        scores.forEach((s, i) => { emotionScores[i] += s; });
        matchCount++;
      }
    }
    // 归一化
    const total = emotionScores.reduce((a, b) => a + b, 0) || 1;
    const emotionDistribution = emotionScores.map(s => Math.round((s / total) * 100) / 100);
    const dominantIdx = emotionDistribution.indexOf(Math.max(...emotionDistribution));
    const emotionLabels = ['快乐', '悲伤', '焦虑', '愤怒', '恐惧', '厌恶', '惊讶', '中性'];
    const dominantEmotion = emotionLabels[dominantIdx];
    const dominantConfidence = emotionDistribution[dominantIdx];

    // === 2. 危机信号检测 ===
    const crisisMarkers: string[] = [];
    let crisisLevel: 'none' | 'low' | 'medium' | 'high' = 'none';
    for (const kw of CRISIS_HIGH) {
      if (textLower.includes(kw)) { crisisMarkers.push(kw); crisisLevel = 'high'; }
    }
    if (crisisLevel !== 'high') {
      for (const kw of CRISIS_MEDIUM) {
        if (textLower.includes(kw)) { crisisMarkers.push(kw); crisisLevel = 'medium'; }
      }
    }
    if (crisisLevel === 'none') {
      for (const kw of CRISIS_LOW) {
        if (textLower.includes(kw)) { crisisMarkers.push(kw); crisisLevel = 'low'; }
      }
    }

    // === 3. 认知扭曲标记 ===
    const countMatches = (patterns: string[]) => {
      const count = patterns.filter(p => textLower.includes(p)).length;
      return Math.min(1, count / 3); // 3个以上标记 = 满分
    };
    const cognitiveMarkers = {
      catastrophizing: countMatches(COGNITIVE_PATTERNS.catastrophizing),
      blackAndWhite: countMatches(COGNITIVE_PATTERNS.blackAndWhite),
      overgeneralization: countMatches(COGNITIVE_PATTERNS.overgeneralization),
      selfBlame: countMatches(COGNITIVE_PATTERNS.selfBlame),
      hopelessness: countMatches(COGNITIVE_PATTERNS.hopelessness),
    };

    // === 4. 语言特征 ===
    const words = textLower.split(/\s+/).filter(Boolean);
    const wordCount = Math.max(words.length, 1);

    // 情感效价 (正面词 - 负面词)
    const positiveScore = emotionScores[0]; // 快乐
    const negativeScore = emotionScores[1] + emotionScores[2] + emotionScores[3] + emotionScores[4]; // 悲+焦+怒+恐
    const sentiment = Math.max(-1, Math.min(1, (positiveScore - negativeScore) / (total || 1)));

    // 情绪强度
    const emotionalIntensity = Math.min(1, matchCount / Math.max(text.length / 10, 1));

    // 第一人称密度
    const firstPersonCount = FIRST_PERSON.filter(w => textLower.includes(w)).length;
    const firstPersonRate = Math.min(1, firstPersonCount / wordCount);

    // 否定词密度
    const negationCount = NEGATION_WORDS.filter(w => textLower.includes(w)).length;
    const negationRate = Math.min(1, negationCount / wordCount);

    // 绝对化用词
    const absoluteWords = ABSOLUTE_WORDS.filter(w => textLower.includes(w)).length;

    // 疑问句比例
    const questionMarks = (text.match(/[？?]/g) || []).length;
    const sentences = Math.max(1, (text.match(/[。！？!?]/g) || []).length);
    const questionRate = questionMarks / sentences;

    // === 5. 书写行为 ===
    const avgMessageLength = messageLengthsRef.current.reduce((a, b) => a + b, 0) / messageLengthsRef.current.length;
    const writingPressure = Math.min(1, (avgMessageLength / 200) * 0.5 + emotionalIntensity * 0.5);

    // 话题连贯性 (最近5条消息词汇重叠度)
    let topicCoherence = 0.5;
    const recentTexts = allTextsRef.current.slice(-5);
    if (recentTexts.length >= 2) {
      const allWords = recentTexts.map(t => new Set(t.split(/\s+/)));
      let overlap = 0;
      for (let i = 1; i < allWords.length; i++) {
        const intersection = [...allWords[i]].filter(w => allWords[i-1].has(w)).length;
        overlap += intersection / Math.max(allWords[i].size, 1);
      }
      topicCoherence = Math.min(1, overlap / (allWords.length - 1));
    }

    console.log('[文本分析] 主导情绪:', dominantEmotion, '危机等级:', crisisLevel, '字符数:', total);
    setMetrics({
      emotionDistribution, dominantEmotion, dominantConfidence,
      crisisMarkers, crisisLevel, cognitiveMarkers,
      sentiment: Math.round(sentiment * 100) / 100,
      emotionalIntensity: Math.round(emotionalIntensity * 100) / 100,
      firstPersonRate: Math.round(firstPersonRate * 100) / 100,
      negationRate: Math.round(negationRate * 100) / 100,
      absoluteWords, questionRate: Math.round(questionRate * 100) / 100,
      avgMessageLength: Math.round(avgMessageLength),
      writingPressure: Math.round(writingPressure * 100) / 100,
      topicCoherence: Math.round(topicCoherence * 100) / 100,
      totalCharsAnalyzed: allTextsRef.current.reduce((sum, t) => sum + t.length, 0),
      lastAnalyzedText: text.slice(0, 100),
      timestamp: Date.now(),
    });
  }, []);

  const reset = useCallback(() => {
    allTextsRef.current = [];
    messageLengthsRef.current = [];
    setMetrics({ ...defaultText });
  }, []);

  return { metrics, analyzeText, reset };
}
