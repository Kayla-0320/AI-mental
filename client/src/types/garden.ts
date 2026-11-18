/**
 * 「心灵花园」首页改版 —— 共享类型。
 *
 * 这些结构**直接对应后端已返回的 JSON**（不是前端臆造的模型）：
 *   · `GrowthOverview` ← GET /api/extra/growth（server/src/services/growth.service.ts 的 GrowthOverview）
 *   · `LearningContent` ← GET /api/learning/recommendations|contents（prisma.learningContent）
 *
 * 改动这些字段前先对照服务端接口，避免前端按不存在的字段渲染导致空态误判。
 */

/** 成长指标卡（与 growth.service.ts 的 MetricCard 一一对应） */
export interface GrowthMetricCard {
  key: string;
  group: 'persistence' | 'mood' | 'investment' | 'exploration';
  label: string;
  /** 已格式化的展示值；无数据为 null */
  value: string | null;
  display: string;
  unit: string;
  /** ok=有数据 / empty=未采集 / building=样本不足 */
  status: 'ok' | 'empty' | 'building';
  hint: string;
  source: string;
  current: number | null;
  target: number | null;
  percent: number | null;
}

/** 成长指数的一个维度 */
export interface GrowthDimension {
  key: string;
  label: string;
  /** null = 该维度样本不足，不参与加权 */
  score: number | null;
  weight: number;
  hint: string;
}

export interface GrowthCounters {
  totalCheckIns: number;
  streakDays: number;
  longestStreak: number;
  checkInRate7: number;
  checkInRate30: number;
  activeDays30: number;
  avgMood7: number | null;
  avgMood14: number | null;
  avgMoodPrev7: number | null;
  moodStability7: number | null;
  moodDelta: number | null;
  healingSessions: number;
  healingMinutes: number;
  assessments: number;
  lastAssessment: { type: string; title: string; totalScore: number | null; level: string | null; createdAt: string } | null;
}

/** 近 N 天心情序列中的一个点（computeMoodSeries） */
export interface MoodTrendPoint {
  dayKey: string;
  /** MM-DD */
  label: string;
  /** 1-10 均分；当天没打卡为 null */
  score: number | null;
  mood: string | null;
}

export interface GrowthBadge {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: string;
  unlocked: boolean;
  unlockedAt: string | null;
  progress: { current: number; target: number; percent: number };
}

export interface GrowthBadgeBoard {
  total: number;
  unlocked: number;
  percent: number;
  badges: GrowthBadge[];
}

export interface GrowthNextAction {
  key: string;
  label: string;
  desc: string;
  path: string;
  icon: string;
}

/** GET /api/extra/growth 的 data */
export interface GrowthOverview {
  index: { score: number; grade: string; dimensions: GrowthDimension[] };
  level: {
    level: number;
    name: string;
    icon: string;
    xp: number;
    currentLevelXp: number;
    nextLevelXp: number;
    progress: number;
    isMax: boolean;
  };
  summary: string;
  counters: GrowthCounters;
  metrics: GrowthMetricCard[];
  keywords: string[];
  trend: MoodTrendPoint[];
  heatmap: Array<{ dayKey: string; level: number; score: number | null }>;
  badges: GrowthBadgeBoard;
  newlyUnlocked: Array<{ id: string; name: string; icon: string }>;
  nextActions: GrowthNextAction[];
}

/** 学习/推荐内容（learningContent 表） */
export interface LearningContent {
  id: string;
  title: string;
  /** article | audio | video | course */
  type: string;
  /** emotion | stress | sleep | self_growth | relationship */
  category: string;
  level: string;
  summary: string | null;
  content: string | null;
  coverImage: string | null;
  /** 分钟 */
  duration: number | null;
  tags: string[] | null;
  viewCount: number;
}
