/**
 * 面部**实时状态摘要** —— 本机表情读数 → 一个可上行、可展示、可留痕的小对象
 *
 * ## 它存在的理由
 *
 * 表情模型的原始读数是"5~10Hz 的一帧一帧结果"（`FaceEmotionReading`），体积与频率
 * 都不适合原样上行；而下游（Node → 算法层 prompt / 画像留痕 / 画像页卡片）需要的
 * 是**同一份口径**的一个瞬时快照。若三处各写一遍转换，口径必然漂移，而且漂移是静默的
 * （与 `_build_llm_messages` 抽出共用逻辑的理由完全一致）。
 *
 * ## 三条硬约束（都来自本仓库既有约定，不是本文件发明的）
 *
 * 1. **没有真实读数就不产出**：没开采集、画面里没人脸、模型没就绪、读数过期 —— 一律返回
 *    `null`。绝不用旧值、默认值、中性值"凑一个"出来（同 `IncongruenceAlert` 删掉伪造声学
 *    分支的理由：把凭空造的数据呈现成测量结果，是本仓库明确禁止的）。
 * 2. **只描述"此刻镜头里"，不描述"这个人怎么了"**：本对象里没有任何心理状态、诊断、
 *    风险字段。它进 prompt 时只是**低权重的内部参考**，绝不参与 `riskLevel` / 融合权重 /
 *    量表分（见 `useFaceEmotion` 的"只上报、不进风险"约定与 AGENTS.md 的安全红线）。
 * 3. **必须带"这条读数多新"**：`ageMs` 与 `sampledAt` 一起上行。人脸离开画面后旧读数会过期
 *    （`useFaceEmotion` 的 `ttlMs`），下游据此判断"还新不新"，而不是把几分钟前的表情
 *    当成当下。
 *
 * ## 与 `emotionMapping` 的关系
 *
 * `FacialAnalysis` 里有两条**独立证据**，本文件只做打包、不做合并：
 *   - `faceEmotion`：本机 CNN 三分类（happy/neutral/sad）→ 写进 `probabilities`；
 *   - `emotionMapping` / `expressionIntensity`：MediaPipe AU 启发式 → 只带 `expressionIntensity`
 *     作为"表情幅度"，因为它与 CNN 的类别概率不是同一件事，混起来会互相污染。
 */
import type { FacialAnalysis, FaceEmotionLabel } from '../types/multimodal.types';
// 从**纯节律模块**取常量，而不是从 `useFaceEmotion`：
// 后者会连带引入 onnxruntime-web，而本文件被 chatStream / 画像页等只需"打包口径"的
// 地方引用，不该把推理运行时拖进这些路径（`faceEmotionPacing` 无任何重依赖）。
import { FACE_EMOTION_READING_TTL_FLOOR_MS } from '../hooks/faceEmotionPacing';

/**
 * 摘要的契约版本。上行到算法层后，算法层只认这个版本 —— 这样将来改口径时，
 * 旧客户端发来的字段不会被新代码静默误读（宁可整段不注入）。
 */
export const FACIAL_LIVE_STATE_VERSION = 1;

/**
 * 摘要允许的最大年龄（ms）兜底。
 *
 * 正常情况下用读数自带的 `ttlMs`（由实测推理耗时推导，见 `faceEmotionReadingTtlMs`）。
 * 这个常量只在读数没带 `ttlMs` 时兜底（例如将来换了模型实现），
 * 取 `useFaceEmotion` 导出的下限，保证两处不会各写一个数。
 */
export const FACIAL_LIVE_STATE_MAX_AGE_MS = FACE_EMOTION_READING_TTL_FLOOR_MS;

/** 上行的面部实时状态快照（纯数据，可 `JSON.stringify`） */
export interface FacialLiveState {
  /** 契约版本，恒为 `FACIAL_LIVE_STATE_VERSION` */
  version: number;
  /** 读数自身的采样时刻（`Date.now()`，epoch ms） */
  sampledAt: number;
  /** 打包时这条读数已经存在了多久（ms） */
  ageMs: number;
  /** 这一拍画面里有没有人脸 —— 恒为 true（没人脸时不产出摘要） */
  present: boolean;
  /** 三分类概率（**未平滑**的模型输出，四舍五入到 3 位小数） */
  probabilities: { happy: number; neutral: number; sad: number };
  /** 概率最大的类别 */
  dominant: FaceEmotionLabel;
  /** 最大类概率 0–1 */
  confidence: number;
  /** 折叠不进三类的概率总量（外部多类模型才有值；三分类模型为 0） */
  offMapProbability: number;
  /** 实测采样率（Hz）；样本不足时为 0 */
  sampleHz: number;
  /** 表情幅度 0–1（AU 启发式通道，与 CNN 类别是两条独立证据） */
  expressionIntensity: number;
  /** 模型执行后端（webgpu / wasm-proxy / wasm），排查用；未知为 null */
  provider: string | null;
}

/** 四舍五入到 3 位，同时把 NaN / 越界值夹回 [0,1] */
function clamp01(v: unknown): number {
  const n = typeof v === 'number' && Number.isFinite(v) ? v : 0;
  return Math.min(1, Math.max(0, Math.round(n * 1000) / 1000));
}

/**
 * 从 `FacialAnalysis` 构造实时状态摘要。
 *
 * @param facial 面部指标（`useAnxiety().facialMetrics` 或 `useCallVision` 的同一份）
 * @param now 当前时刻（默认 `Date.now()`）；显式传入只为可测
 * @returns 摘要；**任何一条前置条件不满足都返回 `null`**（调用方必须处理 null，
 *          不许用中性值兜底）
 */
export function buildFacialLiveState(
  facial: FacialAnalysis,
  now: number = Date.now(),
): FacialLiveState | null {
  // ① 采集没在跑：没有任何东西可描述
  if (!facial.isDetecting) return null;
  // ② 画面里没有人脸：不产出（"人脸不在画面"本身太歧义，可能只是光线/侧脸/离开，
  //    不足以支撑任何关于 ta 的判断，见 useCallVision 对 headPitch 的同款取舍）
  if (!facial.facePresent) return null;
  // ③ 模型没就绪 / 状态未知：不产出
  if (facial.faceEmotionStatus !== 'ready') return null;
  const reading = facial.faceEmotion;
  if (!reading) return null;

  // ④ 读数过期：人脸离开画面后 `useFaceEmotion` 会清读数，但这里再兜一道，
  //    因为"摘要被构造"与"读数被清"之间可能隔着一次渲染。
  const ageMs = now - reading.sampledAt;
  const ttlMs = typeof reading.ttlMs === 'number' && reading.ttlMs > 0
    ? reading.ttlMs
    : FACIAL_LIVE_STATE_MAX_AGE_MS;
  if (!Number.isFinite(ageMs) || ageMs < 0 || ageMs > ttlMs) return null;

  const p = reading.probabilities;
  if (!p) return null;

  return {
    version: FACIAL_LIVE_STATE_VERSION,
    sampledAt: reading.sampledAt,
    ageMs: Math.round(ageMs),
    present: true,
    probabilities: {
      happy: clamp01(p.happy),
      neutral: clamp01(p.neutral),
      sad: clamp01(p.sad),
    },
    dominant: reading.dominant,
    confidence: clamp01(reading.confidence),
    offMapProbability: clamp01(reading.offMapProbability ?? 0),
    sampleHz: Math.max(0, Math.round((facial.faceEmotionSampleHz || 0) * 10) / 10),
    expressionIntensity: clamp01(facial.expressionIntensity),
    provider: facial.faceEmotionProvider ?? null,
  };
}

/**
 * 给界面用的一句人话说明（**不出现任何心理状态断言**）。
 *
 * 为什么放在这里而不是各页面各写一份：通话页小窗、画像页卡片、对话页情绪条
 * 都要说同一件事，三处各写一遍必然出现"同一个读数三种说法"。
 */
export function describeFacialLiveState(state: FacialLiveState | null): string {
  if (!state) return '暂无新鲜的面部读数';
  const pct = (v: number): string => `${Math.round(v * 100)}%`;
  const label: Record<FaceEmotionLabel, string> = {
    happy: '开心',
    neutral: '中性',
    sad: '偏低落',
  };
  return `本机表情读数（${label[state.dominant]} ${pct(state.confidence)}，`
    + `开心 ${pct(state.probabilities.happy)} / 中性 ${pct(state.probabilities.neutral)} / `
    + `悲伤 ${pct(state.probabilities.sad)}，${state.ageMs}ms 前）`;
}
