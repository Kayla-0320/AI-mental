/**
 * 面部三分类读数（开心 / 中性 / 悲伤）的**共享展示层**。
 *
 * ## 为什么要抽出来
 *
 * 这套读数以前只有左下角悬浮工具条（`EmotionStatusDock`）一处出口。倾诉工作区改成
 * 「聊天 + 右栏（陪伴与感知）」以后，悬浮条会压住输入区，被从 `/chat` 撤掉了 ——
 * 于是用户在**最需要它的页面**上看不到这三个数了。现在右栏顶部也放一份，两处必须
 * 显示同一件事，所以把"取数 + 上屏口径"集中到这里，浮层与右栏各自只负责摆放。
 *
 * ## 口径（改动前请读完，这里有明确约束）
 *
 *   - 数据**全部**来自 `AnxietyContext`，本模块不新增任何采集或推理；
 *   - 百分比上屏用 `smoothedProbabilities`（时间常数公开的 EMA，见 `faceEmotionPacing.ts`），
 *     与模型**未平滑**的原始判定**并列**展示，避免把平滑值误当成模型输出；
 *   - 读数有**有效期**（读数自带 `ttlMs`，随实测推理耗时自适应）：人脸离开画面后读数
 *     不会被刷新，过期就显示「暂无新读数」，绝不把几秒前的值当成"当前表情"；
 *   - 没有读数时百分比显示 `—` 而不是 `0%`：`0%` 会被读成"测出来是 0"，
 *     而真相是"这一拍没测到"；
 *   - 措辞必须**非诊断性**（AGENTS.md 安全红线）：说「识别到 / 当前读数」，
 *     不说「你有抑郁 / 焦虑症」，也不给建议；
 *   - `faceEmotion` 与闭眼二通道 `eyeStateModel*` **只展示与上报，不参与 `riskScore` /
 *     `riskLevel`**（表情模型训练集与输入尺寸本机无留档，闭眼模型训练域是近红外眼图）。
 *     按 AGENTS.md「自动化流程不得自行修改安全阈值或升级条件」先只观测 ——
 *     风险等级一律取融合结果 `comprehensiveState.riskLevel`。
 *
 * 隐私：表情与眼动推理在**浏览器本机**完成，视频帧不出设备（与仓库既有说明一致）。
 */
import { useRef } from 'react';
import { useAnxiety } from '../context/AnxietyContext';
import {
  FACE_EMOTION_LABELS,
  FACE_EMOTION_LABELS_ZH,
} from '../services/faceEmotionClassifier';
import { FACE_EMOTION_READING_TTL_MS } from '../hooks/useFaceEmotion';
import type {
  FaceEmotionLabel,
  FaceEmotionModelStatus,
  FaceEmotionReading,
} from '../types/multimodal.types';

/** 三类表情的展示色与图标（与全局情绪色系保持一致：绿=正向、蓝=中性、紫=负向） */
export const FACE_STYLE: Record<FaceEmotionLabel, { color: string; icon: string }> = {
  happy: { color: '#52c41a', icon: '😊' },
  neutral: { color: '#1890ff', icon: '😐' },
  sad: { color: '#722ed1', icon: '😔' },
};

/** 风险等级的中文与非诊断性配色 */
export const RISK_STYLE: Record<string, { label: string; color: string }> = {
  low: { label: '低', color: '#52c41a' },
  medium: { label: '中', color: '#faad14' },
  high: { label: '高', color: '#fa541c' },
  crisis: { label: '危机', color: '#cf1322' },
};

/**
 * 眨眼来源的中文标签。
 *
 * `blendshape` 是 MediaPipe 的眼睑闭合度（首选）；`ear` 是眼关键点纵横比的降级路径
 * （主路径取不到时用）；`none` 表示这一拍没能判定。
 * 必须如实显示——降级路径的时长精度更差，不能看起来和首选路径一样。
 */
const BLINK_METHOD_LABEL: Record<string, string> = {
  blendshape: '眼睑闭合度',
  ear: '关键点降级',
  brightness: '亮度降级',
  none: '未判定',
};

/** 执行后端的中文说明（如实告诉用户"跑在什么上面"，这直接决定刷新速度） */
const PROVIDER_LABEL: Record<string, string> = {
  webgpu: 'WebGPU',
  'wasm-proxy': 'WASM(Worker)',
  'wasm-main-thread': 'WASM(主线程)',
  wasm: 'WASM',
};

/** 峰值窗口（ms）：短期记忆窗口，用来抓住"一闪而过的微笑" */
export const PEAK_WINDOW_MS = 10_000;

/** 取展示用概率：优先平滑值（有则用），否则退回原始值 */
function displayProbabilities(face: FaceEmotionReading) {
  return face.smoothedProbabilities ?? face.probabilities;
}

/** 取概率最大的类别（与展示用概率保持同一口径，避免"条和标签打架"） */
function argmaxLabel(probs: Record<FaceEmotionLabel, number>): FaceEmotionLabel {
  let best: FaceEmotionLabel = FACE_EMOTION_LABELS[0];
  for (const l of FACE_EMOTION_LABELS) if ((probs[l] ?? 0) > (probs[best] ?? 0)) best = l;
  return best;
}

/** 一份读数的全部上屏口径 —— 浮层与右栏都只读它，保证两处永远不会说法不一致 */
export interface FaceEmotionSnapshot {
  face: FaceEmotionReading | null;
  faceStatus: FaceEmotionModelStatus;
  faceReason: string;
  facePresent: boolean;
  faceProvider: string | null;
  faceHz: number;
  cameraEnabled: boolean;
  isStale: boolean;
  showFaceReading: boolean;
  probs: Record<FaceEmotionLabel, number> | null;
  dominant: FaceEmotionLabel | null;
  dominantStyle: { color: string; icon: string } | null;
  /** 模型**未平滑**的原始判定 */
  rawDominant: FaceEmotionLabel | null;
  /** 折叠不进三分类的概率质量（8 类模型的惊讶/恐惧/愤怒/厌恶/轻蔑） */
  offMap: number;
  offMapDominant: boolean;
  /** 近 10 秒峰值 */
  peak: Record<FaceEmotionLabel, number> | null;
  /** 收起态一句话状态（"未开启摄像头" / "识别中…" / 当前主导类 …） */
  faceSummary: string;
  risk: { label: string; color: string };
  textEmotion: string;
  textConfidence: number;
  blink: {
    measuring: boolean;
    method: string;
    rate: number;
    samples: number;
    avgMs: number;
    longCount: number;
    quality: number;
    fps?: number;
    quantum?: number;
    hint: string;
    eyeStateStatus?: string;
    eyeStateAgreement?: number;
  };
}

/**
 * 把 `AnxietyContext` 的面部/眼动读数整理成可直接上屏的一份快照。
 *
 * 每拍（10fps）都会重渲染，这里**不缓存**任何值：直接算就是最新的。
 */
export function useFaceEmotionSnapshot(): FaceEmotionSnapshot {
  const { facialMetrics, comprehensiveState, cameraEnabled, eyeMetrics } = useAnxiety();

  const face = facialMetrics.faceEmotion ?? null;
  const faceStatus = facialMetrics.faceEmotionStatus ?? (cameraEnabled ? 'loading' : 'disabled');
  const faceReason = facialMetrics.faceEmotionUnavailableReason ?? '';
  const facePresent = facialMetrics.facePresent;
  const faceProvider = facialMetrics.faceEmotionProvider ?? null;
  const faceHz = facialMetrics.faceEmotionSampleHz ?? 0;

  // 读数过期判定：有效期随实测推理耗时自适应（慢后端下采样周期变长，
  // 写死 3 秒会让读数在两次成功之间就过期、界面闪成「暂无新读数」）。
  const ttl = face?.ttlMs ?? FACE_EMOTION_READING_TTL_MS;
  const isStale = !face || Date.now() - face.sampledAt > ttl;

  const showFaceReading = faceStatus === 'ready' && !!face && !isStale;
  const probs = showFaceReading && face ? displayProbabilities(face) : null;
  const dominant = probs ? argmaxLabel(probs) : null;
  const dominantStyle = dominant ? FACE_STYLE[dominant] : null;
  const rawDominant = showFaceReading && face ? face.dominant : null;
  /**
   * 折叠不进三分类的概率质量。
   *
   * 为什么必须显示：`probabilities` 是"三类**内部**重新归一化"的占比，
   * 如果 80% 的质量落在三类之外，那三个百分比再高也不代表"就是这个表情"。
   * 超过一半时，标题也不该再报三类里的第一名，否则等于把"认不出"说成了"识别结果"。
   */
  const offMap = showFaceReading && face ? (face.offMapProbability ?? 0) : 0;
  const offMapDominant = offMap > 0.5;

  // ── 近 10 秒峰值：抓住"一闪而过"的微笑 ────────────────────────────────
  // 用 ref 累积（不触发额外渲染），只按 `sampledAt` 变化追加，避免同一读数重复计入。
  const peakRef = useRef<{ at: number; probs: Record<FaceEmotionLabel, number> }[]>([]);
  const lastPeakAtRef = useRef(0);
  if (face && showFaceReading && face.sampledAt !== lastPeakAtRef.current) {
    lastPeakAtRef.current = face.sampledAt;
    peakRef.current.push({ at: face.sampledAt, probs: face.probabilities });
    const cutoff = face.sampledAt - PEAK_WINDOW_MS;
    peakRef.current = peakRef.current.filter((p) => p.at >= cutoff);
  }
  const peak: Record<FaceEmotionLabel, number> | null = peakRef.current.length
    ? FACE_EMOTION_LABELS.reduce((acc, l) => {
        acc[l] = Math.max(...peakRef.current.map((p) => p.probs[l] ?? 0));
        return acc;
      }, {} as Record<FaceEmotionLabel, number>)
    : null;

  const risk = RISK_STYLE[comprehensiveState?.riskLevel ?? 'low'] ?? RISK_STYLE.low;
  const textEmotion = comprehensiveState?.dominantEmotion || '';
  const textConfidence = Math.round((comprehensiveState?.confidence ?? 0) * 100);

  // ── 眨眼检测读数（MediaPipe 眼睑闭合度状态机）──────────────────────
  //
  // `isMeasuring` 为 false 表示这一轮根本没有眼动数据（摄像头未开 / 已 reset），
  // 这时 `blinkRate` 是默认值 0，**不能显示成"0 次/分"**——那是把"没测"说成"测得 0"。
  const blinkMeasuring = !!eyeMetrics?.isMeasuring;
  const blinkRate = eyeMetrics?.blinkRate ?? 0;
  // 参照 hooks/ageConfig.ts 的年龄分层阈值（青少年 highBlinkThreshold=30 / lowBlinkThreshold=8），
  // 只给一个「偏快 / 偏慢 / 正常范围」的**参照**，不下任何结论。
  const blinkHint = !blinkMeasuring
    ? ''
    : blinkRate > 30
      ? '偏快'
      : blinkRate > 0 && blinkRate < 8
        ? '偏慢'
        : '正常范围';

  /** 收起态下统一的一句话状态，避免用户看不出"到底有没有在识别" */
  const faceSummary = (() => {
    switch (faceStatus) {
      case 'ready':
        if (isStale) return facePresent ? '识别中…' : '画面中没有人脸';
        return dominant ? FACE_EMOTION_LABELS_ZH[dominant] : '识别中…';
      case 'loading':
        return '模型加载中…';
      case 'unavailable':
        return '模型不可用';
      default:
        return cameraEnabled ? '等待画面…' : '未开启摄像头';
    }
  })();

  return {
    face, faceStatus, faceReason, facePresent, faceProvider, faceHz, cameraEnabled,
    isStale, showFaceReading, probs, dominant, dominantStyle, rawDominant, offMap, offMapDominant,
    peak, faceSummary, risk, textEmotion, textConfidence,
    blink: {
      measuring: blinkMeasuring,
      method: eyeMetrics?.blinkMethod ?? 'none',
      rate: blinkRate,
      samples: eyeMetrics?.blinkSampleCount ?? 0,
      avgMs: eyeMetrics?.avgBlinkDuration ?? 0,
      longCount: eyeMetrics?.longBlinkCount ?? 0,
      quality: Math.round((eyeMetrics?.signalQuality ?? 0) * 100),
      fps: eyeMetrics?.blinkChannelFps,
      quantum: eyeMetrics?.blinkDurationQuantumMs,
      hint: blinkHint,
      eyeStateStatus: eyeMetrics?.eyeStateModelStatus,
      eyeStateAgreement: eyeMetrics?.eyeStateModelAgreement,
    },
  };
}

/** 展开卡片里的一行「标签 — 值」 */
export function DockRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
      <span style={{ color: '#999', fontSize: 11 }}>{label}</span>
      <span style={{ flex: 1 }} />
      <span style={{ color: '#595959', fontSize: 11 }}>{value}</span>
    </div>
  );
}

/**
 * 三色占比条。
 *
 * ⚠️ 必须定义在**模块作用域**，不能写在渲染函数内部：
 * 组件内部定义会在每次渲染产生一个新的组件类型，React 会把它当成"换了个组件"
 * 从而卸载重建 DOM 节点 —— `transition: width` 就永远不会触发（看着还是跳变）。
 */
export function StackedBar({
  probs,
  height,
}: {
  probs: Record<FaceEmotionLabel, number> | null;
  height: number;
}) {
  return (
    <div
      style={{
        display: 'flex',
        width: '100%',
        height,
        borderRadius: height / 2,
        overflow: 'hidden',
        background: '#f0f0f0',
      }}
    >
      {FACE_EMOTION_LABELS.map((label) => (
        <div
          key={label}
          style={{
            width: `${Math.round((probs?.[label] ?? 0) * 1000) / 10}%`,
            background: FACE_STYLE[label].color,
            transition: 'width 0.35s linear',
          }}
        />
      ))}
    </div>
  );
}

/**
 * 三条「标签 + 百分比」常显行（收起态用紧凑版）。
 *
 * 没有读数时显示 `—` 而不是 `0%`：后者会被读成"测出来是 0"，
 * 而真相是"这一拍没测到 / 摄像头没开"。
 */
export function ProbabilityRow({
  probs,
  compact,
}: {
  probs: Record<FaceEmotionLabel, number> | null;
  compact: boolean;
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: compact ? 8 : 10 }}>
      {FACE_EMOTION_LABELS.map((label) => (
        <span
          key={label}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 3,
            fontSize: compact ? 11 : 12,
            color: '#666',
            whiteSpace: 'nowrap',
          }}
        >
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: 3,
              background: FACE_STYLE[label].color,
              display: 'inline-block',
            }}
          />
          {FACE_EMOTION_LABELS_ZH[label]}
          <b style={{ color: FACE_STYLE[label].color, fontVariantNumeric: 'tabular-nums' }}>
            {probs ? `${Math.round((probs[label] ?? 0) * 100)}%` : '—'}
          </b>
        </span>
      ))}
    </div>
  );
}

/**
 * 面部三分类的**完整读数**（含"为什么没读数"、平滑 vs 原始、刷新率、裁剪源）。
 *
 * 悬浮条的展开态与右栏的展开态共用这一份，避免两处对同一份数据给出不同说法。
 */
export function FaceEmotionDetail({ snap }: { snap: FaceEmotionSnapshot }) {
  const { face, faceStatus, faceReason, cameraEnabled, showFaceReading, probs, dominantStyle, rawDominant, offMap, offMapDominant, peak, faceSummary, faceHz, faceProvider } = snap;

  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5 }}>
        <span style={{ color: '#8c8c8c' }}>面部表情</span>
        <span style={{ flex: 1 }} />
        {showFaceReading && face ? (
          <span style={{ color: offMapDominant ? '#8c8c8c' : dominantStyle?.color, fontWeight: 600 }}>
            {offMapDominant ? '○ 其他表情' : `${dominantStyle?.icon} ${FACE_EMOTION_LABELS_ZH[face.dominant]}`}{' '}
            {Math.round((offMapDominant ? offMap : face.confidence) * 100)}%
          </span>
        ) : (
          <span style={{ color: '#bbb' }}>{faceSummary}</span>
        )}
      </div>

      {showFaceReading && face && probs ? (
        <>
          {/* 三条概率条（数值 = 平滑值，与常显数字同一口径）
              注意：这是**三类内部重新归一化**后的占比；落在三类之外的概率
              由下面「其他表情」那一行如实给出，不能假装三类已穷尽所有可能。 */}
          {FACE_EMOTION_LABELS.map((label) => {
            const p = probs[label] ?? 0;
            return (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                <span style={{ width: 28, color: '#999', fontSize: 11 }}>
                  {FACE_EMOTION_LABELS_ZH[label]}
                </span>
                <div style={{ flex: 1, height: 4, background: '#f0f0f0', borderRadius: 2, overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${Math.round(p * 100)}%`,
                      background: FACE_STYLE[label].color,
                      borderRadius: 2,
                      transition: 'width 0.35s linear',
                    }}
                  />
                </div>
                <span style={{ width: 30, textAlign: 'right', color: '#bbb', fontSize: 10 }}>
                  {Math.round(p * 100)}%
                </span>
              </div>
            );
          })}

          {/* 落在三分类之外的概率（惊讶/恐惧/愤怒/厌恶/轻蔑）。
              >2% 就显示：它解释"为什么三个数加起来是 100% 却不像我"，
              也避免把"模型认不出"读成"模型认出来了"。 */}
          {offMap > 0.02 && (
            <DockRow
              label="其他表情"
              value={`${Math.round(offMap * 100)}%` +
                (offMapDominant ? '（多数判定落在三类之外，上面三条仅供参考）' : '')}
            />
          )}

          {/* 近 10 秒峰值：一闪而过的微笑不会被"当前是中性"完全盖掉 */}
          {peak && (
            <DockRow
              label="近 10 秒峰值"
              value={FACE_EMOTION_LABELS
                .map((l) => `${FACE_EMOTION_LABELS_ZH[l]} ${Math.round((peak[l] ?? 0) * 100)}%`)
                .join(' · ')}
            />
          )}
          {/* 平滑值 vs 模型原始输出：必须并列，避免把平滑值当成模型判定 */}
          {rawDominant && (
            <DockRow
              label="模型原始判定"
              value={`${FACE_EMOTION_LABELS_ZH[rawDominant]} ${Math.round(face.confidence * 100)}%（未平滑）`}
            />
          )}
          <DockRow
            label="刷新"
            value={
              (faceHz > 0 ? `${faceHz.toFixed(1)} 次/秒` : '—') +
              ` · 单次 ${Math.round(face.latencyMs)}ms` +
              (faceProvider ? ` · ${PROVIDER_LABEL[faceProvider] ?? faceProvider}` : '')
            }
          />
          {typeof face.sourceSize === 'number' && face.sourceSize > 0 && (
            <DockRow
              label="裁剪源"
              value={`${face.sourceSize}px → ${face.inputSize}px` +
                (face.sourceSize < face.inputSize ? '（放大，细节不足）' : '')}
            />
          )}
          {/* 可操作的建议：源分辨率不足时，唯一真正有效的做法是让脸在画面里更大 */}
          {typeof face.sourceSize === 'number' && face.sourceSize < face.inputSize && (
            <div style={{ fontSize: 10, color: '#d48806', marginTop: 2, lineHeight: 1.35 }}>
              画面中的人脸偏小：离摄像头近一些识别会更准（实测开心类对清晰度最敏感）。
            </div>
          )}
        </>
      ) : (
        <div style={{ color: '#bbb', fontSize: 11 }}>
          {faceStatus === 'unavailable'
            ? faceReason || '表情模型不可用'
            : cameraEnabled
              ? '暂无新读数'
              : '打开摄像头后可识别面部表情'}
        </div>
      )}
    </div>
  );
}

/**
 * 眨眼读数区块（MediaPipe 眼睑闭合度状态机 + 可选闭眼二通道）。
 *
 * 同样只展示、不参与风险；`isMeasuring` 为 false 时整块显示「未测量」，
 * 绝不把"没测"写成"0 次/分"。
 */
export function BlinkDetail({ snap }: { snap: FaceEmotionSnapshot }) {
  const { blink, cameraEnabled } = snap;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <span style={{ color: '#8c8c8c' }}>眨眼</span>
        <span style={{ flex: 1 }} />
        {blink.measuring ? (
          <span style={{ color: '#595959' }}>
            {blink.rate} 次/分
            {blink.hint && <span style={{ color: '#bfbfbf' }}> · {blink.hint}</span>}
          </span>
        ) : (
          <span style={{ color: '#bbb' }}>未测量</span>
        )}
      </div>

      {blink.measuring ? (
        <>
          <DockRow
            label="平均时长"
            value={
              blink.avgMs > 0
                ? `${Math.round(blink.avgMs)} ms${blink.quantum ? ` ±${Math.round(blink.quantum)}` : ''}`
                : '—'
            }
          />
          <DockRow label="长眨眼" value={`${blink.longCount} 次`} />
          <DockRow label="窗口内" value={`${blink.samples} 次`} />
          <DockRow
            label="来源"
            value={`${BLINK_METHOD_LABEL[blink.method] ?? blink.method}${blink.fps ? ` @${blink.fps}fps` : ''}`}
          />
          <DockRow label="信号质量" value={`${blink.quality}%`} />
          {/* 闭眼二通道（浏览器内 ONNX）：只上报、不参与风险，标注一致率供判断可信度 */}
          {blink.eyeStateStatus === 'ready' && typeof blink.eyeStateAgreement === 'number' && (
            <DockRow
              label="闭眼二通道"
              value={`一致率 ${Math.round(blink.eyeStateAgreement * 100)}%`}
            />
          )}
        </>
      ) : (
        <div style={{ color: '#bbb', fontSize: 11 }}>
          {cameraEnabled ? '暂无眼动读数' : '打开摄像头后可检测眨眼'}
        </div>
      )}
    </div>
  );
}

/** 文本/融合情绪与风险等级两行（两个出口共用） */
export function FusionSummaryRows({ snap }: { snap: FaceEmotionSnapshot }) {
  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <span style={{ color: '#8c8c8c' }}>文本情绪</span>
        <span style={{ flex: 1 }} />
        {snap.textEmotion ? (
          <span style={{ fontWeight: 600, color: '#6366f1' }}>
            {snap.textEmotion} {snap.textConfidence}%
          </span>
        ) : (
          <span style={{ color: '#bbb' }}>暂无</span>
        )}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ color: '#8c8c8c' }}>风险等级</span>
        <span style={{ flex: 1 }} />
        <span style={{ color: snap.risk.color, fontWeight: 600 }}>{snap.risk.label}</span>
      </div>
    </>
  );
}

/**
 * 诚实边界：模型来源 / 精度 / 不参与风险判定。
 *
 * 两处出口都必须带上这段 —— 读数是"当前识别到什么"，不是临床结论。
 */
export function HonestyNote({ snap }: { snap: FaceEmotionSnapshot }) {
  return (
    <div style={{ marginTop: 8, fontSize: 10, color: '#bbb', lineHeight: 1.4 }}>
      表情与眼动均在本机识别，视频不出设备；百分比为 τ=450ms 平滑后的展示值，
      判定与风险一律不使用它。眨眼时长受采样帧率量化
      {snap.blink.quantum ? `（当前 ±${Math.round(snap.blink.quantum)}ms）` : ''}，仅供展示，不参与风险判定。
    </div>
  );
}
