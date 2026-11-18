/**
 * 表情模型通道（浏览器内 ONNX；模型见 `FACE_EMOTION_SPEC`，**展示口径恒为** happy/neutral/sad）
 *
 * 位置与职责：
 *   `useFacialAnalysis`（指标拍，默认 10fps）在**同一帧**上：眨眼/头姿 → 眼 patch，
 *   并把人脸帧交给本 Hook → 本 Hook 按节流裁人脸、跑 `faceEmotionClassifier`
 *   → 结果写进 `overlayRef`，由 `useFacialAnalysis` 的下一拍并进 `FacialAnalysis`。
 *
 * 为什么结果要经过 `overlayRef` 而不是直接 `setState`：
 *   `AnxietyContext` 用 `useEffect(() => setFacialMetrics(facial.metrics), [JSON.stringify(facial.metrics)])`
 *   单向同步面部指标，任何"旁路"写进 `facialMetrics` 的字段都会在下一拍被覆盖。
 *   这与 `useEyeTracking` 处理眼状态模型的做法一致：**只更新 ref，在下一拍一起 setMetrics**。
 *
 * 三条硬约束（都来自本仓库既有约定）：
 * 1. **只上报、不进风险**：结果不参与 `riskScore`/`riskLevel` 融合（安全阈值变更需人工审核）。
 * 2. **失败不拖累主链路**：模型加载/推理失败只把状态置为 `unavailable`，眨眼照常工作。
 * 3. **不额外开流**：只消费 `useFacialAnalysis` 已经拿到的那一帧，不新开摄像头、不新开检测器。
 *
 * ## 2026-10-10 采样节律重做（"一直在等结果"）
 *
 * 旧实现：`minGap = Math.max(1000, 上次耗时 × 2)`。慢后端上一次推理 600~1500ms，
 * 实际周期就被抬到 1.2~3 秒 —— 界面几秒才跳一次，看起来像"卡住了"。
 * 现在按**后端实测耗时**贴着跑（`faceEmotionPacing.ts`）：
 *   - 快后端 → 触到 60ms 下限，而人脸帧只在指标拍（10fps）到来 ⇒ **每拍都跑**；
 *   - 慢后端 → `1.15 × 耗时` 背靠背，比旧的 `2 × 耗时` 快近一倍。
 * 同时把**实测采样率**与**实际执行后端**如实发到 overlay，界面据此说明刷新速度，
 * 而不是让用户对着一个不动的数字猜"是不是坏了"。
 *
 * 另外两条随之而来的修正：
 *   - 读数有效期由**实测耗时**推导（`faceEmotionReadingTtlMs`），不再写死 3 秒 ——
 *     否则慢后端下读数会在两次成功之间过期，界面闪成「暂无新读数」；
 *   - 展示用概率做**时间常数公开**的指数滑动平均（`τ = 450ms`），
 *     让 5~10Hz 的读数在视觉上连续变化；判定一律用未平滑的 `probabilities`。
 */
import { useCallback, useEffect, useRef, useState, type MutableRefObject } from 'react';
import type {
  FacialMetricsOverlay,
  FaceEmotionModelStatus,
} from '../types/multimodal.types';
import {
  FaceEmotionClassifier,
  FACE_EMOTION_SPEC,
  type FaceEmotionResult,
} from '../services/faceEmotionClassifier';
import { FacePatchExtractor, type FaceFrameCallback } from './facePatchCanvas';
import {
  FACE_EMOTION_READING_TTL_FLOOR_MS,
  faceEmotionReadingTtlMs,
  measuredSampleHz,
  shouldSampleFaceEmotion,
  smoothProbabilities,
} from './faceEmotionPacing';

// 状态的唯一来源是 `types/multimodal.types.ts`（overlay 字段用的是同一个联合类型）
export type { FaceEmotionModelStatus };

/**
 * 读数有效期的**下限**（ms）。
 *
 * 实际有效期是自适应的，见 `faceEmotionReadingTtlMs`：慢后端下采样周期变长，
 * 写死的有效期会让读数在两次成功之间就过期。这里保留常量只为兼容旧引用，
 * 新代码请读读数自带的 `ttlMs`（没有时再回退到这个下限）。
 */
export const FACE_EMOTION_READING_TTL_MS = FACE_EMOTION_READING_TTL_FLOOR_MS;

/** 模型不可用时界面统一展示的说明（拿不到具体原因时的兜底） */
export const FACE_EMOTION_UNAVAILABLE_REASON =
  '表情模型未加载（模型文件下载失败或浏览器不支持 ONNX），眨眼检测不受影响';

/**
 * 把"为什么不可用"拼出来。
 *
 * 原来的固定文案把五种完全不同的病因（wasm 运行时 CDN 不可达 / dev server 依赖失效 /
 * 模型文件 404 / 浏览器不支持 wasm SIMD / WebGPU 建会话失败）压成同一句话 ——
 * 用户问"为什么不可用"时，界面和日志都给不出答案。现在把分类器留下的
 * 一句话原因带上（拿不到时才回退到兜底文案）。
 */
export function faceEmotionUnavailableReason(detail: string | null): string {
  return detail
    ? `表情模型未加载，眨眼检测不受影响。原因：${detail}`
    : FACE_EMOTION_UNAVAILABLE_REASON;
}

export interface UseFaceEmotionResult {
  status: FaceEmotionModelStatus;
  isReady: boolean;
  /** 实际执行后端（webgpu / wasm-proxy / wasm），未加载时为 null */
  provider: string | null;
  /** 最近一次成功结果（可能为 null） */
  latest: FaceEmotionResult | null;
  /** 累计成功采样次数 */
  samples: number;
  /** 实测采样率（Hz，4 秒窗口）；样本不足时为 0 */
  sampleHz: number;
  /** 最近一次推理耗时（ms）；尚无成功样本时为 0 */
  lastLatencyMs: number;
  /** 供 `useFacialAnalysis` 的人脸帧回调直接使用 */
  onFaceFrame: FaceFrameCallback;
  /** 关闭多模态时复位（清结果与 overlay；会话由 enabled=false 的清理释放） */
  reset: () => void;
  unavailableReason: string;
}

export function useFaceEmotion(
  enabled: boolean,
  overlayRef?: MutableRefObject<FacialMetricsOverlay> | null,
  onReport?: ((r: FaceEmotionResult) => void) | null,
): UseFaceEmotionResult {
  const [status, setStatus] = useState<FaceEmotionModelStatus>(enabled ? 'loading' : 'disabled');
  const [provider, setProvider] = useState<string | null>(null);
  const [latest, setLatest] = useState<FaceEmotionResult | null>(null);
  const [samples, setSamples] = useState(0);
  const [sampleHz, setSampleHz] = useState(0);
  /** 加载失败的具体原因（分类器给的），用于界面回答"为什么不可用" */
  const [detail, setDetail] = useState<string | null>(null);

  const classifierRef = useRef<FaceEmotionClassifier | null>(null);
  const extractorRef = useRef<FacePatchExtractor | null>(null);
  /** 串行化推理：人脸帧以 10fps 到来，上一拍没跑完就丢帧，不排队 */
  const inFlightRef = useRef(false);
  /**
   * 会话代号：`reset()` 或关闭多模态时 +1。
   * 用来丢弃"上一段会话仍在飞"的推理结果 —— 否则关闭摄像头后归零的读数
   * 会被一个迟到的 `classify()` 重新写回 overlay，下一段会话第一拍就会把旧读数发出去。
   */
  const generationRef = useRef(0);
  /** 「读数过期」定时器句柄 */
  const expiryTimerRef = useRef<number | null>(null);
  /** 加载失败后那次自动重试的定时器句柄 */
  const retryTimerRef = useRef<number | null>(null);
  /** 当前最新读数的采样时刻（0 = 无读数），用于让过期定时器只清掉"自己那一拍" */
  const latestSampledAtRef = useRef(0);
  /** 上一次真正发起推理的时间戳（节律用） */
  const lastAtRef = useRef(0);
  /** 上一次推理耗时，用于按后端能力定节律（见 `faceEmotionPacing`） */
  const lastLatencyRef = useRef(0);
  /** 连续失败次数：失败越多退避越长，成功一次立即清零 */
  const failuresRef = useRef(0);
  /** 近期成功采样时刻（performance.now），用于如实上报实测采样率 */
  const sampleTimesRef = useRef<number[]>([]);
  /** 平滑状态：上一拍发布的平滑概率 + 它的采样时刻（`Date.now()`） */
  const smoothedRef = useRef<{ probs: FaceEmotionResult['probabilities']; at: number } | null>(null);
  const samplesRef = useRef(0);
  /** 上报回调放 ref，避免调用方每次渲染换函数导致闭包抖动 */
  const reportRef = useRef<((r: FaceEmotionResult) => void) | null>(null);
  reportRef.current = onReport ?? null;

  // ── 会话生命周期：跟着多模态开关走（关闭即释放约 90 MB 的 wasm 堆）──
  useEffect(() => {
    if (!enabled) {
      setStatus('disabled');
      setProvider(null);
      classifierRef.current?.dispose();
      classifierRef.current = null;
      return undefined;
    }
    const classifier = new FaceEmotionClassifier();
    classifierRef.current = classifier;
    let cancelled = false;
    setStatus('loading');
    setDetail(null);

    /**
     * 失败后**自动重试一次**（2s 后）。
     *
     * 为什么不能一次定生死：这条链路上有两类**瞬时**失败——
     *   · Vite dev server 重新优化依赖时，`import('onnxruntime-web')` 会拿到
     *     "Outdated Optimize Dep"（504）；
     *   · wasm 运行时来自 CDN，一次网络抖动就没了。
     * 原来第一次失败就直接把模态钉成"不可用"，用户只能靠反复开关或刷新，
     * 而且界面还告诉他"浏览器不支持 ONNX" —— 那是错的。
     */
    const finish = (ok: boolean) => {
      if (cancelled) return;
      setStatus(ok ? 'ready' : 'unavailable');
      setProvider(ok ? classifier.executionProvider : null);
      setDetail(ok ? null : classifier.unavailableDetail);
    };
    classifier.load().then((ok) => {
      if (cancelled) return;
      if (ok) {
        finish(true);
        return;
      }
      retryTimerRef.current = window.setTimeout(() => {
        retryTimerRef.current = null;
        if (cancelled) return;
        classifier.load().then(finish);
      }, 2000);
    });
    return () => {
      cancelled = true;
      if (retryTimerRef.current) {
        window.clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
      }
      // 关掉/卸载时作废本会话所有在飞结果，再释放会话
      generationRef.current += 1;
      if (expiryTimerRef.current) {
        window.clearTimeout(expiryTimerRef.current);
        expiryTimerRef.current = null;
      }
      classifier.dispose();
      if (classifierRef.current === classifier) classifierRef.current = null;
    };
  }, [enabled]);

  // ── 结果/状态 → overlay（下一拍随 FacialAnalysis 一起发布）──
  useEffect(() => {
    if (!overlayRef) return;
    overlayRef.current = {
      ...overlayRef.current,
      faceEmotion: latest,
      faceEmotionStatus: status,
      faceEmotionUnavailableReason: status === 'unavailable' ? faceEmotionUnavailableReason(detail) : '',
      faceEmotionProvider: provider,
      faceEmotionSampleHz: sampleHz,
    };
  }, [latest, status, provider, sampleHz, detail, overlayRef]);

  /**
   * 给最新读数排一个过期定时器。
   *
   * 为什么需要：人脸离开画面 / MediaPipe 降级后不再有人脸帧，`latest` 不会被更新，
   * 而 overlay 会在每一拍指标里被重新发布 —— 界面就会把几分钟前的读数当成"当前表情"。
   * 这个仓库对"没人了还在读情绪"这类问题是明确不能接受的（见 useFacialAnalysis 的
   * facePresent 说明），所以读数必须自带有效期。
   *
   * 有效期由 `ttlMs` 传入（随实测耗时自适应），不再写死 3 秒。
   */
  const scheduleExpiry = useCallback((sampledAt: number, ttlMs: number) => {
    if (expiryTimerRef.current) window.clearTimeout(expiryTimerRef.current);
    expiryTimerRef.current = window.setTimeout(() => {
      expiryTimerRef.current = null;
      // 期间又来了更新的读数 → 定时器已被重置，这里不该清
      if (latestSampledAtRef.current !== sampledAt) return;
      latestSampledAtRef.current = 0;
      smoothedRef.current = null;
      setLatest(null);
    }, ttlMs);
  }, []);

  const onFaceFrame = useCallback<FaceFrameCallback>((video, landmarks) => {
    const classifier = classifierRef.current;
    if (!classifier || !classifier.isReady) return;
    if (inFlightRef.current) return;

    try {
      const now = performance.now();
      // 节律：贴着后端能力连续推理（旧实现是 max(1s, 2×耗时)，慢后端下周期被抬到秒级）
      if (!shouldSampleFaceEmotion({
        lastAt: lastAtRef.current,
        now,
        latencyMs: lastLatencyRef.current,
        consecutiveFailures: failuresRef.current,
      })) return;

      // 裁剪边长跟**模型规格**走：换 224px 的外部 FER 模型时，这里直接从
      // 摄像头原帧裁 224px（不是把 96 放大）—— 这是换模型最大的收益点。
      if (!extractorRef.current || extractorRef.current.outputSize !== FACE_EMOTION_SPEC.inputSize) {
        extractorRef.current = new FacePatchExtractor(FACE_EMOTION_SPEC.inputSize);
      }
      const frame = extractorRef.current.extract(video, landmarks);
      // 裁剪框越界的帧不予采信：画布会留空边，判出来的分数没有意义（与眼 patch 同一规矩）
      if (!frame || !frame.box.fullyInsideFrame) return;

      lastAtRef.current = now;
      inFlightRef.current = true;
      const gen = generationRef.current;
      classifier
        .classify(frame.image)
        .then((result) => {
          if (!result) {
            failuresRef.current += 1;
            return;
          }
          // 会话已经结束（关闭多模态/复位）：这一拍结果作废，不能写回 overlay
          if (gen !== generationRef.current) return;
          failuresRef.current = 0;
          lastLatencyRef.current = result.latencyMs;
          samplesRef.current += 1;

          // 实测采样率：只用**真实成功**的时刻算，样本少时不上报
          const times = sampleTimesRef.current;
          times.push(performance.now());
          if (times.length > 60) times.shift();
          setSampleHz(measuredSampleHz(times));
          setSamples(samplesRef.current);

          // 展示用平滑：α = 1 - exp(-Δt/τ)，Δt = 距上次采样的墙钟时间。
          // 判定一律用未平滑的 probabilities；这里只影响上屏观感。
          const prev = smoothedRef.current;
          const smoothed = smoothProbabilities(
            prev?.probs ?? null,
            result.probabilities,
            prev ? result.sampledAt - prev.at : 0,
          );
          smoothedRef.current = { probs: smoothed, at: result.sampledAt };

          const ttlMs = faceEmotionReadingTtlMs(result.latencyMs);
          const enriched: FaceEmotionResult = {
            ...result,
            smoothedProbabilities: { ...smoothed },
            ttlMs,
            // 裁剪源矩形边长：`sourceSize < inputSize` 就是放大喂模型（必然糊），
            // 界面据此提示"靠近一些"，而不是让用户对着一个偏低的读数猜原因。
            // `inputSize` 跟着模型规格走（当前 224），所以阈值不是写死的 96。
            sourceSize: Math.round(frame.box.size),
          };
          latestSampledAtRef.current = result.sampledAt;
          setLatest(enriched);
          // 排一个"读数过期"定时器：人脸离开画面后没有人脸帧，
          // 若不清空，旧读数会被每一拍重新发布、界面永远显示最后一次结果。
          scheduleExpiry(result.sampledAt, ttlMs);
          reportRef.current?.(enriched);
        })
        .catch((err) => {
          // 这一拍失败绝不能冒泡到 useFacialAnalysis 的帧循环里（那会把错误记到眨眼通道头上）
          failuresRef.current += 1;
          console.warn('[表情模型] 推理链路异常，本帧跳过：', err);
        })
        .finally(() => {
          inFlightRef.current = false;
        });
    } catch (err) {
      // extract()/drawImage 同步抛错也不能污染眨眼通道的帧循环
      failuresRef.current += 1;
      console.warn('[表情模型] 人脸裁剪异常，本帧跳过：', err);
      inFlightRef.current = false;
    }
  }, [scheduleExpiry]);

  const reset = useCallback(() => {
    // 作废在飞结果（gen 变了，迟到的 then 会被丢弃）
    generationRef.current += 1;
    if (expiryTimerRef.current) {
      window.clearTimeout(expiryTimerRef.current);
      expiryTimerRef.current = null;
    }
    latestSampledAtRef.current = 0;
    inFlightRef.current = false;
    lastAtRef.current = 0;
    lastLatencyRef.current = 0;
    failuresRef.current = 0;
    sampleTimesRef.current = [];
    smoothedRef.current = null;
    samplesRef.current = 0;
    setSamples(0);
    setSampleHz(0);
    setLatest(null);
    if (overlayRef) {
      overlayRef.current = {
        ...overlayRef.current,
        faceEmotion: null,
        faceEmotionUnavailableReason: '',
        faceEmotionSampleHz: 0,
      };
    }
  }, [overlayRef]);

  return {
    status,
    isReady: status === 'ready',
    provider,
    latest,
    samples,
    sampleHz,
    lastLatencyMs: lastLatencyRef.current,
    onFaceFrame,
    reset,
    unavailableReason: status === 'unavailable' ? faceEmotionUnavailableReason(detail) : '',
  };
}
