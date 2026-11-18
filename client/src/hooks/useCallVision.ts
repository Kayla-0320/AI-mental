/**
 * 通话期视觉：存在感、画面质量与临时图文辅助参考。
 * 用户开启摄像头后，在内存中保留最近三张224px缩略帧；每轮发送到项目算法服务。
 * 服务端先检查像素质量，再执行匹配的冻结图文专家，差画面退回文字参考。
 * 关闭、切后台、设备断开和读数过期时清除快照；原图不保存。
 * 面部和图文参考不参与风险分级、量表或升级，用户自己的说明优先。
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { mediaStreamManager } from '../services/mediaStreamManager';
import { useFacialAnalysis } from './useFacialAnalysis';
import { useFaceEmotion } from './useFaceEmotion';
import { FACE_EMOTION_MODEL_AVAILABLE } from './perceptionCapabilities';
import { buildFacialLiveState, buildFacialTurnInput, type FacialLiveState, type FacialTurnInput } from '../services/facialLiveState';
import { VisualFrameBuffer } from '../services/visualFrameBuffer';
import type { LandmarkPoint } from './eyePatch';
import type { AgeGroup } from './ageConfig';
import type {
  FaceEmotionModelStatus,
  FaceEmotionReading,
} from '../types/multimodal.types';

/** 通话期视觉的公开状态（全部是实测值或明确的"未知"） */
export interface CallVisionState {
  /** 摄像头是否已开（由页面开关控制） */
  enabled: boolean;
  /** 会话级：采集循环在跑 */
  detecting: boolean;
  /** 帧级：这一拍画面里有没有人脸（存在感，最诚实的那个信号） */
  present: boolean;
  /** 画面是否新鲜（>3× 采样间隔没新帧 ⇒ false） */
  frameFresh: boolean;
  /** 绿通道均值 0–255；没测到时为 null */
  brightness: number | null;
  /** 由亮度粗分档，用于给一句人话提示 */
  lightLevel: 'ok' | 'dim' | 'dark' | 'unknown';
  /**
   * 头部俯仰角（度，正=低头）。
   * ⚠️ 名字刻意叫 `headPitchDeg` 而不是"目光向下"：它是头姿，不是注视（F8）。
   */
  headPitchDeg: number | null;
  /**
   * 通话期口径的"低头"时长占比 0–1。
   *
   * 与全局默认阈值的区别（F8）：通话里 `>25°` **且持续 >3 秒**才起算一次低头；
   * 全局那套 `>10°` 单帧即计，在通话场景会刷出大量假信号。
   */
  headDownRatio: number;
  /** 实测采样频率（Hz），用于如实说明精度 */
  fps: number;
  /** 不可用原因（中文，可直接上屏）；正常时为空串 */
  unavailableReason: string;
  /** 页面在后台导致采集暂停 */
  paused: boolean;
  /** 视觉采样是否处于低帧率模式（存在感只需 2–5 fps） */
  lowPower: boolean;
  /**
   * 本机三分类表情读数（happy/neutral/sad）。**只展示**：
   * 不进风险、不进融合、不写库；仅供通话页小窗与"临时提示词"取用。
   */
  faceEmotion: FaceEmotionReading | null;
  /** 表情模型通道状态（`disabled` = 用户没开摄像头） */
  faceEmotionStatus: FaceEmotionModelStatus;
  /** 表情模型不可用时的中文原因（可直接上屏）；正常时为空串 */
  faceEmotionUnavailableReason: string;
}

const DEFAULT_STATE: CallVisionState = {
  enabled: false,
  detecting: false,
  present: false,
  frameFresh: false,
  brightness: null,
  lightLevel: 'unknown',
  headPitchDeg: null,
  headDownRatio: 0,
  fps: 0,
  unavailableReason: '',
  paused: false,
  lowPower: true,
  faceEmotion: null,
  faceEmotionStatus: 'disabled',
  faceEmotionUnavailableReason: '',
};

/** 通话期视觉采样周期（毫秒）—— 5 fps。存在感不需要 10 fps。 */
export const CALL_VISION_INTERVAL_MS = 200;

/** 帧陈旧判定：超过这么多倍采样间隔没新帧，就认为画面停了 */
export const STALE_FACTOR = 3;

/** 通话期的"低头"阈值：俯仰角超过它、且连续够久才算 */
export const CALL_HEAD_DOWN_DEG = 25;
export const CALL_HEAD_DOWN_HOLD_MS = 3000;

export interface UseCallVisionOptions {
  /** 年龄组：只影响表情校准，不影响本文件暴露的三个量 */
  ageGroup?: AgeGroup | null;
}

export interface UseCallVisionResult extends CallVisionState {
  /** 视频元素（页面拿来当自拍小窗）；未开时为 null */
  videoRef: React.RefObject<HTMLVideoElement | null>;
  /** 开：申请摄像头并开始采样；**失败会抛出**带中文原因的 `MediaUnavailableError` */
  start: () => Promise<void>;
  /** 关：释放摄像头 + 归还设备 + 停止采样 */
  stop: () => void;
  /**
   * 把自拍小窗挂到页面给的宿主元素上。
   *
   * 为什么要这么一个方法：`useFacialAnalysis` 的 `<video>` 是 `document.createElement`
   * 造出来的（不挂 DOM 也能给 MediaPipe 喂帧），而通话页要**让用户看见自己**。
   * 直接把元素交给页面去 append 也行，但那会让"谁负责摘下来"变得含糊 ——
   * 放在这里，`stop()` 与卸载时一定能摘干净。
   */
  attachPreview: (host: HTMLElement | null) => void;
  /**
   * 取**当前**的面部实时状态快照（供通话轮次随消息上行）。
   *
   * 为什么是一个 getter 而不是一个 state 字段：通话轮次的发送发生在
   * "用户说完话"那一刻，需要的是**那一刻**的读数，而不是最近一次渲染时的读数。
   * 这个函数身份稳定（`useCallback([])`），内部读 ref，因此不会让
   * `useVoiceCall` 的轮次回调因 200ms 一次的采样而重建。
   *
   * @returns 快照；**没有新鲜读数时为 `null`**（调用方必须处理，不许用中性值兜底）
   */
  getFacialLiveState: () => FacialTurnInput | null;
}

export function useCallVision(options: UseCallVisionOptions = {}): UseCallVisionResult {
  const { ageGroup = null } = options;

  const [state, setState] = useState<CallVisionState>({ ...DEFAULT_STATE });

  // ── 面部帧回调的中转 ref ────────────────────────────────────────────────
  // `useFacialAnalysis` 必须先于 `useFaceEmotion` 创建（结果要写进它的 `overlayRef`），
  // 所以用 ref 把后者的帧回调转发进去 —— 与 `AnxietyContext` 同一手法。
  const onFaceFrameRef = useRef<
    ((video: HTMLVideoElement, landmarks: readonly LandmarkPoint[]) => void) | null
  >(null);
  const onFaceFrame = useCallback((video: HTMLVideoElement, landmarks: readonly LandmarkPoint[]) => {
    onFaceFrameRef.current?.(video, landmarks);
  }, []);

  const face = useFacialAnalysis(ageGroup, null, null, onFaceFrame);
  const currentFacialRef = useRef(face.metrics);
  currentFacialRef.current = face.metrics;

  // 表情模型通道：**跟随用户开关**（`state.enabled` 由 `start()` 置真）。
  // 关闭时 `enabled=false` → 会话被释放（约 90 MB wasm 堆），不残留资源。
  const faceEmotion = useFaceEmotion(state.enabled && FACE_EMOTION_MODEL_AVAILABLE, face.overlayRef);
  onFaceFrameRef.current = faceEmotion.onFaceFrame;

  /** 最近一拍的实时状态快照（`getFacialLiveState` 读它） */
  const latestLiveStateRef = useRef<FacialLiveState | null>(null);

  // 我们自己的一拍一次采样（与面部管线解耦：它跑它的，我们只读结果）
  const timerRef = useRef(0);
  const brightCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastFrameTsRef = useRef(0);
  const lastFaceFrameTsRef = useRef(0);
  const tickTimesRef = useRef<number[]>([]);
  /** 当前**连续**低头了多久（毫秒） */
  const downRunMsRef = useRef(0);
  const downMsRef = useRef(0);
  const totalMsRef = useRef(0);
  const pausedRef = useRef(false);
  const enabledRef = useRef(false);
  const visualFramesRef = useRef(new VisualFrameBuffer());
  /** 自拍小窗的宿主元素（页面给的）；`stop()` 与卸载时把 video 摘下来 */
  const previewHostRef = useRef<HTMLElement | null>(null);

  // ── 采样一拍：亮度 + 存在感 + 头姿 ──────────────────────────────────────
  const tick = useCallback(() => {
    const now = performance.now();
    if (pausedRef.current) return;

    // 帧率统计（最近 1 秒的拍数）
    const times = tickTimesRef.current;
    times.push(now);
    while (times.length > 0 && now - times[0] > 1000) times.shift();

    const video = face.videoRef.current;
    let brightness: number | null = null;
    if (video && video.videoWidth > 0 && video.videoHeight > 0) {
      let canvas = brightCanvasRef.current;
      if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.width = 32;
        canvas.height = 32;
        brightCanvasRef.current = canvas;
      }
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (ctx) {
        try {
          ctx.drawImage(video, 0, 0, 32, 32);
          const data = ctx.getImageData(0, 0, 32, 32).data;
          let sum = 0;
          let n = 0;
          for (let i = 0; i < data.length; i += 4) {
            sum += data[i + 1]; // 绿通道：与人眼亮度感受最接近
            n += 1;
          }
          if (n > 0) {
            brightness = sum / n;
            lastFrameTsRef.current = now;
          }
        } catch {
          /* 视频还没就绪 / 画面被跨源污染：当作测不到，不编造 */
        }
      }
    }

    const m = face.metrics;
    const pitch = typeof m.actionUnits?.headPitch === 'number' ? m.actionUnits.headPitch : null;

    // 帧陈旧：画面停了就不该继续拿旧读数说话
    const stale =
      lastFrameTsRef.current === 0 ||
      now - lastFrameTsRef.current > STALE_FACTOR * CALL_VISION_INTERVAL_MS;

    // 存在感：只认"这一帧真的有人脸"，且画面是新的
    if (m.facePresent && !stale) lastFaceFrameTsRef.current = now;
    const present =
      !stale && now - lastFaceFrameTsRef.current <= STALE_FACTOR * CALL_VISION_INTERVAL_MS;

    // 用户开摄像头且当前有人脸时采集；内存最多三张，不后台录制。
    if (!present) visualFramesRef.current.clear();
    if (present && video) visualFramesRef.current.capture(video);

    // 通话期口径的低头：`>25°` 且**连续超过 3 秒**才起算。
    //
    // 实现上按"连续的低头拍"累计：这一拍是低头就 +1 拍；一旦当前连续段超过 3 秒，
    // 之后的每一拍都计入 `downMs`。
    // （第一版写成"松手时一次性补记 + 每拍重复记"，会把占比算成两倍。这类错误不抛异常，
    //   只会让屏幕上的数字悄悄偏大 —— 所以改成逐拍累计，口径唯一。）
    if (pitch !== null && pitch > CALL_HEAD_DOWN_DEG) {
      downRunMsRef.current += CALL_VISION_INTERVAL_MS;
      if (downRunMsRef.current >= CALL_HEAD_DOWN_HOLD_MS) {
        downMsRef.current += CALL_VISION_INTERVAL_MS;
      }
    } else {
      downRunMsRef.current = 0;
    }
    totalMsRef.current += CALL_VISION_INTERVAL_MS;

    const lightLevel: CallVisionState['lightLevel'] =
      brightness === null ? 'unknown' : brightness < 45 ? 'dark' : brightness < 80 ? 'dim' : 'ok';

    // 面部实时状态快照：本拍一并算好，供"用户说完话那一刻"取用。
    // 不新鲜/没人脸/模型没就绪时它是 null —— 这是有意的，调用方据此**不注入**。
    latestLiveStateRef.current = buildFacialLiveState(m);

    setState({
      enabled: enabledRef.current,
      detecting: m.isDetecting,
      present,
      frameFresh: !stale,
      brightness: brightness === null ? null : Math.round(brightness),
      lightLevel,
      headPitchDeg: pitch === null ? null : Math.round(pitch * 10) / 10,
      headDownRatio:
        totalMsRef.current > 0 ? Math.min(1, downMsRef.current / totalMsRef.current) : 0,
      fps: times.length,
      unavailableReason: m.unavailableReason || '',
      paused: pausedRef.current,
      lowPower: true,
      faceEmotion: m.faceEmotion ?? null,
      faceEmotionStatus: m.faceEmotionStatus ?? 'disabled',
      faceEmotionUnavailableReason: m.faceEmotionUnavailableReason ?? '',
    });
  }, [face.metrics, face.videoRef]);

  // ── 后台标签页：停表（切后台时定时器被节流，读数会冻在最后一帧）────────
  useEffect(() => {
    const onVisibility = (): void => {
      visualFramesRef.current.clear();
      const hidden = document.visibilityState === 'hidden';
      pausedRef.current = hidden && enabledRef.current;
      if (hidden) {
        // 停表：回来时**重新计时**，绝不把暂停期算进"低头占比"
        downRunMsRef.current = 0;
        tickTimesRef.current = [];
      }
      setState((prev) => ({ ...prev, paused: pausedRef.current }));
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => document.removeEventListener('visibilitychange', onVisibility);
  }, []);

  // ── 设备被抢走 / 被拔掉：立刻如实报出（F9）────────────────────────────
  useEffect(() => {
    return mediaStreamManager.onEnded('camera', (event) => {
      visualFramesRef.current.clear();
      if (!enabledRef.current) return;
      // 设备被收回：读数作废，别再让"最后一次看到的画面"当成本轮的状态
      latestLiveStateRef.current = null;
      setState((prev) => ({
        ...prev,
        detecting: false,
        present: false,
        frameFresh: false,
        faceEmotion: null,
        unavailableReason: event?.trackLabel
          ? `摄像头已断开（${event.trackLabel}）`
          : '摄像头已断开',
      }));
    });
  }, []);

  // ── 采样循环 ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!state.enabled) return;
    timerRef.current = window.setInterval(tick, CALL_VISION_INTERVAL_MS);
    return () => {
      window.clearInterval(timerRef.current);
      timerRef.current = 0;
    };
  }, [state.enabled, tick]);

  // ── 卸载：一定要还设备 ─────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      enabledRef.current = false;
      visualFramesRef.current.clear();
      pausedRef.current = false;
      latestLiveStateRef.current = null;
      window.clearInterval(timerRef.current);
      timerRef.current = 0;
      try {
        face.stop();
      } catch {
        /* 卸载路径上尽力而为 */
      }
    };
    // 只在卸载时跑一次；`face.stop` 的身份是稳定的（内部 useCallback 无依赖）
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const detachPreview = useCallback(() => {
    const host = previewHostRef.current;
    const video = face.videoRef.current;
    if (host && video && video.parentElement === host) host.removeChild(video);
  }, [face.videoRef]);

  const attachPreview = useCallback(
    (host: HTMLElement | null): void => {
      const video = face.videoRef.current;
      const prev = previewHostRef.current;
      // 先把上一个宿主里的 video 摘下来（React 在卸载时会用 null 调一次 ref，
      // 若这时才去读 previewHostRef 就已经是 null 了 —— 那种写法摘不掉）
      if (prev && prev !== host && video && video.parentElement === prev) {
        prev.removeChild(video);
      }
      previewHostRef.current = host;
      if (!host || !video) return;
      video.style.width = '100%';
      video.style.height = '100%';
      video.style.objectFit = 'cover';
      video.style.transform = 'scaleX(-1)'; // 自拍镜像：像照镜子
      video.style.display = 'block';
      if (video.parentElement !== host) host.appendChild(video);
    },
    [face.videoRef],
  );

  const start = useCallback(async (): Promise<void> => {
    enabledRef.current = true;
    try {
      await face.start();
    } catch (err) {
      enabledRef.current = false;
      // 失败必须可见：绝不显示"已开启"
      setState((prev) => ({
        ...prev,
        enabled: false,
        detecting: false,
        unavailableReason: err instanceof Error ? err.message : '摄像头启动失败（原因未知）',
      }));
      throw err;
    }
    if (!enabledRef.current) return;
    // 复位统计：新一段采集不继承上一段的时长
    downRunMsRef.current = 0;
    downMsRef.current = 0;
    totalMsRef.current = 0;
    lastFrameTsRef.current = 0;
    lastFaceFrameTsRef.current = 0;
    tickTimesRef.current = [];
    pausedRef.current = document.visibilityState === 'hidden';
    latestLiveStateRef.current = null; // 新一段采集从"没有读数"开始
    setState({ ...DEFAULT_STATE, enabled: true, detecting: true, paused: pausedRef.current });
    attachPreview(previewHostRef.current);
  }, [attachPreview, face]);

  const stop = useCallback((): void => {
    visualFramesRef.current.clear();
    enabledRef.current = false;
    pausedRef.current = false;
    window.clearInterval(timerRef.current);
    timerRef.current = 0;
    downRunMsRef.current = 0;
    // 采样循环停了，不会再有 tick 来刷新快照 —— 必须在这里显式作废，
    // 否则"关掉摄像头"之后 `getFacialLiveState()` 仍会返回关机前的那一拍。
    latestLiveStateRef.current = null;
    detachPreview();
    try {
      face.stop();
    } catch {
      /* 尽力而为 */
    }
    setState({ ...DEFAULT_STATE });
  }, [detachPreview, face]);

  /**
   * 取当前快照。两道闸门：
   *   ① 采集没在跑（用户没开摄像头 / 已关闭）→ `null`；
   *   ② 本拍没有新鲜读数 → `null`。
   * 身份稳定，供 `useVoiceCall` 在"用户说完话那一刻"直接调用。
   */
  const getFacialLiveState = useCallback((): FacialTurnInput | null => {
    if (!enabledRef.current || pausedRef.current || document.hidden) return null;
    return buildFacialTurnInput(currentFacialRef.current, visualFramesRef.current.snapshot()) ?? null;
  }, []);

  return {
    ...state,
    videoRef: face.videoRef,
    start,
    stop,
    attachPreview,
    getFacialLiveState,
  };
}
