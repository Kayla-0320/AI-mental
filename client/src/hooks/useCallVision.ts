/**
 * 通话期视觉 —— **只做三件诚实的事**（施工单 §S7 / `video_call_build_plan.md` 阶段 4）
 *
 * ## 它做什么
 *
 * | 做什么 | 具体 | 用什么算 |
 * |---|---|---|
 * | ① 存在感 | "你在画面里 / 暂时看不到你" | `FacialAnalysis.facePresent`（帧级、抗噪） |
 * | ② 画面质量 | "光线有点暗" | 视频帧的绿通道均值（自己画到 32×32 画布上） |
 * | ③ 给自己看的小读数 | 头部俯仰角 / 低头时长占比 | `actionUnits.headPitch` |
 *
 * ## 它**不**做什么（这条比做什么更重要）
 *
 * - ❌ **不进融合、不进风险、不上报**（F7）。这个 hook 里没有任何 `fetch`、
 *   没有任何 `reportToServer`、不碰 `AnxietyContext`。它只是"把读数摆出来"。
 *   硬验收标准是：**开/关摄像头两次，落库的 `riskLevel` 完全一致**。
 * - ❌ 不做眨眼频率。10 fps 采样下"次/分"是量纲级不准（`§S7.0`），
 *   上一轮已按用户决定暂缓；这里连字段都不暴露，免得将来被"顺手加回去"。
 * - ❌ 不做注视方向。`EyeMovementAnalysis.downwardGazeRatio` 名字说的是"目光向下"，
 *   实际算的是**头部角度** —— 通话里低头看手机是常态，会被持续误报成"回避"（F8）。
 *   所以这里用的是 `headPitchDeg` 这个**诚实名字**，而且**只上屏、不告警**。
 *
 * ## 为什么复用 `useFacialAnalysis` 而不是自己写一条管线
 *
 * 那条管线（MediaPipe + 单例设备管理 + 幂等 start + 失败可见）已经在前几个阶段
 * 改造过并通过实测验收。通话期需要的差别只有三点：低帧率、自己算亮度、把"只上屏不报"
 * 落实。重写一遍等于把那些已经踩过的坑再踩一遍。
 *
 * ## F9：设备被抢 / 切后台 / 帧陈旧
 *
 * 三个场景都要**如实说**，而不是让读数冻在最后一帧假装还活着：
 *
 * | 场景 | 这里怎么处理 |
 * |---|---|
 * | 流被系统收回 | 订阅 `mediaStreamManager.onEnded('camera')` → 立刻置不可用，UI 说"摄像头已断开" |
 * | 切到后台标签页 | `visibilitychange` → `paused`，采样空转；回来时**重置计时** |
 * | 帧陈旧（>3× 采样间隔没新帧） | `frameFresh=false` → 读数显示为"—"，不参与"低头占比" |
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { mediaStreamManager } from '../services/mediaStreamManager';
import { useFacialAnalysis } from './useFacialAnalysis';
import type { AgeGroup } from './ageConfig';

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
}

export function useCallVision(options: UseCallVisionOptions = {}): UseCallVisionResult {
  const { ageGroup = null } = options;

  const face = useFacialAnalysis(ageGroup, null, null);
  const [state, setState] = useState<CallVisionState>({ ...DEFAULT_STATE });

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
    });
  }, [face.metrics, face.videoRef]);

  // ── 后台标签页：停表（切后台时定时器被节流，读数会冻在最后一帧）────────
  useEffect(() => {
    const onVisibility = (): void => {
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
      if (!enabledRef.current) return;
      setState((prev) => ({
        ...prev,
        detecting: false,
        present: false,
        frameFresh: false,
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
      pausedRef.current = false;
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
    // 复位统计：新一段采集不继承上一段的时长
    downRunMsRef.current = 0;
    downMsRef.current = 0;
    totalMsRef.current = 0;
    lastFrameTsRef.current = 0;
    lastFaceFrameTsRef.current = 0;
    tickTimesRef.current = [];
    pausedRef.current = document.visibilityState === 'hidden';
    setState({ ...DEFAULT_STATE, enabled: true, detecting: true, paused: pausedRef.current });
    attachPreview(previewHostRef.current);
  }, [attachPreview, face]);

  const stop = useCallback((): void => {
    enabledRef.current = false;
    pausedRef.current = false;
    window.clearInterval(timerRef.current);
    timerRef.current = 0;
    downRunMsRef.current = 0;
    detachPreview();
    try {
      face.stop();
    } catch {
      /* 尽力而为 */
    }
    setState({ ...DEFAULT_STATE });
  }, [detachPreview, face]);

  return {
    ...state,
    videoRef: face.videoRef,
    start,
    stop,
    attachPreview,
  };
}
