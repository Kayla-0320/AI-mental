/**
 * 眼 patch 像素提取（DOM 侧）—— `eyePatch.ts` 的落地产物
 *
 * 职责很窄：把 `computeEyePatchGeometry` 算出的几何，用 canvas 仿射变换真正
 * 裁成像素块，供下游的**浏览器内**每眼分类器使用（onnxruntime-web）。
 *
 * 接入状态：**已接入**（2026-09-27）。调用方是 `useFacialAnalysis.ts` 的指标拍
 * （默认 10fps），产出经 `onEyePatchFrame` → `useEyeStateModel` 做浏览器内 ONNX 判定；
 * 由 env 开关 `VITE_ENABLE_EYE_STATE_MODEL` 控制，默认关闭、且结果只上报不进风险。
 *
 * 设计约束：
 * - **视频不出设备**。整条链路都在本机 canvas + WASM 内完成，不上传帧、不落盘；
 *   这是本平台相对"把眼部图像发给服务器"方案的核心优势，接入时不要破坏它。
 * - 与 `useFacialAnalysis` **共用同一路摄像头与同一帧关键点**，不新开流、不新开检测器。
 * - 几何为空（关键点退化）时返回 `null`，由调用方如实上报，而不是给一张黑图。
 */
import {
  computeEyePatchGeometry,
  eyePatchAffine,
  EYE_PATCH_TARGETS,
  type EyePatchBox,
  type EyePatchGeometry,
  type EyePatchTarget,
  type LandmarkPoint,
} from './eyePatch';

/** 一帧的双眼 patch 像素 */
export interface EyePatchPixels {
  /** 图像左侧眼（RT-BENE 的 36/39） */
  left: ImageData;
  /** 图像右侧眼（RT-BENE 的 42/45） */
  right: ImageData;
  /** 本帧的几何（含 roll 与越界标记，供 signalQuality 使用） */
  geometry: EyePatchGeometry;
  /** 两只眼是否都完整落在画面内 */
  fullyInsideFrame: boolean;
}

interface TargetCanvas {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
}

function createTargetCanvas(target: EyePatchTarget): TargetCanvas {
  const canvas = document.createElement('canvas');
  canvas.width = target.width;
  canvas.height = target.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('眼 patch 提取失败：无法获取 2D 画布上下文');
  return { canvas, ctx };
}

/** 把一只眼按几何画进目标画布，返回该画布的 ImageData */
function renderEye(
  slot: TargetCanvas,
  source: CanvasImageSource,
  box: EyePatchBox,
  roll: number,
): ImageData {
  const { canvas, ctx } = slot;
  const m = eyePatchAffine(box, roll);

  // 先用单位矩阵清空，避免在仿射矩阵下 clearRect 只清掉一个斜矩形
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 整帧画一次，画布尺寸即输出 ROI —— 越界部分自然留空（透明），不伪造像素
  ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
  ctx.drawImage(source, 0, 0);

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/**
 * 眼 patch 提取器。
 *
 * 复用两个离屏画布（每只眼一个），因此**可以按帧调用**，不会每帧分配新画布。
 * 非线程安全（持有画布），每个消费方各自持有一个实例即可。
 */
export class EyePatchExtractor {
  private readonly left: TargetCanvas;
  private readonly right: TargetCanvas;
  private readonly target: EyePatchTarget;

  constructor(target: EyePatchTarget = EYE_PATCH_TARGETS.rtBene) {
    this.target = target;
    this.left = createTargetCanvas(target);
    this.right = createTargetCanvas(target);
  }

  /** 输出尺寸（像素） */
  get outputSize(): { width: number; height: number } {
    return { width: this.target.width, height: this.target.height };
  }

  /**
   * 从一帧视频 + 该帧的 MediaPipe 关键点提取双眼 patch。
   *
   * @returns 关键点退化或帧尺寸不可用时返回 `null`
   */
  extract(
    video: HTMLVideoElement | HTMLCanvasElement | ImageBitmap,
    landmarks: readonly LandmarkPoint[] | null | undefined,
  ): EyePatchPixels | null {
    const width = 'videoWidth' in video ? video.videoWidth : video.width;
    const height = 'videoHeight' in video ? video.videoHeight : video.height;
    const geometry = computeEyePatchGeometry(landmarks, { width, height }, this.target);
    if (!geometry) return null;

    const left = renderEye(this.left, video, geometry.eyeLeftInImage, geometry.roll);
    const right = renderEye(this.right, video, geometry.eyeRightInImage, geometry.roll);

    return {
      left,
      right,
      geometry,
      fullyInsideFrame:
        geometry.eyeLeftInImage.fullyInsideFrame && geometry.eyeRightInImage.fullyInsideFrame,
    };
  }
}
