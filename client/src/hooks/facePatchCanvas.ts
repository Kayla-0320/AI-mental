/**
 * 人脸 patch 像素提取（DOM 侧）—— 供浏览器内三分类表情模型使用
 *
 * 职责很窄：把同一帧的 MediaPipe 关键点围成的**整张脸**裁成
 * `size×size` 的方形像素块，喂给 `faceEmotionClassifier`。
 *
 * 设计约束（与 `eyePatchCanvas.ts` 保持同一条规矩）：
 * - **视频不出设备**：整条链路在本机 canvas + WASM/WebGPU 内完成，不上传帧、不落盘。
 * - 与 `useFacialAnalysis` **共用同一路摄像头与同一帧关键点**，不新开流、不新开检测器。
 * - 裁剪框越界时不"补假像素"：`drawImage` 的源矩形被画面裁掉的部分自然留空，
 *   同时用 `fullyInsideFrame=false` 如实告诉上层"这帧不可采信"（下游直接丢帧）。
 *
 * 为什么裁剪框要按"脸框外扩 + 取正方形"算：
 *   MediaPipe 的 468 点包含脸廓，最上沿大约在额头上沿，直接当 bbox 会切掉额头，
 *   而表情恰恰要靠眉额。所以纵向上下各扩一段、左右各扩一段（具体数值见下面
 *   `FACE_PATCH_MARGIN` —— 它们是**按实测校正过**的，不是拍脑袋定的），
 *   再取最大边做正方形 —— 正方形是为了不做长宽比变形，
 *   因为训练时 `3.1.train_ResNet101.py` 只用 `ToTensor()`、没有 resize。
 *
 * ⚠️ 这几个比例直接决定模型的输出（实测纵向差 4px 就能把 35% 的"开心"改判成"悲伤"），
 *    改动前请先跑 `scripts/face_framing_audit.py`，改完再跑
 *    `scripts/face_emotion_domain_shift_check.py` 的 shift_* 行复核。
 */
import type { LandmarkPoint } from './eyePatch';
import { FACE_EMOTION_INPUT_SIZE } from '../services/faceEmotionClassifier';

/** 裁剪框（像素坐标，相对当前帧） */
export interface FacePatchBox {
  x: number;
  y: number;
  /** 正方形边长 */
  size: number;
  /** 是否完整落在画面内（false 时下游应丢弃该帧） */
  fullyInsideFrame: boolean;
}

/** 一帧的人脸 patch */
export interface FacePatchFrame {
  /** 已缩放到目标尺寸的像素 */
  image: ImageData;
  box: FacePatchBox;
  sourceWidth: number;
  sourceHeight: number;
}

/** 外扩比例（相对脸框宽/高）
 *
 * ## 2026-10-10：这两个纵向比例是按**实测**校正过的，不要凭感觉改
 *
 * `scripts/face_framing_audit.py` 用**与前端同一个** MediaPipe FaceLandmarker
 * 量了 1195 张训练/验证图，得到模型训练时看到的口径：
 *   人脸关键点包围盒高度 / 图高 = **0.6608**，包围盒中心 y / 图高 = **0.5884**。
 *
 * 而旧参数 (top 0.35 / bottom 0.15) 让前端产出的是 0.6667 / 0.5667 ——
 * 缩放几乎对得上（+0.006），但**人脸在 patch 里偏高 0.0217（≈2.1px）**。
 *
 * 为什么 2px 也值得改：该模型对纵向位置极其敏感（`face_emotion_domain_shift_check.py`
 * 的 shift_* 行，validation n=240，真实开心样本）：
 *   上移 2px → 开心召回 95.0%（无害）；4px → 52.5%，其中 **35% 被判成"悲伤"**；
 *   6px → 11.2%，**76.2% 被判成"悲伤"**。也就是说"咧嘴笑被识别成悲伤"正是这个方向。
 *   旧参数的 2.1px 恰好落在临界点附近，校正到 ~0 是零成本的。
 *
 * 推导（设关键点包围盒高 hb）：
 *   patch 内人脸高度占比 = hb / side，要求 = 0.6608 → side = 1.513·hb
 *   patch 内人脸中心 y   = 0.5 − (bottom−top)·(hb/side)/2，要求 = 0.5884
 *   → top + bottom = 1/0.6608 − 1 = 0.513，bottom − top = 2(0.5−0.5884)/0.6608 = −0.2676
 * 复核命令：`python scripts/face_framing_audit.py`（会打印"现状 vs 目标"的偏差）。
 */
export const FACE_PATCH_MARGIN = {
  left: 0.147,
  right: 0.149,
  top: 0.390,
  bottom: 0.123,
};


/**
 * 由关键点算裁剪框。关键点必须与帧尺寸同源（MediaPipe 输出的是归一化坐标）。
 *
 * @returns 关键点退化（空/零面积/过小）时返回 `null`
 */
export function computeFacePatchBox(
  landmarks: readonly LandmarkPoint[] | null | undefined,
  frame: { width: number; height: number },
): FacePatchBox | null {
  if (!landmarks || landmarks.length === 0) return null;
  if (!(frame.width > 0) || !(frame.height > 0)) return null;

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of landmarks) {
    if (!Number.isFinite(p?.x) || !Number.isFinite(p?.y)) continue;
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  if (!Number.isFinite(minX) || !Number.isFinite(minY)) return null;

  const x0 = minX * frame.width;
  const y0 = minY * frame.height;
  const x1 = maxX * frame.width;
  const y1 = maxY * frame.height;
  const w = x1 - x0;
  const h = y1 - y0;
  // 关键点退化（全部重合）→ 不产出一张纯噪声图
  if (w < 8 || h < 8) return null;

  const left = x0 - w * FACE_PATCH_MARGIN.left;
  const right = x1 + w * FACE_PATCH_MARGIN.right;
  const top = y0 - h * FACE_PATCH_MARGIN.top;
  const bottom = y1 + h * FACE_PATCH_MARGIN.bottom;

  const side = Math.max(right - left, bottom - top);
  // 裁剪框比整帧还大（关键点明显异常）→ 与其喂一张缩小的怪图，不如不报
  if (side > Math.max(frame.width, frame.height)) return null;

  const cx = (left + right) / 2;
  const cy = (top + bottom) / 2;
  const x = cx - side / 2;
  const y = cy - side / 2;

  return {
    x,
    y,
    size: side,
    fullyInsideFrame: x >= 0 && y >= 0 && x + side <= frame.width && y + side <= frame.height,
  };
}

/**
 * 人脸 patch 提取器。
 *
 * 复用一块离屏画布，因此**可以按帧调用**而不会每帧分配新画布。
 * 非线程安全（持有画布），每个消费方各自持有一个实例即可。
 */
export class FacePatchExtractor {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  constructor(private readonly size: number = FACE_EMOTION_INPUT_SIZE) {}

  /** 输出边长（像素） */
  get outputSize(): number {
    return this.size;
  }

  /**
   * 从一帧视频 + 该帧的 MediaPipe 关键点提取人脸 patch。
   *
   * @returns 关键点退化或帧尺寸不可用时返回 `null`
   */
  extract(
    video: HTMLVideoElement | HTMLCanvasElement | ImageBitmap,
    landmarks: readonly LandmarkPoint[] | null | undefined,
  ): FacePatchFrame | null {
    const width = 'videoWidth' in video ? video.videoWidth : video.width;
    const height = 'videoHeight' in video ? video.videoHeight : video.height;
    const box = computeFacePatchBox(landmarks, { width, height });
    if (!box) return null;

    if (!this.canvas || !this.ctx) {
      const canvas = document.createElement('canvas');
      canvas.width = this.size;
      canvas.height = this.size;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return null;
      // ── 下采样质量：必须是 'high' ────────────────────────────────────
      //
      // 默认的 `imageSmoothingQuality` 是 `'low'`，在源矩形明显大于 96px 时
      // 只做 2×2 双线性采样 —— 相当于**欠采样**：卷积核只覆盖 2 个源像素，
      // 中间的高频被直接丢掉（还带混叠）。而模型判"开心"依赖的恰恰是
      // 嘴角弧度、鼻唇沟、卧蚕这些**高频**特征。
      //
      // 实测（`scripts/face_emotion_domain_shift_check.py`，validation n=240，
      // 在 96×96 patch 上人为施加等价模糊）：
      //   清晰 97.5%（开心召回 96.2%） → blur1 92.1%（83.8%）→ blur2 64.6%（28.7%）
      // 即：模糊一旦上来，开心会被成片判成"中性"（blur2 有 68.8% 的开心被判中性）。
      // `'high'` 让浏览器用更完整的降采样滤波，成本只有这一层 96×96 的绘制。
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      this.canvas = canvas;
      this.ctx = ctx;
    }

    const ctx = this.ctx;
    // 源矩形可能部分越界：按 canvas 规范，越界部分不会绘制（不是伪造像素），
    // 与 `fullyInsideFrame` 标记配合，由上层决定是否采信。
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.size, this.size);
    ctx.drawImage(video, box.x, box.y, box.size, box.size, 0, 0, this.size, this.size);

    return {
      image: ctx.getImageData(0, 0, this.size, this.size),
      box,
      sourceWidth: width,
      sourceHeight: height,
    };
  }
}

/** `useFacialAnalysis` 的人脸帧回调签名（指标拍，默认 10fps） */
export type FaceFrameCallback = (
  video: HTMLVideoElement,
  landmarks: readonly LandmarkPoint[],
) => void;
