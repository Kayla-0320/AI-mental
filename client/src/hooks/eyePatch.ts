/**
 * 眼部 patch 几何（纯逻辑，不依赖 React / DOM）
 *
 * ## 借鉴来源与许可证
 *
 * 裁剪口径借鉴 RT-GENE / RT-BENE 仓库的
 * `rt_gene/src/rt_gene/tracker_generic.py::TrackedSubject.get_eye_image_from_landmarks`
 * （tag 5.0.0）。它是这样取眼图的：
 *
 *   1. **只用眼部角点**（68 点模型里的 36/39/42/45），不做头姿 PnP；
 *   2. 以双眼中心为基准做仿射对齐，把双眼连线转成水平（去 roll）；
 *   3. 水平方向各留 `margin = 眼宽 × margin_ratio`（原实现 margin_ratio = 1.0，
 *      即框宽 = 2 × 眼宽）；竖直方向不自由取值，而由固定宽高比
 *      `desired_ratio = (outH / outW) / 2` 推出 —— 60×36 时即框高 = 0.6 × 框宽；
 *   4. 最后 resize 到目标尺寸（RT-BENE 用 60×36）。
 *
 * ⚠️ **许可证**：RT-GENE / RT-BENE 的代码、权重、数据集均为 **CC BY-NC-SA 4.0**
 * （禁止商用）。本文件是**按上述口径用 TypeScript 重写**的方法借鉴，**不含其任何代码**；
 * 但若将来用其数据集/权重训练模型，必须在报告与许可证清单中标注来源与非商用约束。
 *
 * ## 与 MediaPipe 的对接
 *
 * 本平台已有 FaceLandmarker 的 468 点归一化坐标，眼部角点无需额外检测：
 *
 * | RT-BENE（68 点） | 含义 | MediaPipe（468 点） |
 * |---|---|---|
 * | 36 / 39 | 受试者右眼外/内角 | 33 / 133 |
 * | 42 / 45 | 受试者左眼内/外角 | 362 / 263 |
 *
 * 内部命名沿用 MediaPipe 的**图像视角** L/R（与 `useFacialAnalysis.ts` 的
 * `eyeOuterL: 33` 一致的约定），避免两套命名互相打架。
 *
 * ## 接入状态：**已接入**（2026-09-27）
 *
 * 调用方：`useFacialAnalysis.ts` 在「指标拍」（默认 10fps）调用
 * `EyePatchExtractor.extract(video, landmarks)` 裁出左右眼 patch，
 * 经 `onEyePatchFrame` 回调交给 `useEyeStateModel` 做浏览器内 ONNX 二次判定。
 *
 * - 该通道由 `perceptionCapabilities.EYE_STATE_MODEL_AVAILABLE`（env 开关）控制，
 *   **默认关闭**：关闭时不裁图、不加载 ONNX、不产生任何开销；
 * - ONNX 的判定结果**只上报，不参与 `riskScore`** —— 换信号源会改变风险与落库内容，
 *   按仓库约定属于需要人工审核的改动。方案与实测见
 *   `docs/blink_signal_source_selection.md` 的步 B/C。
 *
 * ## 与 RT-BENE 原实现的一处**有意偏差**
 *
 * 原实现先把整张脸旋转、再在旋转后的图上重算 x 方向宽度；这里改用两个角点的
 * **欧氏距离**（旋转不变量）作为眼宽。小角度下二者等价，大角度下前者会因为
 * "眼线不再水平"而系统性低估眼宽 —— 距离更稳。差高比、margin 语义完全保持。
 */
/**
 * 归一化关键点（0~1）。本模块**自带**这个类型而不从 `multimodal.types` 导入：
 * 一是不用为一个两字段的结构去改公共类型文件，二是让本文件能被
 * `tools/blink_tests` 用 `tsc --lib es2020` 单独编译（零 DOM、零依赖）。
 */
export interface LandmarkPoint {
  x: number;
  y: number;
}

/** 眼部 patch 目标规格 */
export interface EyePatchTarget {
  /** 输出宽（像素），对应 RT-BENE 的 `eye_image_size[0]` */
  width: number;
  /** 输出高（像素），对应 `eye_image_size[1]` */
  height: number;
  /**
   * 水平 margin 比例（相对眼宽）。RT-BENE 原实现取 1.0，
   * 即裁剪框宽 = 眼宽 × (1 + marginRatio) = 2 × 眼宽。
   */
  marginRatio: number;
}

/** 常用目标规格（都是实测核对过的模型输入） */
export const EYE_PATCH_TARGETS = {
  /** RT-BENE 眼 patch：60×36，见 tracker_generic.py `eye_image_size = (60, 36)` */
  rtBene: { width: 60, height: 36, marginRatio: 1.0 },
  /** OCEC（MIT，ONNX 输入 N×3×24×40）：40×24 */
  ocec: { width: 40, height: 24, marginRatio: 1.0 },
  /** OpenVINO open-closed-eye-0001（Apache-2.0，输入 1×3×32×32） */
  openClosedEye: { width: 32, height: 32, marginRatio: 1.0 },
} as const satisfies Record<string, EyePatchTarget>;

/** 单只眼睛的裁剪几何（源图**像素**坐标） */
export interface EyePatchBox {
  /** 眼睛中心（源图像素坐标） */
  center: LandmarkPoint;
  /** 源图上裁剪框的宽（像素） */
  sourceWidth: number;
  /** 源图上裁剪框的高（像素） */
  sourceHeight: number;
  /** 输出宽（像素） */
  outWidth: number;
  /** 输出高（像素） */
  outHeight: number;
  /** 本框是否完全落在画面内（越界时画布会补黑边，调用方据此降低 signalQuality） */
  fullyInsideFrame: boolean;
}

/** 双眼的裁剪几何 */
export interface EyePatchGeometry {
  /** 头 roll（弧度）：双眼连线相对水平的夹角；正值 = 图像右侧的眼睛更低 */
  roll: number;
  /** 图像左侧的那只眼（RT-BENE 的 36/39） */
  eyeLeftInImage: EyePatchBox;
  /** 图像右侧的那只眼（RT-BENE 的 42/45） */
  eyeRightInImage: EyePatchBox;
  /** 双眼中心的中点（源图像素坐标）—— 也是旋转中心 */
  midpoint: LandmarkPoint;
}

/** MediaPipe 468 点里用于眼 patch 的 4 个角点索引 */
export const EYE_CORNER_INDICES = {
  /** 图像左侧眼：外角 33、内角 133 */
  leftOuter: 33,
  leftInner: 133,
  /** 图像右侧眼：内角 362、外角 263 */
  rightInner: 362,
  rightOuter: 263,
} as const;

/** 源帧尺寸 */
export interface FrameSize {
  width: number;
  height: number;
}

function dist(a: LandmarkPoint, b: LandmarkPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** 归一化坐标 → 像素坐标 */
function toPixels(p: LandmarkPoint, frame: FrameSize): LandmarkPoint {
  return { x: p.x * frame.width, y: p.y * frame.height };
}

function isFinitePoint(p: LandmarkPoint | undefined): p is LandmarkPoint {
  return !!p && Number.isFinite(p.x) && Number.isFinite(p.y);
}

/**
 * 计算双眼 patch 的裁剪几何。
 *
 * @param landmarks MediaPipe 面部关键点（归一化 0~1），至少 468 个
 * @param frame     源帧像素尺寸（videoWidth / videoHeight）
 * @param target    目标规格，默认 RT-BENE 的 60×36
 * @returns 几何；**关键点缺失或退化（眼宽为 0）时返回 `null`** —— 不猜、不兜底，
 *          由调用方显式处理（与 RT-BENE 原实现返回 `None` 的行为一致）
 */
export function computeEyePatchGeometry(
  landmarks: readonly LandmarkPoint[] | null | undefined,
  frame: FrameSize,
  target: EyePatchTarget = EYE_PATCH_TARGETS.rtBene,
): EyePatchGeometry | null {
  if (!landmarks || landmarks.length <= EYE_CORNER_INDICES.rightOuter) return null;
  if (!(frame.width > 0) || !(frame.height > 0)) return null;

  const idx = EYE_CORNER_INDICES;
  const lo = landmarks[idx.leftOuter];
  const li = landmarks[idx.leftInner];
  const ri = landmarks[idx.rightInner];
  const ro = landmarks[idx.rightOuter];
  if (!isFinitePoint(lo) || !isFinitePoint(li) || !isFinitePoint(ri) || !isFinitePoint(ro)) {
    return null;
  }

  const loPx = toPixels(lo, frame);
  const liPx = toPixels(li, frame);
  const riPx = toPixels(ri, frame);
  const roPx = toPixels(ro, frame);

  // 眼宽用角点欧氏距离（旋转不变量），见文件头的「有意偏差」
  const leftWidth = dist(loPx, liPx);
  const rightWidth = dist(riPx, roPx);
  if (!(leftWidth > 0) || !(rightWidth > 0)) return null;

  // 眼睛中心 = 两个角点的中点
  const leftCenter = { x: (loPx.x + liPx.x) / 2, y: (loPx.y + liPx.y) / 2 };
  const rightCenter = { x: (riPx.x + roPx.x) / 2, y: (riPx.y + roPx.y) / 2 };

  // 双眼连线相对水平的夹角。图像右侧眼更低 → 正 roll
  const roll = Math.atan2(rightCenter.y - leftCenter.y, rightCenter.x - leftCenter.x);

  const aspect = target.height / target.width;
  const boxOf = (center: LandmarkPoint, eyeWidth: number): EyePatchBox => {
    const sourceWidth = eyeWidth * (1 + target.marginRatio);
    const sourceHeight = sourceWidth * aspect;
    const halfW = sourceWidth / 2;
    const halfH = sourceHeight / 2;
    const fullyInsideFrame =
      center.x - halfW >= 0 &&
      center.y - halfH >= 0 &&
      center.x + halfW <= frame.width &&
      center.y + halfH <= frame.height;
    return {
      center,
      sourceWidth,
      sourceHeight,
      outWidth: target.width,
      outHeight: target.height,
      fullyInsideFrame,
    };
  };

  return {
    roll,
    eyeLeftInImage: boxOf(leftCenter, leftWidth),
    eyeRightInImage: boxOf(rightCenter, rightWidth),
    midpoint: {
      x: (leftCenter.x + rightCenter.x) / 2,
      y: (leftCenter.y + rightCenter.y) / 2,
    },
  };
}

/** 画布 2D 仿射矩阵（与 `CanvasRenderingContext2D.setTransform` 参数同序） */
export interface AffineMatrix {
  a: number;
  b: number;
  c: number;
  d: number;
  e: number;
  f: number;
}

/**
 * 生成「把某个眼 patch 从源帧搬到输出画布」的仿射矩阵。
 *
 * 矩阵 = 平移到输出中心 ∘ 反向旋转 roll ∘ 等比缩放 ∘ 平移到 −眼睛中心，
 * 于是**源图的眼睛中心正好落在输出画布中心**，且眼线方向在输出里是水平的。
 *
 * 用法（`eyePatchCanvas.ts` 里有现成封装）：
 * ```ts
 * const m = eyePatchAffine(box, geom.roll);
 * ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
 * ctx.drawImage(video, 0, 0);
 * ```
 */
export function eyePatchAffine(box: EyePatchBox, roll: number): AffineMatrix {
  const k = box.outWidth / box.sourceWidth;
  const cos = Math.cos(-roll) * k;
  const sin = Math.sin(-roll) * k;
  return {
    a: cos,
    b: sin,
    c: -sin,
    d: cos,
    e: box.outWidth / 2 - (cos * box.center.x - sin * box.center.y),
    f: box.outHeight / 2 - (sin * box.center.x + cos * box.center.y),
  };
}

/** 用仿射矩阵变换一个点（测试与调试用） */
export function applyAffine(m: AffineMatrix, p: LandmarkPoint): LandmarkPoint {
  return {
    x: m.a * p.x + m.c * p.y + m.e,
    y: m.b * p.x + m.d * p.y + m.f,
  };
}
