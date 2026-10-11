/**
 * 浏览器内「表情」模型（ONNX Runtime Web）—— 模型可换，**展示口径恒为** happy / neutral / sad
 *
 * ## 2026-10-10 起：模型是**规格**，不是硬编码
 *
 * 本文件原先把"输入 96px、RGB、只 /255、输出 3 类"写死在代码里，于是**换模型 = 改代码**。
 * 现在这段事实集中在 `FaceEmotionModelSpec`（`FACE_EMOTION_SPEC`）里：
 *
 * | 字段 | 含义 | 写错的后果 |
 * |---|---|---|
 * | `inputSize` | 方形输入边长 | 自训练模型 96→224：95.2% → 40.6%（已实测） |
 * | `labels` | **模型自己的**类别名，顺序 = 输出索引顺序 | 名字贴错 = "全预测对、全贴错"（实测 95.2% → 40.6%） |
 * | `fold` | 模型类 → 展示三分类；`null` = 不参与 | 8 类模型不折叠就无法上屏 |
 * | `mean`/`std` | 归一化；`null` = 只 `x/255` | 猜错就退化：FER+ 用 `/255` 时 69.1% → **33.3%** 且输出恒为 neutral |
 * | `grayscale`/`bgr`/`channels` | 输入通道口径 | FER+ 那类模型是单通道灰度 |
 *
 * 关键约束：**下游一律只说三分类**。`FaceEmotionReading.probabilities` 只有 3 个键，
 * 所以 `faceEmotionPacing`（节律/TTL/平滑）与 `Profile` 全部不用改。
 * 外部模型多出来的类（惊讶/恐惧/愤怒…）不丢：折叠不进去的概率放在
 * `offMapProbability` 里如实上报，工具栏显示「其他表情 X%」。
 *
 * ## 当前线上模型（2026-10-10 起）
 *
 * **HSEmotion `enet_b0_8_best_vgaf`**（EfficientNet-B0，AffectNet-8 + VGAF 微调）：
 *   - 输入 `input` `[b,3,224,224]`，RGB，`x/255` 后 ImageNet 归一化
 *   - 输出 `output` `[b,8]`，顺序见 `FACE_EMOTION_SPEC.labels`
 *   - 15.3 MB / Apache-2.0 / 单张 ≈17 ms（CPU 单线程）、47 ms（ORT Web wasm）
 *
 * 换掉自训练 ResNet50 的**实测理由**（RAF-DB test，n=794 真实成人人脸）：
 * 旧模型把真实笑脸判成「悲伤」56.9%、真实中性脸判成「悲伤」92.5%、sad AUC 0.630；
 * 新模型分别是 0.8% / 10.8% / **0.965**。完整证据链见
 * `docs/fer_external_model_integration.md`。
 *
 * ## 历史（自训练模型为什么曾经"看起来 95%"）
 *
 * 那个模型在本仓库数据集上是 95.2%~95.8%，但那个数据集的 `sad` 大量是**婴幼儿嚎哭**，
 * 8 类模型把它读成 Fear/Surprise —— 两者不是同一个概念。
 * 它当年踩过的两个坑仍然值得记住（都是"训练时的事实没被如实记录"）：
 *   1. 数据集图片原生 96×96，训练没 resize，喂 224 → 40.6%；
 *   2. 训练脚本用 `classes = os.listdir(data_dir)`（不是 `sorted()`），
 *      实测顺序是 `neutral, happy, sad`，按字典序假设会让准确率掉到 40.6%。
 *
 * ⚠️ 无论换哪个模型：**类别顺序必须用数据实测**（`fer_external_eval.py --probe-labels`
 * 会穷举所有映射），不能只看 README 或目录名。
 *
 * ## 隐私与降级
 *
 * 全程在本机 canvas + WASM/WebGPU 内完成，**不传帧、不落盘**；与眨眼检测共用
 * 同一路摄像头与同一帧关键点（`useFacialAnalysis` 的指标拍），不新开流。
 * 任何一步失败都不抛给调用方 —— 返回 `null` 由上层如实上报"模型不可用"，
 * 眨眼主链路完全不受影响（与 `eyeStateClassifier` 的约束一致）。
 */

import type { FaceEmotionLabel, FaceEmotionReading } from '../types/multimodal.types';
import type { RgbaImage } from './eyeStateClassifier';

export type { RgbaImage };

/**
 * **展示口径**的三个类别（顺序固定，与模型输出顺序无关）。
 *
 * 界面、平滑、节律、上报、风险融合的措辞全都只说这三种；
 * 某个模型自己的类别顺序在 `FaceEmotionModelSpec.labels` 里，两者靠 `fold` 对应。
 *
 * ⚠️ 历史坑：这里曾经同时兼职"模型输出索引顺序"。训练脚本用的是
 * `classes = os.listdir(data_dir)`（**不是 `sorted()`**），实测顺序是
 * `neutral(0) / happy(1) / sad(2)` —— 用它对映射得 95.2%，用 `happy, neutral, sad`
 * 只有 40.6%。现在这两件事已经拆开，但**换模型后仍必须重跑**
 * `scripts/face_emotion_accuracy_check.py`（自带 6 种排列扫描）复核。
 */
export const FACE_EMOTION_LABELS: readonly FaceEmotionLabel[] = ['neutral', 'happy', 'sad'];

/** 上屏用的中文短标签（界面与证据文本共用，避免各处各写一份） */
export const FACE_EMOTION_LABELS_ZH: Record<FaceEmotionLabel, string> = {
  happy: '开心',
  neutral: '中性',
  sad: '悲伤',
};

/**
 * 模型规格：把"换一个 FER 模型"变成**改这一份数据**（而不是改本文件里散落的常量）。
 *
 * 每个字段都必须来自**模型自己的事实**，不能猜：
 * 归一化猜错、输入尺寸猜错、类别顺序猜错，三者任一都会让准确率掉到随机水平附近
 * （本项目已经踩过其中两个，见文件头）。
 */
export interface FaceEmotionModelSpec {
  /** 展示用名字（日志/界面/文档引用同一份，避免各处叫法不一） */
  readonly name: string;
  /** 模型地址（Vite 的 `public/` 直接按根路径提供） */
  readonly url: string;
  /** 方形输入边长（模型训练时看到的边长） */
  readonly inputSize: number;
  /** **模型自己的**类别名，数组顺序 = 输出张量索引顺序 */
  readonly labels: readonly string[];
  /**
   * 模型类 → 展示三分类；长度必须等于 `labels`。
   * `null` = 这一类折叠不到 neutral/happy/sad（例如 8 类模型的 fear/surprise），
   * 它不算命中也不算误判，但会进 `offMapProbability` 如实上报。
   */
  readonly fold: readonly (FaceEmotionLabel | null)[];
  /** 归一化 `(x/255 - mean) / std`；`null` = 只做 `x/255` */
  readonly mean?: readonly number[] | null;
  readonly std?: readonly number[] | null;
  /** 输入通道数（FER+ 那类是单通道灰度） */
  readonly channels?: 1 | 3;
  /** 是否转灰度（转完仍按 `channels` 展开） */
  readonly grayscale?: boolean;
  /** 通道序是否为 BGR */
  readonly bgr?: boolean;
  /** 模型文件字节数（界面如实展示"要下载多少"） */
  readonly bytes?: number;
}

/**
 * 线上在用的规格。
 *
 * ## 2026-10-10 换模型：自训练 ResNet50(3类) → HSEmotion enet_b0_8_best_vgaf(8类)
 *
 * 换的理由是**实测**，不是口味。在公开的**真实世界成人人脸**数据集 RAF-DB test 上
 * （`scripts/fer_cross_domain_bench.py`，n=794），旧模型把
 *   - 真实**笑脸**判成「悲伤」**56.9%**（喂紧取景原图时 98.3%）
 *   - 真实**中性脸**判成「悲伤」**92.5%**
 *   - sad 的 AUC 只有 **0.630**（排序能力接近随机）
 * 而 HSEmotion：笑脸误判 **0.8%**、中性误判 10.8%、sad AUC **0.965**；
 * 体积 89.6 MB → **15.3 MB**，许可 Apache-2.0（EmotiEffLib，明示学术与商用均无限制）。
 *
 * 代价（必须知道）：本仓库自带数据集（`数据集/Face`）上的三分类准确率从 97.2% 降到 ~84% ——
 * 但那个数据集的 `sad` 大量是**婴幼儿嚎哭**，在 8 类模型眼里是 Fear/Surprise，
 * **它衡量的不是"成年人难过"**。完整证据与取舍见 `docs/fer_external_model_integration.md`。
 *
 * ## 规格事实（每一项都实测过，不是抄 README）
 *
 * | 项 | 值 | 怎么确认的 |
 * |---|---|---|
 * | 输入 | `input` [b,3,224,224]，RGB | onnxruntime introspection |
 * | 归一化 | `x/255` 后 ImageNet mean/std | 不加归一化 84.3% → 75.8%（本地域） |
 * | 类别顺序 | 见下 `labels` | `fer_external_eval.py --probe-labels` 穷举 336 种映射，最佳排列与之一致 |
 * | 输出 | `output` [b,8] | introspection |
 *
 * SHA-256 `fa07e841fd06c7a67ee651ea4e6e4a3a2bb5695f47b37a7da50492526f59c898`
 * （`client/public/models/face_emotion_8cls_hsemotion.onnx`，16,039,595 字节）
 */
export const FACE_EMOTION_SPEC: FaceEmotionModelSpec = {
  name: 'HSEmotion enet_b0_8_best_vgaf（8类 → 折叠三分类）',
  url: '/models/face_emotion_8cls_hsemotion.onnx',
  inputSize: 224,
  // ⚠️ 顺序 = 模型输出索引顺序。改这里之前必须先跑 --probe-labels 复核。
  labels: ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise'],
  // 折叠不到三类的一律 null：它们不进"三类内部的占比"，
  // 但会如实计入 offMapProbability，界面显示「其他表情 X%」，
  // 不假装"三类已经穷尽了所有可能"。
  fold: [null, null, null, null, 'happy', 'neutral', 'sad', null],
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
  channels: 3,
  bytes: 16_039_595,
};

/**
 * wasm 运行时目录的**兜底值**。
 * 正常路径是用 `versionedWasmPaths()` 按运行时实际加载到的 ORT 版本拼 CDN 地址，
 * 只有取不到版本号时才用这个写死的值（避免 package.json 的 `^` 范围升级后
 * glue 与 wasm 版本不一致而静默失效）。
 */
export const FACE_EMOTION_WASM_PATHS = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

export type FaceEmotionResult = FaceEmotionReading;

// ── 由规格派生的兼容常量（外部引用不必跟着改）──

/**
 * 输入边长。
 *
 * 实测（`scripts/face_emotion_accuracy_check.py`）：数据集所有图片原生就是 96×96，
 * 训练脚本没有 resize，所以当前模型只看过 96×96 —— 喂 224 会把准确率从 95.2%
 * 打到 40.6%。**换模型时这个值跟着 `FACE_EMOTION_SPEC.inputSize` 走**。
 */
export const FACE_EMOTION_INPUT_SIZE = FACE_EMOTION_SPEC.inputSize;

/** 当前模型的静态资源路径 */
export const FACE_EMOTION_MODEL_URL = FACE_EMOTION_SPEC.url;

/** 当前产物的字节数（用于如实展示"要下载多少"） */
export const FACE_EMOTION_MODEL_BYTES = FACE_EMOTION_SPEC.bytes ?? 0;

/**
 * 把一张 RGBA 图打包成模型输入 `[1, channels, size, size]` 的 NCHW float32。
 *
 * 纯函数（可在 Node 下测）：生产路径里 `FacePatchExtractor` 已经把画面裁成
 * `size×size`，这里只做通道序、缩放与归一化；尺寸不符时用**最近邻**兜底，
 * 保证函数在任何输入下都有确定行为（可测试、可复现）。
 *
 * 前两个参数保持历史签名（`packFaceImage(img)` / `packFaceImage(img, 96)` 仍然可用），
 * 第三个参数才是模型规格。默认规格 = 当前线上模型（RGB + 只 /255）。
 */
export function packFaceImage(
  img: RgbaImage,
  size: number = FACE_EMOTION_INPUT_SIZE,
  spec: Pick<FaceEmotionModelSpec, 'mean' | 'std' | 'channels' | 'grayscale' | 'bgr'> = FACE_EMOTION_SPEC,
): Float32Array {
  const channels = spec.channels ?? 3;
  const out = new Float32Array(channels * size * size);
  const sw = img.width;
  const sh = img.height;
  if (sw <= 0 || sh <= 0) return out;

  const plane = size * size;
  const mean = spec.mean ?? null;
  const std = spec.std ?? null;
  const bgr = spec.bgr === true;
  // 通道序：默认 RGB（当前模型的训练口径），BGR 时 0↔2 互换
  const c0 = bgr ? 2 : 0;
  const c2 = bgr ? 0 : 2;
  for (let y = 0; y < size; y++) {
    const sy = Math.min(sh - 1, Math.floor(((y + 0.5) * sh) / size));
    for (let x = 0; x < size; x++) {
      const sx = Math.min(sw - 1, Math.floor(((x + 0.5) * sw) / size));
      const si = (sy * sw + sx) * 4;
      const di = y * size + x;
      let r = img.data[si] / 255;
      let g = img.data[si + 1] / 255;
      let b = img.data[si + 2] / 255;
      if (spec.grayscale) {
        // Rec.601 亮度（ONNX model zoo 的 FER+ 预处理口径）
        const l = 0.299 * r + 0.587 * g + 0.114 * b;
        r = l;
        g = l;
        b = l;
      }
      if (channels === 1) {
        // 单通道输入：取灰度（未声明 grayscale 时按 Rec.601 折算，行为确定）
        const l = spec.grayscale ? r : 0.299 * r + 0.587 * g + 0.114 * b;
        out[di] = normalize(l, mean, std, 0);
        continue;
      }
      out[di] = normalize(r, mean, std, c0);
      out[plane + di] = normalize(g, mean, std, 1);
      out[2 * plane + di] = normalize(b, mean, std, c2);
    }
  }
  return out;
}

/** `(x - mean[c]) / std[c]`；mean/std 缺省时**只做 x/255**（不做任何归一化）。 */
function normalize(v: number, mean: readonly number[] | null, std: readonly number[] | null, c: number): number {
  if (!mean || !std) return v;
  const m = mean[c] ?? mean[0] ?? 0;
  const s = std[c] ?? std[0] ?? 1;
  return (v - m) / (s === 0 ? 1 : s);
}

/** 数值稳定的 softmax（模型输出是 logits，不是概率） */
export function softmax(logits: ArrayLike<number>): number[] {
  const n = logits.length;
  const out = new Array<number>(n).fill(0);
  if (n === 0) return out;
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (Number.isFinite(logits[i]) && logits[i] > max) max = logits[i];
  if (!Number.isFinite(max)) return out;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = Math.exp(logits[i] - max);
    out[i] = v;
    sum += v;
  }
  if (sum <= 0) return out;
  for (let i = 0; i < n; i++) out[i] /= sum;
  return out;
}

/**
 * 模型输出 → 展示三分类概率 + 主导类。
 *
 * 两步，都不能省：
 *   1. **按模型自己的类名**取第 i 个输出（`spec.labels[i]`），不要写死
 *      `happy=probs[0]` —— 历史上正是这条写死导致"只改标签数组修不好"（实测仍 40%）。
 *   2. **折叠**到展示三分类并重新归一化，使上屏的三个百分比仍然加满 100%。
 *      折叠不进去的概率（8 类模型的 fear/surprise/anger…）不丢，放进
 *      `offMapProbability` 如实上报。
 */
export function toReading(
  logits: ArrayLike<number>,
  extra: { latencyMs: number; inputSize: number; sampledAt: number },
  spec: FaceEmotionModelSpec = FACE_EMOTION_SPEC,
): FaceEmotionResult {
  const probs = softmax(logits);
  const folded: Record<FaceEmotionLabel, number> = { happy: 0, neutral: 0, sad: 0 };
  let offMap = 0;
  let mapped = 0;
  for (let i = 0; i < spec.labels.length; i++) {
    const p = probs[i] ?? 0;
    const target = spec.fold[i] ?? null;
    if (target) {
      folded[target] += p;
      mapped += p;
    } else {
      offMap += p;
    }
  }
  // 归一化到"三分类内部"的占比；三分类全为 0 时保持 0（界面会显示"识别中…"）
  const probabilities = {} as Record<FaceEmotionLabel, number>;
  for (const label of FACE_EMOTION_LABELS) {
    probabilities[label] = mapped > 0 ? folded[label] / mapped : 0;
  }
  let dominant: FaceEmotionLabel = FACE_EMOTION_LABELS[0];
  let confidence = -1;
  for (const label of FACE_EMOTION_LABELS) {
    if (probabilities[label] > confidence) {
      confidence = probabilities[label];
      dominant = label;
    }
  }
  const round = (v: number) => Math.round(v * 1000) / 1000;
  return {
    probabilities: {
      happy: round(probabilities.happy),
      neutral: round(probabilities.neutral),
      sad: round(probabilities.sad),
    },
    dominant,
    confidence: round(Math.max(0, confidence)),
    offMapProbability: round(Math.max(0, offMap)),
    latencyMs: Math.round(extra.latencyMs * 10) / 10,
    inputSize: extra.inputSize,
    sampledAt: extra.sampledAt,
  };
}

export interface FaceEmotionClassifierOptions {
  /** 模型地址，默认走前端静态目录 */
  modelUrl?: string;
  /** onnxruntime-web 的 wasm 目录（默认 jsDelivr 固定版本，与 package.json 对齐） */
  wasmPaths?: string;
  /** 输入边长（默认 `FACE_EMOTION_SPEC.inputSize`） */
  inputSize?: number;
  /**
   * 模型规格（默认 `FACE_EMOTION_SPEC`）。传入即按该规格做预处理与折叠，
   * 因此**外部 FER 模型（8 类 / 224px / ImageNet 归一化）不需要改本文件**。
   */
  spec?: FaceEmotionModelSpec;
  /**
   * 是否优先尝试 WebGPU。90 MB 的 ResNet50 在单线程 wasm 上约 1~3 s/次，
   * WebGPU 可用时会快一个数量级；不可用则自动回退 wasm（不影响可用性，只影响频率）。
   */
  preferWebGPU?: boolean;
  /**
   * 是否允许**主线程** wasm 兜底（默认 false）。
   *
   * 为什么默认关：单线程 wasm 在调用线程上执行，一次 224px 前向约 300~400 ms，
   * 会把 JS 主线程卡住、`setInterval` 排队，直接损害眨眼通道的时长测量。
   * 正常情况下走 `webgpu` 或 `wasm` + Worker 代理（`env.wasm.proxy`）都不占主线程；
   * 只有在"没有 WebGPU 且 Worker 代理也建不起来"（例如 CSP 禁了 blob worker）时
   * 才会需要它 —— 那种环境下**宁可本模态不可用，也不拖累眨眼主链路**。
   */
  allowMainThreadWasm?: boolean;
}

/**
 * 用**运行时真正加载到的 ORT 版本**拼 wasm 的 CDN 路径。
 *
 * 为什么不写死 `onnxruntime-web@1.30.0`：package.json 里是 `^1.30.0`，
 * 一旦将来装到 1.31/2.x，glue 代码是新版、wasm 还是旧版，两者不匹配会静默降级为
 * "模型不可用"。用 `ort.env.versions.web` 取实际版本就不会漂移。
 */
export function versionedWasmPaths(ort: any): string | null {
  const v = ort?.env?.versions?.web;
  return typeof v === 'string' && v.length > 0
    ? `https://cdn.jsdelivr.net/npm/onnxruntime-web@${v}/dist/`
    : null;
}

/**
 * 懒加载的表情分类器。任何一步失败都不会抛到调用方 ——
 * 返回 `null` 由上层降级（本模块只做增强，绝不影响眨眼主链路）。
 */
export class FaceEmotionClassifier {
  private session: unknown = null;
  private loading: Promise<boolean> | null = null;
  /** 实际用上的执行后端（用于界面/日志如实说明"跑在什么上面"） */
  private provider: string | null = null;
  /**
   * 是否已被释放。用来堵住"边下载边关闭"的泄漏：
   * `InferenceSession.create()` 解析完（= 90 MB 下完）之后如果组件已经关了，
   * 必须把刚建好的会话立刻释放，否则那 90 MB 会话再也没人回收。
   */
  private disposed = false;
  /** 调用方显式指定的 wasm 路径（不指定时才用版本推导/兜底值） */
  private readonly explicitWasmPaths: string | null;
  /**
   * 上一次加载失败的原因（一句话，给界面用）。
   *
   * 存在的理由：界面原来只会说「模型文件下载失败或浏览器不支持 ONNX」，
   * 而实际病因至少有五种（wasm CDN 不可达 / dev server 依赖失效 / 模型 404 /
   * 浏览器不支持 wasm SIMD / WebGPU 建会话失败）。不留原因，用户问了"为什么"
   * 没人答得上来。
   */
  private lastError: string | null = null;
  private readonly options: Required<FaceEmotionClassifierOptions>;

  constructor(options: FaceEmotionClassifierOptions = {}) {
    this.explicitWasmPaths = options.wasmPaths ?? null;
    // 规格是"模型事实"的唯一来源：地址、输入边长、归一化、类别折叠都从这里取，
    // 显式传入的 modelUrl/inputSize 仍然优先（回归自检脚本靠它故意造错配）。
    const spec = options.spec ?? FACE_EMOTION_SPEC;
    this.options = {
      modelUrl: options.modelUrl ?? spec.url ?? FACE_EMOTION_MODEL_URL,
      wasmPaths: options.wasmPaths ?? FACE_EMOTION_WASM_PATHS,
      inputSize: options.inputSize ?? spec.inputSize ?? FACE_EMOTION_INPUT_SIZE,
      preferWebGPU: options.preferWebGPU ?? true,
      allowMainThreadWasm: options.allowMainThreadWasm ?? false,
      spec,
    };
  }

  /** 本会话使用的模型规格（预处理与类别折叠都按它走） */
  get spec(): FaceEmotionModelSpec {
    return this.options.spec;
  }

  /** 上一次加载失败的一句话原因；从未失败过时为 null */
  get unavailableDetail(): string | null {
    return this.lastError;
  }

  get isReady(): boolean {
    return this.session !== null;
  }

  /** 实际执行后端：`webgpu` / `wasm` / null（未加载） */
  get executionProvider(): string | null {
    return this.provider;
  }

  get inputSize(): number {
    return this.options.inputSize;
  }

  /** 懒加载会话；重复调用只加载一次。失败返回 false（不抛） */
  async load(): Promise<boolean> {
    if (this.disposed) return false;
    if (this.session) return true;
    if (this.loading) return this.loading;
    this.loading = (async () => {
      try {
        const ort: any = await import('onnxruntime-web');
        if (ort.env?.wasm) {
          // 显式指定 > 按实际 ORT 版本推导 > 写死兜底（避免 glue/wasm 版本漂移）
          ort.env.wasm.wasmPaths =
            this.explicitWasmPaths ?? versionedWasmPaths(ort) ?? this.options.wasmPaths;
          // 多线程 wasm 需要 SharedArrayBuffer（页面必须带 COOP/COEP 响应头）。
          // 本仓库的 dev/prod 都没有这两个头，所以默认单线程；能隔离时才用多线程。
          const isolated = typeof (globalThis as any).crossOriginIsolated === 'boolean'
            ? (globalThis as any).crossOriginIsolated === true
            : false;
          ort.env.wasm.numThreads = isolated
            ? Math.max(1, Math.min(4, ((globalThis as any).navigator?.hardwareConcurrency as number) || 1))
            : 1;
        }
        console.log(
          `[表情模型] ORT ${ort?.env?.versions?.web ?? '?'} 就绪；`
          + `wasm 目录 = ${ort?.env?.wasm?.wasmPaths ?? '(未设置)'}；`
          + `executionProviders 本次将依次尝试：`
          + (this.options.preferWebGPU ? 'webgpu → ' : '') + 'wasm(worker 代理)'
          + (this.options.allowMainThreadWasm ? ' → wasm(主线程)' : ''),
        );

        // 依次尝试：WebGPU → wasm(Worker 代理) → [可选] wasm(主线程)。
        //
        // 为什么非要试「Worker 代理」：这是一个 90 MB 的 ResNet50，单线程 wasm 跑一次
        // 可能上千毫秒，而**单线程 wasm 是在调用线程上执行的** —— 直接跑就会把主线程
        // 卡住，`setInterval` 排队、眨眼时间戳被推后，把主链路（眨眼时长）一起搞坏。
        // `proxy = true` 让 onnxruntime-web 把会话放进 Web Worker，主线程只等 Promise。
        // WebGPU 走 GPU，天然不占主线程，所以优先。
        const attempts: Array<{ label: string; providers: string[]; proxy: boolean }> = [
          ...(this.options.preferWebGPU
            ? [{ label: 'webgpu', providers: ['webgpu'], proxy: false }]
            : []),
          { label: 'wasm-proxy', providers: ['wasm'], proxy: true },
          // 主线程兜底默认关闭：宁可本模态不可用，也不让 30fps 眨眼通道排队（见选项注释）
          ...(this.options.allowMainThreadWasm
            ? [{ label: 'wasm-main-thread', providers: ['wasm'], proxy: false }]
            : []),
        ];

        let lastErr: unknown = null;
        let created: unknown = null;
        /** 每一路后端失败的一句话原因；全失败时拼给界面，回答"为什么不可用" */
        const failures: string[] = [];
        for (const attempt of attempts) {
          try {
            if (ort.env?.wasm) ort.env.wasm.proxy = attempt.proxy;
            created = await ort.InferenceSession.create(this.options.modelUrl, {
              executionProviders: attempt.providers,
            });
            this.provider = attempt.label;
            if (attempt.label === 'wasm-main-thread') {
              console.warn('[表情模型] ⚠️ 正在用主线程 wasm 推理，可能影响眨眼时间戳精度');
            }
            break;
          } catch (err) {
            lastErr = err;
            created = null;
            const msg = err instanceof Error ? err.message : String(err);
            failures.push(`${attempt.label}: ${msg}`);
            console.warn(`[表情模型] ${attempt.label} 建会话失败，尝试下一种后端：`, err);
          }
        }
        if (!created) {
          this.lastError = failures.length
            ? `所有执行后端都失败 —— ${failures.join('；')}`
            : '所有执行后端都不可用';
          throw lastErr ?? new Error(this.lastError);
        }

        // 「边下载边关闭」：会话是在 await 之后才建好的，此时若已 dispose，
        // 必须立刻释放，不能挂到 this.session 上（那之后就没人回收了）。
        if (this.disposed) {
          try {
            (created as any)?.release?.();
          } catch {
            /* 释放失败不影响功能 */
          }
          this.provider = null;
          return false;
        }
        this.session = created;

        console.log(
          `[表情模型] ✅ ${this.options.spec.name} 已加载（${(
            (this.options.spec.bytes ?? FACE_EMOTION_MODEL_BYTES) / 1024 / 1024
          ).toFixed(1)} MB，` +
          `输入 ${this.options.inputSize}px，${this.options.spec.labels.length} 类 → ` +
          `折叠为 ${FACE_EMOTION_LABELS.join('/')}，后端 ${this.provider}）`,
        );
        return true;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        // 把"为什么不可用"留下来给界面用。原来这里只有一句
        // 「模型文件下载失败或浏览器不支持 ONNX」，把 5 种完全不同的病因
        // （CDN 不可达 / dev server 依赖失效 / 浏览器不支持 wasm SIMD / 模型 404 /
        //   WebGPU 建会话失败）压成同一句话，用户和我们都无从下手。
        this.lastError = msg;
        console.warn('[表情模型] ⚠️ 加载失败，本模态如实上报为不可用：', err);
        this.session = null;
        this.provider = null;
        return false;
      } finally {
        this.loading = null;
      }
    })();
    return this.loading;
  }

  /**
   * 对一张人脸裁剪图跑一次推理。
   *
   * 模型输入是固定 batch 1（图里 batch 维虽然标了动态，运行时仍按 1 喂），
   * 所以这里一次只处理一张。
   */
  async classify(img: RgbaImage): Promise<FaceEmotionResult | null> {
    const ok = await this.load();
    if (!ok || !this.session) return null;
    const spec = this.options.spec;
    const size = this.options.inputSize;
    const channels = spec.channels ?? 3;
    const t0 = performance.now();
    try {
      const ort: any = await import('onnxruntime-web');
      const session: any = this.session;
      const inputName = session.inputNames?.[0] ?? 'input';
      const tensor = new ort.Tensor('float32', packFaceImage(img, size, spec),
        [1, channels, size, size]);
      const out = await session.run({ [inputName]: tensor });
      const key = Object.keys(out)[0];
      const logits = out[key].data as Float32Array;
      // 输出维度必须等于**该模型自己的类别数**（不是展示的 3 类）
      if (!logits || logits.length !== spec.labels.length) {
        console.warn(
          `[表情模型] 输出维度异常（期望 ${spec.labels.length}，实得 ${logits?.length}），本帧跳过：` +
          '规格与模型不匹配，请核对 FACE_EMOTION_SPEC.labels',
        );
        return null;
      }
      return toReading(logits, {
        latencyMs: performance.now() - t0,
        inputSize: size,
        sampledAt: Date.now(),
      }, spec);
    } catch (err) {
      console.warn('[表情模型] 推理失败，本帧跳过：', err);
      return null;
    }
  }

  /** 释放会话（关闭多模态 / 组件卸载时调用，回收约 90 MB 的 wasm 堆） */
  dispose(): void {
    this.disposed = true;
    const session: any = this.session;
    this.session = null;
    this.provider = null;
    try {
      session?.release?.();
    } catch {
      /* 释放失败不影响功能 */
    }
  }
}
