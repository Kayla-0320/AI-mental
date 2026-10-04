/**
 * 浏览器内「睁眼 / 闭眼」每眼分类器（ONNX Runtime Web）
 *
 * ## 它是什么、不是什么
 *
 * 这是**独立的第二意见**，不是眨眼状态机的替代品：
 *   - 输入：`eyePatch` 裁出的左右眼 patch（复用 MediaPipe 同一帧的关键点，不新开摄像头）
 *   - 输出：每只眼的「闭眼概率」（0–1），与 MediaPipe `eyeBlinkLeft/Right` 语义同级
 *   - 时序判定仍然由 `blinkLogic.BlinkStateMachine` 负责（迟滞 + 时长窗 + 最小间隔）
 *
 * ## 模型与许可证
 *
 * `open-closed-eye-0001`（OpenVINO Open Model Zoo，**Apache-2.0**，46 164 B，SHA-384 已校验）：
 *   - 输入 `input.1`：`[1, 3, 32, 32]`，**BGR** 通道序，归一化 `(x − 127) / 255`
 *   - 输出 `19`：`[1, 2, 1, 1]` softmax
 *   - ⚠️ **类别顺序按实测取 index 0 = 闭眼**：官方 model card 写的是 `[open, closed]`，
 *     但在 RT-BENE 标注 patch 上实测 index 0 的 AUC 为 0.918、index 1 为 0.082
 *     （= 1 − 0.918，完全反向）。详见 `docs/blink_signal_source_selection.md` §3.3。
 *
 * ## 隐私
 *
 * 全程在本机 canvas + WASM 内完成，**不传帧、不落盘**。模型文件 46 KB，随前端一起发布。
 *
 * ## 为什么默认不启用
 *
 * 首次使用需要下载 onnxruntime-web 的 wasm 运行时（约 13.6 MB，之后浏览器缓存）。
 * 这对"只加 46 KB 模型"来说是一笔真实成本，因此由
 * `perceptionCapabilities.EYE_STATE_MODEL_AVAILABLE`（env 开关）控制，
 * 默认关闭；启用前请先按 `docs/blink_signal_source_selection.md` 的步 B 做阈值标定。
 */

/** 结构化的 RGBA 图像（浏览器里就是 ImageData；Node 测试里可以是普通对象） */
export interface RgbaImage {
  data: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
}

/** 模型输入规格 */
export const EYE_STATE_INPUT = {
  width: 32,
  height: 32,
  /** OpenVINO model.yml：--mean_values=[127,127,127] --scale_values=[255,255,255] */
  mean: 127,
  scale: 255,
  /** 官方要求 BGR 通道序 */
  channelOrder: 'BGR' as const,
};

/** 「闭眼」在输出向量中的下标（实测结论，勿照抄文档） */
export const CLOSED_CLASS_INDEX = 0;

/**
 * 把一张 RGBA 图缩放到 32×32 并打包成 NCHW float32（纯函数，可在 Node 下测）。
 *
 * 用**最近邻**而不是双线性：32×32 的输入本来就没有细节可言，
 * 而最近邻没有浮点插值开销，且行为完全确定（可测试、可复现）。
 */
export function packEyePatch(img: RgbaImage): Float32Array {
  const { width: W, height: H } = EYE_STATE_INPUT;
  const out = new Float32Array(3 * W * H);
  const sw = img.width;
  const sh = img.height;
  if (sw <= 0 || sh <= 0) return out;

  const plane = W * H;
  for (let y = 0; y < H; y++) {
    const sy = Math.min(sh - 1, Math.floor((y + 0.5) * sh / H));
    for (let x = 0; x < W; x++) {
      const sx = Math.min(sw - 1, Math.floor((x + 0.5) * sw / W));
      const si = (sy * sw + sx) * 4;
      const r = img.data[si];
      const g = img.data[si + 1];
      const b = img.data[si + 2];
      const di = y * W + x;
      // BGR 通道序 + (x − 127)/255
      out[di] = (b - EYE_STATE_INPUT.mean) / EYE_STATE_INPUT.scale;
      out[plane + di] = (g - EYE_STATE_INPUT.mean) / EYE_STATE_INPUT.scale;
      out[2 * plane + di] = (r - EYE_STATE_INPUT.mean) / EYE_STATE_INPUT.scale;
    }
  }
  return out;
}

/** 从模型输出取「闭眼概率」 */
export function closedProbability(output: ArrayLike<number>): number {
  const p = output[CLOSED_CLASS_INDEX];
  return Number.isFinite(p) ? p : 0;
}

export interface EyeStateResult {
  /** 图像左侧眼的闭眼概率 */
  closedLeft: number;
  /** 图像右侧眼的闭眼概率 */
  closedRight: number;
  /** 两眼中较小的那个 —— 与线上 `min(eyeBlinkLeft, eyeBlinkRight)` 口径对齐，便于比较 */
  closedBoth: number;
  /** 本次推理耗时 ms */
  latencyMs: number;
}

export interface EyeStateClassifierOptions {
  /** 模型地址，默认走前端静态目录 */
  modelUrl?: string;
  /** onnxruntime-web 的 wasm 目录（默认 jsDelivr 固定版本，与 package.json 对齐） */
  wasmPaths?: string;
}

/**
 * 懒加载的每眼分类器。任何一步失败都不会抛到调用方 ——
 * 返回 `null` 由上层降级回 blendshape（本模块只做增强，绝不影响主链路可用性）。
 */
export class EyeStateClassifier {
  private session: unknown = null;
  private loading: Promise<boolean> | null = null;
  private readonly options: Required<EyeStateClassifierOptions>;

  constructor(options: EyeStateClassifierOptions = {}) {
    this.options = {
      modelUrl: options.modelUrl ?? '/models/open_closed_eye.onnx',
      // 固定版本号而不是 @latest：模型输入/输出名字是与具体版本一起验证过的
      wasmPaths: options.wasmPaths
        ?? 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/',
    };
  }

  /** 是否已就绪（可用于界面显示"第二意见：可用/不可用"） */
  get isReady(): boolean {
    return this.session !== null;
  }

  /** 懒加载会话；重复调用只加载一次 */
  async load(): Promise<boolean> {
    if (this.session) return true;
    if (this.loading) return this.loading;
    this.loading = (async () => {
      try {
        const ort: any = await import('onnxruntime-web');
        ort.env.wasm.wasmPaths = this.options.wasmPaths;
        // 单线程：多线程 wasm 需要 SharedArrayBuffer，即页面必须带 COOP/COEP 响应头。
        // 46 KB 的模型单线程足够（实测 CPU 上 <1ms），不值得为它改部署头。
        if (ort.env?.wasm) ort.env.wasm.numThreads = 1;
        this.session = await ort.InferenceSession.create(this.options.modelUrl, {
          executionProviders: ['wasm'],
        });
        console.log('[眼状态模型] ✅ open-closed-eye-0001 已加载（46 KB，Apache-2.0）');
        return true;
      } catch (err) {
        console.warn('[眼状态模型] ⚠️ 加载失败，降级为只用 MediaPipe blendshape：', err);
        this.session = null;
        return false;
      } finally {
        this.loading = null;
      }
    })();
    return this.loading;
  }

  /**
   * 对左右眼 patch 各跑一次推理。
   *
   * 注意模型输入是**固定 batch 1**（`[1,3,32,32]`，ONNX 图里没有动态维），
   * 所以这里跑两次而不是拼成一个 batch=2 的张量。
   */
  async classify(left: RgbaImage, right: RgbaImage): Promise<EyeStateResult | null> {
    const ok = await this.load();
    if (!ok || !this.session) return null;
    const t0 = performance.now();
    try {
      const ort: any = await import('onnxruntime-web');
      const session: any = this.session;
      const inputName = session.inputNames?.[0] ?? 'input.1';
      const run = async (img: RgbaImage): Promise<number> => {
        const tensor = new ort.Tensor('float32', packEyePatch(img), [1, 3, 32, 32]);
        const out = await session.run({ [inputName]: tensor });
        const key = Object.keys(out)[0];
        return closedProbability(out[key].data as Float32Array);
      };
      const closedLeft = await run(left);
      const closedRight = await run(right);
      return {
        closedLeft,
        closedRight,
        closedBoth: Math.min(closedLeft, closedRight),
        latencyMs: Math.round((performance.now() - t0) * 10) / 10,
      };
    } catch (err) {
      console.warn('[眼状态模型] 推理失败，本帧跳过：', err);
      return null;
    }
  }

  /** 释放会话（组件卸载时调用） */
  dispose(): void {
    const session: any = this.session;
    this.session = null;
    try {
      session?.release?.();
    } catch {
      /* 释放失败不影响功能 */
    }
  }
}
