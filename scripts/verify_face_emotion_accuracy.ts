/**
 * 端到端**精度**复核：用真实数据集图像跑**真正的前端代码**。
 *
 * 与 `face_emotion_accuracy_check.py` 的分工：
 *   - Python 脚本测的是 **ONNX 本身**（onnxruntime 原生），负责扫输入尺寸与类别排列；
 *   - 本脚本测的是 **前端这一层**：`FaceEmotionClassifier.classify` →
 *     `packFaceImage` → `toReading` → `FACE_EMOTION_LABELS` 的索引→标签映射。
 *
 * 为什么必须单独测这一层：`tsc` 只能保证类型对，**保证不了映射语义对**。
 * 2026-10-10 的实测里，`toReading` 曾经把 `probs[0]` 写死成 happy，
 * 只改 `FACE_EMOTION_LABELS` 是修不好的 —— 那种错误只有跑真实数据才能发现。
 *
 * 数据来源：先用 Python 把图像导成裸 RGBA（Node 侧没有图像解码库）：
 *   & "D:\Develop\AIC\.venv\Scripts\python.exe" scripts/face_emotion_accuracy_check.py \
 *       --dump-rgba .tmp-face-rgba --dump-limit 200
 *
 * 然后：
 *   node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_face_emotion_accuracy.ts \
 *       --data .tmp-face-rgba
 */
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  FaceEmotionClassifier,
  FACE_EMOTION_INPUT_SIZE,
  FACE_EMOTION_LABELS,
  type FaceEmotionModelSpec,
  type RgbaImage,
} from '../client/src/services/faceEmotionClassifier';

const here = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(here, '..');
/**
 * 默认用前端真正加载的那份模型；`--model` 可换成候选模型 ——
 * **换模型必须先在这里跑通再覆盖 `client/public/models/`**（见 docs/face_fix_diagnosis.md §3）。
 *
 * ⚠️ 2026-10-10 起线上是 HSEmotion 8 类（15.3 MB）。本脚本测的是**本仓库数据集**，
 * 而该数据集的 `sad` 大量是婴幼儿嚎哭，8 类模型把它读成 Fear/Surprise ——
 * 所以这里的数字只能用来确认"规格与折叠没配错"（门槛 60%，40%/33% 那些错配都远低于它），
 * **不能**用来判断"成年人难过识别得好不好"。
 * 后者看 `docs/fer_external_model_integration.md`（RAF-DB 成人集，sad AUC 0.965 vs 0.630）。
 */
const defaultModelPath = path.join(appRoot, 'client', 'public', 'models', 'face_emotion_8cls_hsemotion.onnx');
/** 浏览器里 wasm 走 CDN；Node 的 ESM loader 不收 https，改用本地同一份运行时 */
const localWasmPaths =
  pathToFileURL(path.join(appRoot, 'client', 'node_modules', 'onnxruntime-web', 'dist')).href + '/';

interface Manifest {
  classes: string[];
  images: { file: string; width: number; height: number; label: string; source: string }[];
}

function argValue(flag: string, fallback: string): string {
  const i = process.argv.indexOf(flag);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

async function main() {
  const dataDir = path.resolve(argValue('--data', '.tmp-face-rgba'));
  const modelPath = path.resolve(argValue('--model', defaultModelPath));
  /**
   * `--spec <json>`：用**外部 FER 模型自己的规格**跑同一条前端代码路径。
   *
   * 为什么必须能这样测：接外部模型时最容易错的三件事（输入尺寸、归一化、类别顺序）
   * 都发生在 `packFaceImage` / `toReading` 这一层。只测 ONNX 是测不到它们的 ——
   * 必须让 `FaceEmotionClassifier` 本尊去跑，测的才是"线上真正会发生的事"。
   * JSON 字段与 `FaceEmotionModelSpec` 一致。
   */
  const specPath = argValue('--spec', '');
  let spec: FaceEmotionModelSpec | undefined;
  if (specPath) {
    const rawSpec = JSON.parse(readFileSync(path.resolve(specPath), 'utf-8'));
    spec = {
      ...rawSpec,
      url: rawSpec.url ?? modelPath,
      // `fold` 缺省时按类名同名折叠（neutral/happy/sad 直接对号入座）
      fold: rawSpec.fold ?? (rawSpec.labels as string[]).map((l: string) => (
        ['neutral', 'happy', 'sad'].includes(l.toLowerCase()) ? l.toLowerCase() : null
      )),
    } as FaceEmotionModelSpec;
    console.log(`[规格] ${specPath}`);
    console.log(`  ${spec.name}  输入 ${spec.inputSize}px  类别=${spec.labels.join('/')}`);
  }
  const manifestPath = path.join(dataDir, 'images.json');
  console.log('='.repeat(78));
  console.log(`[模型] ${modelPath}`);
  console.log(`[前端常量] FACE_EMOTION_INPUT_SIZE = ${spec ? spec.inputSize : FACE_EMOTION_INPUT_SIZE}`);
  console.log(`[前端常量] FACE_EMOTION_LABELS     = [${FACE_EMOTION_LABELS.join(', ')}]（展示口径，固定）`);
  if (spec) {
    console.log(`[规格] 模型自己的类别 → 展示类：` +
      spec.labels.map((l, i) => `${l}→${spec.fold[i] ?? '其他'}`).join(', '));
  } else {
    console.log('[规格] 即 输出索引 i → 上述数组第 i 项');
  }
  if (!existsSync(manifestPath)) {
    throw new Error(`找不到 ${manifestPath}\n请先用 face_emotion_accuracy_check.py --dump-rgba 导出。`);
  }
  if (!existsSync(modelPath)) throw new Error(`模型不存在：${modelPath}`);

  const manifest = JSON.parse(readFileSync(manifestPath, 'utf-8')) as Manifest;
  console.log(`[数据] ${dataDir}  共 ${manifest.images.length} 张  类别=${manifest.classes.join(', ')}`);

  // ── 只用于「回归自检」：故意把输入尺寸改回历史上有 bug 的那一版（224），
  //    验证本脚本确实能发现问题，而不是恒过。正常复核**不要传**。
  //    （标签顺序无法在运行时覆盖——那正是 `toReading` 的单一口径；
  //      类别排列由 Python 脚本的 6 种排列扫描负责。）
  const forceSize = argValue('--force-input-size', '');
  const effectiveSize = forceSize ? Number(forceSize) : (spec ? spec.inputSize : FACE_EMOTION_INPUT_SIZE);
  if (forceSize) {
    console.log(`\n⚠️ 回归自检模式：inputSize=${effectiveSize}（**不是**发布配置，仅用于验证本脚本能识别错配）`);
  }

  const classifier = new FaceEmotionClassifier({
    modelUrl: modelPath,
    inputSize: effectiveSize,
    spec,
    preferWebGPU: false,
    wasmPaths: localWasmPaths,
  });
  const ok = await classifier.load();
  if (!ok) throw new Error('ORT Web 加载模型失败');
  console.log(`[模型] 已加载 | 后端=${classifier.executionProvider}`);

  // 混淆矩阵：行=真实类别，列=前端给出的标签
  const labels = [...FACE_EMOTION_LABELS];
  const classes = manifest.classes;
  const conf: Record<string, Record<string, number>> = {};
  for (const c of classes) {
    conf[c] = {};
    for (const l of labels) conf[c][l] = 0;
  }

  let correct = 0;
  let failed = 0;
  let latencySum = 0;
  /** 折叠不进三分类的概率总量（外部 7/8 类模型的 fear/surprise/… 去哪了） */
  let offMapSum = 0;
  let offMapN = 0;
  const t0 = Date.now();
  for (const item of manifest.images) {
    const buf = readFileSync(path.join(dataDir, item.file));
    const img: RgbaImage = {
      // 复制一份，避免受 Buffer 内存池影响
      data: new Uint8ClampedArray(buf),
      width: item.width,
      height: item.height,
    };
    const result = await classifier.classify(img);
    if (!result) {
      failed += 1;
      continue;
    }
    latencySum += result.latencyMs;
    offMapSum += result.offMapProbability ?? 0;
    offMapN += 1;
    const predicted = result.dominant;
    conf[item.label][predicted] = (conf[item.label][predicted] ?? 0) + 1;
    if (predicted === item.label) correct += 1;
  }
  classifier.dispose();

  const n = manifest.images.length - failed;
  console.log('\n' + '='.repeat(78));
  console.log(`准确率 = ${((correct / n) * 100).toFixed(1)}%  (${correct}/${n}` +
    `${failed ? `，${failed} 张推理失败` : ''})`);
  console.log(`平均单张推理 ${(latencySum / Math.max(1, n)).toFixed(1)} ms | ` +
    `总耗时 ${((Date.now() - t0) / 1000).toFixed(1)} s`);
  if (spec) {
    const meanOffMap = offMapSum / Math.max(1, offMapN);
    console.log(`折叠不进三分类的平均概率质量 = ${(meanOffMap * 100).toFixed(1)}%` +
      `（外部模型多出来的 fear/surprise/anger 等类；` +
      `${meanOffMap > 0.5 ? '❗超过一半的判定落在三类之外，折叠会严重失真' : '在可接受范围'}）`);
  }
  console.log(`\n混淆矩阵（行=真实，列=前端给出的标签）`);
  console.log('        ' + labels.map((l) => l.padStart(9)).join(''));
  for (const c of classes) {
    const row = labels.map((l) => String(conf[c][l]).padStart(9)).join('');
    const total = labels.reduce((s, l) => s + conf[c][l], 0);
    const recall = total ? ((conf[c][c] ?? 0) / total) * 100 : 0;
    console.log(c.padStart(8) + row + `   recall ${recall.toFixed(1)}%`);
  }
  console.log('='.repeat(78));

  // 前端常量与数据集类别对不上时直接判失败：这正是 40% 那一版的形态
  const mismatch = classes.filter((c) => !labels.includes(c));
  if (mismatch.length) {
    console.log(`⚠️ 数据集类别 ${mismatch.join(', ')} 不在前端 FACE_EMOTION_LABELS 里，` +
      `这些样本必然算错（模型没有对应输出）。`);
  }
  /**
   * 门槛的作用是**发现规格错配**，不是给模型打分。
   *
   * 实测的三种错配形态：输入尺寸错 40.6%、类别顺序错 40.6%、归一化错 33.3%。
   * 所以 60% 足够把它们全部拦下，同时不会因为"换了一个在本数据集上天生更弱的模型"
   * 而误报 —— 比如 8 类 HSEmotion 在这里约 84%（它把本数据集的婴幼儿嚎哭读成 Fear）。
   * 60%~90% 之间额外提示一句，说明这个数字偏低可能是数据域问题而非配置问题。
   */
  const acc = correct / n;
  if (acc < 0.6) {
    console.log('❌ 准确率低于 60%：前端常量（尺寸/排列/归一化）很可能与模型不一致，' +
      '请跑 Python 脚本重扫（face_emotion_accuracy_check.py / fer_external_eval.py --probe-labels）。');
    process.exitCode = 1;
  } else if (acc < 0.9) {
    console.log('✅ 前端代码路径（预处理 + 折叠 + 索引→标签映射）与模型自洽；' +
      '准确率偏低属**数据域**差异，不是配置问题 —— 见 docs/fer_external_model_integration.md。');
  } else {
    console.log('✅ 前端代码路径（预处理 + 折叠 + 索引→标签映射）与模型一致。');
  }
}

main().catch((err) => {
  console.error('[复核失败]', err);
  process.exit(1);
});
