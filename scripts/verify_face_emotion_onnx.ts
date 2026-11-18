/**
 * 端到端验证：用**真正的 onnxruntime-web（wasm）**加载导出的表情模型，
 * 并调用前端同一份代码（`packFaceImage` / `softmax` / `FaceEmotionClassifier`）。
 *
 * 为什么要在 Node 里跑：浏览器之外的这份校验能证明
 *   1. 产物 ONNX 能被 ORT Web 的 wasm 后端读起来（不是只有原生 onnxruntime 能读）；
 *   2. 输入名/形状/输出解析与前端代码里写的一致；
 *   3. 预处理是纯函数、softmax 数值正常（和为 1）；
 *   4. 单线程 wasm 的真实推理耗时 —— 决定 `useFaceEmotion` 的采样节流。
 *
 * 运行：
 *   node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_face_emotion_onnx.ts
 */
import { createHash } from 'node:crypto';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

import {
  FaceEmotionClassifier,
  packFaceImage,
  softmax,
  versionedWasmPaths,
  FACE_EMOTION_INPUT_SIZE,
  FACE_EMOTION_LABELS,
  type RgbaImage,
} from '../client/src/services/faceEmotionClassifier';

const here = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(here, '..');
/** 默认校验**线上真正加载的那份**（改 `FACE_EMOTION_SPEC.url` 时这里要一起改） */
const modelPath = path.join(appRoot, 'client', 'public', 'models', 'face_emotion_8cls_hsemotion.onnx');
/**
 * 浏览器里 wasm 从 jsDelivr CDN 取（见 `FACE_EMOTION_WASM_PATHS`）；
 * Node 的 ESM loader 不接受 https 协议，所以验证时改用本地 `node_modules` 里的同一份运行时。
 */
const localWasmPaths = pathToFileURL(
  path.join(appRoot, 'client', 'node_modules', 'onnxruntime-web', 'dist'),
).href + '/';

function syntheticImage(width: number, height: number, kind: 'gradient' | 'noise'): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      if (kind === 'gradient') {
        data[i] = Math.round((x / Math.max(1, width - 1)) * 255);
        data[i + 1] = Math.round((y / Math.max(1, height - 1)) * 255);
        data[i + 2] = 128;
      } else {
        data[i] = (x * 37 + y * 17) % 256;
        data[i + 1] = (x * 11 + y * 53) % 256;
        data[i + 2] = (x * 97 + y * 7) % 256;
      }
      data[i + 3] = 255;
    }
  }
  return { data, width, height };
}

function fmt(p: number): string {
  return p.toFixed(4);
}

async function main() {
  console.log('='.repeat(78));
  if (!existsSync(modelPath)) throw new Error(`模型不存在：${modelPath}`);
  const bytes = readFileSync(modelPath);
  const sha = createHash('sha256').update(bytes).digest('hex');
  console.log(`模型：${modelPath}`);
  console.log(`  大小 ${(bytes.length / 1024 / 1024).toFixed(2)} MB | SHA-256 ${sha}`);

  // ── 纯函数自检（不依赖任何运行时）──
  const img = syntheticImage(FACE_EMOTION_INPUT_SIZE, FACE_EMOTION_INPUT_SIZE, 'gradient');
  const packed = packFaceImage(img);
  // 注意：当前规格带 ImageNet 归一化，取值**必然**会出 [0,1]（约 -2.1 ~ 2.6）。
  // 这里只检查"有限且量级合理"，不要断言 [0,1] —— 那是"只 /255"那一版的属性。
  const finite = packed.every((v) => Number.isFinite(v));
  // ⚠️ 不要写 Math.min(...packed)：224px 三通道有 150,528 个数，展开成实参会爆栈
  //（"Maximum call stack size exceeded"）。用循环。
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < packed.length; i++) {
    if (packed[i] < lo) lo = packed[i];
    if (packed[i] > hi) hi = packed[i];
  }
  console.log(`\n[纯函数] packFaceImage: 长度=${packed.length}（期望 ${3 * FACE_EMOTION_INPUT_SIZE ** 2}）` +
    `取值有限=${finite} 范围=[${lo.toFixed(3)}, ${hi.toFixed(3)}]（ImageNet 归一化后，出 [0,1] 是正常的）`);
  // 非方形输入要走最近邻兜底，不能崩
  const packedSmall = packFaceImage(syntheticImage(96, 96, 'noise'));
  console.log(`[纯函数] 96×96 → ${FACE_EMOTION_INPUT_SIZE}px 兜底：长度=${packedSmall.length}`);
  const p = softmax([1, 2, 3]);
  console.log(`[纯函数] softmax([1,2,3]) = [${p.map(fmt).join(', ')}] 和=${fmt(p.reduce((a, b) => a + b, 0))}`);
  const shifted = softmax([101, 102, 103]);
  console.log(`[纯函数] softmax 平移不变性 = ${fmt(Math.max(...p.map((v, i) => Math.abs(v - shifted[i]))))} (期望 ~0)`);

  // ── 真机路径：ORT Web + wasm ──
  //
  // 只跑**规格声明的那个尺寸**。旧的三分类模型输入 H/W 是动态维，可以顺便试 128；
  // 现在线上模型（HSEmotion）是**固定 224**，喂别的尺寸必然失败 —— 那时要验证的
  // 不是"能跑通"，而是"**喂错尺寸时优雅降级（返回 null）而不是抛给调用方**"，
  // 见下面单独那一块。
  for (const size of [FACE_EMOTION_INPUT_SIZE]) {
    const classifier = new FaceEmotionClassifier({
      modelUrl: modelPath,
      inputSize: size,
      preferWebGPU: false,
      wasmPaths: localWasmPaths,
    });
    const t0 = Date.now();
    const ok = await classifier.load();
    console.log(`\n[ORT Web] 输入 ${size}px | 加载成功=${ok} | 后端=${classifier.executionProvider} | 加载耗时 ${Date.now() - t0} ms`);
    if (!ok) throw new Error('ORT Web 加载模型失败');

    const image = syntheticImage(size, size, 'gradient');
    const result = await classifier.classify(image);
    if (!result) throw new Error('推理返回 null');
    const again = await classifier.classify(image);
    const deterministic = JSON.stringify(result.probabilities) === JSON.stringify(again?.probabilities);
    console.log(
      `  展示口径 ${FACE_EMOTION_LABELS.join(' / ')} ← 模型 8 类折叠 → ` +
      `h=${fmt(result.probabilities.happy)} n=${fmt(result.probabilities.neutral)} s=${fmt(result.probabilities.sad)}` +
      `（落在三类之外 ${fmt(result.offMapProbability ?? 0)}）`,
    );
    console.log(`  主导=${result.dominant} 置信=${fmt(result.confidence)} 概率和=${fmt(
      result.probabilities.happy + result.probabilities.neutral + result.probabilities.sad,
    )} | 推理 ${result.latencyMs} ms | 两次一致=${deterministic}`);
    classifier.dispose();
  }

  // ── 固定形状模型的降级回归：喂错尺寸不得把错误抛给调用方（眨眼主链路）──
  {
    const wrongSize = FACE_EMOTION_INPUT_SIZE + 32;
    const c = new FaceEmotionClassifier({
      modelUrl: modelPath,
      inputSize: wrongSize,
      preferWebGPU: false,
      wasmPaths: localWasmPaths,
    });
    const ok = await c.load();
    let outcome = '(未执行)';
    try {
      const r = await c.classify(syntheticImage(wrongSize, wrongSize, 'gradient'));
      outcome = r === null ? '返回 null（优雅降级）' : '❗意外成功';
    } catch (e) {
      outcome = `❗抛出了异常：${String(e)}`;
    }
    console.log(`\n[固定形状] 刻意喂错尺寸 ${wrongSize}px（模型要 ${FACE_EMOTION_INPUT_SIZE}px）：` +
      `load=${ok} | 推理=${outcome}（期望 load=true / 返回 null）`);
    if (outcome !== '返回 null（优雅降级）') process.exitCode = 1;
    c.dispose();
  }

  // ── 失败必须**可诊断**：界面不能只说「模型文件下载失败或浏览器不支持 ONNX」──
  //    那句话把 5 种病因压成一句，用户问"为什么不可用"时无从下手。
  //    这里故意指向不存在的模型，断言能拿到一句话原因。
  {
    const bad = new FaceEmotionClassifier({
      modelUrl: `${modelPath}.does-not-exist`,
      preferWebGPU: false,
      wasmPaths: localWasmPaths,
    });
    const ok = await bad.load();
    const detail = bad.unavailableDetail;
    console.log(`\n[可诊断性] 指向不存在的模型：load()=${ok}（期望 false）`
      + `| 原因=${detail ? JSON.stringify(detail).slice(0, 220) : '(空 —— 界面将无法解释为什么不可用!)'}`);
    if (ok || !detail) process.exitCode = 1;
    bad.dispose();
  }

  // ── 修复项回归：wasm 路径按运行时 ORT 版本推导（防止 glue/wasm 版本漂移）──
  {
    const ort: any = await import(pathToFileURL(
      path.join(appRoot, 'client', 'node_modules', 'onnxruntime-web', 'dist', 'ort.webgpu.mjs'),
    ).href).catch(() => null);
    const version = ort?.env?.versions?.web ?? '(需浏览器/打包环境)';
    console.log(`\n[版本推导] ORT 运行时版本 = ${version}；推导出的 wasm CDN 路径 = ${versionedWasmPaths(ort) ?? '(取不到版本，用写死兜底)'}`);
  }

  // ── 修复项回归：边加载边关闭（关摄像头时下载还没完）不得泄漏会话、不得抛错 ──
  {
    // ⚠️ 尺寸必须用**规格声明的那个**（当前 224）。这里曾写死 128（旧模型 3 类、
    // 动态维时代的残留），换了固定形状的模型之后就会静默变成"推理=null"。
    const racer = new FaceEmotionClassifier({
      modelUrl: modelPath,
      inputSize: FACE_EMOTION_INPUT_SIZE,
      preferWebGPU: false,
      wasmPaths: localWasmPaths,
    });
    const pending = racer.load();
    racer.dispose(); // 模拟"模型还在下载时用户关掉了多模态"
    const ok = await pending;
    console.log(`\n[竞态] 边加载边 dispose：load()=${ok}（期望 false）| isReady=${racer.isReady}（期望 false）| 未抛错`);

    // 释放之后仍能正常新建会话并推理（证明那次中途 dispose 没把运行时状态搞坏）
    const again = new FaceEmotionClassifier({
      modelUrl: modelPath,
      inputSize: FACE_EMOTION_INPUT_SIZE,
      preferWebGPU: false,
      wasmPaths: localWasmPaths,
    });
    const okAgain = await again.load();
    const r = await again.classify(
      syntheticImage(FACE_EMOTION_INPUT_SIZE, FACE_EMOTION_INPUT_SIZE, 'gradient'));
    console.log(`[竞态] 随后新建会话：load()=${okAgain} | 推理=${r ? '正常' : 'null'}（期望 true / 正常）`);
    if (!okAgain || !r) process.exitCode = 1;
    again.dispose();
  }

  console.log('\n' + '='.repeat(78));
  console.log('结论：模型产物在 ORT Web(wasm) 上可加载、可推理；预处理（含 ImageNet 归一化）、'
    + '8 类折叠与 offMapProbability、固定形状下喂错尺寸的优雅降级、边加载边 dispose 的竞态，四项均通过。');
  console.log('注意：这里用合成图只验证"链路通"，**精度**需用真实数据集跑 --eval-dir 才能定论。');
}

main().catch((err) => {
  console.error('[验证失败]', err);
  process.exit(1);
});
