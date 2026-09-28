#!/usr/bin/env node
/**
 * 演示前预热（施工单 §S9.1）—— **不做的话演示会很难看**
 *
 * ## 为什么需要它
 *
 * 算法服务第一次被用到时是**冷启动**：感知模型 + 认知扭曲分类器 + 向量检索
 * 首次加载实测 6.3 s。演示时如果第一句话就吃这 6.3 秒，观感直接崩 ——
 * 而且它会被误读成"这个功能很慢"，其实是"还没热身"。
 *
 * 本脚本按顺序把四件事都预热一遍，并把每一步的实测耗时打出来：
 *
 * | # | 预热什么 | 怎么做 |
 * |---|---|---|
 * | 1 | 算法服务已就绪 | `GET /api/v1/asr/status`（`streaming_available` 必须 true）+ `GET /api/v1/tts/status` |
 * | 2 | LLM 冷启动 | `POST /api/v1/intervention/smart-chat`（非流式，最简单的一句） |
 * | 3 | TTS 模型加载 | 连 `/tts/ws`、合成一句短文本 → **首段 PCM 到手** |
 * | 4 | 流式 ASR 模型加载 | 连 `/asr/ws`、推一小段音频 + `finalize`（顺带把离线标点/复识别模型拉起来） |
 *
 * 另外**顺带**探一次 Node 服务的 `/algorithm` 代理（§S8）。
 * 那条路径在生产包里是 TTS/ASR 唯一的入口，缺了它"页面正常但没声音" ——
 * 这一项失败了要能一眼看出来，所以它不是"可选步骤"而是"报告出来"。
 *
 * 用法：
 * ```powershell
 * npm run warmup
 * npm run warmup -- --base http://127.0.0.1:8001
 * ```
 *
 * 退出码：0 = 必需项全过；1 = 有必需项失败（脚本会明确说是哪一项、怎么修）。
 */

import process from 'node:process';

const argv = process.argv.slice(2);
const argOf = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
};

const BASE = argOf('--base', process.env.ALGORITHM_BASE || 'http://127.0.0.1:8001').replace(/\/+$/, '');
const SERVER_BASE = argOf('--server', process.env.SERVER_BASE || 'http://127.0.0.1:3000').replace(/\/+$/, '');

const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const DIM = '\x1b[2m';
const OFF = '\x1b[0m';

/** 每一项的结果，最后统一打表 */
const results = [];

function record(name, ok, detail, ms, { required = true } = {}) {
  results.push({ name, ok, detail, ms, required });
  const mark = ok ? `${GREEN}✓${OFF}` : required ? `${RED}✗${OFF}` : `${YELLOW}—${OFF}`;
  const time = ms === undefined || ms === null ? '' : ` ${DIM}${Math.round(ms)} ms${OFF}`;
  console.log(`${mark} ${name}${time}${detail ? `  ${DIM}${detail}${OFF}` : ''}`);
}

async function timed(fn) {
  const t0 = performance.now();
  try {
    const value = await fn();
    return { ok: true, value, ms: performance.now() - t0 };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err), ms: performance.now() - t0 };
  }
}

async function getJson(url) {
  const res = await fetch(url, { signal: AbortSignal.timeout(15000) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/** 连一次 WebSocket 并按事件流处理；返回一个可控的 Promise */
function withWebSocket(url, handler, timeoutMs = 30000) {
  return new Promise((resolve, reject) => {
    let ws;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      reject(err instanceof Error ? err : new Error(String(err)));
      return;
    }
    ws.binaryType = 'arraybuffer';
    const timer = setTimeout(() => {
      try {
        ws.close();
      } catch {
        /* 忽略 */
      }
      reject(new Error(`等待超时（${timeoutMs} ms）`));
    }, timeoutMs);
    const done = (value) => {
      clearTimeout(timer);
      try {
        ws.close();
      } catch {
        /* 忽略 */
      }
      resolve(value);
    };
    const fail = (err) => {
      clearTimeout(timer);
      try {
        ws.close();
      } catch {
        /* 忽略 */
      }
      reject(err instanceof Error ? err : new Error(String(err)));
    };
    ws.onerror = () => fail(new Error('WebSocket 连接失败'));
    ws.onopen = () => handler({ ws, done, fail });
  });
}

/** 22050 Hz 单声道 → int16 小端（TTS 模型的实际采样率由服务端报） */
function tone(seconds, sampleRate, freq = 220, amp = 0.08) {
  const n = Math.round(seconds * sampleRate);
  const buf = new Int16Array(n);
  for (let i = 0; i < n; i++) {
    buf[i] = Math.round(Math.sin((2 * Math.PI * freq * i) / sampleRate) * amp * 32767);
  }
  return buf;
}

/** 16 kHz 单声道 int16 小端（ASR 固定 16k） */
function asrBlip(seconds = 0.6, sampleRate = 16000) {
  return tone(seconds, sampleRate, 180, 0.05);
}

// ── 1. 服务就绪 ────────────────────────────────────────────────────────────
async function checkStatus() {
  const asr = await timed(() => getJson(`${BASE}/api/v1/asr/status`));
  if (!asr.ok) {
    record('算法服务可达', false, `${BASE} 无响应：${asr.error}`, asr.ms);
    return { fatal: true };
  }
  record('算法服务可达', true, BASE, asr.ms);
  const a = asr.value;
  record(
    'ASR 流式引擎可用（边说边出字）',
    a.streaming_available === true,
    `engine=${a.streaming_engine} model=${a.streaming_model_name ?? '-'}${a.streaming_error ? ` err=${a.streaming_error}` : ''}`,
  );
  record(
    '离线标点模型可用',
    a.punctuation_available === true,
    a.punctuation_error || '',
    null,
    { required: false },
  );

  const tts = await timed(() => getJson(`${BASE}/api/v1/tts/status`));
  if (!tts.ok) {
    record('TTS 引擎可用（能出声）', false, `${BASE}/api/v1/tts/status 无响应：${tts.error}`, tts.ms);
    return { fatal: false };
  }
  const t = tts.value;
  record(
    'TTS 引擎可用（能出声）',
    t.available === true,
    `engine=${t.engine} ${t.sample_rate}Hz 音色=${t.num_speakers} 线程=${t.num_threads} max_num_sentences=${t.max_num_sentences}`,
    tts.ms,
  );
  // 这两条是 F-6 踩过的坑：max_num_sentences=1 会静默截断，采样率不能被客户端写死
  if (t.max_num_sentences !== undefined && t.max_num_sentences !== -1) {
    record('TTS max_num_sentences = -1', false, `实际是 ${t.max_num_sentences}（长句会被静默截断）`);
  }
  return { fatal: false, ttsSampleRate: t.sample_rate || 22050 };
}

// ── 2. LLM 冷启动 ─────────────────────────────────────────────────────────
async function warmLlm() {
  const r = await timed(async () => {
    const res = await fetch(`${BASE}/api/v1/intervention/smart-chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: 'warmup',
        message: '你好',
        conversation_history: [],
      }),
      signal: AbortSignal.timeout(120000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  });
  if (!r.ok) {
    record('LLM 冷启动（smart-chat）', false, r.error, r.ms);
    return;
  }
  const reply = String(r.value?.reply ?? '');
  record(
    'LLM 冷启动（smart-chat）',
    reply.length > 0,
    `risk=${r.value?.risk_level ?? '?'} reply=${reply.slice(0, 18)}${reply.length > 18 ? '…' : ''}`,
    r.ms,
  );
}

// ── 3. TTS 模型加载 ───────────────────────────────────────────────────────
async function warmTts(sampleRate = 22050) {
  const r = await timed(() =>
    withWebSocket(`${BASE.replace(/^http/, 'ws')}/api/v1/tts/ws`, ({ ws, done, fail }) => {
      let saidAt = 0;
      let seq = 1;
      ws.onmessage = (ev) => {
        if (typeof ev.data !== 'string') {
          // 第一个二进制帧 = 首段 PCM 到手（这才是"预热完成"的时刻）
          if (saidAt > 0) {
            done({
              firstPcmMs: performance.now() - saidAt,
              seq,
            });
          }
          return;
        }
        let msg;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (msg.type === 'status') {
          if (!msg.available) {
            fail(new Error(`TTS 不可用：${msg.error || '未知'}`));
            return;
          }
          saidAt = performance.now();
          ws.send(JSON.stringify({ type: 'say', seq, text: '预热一下声音。' }));
        } else if (msg.type === 'end') {
          // 没等到二进制帧就说完了（空音频）也算失败
          fail(new Error('这句没有合成出音频'));
        } else if (msg.type === 'error') {
          fail(new Error(String(msg.message || '合成报错')));
        }
      };
    }),
  );
  if (!r.ok) {
    record('TTS 模型加载（合成一句）', false, r.error, r.ms);
    return;
  }
  record(
    'TTS 模型加载（合成一句）',
    true,
    `首段 PCM 到手 ${Math.round(r.value.firstPcmMs)} ms`,
    r.ms,
  );
  void sampleRate;
}

// ── 4. 流式 ASR 模型加载 ──────────────────────────────────────────────────
async function warmAsr() {
  const r = await timed(() =>
    withWebSocket(`${BASE.replace(/^http/, 'ws')}/api/v1/asr/ws`, ({ ws, done, fail }) => {
      const startedAt = performance.now();
      // ⚠️ **首帧必须是 `hello`**：服务端据此决定"说话停顿要不要自动定稿"
      //    （`utterance` = 聊天页整段录入；`streaming` = 通话页一句一轮）。
      //    不发它，服务端不会建会话、也不会回 `status` ——
      //    表现是"连接建上了但一直没有响应"，本脚本第一版就撞上了这个（30 s 超时）。
      ws.send(JSON.stringify({ type: 'hello', mode: 'streaming' }));
      ws.onmessage = (ev) => {
        let msg;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        if (msg.type === 'status') {
          if (!msg.available) {
            fail(new Error(`ASR 不可用：${msg.error || '未知'}`));
            return;
          }
          // 推一小段音频再 finalize：让流式模型 + 离线标点/复识别模型都真正跑一次。
          // 这段音频是合成音，识别出什么无所谓 —— 这里只关心"模型加载完了没有"。
          const pcm = asrBlip();
          ws.send(pcm.buffer);
          ws.send(JSON.stringify({ type: 'finalize' }));
        } else if (msg.type === 'final') {
          // 定稿到了 ⇒ 模型链路整条都通了
          done({ elapsedMs: performance.now() - startedAt, text: String(msg.text || '') });
        } else if (msg.type === 'error') {
          fail(new Error(String(msg.message || '识别报错')));
        }
      };
    }),
  );
  if (!r.ok) {
    record('流式 ASR 模型加载', false, r.error, r.ms);
    return;
  }
  record('流式 ASR 模型加载', true, `定稿往返 ${Math.round(r.value.elapsedMs)} ms`, r.ms);
}

// ── 附加：Node 服务的 /algorithm 代理（§S8）───────────────────────────────
async function checkProxy() {
  const direct = await timed(() => getJson(`${SERVER_BASE}/algorithm/api/v1/asr/status`));
  if (!direct.ok) {
    record(
      'Node 服务 /algorithm 代理（生产包唯一入口）',
      false,
      `经 ${SERVER_BASE} 取不到：${direct.error}（Node 服务没起来时属正常，演示前请确认）`,
      direct.ms,
      { required: false },
    );
    return;
  }
  record(
    'Node 服务 /algorithm 代理（生产包唯一入口）',
    direct.value?.streaming_available === true,
    'HTTP 通',
    direct.ms,
    { required: false },
  );
}

async function main() {
  console.log(`\n${DIM}预热算法服务 ${BASE}${OFF}\n`);
  const status = await checkStatus();
  if (status.fatal) {
    console.log(`\n${RED}算法服务没起来，后面的预热没有意义。${OFF}`);
    console.log('  先运行： npm run dev:algorithm\n');
    process.exit(1);
  }
  await warmLlm();
  await warmTts(status.ttsSampleRate);
  await warmAsr();
  await checkProxy();

  const failed = results.filter((r) => r.required && !r.ok);
  const total = results.filter((r) => r.ms !== null && r.ms !== undefined).reduce((a, r) => a + r.ms, 0);
  console.log('');
  if (failed.length === 0) {
    console.log(`${GREEN}预热完成${OFF} —— 总共 ${Math.round(total)} ms。这一通电话的第一句不会再吃冷启动。`);
    console.log(`${DIM}提示：演示就用 npm run dev（vite 5173 代理）。生产构建也已补上 /algorithm 代理（§S8）。${OFF}\n`);
    process.exit(0);
  }
  console.log(`${RED}预热失败 ${failed.length} 项：${OFF}`);
  for (const f of failed) console.log(`  · ${f.name} —— ${f.detail}`);
  console.log('\n常见原因：算法服务用错了 Python（缺 sherpa-onnx）→ 用 npm run dev:algorithm:check 查。\n');
  process.exit(1);
}

main().catch((err) => {
  console.error(`${RED}预热脚本异常：${OFF}`, err);
  process.exit(1);
});
