/**
 * 延迟复核：**"发一句 → 首块音频到手"** 的干净测量（单句、反复多次）。
 *
 * ## 为什么单独一个探针（2026-10-11 用户："那现在延迟怎么样"）
 *
 * `probe_tts_supply.mjs` 里 `seq=1` 的"首块 1.5~1.9s"是**从连接建立算起**的，
 * 含连接握手，而且它与用户真机面板上的 `首段到达 1844ms`（从 `say()` 算起）**不是同一口径**。
 * 要回答"延迟怎么样"，必须用**与客户端一致的口径**：从发 `say` 那一刻计时。
 *
 * 本探针每条连接只发**一句**，连发 5 轮，报每次的：
 *   · `say → 首块`（用户"说完到听到"里的合成段）
 *   · `say → 末块`（整句产出耗时）
 *   · 服务端自报 `first_chunk_ms`（交叉印证）
 *   · 音频时长与有效 RTF
 * 最后给中位数/极值 —— 单次读数在真机上抖动很大，只有分布才说明问题。
 *
 * 用法：
 *   node scripts/probe_tts_latency.mjs [文本] [次数]
 */
const WS_URL = process.env.TTS_WS_URL ?? 'ws://127.0.0.1:8001/api/v1/tts/ws';
const TEXT = process.argv[2] ?? '听你这么说我挺高兴的。';
const ROUNDS = Number(process.argv[3] ?? 5);

const samples = [];

function runOnce(round) {
  return new Promise((resolve) => {
    const ws = new WebSocket(WS_URL);
    let sayAt = 0;
    let firstAt = null;
    let bytes = 0;
    let firstChunkReported = null;
    let done = false;

    const finish = () => {
      if (done) return;
      done = true;
      try {
        ws.close();
      } catch {
        /* 忽略 */
      }
      resolve();
    };

    const timer = setTimeout(() => {
      console.log(`  #${round} ⏰ 超时（30s）`);
      finish();
    }, 30000);

    ws.binaryType = 'arraybuffer';
    ws.onerror = () => {
      console.log(`  #${round} ❌ 连接/传输错误`);
      clearTimeout(timer);
      finish();
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data !== 'string') {
        if (firstAt === null) firstAt = Date.now() - sayAt;
        bytes += ev.data.byteLength ?? 0;
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
          console.log(`  #${round} ❌ 服务端不可用: ${msg.error ?? ''}`);
          clearTimeout(timer);
          finish();
          return;
        }
        sayAt = Date.now();
        ws.send(JSON.stringify({ type: 'say', seq: 1, text: TEXT }));
        return;
      }
      if (msg.type === 'end') {
        const lastAt = Date.now() - sayAt;
        firstChunkReported = Number(msg.first_chunk_ms ?? 0);
        const audioMs = Number(msg.audio_ms ?? 0);
        const row = {
          round,
          firstMs: firstAt,
          lastMs: lastAt,
          audioMs,
          reportedFirstMs: firstChunkReported,
          synthMs: Number(msg.synth_ms ?? 0),
          rtf: Number(msg.rtf ?? 0),
          bytes,
        };
        samples.push(row);
        console.log(
          `  #${round} say→首块=${row.firstMs}ms（服务端自报 ${row.reportedFirstMs.toFixed(0)}ms）` +
            ` say→末块=${row.lastMs}ms 音频=${audioMs.toFixed(0)}ms rtf=${row.rtf.toFixed(2)}`,
        );
        clearTimeout(timer);
        finish();
        return;
      }
      if (msg.type === 'cancelled' || msg.type === 'error') {
        console.log(`  #${round} ❌ ${msg.type}: ${msg.message ?? ''}`);
        clearTimeout(timer);
        finish();
      }
    };
  });
}

function stats(nums) {
  const s = [...nums].sort((a, b) => a - b);
  const mid = s[Math.floor(s.length / 2)];
  const p = (q) => s[Math.min(s.length - 1, Math.floor(q * s.length))];
  const mean = s.reduce((a, b) => a + b, 0) / s.length;
  const sd = Math.sqrt(s.reduce((a, b) => a + (b - mean) ** 2, 0) / s.length);
  return { min: s[0], mid, max: s[s.length - 1], p95: p(0.95), mean, sd };
}

console.log(`文本（${TEXT.length} 字）：${TEXT}`);
console.log(`测量 ${ROUNDS} 轮，每轮一条新连接、只发一句：`);

for (let i = 1; i <= ROUNDS; i += 1) {
  // eslint-disable-next-line no-await-in-loop
  await runOnce(i);
}

console.log('');
console.log('══════ 汇总（与客户端 `首段到达` 同口径：从发 say 算起）══════');
if (samples.length === 0) {
  console.log('没有有效样本');
  process.exit(1);
}
const first = stats(samples.map((s) => s.firstMs));
const last = stats(samples.map((s) => s.lastMs));
const audio = stats(samples.map((s) => s.audioMs));
console.log(`say → 首块：中位 ${first.mid}ms（${first.min}~${first.max}）· 均值 ${first.mean.toFixed(0)}ms · 标准差 ${first.sd.toFixed(0)}ms`);
console.log(`say → 末块：中位 ${last.mid}ms（${last.min}~${last.max}）`);
console.log(`该句音频时长：中位 ${audio.mid.toFixed(0)}ms`);
console.log('');
console.log('解读：');
console.log(`  · 用户在"停口 → 听到第一声"里，**这一项就要占 ${first.mid}ms**（还要加上 LLM 首句与宽限 350ms）；`);
console.log(`  · say → 末块 ${last.mid}ms 是"这一句全部产出"，它决定缓冲区能不能攒起来（音频只有 ${audio.mid.toFixed(0)}ms）。`);
// 预缓冲的诚实结论（这里**不再**给"能吸收多少"的经验数字 ——
// 预缓冲的作用是把**播放起点**推后，能不能消掉某次断档必须拿同一条时间线重算，
// 见 `scripts/probe_tts_supply.mjs` 的"各预缓冲档位"那一节。）
console.log('');
console.log('关于预缓冲：它只是把**播放起点**推后（首声随之多等），');
console.log('  能否消掉某次断档要用同一条时间线重算 —— 跑 `node scripts/probe_tts_supply.mjs`');
console.log('  看"各预缓冲档位"那一节（那里会对 0/300/600/1000/1500ms 逐档给最低水位）。');
process.exit(0);
