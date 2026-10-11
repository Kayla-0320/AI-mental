/**
 * 真机时序复现：**`say` → `cancel` → 再 `say`** —— 通话每一轮改稿时客户端都会走这条路。
 *
 * ## 为什么单写这一个探针（2026-10-11 用户："我两次说我感觉好一些了都没有声音出来"）
 *
 * 用户那一通电话里 `改稿/未朗读 = 3/0`：**每一条回复都被安全审计改稿**（回复里在复述
 * 用户的话，改稿文本与流式 delta 自然不同），于是客户端每轮都执行：
 *
 *     say(seq=N, A) ──► 350ms 宽限期到 → 送合成
 *     revise 到达 ────► cancel(seq≥N) 取消在途
 *                    └► say(seq=N+k, 改稿全文) 补念
 *
 * 服务端的行为是"cancel 记进 `cancelled` 集合 + 清空同 seq 的待合成队列"，
 * 而**补念用的是新 seq**。所以正确行为是：取消掉旧句，新句照常出音频。
 * 这个探针就是来钉死这一点的 —— 如果补念句收不到任何 PCM，那问题在服务端；
 * 如果补念句正常出音频，那"没声音"就在客户端播放侧（面板新增的几行读数负责定位）。
 *
 * 退出码 0 = 取消之后的补念句**确实收到了音频**（协议层正常）。
 *
 * 用法：
 *   node scripts/probe_tts_cancel_resay.mjs
 */
const WS_URL = process.env.TTS_WS_URL ?? 'ws://127.0.0.1:8001/api/v1/tts/ws';

const BEFORE = '又见到你啦~刚才说到考试的事，心里还沉沉的呀？';
const AFTER = '听你这么说我挺高兴的。你愿意说出来，本身就是一种力气。';

const t0 = Date.now();
const ws = new WebSocket(WS_URL);
/** seq → 收到的 PCM 字节数 */
const bytes = new Map();
let lastStart = null;
const events = [];

const timer = setTimeout(() => {
  events.push('⏰ 20 秒超时');
  finish();
}, 20000);

function finish() {
  clearTimeout(timer);
  try {
    ws.close();
  } catch {
    /* 忽略 */
  }
  console.log('── 事件 ──');
  for (const e of events) console.log(' ', e);
  console.log('── 每句收到的音频字节 ──');
  for (const [seq, n] of [...bytes.entries()].sort((a, b) => a[0] - b[0])) {
    console.log(`  seq=${seq}: ${n} 字节${n === 0 ? '  ← 一个字节都没有' : ''}`);
  }
  const seq1 = bytes.get(1) ?? 0;
  const seq2 = bytes.get(2) ?? 0;
  const ok = seq2 > 0;
  console.log('');
  console.log(`取消前那句 seq=1 收到 ${seq1} 字节（被取消，允许为 0）`);
  console.log(`**取消后补念的 seq=2 收到 ${seq2} 字节**`);
  console.log(ok ? '✅ 取消之后的补念句照常出音频（协议层正常）' : '❌ 取消之后的补念句没有音频 —— 问题在服务端取消路径');
  process.exit(ok ? 0 : 1);
}

ws.binaryType = 'arraybuffer';
ws.onerror = (e) => events.push(`onerror: ${e?.message ?? e}`);
ws.onmessage = (ev) => {
  const dt = Date.now() - t0;
  if (typeof ev.data !== 'string') {
    const seq = lastStart;
    if (seq !== null) bytes.set(seq, (bytes.get(seq) ?? 0) + (ev.data.byteLength ?? 0));
    return;
  }
  let msg;
  try {
    msg = JSON.parse(ev.data);
  } catch {
    return;
  }
  events.push(`${msg.type}${msg.seq !== undefined ? ` seq=${msg.seq}` : ''} +${dt}ms`);
  if (msg.type === 'status') {
    if (!msg.available) {
      console.log('❌ 服务端自述不可用:', msg.error ?? '（无原因）');
      finish();
      return;
    }
    console.log('① say seq=1（改稿前的流式文本）');
    ws.send(JSON.stringify({ type: 'say', seq: 1, text: BEFORE }));
    // 350ms 后（就是客户端 `SPEAK_PLAN_GRACE_MS` 的时机）模拟"改稿到了"：
    // 先取消 seq≥1，再用新 seq 补念改稿全文
    setTimeout(() => {
      console.log('② cancel seq=1（改稿到达）');
      ws.send(JSON.stringify({ type: 'cancel', seq: 1 }));
      console.log('③ say seq=2（补念改稿全文）');
      ws.send(JSON.stringify({ type: 'say', seq: 2, text: AFTER }));
    }, 350);
    return;
  }
  if (msg.type === 'start') {
    lastStart = Number(msg.seq);
    return;
  }
  if (msg.type === 'end' || msg.type === 'cancelled' || msg.type === 'error') {
    // 等两句都有结论再收尾
    const done1 = events.some((e) => (e.startsWith('end seq=1') || e.startsWith('cancelled seq=1')));
    const done2 = events.some((e) => (e.startsWith('end seq=2') || e.startsWith('cancelled seq=2')));
    if (msg.seq === 2 && done1 && done2) finish();
  }
};
