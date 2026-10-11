/**
 * TTS 供音速度探针 —— **从浏览器这一侧**量"音频到底以多快的速度流出来"。
 *
 * ## 为什么不用 `end` 帧自报的 rtf（2026-10-11 用户："中途还有些小卡顿"）
 *
 * 自报 `rtf` 来自合成内部的计时（`synth_ms / audio_ms`），它回答"模型算了多久"；
 * 而"**缓冲区会不会被吃空**"问的是另一件事：**音频块到达的速度**。
 * 两者在流式合成下并不相等（首块慢、后面快；或反过来），所以这里**独立量一遍**：
 *
 *     有效RTF = (最后一块到达时刻 − 第一块到达时刻) ÷ 该段音频总时长
 *
 * * `有效RTF ≤ 1` ⇒ 供音速度 ≥ 播放速度，缓冲区只涨不跌（卡顿只可能来自接缝/调度）；
 * * `有效RTF > 1` ⇒ 供音持续慢于播放 ⇒ 缓冲区被逐渐吃空，长回复必然越来越卡。
 *
 * ## 复刻真机时序
 *
 * 通话页的做法是：**一句话一个 seq，一次把整轮回复的所有句都发出去**，
 * 服务端串行合成。本探针照做：
 *
 *   1. 连上 `/tts/ws` 等 `status`；
 *   2. 立刻把 N 句 `say` 全部发出（与客户端一致）；
 *   3. 逐块记 `(seq, 到达时刻, 字节数)`；
 *   4. 每句 `end` 后算：首块 / 流出跨度 / 音频时长 / **有效RTF** / 最大块间隔；
 *   5. 另外算"**句间接缝**"：上一句末块 → 下一句首块的间隔（跨句卡顿的直接读数）。
 *
 * 用法（仓库根目录）：
 *   node scripts/probe_tts_supply.mjs                 # 默认三句
 *   node scripts/probe_tts_supply.mjs --long          # 一句长回复（看"越说越卡"）
 */
const WS_URL = process.env.TTS_WS_URL ?? 'ws://127.0.0.1:8001/api/v1/tts/ws';

const DEFAULT_LINES = [
  '听你这么说我挺高兴的。',
  '你刚才说，考试没考好，心里一直沉沉的。',
  '难受的时候，可以试试深呼吸三次，或者找一个你愿意说话的人，说一句今天有点累。',
];

/** 一句长回复：用来暴露"供音慢于播放时缓冲区被逐渐吃空" */
const LONG_LINE =
  '我听见你说考试没考好，心里很难受。先说一句，这种难受不是矫情，' +
  '它说明你其实很在意这件事。我们可以把它拆小一点看：哪一科最让你堵得慌，' +
  '是怕辜负谁，还是怕自己就这样了？你愿意从哪儿开始说，我都在这儿听着，' +
  '不着急，也不用一次说完。';

const lines = process.argv.includes('--long') ? [LONG_LINE] : DEFAULT_LINES;

const t0 = Date.now();
const ws = new WebSocket(WS_URL);
/** seq → 明细 */
const seg = new Map();
let lastArrival = null;
let lastSeq = null;
/** 句间接缝（上一句末块 → 本句首块） */
const seams = [];

const done = new Set();
let expected = lines.length;

function ensure(seq) {
  if (!seg.has(seq)) {
    seg.set(seq, {
      seq,
      firstMs: null,
      lastMs: null,
      bytes: 0,
      chunks: 0,
      gaps: [],
      end: null,
      /** 逐块 (到达时刻, 累计音频 ms)，用来算"播放缓冲区水位" */
      arrivals: [],
    });
  }
  return seg.get(seq);
}

/**
 * 按"播放头"算缓冲区水位 —— 这才是"会不会卡"的直接判据。
 *
 * 模型：从**第一块到达**开始按实时消耗（1 ms 播放消 1 ms 音频）。
 * 水位 = 已到达音频累计 − 已播时长；水位降到 0 就是**听得见的断档**。
 *
 * ⚠️ 只按**单句**算（句内）。跨句的水位需要用整条流算，见 {@link overallBuffer}。
 */
function bufferCurve(s) {
  if (s.arrivals.length === 0) return null;
  let cum = 0;
  let min = Infinity;
  let minAt = null;
  const t0 = s.arrivals[0][0];
  let starveMs = 0;
  for (const [at, bytes] of s.arrivals) {
    const audioMs = bytes / 48; // 48000 B/s ⇒ 48 B/ms
    const playedMs = at - t0;
    const level = cum - playedMs;
    if (level < min) {
      min = level;
      minAt = at;
    }
    if (level < 0) starveMs += -level; // 累积见底量（粗略：只用于排序）
    cum += audioMs;
  }
  return { min, minAt, starveMs, totalAudioMs: cum };
}

/**
 * 整轮（跨句）的缓冲区水位：把整轮所有块按到达顺序拼起来算。
 *
 * @param prebufferMs **开场预缓冲**（毫秒）。现行播放器是"第一块一到就开始播"，
 *   而预缓冲把**播放起点**推后这么多：这段时间里音频不被消耗，只有新块累积。
 *   于是水位 = 已到达音频 − max(0, 到达时刻 − 预缓冲)。这是评估档位好坏的唯一正确模型
 *   （"预缓冲能吸收多少"不能靠拍脑袋，见 `probe_tts_latency.mjs` 里那段被修正的说明）。
 */
function overallBuffer(prebufferMs = 0) {
  const all = [];
  for (const s of seg.values()) {
    for (const [at, bytes] of s.arrivals) all.push([at, bytes]);
  }
  all.sort((a, b) => a[0] - b[0]);
  if (all.length === 0) return null;
  let cum = 0;
  let min = Infinity;
  let minAt = null;
  const t0 = all[0][0];
  const level = (at) => cum - Math.max(0, at - t0 - prebufferMs);
  for (const [at, bytes] of all) {
    cum += bytes / 48;
    const lv = level(at);
    if (lv < min) {
      min = lv;
      minAt = at;
    }
  }
  return {
    min,
    minAt,
    totalAudioMs: cum,
    lastLevel: level(all[all.length - 1][0]),
    /** 逐块水位曲线 `[t_ms, level_ms]`，供压缩成消耗段 */
    curve: (() => {
      let c = 0;
      const out = [];
      for (const [at, bytes] of all) {
        c += bytes / 48;
        out.push([at - t0, c - Math.max(0, at - t0 - prebufferMs)]);
      }
      return out;
    })(),
    /** 见底累计时长（毫秒）—— 耳朵里"卡"的总量 */
    starveTotalMs: (() => {
      let c = 0;
      let total = 0;
      let prevT = null;
      for (const [at, bytes] of all) {
        const t = at - t0;
        c += bytes / 48;
        const lv = c - Math.max(0, t - prebufferMs);
        if (lv < 0 && prevT !== null) total += Math.min(-lv, t - prevT);
        prevT = t;
      }
      return Math.round(total);
    })(),
  };
}

/**
 * 把逐块水位曲线压成「**正在消耗**」的区间（水位单调下降的那些段）。
 *
 * 为什么要压：几十块拼出来的曲线逐块看没有意义；用户要的是
 * "**卡顿那一刻缓冲区还有多少**"，也就是"从一个峰开始一路消耗到哪里"。
 * 每个区间给：起点水位、终点水位、时长、终点时刻。
 */
function drainSegments(curve) {
  const segs = [];
  let peak = null;
  let prevLevel = null;
  let prevT = null;
  for (const [t, level] of curve) {
    if (prevLevel === null) {
      peak = { t, level };
      prevLevel = level;
      prevT = t;
      continue;
    }
    if (level < prevLevel) {
      // 还在消耗
    } else {
      // 水位回升 ⇒ 上一次消耗结束
      if (peak && prevT > peak.t) {
        segs.push({ fromT: peak.t, fromLevel: peak.level, toT: prevT, toLevel: prevLevel, durMs: prevT - peak.t });
      }
      peak = { t, level };
    }
    prevLevel = level;
    prevT = t;
  }
  if (peak && prevT !== null && prevLevel !== null && prevT > peak.t) {
    segs.push({ fromT: peak.t, fromLevel: peak.level, toT: prevT, toLevel: prevLevel, durMs: prevT - peak.t });
  }
  return segs;
}

const timer = setTimeout(() => {
  console.log('⏰ 60 秒超时，按已有数据汇总');
  finish();
}, 60000);

function finish() {
  clearTimeout(timer);
  try {
    ws.close();
  } catch {
    /* 忽略 */
  }
  console.log('');
  console.log('══════ 每句供音时间线（24kHz 单声道 PCM16：48000 字节 = 1 秒）══════');
  const rows = [...seg.values()].sort((a, b) => a.seq - b.seq);
  let totalAudioMs = 0;
  let totalSpanMs = 0;
  for (const s of rows) {
    const audioMs = s.bytes / 48; // 48000 B/s ⇒ 48 B/ms
    const spanMs = s.firstMs !== null && s.lastMs !== null ? s.lastMs - s.firstMs : 0;
    const rtf = audioMs > 0 ? spanMs / audioMs : 0;
    const maxGap = s.gaps.length ? Math.max(...s.gaps.map((g) => g.gap)) : 0;
    totalAudioMs += audioMs;
    totalSpanMs += spanMs;
    console.log(
      `seq=${s.seq} 块=${s.chunks} 字节=${s.bytes} 音频=${audioMs.toFixed(0)}ms ` +
        `首块=${s.firstMs === null ? '-' : s.firstMs.toFixed(0)}ms ` +
        `流出=${spanMs.toFixed(0)}ms **有效RTF=${rtf.toFixed(2)}** ` +
        `最大块间隔=${maxGap.toFixed(0)}ms ${s.end ? '已 end' : '未 end'}`,
    );
    // 服务端自报（相对 `say`）：这是"用户说完 → 第一块音频到手"里的**合成段**，
    // 客户端那边的首声延迟要用它，而不是用"从连接建立算起"的首块时刻。
    if (s.rtfReported !== undefined) {
      console.log(
        `    服务端自报：first_chunk=${s.firstChunkMsReported.toFixed(0)}ms` +
          ` synth=${s.synthMs.toFixed(0)}ms audio=${s.audioMsReported.toFixed(0)}ms rtf=${s.rtfReported.toFixed(2)}`,
      );
    }
    if (s.gaps.length) {
      console.log('    ⚠️ 停顿（>250ms）: ' + s.gaps.map((g) => `${g.gap.toFixed(0)}ms@${g.at.toFixed(0)}`).join(' '));
    }
  }

  console.log('');
  console.log('══════ 句间接缝（上一句末块 → 下一句首块）══════');
  if (seams.length === 0) {
    console.log('  （没有可比的接缝）');
  }
  for (const s of seams) {
    console.log(`  seq=${s.from} → seq=${s.to}: 间隔 ${s.gapMs.toFixed(0)}ms`);
  }

  console.log('');
  console.log('══════ 播放缓冲区水位（按"到达即排队、实时消耗"模型算）══════');
  for (const s of rows) {
    const b = bufferCurve(s);
    if (!b) continue;
    console.log(
      `seq=${s.seq} 单句最低水位=${b.min.toFixed(0)}ms` +
        (b.min < 0 ? ` ⚠️ **见底 ${(-b.min).toFixed(0)}ms@${b.minAt.toFixed(0)}ms**` : ' ✅ 从未见底') +
        `（音频共 ${b.totalAudioMs.toFixed(0)}ms）`,
    );
  }
  const ob = overallBuffer(0);
  if (ob) {
    console.log(
      `整轮最低水位=${ob.min.toFixed(0)}ms` +
        (ob.min < 0 ? ` ⚠️ **见底 ${(-ob.min).toFixed(0)}ms@${ob.minAt.toFixed(0)}ms**` : ' ✅ 从未见底') +
        ` · 收尾时水位=${ob.lastLevel.toFixed(0)}ms · 见底合计 ${ob.starveTotalMs}ms`,
    );
    // ── 档位对比：用**同一条时间线**按不同预缓冲重算（不是拍脑袋的"能吸收多少"）──
    console.log('  各预缓冲档位（同一条时间线重算）：');
    for (const p of [0, 300, 600, 1000, 1500]) {
      const r = overallBuffer(p);
      const mark = r.min < 0 ? `⚠️ 见底 ${(-r.min).toFixed(0)}ms，累计 ${r.starveTotalMs}ms` : '✅ 从未见底';
      console.log(`    prebuffer=${String(p).padStart(4)}ms ⇒ 最低水位 ${r.min.toFixed(0)}ms ${mark}`);
    }
    console.log('    （代价：每一档都让**首声**再多等同样时长）');
    // 「正在消耗」的区间：每个区间就是"从上一次供音涌进来到下一次供音到达之间，缓冲被吃掉多少"
    const drains = drainSegments(ob.curve).filter((d) => d.durMs >= 200);
    if (drains.length > 0) {
      console.log('  消耗区间（水位从高到低，就是"等下一块"的那段）:');
      let starveTotal = 0;
      for (const d of drains.slice(0, 12)) {
        const flag = d.toLevel < 0 ? ' ⚠️ 见底' : d.toLevel < 200 ? ' ⚠️ 逼近见底' : '';
        if (d.toLevel < 0) starveTotal += -d.toLevel;
        console.log(
          `    t=${d.fromT.toFixed(0)}→${d.toT.toFixed(0)}ms（${d.durMs.toFixed(0)}ms）` +
            ` 水位 ${d.fromLevel.toFixed(0)} → ${d.toLevel.toFixed(0)}ms${flag}`,
        );
      }
      console.log(`  ⇒ 本轮**见底合计约 ${starveTotal.toFixed(0)}ms**（耳朵里那几下停顿的总量）`);
      const risky = drains.filter((d) => d.toLevel < 200);
      if (risky.length > 0) {
        const worst = risky.reduce((a, b) => (a.fromLevel - a.toLevel > b.fromLevel - b.toLevel ? a : b));
        const need = Math.ceil((worst.fromLevel - worst.toLevel + 150) / 100) * 100;
        console.log(
          `  ⇒ **开场预缓冲建议**：最险的一段从水位 ${worst.fromLevel.toFixed(0)}ms 一路吃到 ` +
            `${worst.toLevel.toFixed(0)}ms（消耗 ${(worst.fromLevel - worst.toLevel).toFixed(0)}ms）。` +
            `按现行"收到第一块就播"的模型，需要 ${need}ms 的预缓冲才能覆盖它 —— ` +
            `代价是首声再多等约 ${need}ms（当前首块本身就要 ${rows[0].firstMs.toFixed(0)}ms）。`,
        );
        // 开场的那个上限：预缓冲只在**开场**等待，之后水位自己会被后续块抬起来
        const firstDrain = drains[0];
        console.log(
          `  ⇒ 若只想覆盖**开场**那一下：第一段消耗 ${firstDrain.durMs.toFixed(0)}ms、` +
            `从 ${firstDrain.fromLevel.toFixed(0)}ms 吃到 ${firstDrain.toLevel.toFixed(0)}ms，` +
            `取 ${Math.max(300, Math.ceil((firstDrain.fromLevel - firstDrain.toLevel + 150) / 100) * 100)}ms 即可。`,
        );
      }
    }
  }

  console.log('');
  console.log('══════ 结论 ══════');
  const overall = totalAudioMs > 0 ? totalSpanMs / totalAudioMs : 0;
  console.log(`整轮：音频合计 ${totalAudioMs.toFixed(0)}ms，块流合计 ${totalSpanMs.toFixed(0)}ms，**整轮有效RTF=${overall.toFixed(2)}**`);
  if (overall > 1.0) {
    console.log('❌ 供音速度**慢于**播放速度 ⇒ 缓冲区会被持续吃空（长回复越来越卡）。先修推理速度/预缓冲。');
  } else {
    console.log('✅ 平均供音快于播放 ⇒ 卡顿若发生，来自**短时抖动**（抖动幅度超过当时水位）或片段接缝/调度。');
  }
  if (ob && ob.min < 0) {
    console.log(
      `⚠️ 缓冲区在 t≈${ob.minAt.toFixed(0)}ms 见底 ${(-ob.min).toFixed(0)}ms ⇒ 那一刻就是一次**听得见的断档**；` +
        `该时刻附近上游在等哪一块，就是病根（1~2s 的块间隔 > 当时水位）。`,
    );
  } else {
    console.log('✅ 缓冲区从未见底 ⇒ 静态播放不会断。若耳朵里仍有卡顿，去查句内块接缝（交叉淡化）与播放调度。');
  }
  const worstSeam = seams.length ? Math.max(...seams.map((s) => s.gapMs)) : 0;
  if (worstSeam > 250) {
    console.log(`ℹ️ 句间接缝最大 ${worstSeam.toFixed(0)}ms —— 但**只要水位足够就不是卡顿**（音频是排队播的，不是到了才播）。`);
  }
  process.exit(0);
}

ws.binaryType = 'arraybuffer';
ws.onerror = (e) => console.log('onerror:', e?.message ?? e);
ws.onmessage = (ev) => {
  const now = Date.now() - t0;
  if (typeof ev.data !== 'string') {
    const s = ensure(lastSeq ?? -1);
    if (s.firstMs === null) {
      s.firstMs = now;
      if (lastArrival !== null && lastSeq !== null && seg.size > 1) {
        // 接缝：上一句的末块 → 本句首块
        seams.push({ from: [...seg.keys()].filter((k) => k !== lastSeq).pop() ?? -1, to: lastSeq, gapMs: now - lastArrival });
      }
    } else {
      const gap = now - lastArrival;
      if (gap > 250) s.gaps.push({ at: now, gap });
    }
    s.lastMs = now;
    s.bytes += ev.data.byteLength ?? 0;
    s.chunks += 1;
    s.arrivals.push([now, ev.data.byteLength ?? 0]);
    lastArrival = now;
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
      console.log('❌ 服务端不可用:', msg.error ?? '（无原因）');
      process.exit(1);
    }
    console.log(`引擎=${msg.engine} 采样率=${msg.sample_rate}Hz 句数=${lines.length}`);
    lines.forEach((text, i) => {
      console.log(`say seq=${i + 1}（${text.length} 字）`);
      ws.send(JSON.stringify({ type: 'say', seq: i + 1, text }));
    });
    return;
  }
  if (msg.type === 'start') {
    lastSeq = Number(msg.seq);
    ensure(lastSeq);
    return;
  }
  if (msg.type === 'end' || msg.type === 'cancelled' || msg.type === 'error') {
    const s = ensure(Number(msg.seq));
    s.end = msg.type;
    if (msg.type === 'end') {
      s.audioMsReported = Number(msg.audio_ms ?? 0);
      s.synthMs = Number(msg.synth_ms ?? 0);
      s.rtfReported = Number(msg.rtf ?? 0);
      s.firstChunkMsReported = Number(msg.first_chunk_ms ?? 0);
    }
    done.add(Number(msg.seq));
    if (done.size >= expected) finish();
  }
};
