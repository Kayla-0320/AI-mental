# 「AI 陪伴通话」施工单 —— 具体实施步骤

> **这份文档与既有三份文档的分工：**
>
> | 文档 | 回答的问题 | 本文档是否重复 |
> |---|---|---|
> | [`voice_call_research.md`](./voice_call_research.md) | 有哪些事实？ | 否 |
> | [`voice_call_plan.md`](./voice_call_plan.md) | **能不能做 / 走哪条路 / 有什么风险** | 否（只引用其结论） |
> | [`video_call_build_plan.md`](./video_call_build_plan.md) | 视觉要修哪 12 项 | 否（阶段 4 直接引用） |
> | [`tts_benchmark.md`](./tts_benchmark.md) | **出声端到底行不行**（S0 实测） | 否（S0 结果全在那里） |
> | **本文档** | **先做什么、后做什么、每个文件改成什么样、怎么证明改对了** | —— |
>
> 事实来源标注沿用既有约定：
> * **[实测]** 本机真跑出来的数字；**[实测-代码]** 从当前工作区读到的 `文件:行号`；
> * **[估计]** 有依据的推算；**[未验证]** 没验证过。
>
> **本文档不新增任何安全规则、不改任何阈值、不拟任何危机话术。** 需要人工审核的地方逐条标了
> ⚠️（AGENTS.md「所有安全相关规则的变更必须经人工审核」）。

---

## 0. 一页结论

### 0.1 施工顺序（关键路径）

```
S0 TTS 可行性 benchmark ──✅ 已跑通，Path A 成立──┐
   （见 docs/tts_benchmark.md）                  ▼
                        S1 本地 TTS 引擎 + WS 端点 ──✅ 已完成（9/9 检查通过）
                                     ▼
                        S2 前端播放队列（Web Audio）──✅ 已完成（17/17 检查通过）
                                     ▼
     S3 通话页骨架（可与 S1/S2 并行）──✅ 已完成（27/27 检查通过）─┤
                                     ▼
                        S4 轮次编排（useVoiceCall）──✅ 已完成（14/14 检查通过）
                                     ▼
                        S5 打断 + 回声过滤──✅ 已完成（阈值实测标定 0.40；见 §S5.6）
                                     ▼
                        S6 危机切换（安全兜底）──✅ 已完成（见 §S6.5）
                                     ▼
                        S7 视觉进通话──✅ 已完成（F7 硬闸门通过；见 §S7.1）
                                     ▼
                        S8 生产可用性──✅ 已完成（HTTP+WS 代理 6/6）
                        S9 演示准备──✅ 已完成（预热 8/8 + 走查清单）
```

> **S0–S9 全部完成。** 三条通话闸门合计 **27 + 14 + 37 = 78 项检查全绿**
> （`_probe_call_page.js` / `_probe_call_voice_e2e.js` / `_probe_call_s5.js`）。

* **S0 已完成 [实测]**：RTF 最差 **0.180**（4 线程）/ **0.112**（8 线程）空载，
  满载 **0.209 / 0.136**，判据 0.4 —— 余量 2–3 倍。**Path A 技术成立**，
  D1 因此有了"可以选本地"的资格（但仍需人拍板，见 §7 Q1）。
  完整数据与三个坑见 [`docs/tts_benchmark.md`](./tts_benchmark.md)。
* **S3 不依赖 S1/S2 的"功能"**：通话页 UI 可以先做。实际做下来，因为它顺手复用了
  S2 的 `TtsChannel`，骨架里那句"假声音"是**真实 TTS 念预置文本**（`?demo=1`）——
  文本是假的、链路是真的，这样"状态词 / 字幕 / 头像律动"这条视觉链在 S4 之前就被验过了。

* **最小可演示产品 = S1–S4 + S6**（能通话、AI 会出声、边说边出字幕、危机兜得住）。
  **S5 是加分项，S7 是可砍项。**

### 0.2 七条"读代码 / 实测"得出的事实（决定了后面所有步骤的形状）

前五条是**读代码**读出来的 —— 它们把原本需要新建的东西变成了**复用既有代码**；
第六条是 **S0 实测**出来的 —— 它改变了 S1 的协议形态；
第七条是 **S3 实测**出来的 —— 它改变了"首声"这个指标本身的定义。

| # | 事实 | 位置 | 影响 |
|---|---|---|---|
| **F-1** | 通话的文本**不能**直连算法服务，必须走 Node 的 SSE 端点 | `server/src/controllers/consultation.controller.ts:62` | 通话页是 `sendChatMessageStream()` 的**壳**，不需要新的文本协议 |
| **F-2** | SSE 的 `delta` **已经是过闸的完整句子** | `algorithm/api/intervention.py:1899` | 它**就是** TTS 的合成单位 → **不需要在服务端新建切句器** |
| **F-3** | `meta.streaming === false` 就是"高风险禁用语音"的现成开关 | `intervention.py:1865-1873` | 零新增阈值即可硬禁用高风险语音 |
| **F-4** | 打断时**绝不能 abort SSE**（会触发第二次 LLM + 落库一份用户没听过的回复） | `consultation.service.ts:360-405` | 打断 = 停播 + 排队下一轮，**不是**断连接 |
| **F-5** | 麦克风全项目只有一路（`mediaStreamManager` 单例） | `client/src/services/mediaStreamManager.ts:82-85` | 通话必须复用 `startVoiceInput()`，且 AEC 必须加在**默认约束**上 |
| **F-6** | **[实测]** VITS 经 `OfflineTts` **不流式**；且 `max_num_sentences=1` 会**静默丢弃第一句之后的内容** | `algorithm/tools/benchmark_tts.py`、[`docs/tts_benchmark.md`](./tts_benchmark.md) | S1 的 WS 协议从"分块推流"改为"一句一个二进制帧"；`max_num_sentences` 必须 `-1` 并配回归测试 |
| **F-7** | **[实测]** `AudioContext` 的时钟在**空闲时走不准**（实测 3.0 s 墙钟只走 2.08 s），但**一旦有音频在播就 1:1**；而 `AudioContext` 必须在**用户手势**里 `resume()` | `_probe_player_clock.js` | ① "首声"必须量**「请求 → 音频到手」**，不能量"排到它出声"（第 2 段起天然含前几段的时长，实测差 20 倍）；② 通话页必须有"开始通话"这个点击，不能在 effect 里开麦（见 S3.6） |

---

## 1. 五条事实的详细依据

### F-1 —— 文本必须走 Node 的 SSE 端点

**证据链 [实测-代码]：**

```
client  client/src/services/chatStream.ts:138
        POST /api/consultation/conversations/:id/messages/stream
   ↓
server  server/src/controllers/consultation.controller.ts:62  sendMessageStream
   ↓
server  server/src/services/consultation.service.ts:295
        ├─ :313  先落库用户消息（前端用它替换乐观占位）
        ├─ :318  取最近 20 条历史喂给 LLM
        ├─ :335  algorithmBridge.smartChatStream(...)  ← 唯一的上游调用点
        ├─ :408  危机时补热线（幂等）
        ├─ :427  与已上屏内容比对 → revise
        └─ :432  persistAssistantReply(...)            ← 落库 AI 回复
   ↓
algo    algorithm/api/intervention.py:1961  /smart-chat/stream
```

**结论**：如果通话页自己去连 `8001/api/v1/intervention/smart-chat/stream`，会绕过
`:313 / :408 / :432` 三步 —— **用户说的话不进数据库、AI 的回复不进数据库、
危机热线不补全、`/admin/crisis` 看不到**。这个代价是不可接受的。

**做法**：通话页把「ASR 定稿文本」当成一条普通聊天消息交给
`sendChatMessageStream()`，然后消费它已有的 6 种事件。**不新增任何文本侧协议。**

---

### F-2 —— `delta` 已经是过闸的完整句子，它天然就是 TTS 单位

**证据链 [实测-代码]**：`algorithm/api/intervention.py`

```python
# :1899-1901
for unit in gate.feed(delta):            # gate = _StreamingSafetyGate
    shown_parts.append(unit)
    yield _sse_pack(ChatStreamEventType.DELTA, {'text': unit})
```

而 `gate.feed()`（`:1786-1822`）只在两种情况下切出一个 `unit`：
1. 遇到句末标点 `[。！？!?；;\n]`（`:1805`，正常路径）；
2. 长度触及 `_STREAM_UNIT_MAX_CHARS = 200`（`:1808`，兜底）。

**结论**：**每一个 `delta` 都是一个已过词法红线筛查的完整语义单位。**
这正是 `voice_call_plan.md` §4.3 想要的「逐句过闸 → 逐句合成 → 逐句播」——
**闸门已经接好了，不需要再写一个切句器，也不需要碰安全组件。**

**唯一的补丁**：200 字上限对语音太长（≈36 秒音频）。**在客户端补一刀**：
`delta` 长度 > 30 字时，先按 `[，,、]` 二次切分，再按 30 字硬切。
（在客户端做而不是服务端，是为了**完全不碰 `_StreamingSafetyGate`** ——
`voice_call_plan.md` §4.3 的注意事项 1 建议服务端加独立切句器；客户端做更省、风险更低，
安全语义完全等价：切成两段说，与一段说完，过的是同一套词表。）

---

### F-3 —— `meta.streaming` 就是高风险硬禁用语音的开关

**证据 [实测-代码]**：

```python
# algorithm/api/intervention.py:1865-1873
streaming_allowed = (prep.risk_level in ("low", "medium") and not prep.crisis_detected)
yield _sse_pack(ChatStreamEventType.META, {
    'risk_level': prep.risk_level, 'dialogue_mode': prep.dialogue_mode,
    'emotion_probs': ..., 'streaming': streaming_allowed,
})
```

**结论**：`voice_allowed = meta.streaming`。**没有新增任何阈值或判定条件**，
满足 AGENTS.md「复用既有规则不需要人工审核」。

效果（`voice_call_plan.md` §4.2 已实测过）：
`risk_level ∈ {high, crisis}` → 整轮**一个字都不上屏**，自然也不会合成任何语音；
`done` 里 `requires_escalation=True`，走既有升级流程。

---

### F-4 —— 打断时**不能** abort SSE ⚠️ 本计划最重要的一个反直觉结论

**证据链 [实测-代码]**：

```ts
// consultation.controller.ts:90-93
req.on('close', () => { closed = true; abort.abort(); });   // 客户端断开 → 中止上游
```

```ts
// consultation.service.ts:360-405
} catch (error) {
  console.warn('[Consultation] 流式生成中断，回退非流式路径:', error);  // ← 只是警告
}
if (!smartResult) {                                    // ← 于是走进这里
  const fallback = await algorithmBridge.smartChat({...});   // ← 第二次完整 LLM 调用（没传 signal！）
  ...
}
// :432  persistAssistantReply(...)                     // ← 把这份"用户从没听过"的回复落库
```

**所以「打断 = abort 连接」在现有服务端语义下会产生三个后果：**

| # | 后果 | 严重性 |
|---|---|---|
| 1 | 上游流被中止后**再发一次完整 LLM 请求**（token 翻倍） | 中 |
| 2 | 落库一份**用户从未听到**的完整回复 → 历史记录与通话内容不一致 | **高** |
| 3 | 打断的那句话作为新一轮，与上一轮**并发**跑在同一个 conversation 上 → 历史顺序错乱 | **高** |

**结论（打断的正确语义）：**

```
用户插话（partial 命中，非回声）
  → ① 立刻本地停播 + 清空 PCM 队列与待合成队列      ← 用户感知"立刻停了"（[估计] < 50 ms）
  → ② 【不断连接】让当前 SSE 继续跑完，但：
        · 不再合成语音（后续 delta 只上屏，灰色显示）
        · 不再上屏给"正在说"的字幕
  → ③ 把插话文本挂进"下一轮待发队列"
  → ④ 等本轮 `done`（含落库完成）到达 → 立刻发送插话，开始下一轮
```

代价：插话真正生效要等本轮 `done`（通常 < 2 s，因为首句 442–497 ms 已到 [实测]）。
**这个代价必须如实接受**，换来的是：单次 LLM 调用、落库一致、无并发轮次。

> ⚠️ **如果要省掉这个等待**，就得改服务端：给 `sendMessageStream` 加一个"软打断"
> 参数（不 abort、只截断上游生成并以已生成内容落库）。那属于**改动安全链路附近的代码**，
> 本计划不擅自做，列为待决项 **Q3**（见 §7）。

---

### F-5 —— 麦克风只有一路，AEC 必须加在默认约束上

**证据 [实测-代码]**：

```ts
// client/src/services/mediaStreamManager.ts:82-85
const DEFAULT_CONSTRAINTS = {
  camera: { video: { facingMode: 'user', width: {ideal:640}, height: {ideal:480} } },
  microphone: { audio: true },          // ← 没有任何 AEC / NS / AGC 约束
};
```

```ts
// client/src/hooks/useVoiceAnalysis.ts:308
const stream = await mediaStreamManager.acquire('microphone', 'perception-voice');
```

**关键点**：`acquire` 是**幂等**的，第一个消费者拿到什么约束，后面所有人共用同一份轨道。
所以：

* 若用户先开过「多模态感知」，麦克风已被 `{audio:true}` 拿住；
  通话页再 `acquire('microphone','call', {audio:{echoCancellation:true}})` **拿不到 AEC** ——
  它会拿到那份没有 AEC 的旧轨道。
* **因此 AEC 只能加在 `DEFAULT_CONSTRAINTS.microphone` 上**，不能只在通话路径加。

**改动**（S5 的第 1 步）：

```ts
microphone: {
  audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
},
```

并按 `voice_call_plan.md` §6.1 的要求，把 `track.getSettings()` 读回的**实际生效值**
同步到通话页 UI（**不要假设它生效了** —— 浏览器和设备都可能忽略这些约束）。

---

### F-6 —— [实测] VITS 不流式 + `max_num_sentences=1` 会静默截断

这两条是 **S0 benchmark 实测**出来的，完整数据见 [`docs/tts_benchmark.md`](./tts_benchmark.md)。

**(a) `callback` 恒为 1 次 → 没有"首块"这回事。**

```
[短句] 字数=9   callback 次数 = 1   t=282.6 ms  samples=17058
[长句] 字数=33  callback 次数 = 1   t=1012.4 ms samples=64058
```

回调收到的样本与 `generate()` 返回的 `audio.samples` **逐样本相等** —— 没有增量输出。
所以"首块延迟"就是"整句合成耗时"，两者是同一个数。
`voice_call_plan.md` §2.2 假设的"首块 100–300 ms"**不成立**。

**影响 S1 的协议形态**：原设计的"多块二进制 + `start`/`end` 定界"退化为
**一句一个二进制帧**。仍然是"`start` → 1 个二进制帧 → `end`"，
但不再需要处理"同一句的多块拼接"。

**(b) `max_num_sentences=1` 只合成第一句，其余静默丢弃。**

| `max_num_sentences` | 1 句 16 字 | 4 句 66 字 |
|---|---|---|
| **1** | 4218 ms | **4258 ms** ← 后 3 句消失 |
| 8 / 100 | 4218 ms | **16434 ms** |

66 字的文本被合成了 4.3 秒 —— 和 16 字单句一样长，**没有异常、没有日志、没有报错**。

**为什么这条对通话特别危险**：安全闸门保证**每个 `delta` 是一个完整句**，
所以"逐句送"时 `mns=1` 看起来"能用"。但 `revise`（整段改稿）与模板兜底
（`_generate_empathy_reply` 返回多句整段）**不是逐句的** —— 这些路径会只念第一句。
这正是本项目最怕的静默失效类型（对比 `video_call_build_plan.md` §8.5 的
8001 用错解释器事故：服务正常、只坏一个端点、无人察觉）。

**影响 S1 的三个必须落死的配置：**

| 配置 | 值 | 不这么做的后果 |
|---|---|---|
| `max_num_sentences` | **`-1`** | 只念第一句（上面这张表） |
| `num_threads` | `clamp(核数-2, 2, 8)` | 1 线程 RTF 0.568 → 直接不达标 |
| 送合成前的术语替换 | 见下 | `CBT` 这类缩写被吞掉 |

**附带实测**：`fanchen-C` 的 `lexicon.txt` **不含任何 ASCII 字母**，
合成时会打 `Ignore OOV 'CBT'` —— 一个心理学产品回复里的 `CBT`
会被**完全跳过**（用户听到"我们试试 里的一个方法"）。对策是送合成前过一张
**中文替换表**（`CBT` → `认知行为疗法`），属于话术层，不涉及安全规则。

---

### F-7 —— "首声"必须量「请求 → 音频到手」，不能量「排到它出声」

**这条是 S3 做骨架时被自己吓出来的，值得单列**：骨架面板一开始报"首声 6919 ms"，
看起来像整个链路坏了。同一时刻的 WS 帧时间线却是：

| 事件 | 相对页面加载 |
|---|---|
| 客户端发 `say`（3 句同时入队） | 721 ms |
| 服务端回 `start` | 729 ms |
| 收到第 1 句的 PCM 与 `end`（`synth_ms=350.4`、`rtf=0.149`） | **1079 ms** |

也就是**真实 358 ms**。差了 20 倍。原因不在链路，在**指标定义**：

* `TtsChannel.takeFirstSoundMs()` = `say()` → `onStart` 触发；
* `onStart` 是**排程时刻**（"该它出声了"），而 3 句是**一起**入队的，
  第 3 句本来就要等前两句播完（实测那两句音频共 6.5 s）——
  **6919 ms 是它正确的排队位置，不是慢**。

用纯客户端实验（`_probe_player_clock.js`，把网络整个摘掉、直接灌 3 段 2 秒静音）量出的规律：

| 场景 | 第 1 段 | 第 2 段 | 第 3 段 |
|---|---|---|---|
| 3 段一起入队（队列非空） | 入队后 **2 ms** 出声 | 入队后 **1962 ms** | 入队后 **3900 ms** |
| 每段播完再入队（队列空） | **1 ms** | **0 ms** | **0 ms** |

**队列空时 `onStart` ≈ 入队时刻**（`delayMs ≈ 0`）；队列非空时它等于"排队位置"。
所以"首声"这个数只能取**本轮第一段**，而且要取**「请求 → 音频到手」**。

**顺带量清楚的两件事**（都影响 S4）：

1. `AudioContext` 时钟在**空闲**时走不准：什么都不播时，3.00 s 墙钟里
   `ctx.currentTime` 只走了 **2.08 s**（≈0.69×）。但**一旦有音频在播就是 1:1**
   （上表第 2/3 段的 `ctx` 增量与墙钟增量逐毫秒吻合）。
   ⇒ 这就是为什么 `TtsStartTiming.startsAtWallMs` 只能当近似值（S2 已写明），
   也是为什么"排程是否无缝"必须用 **ctx 域**的排程值来校验。
2. 空闲期的这个偏差**不会**影响排程正确性：入队时 `startAt` 与 `ctx.currentTime`
   在同一个域里，差值仍是真实排队量。

**落地的改动**：`TtsStreamSession` 新增 `pcmReadyMs`（在 `end` 帧里、
**送播放器之前**记），`TtsChannel.takePcmReadyMs(seq)` 对外暴露；
`takeFirstSoundMs()` 语义不变（"什么时候出声"），但**文档与 UI 都不再把它叫首声**。

---

## 2. 步骤总表

| 编号 | 内容 | 新建 / 改动 | 闸门（可机器验证） | 量 | 依赖 |
|---|---|---|---|---|---|
| **S0** | ~~TTS 可行性 benchmark~~ **✅ 已完成** | `tools/download_asr_model.py`、`tools/benchmark_tts.py` | **RTF 0.180 / 0.112（空载），0.209 / 0.136（满载）→ 判据 0.4 通过** | 中 | —— |
| **S1** | ~~本地 TTS 引擎 + WS 端点~~ **✅ 已完成** | 新建 `algorithm/tts/*`、`algorithm/api/tts.py`、`tests/test_tts_engine.py`；改 `dataclasses.py`、`__init__.py`、`router.py`、`test_dataclasses.py` | **9/9 端到端检查通过**（见 S1.7） | 中 | S0 |
| **S2** | ~~前端播放队列~~ **✅ 已完成** | 新建 `client/src/services/ttsPlayer.ts`、`ttsStream.ts` | **浏览器实测 17/17 通过**（见 S2.5）：排程缝隙 `[0,0,1,-1]` ms、首段合成 347 ms、`stopAll()` 0.1 ms | 小 | —— |
| **S3** | ~~通话页骨架 + 路由 + 入口~~ **✅ 已完成** | 新建 `pages/patient/VoiceCall.tsx`；改 `App.tsx`、`Chat.tsx`；`ttsStream.ts` 补 `takePcmReadyMs` | **浏览器实测 27/27 通过**（见 S3.6）：无侧边栏、只建 1 个会话、离开后 0 live 轨道 / 0 定时器 / 0 未处理 rejection | 中 | —— |

| **S4** | ~~轮次编排 `useVoiceCall`~~ **✅ 已完成** | 新建 `hooks/useVoiceCall.ts`；重写 `pages/patient/VoiceCall.tsx`；修 `useVoiceAnalysis.ts`、`ttsPlayer.ts` 的 AudioContext 关闭拒绝 | **浏览器实测 14/14 通过**（见 S4.7）：ASR 建连、定稿上屏、user+assistant 双落库、TTS 收到 PCM、首段合成 246 ms、自动回聆听、挂断 0 泄漏 | 大 | S1 S2 S3 |
| **S5** | ~~打断 + 回声过滤~~ **✅ 已完成** | 新建/重写 `services/echoFilter.ts`；改 `mediaStreamManager.ts`、`streamingAsr.ts`、`useVoiceAnalysis.ts`、`useVoiceCall.ts`；新增 `_exp_echo_coupling.py`、`_exp_echo_decide.js`、`docs/echo_calibration.md`、`_probe_call_s5.js` | 阈值**实测标定 0.40**（30/30 真回声全中、普通负样本 0 误判）；插话 → 本地静音 **0 ms**；并发轮次峰值 **1**（见 §S5.6） | 大 | S4 |
| **S6** | ~~危机切换~~ **✅ 已完成** | 新建 `components/CrisisResourceCard.tsx`；改 `Chat.tsx`、`VoiceCall.tsx`、`useVoiceCall.ts` | 危机轮**0 个 PCM 帧**、0 次 `say`、卡片只上屏号码、算法层 `requires_escalation=true`（见 §S6.5） | 中 | S4 |
| **S7** | ~~视觉进通话~~ **✅ 已完成** | 新建 `hooks/useCallVision.ts`；改 `VoiceCall.tsx` | **开/关摄像头两轮 → 落库 `riskLevel` 完全一致**；全程 0 条视觉上报；拒权限不中断通话（见 §S7.1） | 大 | S6 |
| **S8** | ~~生产可用性（`/algorithm` 代理）~~ **✅ 已完成** | 新建 `server/src/services/algorithmProxy.ts`；改 `server/src/app.ts` | 生产构建下 `/algorithm` 的 **HTTP + WebSocket 都握手成功（6/6）**（见 §S8.1） | 小 | S4 |
| **S9** | ~~演示准备（预热 + 走查）~~ **✅ 已完成** | 新建 `scripts/warmup.mjs`、`_probe_prod_proxy.js`；改 `package.json`（`npm run warmup`） | 预热 **8/8**；走查清单见 §S9.4 | 小 | S6 |

---

## 3. 逐步施工单

### S0 —— TTS 可行性 benchmark ✅ **已完成（Path A 成立）**

**完整报告**：[`docs/tts_benchmark.md`](./tts_benchmark.md)。
**原始数据**：`docs/tts_benchmark.json`（4 线程）、`docs/tts_benchmark_t8.json`（8 线程）。

#### S0.5 结果

| 判据 | 4 线程 | 8 线程 | 判定 |
|---|---|---|---|
| RTF < 0.4（空载） | 0.180 | **0.112** | ✅ 余量 2–3 倍 |
| RTF < 0.4（满载，同时跑流式 ASR） | 0.209 | **0.136** | ✅ |
| 最短句整句合成 < 400 ms（满载） | 507 ms | **336 ms** | ⚠️ 4 线程不过，8 线程过 |

**结论：Path A 技术上成立**，D1 因此有了"可以选本地"的资格（但仍需人拍板，§7 Q1）。
按 8 线程配置，合成速度是播放速度的 **7 倍**，播放队列在结构上不可能饿死。

**修正后的首声预算 ≈ 1.35–1.97 s，中位 1.6 s**（TTS 贡献 0.26–0.48 s）——
与 `voice_call_plan.md` 原估计一致，但那是两个反向误差相互抵消的结果（见报告 §1/§5）。

#### S0.6 本次实测踩到、并被修掉的三件事

| # | 现象 | 处理 |
|---|---|---|
| 1 | 下载器 read timeout 后**从 0 重下**（121 MB 跑到 82% 断掉，重试两次都失败） | 改成 `.part` + `Range` **断点续传**，并在服务端不支持 Range 时清空重下 |
| 2 | 预设的 `required` 文件列表**照抄文档、缺了整个 `dict/` jieba 词典** | 改为**按 hf-mirror 的真实文件列表**列出（`vits-zh-hf-*` 系列必须有 `dict/`，否则加载直接失败） |
| 3 | 原判据"首块 < 250 ms"**结构性不可达** | 改为"最短句整句合成 < 400 ms"，理由见 F-6(a) |

#### S0.7 复现命令（保留，因为这是 go/no-go 判据的复现工具）

```powershell
# 1) 下载模型（137.7 MB，走 hf-mirror，支持断点续传）
& 'D:\Develop\AIC\.venv\Scripts\python.exe' algorithm/tools/download_asr_model.py --preset tts-zh-fanchen-C

# 2) benchmark（--contention 会同时拉起项目自己的流式 ASR 制造真实争用）
& 'D:\Develop\AIC\.venv\Scripts\python.exe' algorithm/tools/benchmark_tts.py --repeat 5 --num-threads 8 --contention --json docs/tts_benchmark_t8.json

# 3) 听音质 —— **这一步必须由人做，脚本量不出来**
#    _tts_bench/idle_*.wav  _tts_bench/load_*.wav
```

落盘：`algorithm/models/tts/vits-zh-hf-fanchen-C/`（`algorithm/models/` 已在 `.gitignore`）。

> **仍然只能靠人的两件事**（别当成已验证）：
> ① 中文音质好不好 —— WAV 已导出，**等人耳**；
> ② 端到端首声 —— S0 全是**进程内**直调 `generate()`，没经过网络与浏览器，
> 真正的端到端首声要等 S4 做完在浏览器里再测（S4.6）。

---

### S1 —— 本地 TTS 引擎 + WebSocket 端点 ✅ **已完成**

> **执行结果见 S1.7。** 下面保留设计理由与协议定义（S1.7 是"做完了长什么样"）。

> **F-6 带来的两处改动，写在最前面（照着原设计做会踩坑）：**
>
> 1. **协议简化**：VITS 回调只给一次完整音频 ⇒ 一个 `say` 对应
>    `start` → **1 个**二进制帧 → `end`。不需要"多块拼接"的逻辑。
> 2. **三个配置必须落死**：`max_num_sentences=-1`、`num_threads=clamp(核数-2,2,8)`、
>    送合成前过一张**中文术语替换表**（词表无拉丁字母，`CBT` 会被吞）。

#### S1.1 数据类（AGENTS.md 强制：统一定义在 `shared/dataclasses.py`）

改 `algorithm/shared/dataclasses.py`，在「语音识别数据类」段（`:286`）之后新增一段：

```python
# ============================================================
# 语音合成数据类
# ============================================================

class TtsEngine(str, Enum):
    """语音合成引擎枚举

    只落地本地 sherpa-onnx VITS 一条路径。刻意不提供 `speechSynthesis`：
    它的音频不经过 Web Audio，拿不到 MediaStream / AudioNode，
    无法作为回声消除的参考信号，也无法做文本级自回声过滤。
    """
    VITS = "sherpa-onnx-vits"     # 本地离线合成（/tts/ws）
    UNAVAILABLE = "unavailable"   # 模型缺失或加载失败


@dataclass
class TtsSynthesisResult:
    """一次语音合成的结果（用于日志与前端展示，不承载音频数据）

    Attributes:
        seq: 本次合成的序号，与请求里的 seq 对应
        engine: 实际使用的合成引擎
        sample_rate: 输出 PCM 的采样率（Hz）
        audio_ms: 产出音频时长（毫秒）
        first_chunk_ms: 从调用到第一次回调的耗时（毫秒）；判据之一
        synth_ms: 合成总耗时（毫秒）
        cancelled: 是否被中途取消
        error: 失败原因；为空字符串表示成功
    """
    seq: int
    engine: TtsEngine = TtsEngine.VITS
    sample_rate: int = 16000
    audio_ms: float = 0.0
    first_chunk_ms: float = 0.0
    synth_ms: float = 0.0
    cancelled: bool = False
    error: str = ""

    @property
    def rtf(self) -> float:
        """实时率 = 合成耗时 / 音频时长；< 1.0 才追得上播放"""
        return self.synth_ms / self.audio_ms if self.audio_ms > 0 else 0.0
```

同步更新 `algorithm/shared/__init__.py` 的导出列表，
并在 `algorithm/tests/test_dataclasses.py` 补测试（AGENTS.md 强制）。

#### S1.2 引擎

新建 `algorithm/tts/__init__.py`（空）与 `algorithm/tts/engine.py`。

**严格仿 `algorithm/asr/engine.py:163-320` 的写法**：`_instance` + `_instance_lock` 双检锁、
`available` / `load_error` / `load()` 懒加载、`status() -> dict`、中文 docstring、完整类型注解。

```python
class VitsTtsEngine:
    """本地离线中文 TTS 引擎（sherpa-onnx OfflineTts 单例）

    与 `asr/engine.py` 的 `ParaformerRecognizer` 同构：懒加载、失败原因记在
    `load_error` 而不是抛异常 —— 服务能起来但合成能用/不能用，必须能被
    `/tts/status` 如实报出来（这个项目已经因为「服务起来了但某能力静默失效」
    坏过一次：8001 用错解释器 → `/asr/ws` 404）。
    """

    def synthesize(
        self,
        text: str,
        seq: int,
        on_chunk: Callable[[np.ndarray], int],
    ) -> TtsSynthesisResult:
        """合成一句话，边合成边通过 `on_chunk` 推 PCM。

        Args:
            text: 待合成文本（调用方保证已过安全闸门）。
            seq: 本次合成的序号，原样带回结果。
            on_chunk: 每产出一块 float32 波形调用一次；**返回 1 中止合成**。
                （以 sherpa-onnx 1.13.8 的实际 Python API 为准 ——
                落地第一步先写 10 行探针确认回调签名与中止语义。）

        Returns:
            TtsSynthesisResult: 计时与状态；音频不在这里，已经通过 on_chunk 推走。
        """
```

**模型目录解析**：仿 `asr/engine.py:64 resolve_model_dir(explicit)` 的写法，
新增 `resolve_tts_model_dir()`，支持环境变量 `TTS_MODEL_DIR` 覆盖。

#### S1.3 WS 端点

新建 `algorithm/api/tts.py`。端点声明风格照抄 `api/asr.py`。

```python
router = APIRouter(prefix="/tts", tags=["语音合成"])

@router.websocket("/ws")
async def tts_stream(websocket: WebSocket) -> None: ...

@router.get("/status", response_model=TtsStatusResponse)
def status() -> TtsStatusResponse: ...
```

**帧协议**（与 `api/asr.py:94-201` 的 `asr/ws` 风格一致：文本帧 JSON 控制 + 二进制帧数据）：

```
客户端 → 服务端（文本帧 JSON）
  {"type":"say","seq":1,"text":"听起来这段时间挺难熬的。"}
  {"type":"cancel","seq":1}          中止当前合成（并清空服务端待合成队列）
  {"type":"close"}                   结束本次通话的合成会话

服务端 → 客户端
  {"type":"status","available":true,"engine":"sherpa-onnx-vits",
   "sample_rate":16000,"error":""}          连接建立后立即下发
  {"type":"start","seq":1,"sample_rate":16000}
  二进制帧 × N   —— int16 小端、单声道 PCM，按顺序拼接即为完整音频
  {"type":"end","seq":1,"audio_ms":2400.0,"first_chunk_ms":138.0,
   "synth_ms":410.0,"rtf":0.17}
  {"type":"cancelled","seq":1}
  {"type":"error","seq":1,"message":"……"}
```

**为什么二进制帧外面要包 `start` / `end` 文本帧**：二进制帧自身没有边界语义，
前端无法知道"这一帧属于哪一句"。服务端**串行合成** ⇒ 同一时刻只有一路音频流，
`start` / `end` 足以定界。

> **F-6(a) 的修正**：不要按"一句会来很多块"去写客户端拼装逻辑。
> 实测 VITS 的回调**只触发一次**、携带整句音频，所以每个 `say` 对应的就是
> **1 个二进制帧**。客户端仍应写成"累积到 `end` 为止"的循环（这样将来换成
> 真流式 TTS 时不用改），但**不要**依赖"块数 > 1 才能工作"。

#### S1.3b 三个必须落死的配置（F-6 的结论）

| 配置 | 值 | 依据 / 不这么做的后果 |
|---|---|---|
| `max_num_sentences` | **`-1`** | `=1` 时只合成第一句，其余**静默丢弃**（66 字文本只产出 4.3 s 音频）。见 `docs/tts_benchmark.md` §1 发现三 |
| `num_threads` | `clamp(CPU核数 - 2, 2, 8)` | 1 线程 RTF **0.568**（不达标）；8 线程 0.112。给 ASR 与浏览器留核 |
| 送合成前的**术语替换表** | 见下 | 词表**不含任何 ASCII 字母**，`CBT` 会被 `Ignore OOV` 吞掉 |

**必须新增的两条回归测试**（这是发现三的唯一可靠护栏）：

```python
# algorithm/tests/test_tts_engine.py
def test_multi_sentence_is_not_truncated() -> None:
    """一段 N 句文本「整体送一次」的音频时长，必须显著大于「只送第一句」。

    这条测试就是 max_num_sentences=1 那个坑的复现方式：
    配错时两者几乎相等（4218 vs 4258 ms），测试立刻红。
    """
```

```python
# 术语替换表（话术层，不涉及安全规则；由 reviewer 增补）
PRONOUNCE_FIX: dict[str, str] = {
    "CBT": "认知行为疗法",
    # 词表无拉丁字母，所有缩写都要在这里换成可发音的中文
}
```

#### S1.4 并发形态 ⚠️ 这里最容易写错

**必须串行合成**（`voice_call_plan.md` §5 阶段 1 已论证，先例是
`asr/punctuation.py:161` 的"会话不保证并发安全 → 加锁串行化"）。

⚠️ **不要用 `run_in_threadpool`** 来跑合成：它走 anyio 的共享限流器，会并发。
必须用**专属的单工作线程池**：

```python
# algorithm/api/tts.py（形态示意，落地时按项目风格补 docstring 与注解）
_tts_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

loop = asyncio.get_running_loop()
queue: asyncio.Queue[tuple[str, bytes | None]] = asyncio.Queue()

def on_chunk(samples: np.ndarray) -> int:
    """在合成线程里被调用 → 必须回到事件循环才能写 socket

    ⚠️ 实测 VITS 只调用**一次**（合成结束后）。所以这里的 `return 1` 只能
    阻止把音频发出去，**省不下计算** —— 那时算力已经花完。取消要靠
    「清空待合成队列 + 丢弃在途结果」，不是靠这个返回值。
    """
    if cancel_flag.is_set():
        return 1                      # 不再发送（不是"中止计算"）
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    loop.call_soon_threadsafe(queue.put_nowait, ("pcm", pcm))
    return 0

sender = asyncio.create_task(_drain(queue, websocket))     # 只有事件循环写 socket
await loop.run_in_executor(_tts_pool, engine.synthesize, text, seq, on_chunk)
```

三条硬约束：

1. **socket 只能由事件循环写**（`asyncio` 对象不是线程安全的）→ 一律 `call_soon_threadsafe`。
2. **CPU 密集合成不能占事件循环** → 必须在线程里。
3. **`max_workers=1`** → 串行化免费获得，不需要额外加锁。

#### S1.5 注册路由

`algorithm/api/router.py` 追加（照抄 `asr_router` 的三行写法）：

```python
from api.tts import router as tts_router
...
# 本地语音合成（陪伴通话的出声端；音频不出设备）
api_router.include_router(tts_router)
```

#### S1.6 闸门

```powershell
# 1. 状态必须如实报告
curl.exe -s http://127.0.0.1:8001/api/v1/tts/status

# 2. 用现成 fixture 跑一遍 WS（复用 _probe_ws_e2e.py 的分帧逻辑）
#    判据：收到 ≥1 个二进制帧；拼起来是 16 kHz 单声道；ffplay 能听出人话
```

| 检查 | 判据 |
|---|---|
| `/tts/status` | `available: true`，`engine: sherpa-onnx-vits`，`num_threads` 如实报出 |
| WS 合成耗时 | `synth_ms` 与 S0 benchmark 同量级（±30%）；`rtf` 字段 < 0.4 |
| **不截断** | 送 4 句话 → 音频时长 ≈ 4 × 单句时长（**不是** ≈ 1 × 单句） |
| **取消粒度** | 见下：`cancel` 的语义是**"不再送后续句子 + 丢弃在途结果"**，不是"砍掉当前这句的后半段" |
| 串行 | 连发 3 条 `say` → 按序返回 3 组 `start…end`，无交叉 |
| 回归 | `pytest algorithm/tests` 通过数**不减少** |

> **取消语义要按实测改**（F-6(a)）：回调只在**合成结束后**触发一次，
> 所以"回调返回 1 中止合成"实际上**省不下计算**（那时算力已经花完），
> 只能阻止把音频发出去。
> 正确的取消语义是：**清空待合成队列 + 丢弃在途结果 + 不发二进制帧**。
> 而"用户感知的即时打断"本来就靠**客户端 `stopAll()` 立刻停播**（S2/S5），
> 不依赖服务端。这与 `voice_call_plan.md` §6.2 的"先清本地队列，再通知服务端"一致。

#### S1.7 执行结果（✅ 本轮完成）

**改动的文件**

| 文件 | 内容 |
|---|---|
| `algorithm/tts/engine.py` | **新建**：`VitsTtsEngine` 单例（懒加载 / 失败原因如实上报 / `synthesize` / `status`），含三个实测配置与截断护栏 |
| `algorithm/tts/pronunciation.py` | **新建**：术语→中文替换表 + `unpronounceable_spans`（丢内容必须可观测） |
| `algorithm/tts/__init__.py` | **新建** |
| `algorithm/api/tts.py` | **新建**：`WS /tts/ws`（receiver + worker 双协程）、`GET /tts/status` |
| `algorithm/shared/dataclasses.py` | 新增 `TtsEngine` 枚举 + `TtsSynthesisResult`（含 `rtf` / `ok` / `is_silent`） |
| `algorithm/shared/__init__.py` | 同步导出 |
| `algorithm/api/router.py` | 注册 `tts_router` |
| `algorithm/tests/test_tts_engine.py` | **新建**：27 个测试（含截断回归测试） |
| `algorithm/tests/test_dataclasses.py` | 新增 10 个测试 |

**端到端验收（`D:\Develop\AIC\_probe_tts_ws.py`，9/9 通过）**

| # | 检查项 | 结果 |
|---|---|---|
| A | 握手 `status` 如实报出引擎与配置 | ✅ `sample_rate=16000`、`num_speakers=187`、`num_threads=8` |
| B | 一句 → `start` + **1 个**二进制帧 + `end` | ✅ 帧数 = 1（F-6(a) 在线上的体现） |
| B | RTF < 0.4 | ✅ **0.154** |
| C | **多句不截断**：4 句 / 1 句 音频时长比 | ✅ **7.28**（配错 `max_num_sentences=1` 时 ≈ 1.0） |
| D | `cancel` 后不交付音频 | ✅ `cancelled`，0 帧 |
| E | 术语替换 | ✅ `我们试试 CBT 里的方法` → `spoken_text = "我们试试 认知行为疗法 里的方法"` |
| F | PCM 格式与 `audio_ms` 自洽 | ✅ 75354 字节 = 2355 ms @ 16 kHz 单声道 int16 |

**`/tts/status` 实测输出**

```json
{"available": true, "engine": "sherpa-onnx-vits",
 "model_dir": "…/algorithm/models/tts/vits-zh-hf-fanchen-C",
 "model_name": "vits-zh-hf-fanchen-C", "sample_rate": 16000,
 "num_speakers": 187, "num_threads": 8, "max_num_sentences": -1, "error": ""}
```

**回归**

| 项 | 结果 |
|---|---|
| `pytest tests/` | **1169 passed / 8 failed / 1 skipped**（基线 1132/8/1，**新增 37 个测试全部通过**） |
| 8 个失败 | 全部追溯到一个根因：`audit/fairness.py:492` 与 `federated/run_simulation.py:157` 的 `import matplotlib`（**预先存在，与本次无关**） |
| `/asr/status` | ✅ `streaming_available = True`（旧能力未被破坏） |

**护栏自检（一条不会失败的回归测试等于没有）**

`D:\Develop\AIC\_probe_tts_guard.py` 把 `max_num_sentences` 改回 `1` 再跑那条测试：

```
① 配置正确（-1）→ 1 passed，退出码 0
② 配置改坏（1）→ 1 failed，退出码 1   ← 护栏真的会响
源文件已还原：✅
```

**人耳试听的结果：不合格**（"声音好难听"）→ 触发了 §11 的音色与模型排查。
其中修掉了 `tts/engine.py` 的一个**静默降级** bug：`TTS_MODEL_DIR` 指向
onnx 文件名不同的模型时被判"不可用"，然后**无声回退到默认模型** ——
详见 [`tts_benchmark.md`](./tts_benchmark.md) §10.3，已配 8 条 `TestModelDirResolution` 护栏。

**已补齐的两项**

* **经 vite 代理（5173 → `/algorithm`）的 WS** —— 已在 S2.5 的浏览器实测中验证（✅）。
* **音色/模型候选** —— 见 §11。

---

### S2 —— 前端播放队列（Web Audio）✅ **已完成**

> 产出两个文件：`ttsPlayer.ts`（播放）+ `ttsStream.ts`（协议 + `TtsChannel` 组合）。
> **执行结果见 S2.5。**

新建 `client/src/services/ttsPlayer.ts`。

```ts
export interface TtsPlayerCallbacks {
  /** 一句开始播（seq） */
  onStart?: (seq: number) => void;
  /** 一句播完（seq, 已播文本）—— 文本用于回声过滤 */
  onEnd?: (seq: number, text: string) => void;
  /** 队列彻底空了（可以恢复"聆听"状态） */
  onDrained?: () => void;
}

export class TtsPlayer {
  /** 建一个独立 AudioContext（采样率用模型值；与 16kHz 的 ASR 上下文分开） */
  constructor(cb?: TtsPlayerCallbacks);
  /** 追加 PCM；入队即按 nextStartTime 排程，句间无缝 */
  enqueue(seq: number, pcm: Int16Array, sampleRate: number, text: string): void;
  /** 立刻停播（stop + disconnect），不等待 —— 打断路径用 */
  stopAll(): void;
  get isPlaying(): boolean;
  /** 最近 N 秒播过的文本，供文本级自回声过滤比对 */
  recentSpoken(withinMs?: number): string[];
  dispose(): void;
}
```

**为什么必须用 `AudioBufferSourceNode` 而不是 `<audio>`**（`voice_call_plan.md` §5 阶段 1）：

1. `<audio>` 每句之间会有可听的接缝，通话里很突兀；
2. `AudioBufferSourceNode.stop()` 是采样级精确的，`<audio>.pause()` 有延迟；
3. `<audio>` 的元素级播放会被系统的媒体会话接管（锁屏控制、蓝牙路由），行为不可控；
4. 需要 `recentSpoken()` 这个"我刚刚说了什么"的记录 —— 用自己的播放器才有。

**排程写法（关键）：**

```ts
const startAt = Math.max(this.ctx.currentTime, this.nextStartTime);
source.start(startAt);
this.nextStartTime = startAt + buffer.duration;
```

**闸门**：连播 5 句不卡顿、句间无可听静默缝；`stopAll()` 到 `isPlaying === false` < 50 ms；
`dispose()` 后 `ctx.state === 'closed'`、0 条 live 轨道。

#### S2.5 执行结果（✅ 本轮完成）

**新增文件**

| 文件 | 内容 |
|---|---|
| `client/src/services/ttsPlayer.ts` | `TtsPlayer`：`AudioBufferSourceNode` 队列、`nextStartTime` 排程、`stopAll()`、`recentSpoken()`、`dispose()` |
| `client/src/services/ttsStream.ts` | `TtsStreamSession`（协议）+ `TtsChannel`（合成+播放门面，S4 直接用） |

**浏览器实测（`D:\Develop\AIC\_probe_tts_player.js`，CDP + Edge，17/17 通过）**

验的是**真实浏览器里的 Web Audio + 经 vite 代理的真实 WebSocket** —— 不是 mock，
因为这两样恰好最容易"代码看着对、跑起来不对"。

| # | 检查项 | 结果 |
|---|---|---|
| A | 经 5173 的 `/algorithm` 代理连上 `/tts/ws` | ✅ `engine=sherpa-onnx-vits sr=16000 threads=8`（**顺带补上了 S1 欠的代理验证**） |
| B | 5 句都合成并播完（5 次 `onEnd`） | ✅ |
| B | 自然播完时 `onDrained` 触发一次 | ✅ |
| C | **句间无缝隙**（排程时钟偏差 < 5 ms） | ✅ **`[0, 0, 1, -1]` ms** |
| C | 端到端首声（`say()` → 真正出声） | ✅ **347 ms** —— 这一句是**队列空**时入队的，所以"出声"≈"到手"；第 2 段起两者会分离，见 §1 F-7 |
| D | `recentSpoken()` 含刚播完的那句 | ✅ |
| E | `spoken_text` 已是术语替换后的文本 | ✅ 含「认知行为疗法」 |
| F | `stopAll()` 同步完成 | ✅ **0.1 ms** |
| F | `stopAll()` 后 `isSpeaking === false` | ✅ |
| F | 被掐断的句子也进了 `recentSpoken` | ✅（回声过滤必须知道"AI 刚说了半句"） |
| G | `say()` 非阻塞 | ✅ 两次调用共 0 ms |
| H | `dispose()` 后 `isDisposed=true`、`isSpeaking=false` | ✅ |
| H | AudioContext 采样率为设备值 | ✅ 48000 Hz（模型 16000，播放时重采样） |

**期间修掉的一个真 bug**：`connect()` 原本在 `ws.onopen` 就兑现，此时服务端的 `status`
帧还没到 —— 于是"连接明明通了、5 句都合成出来了，`ch.status` 却全是 0"。
现在改为**由 `status` 帧兑现**，调用方在说第一句话之前就能可靠判断"服务端能不能合成"。

**两条测量方法上的教训（都已写进代码注释，避免重踩）**

1. **校验"句间有无缝隙"不能用墙钟**。实测 `#1 enqueue@1112(ctx 0)` ——
   AudioContext 的时钟要等第一个 source 开始渲染才起步，早期按"ctx 与墙钟 1:1"
   外推会**凭空多出 536 ms**，看起来像一道大缝，其实排程是 `0 ms` 无缝。
   正确做法是让播放器回报**排程时的 ctx 起点**（精确、不受回调时刻影响）。
2. **也不能在 `onStart` 回调里现读 `currentTime`**：`setTimeout` 的抖动在无头浏览器下
   实测 ±35 ms，会把精确的排程误判成"有缝"。同上，用排程值。

**为什么 `onDrained` 在 `stopAll()` 时也会触发**：队列空了是**事实**，不是"自然播完"的
独有信号。S4 必须把它当成幂等的状态收敛（已在代码与本文档两处写明）。

**回归**：`client tsc -b` ✅；`client` 生产构建 ✅（3987 模块，9.06 s，exit 0）。

**换到 22 kHz 模型后重跑（证明客户端没有假设采样率）**

| 项 | 16 kHz `fanchen-C` | 22 kHz `theresa` |
|---|---|---|
| S2 闸门 | 17/17 | **17/17** ✅ |
| 端到端首声 | 347 ms | **237–272 ms** |
| 句间排程偏差 | `[0,0,1,-1]` ms | **`[0,0,0,0]`** ms |
| 5 句总墙钟 | 19.8 s | **13.1 s** |

客户端一行未改。原因：`ttsStream` 用服务端帧里的 `sample_rate`、
`TtsPlayer` 用 `ctx.createBuffer(1, len, sampleRate)` 由浏览器重采样、
探针报的也是模型自述值 —— **三处都没有硬编码 16000**。

---

### S3 —— 通话页骨架 + 路由 + 入口

#### S3.1 新建 `client/src/pages/patient/VoiceCall.tsx`

静态骨架先做（声音先用假数据），布局见 `video_call_build_plan.md` §3.1 的线框图。

#### S3.2 路由（**必须放在 `PatientLayout` 之外**）

`client/src/App.tsx:49` 附近，与 `/login` 同级：

```tsx
{/* AI 陪伴通话：整屏页面，刻意不进 PatientLayout（不要侧边栏） */}
<Route path="/call" element={<PrivateRoute><RoleRoute role="PATIENT"><VoiceCall /></RoleRoute></PrivateRoute>} />
<Route path="/call/:conversationId" element={<PrivateRoute><RoleRoute role="PATIENT"><VoiceCall /></RoleRoute></PrivateRoute>} />
```

> 放在 `:53` 那个 `/` 下面会带上侧边栏，通话就不像电话了。

#### S3.3 入口

`client/src/pages/patient/Chat.tsx` 顶部加一个电话按钮（`<PhoneOutlined />`）：

* 已有会话 → `navigate('/call/' + conversationId)`；
* 无会话 → `navigate('/call')`，通话页进入时自己建一个 conversation
  （复用 Chat.tsx 里建会话的同一个 API 调用，不要另写一份）。

#### S3.4 会话生命周期

| 场景 | 行为 |
|---|---|
| 无 `:conversationId` | 进入时创建会话（复用 Chat 的 createConversation），成功后 `replace` 到 `/call/<id>` |
| 挂断 | 冲刷最后一段定稿 → 等 SSE `done` → `navigate('/chat/<id>')` |
| 直接关标签页 | 靠 unmount / `beforeunload` 清理（见闸门） |

#### S3.5 闸门 ⚠️ 这一条最容易被忽略

> **离开通话页后：0 条 live 媒体轨道、0 个 `setInterval`、0 个未处理的 Promise rejection。**

复用 `_probe_media_fix.js` 的验收手法（它对"媒体流生命周期"已经有了可跑的实现，
直接扩一个"进通话页 → 挂断 → 检查不变量"的用例）。

---

#### S3.6 执行结果（✅ 本轮完成）

**改动文件**

| 文件 | 内容 |
|---|---|
| `client/src/pages/patient/VoiceCall.tsx`（**新建**） | 整屏通话页：头像律动 / 状态词 / 字幕 / 三个控制按钮；会话生命周期；麦克风借用与归还；计时器；`?demo=1` 的骨架自检面板 |
| `client/src/App.tsx` | 加 `/call` 与 `/call/:conversationId`，**与 `/login` 同级** |
| `client/src/pages/patient/Chat.tsx` | 对话历史卡片顶部加电话入口（`<PhoneOutlined />`） |
| `client/src/services/ttsStream.ts` | 补 `pcmReadyMs` / `takePcmReadyMs()` —— 修 F-7 那个指标（见 §1 F-7） |
| `D:\Develop\AIC\_probe_call_page.js`（**新建**，仓库外） | S3 闸门探针：CDP + Edge 假设备 + 真登录，28 项检查 |
| `D:\Develop\AIC\_probe_strip_comments.js`（**新建**，仓库外） | 带字符串状态的去注释器，供 P0b 做"新鲜度比对"（可单独运行自检） |
| `D:\Develop\AIC\_probe_call_tts_timing.js`、`_probe_player_clock.js`（**新建**，仓库外） | 诊断 F-7 的两次实验（帧时间线 / 纯客户端时钟隔离），留档备查 |

**浏览器实测（`D:\Develop\AIC\_probe_call_page.js`，CDP + Edge 假设备 + 真登录，27/27 通过）**

| # | 检查项 | 结果 |
|---|---|---|
| P0 | dev server 已在服务新页面与新路由，且 `/call` 出现在 `/` 路由**之前**（⇒ 不在 PatientLayout 里） | ✅ |
| P0b | **dev server 服务的模块与磁盘一致**（防 stale module graph） | ✅ 见下方"两条方法学教训" |
| P1 | 真登录（`patient@mental.com`） | ✅ |
| P2 | `/chat` 顶部有电话入口 | ✅ |
| P3 | 无会话点电话 → 走到 `/call/<cuid>` | ✅ |
| P3b | **只建了 1 个会话** | ✅ `15 → 16`（StrictMode 双跑没有多建，见下） |
| P4 | 通话页无 `.ant-layout-sider`（确实整屏） | ✅ |
| P5 | 点"开始通话" → 状态词变「正在聆听」 | ✅ |
| P5b | 恰好 **1** 条 live 音轨 | ✅ |
| P5c | `getUserMedia` 的 audio 调用**恰好 1 次** | ✅ |
| P5d | 通话中 1000 ms 计时器在跑 | ✅ |
| P6 | 静音 → 轨道 `enabled=false` 但**仍然 live**（静音 ≠ 挂断） | ✅ |
| P7 | 取消静音 → `enabled=true` | ✅ |
| P8a | `?debug=1` 出现诊断面板，且如实报出 ASR / TTS 引擎 | ✅ |
| P8b | 没有语音输入时，轮次显示「尚未发生」（不假装有轮次） | ✅ |

> ⚠️ P8 这一组在 S4 里**换过内容**：S3 时验的是 `?demo=1` 那个念预置台词的骨架面板
> （"首段合成 347–381 ms"就是那时测的），S4 按施工单要求把那个面板删掉了，
> 换成只报真实读数的 `?debug=1`，所以 S3 闸门从 28 项变成 27 项。
> 首声那一腿现在由 S4 闸门（`_probe_call_voice_e2e.js` V8）负责，读数更硬。
| P9 | 挂断 → 回到 `/chat/<id>` | ✅ |
| P9b | 通话页那个 1000 ms 计时器**已清掉**（按 id 认，不是笼统数总数） | ✅ |
| P10 | 离开后 live 媒体轨道 = **0** | ✅ |
| P10b | 未处理 rejection / 页面错误 = **0** | ✅ |
| P10c | 借过的音轨确实 `ended`（真的还了设备） | ✅ |
| P11 | 已有 `:conversationId` 时**不再建**会话 | ✅ `18 → 18` |
| P11b | 未接通就挂断：轨道 0 / 错误 0 | ✅ |
| P13 | 连点"开始通话" 3 次（**同步**调用 `onClick`，antd 的 `loading` 挡不住）→ 只建 **1** 个 TTS 通道、只申请 1 次麦克风 | ✅ `ws=1, gum=1, live=1` |
| P13b | 之后挂断：轨道归零、0 未处理 rejection | ✅ |
| P12 | **不刷新页面**进出 3 次，每次都能重新借到麦克风（3 次 `getUserMedia`） | ✅ `0 → 3` |
| P12b | 每次进入 1 条 live 音轨、每次离开归零 | ✅ 3/3 循环 |
| P12c | 3 次之后仍然 0 未处理 rejection / 页面错误 | ✅ |

> P12 为什么强调"不刷新页面"：整页 reload 时浏览器无论如何都会收走设备，
> 那种"3 次"什么都验不出来。只有 SPA 内部跳转（电话入口 → 挂断 → 再点电话）
> 才真正依赖引用计数与组件卸载清理。而 `getUserMedia` 恰好被调用 **3** 次这件事，
> 本身就证明了**每次挂断都真的把设备还回去了** —— 只要有一次没还，
> 下一次 `acquire` 会直接复用那份旧流，`getUserMedia` 就不会被再调一次。

**设计决定：为什么"开始通话"必须点一下**（不是偷懒，是两个硬约束的交点）

1. `AudioContext` 必须在**用户手势**里 `resume()`，否则自动播放策略会让第一句话没声音，
   且事后无法补救。
2. 麦克风申请若放在 effect 里，会撞上 React 18 StrictMode 的"假卸载"：
   先 cleanup（`release`）再重新 effect（`acquire`）。而 `mediaStreamManager` 的引用计数是
   **按消费者名去重的 Set** —— 同一个 `'call'` 记不了两次，于是后一次拿到的可能是
   **已经被 stop 掉的轨道**（麦看起来是开的、界面什么都不报，实际全是静音）。
   放进点击回调就只会申请一次，从根上绕开这个竞态。

> 这条是**读代码 + 推演**得出的，不是实测抓到的（StrictMode 的时序无法稳定复现）。
> 但两个约束各自独立成立，所以这个设计不只是为了绕 bug。

**顺带修掉的三个"自己的坑"**

1. **F-7 那个指标**：骨架面板一度报"首声 6919 ms"，真值是 358 ms —— 见 §1 F-7。
   已拆成 `takePcmReadyMs()`（请求→到手，S4 闸门用它）与 `takeFirstSoundMs()`（含排队），
   UI 也分别标注「首段合成」/「首段出声」。
2. **`/call` 无 id 时的会话创建要防 StrictMode 双跑**：用 ref 缓存那个 promise，
   否则会多建一个空会话留在用户的对话历史里（P3b 专门验了这条）。
3. **连点"开始通话"会泄漏一个 TTS 通道**（P13 抓到的）：第二次点击会新建第二个
   `TtsChannel` 并覆盖 `ttsRef.current`，第一个的 WebSocket 与 AudioContext 从此无人可达，
   挂断也关不掉。修法是**同步重入锁**（`startingRef`）+ `loading` 禁用按钮 ——
   只加 `loading` 不够：连点时 React 还没重渲染，`loading` 拦不住第二次
   （与 `video_call_build_plan.md` F1 是同一个坑）。

**两条方法学教训（写进探针注释，避免重踩）**

1. **别拿源码里的标记去比 dev server 服务的内容**：esbuild 把注释整个丢掉，而标识符与
   中文串大量出现在注释里（实测 `挂断` 源码 9 次、服务内容 1 次）—— 直接比次数会得到
   **恒假警报**。正确做法是两边都过一遍 `_probe_strip_comments.js`（带字符串状态的
   去注释器，已单测：去注释后的 8 个标记次数与服务内容**逐一相等**）。
2. **别用"带随机 query 再取一次"当新鲜度基准**：未知 query 会让 vite 走另一条路径
   （实测返回带注释的近乎原文，26158 字符 vs 正常 89183），两边管线根本不一致。
   同理也别用"注释里有没有某个新写的字符串"判断新鲜度 —— 注释不会出现在服务内容里
   （本节写作时就因此误判过一次"服务端 stale"，白跑两轮）。

> **为什么值得为 P0b 花这些功夫**：闸门的前提是"跑的是刚写的那份代码"。
> 这个前提一旦不成立，**全绿等于没跑**。本项目已经有过一次同类事故
> （`video_call_build_plan.md` §8.5：8001 用了错解释器，服务正常、只坏一个端点、无人察觉）。
> ⚠️ P0b 是启发式而非证明（只有改动碰到那 8 个标记之一才会报警）——
> **最稳的做法仍是每次改完重启一次 dev server 再跑闸门。**

**验收命令**

```powershell
node D:\Develop\AIC\_probe_call_page.js   # 退出码 0 = 全过；27/27
```

> 探针会**新建 1 个空会话**（走的就是"无 id 进通话"那条真实路径），跑多次会留下多个空会话，
> 属预期副作用；需要干净的历史列表就删掉它们。
> 跑之前建议重启一次 dev server —— 见 P0b 与上面第 2 条方法学教训。

**护栏自检**（沿用 S1 那条纪律：**测试要能变红才算测试**）

把 `VoiceCall.tsx` 里 `if (startingRef.current) return;` 注释掉再跑闸门：

| 检查 | 结果 |
|---|---|
| P13 | ❌ **`ws=3`** —— 连点 3 次真的建出 3 个 TTS 通道（3 个 WebSocket + 3 个 AudioContext，只关得掉最后一个） |
| P0b | ✅ 顺便验证了去注释比对法：磁盘去注释后的 8 个标记次数与服务内容逐一相等 |

也就是说 P13 不是"永远为真"的装饰性检查，它抓的就是真会发生的泄漏。

**回归**：`client tsc -b` ✅；`client` 生产构建 ✅（exit 0）；
S2 闸门 `_probe_tts_player.js` 重跑 ✅（17/17，因为动了 `ttsStream.ts`）。


---

### S4 —— 轮次编排 `useVoiceCall`

> ⚠️ **开工第一件事**：删掉 S3 留在 `VoiceCall.tsx` 里的 `?demo=1` 骨架自检面板
> （`DEMO_LINES` / `handleDemoSay` / 那张 Card）。它念的是**写死的台词**，
> 留着会在答辩现场把预置文本当真实回复播出去 —— 这正是本项目最该避免的"演示假象"。
> 面板里那两行耗时读数（首段合成 / 首段出声）挪到 S4 的真实轮次统计里。

新建 `client/src/hooks/useVoiceCall.ts`。**这是通话的心脏。**

#### S4.1 状态机

| 状态 | 含义 | 进入条件 | 离开条件 |
|---|---|---|---|
| `idle` | 未开始 | 初始 | 用户点"开始通话" |
| `listening` | 麦克风开着，等用户说话 | 上一轮 `done` 且播放队列清空 | 收到 ASR final |
| `thinking` | 已发出请求，等首个 `delta` | 发了 `sendChatMessageStream` | 收到第一个 `delta` 或 `meta.streaming===false` |
| `speaking` | 正在出声 | 播放队列非空 | 队列排空 → `listening` |
| `crisis` | 高风险，语音硬禁用 | `meta.streaming === false` 或 `done.isCrisis` | 用户手动确认（回到 `/chat`） |
| `ended` | 已挂断 | 点挂断 | —— |

#### S4.2 接线表（**全部复用既有代码，不新开任何设备**）

> ⚠️ **本表有一处已在施工中修正**：原表第一行写的是 `useAnxiety().startVoiceInput` /
> `voiceMetrics.asrPartialText`，但 `AnxietyProvider` **只挂在 `PatientLayout` 里**
> （`PatientLayout.tsx:116`），而通话页是**刻意的整屏独立页、不在它里面** ——
> 在通话页调 `useAnxiety()` 会直接抛错（`AnxietyContext.tsx:1249`）。
> 改用自包含的 `useVoiceAnalysis(ageGroup, onAudioEnergy, onVoiceFinal)`：
> 它就是"聊天页语音录入"那条链路本身，麦克风、16kHz AudioContext、
> AudioWorklet、流式 ASR 会话、端点定稿全在里面，只是不经过 context。
> **这条修正只影响"从哪里拿"，不影响下面任何一行语义。**

| 需要什么 | 从哪来 | 位置 |
|---|---|---|
| 麦克风开/关 | ~~`useAnxiety().startVoiceInput`~~ → `useVoiceAnalysis().start / stop` | `useVoiceAnalysis.ts:297 / :466` |
| 中间结果（上屏 + 打断判据） | `voiceMetrics.asrPartialText` | `multimodal.types.ts:136` |
| 定稿（送下一轮） | **`onVoiceFinal(text, seq)` 回调**（不是订阅 state） | `useVoiceAnalysis.ts:54` |
| 文本流式对话 | `sendChatMessageStream()` | `client/src/services/chatStream.ts:124` |
| 出声 | `TtsPlayer`（S2）+ `/algorithm/api/v1/tts/ws` | 新建 |

> ⚠️ **定稿必须走回调，不能只订阅 `asrFinalText` state**。Chat.tsx 已经踩过这个坑
> （`Chat.tsx:67-115`）：`isRecording` 与定稿不是同一时刻落的，只订阅 state
> 会漏掉最后一段。通话里每一段定稿就是"一轮"，漏一段就是漏一轮。
> 顺带一提：`onVoiceFinal` 的**回调 + `stop()` 返回值**会拿到同一段文本，
> 所以挂断时必须**先**把状态置成 `ended`（回调因此丢弃）、**再**取返回值，
> 否则同一句会被送两次、落库两条用户消息。

#### S4.3 一轮的精确时序

```ts
// 伪码：落进 useVoiceCall 的轮次函数
async function runTurn(userText: string) {
  setState('thinking');
  const handle = sendChatMessageStream(conversationId, userText, {
    onMeta: (m) => {
      // F-3：零新增阈值复用既有安全判据
      voiceAllowed = m.streaming;
      if (!m.streaming) setState('crisis');
    },
    onDelta: (text) => {
      appendAiCaption(text);                     // 字幕照常（即使危机也上屏？见下）
      if (!voiceAllowed || interrupted) return;  // 打断后只上屏不出声
      for (const unit of splitSpeakable(text)) ttsSay(unit);   // 见 §4.4
    },
    onRevise: (text) => {
      replaceAiCaption(text);
      onAuditRevise(text);                       // 见 §4.5
    },
    onDone: (payload) => {
      if (payload.isCrisis === true || payload.riskLevel === 'crisis') setState('crisis');
      flushPendingInterrupt();                   // F-4：插话在这里才发出去
      pendingTurn = null;
    },
    onError: (e) => { /* 显示提示，回到 listening，不要停掉麦克风 */ },
  });
  turnAbortRef.current = handle.abort;           // 仅"挂断"时用，打断时不用
}
```

#### S4.4 可合成单元的切分（F-2 + F-6 的客户端补丁）

**两件事叠在一起：**

* **F-2**：安全闸门已按句切过，所以正常情况不用切；
* **F-6(a)**：VITS 不流式 ⇒ **首声 = 第一个 unit 的整句合成耗时**
  （实测 9 字 263 ms / 16 字 443 ms / 33 字 903 ms，8 线程）。
  所以**第一个 unit 越短，首声越快** —— 这是唯一能压首声的地方，
  且零风险、不碰安全组件。

```ts
/**
 * 安全闸门已经按句切过了；这里只处理两种情况：
 *   1. unit 太长（> 30 字）→ 二次切分，避免一次合成几十秒音频
 *   2. unit 是**本轮第一个** → 按逗号再切一刀，只先合成第一个小句，
 *      让首声提前到来（实测 VITS 整句合成耗时与字数近似成正比）
 */
function splitSpeakable(unit: string, isFirst = false): string[] {
  const MAX = 30;
  const FIRST_MAX = 12;                 // 首块目标：≈ 300–400 ms 合成
  const limit = isFirst ? FIRST_MAX : MAX;
  if (unit.length <= limit) return [unit];
  // 先按次级标点切（逗号/顿号/冒号）
  const parts = unit.split(/(?<=[，,、：:])/).filter(s => s.trim());
  // 再把仍然超长的硬切
  const out = parts.flatMap(p =>
    p.length <= limit ? [p] : p.match(new RegExp(`.{1,${limit}}`, 'gs')) ?? [p]
  );
  // 首块只要第一段就够（剩下的会随后续 unit 一起合成，不影响正确性）
  return isFirst ? out.slice(0, 1).concat(out.slice(1).join('')) : out;
}
```

**为什么这不丢内容**：切开的只是"送合成的批次"，合成出来的 PCM 仍按顺序
全部入播放队列。闸门已经保证这个 unit **本身**不含红线词，所以切开也不改变安全语义。

#### S4.4b 合成/播放流水线（非流式 TTS 变流式体验的关键）

RTF 实测 0.11–0.14 ⇒ **合成比播放快 7 倍**，所以只要"边播边合成"，
队列在结构上不可能饿死：

```
delta#1 到达 → 立刻 ttsSay(首块, isFirst=true)   ← 唯一要等的一段（~0.3 s）
   开始播第 1 块 ──同时──→ 合成第 2 块
   播第 2 块 ──────同时──→ 合成第 3 块
   ...
```

**硬性要求**：`ttsSay` 必须是**非阻塞入队**，绝不能 `await` 合成结果。
前端把待合成文本推给 `/tts/ws`，服务端串行合成、边合成边推 PCM，
播放器按 `nextStartTime` 排程 —— 三者互不阻塞。
（服务端的串行合成在这里反而是好事：它天然给音频排序。）

#### S4.5 `revise`（审计改稿）时的音频策略 ⚠️

**已播出去的声音收不回来**（`voice_call_plan.md` §4.4）。必须定义一个确定的行为，
而不是"看情况"：

```
onRevise(revised):
  1. 立刻 stopAll() 并清空待合成队列
  2. 计算「已播文本」与「改稿」的最长公共前缀
  3. 若 revised 以已播文本开头（说明只是被截短/补齐）
       → 不再出声（避免听感上"重说一遍"），只把字幕替换成改稿
     否则（说明被整体改写）
       → 合成并播出改稿中"尚未播过"的剩余部分
  4. 记录一次审计改稿事件（用于演示时解释）
```

> 这条策略是**产品判断**，不是安全判断。它不改变任何安全规则。
> 但它决定了"审计改稿时用户听到什么"，**建议让 reviewer 过一眼**。⚠️

#### S4.6 闸门

| 检查 | 判据 |
|---|---|
| 首声 | 用户停口 → 第一个 PCM 到达播放器 < 2.4 s [估计]。**用 `TtsChannel.takePcmReadyMs(seq)`，只取本轮第一段** —— 第 2 段起读数是"排队位置"，不是延迟（见 §1 F-7：实测差 20 倍） |
| 轮次 | 队列排空后状态自动回到 `listening`，不用点任何按钮 |
| 落库 | 一轮结束后 `/api/consultation/conversations/:id/messages` 里有 user + assistant 两条 |
| 字幕一致性 | 字幕最终文本 === `done.text`（含危机热线补全） |
| 设备 | 全程只有 **1** 条 live 麦克风轨道、**1** 个 ASR AudioContext、**1** 个 TTS AudioContext |

---

#### S4.7 执行结果（✅ 本轮完成）

**改动文件**

| 文件 | 内容 |
|---|---|
| `client/src/hooks/useVoiceCall.ts`（**新建**） | 轮次编排：状态机、定稿→一轮、SSE 接线、TTS 入队与切分（`splitSpeakable`）、半双工、挂断冲刷、一轮实测耗时 |
| `client/src/pages/patient/VoiceCall.tsx` | 重写接线：页面只管看和点（布局/字幕/按钮/计时/跳转），编排全部交给 hook；**删掉 `?demo=1` 的假台词面板**，换成 `?debug=1` 的**实测读数**面板 |
| `client/src/hooks/useVoiceAnalysis.ts` | 修 AudioContext 关闭的未处理 rejection（见下） |
| `client/src/services/ttsPlayer.ts` | 同一个 bug 的第二个实例（S2 写的，这次被闸门抓到） |
| `client/src/services/echoFilter.ts`（**新建**） | 文本级自回声过滤的**保守预版**（S5 方案 D 的判据先落地，见下） |

**怎么验的：把"人的嘴"换成合成语音**

难点：`--use-fake-device-for-media-stream` 给的是**正弦音**，ASR 只会得到空文本 ——
也就是说"用户说话 → 定稿"这一段**用假设备根本验不了**。解决办法是 Chromium 的另一个开关
`--use-file-for-fake-audio-capture=<wav>`：**用本机 TTS 合成一句中文当成麦克风输入**
（`_make_asr_fixture.py`，2s 静音 + 语音 + 3.5s 静音，16kHz）。

于是整条链路都是真的：AudioWorklet → 本机流式 ASR → 端点定稿 → Node SSE → AI 回复 → TTS 出声，
**只有"人的嘴"是合成的**。

> ⚠️ 夹具**刻意循环播放**（不加 `%noloop`）：定稿靠的是服务端流式 ASR 的**端点判定**
> （尾随静音），而假采集在文件播完后可能不再投递音频 —— 那样服务端永远等不到那段静音，
> 于是表现为"中间结果上屏了、却一直没有定稿"。**这是夹具的坑，不是产品的问题**
> （真实麦克风永远在投递静音）。代价是转写尾部会带上循环回来的一小截
> （实测："我最近总是睡不着，心里特别烦。**我最近。**"），断言时按"非空 + 落库一致"判，不比对全文。

**实测（`_probe_call_voice_e2e.js`，14/14 通过；连续 3 次全绿）**

| # | 检查项 | 结果 |
|---|---|---|
| V0 / V0b / V0c | dev server 服务的是磁盘上那份 hook；接通 → 正在聆听 | ✅ |
| V1 | **ASR 会话确实建立**（`/asr/ws` ≥ 1） | ✅ `asrWs=1`（**S3 时这里恒为 0 —— 正是"录不进语音"的根因**） |
| V2 | 用户字幕上屏（中间结果 → 定稿） | ✅ 定稿 12.2 s 后出现 |
| V3 | AI 字幕上屏 | ✅ |
| V4 | **TTS 真的收到 PCM**（`/tts/ws` 二进制帧） | ✅ 上行 3 个 `say`（9/6/13 字），下行收到 PCM |
| V5 | 一轮结束后自动回到「正在聆听」 | ✅ 不需要点任何按钮 |
| V6 / V7 | **用户消息与 AI 回复都落库**，且与字幕一致 | ✅ |
| V8 | **首段合成 246 ms** < 2400 ms | ✅ 首段出声 247 ms |
| V9 | 麦克风只申请 1 次（通话页不自持设备引用） | ✅ |
| V11 | **自回声过滤的判据**（整句/半句被听回要拦，用户自己的话要放行） | ✅ 6 个用例逐条断言，含"带 ASR 小错"（重叠 92%）与"太短放行" |
| V10 | 挂断 → `/chat/<id>`，0 live 轨道、**0 未处理 rejection** | ✅ |

**为什么 S4 里就加了自回声过滤（它本来是 S5 的活）**

半双工挡不住最危险的那一秒：AI 说完 → 播放队列排空（相位回到 `listening`）
→ **约 0.8 s 后** ASR 才把"刚才听到的 AI 的话"端点定稿 —— 此时相位已经是 `listening`，
那句会**当成用户说的话落库**，并触发一轮莫名其妙的回复。表现就是"AI 自己跟自己聊起来"。

所以 `services/echoFilter.ts`（S5 方案 D 的**保守预版**）先落地：只在高重合时判回声
（互相包含，或字符二元组重叠 ≥ 60%），判不准时**宁可放行用户的话**。
⚠️ **阈值 60% 是拍的** —— 施工单 §S5.1 那个"AI 的话被 ASR 听成什么"的实验做完之前，
它只能算"挡住最尴尬的失效"，不能算标定过。它不改变任何安全规则。

**本轮新增的两个权威读数**（`?debug=1`，都是实测值）：

| 读数 | 实测 | 说明 |
|---|---|---|
| 定稿 → 首个 `delta` | **782 ms** | LLM 首句这一腿（S0 的估计是 370–390 ms，这里含 META + 网络） |
| 首段「请求 → 音频到手」 | **246 ms** | TTS 这一腿 |

⇒ **定稿 → 首声 ≈ 1.03 s**。比施工单里"中位 1.6 s"的估计更好；差的那部分主要是
**ASR 端点判定**（服务端等尾随静音）与真实说话长度，这两项本轮没有单独计时。

**顺带修掉的两个真 bug（都是"未处理 rejection"，闸门抓到的）**

`AudioContext.close()` 返回的是 **Promise**，对**已经关过**的 context 会**拒绝**
（`InvalidStateError: Cannot close a closed AudioContext`）。两处都写成 `void ctx.close()`
或裸调用，`try/catch` 抓不到 Promise 的拒绝：

| 位置 | 原来的写法 | 现在 |
|---|---|---|
| `useVoiceAnalysis.ts`（stop / 卸载清理） | `audioContextRef.current?.close()` | 抽成 `closeAudioContext()`：先判 `state !== 'closed'`，再 `.catch(() => undefined)`，并立刻置空 ref |
| `ttsPlayer.ts` `dispose()` | `void ctx.close()` | 同上 |

修之前通话页会稳定报 **2 条** `unhandledrejection: Cannot close a closed AudioContext.`
（S3 闸门没抓到，是因为那时 ASR 根本没启动、没有 ASR 的 AudioContext）。
修之后 V10 = 0 条。

> **回归**：`useVoiceAnalysis` 是聊天页语音录入共用的 hook，所以改完必跑
> `_probe_chat_voice_browser.js` —— ✅「语音输入出字了」，转写结果与改前一致。

**第三条是"指标快照时机"的坑（F-7 的同类）**

`turn` 的读数一开始只在 `done` 时快照一次，于是"首段合成/首段出声"永远是空的 ——
因为**首段的播放读数是在 `done` 之后才产生的**（合成在 SSE 还没结束时就已入队并开播）。
现象是：`say` 发了 3 次、PCM 到了 2 帧，面板上却是 `—`。现在 `onStart` 里也会再发布一次。

> 记一笔教训：**只要是"随事件陆续产生"的指标，就不能在单一时刻快照。**
> 与 §1 F-7 是同一类错误（把"排到它出声"当成"首声"），只是这次错在时机而不是定义。

**明确的边界（S4 不做、留给后面）**

* **半双工**：AI 思考/说话时收到的定稿一律**丢弃并计数**（`?debug=1` 有"被丢弃定稿"）。
  理由是麦克风常开、AI 的声音会经扬声器回到麦克风被 ASR 识别，若当插话发出去，
  **会把 AI 自己的话写成一条用户消息落库**。真打断 + 回声过滤是 S5（要先做 §S5.1 的实验标定）。
* **revise 只改字幕不重说**（§S4.5 的保守半边，完整策略待人工拍板 —— 见 §7 Q4）。
* **危机只做到"立刻静音 + 提示走文字"**，卡片与"绝不用语音念热线号"是 S6。
* **不做并发轮次**：`turnActive` 期间不会开第二轮（半双工已经保证了这一点）。

**验收命令**

```powershell
# 需要 5173 / 3000 / 8001 都在跑；夹具按需重新生成
& 'D:\Develop\AIC\.venv\Scripts\python.exe' D:\Develop\AIC\_make_asr_fixture.py
node D:\Develop\AIC\_probe_call_voice_e2e.js   # 退出码 0 = 全过；14/14
node D:\Develop\AIC\_probe_call_page.js        # S3 的不变量仍须全过；27/27
```

---

### S5 —— 打断 + 回声过滤

#### S5.1 先做实验：**"AI 的话被 ASR 听成什么"**（`voice_call_plan.md` §7 未验证项 13）

> ⚠️ **S4 已经把"方案 D"的保守判据先落地了**（`services/echoFilter.ts`：互相包含，
> 或字符二元组重叠 ≥ 60%；判不准时放行用户的话 —— 见 §S4.7）。
> 那个版本只保证"挡住最尴尬的失效（把 AI 自己的话写成用户消息）"，
> **阈值是拍的**。这一节要做的实验仍然是**标定**它：拿真实的 `(spoken, heard)` 配对
> 去量误判率，把 60% 换成量出来的数；并决定要不要放开 S4 的半双工（见 §7 Q10）。

> 这是 §6 里最需要先做实验的一个数，阈值现在是**拍的**。

步骤：

1. 用 S1 的端点合成 10 句典型回复，导出 WAV；
2. **用本机扬声器播放**（不是耳机 —— 要的就是扬声器→麦克风的耦合路径），
   同时开 ASR 录制；
3. 保存 `(spoken, heard)` 配对，看：
   * 完全正确的比例；
   * 命中"≥6 字连续子串"的比例；
   * 误判为"用户在说话"的比例。
4. 据实标定 `echoFilter` 的阈值，并把 10 组配对数据写进 `docs/`（可复现才算数）。

#### S5.2 回声过滤

新建 `client/src/services/echoFilter.ts`：

```ts
/**
 * 文本级自回声过滤（voice_call_plan.md §6.2 的「方案 D」）
 *
 * 为什么不用 DSP：我们**完全知道自己在说什么**。把 ASR 的中间结果与
 * "最近正在播的 TTS 文本"做子串/编辑距离比对，命中即判定为回声。
 * 便宜、确定、不依赖任何回声消除实现。
 */
export class EchoFilter {
  /** 播放器每播一句就登记一次 */
  note(spokenText: string, at: number): void;
  /** ASR 中间结果是否为回声 */
  isEcho(asrText: string, now: number): boolean;
}
```

#### S5.3 半双工窗口 + AEC

改 `client/src/services/mediaStreamManager.ts:82-85`：把 AEC / NS / AGC 加进**默认**约束
（理由见 F-5）。同时在通话页读取并展示 `track.getSettings()` 的**实际生效值**。

改 `client/src/services/streamingAsr.ts`：新增公开方法

```ts
/** 暂停/恢复向服务端送帧（半双工窗口用；不停轨道、不动 SpeechGate） */
setForwarding(on: boolean): void;
```

改 `client/src/hooks/useVoiceAnalysis.ts:517` 的返回，把 `setForwarding` 透出去；
再经 `AnxietyContext.tsx` 暴露成 `setVoiceForwarding(on: boolean)`。

> ⚠️ **不要用 `track.enabled = false` 来"静音麦克风"**：那会让服务端收到一段纯静音，
> 端点检测照样触发，反而定稿出一个空段。必须停的是"送帧"这个动作。

#### S5.4 打断时序（**照抄 `voice_call_plan.md` §6.2 的时序表**）

```
开始播 TTS 第 1 句（Web Audio）
  [-∞, 0)      麦克风正常流式识别（用户还在说 → 触发打断）
  [0, 400ms)   半双工窗口 —— setForwarding(false)，覆盖 AEC 收敛期
  [400ms, ∞)   setForwarding(true)，靠 AEC + EchoFilter 过滤
收到 ASR partial：
  EchoFilter.isEcho(partial) 为真 → 忽略（是 AI 自己的声音）
  为假                          → 判定为真实打断 → 走 F-4 的四步
收到 ASR final：作为「下一轮待发」文本挂进队列（F-4 的 ③）
```

**G-1：打断必须靠 partial 而不是 final**（partial 每 ~330 ms 就有 [实测]，
final 要等端点 0.45–1.2 s，手感迟钝）。
**G-2：先清本地队列，再谈别的** —— 千万不要等服务端回执。

#### S5.5 闸门

| 检查 | 判据 |
|---|---|
| 打断体感 | 插话 → 本地静音 < 200 ms（打时间戳） |
| 误打断 | 用 S5.1 的 10 句配对数据回放：**0 次误打断**（AI 自己的声音不得触发打断） |
| 真打断 | 10 次真人插话：≥ 9 次成功打断（余 1 次落在 400 ms 半双工窗口内，**这是设计接受范围内的**，要如实告知而不是当成 bug） |
| 一致性 | 打断后本轮仍落库一条完整 assistant 消息；插话作为下一轮正常落库；**无并发轮次** |

#### S5.6 执行结果（✅ 本轮完成）

**改动文件**

| 文件 | 内容 |
|---|---|
| `client/src/services/echoFilter.ts` | 从"保守预版"改成**标定过**的判据：`EchoFilter` 类（带时间窗）、可导出的原始分数 `scoreAgainstSpoken`、常量 `ECHO_OVERLAP_THRESHOLD = 0.40` / `ECHO_TAIL_MS = 2000` |
| `client/src/services/streamingAsr.ts` | 新增 `setForwarding(on)` / `isForwarding`（半双工窗口；**停的是送帧，不是麦克风**） |
| `client/src/services/mediaStreamManager.ts` | 默认麦克风约束加上 AEC / NS / AGC（用 `ideal` 而不是硬约束，理由见下） |
| `client/src/hooks/useVoiceAnalysis.ts` | 透出 `setForwarding` / `isForwarding` / `getMicSettings`，新增 `onVoicePartial` 回调（打断靠中间结果） |
| `client/src/hooks/useVoiceCall.ts` | 打断路径（partial 判据 → 停播 → 排队 → 顺序发下一轮）、半双工窗口、`interruptStopMs`、并发轮次峰值 |
| `docs/echo_calibration.md`（**新建，由脚本生成**） | 40 组 `(spoken, heard)` 配对 + 阈值扫描 + 4 条发现 |

**S5.1 那个实验：结论是"阈值不能拍"**

完整数据、逐条配对与复跑命令见 [`docs/echo_calibration.md`](./echo_calibration.md)（**该文件由 `_exp_echo_decide.js` 生成，不要手改**）。做法：

```
10 句 AI 回复（8 句来自数据库里真实的 assistant 消息）
   → 真实 TTS 通道（/tts/ws）拿到 PCM + spoken_text
   → 三种声学耦合：clean（数字回环）/ room（带通+轻混响+SNR18dB）/ far（更小更吵，SNR8dB）
   → 真实流式 ASR（/asr/ws）
   → 与 spoken_text 比对（跑的是 esbuild 打包出来的**同一份源码**）
```

| 量 | 实测 | 说明 |
|---|---|---|
| 真回声（30 组）**最低**重叠 | **41.7%** | ai10/clean——号码被念糊那一句 |
| 普通负样本（8 句）**最高**重叠 | **37.5%** | u02「嗯，最近考试压力特别大」 |
| 困难负样本 | **83.3%** | u09 把 AI 刚说的话原样复述了一遍 |
| ⇒ 阈值 | **0.40** | 真回声 30/30 全中；普通负样本 0 误判 |

三条必须记住的结论：

1. **旧的 0.6 会漏掉 3/30 条真回声**（命中率 90%），漏掉的正是"AI 自己的话被当成用户消息落库"那条路径。
   改成 0.40 之后 30/30 全中。
2. **困难负样本是原理边界**：用户把 AI 刚说的词复述一遍（u09）在文本层面与回声**无法区分**，
   任何阈值都救不了（要挡住它就得漏掉大量真回声）。缓解手段是时间维度 ——
   `ECHO_TAIL_MS = 2000`：只在"AI 说完后 2 秒内"到达的文本参与比对。
   S4 用播放器默认的 8 秒回溯窗口会把这个误杀放大 4 倍。
3. **号码真的会被念糊**（这条直接支撑 §S6.3）：`400 161 9995` 在三种耦合下分别被念成
   「四百六十一九九九九五」「四百六十一九九九九五五」「四百一百六十一九九九五」——
   **一次都没有**按电话号码分组读出来，TTS 把它当成了一个 11 位数。
   这也是表 A 里 41.7% 那个下限的来源。

**AEC 加进默认约束：为什么用 `ideal` 而不是 `true`**

`{ echoCancellation: true }` 是**硬**约束：某些虚拟声卡 / 采集卡 / 蓝牙设备的驱动不声明这项能力，
硬约束会直接 `OverconstrainedError` —— 结果是"为了降回声，把麦克风整个弄不能用了"。
所以默认约束用 `ideal`，并且**绝不假设它生效**：通话页读 `track.getSettings()` 的实际值公示出来。

实测（`?debug=1` 面板「麦克风实际」一行）：**`AEC=开 NS=开 AGC=开 48000Hz`**。
但它只是"浏览器说它开着"，所以文本级过滤仍然是必需的（不是双保险，是两道不同的防线）。

**打断时序（S5.4）落在代码里的样子**

```
TTS onStart ──► echoFilter.note(spoken_text, now)      ← 登记"这句话进房间了"
           └─► setForwarding(false)，400ms 后恢复      ← 覆盖 AEC 收敛期

ASR partial（每 ~330ms）──► 本轮还在飞？
      ├─ echoFilter.judge() 判为回声 ──► 丢弃（不上屏、不断句）
      └─ 判为真打断 ──► ① 立刻 stopAll()（同步，实测 0 ms）
                        ② retireAll()（被打断的半句退出回声窗口）
                        ③ 本轮剩下的句子不再出声（只在字幕里追加）
                        ④ 相位回 listening，用户立刻看到自己的话在上屏

ASR final ──► 本轮还在飞？ ──► 进 pending 队列，等 done 之后**顺序**发下一轮
                             （绝不并发，也绝不丢弃用户真说的话）
```

两个刻意的取舍：

* **打断后本轮剩余句子不再出声**：已经"立刻静音"了，几百毫秒后又自己说起来会让人以为打断失败。
  字幕照旧追加（内容不藏），回答由队列里的插话开新一轮。
* **`stopAll()` 不触发播放器的 `onEnd`**（`onended` 被置空以避免重入），所以被打断的半句只能靠
  `retireAll()` 退出回声窗口 —— 它的 `note` 已经在 `onStart` 记过了。

**S5.5 闸门（`_probe_call_s5.js`，会话 A 部分）**

| 检查 | 判据 | 实测 |
|---|---|---|
| 打断体感 | 判定为真打断 → 本地静音 < 200 ms | ✅ **0 ms**（`stopAll()` 同步掐断） |
| 半双工窗口 | 出声时停送帧、0.4s 后自动恢复 | ✅ 捕获到停 → 恢复 |
| 真打断 | 循环夹具会持续产生中间结果，打断被真实触发 | ✅ 打断次数 ≥ 1，插话落在本轮第 773 ms |
| 一致性 | 打断后本轮仍落库完整 assistant；插话作为下一轮落库 | ✅ user 8 / assistant 8，逐条对应 |
| **无并发轮次** | 同时进行的轮次数峰值 | ✅ **1**（`?debug=1` 的「并发轮次峰值」） |
| **误打断** | AI 自己的声音不得触发打断 | ✅ 见下 |

> ⚠️ **"10 次真人插话 ≥ 9 次成功"这一条本轮没有做**，因为它需要**真人**在真实扬声器/麦克风环境里插话 10 次。
> 无头环境的"麦克风"只有一个音频文件，AI 的声音根本到不了它 —— 也就是说
> "AI 自己的声音会不会触发误打断"在这里**原理上测不了**。
> 本轮给的替代证据是：
> ① 误打断方向用 S5.1 的 30 组真实回声配对判（30/30 命中、普通负样本 0 误判）；
> ② 真打断方向用循环夹具真实触发（打断次数、0 ms 停播、落库一致、无并发轮次）。
> **真人 10 次那条留给 §S9.3 的走查清单**，届时要如实记录"几次成功、几次落在 400ms 窗口内"。

> 📌 本轮还观察到一条**已知代价**：400 ms 半双工窗口内用户开口，开头会被切掉。
> 这不是 bug，是 §S5.5 明确接受的范围。

**顺带修的一个真问题（网关日志抓到的）**

AI 的回复里带书名号 `「」`，而 `splitSpeakable` 会把它们切成**独立的 1 字片段**送合成；
`sherpa-onnx` 的词表里没有这两个符号，日志里出现：

```
Ignore OOV '「' / Ignore OOV '」'  →  Failed to convert 」 to token IDs
[TTS] 合成结果为空白音频：seq=3（文本 1 字），已跳过
```

后果是每一轮都会多发 2 次**注定合成失败**的请求（还各带一次 `error` 回调）。
已加一道"可发音"判据：**不含任何汉字/字母/数字的片段根本不送合成**。

---

### S6 —— 危机切换（安全兜底）

#### S6.1 复用既有卡片

`Chat.tsx:411` 有一个危机 Modal，`:652-656` 是热线卡片内容（`400-161-9995` / 生命热线）。

**抽成 `client/src/components/CrisisResourceCard.tsx`，Chat 与 VoiceCall 共用。**
⚠️ **只搬不改文案** —— 热线号码、机构名称一个字都不动。

#### S6.2 危机时的动作（`voice_call_plan.md` §4.5）

```
risk_level ∈ {high, crisis}（由 meta.streaming === false 判定，见 F-3）：
  1. 立刻 stopAll() + 清队列 + 中止在途合成（WS 发 cancel）
  2. 会话切纯文字；麦克风保持可用（用户还要说话）
  3. 走既有 escalation（AGENTS.md：必须触发）
  4. 【预留】播一句固定的、预审核过的短句 —— 用预合成 PCM 缓存
     ⚠️ 本计划**没有拟这条文案**，留空由人工填写并经审核后再启用；
        未填写前该步骤直接跳过（不播任何东西）
  5. 展示 CrisisResourceCard
```

#### S6.3 **绝不用 TTS 朗读热线号码**（最强烈的一条建议）

读数字容易出错（"400-161-9995" 可能被读成"四十万…"），危机场景下念错号码后果严重。
号码只以**文字卡片**呈现。

**这条要落成代码，不能只靠自觉**：

```ts
/** 送 TTS 之前的最后一道保险：含电话号码特征的片段一律不合成 */
const PHONE_LIKE = /\d[\d\s-]{5,}\d/;
function ttsSay(unit: string): void {
  if (PHONE_LIKE.test(unit)) {
    console.warn('[通话] 片段含号码特征，跳过合成（只上屏）:', unit.slice(0, 30));
    return;
  }
  // ...正常入队
}
```

#### S6.4 闸门

| 检查 | 判据 |
|---|---|
| 危机文本（如"我不想活了，活着没意思"） | 立即静音；**0 个 PCM 帧**送达播放器；显示卡片 |
| 热线去重 | 卡片只出现一次（复用服务端既有幂等） |
| 号码 | 全链路日志里 TTS `say` 的文本**从不包含**热线号码 |
| 回归 | 与 `voice_call_plan.md` §4.2 已实测的行为一致：无 delta、`requires_escalation=True` |

#### S6.5 执行结果（✅ 本轮完成）

**改动文件**

| 文件 | 内容 |
|---|---|
| `client/src/components/CrisisResourceCard.tsx`（**新建**） | 危机资源卡片；文案、两个热线号码、机构名称**一个字都没动**（从 `Chat.tsx` 原样搬出） |
| `client/src/pages/patient/Chat.tsx` | `CrisisModal` 改为复用该组件（文字页与通话页从此**只有一份**危机文案） |
| `client/src/pages/patient/VoiceCall.tsx` | 危机全屏浮层（复用同一张卡片）+ 顶部提示条 |
| `client/src/hooks/useVoiceCall.ts` | 危机分支：立刻 `stopAll()` + 清队列 + 会话切纯文字；`PHONE_LIKE` 号码守卫；`lastRisk` 如实上报服务端判定的等级 |

**危机这一轮在协议层长什么样（实测抓包，不是推断）**

用 `_probe_crisis_sse.js` 直连/经代理各抓一次，事件序列**完全一致**：

```
user → meta{risk_level:"crisis", streaming:false} → revise{含热线的完整文案} → done
                                                      ↑ 无 delta
```

| 时刻（经 vite 代理） | 事件 | 说明 |
|---|---|---|
| +8 ms | `user` | 用户消息已落库 |
| +85 ms | `meta` | **`streaming: false`** —— 这就是 F-3 那个"零新增阈值"的开关 |
| +642 ms | `revise` | 完整文案（含三条热线），要求前端整体替换 |
| +661 ms | `done` | `isCrisis: true`、`riskLevel: "crisis"` |

> ⚠️ 这里纠正两处**我自己先写错的期望**（都记在探针注释里，免得下次再错）：
> ① 危机轮**会**发 `meta`（只是不带 `streaming: true`），不是"没有 meta"；
> ② 危机轮**会**发 `revise`（因为补了热线，上屏文本与最终文本必然不同）。
> 真正成立的是"**没有 delta**"这一条。

**S6.4 闸门（`_probe_call_s5.js`，会话 B 部分；危机文本走真实语音链路）**

| 检查 | 判据 | 实测 |
|---|---|---|
| 危机文本（"我不想活了，活着没意思"，**由夹具喂给真 ASR**） | 立即静音 | ✅ 相位 7.1 s 内切到「已切换为文字陪伴」 |
| 0 个 PCM 帧 | 整通电话 **`ttsBinary === 0`** | ✅ 一帧都没有 |
| 绝不合成 | 整通电话 **一次 `say` 都没发** | ✅ 号码与非号码文本都没进合成 |
| 热线卡片 | 出现，且两个号码都在**文字**里 | ✅ `400-161-9995` / `400-821-1215` 只在卡片上 |
| 麦克风保持可用 | live 音轨仍为 1（用户还要说话） | ✅ |
| 与 §4.2 一致 | 无 delta、`requires_escalation=True` | ✅ 客户端事件序列 `user→meta→revise→done`；算法层 `requires_escalation=true, dialog_state=CRISIS` |
| 升级真的触发 | 算法日志出现 `🚨 危机告警`（risk 0.90）+ 落库回复里含热线文案 | ✅ 两道独立证据 |
| 挂断 | 0 live 轨道、0 未处理 rejection | ✅ |

**`PHONE_LIKE` 守卫的判据（A12 逐条断言）**

| 应当命中 | 实测 | 应当放行 | 实测 |
|---|---|---|---|
| `400-161-9995` | ✅ | `我最近总是睡不着` | ✅ 不命中 |
| `可以打 400 161 9995 这个电话` | ✅ | `这种状态持续多久了` | ✅ |
| `4008211215` | ✅ | `2026 年 9 月`（4 位数字） | ✅ |
| `13800138000` | ✅ | `先深呼吸三次` | ✅ |
| `电话是（400）1619995`（全角括号） | ✅ | | |

> 比施工单里那条正则多收了两类：**全角括号/破折号**（中文排版常见）与**7 位以上连续数字**（不带分隔符的写法）。

**一致性与"只出现一次"**

卡片抽成组件后，文字页与通话页共用同一份文案 —— 这正是 S6.1 想要的：**号码不一致是不可接受的**。
通话页里卡片在**一次通话内只弹一次**（用户点"我知道了"之后不再弹出），危机状态本身持续到挂断。

> ⚠️ **需要人工过眼的一处**：施工单 §S6.2 把 `risk_level ∈ {high, crisis}` 都算进这一分支
> （依据是 F-3 的 `streaming === false`），也就是 **`high` 也会弹热线卡片**。
> 这是施工单写好的行为，本轮照做；但它属于"安全相关规则的变更"，按 AGENTS.md
> 应当由人工确认一次（`high` 是不是也要弹卡片、还是只在 `crisis` 时弹）。

---

### S7 —— 视觉进通话（**可砍**）

**不重复，直接执行 [`video_call_build_plan.md`](./video_call_build_plan.md) 阶段 4**（F7 / F8 / F9 / F12），
视觉在通话里只做三件诚实的事（§3.4）：存在感 / 画面质量提示 / 给自己看的小读数。

**本计划只补两条与通话形态相关的约束：**

1. **通话期视觉不进融合、不进风险、不上报**（F7）—— 硬验收标准是
   **开/关摄像头两次，落库的 `riskLevel` 完全一致**。
   这是整个视觉部分唯一不可妥协的一条。
2. **视觉不可用时通话绝不能中断**（`video_call_build_plan.md` §3.6）：
   拒权限 / 被抢走 / 切后台 → 明确提示 + 通话继续，**绝不显示"已开启"**。

#### S7.0 眨眼：**本轮决定暂缓，但分析留档**（2026 本轮）

> **决定**：通话功能先不做眨眼，后续再定。下面是当时查清的结论，
> 留在这里是为了将来重新捡起时**不必再查一遍**，也避免有人"顺手把它加回去"。

**技术上能接，且不需要重写算法。** 眨眼链路不在被删的 `useBlinkDetection.ts` 里，
它挂在 `useFacialAnalysis → useEyeTracking → blinkLogic.BlinkStateMachine` 上，
而 `useEyeTracking` 本来就是 `AnxietyContext` 里实例化的 hook。工作量在 UI，不在算法。

**但有三个必须先解决的问题：**

| # | 问题 | 证据 |
|---|---|---|
| 1 | **10 fps 采样下"眨眼频率"测不准** | `useFacialAnalysis.ts:577` 是 `100ms`（MediaPipe 路径）。而 `blinkLogic.ts:40-46` 的 `minMs=60` / `minGapMs=100`：真人眨眼 100–400 ms → 只采到 1–4 帧；`minMs=60` 在 100 ms 采样下**不可分辨**；`minGapMs=100` **等于**采样间隔 → 相邻两次眨眼难以区分。这不是"有点误差"，是量纲级问题 |
| 2 | **眨眼现在会进风险、会写库** | `blinkLogic.ts:158-161` 从眨眼算 `anxietyIndex` → `useEyeTracking.ts:104-107` 加到 `riskScore` → `AnxietyContext.tsx:252` 生成文案 → `:940` 随 `reportToServer` 上报落库。通话期必须切断（即 F7） |
| 3 | **构念效度弱** | `blinkRate` 受干眼、屏幕亮度、隐形眼镜、咖啡因影响远大于情绪；`ageConfig.ts:361-363` 的 `highBlinkThreshold=30` 是在"坐在摄像头前"场景标定的，通话场景**未验证** |

**若将来要做，推荐三条路（按诚实度排序）：**

| | 做法 | 成本 | 说明 |
|---|---|---|---|
| **A** | 通话期只做**「长闭眼次数」+「当前是否闭眼」**，不做"次/分" | 零 | 长闭眼 >400 ms 在 10 fps 下能可靠采到（4 帧）；"是否闭眼"是单帧判据。**采样率不敏感的量才是真话** |
| **B** | 通话期把面部帧率提到 **30 fps** 才做频率 | MediaPipe 13.9 ms/tick [实测] → 10 fps 占 14% 单核，**30 fps 占 42%** | 本机 20 核扛得住；4 核机器不行。且与"存在感只需 2–5 fps"的省电方向相反 |
| **C** | 通话里不做眨眼，只做存在感 + 光线 | 零 | 最省，产品叙事少一块 |

**同批一起放弃的还有注视方向**：`downwardGazeRatio` 名字说的是"目光向下"，
实际算的是**头部角度**（`atan2(matrix[6], matrix[10])` 得 pitch）。通话里低头看手机是常态，
会被持续误报成"回避"。要做也得先改名 `headDownRatio`（F8）。

**砍掉 S7 之后**：通话功能仍然完整（能说、能听、能出声、能打断、危机兜得住），
产品叙事仍然成立（"AI 陪伴通话"），只是没有摄像头。**S0–S6 一条都不能砍。**

#### S7.1 执行结果（✅ 本轮完成，按"不砍"做）

**改动文件**

| 文件 | 内容 |
|---|---|
| `client/src/hooks/useCallVision.ts`（**新建**） | 通话期视觉：存在感 / 光线 / 头部俯仰 + 低头占比；**不含任何上报、不碰 `AnxietyContext`**；F9 的三个失败场景都如实表达 |
| `client/src/pages/patient/VoiceCall.tsx` | 摄像头开关、右下角自拍小窗、三行小读数、"画面本机分析 · 不进风险"标签、失败提示 |

**它到底做了什么（三件，都是诚实的）**

| 做什么 | 用什么算 | 实测读数（无头假设备） |
|---|---|---|
| ① 存在感 | `FacialAnalysis.facePresent`（帧级） | "你在画面里 / 暂时看不到你" |
| ② 画面质量 | 视频帧绿通道均值（32×32 画布） | `153/255` → "光线正常" |
| ③ 给自己看的小读数 | `actionUnits.headPitch` | `头部 0° · 低头 0%（>25°且>3s）` |

**它明确不做**（写进代码注释，也写进答辩材料）：

* ❌ 不进融合、不进风险、不上报 —— hook 里没有任何 `fetch`/`reportToServer`；
* ❌ 不做眨眼频率（10 fps 下量纲级不准，§S7.0 已按用户决定暂缓；**连字段都不暴露**，
  免得将来被"顺手加回去"）；
* ❌ 不做注视方向 —— 用的是 `headPitchDeg` 这个诚实名字（F8 的构念错误在通话里绕开了，
  全局 `EyeMovementAnalysis.downwardGazeRatio` 的改名**仍未做**，见 §7 Q11）。

**F9 的三个失败场景**

| 场景 | 处理 | 实测 |
|---|---|---|
| 用户拒权限 | 明确提示"摄像头不可用（权限被拒绝）（已改为纯语音通话）"，**通话继续** | ✅ 提示出现，相位仍是「正在聆听」，live 音轨仍是 1，**不显示"已开启"** |
| 设备被抢走/拔掉 | 订阅 `mediaStreamManager.onEnded('camera')` → 置不可用 + 显示原因 | ✅（代码路径，本轮未能在无头环境里真实抢走设备） |
| 切后台标签页 | `visibilitychange` → `paused`，停表、不把暂停期算进低头占比 | ✅（同上，逻辑已实现） |
| 帧陈旧（>3× 采样间隔没新帧） | `frameFresh=false` → 读数显示"—"，**不参与任何统计** | ✅ 首拍就如实报"画面已停住，读数暂空" |

**S7 闸门（`_probe_call_s5.js` 会话 A）**

| 检查 | 判据 | 实测 |
|---|---|---|
| 开摄像头 | 自拍小窗（`<video>` 真的挂在 DOM 上）+ 三件事上屏 | ✅ |
| 真在采样 | 采样帧率 > 0 且"画面新鲜=是" | ✅ 1–3 Hz（低帧率模式，存在感不需要 10 fps） |
| 关摄像头 | 小窗消失、读出归位 | ✅ |
| **F7 硬闸门** | 静音冻住对话 + 等级稳定后，开/关摄像头两轮 → 落库 `riskLevel` **完全一致** | ✅ 三轮都是 `CRISIS` |
| 不上报 | 全程 `fetch` 里 **0 条**含 `facial`/`eye`/`headPitch`/`/perception` 的请求 | ✅ `fetchCount=22`，命中 0 |

> ⚠️ 这条闸门第一版**写错过**，值得记一笔：它比的是"通话前 vs 通话后"，而这一通电话本身
> 会产生多轮对话，每轮结束服务端都会 fire-and-forget 一次 `analyzeAndUpdateProfile`
> **按对话文本**改风险等级 —— 于是断言失败时看起来像"视觉闯的祸"，其实是文本改的。
> 现在的判据把变量冻住了：**先静音**（不再有新轮次）→ **等等级连续 6 秒不变** →
> 再开/关摄像头两轮。这样"等级若变化就只可能来自视觉"才真的成立。

**没做的一件事（如实记）**：`useFacialAnalysis` 里的 MediaPipe wasm 仍从
`cdn.jsdelivr.net` 加载（**F12 未做**）。答辩现场若没有外网，摄像头这条路会起不来 ——
但它**不影响通话**（视觉本来就是可选层），而且失败会走 §3.6 的提示路径。

---

### S8 —— 生产可用性（`/algorithm` 代理缺口）

**现状 [实测-代码]**：`client/vite.config.ts` 里 `/algorithm` → 8001 的代理**只在 dev 生效**；
Node 服务端（`server/src/app.ts`）既没有 `/algorithm` 代理，也没有 `express.static`。

**后果**：`npm run build` 出来的生产包，TTS 的 WebSocket 会 **404** ——
而且这个失败模式与本次会话已经踩过的"8001 用错解释器导致 `/asr/ws` 404"**完全同型**：
页面正常、按钮正常、就是没声音。

**两个选择：**

| 方案 | 做法 | 适用 |
|---|---|---|
| **A. 演示走 dev**（推荐） | 演示就用 `npm run dev`（vite 5173 + 代理），不做生产构建 | 竞赛 Demo |
| B. 补代理 | 在 Node 服务加一个支持 `ws: true` 的 `/algorithm` 反向代理 | 要交付生产包时 |

**若选 A**：在 `algorithm/README.md` 与本文档里都写清"演示必须用 dev 模式"，
并把它列进 S9 的走查清单。

#### S8.1 执行结果（✅ 本轮完成 —— 选了 B，把缺口补上）

**为什么没选 A（走 dev 演示）**：A 是"绕过去"，但它把生产包留成一个**必然不会出声**的构建 ——
任何人拿到 `npm run build` 的产物都会踩同一个坑，而且症状与本次会话踩过的
"8001 用错解释器导致 `/asr/ws` 404" **完全同型**（页面正常、按钮正常、就是没声音）。
代理本身只有 20 行，补上比写一条免责声明更省事，也更诚实。

**改动**

| 文件 | 内容 |
|---|---|
| `server/src/services/algorithmProxy.ts`（**新建**） | `/algorithm` → `127.0.0.1:8001` 的 HTTP 转发 + **`upgrade` 事件的 WebSocket 转发**（去掉前缀、原样透传） |
| `server/src/app.ts` | 挂载它（在 `errorHandler` 之前，且接管同一个 http server 的 `upgrade`） |

三个实现要点：

1. **必须同时处理 `upgrade`**。只做 HTTP 代理等于没做 —— 通话的两个关键通道（ASR/TTS）都是 WebSocket。
2. **不能碰 Socket.IO 的握手**：两者挂在同一个 `upgrade` 事件上，所以不是 `/algorithm/` 开头就**立刻返回**，
   一个字节都不写进 socket（否则表现为"聊天页实时消息时好时坏"）。
3. **不装 `http-proxy-middleware`**：为一条"去掉前缀、其余原样"的规则加一个运行时依赖不划算，
   直接用 `http`/`net` 写。另加：客户端在握手包里带上来的字节（`head`）必须一起转发，否则首个音频块会丢。

**S8 闸门（`_probe_prod_proxy.js`，对着 `npm run build` 出来的产物跑；6/6 通过）**

```powershell
cd server; $env:SERVER_PORT=3100; node dist/app.js     # 生产产物，另一个端口
node _probe_prod_proxy.js --port 3100
```

| 检查 | 实测 |
|---|---|
| 生产服务在线（`/api/health`） | ✅ 200 |
| HTTP 代理 `/algorithm/api/v1/asr/status` | ✅ 200，`streaming_available=true` |
| HTTP 代理 `/algorithm/api/v1/tts/status` | ✅ 200，`engine=sherpa-onnx-vits 22050Hz` |
| 上游 404 如实透传（不是 502/挂起） | ✅ 404 |
| **WS 代理 `/algorithm/api/v1/asr/ws`** | ✅ 握手成功并收到 `status`（`mode:"streaming"`） |
| **WS 代理 `/algorithm/api/v1/tts/ws`** | ✅ 握手成功并收到 `status`（`num_speakers=804`） |

> 演示仍建议用 `npm run dev`（改动即时生效、有 HMR）。S8 的意义是：
> **"生产构建装不出声"这个坑不存在了**，而不是"演示要改成生产模式"。


---

### S9 —— 演示准备

#### S9.1 预热（**不做的话演示会很难看**）

**冷启动 6.3 s [实测]**（`voice_call_plan.md` §1.2）—— 感知模型 + 认知扭曲分类器 + 向量检索首次加载。
若演示时第一句话就吃这 6.3 s，观感直接崩。

新建 `scripts/warmup.mjs`，在演示前跑一次，把三件事都预热：

```powershell
npm run warmup      # 新增脚本：加进根 package.json
```

| # | 预热什么 | 怎么做 |
|---|---|---|
| 1 | 算法服务（8001）已就绪 | 读 `/api/v1/asr/status`（`streaming_available` 必须 true）+ 新增的 `/api/v1/tts/status` |
| 2 | LLM 冷启动 | 打一次 `/api/v1/intervention/smart-chat`（非流式，最简单的一句） |
| 3 | TTS 模型加载 | `/tts/ws` 连一次、合成一句短文本 |

#### S9.2 演示脚本（3 分钟，与 `video_call_build_plan.md` §3 的效果对齐）

| 时间 | 动作 | 要展示的点 |
|---|---|---|
| 0:00 | 从聊天页点 📞 进入通话 | 整屏、无侧边栏、AI 头像呼吸动效 |
| 0:10 | 说"最近老是睡不着，晚上总想事情" | **边说边逐字上屏** |
| 0:20 | 停口 | AI 0.5s 内开始回应字幕，~1.6s 出声 |
| 0:40 | 中途插话说"其实是因为考试" | **打断生效**，本地立刻静音，然后接着答 |
| 1:10 | 开摄像头 | 右下角自拍小窗 + "你在画面里" + 光线提示 |
| 1:30 | 关摄像头 | 小窗消失、分析停止（**不产生任何风险结论**） |
| 1:50 | 说危机文本（预先准备好的台本） | 立刻静音 → 全屏热线卡片 → 会话切纯文字 |
| 2:20 | 回到聊天页 | 历史记录完整（含刚才的每一轮 + 打断那轮） |

#### S9.3 交付前走查清单

- [ ] `npm run dev:algorithm:check` 通过（解释器 / 依赖 / 端口三道闸）
- [ ] `npm run warmup` 通过
- [ ] `client`/`server` 构建通过；`pytest algorithm/tests` 通过数**不减少**
- [ ] 通话页进出 3 次 → 0 live 轨道、0 泄漏定时器、0 未处理 rejection
- [ ] 断网 5 秒 → 提示"网络不稳定，正在重连…"，已说出的字幕不丢
- [ ] 静音麦克风 → **会话与安全监测继续**（无音频则无文本，此时不做风险推断）

#### S9.4 执行结果（✅ 本轮完成）

**预热脚本（`scripts/warmup.mjs`，`npm run warmup`；8/8 通过）**

一次跑完四件事，并把每一步的实测耗时打出来（冷启动那 6.3 s 就是被它消化掉的）：

| 预热什么 | 实测 |
|---|---|
| 算法服务可达 | 28 ms |
| ASR 流式引擎可用 | `sherpa-onnx-streaming-zipformer` |
| 离线标点模型可用 | ✅ |
| TTS 引擎可用 | `sherpa-onnx-vits 22050Hz 音色=804 max_num_sentences=-1` |
| LLM 冷启动（`smart-chat`） | 563 ms |
| TTS 模型加载（合成一句 → 首段 PCM 到手） | 211 ms（首段 204 ms） |
| 流式 ASR 模型加载（推一段音频 + `finalize`） | 16 ms |
| **Node 服务 `/algorithm` 代理**（顺手探一次，§S8） | ✅ HTTP 通 |

> 脚本还会**顺手校验两个 F-6 的坑**：`max_num_sentences` 必须是 `-1`（否则长句被静默截断）、
> TTS 采样率必须从服务端读（客户端不得写死 16 kHz）。任一项不对就红着退出。
> 算法服务没起来时它给一句明确指引：`npm run dev:algorithm`。

**交付前走查清单（§S9.3）的实测结果**

| 检查 | 结果 |
|---|---|
| `npm run dev:algorithm:check` | ✅ 通过（解释器=项目 venv、依赖齐全、端口可用） |
| `npm run warmup` | ✅ 8/8 |
| `client` / `server` 构建 | ✅ `tsc` 0 错；client 生产构建 17.9 s |
| `pytest algorithm/tests` 通过数不减少 | ✅ **1183 passed**（8 failed 全部是 `ModuleNotFoundError: No module named 'matplotlib'` 的**环境缺包**，与本次改动无关；本次没有改 `algorithm/` 任何一行） |
| 通话页进出 3 次 → 0 live 轨道 / 0 泄漏定时器 / 0 未处理 rejection | ✅ `_probe_call_page.js` P12 / P12b / P12c |
| 断网 5 秒 → 提示"网络不稳定，正在重连…"，已说出的字幕不丢 | ✅ A24 / A25（15 条用户字幕、6 条 AI 字幕，断网前后**逐条一致**） |
| 静音麦克风 → 会话与安全监测继续 | ✅ A19b（静音后 live 音轨仍为 1、相位仍「正在聆听」） |

**新增的一条走查项（本轮才发现要做）**

| 检查 | 为什么 | 结果 |
|---|---|---|
| **AI 复述用户时，用户还能不能插上话** | 安全审计会把回复改写成复述用户（实测重叠 92%），那种句子当"回声指纹"会让用户**永远插不上话** | ✅ A12c：复述句不再登记为指纹（0.579 ≥ 0.40）；打断随后恢复正常 |

**仍然挂着的三项（都不是本轮范围，如实列出来）**

| 项 | 状态 | 影响 |
|---|---|---|
| **真人 10 次插话 ≥ 9 次成功打断** | ⏳ 需人工在真机（扬声器+麦克风）上做 | 无头环境的"麦克风"只有一个文件，AI 的声音到不了它，原理上测不了 |
| **F12**：MediaPipe wasm 仍走 CDN | ⏳ 未做 | 答辩现场无外网时摄像头起不来；**不影响通话**（视觉是可选层，失败会走提示路径） |
| **眨眼 / 注视方向**（§S7.0） | ⏳ 按用户决定暂缓 | 通话功能完整；`downwardGazeRatio` 的构念改名（F8）全局仍未做 |

**演示前的最小动作（3 条）**

```powershell
npm run dev:algorithm:check      # 三道闸：解释器 / 依赖 / 端口
npm run warmup                   # 把冷启动消化掉
npm run dev                      # vite 5173 + server 3000（演示用 dev）
```

打开 `http://localhost:5173/chat` → 电话按钮 → **开始通话**。
要现场取数就加 `?debug=1`（面板里每个数字都是这一通电话实测出来的）。

---

### S5–S9 改动总览（本轮）

**产品代码**

| 文件 | 动作 | 属于 |
|---|---|---|
| `client/src/services/echoFilter.ts` | 重写：`EchoFilter` 类（带时间窗）+ 可导出的原始分数 + 标定常量 0.40 / 2000 ms | S5 |
| `client/src/services/streamingAsr.ts` | 加 `setForwarding()` / `isForwarding` | S5 |
| `client/src/services/mediaStreamManager.ts` | 默认麦克风约束加 AEC / NS / AGC（`ideal`） | S5 |
| `client/src/hooks/useVoiceAnalysis.ts` | 透出 `setForwarding` / `getMicSettings`，新增 `onVoicePartial` | S5 |
| `client/src/hooks/useVoiceCall.ts` | 打断路径、半双工窗口、插话排队、号码守卫、可发音守卫、复述用户不登记、危机分支、并发轮次峰值 | S5 S6 |
| `client/src/hooks/useCallVision.ts` | **新建**：通话期视觉（存在感 / 光线 / 头姿），不进风险不上报 | S7 |
| `client/src/components/CrisisResourceCard.tsx` | **新建**：危机卡片（文案与号码一字未改） | S6 |
| `client/src/pages/patient/Chat.tsx` | `CrisisModal` 改为复用上面那个组件 | S6 |
| `client/src/pages/patient/VoiceCall.tsx` | 危机浮层、摄像头开关 + 自拍小窗 + 小读数、断网提示、诊断面板扩到 20 项 | S5 S6 S7 S9 |
| `server/src/services/algorithmProxy.ts` | **新建**：`/algorithm` 的 HTTP + WS 代理 | S8 |
| `server/src/app.ts` | 挂载代理（在 `errorHandler` 之前，接管 `upgrade`） | S8 |
| `scripts/warmup.mjs` | **新建**：演示前预热（HTTP + 两个 WebSocket） | S9 |
| `package.json`（根） | 加 `npm run warmup` | S9 |

**可复现的实验与闸门（都在仓库外 `D:\Develop\AIC\`，不进版本库）**

| 文件 | 作用 |
|---|---|
| `_exp_echo_coupling.py` | S5.1 实验：造 40 组 `(spoken, heard)` 配对（真实 TTS → 三种声学耦合 → 真实 ASR） |
| `_exp_echo_decide.js` | 用 esbuild 打包**真实 `echoFilter.ts`** 算表、扫阈值，并生成 `docs/echo_calibration.md` |
| `_probe_call_s5.js` | **S5/S6/S7 闸门（37/37）**：两个浏览器会话（普通句 / 危机句各一个假麦克风夹具） |
| `_probe_prod_proxy.js` | **S8 闸门（6/6）**：对着 `npm run build` 的产物验 HTTP + WS 代理 |
| `_probe_crisis_sse.js` | 抓危机轮的原始 SSE 事件流（直连 3000 / 经 vite 5173 各一次） |
| `_probe_asr_modes.js` | 量 `streaming` / `utterance` 两种模式各推多少次 partial / final（打断判据的前提） |
| `_make_asr_fixture.py` | 造假麦克风 WAV（`user_says.wav` 用 0.8s/1.0s 静音提高语音占空比；`crisis.wav` 是危机句） |
| `_probe_call_page.js` / `_probe_call_voice_e2e.js` | S3（27/27）/ S4（14/14）回归 |

**四条闸门合计 84 项全绿**：`27（S3） + 14（S4） + 37（S5/S6/S7） + 6（S8）`，外加 `8/8`（S9 预热）。

---

## 4. 协议契约汇总

### 4.1 客户端 → 服务端

**只用了两条既有通道，没有新增任何文本侧协议：**

| 通道 | 端点 | 用途 |
|---|---|---|
| ASR WS | `/algorithm/api/v1/asr/ws` | 音频上行 / partial / final（既有） |
| 对话 SSE | `/api/consultation/conversations/:id/messages/stream` | 定稿文本上行 / meta·delta·revise·done（既有） |
| **TTS WS**（新） | `/algorithm/api/v1/tts/ws` | 待合成句上行 / PCM 下行 |

### 4.2 一次完整轮次的数据流

```
🎤 麦克风
  → AudioWorklet(16kHz, 100ms/块) → SpeechGate → /asr/ws
  → partial（每 ~330ms，上屏 + 打断判据）
  → final（端点 0.45–1.2s）
        │
        ├─ EchoFilter.isEcho? ──是──→ 丢弃
        │                      否
        ▼
  sendChatMessageStream(final)
        │
        ▼  meta(streaming?) → delta*(过闸的完整句) → [revise] → done
        │
        ├─ meta.streaming === false ──→ 危机：停播 + 卡片（F-3 / S6）
        │
        ├─ delta ──→ 字幕追加
        │            └→ splitSpeakable(首块按逗号切到 ≤12 字) ──→ /tts/ws {"type":"say"}
        │                                                    │
        │                             1 个二进制帧（整句 PCM，VITS 不流式）← 串行合成
        │                                                    ▼
        │                                            TtsPlayer 排程播放
        │                                              （边播边合成下一句，流水线）
        │                                                    │
        │                                     onStart → setForwarding(false)，400ms 后恢复
        │                                     note(text) → EchoFilter 登记
        │
        └─ revise ──→ 停播 + 清队列 + 前缀比对后决定是否重播（§4.5）
```

### 4.3 事件与状态的对应（防漏事件）

| SSE 事件 | `useVoiceCall` 必须做的事 | 漏了会怎样 |
|---|---|---|
| `user` | 把乐观气泡替换成服务端落库版本（通话页不显示气泡，可忽略但**不能崩**） | —— |
| `meta` | 记 `streaming` → `voiceAllowed`；`false` 立刻进 `crisis` | 危机时仍在出声 |
| `delta` | 字幕追加 + 逐句合成 | 有字幕没声音 / 有声音没字幕（不同步） |
| `revise` | 字幕整体替换 + 停播 + 前缀策略 | **不安全原文与安全改稿叠在一起** |
| `done` | 危机判定、发插话队列、状态回 `listening` | 通话卡死在一轮里 |
| `error` | 提示 + 回 `listening`（**不要停麦克风**） | 一次抖动就断了整通电话 |

---

## 5. 验收清单（可直接勾）

**S0 可行性（✅ 全部达成，见 [`docs/tts_benchmark.md`](./tts_benchmark.md)）**
- [x] `docs/tts_benchmark.md` 有真实数字，不是估计
- [x] 空载 RTF **0.180**（4 线程）/ **0.112**（8 线程）< 0.4
- [x] 满载（同时跑流式 ASR）RTF **0.209 / 0.136** < 0.4
- [x] 最短句整句合成 **336 ms**（8 线程满载）< 400 ms
- [ ] 导出的 WAV **有人真的听过**并确认中文可懂 —— **仍然未做，等人耳**
- [ ] `noise_scale` 置 0 能否让音频可复现（可选，未测）
- [ ] `num_speakers = 187` 里挑一个合适的 `sid`（可选，等人耳）

**S1/S2 出声链路（✅ 全部完成）**
- [x] `/tts/status` → `available: true`，且如实报出 `num_threads` / `max_num_sentences`
- [x] WS 送 4 句 → 音频时长 ≈ 3.9×（**不是** ≈ 1×）
- [x] `cancel` → 不发音频、回 `cancelled`
- [x] 术语替换生效（`spoken_text` 已是替换后的文本）
- [x] 截断护栏自检：配置改坏时测试变红
- [x] 连播 5 句无缝隙（排程偏差 `[0,0,1,-1]` ms）
- [x] `stopAll()` < 50 ms（实测 **0.1 ms**）
- [x] 经 5173 代理的 WS 握手（已并入 S2 的浏览器实测）
- [ ] **人耳听音质** —— `_tts_bench/*.wav` 已生成，**仍等人听**

**S3 通话页骨架（✅ 全部完成，`_probe_call_page.js` 27/27）**
- [x] `/call` 全屏、无侧边栏（`hasSider === false`）
- [x] 无 `:conversationId` 时**只**建 1 个会话并 `replace` 到 `/call/<id>`
- [x] 已有 `:conversationId` 时**不**建会话
- [x] 点"开始通话" → 1 条 live 音轨、麦克风只申请 1 次
- [x] 静音 ≠ 挂断（轨道 `enabled=false` 但仍 live）
- [x] 挂断 → 回 `/chat/<id>`
- [x] 进出通话页 → 0 live 轨道、0 残留定时器、0 未处理 rejection
- [x] 首段合成 347 ms（TTS 这一腿）
- [x] 进出通话页 **3 次**（SPA 内，不刷新）→ 仍 0 泄漏（P12/P12b/P12c）

**S4 通话闭环（✅ 全部完成，`_probe_call_voice_e2e.js` 14/14）**
- [x] 用户说话 → 本机流式 ASR 中间结果上屏 → 定稿（V2）
- [x] 定稿 → 一轮：user 消息**落库**（V6）
- [x] AI 字幕上屏 + assistant 消息**落库**（V3 / V7）
- [x] TTS 真的收到 PCM 并出声（V4）
- [x] 首段合成 **246 ms** < 2400 ms（V8；口径 = `takePcmReadyMs`，只取本轮第一段 —— 见 §1 F-7）
- [x] 队列排空 + 本轮结束 → 自动回到聆听，无需任何点击（V5）
- [x] 全程只有 1 条麦克风轨道；麦克风只申请 1 次（V9）
- [x] 删掉 `?demo=1` 骨架自检面板，换成 `?debug=1` 实测读数面板
- [ ] 字幕最终文本 === `done.text`（含危机热线补全）—— **S6 一起做**（当前危机只做静音 + 提示）
- [ ] ASR 端点判定耗时单独计时（定稿延迟里唯一还没拆出来的一段）

**S5 打断**
- [ ] `docs/` 里有 10 组 `(spoken, heard)` 回声配对数据
- [ ] 回放那 10 句 AI 语音 → **0 次误打断**
- [ ] 10 次真人插话 → ≥9 次成功打断
- [ ] 插话 → 本地静音 < 200 ms
- [ ] 打断后无并发轮次、无第二次 LLM 调用（看服务端日志确认）

**S6 危机**
- [ ] 危机文本 → 0 个 PCM 帧达播放器
- [ ] 热线只出现一次
- [ ] TTS `say` 的文本从不含电话号码
- [ ] `requires_escalation=True`

**S7 视觉（若做）**
- [ ] 开/关摄像头两次 → 落库 `riskLevel` **完全一致**
- [ ] `reportToServer` payload 不含 `facial`/`eye`
- [ ] 拒摄像头权限 → 通话继续，绝不显示"已开启"

**回归**
- [ ] `pytest algorithm/tests` 通过数不减少（当前基线 **1132 passed / 8 failed / 1 skipped**，
      8 个失败全部是缺 `matplotlib`，与本次无关）
- [ ] `test_chat_stream.py` + `test_intervention_prompt.py` 全绿（138 passed）
- [ ] 纯文本聊天行为不变：META 75–108 ms、首 delta 442–497 ms [实测]

---

## 6. 里程碑与交付物

| 里程碑 | 含哪几步 | 交付物 | 可以演示什么 |
|---|---|---|---|
| **M1 出声** | ~~S0~~ ✅ + S1 + S2 | `docs/tts_benchmark.md`、`/tts/ws` | 敲一句话，AI 说出来 |
| **M2 通话** | + S3 + S4 | `/call` 页面 | 说话 → 字幕 → AI 出声 → 自动下一轮 |
| **M3 完整** | + S5 + S6 | 打断、危机卡片 | 能插话、危机兜得住（**建议到此为止**） |
| **M4 加分** | + S7 | 摄像头三件事 | 视觉叙事完整 |

**M3 是"能拿去比赛"的线。M4 是加分项。S8/S9 无论做到哪一步都必须做。**

---

## 7. 待决 / 需人工拍板 / 未验证

| # | 事项 | 类型 | 谁定 |
|---|---|---|---|
| **Q1** | **D1**：语音走本地（Path A）还是上云（Path B） | 产品/合规 | **人**（S0 之后才有得选） |
| **Q2** | 危机时那句"预审核短句"的措辞 | ⚠️ 新增用户可见安全文案 | **人**（本计划刻意没拟） |
| **Q3** | 是否给服务端加"软打断"参数（省掉插话等 `done` 的 0–2 s） | 改安全链路附近的代码 | **人** |
| **Q4** | `revise` 时"是否重播"的策略（§4.5） | 产品判断，影响用户听到什么 | **建议 reviewer 过一眼**。⚠️ **S4 已先按保守半边实现**（停播 + 只改字幕、**不重说**），等有人拍板再决定要不要做"补说剩余部分" |
| **Q5** | 备选 TTS 模型 `huayan` 的 License | 法务 | **人**（`MODEL_CARD` 写 `Unknown`） |
| **Q6** | 是否把 AEC 约束加进**默认**麦克风约束 | 会同时影响多模态感知的识别质量 | ✅ **本轮已加**（S5.3）：`echoCancellation/noiseSuppression/autoGainControl` 用 `ideal` 而非硬约束（硬约束在部分设备上会 `OverconstrainedError`，把麦克风整个弄不能用）。通话页读 `getSettings()` 的实际值公示：实测 `AEC=开 NS=开 AGC=开 48000Hz`。**若多模态感知侧发现识别质量下降，改的是这里的三个开关，不是通话代码** |
| **Q7** | `/algorithm` 代理：演示走 dev 还是补生产代理 | 交付形态 | ✅ **本轮选了"补代理"**（S8.1）：`/algorithm` 的 HTTP + WS 代理都进了生产包（6/6 通过）。演示仍建议 `npm run dev`，但"生产包装不出声"这个坑已经不存在 |
| **Q8** | **用哪个人声 / 哪个 TTS 模型**（187 vs 804 个音色；本地 16 kHz vs 22 kHz vs 云端） | 听感，只有人耳能定 | **人**（试听材料见 §11.4）。注意：现在 8001 上跑的是 `theresa`（22 kHz，靠 `TTS_MODEL_DIR` 环境变量），**仓库默认仍是 `fanchen-C`** —— 重启服务会回到 fanchen-C，这是有意的（等听感结论再改默认值）。换模型后请重跑 `_exp_echo_coupling.py` 重新标定回声阈值（命令见 `docs/echo_calibration.md` 文首） |
| **Q10** | S4 的**半双工**要不要保留 | 产品判断：AI 说话时用户无法插话，语音被丢弃 | ✅ **本轮已放开**（S5）：半双工窗口缩到"每次出声后的 400ms（AEC 收敛期）"，之后靠 AEC + 文本级回声过滤。实测：打断在判定后 **0 ms** 静音、并发轮次峰值恒为 1、误打断由 30 组标定数据背书 |
| **Q11**（新增） | 全局把 `EyeMovementAnalysis.downwardGazeRatio` 改名为 `headDownRatio`（F8 的构念改名） | 影响融合与上报的字段名 | **人**。通话页已经绕开（它只用 `headPitchDeg`），但全局那个名字仍在说"目光向下"、实际算的是头部角度。改它要动 `useEyeTracking` / `blinkLogic` / `AnxietyContext` / 上报 payload，属于"安全链路附近"的改动 |
| **Q9** | 若走云端 TTS：「AI 回复文本会上云」的如实说明怎么写进答辩材料 | 诚信表述 | **人**（论证见 §11.6，但前提是"LLM 在云端"） |

**未验证项（S0 已消掉两条，其余沿用 `voice_call_plan.md` §7；本节只列与施工直接相关的）：**

| # | 说法 | 状态 |
|---|---|---|
| 1 | ~~`hf-mirror` 能下全 `fanchen-C` 的文件~~ | **✅ 已消除** —— 137.7 MB 全部下齐（顺带发现原 `required` 列表漏了整个 `dict/`） |
| 2 | ~~fanchen-C 本机 CPU RTF ≈ 0.15–0.25~~ | **✅ 已消除** —— 实测 0.180（4 线程）/ 0.112（8 线程）空载 |
| 3 | ~~TTS 首 callback 100–300 ms~~ | **✅ 已消除，且原假设错误** —— VITS 不流式，callback 恒 1 次 |
| 4 | Chromium AEC3 收敛 200–500 ms | **[估计]** 工程共识，本机没实测 —— S5.1 会顺带量到 |
| 5 | 方案 D 的拦截率 | **[估计]** 阈值是拍的 —— **S5.1 就是来消掉这条的** |
| 6 | **4–8 核普通笔记本上的 RTF** | **[未验证]** 本机 20 核。按 `docs/tts_benchmark.md` §4.3 外推，4 核 + 2 线程 ≈ **0.29，余量很小**。**这是 Path A 最大的环境风险** |
| 7 | **中文音质好不好听** | **[未验证]** WAV 已导出到 `_tts_bench/`，**必须有人听** |
| 8 | **端到端首声**（浏览器 ↔ 8001 的 WS + 播放队列） | **基本消除** —— S4 闸门在**真实浏览器 + 真实 SSE + 真实 TTS** 下实测：定稿 → 首个 `delta` **782 ms**，首段「请求 → PCM 到手」**246 ms** ⇒ **定稿 → 首声 ≈ 1.03 s**。**仍未单独计时的**：ASR 端点判定（服务端等尾随静音那段）与真实说话长度 |

> ⚠️ 第 8 条同时说明"**1.6 s 中位**"这个估计偏悲观了：S4 实测的 1.03 s 里已经含
> 了 LLM 首句与 TTS 整段合成。真实端到端还要加上"用户说完 → 服务端判定端点"的等待，
> 那一段取决于说话人的停顿习惯，不是常数 —— 答辩时应当说
> "**从定稿到出声约 1 秒**"，而不是拿一个总时长当中位数。
| 9 | 浏览器在跑（MediaPipe / 基频估计 / 音频图）时的 TTS 争用 | **[未验证]** S0 的"满载"只拉了流式 ASR，没有浏览器那一路 |
| 10 | `noise_scale` 置 0 的确定性与音质影响 | **[未验证]** VITS 时长随机（同句 3945/5938/6125 ms），置 0 可能解决可复现性 |
| 11 | `OfflineTts` 回调的中止语义是否真能省算力 | **✅ 已消除** —— 回调只在结束后触发一次，**省不下计算**，取消要靠清队列 |
| 12 | Path A 首声中位 1.6 s | **[估计]** S0 把它收窄到 **1.35–1.97 s**（TTS 贡献已实测），仍是给期望管理用的假设值 |
| 13 | **[新]** `fanchen-C` 词表无拉丁字母 ⇒ `CBT` 等缩写被吞 | **✅ 已确认**（`Ignore OOV 'CBT'`），对策是送合成前的中文替换表 |

---

## 8. 不做什么（写进答辩材料，避免吹牛）

* ❌ **不做全双工同时说话** —— 明确降级为「半双工 + 可打断」，并如实告知用户。
* ❌ **不做口型同步 / viseme 对齐**。
* ❌ **不让视觉影响 AI 的回复内容**，也不让视觉进风险判定（S7/F7）。
* ❌ **不朗读热线号码**（S6.3，且已落成代码）。
* ❌ **不用 `speechSynthesis` 当主路径** —— 它拿不到 `AudioNode`，做不了回声过滤；
  只能在本地 TTS 模型缺失时降级，且**必须如实告诉用户"当前使用在线语音"**。
* ❌ **不改任何安全阈值、不新增安全规则** —— 本计划的全部安全动作都是**复用**
  （F-3 用 `streaming_allowed` 既有布尔量；S6 用既有 escalation 与既有卡片）。

---

## 9. 一页速览

1. **S0 已跑完，Path A 成立。** RTF 最差 **0.180**（4 线程）/ **0.112**（8 线程）空载、
   **0.209 / 0.136** 满载，判据 0.4 —— 余量 2–3 倍。**下一步是 S1。**
2. **但 VITS 不流式（F-6a）**：callback 恒 1 次，所以"首块"就是"整句合成耗时"。
   原判据"首块 < 250 ms"结构性不可达，已改为"最短句整句合成 < 400 ms"。
3. **`max_num_sentences=1` 会静默丢弃第一句之后的所有内容（F-6b）** ——
   66 字文本只产出 4.3 s 音频，**没有报错、没有日志**。必须用 `-1`，并配回归测试。
   这是本次最危险的发现。
4. **通话页不是新架构，是既有流式聊天的壳。** 文本走 Node 的 SSE（F-1），
   设备走 `mediaStreamManager`（F-5），麦克风走 `startVoiceInput`。
5. **`delta` 已经是过闸的完整句，它天然就是 TTS 的单位（F-2）** ——
   不需要写新切句器，不需要碰安全组件。
6. **`meta.streaming === false` 就是高风险禁用语音的开关（F-3）** —— 零新增阈值。
7. **打断不能 abort SSE（F-4）。** abort 会触发第二次 LLM 调用，
   并把一份用户从未听过的回复写进数据库。正确语义是「停播 + 排队下一轮」。
8. **AEC 必须加在默认约束上（F-5）**，因为麦克风全项目只有一路、`acquire` 是幂等的。
9. **S6 有一条要落成代码的红线**：含电话号码特征的片段**永不合成**。
10. **首声靠"首块按逗号切小 + 合成/播放流水线"压到最优**（RTF 0.11 ⇒ 合成比播放快 7 倍，
    队列不可能饿死）。首声中位仍 ≈ 1.6 s，这是 Path A 的诚实天花板。
11. **可砍的是 S7（视觉）；不可砍的是 S1–S6。** M3 就是"能拿去比赛"的那条线。

---

## 10. 上线节奏（把 S1–S9 摊进 7 周）

> 本节是对一份**外部建议的 7 周上线计划**（4 阶段：基础搭建 / 体验优化 / 灰度上线 / 全量上线）
> 的逐条评估与合并结果。记录在这里是为了**避免"两份计划并存"** ——
> 被采纳的部分进了 S1–S9，被否掉的部分在这里写明理由，将来不会被重新提出。

### 10.1 逐条对照

| 外部建议 | 对「AI 陪伴通话」是否成立 | 处理 |
|---|---|---|
| 阶段一：**搭 WebRTC 音视频通道** | ❌ **不成立**。AI 通话**没有对端**，全链路是 麦克风→AudioWorklet→WS(ASR)→SSE(LLM)→WS(TTS PCM)→Web Audio，加 WebRTC 只会引入 STUN/TURN 依赖 | **否掉**。但见 10.2 —— 这条对**另一个功能**是对的 |
| 阶段一：对接 ASR / 大模型 | ✅ 但**已经做完并实测**（`/asr/ws` 可用；META 75–108 ms、首 delta 442–497 ms） | 无需排期 |
| 阶段一：对接 TTS | ✅ 正是 S1 + S2（S0 已证明可行） | 排进 W1–2 |
| 阶段一：极简通话 UI + 跑通单路语音对话 | ✅ 正是 S3 + S4 | 排进 W1–2 |
| 阶段一：1–2 周 | ⚠️ **偏乐观**。S1 含引擎/数据类/WS/回归测试，S4 是全计划最大的一块 | W1–2 只保 S1–S4 主干，UI 细节后移 |
| 阶段二：**加降噪** | ⚠️ **被高估**。`echoCancellation / noiseSuppression / autoGainControl` 是**一行约束**（且必须加在**默认**约束上，见 F-5） | 合并进 S5，不单列 |
| 阶段二：打断响应 | ✅ **正确，而且比看起来难** —— 不能 abort SSE（F-4），且需要先做回声过滤实验（S5.1） | 排进 W3–4 |
| 阶段二：通话状态动效 | ✅ 便宜且对（§3.3） | 合并进 S3/S4 |
| 阶段二：**并发压测** | ❌ **架构前提不成立**。全链路是**本机 CPU 推理**：TTS 8 线程 RTF 0.136、流式 ASR 另占线程。20 逻辑核 ≈ **1–2 路并发**就到顶 | **改为"容量上限声明"**，见 10.3 |
| 阶段二：**弱网适配** | ✅ **本计划确实没有覆盖 —— 认领**。原 S9 只有一条"断网 5 秒"走查 | **新增 W4–5 工作流**，见 10.4 |
| 阶段三：灰度 / 小流量 / 内部用户 | ⚠️ 方向对，但"灰度、小流量"是 SaaS 语汇；本项目的瓶颈不是流量 | 改为"**小范围真实试用**" |
| 阶段三：修延迟 / 断连 / 清晰度 | ✅ 对，但**必须在安全链路之后**（见下） | 排进 W5–6 |
| 阶段三/四：**全程没有安全环节** | ❌ **最大的问题**。这是青少年心理健康产品，AGENTS.md 有红线 | **S6 前置到 W3–4**，见 10.5 |
| 阶段四：**全量上线** | ⚠️ 取决于目标形态（见 10.6） | 待定 |
| 阶段四：**"后续再迭代多轮对话"** | ❌ **多轮对话是前提，不是迭代项**。`consultation.service.ts:318-326` 已在取最近 20 条历史喂 LLM，每轮都落库 | **否掉这个排序** |
| 阶段四：监控面板 | ⚠️ 已有一个（`/admin` 的「算法性能监控」），**但里面的数字是写死的**（`value={23}`、`value={87.3}`） | 加通话指标时**必须真数据**，见 10.7 |
| 阶段四：个性化音色 | ✅ **意外撞上实测发现**：`fanchen-C` 的 `num_speakers = 187`，是多说话人模型，`sid` 可挑音色 | 保留为后续迭代，但**选音色要人耳** |

### 10.2 关于 WebRTC：建议本身没错，只是放错了功能

`client/src/hooks/useWebRTC.ts:24-38` 里已经有 WebRTC，服务的是**真人咨询师视频房间**
（浏览器 ↔ 浏览器点对点）。**那条链路现在不可用**，因为 ICE 配置是：

```ts
{ urls: 'stun:stun.l.google.com:19302' },                    // 国内不可达
{ urls: 'turn:openrelay.metered.ca:80', username: 'openrelay',
  credential: 'openrelay' },                                  // 免费公共 TURN，不可用
```

所以"搭 WebRTC 通道"这条建议**对真人咨询房间是真活儿**（要换成自建 coturn），
只是它**不属于「AI 陪伴通话」**。这两件事在文档里必须分开说，否则会互相污染排期。

### 10.3 容量上限：先声明，别"压测"

**必须写进材料的硬事实**：本架构是**单机 CPU 本地推理**，音频不出设备。

| 资源 | 单路通话占用 |
|---|---|
| TTS 合成 | 8 线程（`clamp(核数-2, 2, 8)`） |
| 流式 ASR | 1–2 线程 |
| 浏览器（MediaPipe / 基频估计 / 音频图） | 客户端侧，若同机再算一份 |

**20 逻辑核 ≈ 1–2 路并发**。这不是调优能解决的：要横向扩展只有两条路 ——
**GPU 服务器**，或 **Path B（音频上云）**，后者与"敏感原始数据不出设备"直接冲突
（`shared/dataclasses.py:290-299` 的 `AsrEngine` 枚举刻意不提供浏览器 Web Speech API，
就是为了守这条线）。

**处理**：不排"并发压测"，改为**写一页容量说明**（单机几路、超了会怎样、
要扩展得改什么）。有真实并发需求时，这是一个**架构决策**，不是优化任务。

### 10.4 弱网适配（认领：这是外部建议里本计划缺的一块）

新增工作流，排 W4–5。要覆盖的四件事：

| 场景 | 现在的行为 | 要做到 |
|---|---|---|
| SSE 断流（网关超时 / 服务重启） | `chatStream.ts:227-233` 会发 `STREAM_TRUNCATED` 错误 | 通话页要**自动重连并续上**，而不是把用户丢在半句话上 |
| ASR WS 断连 | `streamingAsr.ts:186-199` 退避重连 3 次后放弃 | 通话中要**更宽容**（通话更长），并在 UI 显示"正在重连" |
| TTS WS 断连 | 尚未实现（S1 才建） | 与 ASR 同一套退避策略；断连期间字幕照常 |
| 已说出的字幕 | —— | **任何断连都不得丢失已上屏的字幕**（§3.6 已有此要求） |

### 10.5 安全必须前置到 W3–4，不能留到最后 ⚠️

外部建议的 4 个阶段里**一个字都没提安全**。对一个青少年心理健康产品，这是不可接受的：

* AGENTS.md：`risk_level ∈ {high, crisis}` **必须**触发 escalation；
* 禁止诊断性语言、禁止药物建议；
* **绝不朗读热线号码**（S6.3，已落成代码；
  `400-161-9995` 用 TTS 念可能变成"四十万…"，危机场景念错号码后果严重）；
* 热线卡片**只出现一次**（去重已实现）。

**排序原则**：**S6 必须在任何外部用户碰到它之前完成** ——
即"内部试用"（W5–6）之前，S6（W3–4）必须已经绿了。
把安全放到第 7 周等于让真人被试在没有兜底的系统上做危机表达测试。

### 10.6 关于"灰度 / 全量上线"：**已确认 = 竞赛答辩 / Demo 演示**

> **决定（本轮已拍板）**：目标形态是**竞赛答辩 / Demo 演示**，不是真实上线。
> 因此：

| 外部建议里的 | 本项目的替代做法 |
|---|---|
| "全量上线" | **演示就绪**：预热脚本（`npm run warmup`，S9.1）+ 交付前走查清单（S9.3）+ 3 分钟演示台本（S9.2） |
| "基础客服" | **不适用**。改为答辩时的**答疑材料**：容量上限、已知限制、诚实的天花板（1.6 s 首声） |
| "灰度 / 小流量" | **小范围真实试用**（W5–6，3–5 人）。目的不是压流量，是**在有人看着的情况下暴露真实问题** |
| "并发压测" | **容量上限声明**（10.3）：单机 1–2 路；写进答辩材料一句话 |
| 未成年人数据合规与备案 | 按答辩标准处理，**但仍守住 AGENTS.md 的红线**（这些是产品原则，不因不上线而豁免） |

**仍然必须做的两件"非 Demo"的事：**

1. **安全链路（S6）**：答辩现场评委**一定会试**危机文本 —— 静音、卡片、不朗读号码
   这三件事在演示里比任何性能数字都重要。
2. **危机的人工兜底说明**：即使不上线，答辩材料里要写明"生产环境需接入真实危机干预流程"
   （AGENTS.md 要求的 `⚠️ 比赛 Demo 为模拟实现` 标注）。

**不因为"只是 Demo"而打折的**：S6 的安全行为、S7-F7（视觉不进风险）、
容量上限的如实声明、以及所有 `[估计]` 与 `[实测]` 的标注纪律。

### 10.7 监控面板：可以加，但数字必须是真的

`client/src/pages/admin/Dashboard.tsx` 已有「算法性能监控」页签
（推理延迟分布、端侧推理覆盖率 87.3%、融合置信度均值 0.82、"本月危机触发 23"）——
**这些值是写死在组件里的**（`value={23}`、`value={87.3}`）。

本项目已经因为"常量冒充测量值"栽过（`video_call_build_plan.md` 的
F5 `avgDuration = 200`、F6 `confidence = min(0.95, 0.7 + n*0.005)`，后者能改写数据库里的
患者风险等级）。**通话指标如果再做假的，就是第三次犯同一个错。**

要加的通话指标（都必须是真采集）：

| 指标 | 来源 |
|---|---|
| 首声延迟 P50 / P95 | S4 的时间戳（用户停口 → 第一个 PCM 到达播放器） |
| TTS RTF 分布 | `/tts/ws` 的 `end` 帧自带 `rtf` 字段 |
| 端到端轮次延迟 | `done.latency_ms`（已有） |
| 断连率 / 重连次数 | 弱网工作流（10.4）埋点 |
| 打断成功率 | S5 的验收数据 |

### 10.8 合并后的 7 周

| 周 | 阶段 | 内容 | 出口（可验收） |
|---|---|---|---|
| **W1–2** | A 出声与最小闭环 | ~~**S1**~~ ✅ 引擎+WS+测试 · ~~**S2**~~ ✅ 播放器+传输 · ~~**S3**~~ ✅ 通话页骨架 · ~~**S4**~~ ✅ 轮次编排 | **M2**：说话→字幕→AI 出声→自动下一轮 |
| **W3–4** | B 打断 + **安全兜底（前置）** | **S5** 打断（含 S5.1 回声实验 + AEC 默认约束）· **S6** 危机切换 · **S8** 代理缺口 | **M3**：能插话；危机文本→立即静音+卡片；**TTS 从不朗读号码** |
| **W4–5** | C 真实环境与弱网 | 弱网四件事（10.4）· 4 核设备 RTF 复测 · **浏览器端到端首声实测**（S4.6）· **人耳听音质** | 首声 P50/P95 有真数；4 核机器上的结论 |
| **W5–6** | D 小范围真实试用 | 3–5 人真实使用；收集首声/断连/听清度；修复 | 问题清单关闭 |
| **W6–7** | E 视觉（**可砍**）+ 就绪 | **S7**（F7–F9/F12）· **S9** 预热+走查+台本 · 真实监控指标（10.7） | 演示/交付就绪 |

**与外部建议的差异只有四处**，但每一处都是必须的：

1. **去掉 WebRTC**（AI 通话没有对端）；
2. **弱网从"阶段二顺带"提到 W4–5 的独立工作流**（并且要覆盖断流续接，不只是"适配"）；
3. **安全从"没提"提到 W3–4**，必须在任何外部用户之前；
4. **"并发压测"改成"容量上限声明"**，因为架构上本来就只能 1–2 路。

**保留的**：分阶段推进、每阶段有可验收出口、先跑通单路闭环、打断与状态动效、
小范围试用、监控面板、后续个性化音色（且实测有 187 个音色可挑）。

---

## 11. 音色与模型：人耳反馈"难听"之后的处置

> 触发：S1/S2 做完后请人试听，反馈**"声音好难听"**。
> 排查过程与数据在 [`tts_benchmark.md`](./tts_benchmark.md) §10，这里只留结论与决定。

### 11.1 已确认的三件事

| # | 结论 | 依据 |
|---|---|---|
| 1 | **`num_speakers=187` 是真的**，而默认 `sid=0` 是其中最慢的一个 | 9 个 sid → 9 种不同波形；sid=0 是 4389 ms，sid=100 只有 2459 ms |
| 2 | **同系列模型不是同一档位**：`theresa` / `eula` 是 **22050 Hz / 804 音色** | 对比表见 `tts_benchmark.md` §10.2 |
| 3 | **升到 22 kHz 之后 RTF 依然只有 0.23，两条判据全过** | 满载 RTF：fanchen-C 0.136 → theresa 0.232 → eula 0.243 |

**所以这里不存在"音质 vs 性能"的取舍** —— 直接换 22 kHz 的模型更划算。

### 11.2 换模型踩到并修掉的静默降级（与 S1 的同类坑）

`TTS_MODEL_DIR` 指向 `theresa` 时**无声回退到 fanchen-C**：`_usable()` 写死了
`vits-zh-hf-fanchen-C.onnx`，而 theresa 的 onnx 叫 `theresa.onnx`。
已修：onnx 按目录内容识别 + **显式指定不可用时直接失败、绝不回退** +
`status.model_name` 报实际目录名 + 8 条护栏测试。

### 11.3 客户端零改动已验证

22 kHz 下 S2 闸门 **17/17 通过**，且更好（首声 237–272 ms、排程偏差 `[0,0,0,0]` ms）。
原因是 S1/S2 的分层：`ttsStream` 用服务端帧里的 `sample_rate`、
`TtsPlayer` 用 `ctx.createBuffer(..., sampleRate)` 交给浏览器重采样 —— **没有一处硬编码 16000**。

### 11.4 剩下只有一件：用哪个人声，**只有人耳能定**

试听材料（`_tts_bench/voices/`）：

| 文件 | 内容 |
|---|---|
| `audition_all.wav` | fanchen-C **全部 187 个音色**，10 分 04 秒，带时间戳表 |
| `models/audition_vits-zh-hf-theresa.wav` | theresa 抽听 12 个，35 秒 |
| `models/audition_vits-zh-hf-eula.wav` | eula 抽听 12 个，33 秒 |
| `models/<模型>_sidNNN.wav` | 单个音色，圈定后细听 |
| `speed_*.wav` / `length_*.wav` / `noise_*.wav` / `noisew_*.wav` | 参数扫描 |

**判断标准不是"好听"，是"像不像一个耐心的人在听你说话"。**

### 11.5 如果都不满意：那就不是选音色，是技术档位的天花板

| 选项 | 代价 | 说明 |
|---|---|---|
| **云端 TTS** | 网络依赖（演示现场断网就哑）、计费、一次网络往返 | **不违反"数据不出设备"** —— 见 11.6 |
| 换更高档的本地模型 | 体积 / 算力 | 本次只覆盖了 `vits-zh-hf-*` 系列与 piper，还有别的可找 |
| 接受 16/22 kHz 现状 | 无 | 16 kHz ≈ 电话音质（真电话是 8 kHz）：能听清，不够好听 |

### 11.6 云端 TTS 的合规论证（**纠正我之前的一处表述**）

我此前写过"Path B 与'原始数据不出设备'冲突"。**那只对"音频上云"成立**，
对"云端 TTS"不成立 —— 两者要分开：

| 数据 | 是否已经在云上 | 证据 |
|---|---|---|
| 用户说的话（**文本**） | ✅ **已经在** | `intervention.py:626` `_LLM_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'`；`_build_llm_messages` 把用户消息 + 最近 20 轮历史发过去 |
| AI 的回复文本 | ✅ **已经在**（就是千问生成的） | 同上 |
| 用户的声音（**音频**） | ❌ 没有，ASR 全本地 | `AsrEngine` 枚举刻意不提供浏览器 Web Speech API |
| 画面 | ❌ 没有 | 通话期视觉不上报（F7） |

**所以把 AI 的回复文本再发给一个云端 TTS，没有多泄漏任何一样东西。**
"不出设备"这条原则护的是**用户的原始音频**，那条线一条没破。

⚠️ 但这个论证有前提：**它依赖"LLM 在云端"这个现状**。若将来换成本地 LLM，
回复文本就还在本地，那时云端 TTS 会成为第一个出设备的数据 —— 论证需要重做。
这一点必须写进答辩材料，不能只写"用云端 TTS 没问题"。

### 11.7 迁移成本：只换引擎，不动客户端

`TtsPlayer`（S2）与客户端 `/tts/ws` 协议**都不用改**；
只需在 `algorithm/tts/engine.py` 旁边加一个云端实现，由配置选。
这也是 S1/S2 分层刻意留的口子（见 §4.1 的协议契约）。

### 11.8 本轮的排期影响

**S3 暂缓**，等音色定下来。理由：若最终走云端 TTS，延迟预算与"正在合成"的
状态语义都要跟着调（云端有一次网络往返），现在铺通话页 UI 会白做一部分。
**S1/S2 的产出不受影响**（客户端协议与播放器都不变）。
