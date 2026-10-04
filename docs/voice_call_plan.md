# 「AI 陪伴通话」实施计划

> 目标形态：像豆包打电话那样，**按住一个按钮就能和 AI 连续对话** —— AI 说完用户可以直接插话，
> 中间没有"点发送"的动作，轮次切换接近真人通话的节奏。
>
> 本计划基于两类事实：
> * **[实测]** —— 本次会话在本机真实跑出来的数字（下面凡是标 [实测] 的都可在本文档找到出处）。
> * **[估计]** / **[未验证]** —— 推算或没验证过的，逐条标注，**不得当作事实使用**。
>
> 证据附录（88 KB，逐条标了验证状态）：[`docs/voice_call_research.md`](./voice_call_research.md)
> —— 该文档列了 40 条未验证项，读结论前建议先看它的第 7 节。
>
> **本文档回答"能不能做、走哪条路、有什么风险"。**
> **要动手施工，看 [`docs/call_implementation_steps.md`](./call_implementation_steps.md)**
> —— 那是逐步施工单（S0–S9，每步有可机器验证的闸门）。本文档的 §5 阶段划分在施工单里
> 被细化成具体文件、协议与命令，且其中三条被实测代码推翻/简化了（见施工单 §1 的 F-1…F-5）。

---

## 0. 先看结论：三个需要人拍板的事

| # | 决策点 | 选项 | 影响 |
|---|---|---|---|
| **D1** | 语音走本地还是上云 | **Path A** 本地 ASR+TTS（音频不出设备）／ **Path B** DashScope Omni-Realtime（音频上云） | 首声中位 [估计] 1.6s vs 0.85s；合规性质完全不同 |
| **D2** | 是否新增"药物建议"词法轴 | 补 / 不补 | 补属于**新增安全规则**，按 AGENTS.md 需人工审核。当前缺口是既有事实（见 §4.1） |
| **D3** | 打断的可靠性期望 | 半双工（AI 说话时不能插话，100% 可靠）／ 真打断（有已知失效场景） | 直接影响"像不像电话" |

**在 D1 拍板之前，不要写通话页面的代码。** 两条路的客户端结构不同（Path B 要把整条链路换成
WebSocket Realtime 协议），先做完再改等于重写。

**但在 D1 之前必须先做一件事：TTS 的 RTF benchmark（§5 阶段 0）。**
它决定 Path A 是否成立 —— 如果本机 RTF > 0.5，Path A 直接出局，D1 就没得选了。

---

## 1. 现状：已经有什么（[实测]，可直接复用）

这一步不是从零开始。本次会话已经把"文本侧流式"打通，通话功能可以直接坐在它上面。

### 1.1 已有的四段能力

| 环节 | 位置 | 状态 |
|---|---|---|
| **边说边出字** | `algorithm/asr/streaming.py` + `/api/v1/asr/ws`（WS） | ✅ 可用 |
| **停顿定稿** | 同上，端点检测 → 离线复识别纠错 → 补标点 | ✅ 可用 |
| **文本流式回复** | `algorithm/api/intervention.py` 的 `/smart-chat/stream`（SSE） | ✅ **本次新建** |
| **增量安全闸门** | `algorithm/api/intervention.py` 的 `_StreamingSafetyGate` | ✅ **本次新建** |
| **出声** | —— | ❌ **完全没有**（全项目 grep `OfflineTts\|speechSynthesis\|语音合成` = 0） |

### 1.2 本次实测的延迟数字（后面做预算要用）

文本流式（真实千问，本机 127.0.0.1）：

| 事件 | 冷启动 | 热（第 2 次起） |
|---|---|---|
| `meta`（感知+风险+场景检索完成） | 6311 ms | **75 – 108 ms** |
| 第一个 `delta`（首句可上屏） | 6797 ms | **442 – 497 ms** |
| `done`（全文+审计结论） | 6920 ms | **469 – 613 ms** |

→ 首字 440–500 ms 里，**META 只占 ~80 ms，LLM 首 token 占 ~370–390 ms**。
→ 冷启动 6.3 s 是模型首次加载（感知模型 + 认知扭曲分类器 + 向量检索），**必须预热**。

语音侧（本次会话早前实测）：首个 partial ~770 ms、partials 每 ~330 ms、端点触发 0.75–1.2 s、定稿 51–61 ms。

---

## 2. 目标体验与延迟预算

### 2.1 什么叫"像打电话"

| 指标 | 真人通话 | 对讲机感（要避免） | 本计划目标 |
|---|---|---|---|
| 用户说完 → AI 开口 | 200 – 500 ms | > 1200 ms | **< 900 ms** |
| 能否打断 | 随时 | 不能 | 能（或明确降级为半双工并告知） |
| 是否需要手动操作 | 不需要 | 每轮要点一次 | 不需要 |

### 2.2 Path A —— 本地 ASR + 云端 LLM + 本地 TTS

| # | 环节 | 值 | 来源 |
|---|---|---|---|
| 1 | 麦克风攒块 | 0–100 ms（期望 50） | [实测] `pcm-capture-processor.js:20` 1600 样本 |
| 2 | 句末静音判定（rule2） | 现值 750–1200 ms → **目标 400–500 ms** | [实测] / 待调 |
| 3 | 定稿（复识别+标点） | **51 – 61 ms** | [实测] |
| 4 | 风险判定 + 安全闸门 | **< 5 ms** | [实测]（META 75–108 ms 里已包含感知全流程） |
| 5 | LLM 首 token | **370 – 390 ms** | [实测] |
| 6 | 首句补齐（够起播） | 150 – 350 ms | [估计] |
| 7 | TTS 首块 | 100 – 300 ms | [估计] ⚠️ **完全没测** |
| 8 | 下行 + 起播 | 40 – 80 ms | [估计] |
| | **合计首声** | **中位 ≈ 1.6 s；区间 1.1 – 2.4 s** | [估计] |

**硬约束：TTS 总 RTF 必须 < 1.0。** 否则合成追不上播放，队列越积越长（听起来 AI 越说越慢）。
这是 Path A 的结构性风险，也是 §5 阶段 0 必须先跑 benchmark 的原因。

### 2.3 Path B —— DashScope Omni-Realtime（端到端语音到语音）

**握手已独立复验**（本次会话我亲自跑通，不是转述）：

```
wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-omni-flash-realtime
Authorization: Bearer $DASHSCOPE_API_KEY
OpenAI-Beta: realtime=v1
→ session.created:
  modalities: ["text","audio"]        voice: "Cherry"
  input_audio_format:  "pcm16"        ← 16kHz 单声道，与现有 AudioWorklet 产物完全同构
  output_audio_format: "pcm24"
  input_audio_transcription: {model: "gummy-realtime-v1"}
  turn_detection: {type:"server_vad", threshold:0.5, prefix_padding_ms:300,
                   silence_duration_ms:800, create_response:true, interrupt_response:true}
```

这条路一次性解决了四件事，**都不需要自己实现**：

1. **`input_audio_format: pcm16` 与现有采集完全兼容** —— 前端可以同一路音频同时喂本地 ASR 和
   Omni-Realtime，不用重采样、不用改 `pcm-capture-processor.js`。
2. **`interrupt_response: true` 是默认值** —— 服务端在用户开口时自动取消正在生成的回复。
   这就是服务端实现的打断。
3. **`silence_duration_ms` 合法区间 [200, 6000]** —— 端点可一路压到 200 ms。
4. **`conversation.item.input_audio_transcription.*` 事件会把用户语音转写成文本回推** ——
   **安全审计链路不会因为"端到端一体化"而失明**。这一点很关键：`risk_level` 判定需要文本。

首声中位 [估计] ≈ 0.85 s，且**没有 TTS RTF 约束**（服务端边生成边推流）。

**代价（都不是技术问题）：**
* **未成年人音频出设备。** 仓库里 `AsrEngine` 枚举刻意不提供浏览器 Web Speech API 选项，注释写明
  "敏感原始数据不出设备"。Path B 与这条原则直接冲突。
* **计费**：[未验证]，没查价格也没做过计费调用。
* **需要重写客户端**为 Realtime 协议（`input_audio_buffer.append` / `response.cancel` / 事件分发）。

### 2.4 最大瓶颈与结构性结论

Path A 的 1.6 s 由三段串联构成：**句末静音（0.5s）+ LLM 首 token 与首句（0.85s）+ TTS 首块（0.25s）**。

* 唯一能显著压缩的单项是**句末静音判定**（省 350–600 ms），但它也是风险最高的 —— 压太狠会变成"AI 抢话"。
* **结构性结论：Path A 的 1.6 s 是"三段串联"的下限。要进 1 s 只能走 Path B，继续抠 Path A 的每一毫秒没有意义。**

---

## 3. 轮次切换（Endpointing）

### 3.1 参数调整（只用于通话模式，不改全局默认）

| 参数 | 现值 | 通话模式 | 依据 |
|---|---|---|---|
| `rule1_min_trailing_silence` | 1.6 s | 1.0 s | 还没开口时的静音，通话里用户刚接起沉默很正常 |
| `rule2_min_trailing_silence` | 0.8 s | **0.40 s** | 主力参数，直接省 400 ms |
| `rule3_min_utterance_length` | 15.0 s | 12.0 s | 通话轮次通常更短 |

### 3.2 ⚠️ 只改 rule2 会反噬，必须配两件事

400–600 ms 的"我想想"停顿会被切成句末 → AI 抢话，**比 0.8 s 更伤体验**。

**(1) 投机启动 + 取消窗口（收益最大）**

```
t0    端点触发 → 立刻用【partial 文本】发 LLM（不等离线复识别）
[t0, t0+300ms] 窗口内若又出 partial（用户其实还在说）
              → abort 这次投机请求，拼接后重发
              窗口内无新 partial → 用 t0+60ms 的定稿校验一次，差异大则重发
```

收益 [估计] 300–500 ms。代价：误判时多花一次 LLM 调用（竞赛 Demo 完全可接受）。
**原料已经齐了**：`StreamingSession.accept()` 在同一次调用里就可能同时返回 partial 和 final，
所以"端点触发时手上已经有 partial 文本"天然成立。

**(2) partial 上的语义判据（零成本启发式）**

* partial 以 `因为 / 然后 / 就是 / 而且 / 但是 / 所以` 结尾 → **话没说完** → 延长等待到 ~700 ms
* partial 以 `吗 / 呢 / 吧 / 啊 / 了` 结尾 → 大概率说完了 → 可缩短等待
* partial 很短（< 4 字）且静音 → 可能只是"嗯"，不要急着定稿

> 这套是**交互时序规则，不是安全规则**，不触发 AGENTS.md 的"安全阈值需人工审核"约束。
> 但它会影响风险判定的时机，建议仍让 reviewer 过一眼。

**(3) "用户还在想"检测**

* 免费层：rule2 延长到 0.70 s + 上面 (2) 的连接词判据。
* 低成本层：rule2 窗口内用 Silero VAD 判"是否仍有语音"（呼吸、咂嘴、"呃…"）。⚠️ [未验证] 本机 `algorithm/models/` 下**没有** VAD 模型文件，需要额外下载。
* 托管层（仅 Path B）：`turn_detection.type = "semantic_vad"`（按语义而非声学静音判断句末）+
  `idle_timeout_ms ∈ [5000, 30000]`（用户沉默过久时模型主动推进对话）。[未验证] 两者都只从文档读到。

---

## 4. 安全设计（本计划里最不能省的一节）

### 4.1 已有的抓手：`_StreamingSafetyGate`（本次新建）

它在 `algorithm/api/intervention.py`，docstring 已经把边界写死了，这里直接引用：

> 本闸门**不引入任何新的安全规则或阈值**。它只复用 `config/audit_rules.yaml` 里既有的
> `stigma_rejection` 词表（含 `teen_specific` 扩展词表），把原本在整段审计时才生效的词法红线
> **提前到句子级**。按 AGENTS.md「自动化流程不得自行修改安全阈值或升级条件」，这里改变的是
> 既有规则的生效时机，不是规则本身。

接口：`feed(delta) -> list[str]`（返回可放行的完整句）/ `flush()` / `screen(text)` / `tripped` / `hits` / `terms_count`。
切句 `_SENTENCE_BOUNDARY_RE = r'[。！？!?；;\n]'`，上限 `_STREAM_UNIT_MAX_CHARS = 200`（**语音用太长，见下**）。

**已知缺口（docstring 原文承认，不是我的推测）**：`audit_rules.yaml` **没有药物词法轴**，
而 AGENTS.md 明令禁止药物建议。补这一轴属于**新增安全规则**，需人工审核 → 这就是决策点 **D2**。

### 4.2 高风险硬禁用语音 —— 复用既有判据，零新增阈值

`intervention.py` 里已经有这行（本次为文本流式新增，语音通道直接复用同一个布尔量）：

```python
streaming_allowed = prep.risk_level in ("low", "medium") and not prep.crisis_detected
```

`voice_allowed = streaming_allowed`。效果：

* **没有新增任何阈值或判定条件** → 满足 AGENTS.md「复用既有规则不需要人工审核」。
* high / crisis 时**整条语音通道不启动**，escalation 路径保持既有的、原子的、纯文本的。
* 这一点很重要：**危机场景最怕的就是"AI 已经大声念出不该念的内容，审计才说要改"**。硬禁用从结构上排除了它。

**已实测的行为**（本次会话真实跑过，危机输入 `我不想活了，活着没意思`）：

```
[ 61.2ms] META  streaming=False risk=crisis
[641.8ms] REVISE(整体替换) '你说的这些我很重视。你现在的安全是最重要的。…'
[642.0ms] DONE  requires_escalation=True
无 delta                      ← 一个字都没有增量上屏
```

### 4.3 逐句过闸 → 逐句合成 → 逐句播

```
用户语音 → ASR 定稿 → voice_allowed? ──否──→ 退出语音模式，走既有 escalation（纯文本）
                          │是
                          ▼
          _build_llm_messages → _iter_llm_deltas (SSE)
                          │
                          ▼
          gate.feed(delta) → 完整句 unit（命中红线 → tripped，停止生成）
                          │
                          ▼
                 TTS 工作线程 generate(unit, callback)  ← 边合成边推 PCM
                          │
                          ▼
                    前端播放队列 → 出声
                          │
        同时：整段回复跑既有 _audit_and_finalize 五轴审计
                          │
              审计改写？ → REVISE → 立刻停播 + 清队列 + 用改稿重新合成
```

**两处必须注意的实现细节：**

1. **切句上限要调小，但不要改这个类。** `_STREAM_UNIT_MAX_CHARS = 200` 对语音太长
   （200 字 ≈ 36 秒音频）。推荐**不动安全组件**：在语音通道里用一个独立的切句器（同样的
   `_SENTENCE_BOUNDARY_RE`，上限 20–24 字）先切句，再把每句喂给闸门的**公有** `screen()`。
   零改动、零审核成本。
2. **首句必须缓冲的延迟代价是必须付的。** 闸门要求先凑出一个完整语义单位才能判红线。
   20–24 字上限时 [估计] 200–350 ms。
   **反过来说，这也是为什么首句应该设计成最安全的一句。** 现有 prompt 的标准范式本来就是
   「简短共情接纳 → 一个开放式追问」，**首句天然是共情反映**。
   ⚠️ 若要把它写死进 prompt（"首句必须是纯共情反映，不得包含任何建议、判断、信息"），
   那属于**安全相关的 prompt 变更，需人工审核** —— 本计划不擅自改。

### 4.4 无法消除的残余风险（必须写进 Demo 免责说明）

**已经播出去的声音收不回来。** 审计改稿（REVISE）只能停播 + 重播，不能撤回已出口的内容。
缓解是三层叠加：句子级闸门（词法）+ prompt 硬边界 + 首句纯共情约束。叠加后"已播出且被审计否掉"
的概率很低，**但不是零**。

### 4.5 危机时的具体动作

```
risk_level ∈ {high, crisis}
  → 1. 立即停止播放（Web Audio stop + 清队列 + 中止合成）
  → 2. 会话切纯文字通道，禁用通话按钮
  → 3. 走既有 escalation（AGENTS.md：必须触发）
  → 4. 播一句【固定的、预审核过的】短句（用预合成 PCM 缓存，不经 LLM、不过闸门）
  → 5. UI 展示 crisis_resource_card（audit_rules.yaml 既有内容）
```

* 第 4 步那句文案是**新增的用户可见安全文案**，严格说不触发"规则变更需审核"条款，
  但**建议仍让人过一眼** —— 它出现在危机时刻。⚠️ 本计划**没有拟**这条文案。
* **绝不要用 TTS 朗读热线电话号码。** 读数字容易出错（"400-161-9995"可能被读成"四十万…"），
  危机场景下念错号码后果严重。号码只以**文字卡片**呈现（既有做法）。这是本计划最强烈的一条建议。

---

## 5. 实施阶段（每阶段都有 go / no-go 闸门）

### 阶段 0：TTS 可行性 benchmark ✅ **已执行（结论：Path A 成立）**

> **结果见 [`docs/tts_benchmark.md`](./tts_benchmark.md)。** 一句话：
> RTF 最差 **0.180**（4 线程）/ **0.112**（8 线程）空载、**0.209 / 0.136** 满载，
> 判据 0.4 —— 余量 2–3 倍。同时实测出三件推翻原假设的事：
> ① VITS 经 `OfflineTts` **不流式**（callback 恒 1 次），"首块"就是"整句合成耗时"；
> ② `max_num_sentences=1` 会**静默丢弃第一句之后的所有内容**（必须用 `-1`）；
> ③ VITS 音频时长是**随机的**（同句 3945/5938/6125 ms），RTF 必须重复取中位数。
> 下面保留原始的阶段 0 定义，作为闸门设计的记录。

**目标**：回答"Path A 到底成不成立"。

1. 扩展现有 `algorithm/tools/download_asr_model.py` 的 `PRESETS`（`Preset` dataclass / `download()` /
   `_BROWSER_UA` 全可复用），加一个 `subdir="tts"` 的预设：
   ```python
   "tts-zh-fanchen-C": Preset(
       repo="csukuangfj/vits-zh-hf-fanchen-C",
       required=("vits-zh-hf-fanchen-C.onnx", "lexicon.txt", "tokens.txt"),
       optional=("date.fst", "number.fst", "phone.fst", "new_heteronym.fst"),
       subdir="tts", note="离线中文 TTS —— 陪伴通话的出声端"),
   ```
   （`algorithm/models/` 已在 `.gitignore`。模型体积 [实测来自 hf-mirror 的 Content-Length]：
   onnx 121.3 MB + lexicon 2.4 MB。**但本阶段之前没有真实下载验证过**，见 §7 未验证项 1。）
2. 写一次性 benchmark：5 句典型中文回复（10 / 20 / 40 / 80 字各一），用 `OfflineTts.generate()` 测：
   * **首个 callback 延迟**（首块出声时间）
   * **总耗时**，并算 `RTF = 总耗时 / 音频时长`
   * ⚠️ 必须**同时打开现有全部模态**一起测（CPU-only，`useVoiceAnalysis` 每帧还有 O(n²) 的基频估计）
3. **同时测 `qwen-turbo` 的首 token 延迟**（本计划里 §2.2 的 370–390 ms 只来自 3 次观测，样本太少）。
4. **同时试听中文音质** —— [未验证] 报告里没有任何一条音质结论（没人听过）。

**go 判据**：`RTF < 0.4` **且** 首块 < 250 ms → 用 `vits-zh-hf-fanchen-C`。
**no-go 兜底**：换 `vits-piper-zh_CN-huayan-medium`（63.2 MB，同架构 RTF 快约 4.5 倍）。
⚠️ 但它的 `MODEL_CARD` 写 `License: Unknown`，**用于竞赛 Demo 前应确认授权**。
**两者都不达标** → Path A 出局，D1 只剩 Path B（或退回"首句预合成 + 后续整句"的保守形态）。

### 阶段 1：单句出声闭环（Path A 最小可用）

**目标**：不做轮次切换、不做打断，只把"文本 → 声音"这一段打通并可测量。

* `algorithm/tts/engine.py`：`OfflineTts` 单例，仿 `asr/engine.py` 的 lazy load + 错误传播写法。
* `algorithm/api/tts.py`：`WS /tts/ws`（**二进制帧，不要往 SSE 里塞 base64 音频**）。
  合成必须照抄 `api/asr.py:162` 的 `run_in_threadpool` —— 这是 CPU 密集操作，不能占事件循环。
* **TTS 必须串行**：单工作线程 + 队列。`punctuation.py:161` 已有"CT-Transformer 会话不保证并发安全
  → 加锁串行化"的先例。⚠️ [未验证] `OfflineTts` 是否真的不能并发，没测。
* 前端：`AudioBufferSourceNode` 播放队列（**不要用 `<audio>`**）。
* **闸门接线**：复用 `_StreamingSafetyGate` 的公有 `screen()`，独立切句器上限 20–24 字。

**go 判据**：能连续播 5 句不卡顿，且**任何一句都不经过"未过闸门"的路径**。

### 阶段 2：通话闭环（Path A）

* `client/src/pages/patient/VoiceCall.tsx` + `useVoiceCall` hook。
* **复用 `AnxietyContext` 的 `startVoiceInput` / `stopVoiceInput`**（接口在 :311/:312，实现 :975/:985），
  **不要自己 `getUserMedia`** —— `useVoiceAnalysis.ts:233` 已经在开一路麦克风，再开一路会
  导致 AEC 参考冲突与 AudioContext 泄漏。
* 打断：见 §6。
* 轮次：见 §3。
* 危机切换：见 §4.5。

### 阶段 3（可选）：Path B 端到端

只有满足下面全部条件才做，否则不做：
* 阶段 0 判定 Path A 不达标，**或**现场实测首声 > 1.2 s 被认为"不够电话感"；
* **D1 已明确接受"音频出设备"**（产品/合规决策，不是技术决策）；
* 计费已确认。

---

## 6. 打断与回声消除（诚实的版本）

### 6.1 `echoCancellation: true` 是必要但不充分的

**必做**：`getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }})`。
现状是 `useVoiceAnalysis.ts:233` 的 `getUserMedia({audio:true})` **没有任何约束**，这就是改动点。
把 `MediaStreamTrack.getSettings()` 读回的实际生效值同步到 UI 上，**不要假设它生效了**。

**已知失效模式** [估计，有公开 issue 佐证，但**没有一条是针对"Windows 11 + 本机扬声器 + 本地 TTS"的实测**]：

1. AEC 的远端参考是**默认输出设备**。用外接音箱 / 蓝牙耳机 / HDMI 时参考信号就对不上。
2. **收敛需要 200–500 ms** —— TTS 开头那一两百毫秒会被漏过去，恰恰是"AI 刚开口"那一刻。
3. 大音量导致削波/非线性失真时，线性自适应滤波器在原理上抵不掉。
4. 双讲（用户真插话）时 AEC 可能把人声一起压掉 → **打断会漏**。

### 6.2 推荐的组合：A（短窗半双工）+ AEC + D（文本级自回声过滤）

| 方案 | 评价 |
|---|---|
| A. 半双工（TTS 播放时静音麦克风） | 绝对可靠，但**完全无法打断** —— 只能作为"起始 400ms 窗口"用 |
| B. 声学门控（按能量/相关性） | 需要精确电平标定，说话人距离/系统音量/耳机都会变。**不建议作为主手段** |
| C. `MediaStreamAudioDestinationNode` 自回声相减 | 原理正确，但落地等于自己写 AEC3。**研究项目，不是 Demo 特性** |
| **D. 文本级自回声过滤** | 我们**完全知道自己在说什么**，把 ASR 定稿与"最近在播的 TTS 文本"做模糊匹配，命中即丢弃。便宜、确定、不依赖任何 DSP |

**时序（可直接实现）：**

```
时刻 0        : 开始播放 TTS 第 1 句（Web Audio）
[-∞, 0)       : 麦克风正常流式识别（用户还在说 → 触发"打断上一轮"）
[0, 400ms)    : 半双工窗口 —— 停止向 ASR 送帧（覆盖 AEC 收敛期）
[400ms, ∞)    : 恢复送帧，靠 AEC + 方案 D 过滤
收到 ASR 定稿  : 命中"最近 TTS 文本" → 丢弃；否则判为真实打断 →
                 1) 停播（Web Audio stop + disconnect，[估计] < 50 ms）
                 2) 清空本地 PCM 队列与待合成句队列
                 3) 通知服务端（Path B 发 response.cancel；Path A abort SSE +
                    让合成 callback 返回 1 中止）
                 4) 打断文本送入正常轮次（审计 + LLM）
```

* **打断必须走 partial 而不是 final**：partial 每 ~330 ms 就有 [实测]，final 要等端点 0.45–1.2 s，手感迟钝。
* **打断时永远先清本地队列，再通知服务端**，不要等服务端回执（否则用户会听到"AI 说了半句就没了"）。
* ⚠️ 方案 D 的阈值（"TTS 文本中 ≥6 字连续子串"）是**拍的**。[未验证] 从没实测过"AI 的话被 ASR 识别成什么"。
  **这是 §6 里最需要先做实验的一个数**。

### 6.3 `speechSynthesis` 只配当降级兜底

**否决理由是架构性的，不是延迟性的**：
1. 它的音频**不经过 Web Audio** —— 拿不到 `MediaStream`/`AudioNode`，**做不了 AEC 参考、做不了方案 D**。
2. Edge 上默认走**在线微软语音**（`useVoiceAnalysis.ts:11-13` 的注释已经实测抓到端点）。演示现场网络一抖，AI 就哑了。
3. 与项目自己的"数据不出设备"叙事冲突。

本地 TTS 模型缺失时可以退回它，但**必须如实告诉用户"当前使用在线语音"**。

---

## 7. 未验证 / 有风险的点（**读结论前先读这一节**）

本计划的**所有 `[估计]` 都是估计**。具体而言：

| # | 说法 | 状态 |
|---|---|---|
| 1 | `hf-mirror.com` 能下载 `vits-zh-hf-fanchen-C` 的全部文件 | **[未验证]** 只读到 `Content-Length`（HTTP 200 + 大小），**没有真实下载过任何一个 TTS 模型**。`lexicon.txt`/`tokens.txt` 的 HEAD 返回 chunked 无 `Content-Length`，是异常信号，下载时应校验大小 |
| 2 | fanchen-C 本机 CPU RTF ≈ 0.15–0.25 | **[估计]** 由 Pi4 的 1.600 外推。**这是全篇最重要的未验证数字**，直接决定 Path A 成不成立 |
| 3 | TTS 首 callback 延迟 100–300 ms | **[估计]** sherpa 文档只给总 RTF，不给首块延迟。**完全没有依据可查** |
| 4 | 任何模型的**中文音质** | **[未验证]** 没人听过。fanchen-C 优于 piper-huayan 是架构推断（自带中文词典 vs espeak-ng cmn 音素化），不是听感结论 |
| 5 | piper `zh_CN-huayan-medium` 的许可证 | **[未验证]** `MODEL_CARD` 明确写 `License: Unknown`。**用于竞赛 Demo 前应确认授权** |
| 6 | `matcha-icefall-zh-baker` 完全不可用 | **[未验证]** 已验证其仓库无 vocoder，且 `csukuangfj/vocos-22khz-univ` 等镜像 **401**。但**没有穷举**所有镜像名，也没试 ModelScope。结论是"在我检查过的路径下不可达" |
| 7 | Realtime 端到端首声 ≈ 0.85 s | **[估计]** 完全没测 |
| 8 | `semantic_vad` / `idle_timeout_ms` 的实际效果 | **[未验证]** 只从文档读到存在与适用范围 |
| 9 | Realtime 计费标准 | **[未验证]** 没查到价格，没做过计费调用 |
| 10 | Realtime 端到端音质与打断可靠性 | **[未验证]** **一个音频字节都没发过**，只做到收到 `session.created` |
| 11 | Chromium AEC3 收敛需 200–500 ms | **[估计]** 工程共识，本机没实测 |
| 12 | `AudioBufferSourceNode.stop()` 端到端 < 50 ms | **[估计]** 按渲染量子推算，未实测 |
| 13 | 方案 D 的拦截率 | **[估计]** 阈值是拍的，**没实测过回声被识别成什么文本** |
| 14 | `rule2 = 0.40 s` 的过早截断率 | **[未验证]** 完全没测。**建议先在通话模式做影子测试**（只记录"若按 0.4s 切会切在哪"，不真打断），跑 20–30 轮真人语音再定 |
| 15 | 本机有 Silero VAD 模型文件 | **[未验证]** `algorithm/models/` 下只有 asr/punct 两类 |
| 16 | 投机启动能省 300–500 ms | **[估计]** 未实现、未测 |
| 17 | `OfflineTts` 并发安全性 | **[未验证]** 按 `punctuation.py` 的先例**建议**串行，但没验证是否真的不能并发 |
| 18 | 通话与现有模态的 CPU 争用 | **[未验证]** 没压测。CPU-only 同时跑 16kHz 流式 ASR + VITS 合成 + O(n²) 基频估计 + 可能的摄像头，**是真实风险** |
| 19 | 把既有 `streaming_allowed` 接到语音通道算不算"新增升级条件" | **[估计]** 这是我的解读。**建议做之前问一句** |
| 20 | 危机时那句固定话术的措辞 | **[未验证]** 本计划**没有拟** |
| 21 | 阶段 0 的 qwen-turbo 首 token 370–390 ms | **样本只有 3 次观测**，不足以作为预算依据，阶段 0 要重测 |

> **§2 的两张延迟表里，没有任何一条端到端链路被真正跑通过**：没下载 TTS 模型、没合成过一句话、
> 没让浏览器播过一个字节的 AI 语音、没向 Realtime API 发过音频。
> 表里的"1.6 s / 0.85 s"是**给现场期望管理用的假设值，不是预测值**。

---

## 8. 一页速览

1. **本计划不是从零开始**：本地流式 ASR、离线纠错、标点、**文本流式回复**、
   **增量安全闸门**（本次新建）都已可用且实测过。缺的只有**出声**。
2. **先跑 TTS RTF benchmark（阶段 0），再决定 D1。** 这一步决定 Path A 成不成立，
   不做完不要写通话页面的代码。
3. **Path A 首声中位 [估计] 1.6 s 是"三段串联"的下限**；要进 1 s 只能走 Path B（音频上云，需拍板合规）。
4. **安全上几乎全是复用**：`_StreamingSafetyGate` 逐句过闸 + `streaming_allowed` 硬禁用高风险语音。
   唯一需要人工决策的是**药物词法轴的缺口（D2）**。
5. **打断不要指望浏览器 AEC**：用「400ms 半双工窗口 + AEC + 文本级自回声过滤」，打断走 partial。
   `speechSynthesis` 只能当降级兜底。
6. **危机时不要用 TTS 念热线号码**，只用文字卡片；已播出的声音收不回来，写进免责说明。
