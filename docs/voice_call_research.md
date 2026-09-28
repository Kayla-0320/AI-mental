# AI 陪伴通话（实时语音电话）实现方案调研报告

> 调研范围：为现有中文青少年心理健康平台（AIC 算法大赛 Demo）新增「实时语音电话式陪伴通话」能力。
> 运行环境：Windows 11、**CPU only（无 GPU）**、Python 3.13.14（`D:\Develop\AIC\.venv`，**无 pip**，安装须用 `uv pip install --python D:\Develop\AIC\.venv\Scripts\python.exe`）。
> 本报告**不修改任何项目代码**，只新增本文档。

## 0. 证据标记与本次调研方法

全文所有断言按下列三种标记之一标注：

| 标记 | 含义 |
|---|---|
| **[实测]** | 本次会话中我**亲自执行命令 / 发起请求并拿到结果**验证过。命令或请求写在正文里。 |
| **[估计]** | 基于已实测事实 + 公开数据的推算，未直接测量。给出推算依据。 |
| **[未验证]** | 我**没有**成功核实。可能是没抓到页面、网络被墙、或没有对应能力。**不要把它当成事实使用。** |

本次调研实际做过的动作（可复现）：

1. **[实测]** `uv pip list --python D:\Develop\AIC\.venv\Scripts\python.exe` 列出已装包。
2. **[实测]** 用 venv 的 python 内省 `sherpa_onnx` 的 TTS 类与 `OfflineTts.generate` 的 pybind docstring。
3. **[实测]** 用 `hf-mirror.com/api/models/<repo>` 列举候选 TTS 仓库的文件清单；用 `hf-mirror.com/<repo>/resolve/main/<file>`（浏览器 UA）读 `Content-Length` 得到**真实文件大小**。UA 与 `algorithm/tools/download_asr_model.py` 中 `_BROWSER_UA` 一致。
4. **[实测]** 用 `D:\...\.env` 里的 `DASHSCOPE_API_KEY` 调 `GET https://dashscope.aliyuncs.com/compatible-mode/v1/models`，拿到账号**实际可用模型 id 列表**。
5. **[实测]** 用 venv 的 `websockets` 向 `wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-omni-flash-realtime` 发起握手，**成功**收到 `session.created`。
6. **[实测]** 抓取阿里云百炼 client-events / server-events 文档并正则抽取事件名。
7. 阅读仓库源码：`client/public/worklets/pcm-capture-processor.js`、`client/src/services/streamingAsr.ts`、`client/src/hooks/useVoiceAnalysis.ts`、`client/src/context/AnxietyContext.tsx`、`algorithm/api/asr.py`、`algorithm/asr/{streaming,engine,punctuation}.py`、`algorithm/api/intervention.py`、`algorithm/config/audit_rules.yaml`、`algorithm/tools/download_asr_model.py`、`algorithm/shared/dataclasses.py`。
8. **未做**：没有安装任何包、没有下载任何模型、没有修改任何项目文件（除本文档）。

引用到的外部来源（均以 markdown 链接给出）：sherpa 官方文档（vits / matcha / piper / rtf 页）、hf-mirror API、阿里云百炼文档、Chromium/WebKit issue 追踪器。

---

## 1. TTS 选型

### 1.1 先确认三件会影响所有选型的事（全部 [实测]）

**(a) `sherpa-onnx==1.13.8` 里没有 `OnlineTts`。**

```python
import sherpa_onnx
print([n for n in dir(sherpa_onnx) if 'Tts' in n or 'TTS' in n])
# ['OfflineTts', 'OfflineTtsConfig', 'OfflineTtsKittenModelConfig', 'OfflineTtsKokoroModelConfig',
#  'OfflineTtsMatchaModelConfig', 'OfflineTtsModelConfig', 'OfflineTtsPocketModelConfig',
#  'OfflineTtsSupertonicModelConfig', 'OfflineTtsVitsModelConfig', 'OfflineTtsZipvoiceModelConfig']
print([n for n in dir(sherpa_onnx) if 'Online' in n])
# ['OnlinePunctuation', 'OnlinePunctuationConfig', 'OnlinePunctuationModelConfig',
#  'OnlineRecognizer', 'OnlineSpeechDenoiser', 'OnlineSpeechDenoiserConfig', 'OnlineStream']
```

→ 「流式 TTS」这一类在 Python 侧**不存在**。网上若看到 `sherpa_onnx.OnlineTts` 的说法，对本版本不成立。

**(b) 但 `OfflineTts.generate()` 有 callback 重载 —— 这是"假流式"的唯一抓手。** [实测，来自已安装 wheel 的 docstring]

```
generate(self, text: str, sid = 0, speed = 1.0,
         callback: Callable[[np.ndarray, float], int] = None) -> GeneratedAudio

callback:
  If not None, it is called during speech generation with
  ``(samples: np.ndarray, progress: float) -> int``.
  Return a non-zero value to stop generation early.
```

`OfflineTts` 实例上还暴露 `num_speakers`、`sample_rate` 两个属性。`OfflineTtsConfig` 可配置字段为：`model`、`rule_fsts`、`rule_fars`、`max_num_sentences`、`silence_scale`、`validate`。`OfflineTtsVitsModelConfig` 字段为：`model`、`lexicon`、`tokens`、`data_dir`、`dict_dir`、`length_scale`、`noise_scale`、`noise_scale_w`、`validate`。 [实测]

→ **结论**：可以在**句内**通过 callback 边合成边把 PCM 分块 WebSocket 推给前端，不必等整句合成完。这是"低延迟冒充流式"的落地方式（见 1.4）。

**(c) 网络可达性决定了候选名单。** [实测] GitHub release 资产不可达，`huggingface.co` 不可达，`hf-mirror.com` 可达且对 LFS 大文件放行（非浏览器 UA 对部分小文件 403）。sherpa 的 TTS 模型**同时**发布在 GitHub release（`tts-models`/`vocoder-models` tag）和 HuggingFace（`csukuangfj/*`）——**只有后者可达**。

我逐个仓库验证的结果（`https://hf-mirror.com/api/models/<repo>` 返回 200 = 可达）：

| 仓库 | api 状态 | 说明 |
|---|---|---|
| `csukuangfj/vits-zh-hf-fanchen-C` | **200 可达** | 9 个顶层文件 |
| `csukuangfj/sherpa-onnx-vits-zh-ll` | **200 可达** | 10 个顶层文件 |
| `csukuangfj/vits-piper-zh_CN-huayan-medium` | **200 可达** | 362 文件（含 355 个 `espeak-ng-data/*`） |
| `csukuangfj/vits-piper-zh_CN-huayan-x_low` | **200 可达** | 仅 7 文件，**无 espeak-ng-data** |
| `csukuangfj/vits-zh-aishell3` | **200 可达** | 含 fp32 与 **int8** 两个 onnx |
| `csukuangfj/matcha-icefall-zh-baker` | **200 可达** | 仅 8 顶层文件，**无 vocoder** |
| `csukuangfj/kokoro-multi-lang-v1_0` / `v1_1` / `kokoro-int8-multi-lang-v1_1` | **200 可达** | 378 文件，含 `voices.bin` |
| `csukuangfj/vits-melo-tts-zh_en` | **200 可达** | 含 `model.int8.onnx` |
| `csukuangfj/vits-icefall-zh-aishell3` | **401 不可达** | 该仓库名不存在/不匿名开放 |
| `csukuangfj/vocos-22khz-univ` | **401 不可达** | vocoder，**这条决定了 matcha 能不能用** |
| `csukuangfj/hifigan_v2`、`csukuangfj/sherpa-onnx-vocoder-hifigan` | **401 不可达** | 同上 |

> ⚠️ 注意：`csukuangfj/vits-icefall-zh-aishell3`（sherpa 官方文档里的名字）**不是** `csukuangfj/vits-zh-aishell3`。前者 401，后者 200。文档名 ≠ 仓库名。

### 1.2 候选对比表

**模型体积全部是我用 `resolve/main` 读 `Content-Length` 得到的真实字节数 [实测]**；RTF 一列是 sherpa 官方文档在 **Raspberry Pi 4 Model B Rev 1.5** 上的实测值 [实测-引用]，用于**横向排序**，不是本机数值。

| # | 候选 | 语种 / 音色数 | 采样率 | 磁盘占用 [实测] | hf-mirror 可达 | 需外置 vocoder | 官方 RTF@Pi4 (1/4 线程) | 本机 CPU RTF [估计] | Python 接入方式 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **`vits-zh-hf-fanchen-C`** | 中 / 187 人 | 16000 Hz | onnx **121.3 MB** + lexicon.txt 2.4 MB + tokens.txt | ✅ | 否 | 4.306 / **1.600** | **0.15 ~ 0.25** | `OfflineTtsVitsModelConfig(model=…onnx, lexicon=…, tokens=…)` |
| 2 | `sherpa-onnx-vits-zh-ll` | 中 / 5 人 | 16000 Hz | onnx **121.1 MB** + lexicon.txt 368 KB + tokens.txt | ✅ | 否 | 4.275 / **1.593** | 0.15 ~ 0.25 | 同上 |
| 3 | **`vits-piper-zh_CN-huayan-medium`** | 中 / 1 女 | **22050 Hz**（MODEL_CARD 实测） | onnx **63.2 MB** + tokens.txt + **仓库自带 espeak-ng-data（355 文件）** | ✅ | 否 | 官方未列；同架构 `en_US-lessac-medium` 为 0.774 / **0.357** | **0.03 ~ 0.08** | `OfflineTtsVitsModelConfig(model=…, tokens=…, data_dir=<espeak-ng-data>)` |
| 4 | `vits-zh-aishell3` | 中 / 175 人 | **[未验证]**（官方 RTF 表里那个 30 MB 版写的是 8000 Hz，但这是另一个产物） | fp32 **121.4 MB** / **int8 39.9 MB** + lexicon + tokens | ✅ | 否 | 官方表 aishell3 0.365 / **0.156**（30 MB 版） | int8 估计 **< 0.05** | 同 1 |
| 5 | `matcha-icefall-zh-baker` | 中 / 1 女 | 22050 Hz | 声学模型 **75.6 MB** ✅ | ⚠️ 模型可达，**vocoder 不可达** | **是** —— `vocos-22khz-univ.onnx`(51 MB) 或 `hifigan_v1/v2/v3.onnx`，官方走 GitHub release `vocoder-models`，本网络**被墙**；hf-mirror 上同名仓库 **401** | 0.892 / **0.391** | — | 需要 `OfflineTtsMatchaModelConfig(acoustic_model, vocoder, …)`，**vocoder 拿不到 → 本条不可用** |
| 6 | `kokoro-multi-lang-v1_1` / `v1_0` | 中英 / 103(54) 人 | 24000 Hz | fp32 **325.6 MB** / `kokoro-int8-multi-lang-v1_1` int8 **114.3 MB** + `voices.bin` **28.2 MB** + `lexicon-zh.txt` 2.4 MB + espeak-ng-data | ✅ | 否 | v1_1 7.635 / **3.191** | 0.3 ~ 0.6 | `OfflineTtsKokoroModelConfig(model, voices, tokens, data_dir, lexicon, dict_dir, lang)` |
| 7 | `vits-melo-tts-zh_en` | 中英 / 1 人 | 44100 Hz | fp32 163 MB / **int8 53.5 MB** | ✅ | 否 | 6.727 / **2.518**(fp32) | int8 估计 0.4 ~ 0.8 | 同 1 |
| 8 | 浏览器 `speechSynthesis` | 取决于系统语音 | 取决于语音 | 0 | 不适用 | 否 | — | — | Web Speech API，**无法接入 WebAudio** |
| 9 | DashScope 云端 TTS | 中 / 多音色 | 云端 | 0 | 需外网（DashScope 可达） | 否 | — | — | HTTP / WebSocket，见 1.5 |

**候选 5（matcha）为什么被排除 [实测]**：`matcha-icefall-zh-baker` 的 HF 仓库顶层只有
`.gitattributes, README.md, date.fst, lexicon.txt, model-steps-3.onnx, number.fst, phone.fst, tokens.txt`
—— 没有任何 `.onnx` vocoder。官方 [Matcha 文档](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html) 明确写「**Remember to also download the vocoder model**」，指向 `https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx`，而 **GitHub release 资产在本环境不可达**，hf-mirror 上也没有对应仓库（401）。因此 matcha 虽然 RTF 最好（0.391，约为 fanchen-C 的 1/4），**在本网络下无法落地**。若将来能拿到 vocoder（例如人工拷贝），它是延迟最优的中文选择。

**候选 3（piper huayan）的两个诚实警告**：
- **[实测]** `MODEL_CARD` 原文：「*License: Unknown*」「Dataset URL: https://github.com/PlayVoice/HuaYan_TTS」，「Finetuned from U.S. English lessac voice (medium quality)」。**许可证未知**，用于竞赛 Demo 前应确认。
- **[实测]** `zh_CN-huayan-medium.onnx.json` 显示 `language.code=zh_CN`、`espeak.voice=cmn`、`num_speakers=1`、`audio.sample_rate=22050`。中文走 **espeak-ng 的 cmn 音素化**，多音字表现通常不如自带 `lexicon.txt` 的中文 VITS（fanchen-C / zh-ll 用中文词典+heteronym FST）。**[未验证]**：我没有实际听过任何一个模型的音质，这一条是架构层面的推断。

**候选 4（aishell3 int8）** 的价值在于 int8 只有 **39.9 MB**、官方同族 RTF 仅 0.156（Pi4）—— **延迟最优且体积最小的中文选择**。风险是音质（若确为 8 kHz 则明显闷）与音色库的授权来源（`jackyqs/vits-aishell3-175-chinese`）。**[未验证]** 我未能确认该仓库产物的采样率。

### 1.3 单一推荐

> **推荐：`sherpa-onnx` `OfflineTts` + VITS `vits-zh-hf-fanchen-C` 作为主模型；把 `vits-piper-zh_CN-huayan-medium` 作为「实测 RTF 不达标时」的确定性兜底。**

理由（逐条对应上表）：

1. **零新增依赖。** `sherpa_onnx` 已装（1.13.8），`OfflineTts` / `OfflineTtsVitsModelConfig` / `OfflineTtsConfig` 全都在。不引入 torch 推理、不引入第二个推理框架。
2. **中文质量最稳。** 16 kHz、187 个音色、带 `lexicon.txt`（2.4 MB）+ `date.fst`/`number.fst`/`phone.fst`/`new_heteronym.fst` 规则 FST，多音字与数字归一化都有官方路径（`rule_fsts`）。这一点对心理健康陪伴场景（语气自然度）比省 60 MB 重要。
3. **模型自包含。** 只需 3 个文件（`vits-zh-hf-fanchen-C.onnx`、`lexicon.txt`、`tokens.txt`），`OfflineTtsVitsModelConfig` 三个参数即可；不需要 espeak-ng-data、不需要外置 vocoder。
4. **可达性已验证。** 三个文件我都用浏览器 UA 从 `hf-mirror.com` 成功读到 `Content-Length` [实测]。可以直接扩展现有 `algorithm/tools/download_asr_model.py` 的 `PRESETS`（它的 `Preset` dataclass / `download()` / `_BROWSER_UA` 全部可复用，见 §6）。
5. **187 个音色 = 可挑选一个「陪伴者」人声并固定 `sid`。** 对 Demo 的叙事（统一的 AI 陪伴者人格）有用。
6. **RTF 风险可控。** Pi4 4 线程 RTF 1.600；本机是 x86 桌面 CPU，单核性能约为 A72 的 6~10 倍，4 线程估计 RTF **0.15~0.25**（即 3 秒语音约 450~750 ms 合成，且 callback 可更早出首块）。**这是本报告最需要实测确认的一个数字。**

**兜底方案（必须准备）**：`vits-piper-zh_CN-huayan-medium`，63.2 MB、仓库自带 espeak-ng-data、同架构 piper medium 在 Pi4 上 RTF 0.357（比 fanchen-C 快 4.5 倍），本机估计 **0.03~0.08**。若实测 fanchen-C 的 RTF > 0.5，直接切 piper；代价是音质与许可证不确定性。

**建议的落地顺序（Demo 就绪判据）**：
1. 先用 `download_asr_model.py` 加一个 `tts-zh-fanchen-C` 预设把模型拉下来（**本报告未下载**）。
2. 写一个一次性 benchmark：取 5 句典型中文回复（10/20/40/80 字各一），测 `OfflineTts.generate()` 的**首 callback 延迟**与**总耗时**，算出 `RTF = 总耗时 / 音频时长` 和 `首块延迟`。
3. `RTF < 0.4 且首块 < 250 ms` → 用 fanchen-C；否则切 piper-huayan-medium 重跑。
4. 若两者都不达标 → 考虑 aishell3 int8（音质换延迟），或退回「首句预合成 + 后续整句」的保守形态。

### 1.4 没有 `OnlineTts`，如何伪造低延迟

三层手段，从"几乎免费"到"工程量大"：

**(1) 句子级流水线（必做，收益最大）**
LLM 是 SSE 流式的（仓库已有 `_iter_llm_deltas` + `_StreamingSafetyGate`，见 §6）。不要等整段回复生成完再合成，而是：

```
LLM delta ──► _StreamingSafetyGate.feed(delta)  ──► 完整句 unit（已过安全词法筛查）
                                                        │
                                                        ▼
                                             TTS 工作线程（单线程 + 锁，串行）
                                                        │  OfflineTts.generate(unit, callback=emit)
                                                        ▼
                                        WebSocket 二进制帧（PCM16） ──► 浏览器 AudioWorklet 播放队列
```

- **首句必须短**：给语音通道一个更小的切分上限（见 §5），让第一句在 ~200–350 ms 内产出并起播；LLM 继续生成第 2 句时，第 1 句正在播。
- **TTS 必须串行**：`PunctuationRestorer` 已经示范了「CT-Transformer 会话不保证并发安全 → 加锁串行化」的做法（`punctuation.py:161`）。`OfflineTts` 同理，用单工作线程 + 队列，不要在 async 事件循环里直接调（`api/asr.py:15-16` 已经写明"识别是 CPU 密集操作，交给 FastAPI 的线程池"）。

**(2) callback 分块出声（必做，收益中等）**
`generate(text, sid, speed, callback)` 的 callback 每次拿到 `(samples: np.ndarray, progress: float)`。在 callback 里立刻把 `samples` 转 int16 推给前端，而不是等函数返回 `GeneratedAudio`。这就是「同一句内也提前出声」。callback 返回非 0 可**立刻中止合成** —— 这正好也是 barge-in 需要的（见 §2）。

**(3) 语音包的预缓冲 / 固定话术缓存（可选）**
把高频开场白（「我在，慢慢说」「嗯，我听着」）在启动时**预合成**成 PCM 缓存。首句用缓存零延迟出声，同时后台合成本轮真实回复。这能把"首声"压到近乎网络往返。**[估计]** 收益 300–600 ms，代价是话术听起来略模板化 —— 与仓库 prompt 里明确要求的"不要使用模板化语言"（`intervention.py:737`）有张力，**建议只用于「嗯/我在」这类反馈音**，不用于实质内容。

**(4) 不建议做的事**：把 `OfflineTtsConfig.max_num_sentences` 调大来"一次合成多句"。那会增加首块延迟，方向相反。

### 1.5 DashScope 侧：能否用一条流把 ASR+LLM+TTS 合成？—— **能，而且我已经握手成功**

这是本次调研**最有价值、也最需要你拍板**的发现。

**(a) 账号实际可用的相关模型 id [实测]**（`GET https://dashscope.aliyuncs.com/compatible-mode/v1/models`，用 `.env` 里的 key，HTTP 200）：

```
qwen3-omni-flash-realtime
qwen3-omni-flash-realtime-2025-09-15 / -2025-12-01
qwen3.5-omni-flash-realtime / qwen3.5-omni-flash-realtime-2026-03-15
qwen3.5-omni-plus-realtime  / qwen3.5-omni-plus-realtime-2026-03-15
qwen3.8-omni-flash-realtime
qwen3-s2s-flash-realtime-2025-09-22          ← speech-to-speech
qwen3-asr-flash-realtime / qwen3-asr-flash-realtime-2026-02-10
qwen3-tts-flash / qwen3-tts-flash-realtime / qwen3-tts-instruct-flash
qwen3-tts-instruct-flash-realtime / qwen3-tts-vc-realtime / qwen3-tts-vd-realtime
qwen-omni-turbo                               ← 仍可用，但官方标注「No longer updated」
qwen3-omni-flash / qwen3.8-omni-flash         ← 非实时，HTTP，可「音频进 / 音频出」
```

**关键否定结论 [实测]**：**`qwen-omni-turbo-realtime` 这个 id 在我的账号模型列表里不存在。** 阿里云文档（client-events 页）里确实提到 `qwen-omni-turbo-realtime` 的默认音色/温度，说明该模型曾经存在，但**用你现有的 key 现在拿不到**。所以「用 `qwen-omni-turbo` 走 realtime API」这条路**走不通**，必须换用 `qwen3*-omni-*realtime` 系列。

**(b) WebSocket 端点与协议 [实测]**：

```
wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-omni-flash-realtime
Headers: Authorization: Bearer $DASHSCOPE_API_KEY
         OpenAI-Beta: realtime=v1
```

握手成功，服务端立刻下发（原文）：

```json
{"event_id":"event_IgXcLThQyGANsbVXnShQD","type":"session.created",
 "session":{"object":"realtime.session","model":"qwen3-omni-flash-realtime",
  "modalities":["text","audio"],"voice":"Cherry",
  "input_audio_format":"pcm16","output_audio_format":"pcm24",
  "input_audio_transcription":{"model":"gummy-realtime-v1"},
  "turn_detection":{"type":"server_vad","threshold":0.5,"prefix_padding_ms":300,
                    "silence_duration_ms":800,"create_response":true,"interrupt_response":true},
  "id":"sess_GtJH2sHdzo9YYTjqZ6LJi"}}
```

**这解决了好几件事，全部不需要自己实现**：

- `input_audio_format: pcm16` = **16 kHz 单声道 PCM16**，与现有 AudioWorklet 的产物**完全同构**（`streamingAsr.ts` 的 `floatToInt16()` 输出就是 16 kHz mono int16）。→ 前端可以**同时**喂本地 ASR 和 Omni-Realtime，不用重采样。
- `output_audio_format: pcm24` = 24 kHz 输出。浏览器侧用 `AudioContext({sampleRate: 24000})` 或直接交给默认 `AudioContext` 重采样即可。
- **`interrupt_response: true` 是默认值** → 服务端在用户开口时会**自动取消正在生成的回复**。这就是服务端实现的 barge-in，客户端再做 `response.cancel` 就是双保险。
- **`turn_detection.silence_duration_ms = 800` 可改，合法区间 [200, 6000]** → 端点可以一路压到 200 ms（见 §3）。
- **`turn_detection.type` 可选 `semantic_vad`** —— 「按语义有效性判断句末，过滤掉附和声与背景噪声」，仅 `qwen3.8-omni-flash-realtime` 与 `qwen3.5-omni-realtime` 系列支持。**这是本报告里唯一一个"真正解决过早截断"的方案。**
- **`idle_timeout_ms` ∈ [5000, 30000]**（仅 `qwen3.5-omni-plus/flash-realtime`，`server_vad` 模式）：官方描述「*After the server finishes audio playback and the user remains silent beyond this duration… the model proactively generates a response to prompt the user to continue*」。**这就是"用户还在想"检测的原生实现**，不需要自己写。

**(c) 事件名 [实测]**（抓 `server-events` / `client-events` 文档正则抽取）：

- 客户端 → 服务端：`session.update`、`input_audio_buffer.append`（base64 音频）、`input_audio_buffer.commit`、`input_audio_buffer.clear`、`response.create`、`response.cancel`、`conversation.item.create`
- 服务端 → 客户端：`session.created`、`conversation.item.created`、`response.created`、`response.audio.delta`、`response.audio.done`、`response.audio_transcript.delta`、`response.audio_transcript.done`、`response.text.delta`、`response.text.done`、`response.content_part.added`、`response.content_part.done`、`response.output_item.added`、`response.output_item.done`、`response.done`、`conversation.item.input_audio_transcription.delta`、`conversation.item.input_audio_transcription.completed`、`input_audio_buffer.committed`、`input_audio_buffer.cleared`

→ **`conversation.item.input_audio_transcription.delta/completed` 意味着服务端会回推用户语音的转写文本**。这非常关键：**安全审计与 `risk_level` 判定需要文本**，而这个事件正好把文本白送给审计链路。也就是说 Omni-Realtime 路线**不会**因为"端到端一体化"而失去安全审计的输入。

**(d) 非实时 HTTP 路线也支持音频出、音频入（备选）** [实测-文档]：`compatible-mode/v1/chat/completions` + `"modalities":["text","audio"]` + `"audio":{"voice":"Tina","format":"wav"}`，`stream=True` 时 `choices[0].delta.audio.data` 是 base64 音频分片。官方示例模型是 `qwen3.5-omni-plus`。这条路的优势是**复用现有 `_call_llm` / `_iter_llm_deltas` 的 HTTP+SSE 基础设施**，不用新写 WebSocket 客户端；劣势是**没有服务端 VAD、没有 `interrupt_response`、没有 `semantic_vad`** —— 打断和轮次切换全部要自己实现。

**(e) 必须写明的重大风险（不是技术问题）**：
1. **数据出境/出设备**。仓库里反复强调「原始音频不出设备」（`client/src/hooks/useVoiceAnalysis.ts:17-20`、`algorithm/asr/engine.py:6-14`、`shared/dataclasses.py:292-296`），`AsrEngine` 枚举甚至**刻意不提供**浏览器 Web Speech API 选项。Omni-Realtime 会把未成年人音频送到阿里云。这是一个**产品/合规决策，不是技术决策**，必须由人拍板，我不能替你决定。
2. **计费**。语音输入输出按 token 计费（音频 token 折算），比 `qwen-turbo` 纯文本贵得多。**[未验证]** 我没有查到本账号的具体价格，也没有做任何计费调用。
3. **`qwen-omni-turbo` 系列已停止更新** [实测-文档]：官方原文「*This series is no longer updated. For text analysis, migrate to Qwen3.8-Omni-Flash; for audio output, use Qwen3.5-Omni.*」→ 不要在新功能里依赖它。
4. **[未验证]** 我没有测过端到端首声延迟、没有测过 WebSocket 长稳、没有测过并发。所有 §4 中 Path B 的数字都是估计。

### 1.6 一句话结论

**主选：本地 `sherpa-onnx` `OfflineTts` + VITS `vits-zh-hf-fanchen-C`（121.3 MB，hf-mirror 可达，句级流水线 + callback 分块伪造流式）；兜底 `vits-piper-zh_CN-huayan-medium`（63.2 MB，RTF 快 ~4.5 倍，但许可证未知、espeak 中文音素化）。**
**并行评估：DashScope `qwen3-omni-flash-realtime`（已实测可握手），它能把 ASR+LLM+TTS+打断+端点全部折叠成一条流，但代价是音频出设备 + 计费 + 需要重写客户端为 WebSocket Realtime 协议。**
**排除：matcha（vocoder 不可达）、kokoro（体积 325 MB + RTF 最差）、melo（RTF 最差）、浏览器 `speechSynthesis`（云依赖 + 无法接入 WebAudio）。**

---

## 2. 打断与回声消除

### 2.1 问题的真实形状

AI 从扬声器出声 → 麦克风收回来 → ASR 把 AI 自己的话转成文字 → 被当成"用户说话" → 新的轮次。这个环一旦闭合，通话会在几句话内彻底崩坏（自问自答、无限循环）。

**注意现有架构的一个好消息**：`useVoiceAnalysis.ts` 用的是 `new AudioContext()`（默认采样率）做声学分析，而 `StreamingAsrSession` 用 `new AudioContext({sampleRate: 16000})` 做采集 —— **两路独立**。这意味着加回声抑制/静音逻辑不需要动声学分析那条路。坏事是 `getUserMedia({audio: true})`（`useVoiceAnalysis.ts:233`）**没有传任何约束**，AEC 用的是浏览器默认值而不是我们显式要的值。

### 2.2 浏览器侧 AEC（`echoCancellation: true`）到底靠不靠谱

**能做的**：
- `navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } })` 会让 Chromium 走 WebRTC 的 AEC3。这是**零成本、必做**的第一道防线，它在"笔记本内置扬声器 + 内置麦克风 + 中等音量"这个最常见配置下**通常**能把回声压到 ASR 不产生有效文本的程度。
- 采集到的实际生效值可以从 `MediaStreamTrack.getSettings()` 里读回（`echoCancellation`、`noiseSuppression`、`autoGainControl`），**应该把这个值同步到"通话质量指示"上**，而不是假设它生效了。

**已知失效模式（这些是工程共识，我标注为 [估计]；有公开 issue 佐证）**：
1. **AEC 需要一个"远端参考信号"**。Chromium 用**默认输出设备**作为参考。如果播放走 `speechSynthesis`（不经过 Web Audio，见 2.4），或者输出设备与麦克风的物理路径不匹配（外接音箱、蓝牙耳机、HDMI 显示器出声），参考信号就对不上，AEC 基本失效。
2. **收敛需要时间**。AEC3 在播放开始后需要**约 200–500 ms** 才能收敛。**TTS 开头的那一两百毫秒会被漏过去** —— 恰恰是"AI 刚开口"这一刻。
3. **非线性失真无法线性抵消**。扬声器音量开大导致削波/失真时，线性自适应滤波器在原理上就抵不掉，残余回声反而增大。
4. **双讲（double-talk）时 AEC 会误伤近端**。用户真的插话时，AEC 可能把人声一起压掉 → **打断检测会漏**。
5. **平台差异与不确定性**。Chromium 追踪器里有「*echoCancellation='all' on macOS is unpredictable*」这类 issue；WebKit 也有「`echoCancellation:false` 仍会在每次采集启动时打断麦克风设备」的 bug。跨平台不能假设行为一致。

**结论 [估计]**：`echoCancellation: true` 是**必要但不充分**的。**绝不能把打断与轮次判定单独托付给它。**

### 2.3 播放侧护栏：三种方案的评价

| 方案 | 做法 | 评价 |
|---|---|---|
| **A. 半双工（TTS 播放时静音麦克风）** | AI 说话期间不发音频给 ASR | **绝对可靠、零风险**，但**完全无法打断** —— 用户必须等 AI 说完。对"电话感"是致命的，因为真人通话里打断是核心体验。 |
| **B. 声学门控（按能量/相关性判断）** | 只在麦克风能量显著高于"我们正在播的电平"时才当作语音 | **需要精确的电平对齐与设备标定**，实测很容易调不出稳定阈值：说话人距离、系统音量、耳机/外放都会改变标定。**不建议作为主手段。** |
| **C. `MediaStreamAudioDestinationNode` + 自回声相减** | 把要播的音频引一份到 Web Audio，从麦克风信号里减掉 | **原理上正确（就是 AEC 本身），但落地等于自己写一个 NLMS/AEC3**。要用 `AudioWorklet` 做自适应滤波、时延估计、双讲检测，还要处理非线性残余。**这是一个研究项目，不是一个 Demo 特性。明确不建议。** |
| **D. 文本级自回声过滤（推荐）** | 我们**完全知道自己在说什么**（TTS 的输入文本由我们自己给定），所以把 ASR 定稿文本与"最近 N 秒正在播的 TTS 文本"做归一化模糊匹配，命中即丢弃 | **便宜、确定、可解释、不依赖任何 DSP**。AEC 失效时仍然有效。缺点：只对"AI 说的话被完整识别出来"有效，对**被截断的半句**会漏（相似度过不了阈值）→ 用较低阈值 + 只比较"是否包含 TTS 文本中长度 ≥ 6 的连续子串"来补。 |

**推荐组合：A（开头短窗）+ AEC + D。**

具体时序（可直接实现）：

```
时刻 0        : 开始播放 TTS 第 1 句（Web Audio）
[-∞, 0)       : 麦克风正常流式识别（用户还在说 → 这里触发的是"打断上一轮"）
[0, 400ms)    : 半双工窗口 —— 停止向 ASR 发送音频帧（覆盖 AEC 未收敛期）
[400ms, ∞)    : 恢复送帧；此时靠 AEC + D 过滤
收到 ASR 定稿  : 若 (与最近 TTS 文本模糊匹配 命中) → 丢弃，不进入轮次
                否则 → 判定为真实打断：
                       1) 立刻停止播放（Web Audio：见 2.5，< 50ms）
                       2) 清空本地 TTS 音频队列与待合成句子队列
                       3) 服务端：若是 Omni-Realtime → 发 response.cancel；
                          若是自建链路 → abort 当前 SSE / 丢弃未播句子，
                          并对 TTS 工作线程中正在跑的 generate 让 callback 返回 1 中止
                       4) 把打断文本送入正常轮次（审计 + LLM）
```

**为什么 400 ms 而不是更长**：400 ms 的半双工损失，用户感受不到（AI 刚开口时用户通常不会在同一瞬间插话）；而 400 ms 正好覆盖 AEC 收敛期。

### 2.4 `speechSynthesis` 能不能秒停？—— 能 `cancel()`，但它是坏选择

- `speechSynthesis.cancel()` 是标准 API，调用后队列立即清空、当前朗读中止。`pause()` / `resume()` 也在。**[未验证]**：Windows Edge/Chrome 上 `cancel()` 之后"已经在音频硬件缓冲区里的那几十毫秒"是否还会响完、以及 `cancel()` 紧接 `speak()` 的竞态，我没有实测。已知 Chromium 社区报告过 `pause()` 状态下 `cancel()` 会让队列卡死的问题。
- **真正的否决理由是架构性的，不是延迟性的**：
  1. **`speechSynthesis` 的音频不经过 Web Audio**。你拿不到 `MediaStream` / `AudioNode`，所以**没法把它登记为 AEC 的参考信号、没法做回声门控、没法混合/闪避**。它在 §2.3 的方案 D 里是个黑洞。
  2. **Edge 上默认走在线微软语音**（`client/src/hooks/useVoiceAnalysis.ts:11-13` 的注释已经实测抓到端点 `wss://speech.platform.bing.com/speech/recognition/...`）。首次出声要等网络往返，且每次都可能抖动；本环境 Google 全封、微软可达，意味着**演示现场网络一抖，AI 就哑了**。
  3. **与项目自己的隐私原则冲突**（"原始音频/文本不出设备"的叙事在 Demo 里是卖点）。
- **结论**：`speechSynthesis` 只适合**作为降级兜底**（本地 TTS 模型缺失时的"能出声"通道），并且要如实告诉用户"当前使用在线语音"。

### 2.5 播放中的音频能不能在词中间停掉？—— 能，而且很快

用 **Web Audio** 播放（不要用 `<audio>` 元素）：

```js
// 播放侧：Web Audio
const src = ctx.createBufferSource();
src.buffer = pcmToAudioBuffer(ctx, int16Chunk, sampleRate);
src.connect(gain); gain.connect(ctx.destination);
src.start();
// 打断：立即
src.stop();            // 或 src.onended = null; src.disconnect();
```

- `AudioBufferSourceNode.stop()` + `disconnect()` 是**立即生效**的（下一个音频渲染量子，按 128 帧/量子、48 kHz 算约 **2.7 ms**；即使加上主线程调度，**[估计]** 端到端 < 50 ms）。这是**远优于 `speechSynthesis.cancel()`** 的地方 —— 后者是跨进程的、行为不保证的。
- **不要用 `<audio>` 元素**：`element.pause()` 也要经过媒体管线，且 pipeline 更难做队列与无缝衔接。
- **推荐结构**：前端维持一个 **PCM 播放队列**（`AudioBuffer[]`），一个调度器在 `ctx.currentTime` 上首尾相接排期。打断 = `clear(queue)` + 停掉当前正在播的那个 `AudioBufferSourceNode`。这个结构天然支持"边收边播"，是自建链路的必备件。
- 后端侧配合：`OfflineTts.generate()` 的 callback **返回非 0 立即停止合成**（[实测] docstring 原文："*Return a non-zero value to stop generation early*"）。所以一次打断能在"合成中/传输中/播放中"三个阶段分别刹住，不会继续浪费 CPU。

### 2.6 诚实的失败模式清单 —— 什么一定不可靠

| # | 会失败的情形 | 后果 | 缓解 |
|---|---|---|---|
| 1 | 用户用**外接音箱**、音量偏大 | AEC 参考错配 + 非线性失真 → 回声被识别成用户话 | 方案 D 兜底；Demo 建议**强制要求耳机或内置扬声器**并在 UI 提示 |
| 2 | 蓝牙耳机 / USB 声卡（额外 50–200 ms 时延） | AEC 时延估计失配 → 回声泄漏 | 同上；`getSettings()` 读回实际生效值并显示 |
| 3 | 用户在 AI 开口的**前 400 ms 内**插话 | 半双工窗口把用户的话吃掉了 → 用户觉得"AI 不理我" | 窗口收到 400 ms；或改用"只在 TTS 起始 400 ms 内做能量门控"而不是完全静音 |
| 4 | 回声被 ASR **截断**（只识别出 AI 那句话的后半截） | 方案 D 的模糊匹配失败 → 误判为打断 | 用"包含 TTS 文本中 ≥6 字的连续子串"判定；并用 AI 说话期间整体调高匹配容忍度 |
| 5 | 服务端 `interrupt_response` 与本地播放队列**不同步** | 服务端已取消但本地缓冲区还在播 → 用户听到"AI 说了半句就没了" | 打断时**永远先清本地队列**，再通知服务端；不要等服务端回执 |
| 6 | 打断判定依赖 ASR 定稿 → 定稿要等端点 0.45–1.2 s | 打断手感迟钝（用户说完才停） | 打断必须走 **partial** 而不是 final：partial 每 ~330 ms 就有 [实测]，用"连续 2 个 partial 都不匹配 TTS 文本"就可提前触发 |
| 7 | 多路麦克风同时打开（`useVoiceAnalysis` 已经在开一路，通话再开一路） | `getUserMedia` 两次、AEC 参考冲突、`AudioContext` 泄漏 | **复用同一个 MediaStream**：`StreamingAsrSession.start(stream)` 已经接受外部 `MediaStream`（`streamingAsr.ts:217`），不要新建 |
| 8 | 本机没有 GPU，AEC 是 CPU 的（WebRTC 的 AEC3 很轻，但加上 16 kHz ASR + TTS 会争 CPU） | 音频卡顿/丢帧 | 用 `AudioWorklet` 而非 `ScriptProcessorNode`（现有 worklet 已经是对的）；[未验证] 未做 CPU 争用压测 |

---

## 3. 轮次切换（Endpointing / Turn-taking）

### 3.1 现状与目标

现状（`algorithm/asr/streaming.py:63-65`）：

```python
RULE1_MIN_TRAILING_SILENCE = 1.6   # 一直没有语音：静音多久算一段
RULE2_MIN_TRAILING_SILENCE = 0.8   # 说过话之后：静音多久定稿
RULE3_MIN_UTTERANCE_LENGTH = 15.0  # 单段最长
```

父会话给出的实测：**端点在语音结束后 0.75–1.2 s 触发**（rule2=0.8 s + 100 ms 块边界 + 解码开销）。

**电话感的目标**：用户说完 → AI 接话的**明显停顿应为 300–600 ms**。人类通话的自然轮次间沉默中位数约 **200 ms** 量级，超过 ~700 ms 就"像在对讲机"。所以目标是把"句末 → 定稿"压到 **0.30–0.45 s**。

### 3.2 具体参数建议

**建议值（仅用于「通话模式」，不要改全局默认）：**

| 参数 | 现值 | 通话模式建议 | 依据 / 代价 |
|---|---|---|---|
| `rule1_min_trailing_silence` | 1.6 s | **1.0 s** | rule1 管的是"还没开始说话时，静音多久算一段"。通话里用户刚接起就沉默很正常，太短会切出空段。1.0 s 足够。 |
| `rule2_min_trailing_silence` | 0.8 s | **0.40 s** | 主力参数。0.8 → 0.40 直接省 400 ms。**代价**：中文口语里"我想想……"式的 400–600 ms 思考停顿会**被误判为句末**，导致 AI 抢话。 |
| `rule3_min_utterance_length` | 15.0 s | **12.0 s** | 保留"防止无限增长"的作用。通话轮次通常更短，12 s 足够，且能更早强制切段（避免用户一口气说 20 s 时 AI 完全插不上话）。 |
| `enable_endpoint_detection` | True | True | 不变。 |
| `decoding_method` | `greedy_search` | 不变 | 不改，避免精度/延迟回归。 |

**关键补充：不要只改 rule2 就上线。** 0.40 s 会带来"过早截断"，必须配三重缓解：

**(1) 「投机启动 + 取消窗口」（收益最大，建议必做）**

不要在端点触发后才做任何事。改成：

```
t0   : sherpa 传出 rule2=0.40s 的端点 → 立刻用【partial 文本】发 LLM 请求（不等离线复识别）
t0+0 : 同时继续喂音频，sherpa 新起一段
窗口 : [t0, t0+300ms]
       若窗口内 sherpa 又出 partial（用户其实还在说）
         → 丢弃这次投机请求（abort SSE / 丢弃响应），
           把两段 partial 拼接后重新发
       若窗口内无新 partial
         → 把 t0+60ms 时产出的【离线复识别定稿】与已发的 partial 比对：
             差异小（编辑距离 < 20%）→ 继续用投机请求的结果
             差异大                    → 重新发一次
```

- 收益 **[估计]** 300–500 ms（端点等待与 LLM 首 token 部分重叠）。
- 代价：思考停顿被误判时会**多花一次 LLM 调用**。竞赛 Demo 场景完全可接受。
- 仓库已有全部原料：`StreamingSession.accept()` 在**同一次调用里**可能同时返回 partial 和 final（`streaming.py:312-367`），所以"端点触发时手上已经有 partial 文本"这件事天然成立。

**(2) 语义端点（在 partial 上做，不需要新模型）**

在 `onPartial` 里对文本做**零成本**的启发式判据，命中则**提前**把端点时间缩短（或在投机窗口里更激进）：

- **句末语气词/收尾词**：`吗 / 呢 / 吧 / 啊 / 了 / 的 / 嘛` 结尾 → 大概率说完了 → 缩短等待。
- **明显的连接词开头未完成**：partial 以 `因为 / 然后 / 就是 / 而且 / 但是 / 所以 / 我想说 / 那个` 结尾 → **说明话没说完** → 延长等待到 ~700 ms。
- **partial 长度很短（< 4 字）且静音** → 可能只是"嗯"，不要急着定稿，等 rule1。

这套规则**不是**"安全规则"，是**交互时序规则**，不触发 AGENTS.md 的"安全阈值需人工审核"约束。但**建议仍让 reviewer 过一眼**，因为它可能间接影响风险判定的时机。

**(3) 「用户还在想」检测**

三个层次，按成本排序：

- **免费层**：rule2 延长（0.40 s → 0.70 s）+ 上面 (2) 的连接词判据。**[估计]** 能消除大部分误截断。
- **低成本层**：本地 VAD 判据 —— 用户在思考时通常**不是纯静音**（有呼吸、咂嘴、"呃…"）。`sherpa-onnx` 已装且暴露 `SileroVadConfig` [实测：`from sherpa_onnx import SileroVadConfig` 存在]。在 rule2 窗口内如果 VAD 仍偶尔判为语音，就延长等待。**代价**：模型缺失/阈值另一处需要调；**[未验证]** 我没有确认本机是否已有 Silero VAD 模型文件（`algorithm/models/` 下我只看到 asr/punct 两类）。
- **托管层（若走 Omni-Realtime）**：直接把 `turn_detection` 换成 `semantic_vad`，并用 `idle_timeout_ms`（[5000, 30000]）让服务端在用户沉默过久时**主动推进对话**。官方原文：「*the model proactively generates a response to prompt the user to continue the conversation based on the current context.*」这是**唯一一个真正解决"过早截断 vs 响应太慢"矛盾的方案**，因为它用的是**语义**而不是声学静音。**[未验证]** 我没有实测 `semantic_vad` 的效果，也没有实测 `idle_timeout_ms`。

### 3.3 一句话结论

`rule2: 0.8 → 0.40 s`，配 `rule1: 1.0 s`、`rule3: 12.0 s`，**并且必须同时实现"投机启动 + 300 ms 取消窗口"** —— 否则 0.40 s 会以"AI 频繁抢话"的形式反噬，比 0.8 s 更伤体验。若要彻底解决，只能上 `semantic_vad`（Omni-Realtime）。

---

## 4. 延迟预算

### 4.1 说明

- `[实测]` 的四个数字（首 partial 770 ms、partial 间隔 330 ms、端点 0.75–1.2 s、定稿 51–61 ms）**来自父会话的测量**，我本次**没有复测**（没有跑服务、没有下载模型）。
- 所有 `[估计]` 都是基于"本机 CPU + 本机 WS + 阿里云 DashScope（国内）"的推算。
- 表中"中位"是给 Demo 现场做期望管理用的单一数字。

### 4.2 Path A —— 自建链路（本地 ASR + 云端 LLM + 本地 TTS）

**A-1：从「用户说完最后一个字」到「音箱里听到 AI 第一个音」**

| # | 环节 | 实测/估计值 | 依据 |
|---|---|---|---|
| 1 | 麦克风 100 ms 块攒满（`BLOCK_SAMPLES=1600`） | 0 – 100 ms（期望 50）**[实测-设计]** | `pcm-capture-processor.js:20,64-67`：攒满 1600 样本才投递 |
| 2 | 上行 WS + 服务端解码成 partial | partial 首结果 **~770 ms**（从说话开始计）；之后每 **~330 ms** 刷新 **[实测-父会话]** | 这是"边说边出字"的延迟，不是句末延迟 |
| 3 | **句末静音判定（rule2）** | **现值 0.75–1.2 s** **[实测-父会话]** → **建议值 0.40–0.60 s** | §3.2 |
| 4 | 定稿：离线 Paraformer 复识别 + CT-Transformer 标点 | **51 – 61 ms** **[实测-父会话]** | `asr/streaming.py:266-301`（RTF 0.008）+ `punctuation.py:150-167`（1–5 ms） |
| 5 | 风险判定 + `_StreamingSafetyGate` 词法筛查 + SSE 打包 | **< 5 ms** **[估计]** | 纯 Python 子串匹配，~50 个词条 |
| 6 | 上行到 DashScope + `qwen-turbo` 首 token | **300 – 700 ms** **[估计]** | `intervention.py:988-1003` 直连 `dashscope.aliyuncs.com`；未测 |
| 7 | 首句补齐（约 12–20 字，够起播） | **+150 – 350 ms** **[估计]** | 中文 ~15–25 tok/s 的流式速率 |
| 8 | TTS 首块（`generate()` 第一次 callback） | **100 – 300 ms** **[估计]** | 假设 fanchen-C 本机 RTF 0.15–0.25；**若模型未预热需再加 300–1500 ms 首次加载** |
| 9 | WS 下行 + AudioWorklet 入队 + `AudioBufferSource.start()` | **40 – 80 ms** **[估计]** | 本机环回 + 播放调度 |
| | **合计：首声** | **中位 ≈ 1.6 s；区间 1.1 – 2.4 s** | 用建议的 rule2=0.5 s |

**A-2：从「用户说完」到「AI 说完最后一个字」**

```
T_end ≈ 首声(1.6 s) + 音频时长 T
T = 回复字数 × ~180 ms/字     （中文 5.5 字/秒 ≈ 180 ms/字）[估计]
```

**硬约束**：**TTS 的总 RTF 必须 < 1.0**，否则合成永远追不上播放，队列越积越长（听起来像 AI 越说越慢/越来越卡）。这是 Path A 的**结构性风险**：
- fanchen-C 官方 Pi4 RTF 1.600（4 线程）→ **如果本机也是这个量级，Path A 直接不成立**。
- 本机若为 0.15–0.25 → 有 4–6 倍余量，安全。
- **所以 1.3 节里那个 benchmark 是 Path A 的 go/no-go 关卡，不是可选项。**

例：20 字回复 = 3.6 s 音频 → `T_end ≈ 1.6 + 3.6 = 5.2 s`。

### 4.3 Path B —— DashScope `qwen3-omni-flash-realtime`（端到端语音到语音）

| # | 环节 | 值 | 依据 |
|---|---|---|---|
| 1 | 麦克风帧（可直接 20–40 ms，不必攒 100 ms） | **20 – 40 ms** **[估计]** | Realtime 是持续 append，无块对齐要求 |
| 2 | 上行网络（本机 → 阿里云） | **30 – 80 ms** **[估计]** | 未测 |
| 3 | **服务端 VAD `silence_duration_ms`** | **200 – 800 ms 可配**，官方默认 800，建议 **300–400** **[实测-文档]** | client-events 文档：合法区间 `[200, 6000]`；或换 `semantic_vad` |
| 4 | 服务端产首 `response.audio.delta`（含 LLM 首 token + 音频首块） | **250 – 700 ms** **[估计]** | 三段折叠成一段，通常比 A 的 (6+7+8) 之和短 |
| 5 | 下行网络 + 播放启动 | **30 – 80 ms** **[估计]** | 24 kHz PCM 流 |
| | **合计：首声** | **中位 ≈ 0.85 s；区间 0.55 – 1.7 s**（silence=800 时中位 ≈ 1.35 s） | |

**B-2：到说完**
`T_end ≈ 首声 + T`，但**没有 TTS RTF 约束**（服务端边生成边推流，天然实时）。这消除了 Path A 最大的结构性风险。

### 4.4 两条路的对比与结论

| 维度 | Path A（自建） | Path B（Omni-Realtime） |
|---|---|---|
| 首声中位 | ~1.6 s **[估计]** | ~0.85 s **[估计]** |
| 到说完 | 1.6 + T，**受 TTS RTF 约束** | 0.85 + T，无 RTF 约束 |
| 数据是否出设备 | **否**（音频不出本机；只有 ASR 文本发 DashScope） | **是**（未成年人音频上云） |
| 依赖 | 已装 sherpa-onnx + 需下载 121 MB 模型 | 需外网 + 计费 + 新写 WS 客户端 |
| 打断 | 自己实现（§2） | 服务端 `interrupt_response: true` 内建 |
| 端点 | 自己调 rule1/2/3 | `silence_duration_ms` + `semantic_vad` + `idle_timeout_ms` |
| 安全审计入口 | ASR 文本（已有） | `conversation.item.input_audio_transcription.*` 事件（已确认存在） |
| 风险 | TTS RTF 可能 > 1（未测） | 合规/计费/未测延迟 |

### 4.5 单一最大瓶颈

**把链路拆成"串联的不可消除等待"，Path A 的首声 1.6 s 由三段构成：**

```
句末静音判定 (0.5 s) + LLM 首 token (0.5 s) + LLM 首句补齐 + TTS 首块 (0.35 s) ≈ 1.35 s
```

**最大且唯一能显著压缩的单项是「句末静音判定」**（占 0.75–1.2 s → 目标 0.4–0.6 s，**[估计]** 净省 350–600 ms）。它同时也是**风险最高**的一项，因为压太狠会变成"AI 抢话"。

**第二大是 LLM 首 token + 首句补齐（合计 ~0.85 s [估计]）**。对策是 §3.2(1) 的**投机启动**：[估计] 净省 300–500 ms，代价是可能浪费一次调用。

**结构性结论**：Path A 的 1.6 s 是"三段串联"的下限；Path B 之所以能到 0.85 s，不是因为它每一步更快，而是因为它**把三段合并成一段**（没有"ASR 定稿 → 单独起 LLM → 单独起 TTS"的串行边界）。**如果 1.6 s 的首声在 Demo 现场被认为"不够电话感"，唯一的结构性解法是 Path B，而不是继续抠 Path A 的每一毫秒。**

**最后必须实测的三件事（按优先级）**：
1. `OfflineTts` + fanchen-C 的**本机 RTF 与首 callback 延迟**（决定 Path A 是否成立）。
2. `qwen-turbo` 在本机的**首 token 延迟**（决定 §3.2 投机启动的收益）。
3. `qwen3-omni-flash-realtime` 的**端到端首声延迟**（决定值不值得为它承担合规成本）。

---

## 5. 安全约束下的实时语音

三大硬约束（AGENTS.md）：**禁止诊断性语言**、**禁止药物建议**、**`risk_level ∈ {high, crisis}` 必须触发升级**；且**新增安全规则/阈值需人工审核，复用既有规则不需要**。

### 5.1 仓库里已经有一个为此而生的组件 —— `_StreamingSafetyGate`

`algorithm/api/intervention.py:1027-1157`。它的 docstring 原文已经把边界写清楚了：

> 「本闸门**不引入任何新的安全规则或阈值**。它只复用 `config/audit_rules.yaml` 里既有的 `stigma_rejection` 词表（含 `teen_specific.stigma_rejection` 的扩展词表），把原本在整段审计时才生效的词法红线**提前到句子级**。按 AGENTS.md「自动化流程不得自行修改安全阈值或升级条件」，这里改变的是既有规则的生效时机，不是规则本身。」
>
> 「已知缺口：`audit_rules.yaml` 目前没有药物相关词法轴，而 AGENTS.md 明令禁止药物建议。补这一轴属于**新增安全规则**，需人工审核后方可合入，故此处不擅自添加。」

它的公开接口正好就是"逐句放行"：

| 成员 | 签名 | 作用 |
|---|---|---|
| `feed` | `feed(delta: str) -> list[str]` | 追加 LLM 增量，返回**可以放行的完整句子**列表；命中红线后一律返回 `[]` 且此后再不放行 |
| `flush` | `flush() -> list[str]` | 流结束时取出残句（同样过筛查） |
| `screen` | `screen(text: str) -> list[str]` | 对整段文本做词法筛查，返回命中词条 |
| `tripped` | `-> bool` (property) | 是否已命中红线 |
| `hits` / `released_chars` / `scanned_units` / `terms_count` | property | 命中词条 / 已放行字数 / 已筛查单位数 / 载入词条数（自检） |

切句规则：`_SENTENCE_BOUNDARY_RE = re.compile(r'[。！？!?；;\n]')`（`intervention.py:943`），上限 `_STREAM_UNIT_MAX_CHARS = 200`（`intervention.py:947`）。

### 5.2 推荐设计：**「逐句过闸 → 逐句合成 → 逐句播」+「高风险硬禁用语音」**

```
                      ┌─ 硬闸门（复用既有 risk_level 判定，不新增阈值）─┐
用户语音 → ASR 定稿 → │ risk_level ∈ {high, crisis} 或 crisis_detected? │
                      └───────────────────┬───────────────────────────┘
                                    是 │            │ 否
                                       ▼            ▼
                     退出语音模式，转文字升级通道     _build_llm_messages → _iter_llm_deltas (SSE)
                     （既有 escalation + 资源卡）              │
                                                              ▼
                                          _StreamingSafetyGate.feed(delta) → 完整句 unit
                                                              │  （命中红线 → tripped，停止生成）
                                                              ▼
                                                    TTS 工作线程 generate(unit, callback)
                                                              │  边合成边推 PCM
                                                              ▼
                                                    前端播放队列 → 出声
                                                              │
                                    同时：整段回复跑既有 _audit_and_finalize 五轴审计
                                                              │
                                        审计改写？ → REVISE → 【立刻停止播放 + 清空队列 + 用改稿重新合成】
```

**(1) 硬禁用语音模式 —— 直接复用仓库既有的先例，零新增规则**

`intervention.py:1181-1183` 已经这么做过了（对文本流式）：

```python
streaming_allowed = (
    prep.risk_level in ("low", "medium") and not prep.crisis_detected
)
```

**语音模式用同一个布尔量即可**：`voice_allowed = streaming_allowed`。这样：
- **没有新增任何阈值或判定条件** → 满足 AGENTS.md「复用既有规则不需要人工审核」。
- high/crisis 时**整条语音通道不启动**，escalation 路径仍然是既有的、原子的、纯文本的 —— 不会被"边生成边播"的异步性污染。
- 这一点很重要：**危机场景下最怕的就是"AI 已经大声念出了不该念的内容，然后审计才说要改"**。硬禁用从结构上排除了这种可能。

**(2) 逐句过闸 —— 已可复用，但需要一处**小**改动**

`_StreamingSafetyGate` 现在**可以原样复用**。唯一的问题是 `_STREAM_UNIT_MAX_CHARS = 200` 对语音太长了（200 字 ≈ 36 秒音频 ≈ 首声要多等十几秒）。

**建议：给 `_StreamingSafetyGate.__init__` 加一个可选参数 `max_unit_chars: int = _STREAM_UNIT_MAX_CHARS`，语音通道传 20–24。** 这**不是**新增安全规则（改的是一次切多长的交互参数，不是判定什么算红线），也不改变任何红线行为 —— 但因为它触及安全组件的接口，**建议仍然告知 reviewer**。另一种更保守的做法是**完全不改这个类**，在语音通道里用一个独立的 `SentenceUnitSplitter`（同样的 `_SENTENCE_BOUNDARY_RE`，更小的上限）先切句，再把每一句喂给 `gate.screen(unit)`（公有方法）。**我推荐后者 —— 零改动、零审核成本。**

**(3) 首句必须缓冲的延迟代价（诚实说明）**

你不可能"一边说一边审"。闸门要求先凑出一个**完整语义单位**才能判红线（docstring 原文：「拿到「你」或「你可」这样的前缀无法判断任何事」）。所以：

- **代价**：AI 首声必然包括"生成第一个完整短句"的时间。用 20~24 字上限时，**[估计] 200–350 ms**。这是必须付的，不是可优化项。
- **反过来说**：这也是**为什么首句应该设计成最安全的一句**。现有 prompt（`intervention.py:691-695`）的标准范式是「简短共情接纳 → 一个开放式追问」——**首句天然就是共情反映**，风险最低。**建议把这一点写进语音模式的 prompt 约束**（"首句必须是纯共情反映，不得包含任何建议、判断、信息"），这样"已经播出去的第一句"几乎不可能踩线。注意：这会改动 prompt 文案，**属于安全相关变更，需人工审核**。

**(4) 审计改写（REVISE）时的处理 —— 必须诚实**

现有 SSE 契约是「**先说、后校**」（`intervention.py:1280-1282` 原文：「本项目的审计是整段级的，无法在吐字之前拿到结论，因此采用「先说、后校」」）。搬到语音上：

- 收到 `revise` → **立即 `stop()` 当前播放 + 清空本地 PCM 队列 + 中止 TTS 工作线程 + 丢弃待合成句队列**，然后用 `final.reply` 重新合成。
- **但已经播出去的声音收不回来。** 这是本方案**无法消除的残余风险**，必须写进 Demo 的免责说明。
- 缓解：闸门（句子级、词法）+ prompt 硬边界（`intervention.py:721-726`）+ 首句纯共情约束，三者叠加后，**"已经播出去且被审计否掉"的概率被压到很低** —— 但**不是零**。

**(5) 药物轴的缺口 —— 必须由人决定**

`_StreamingSafetyGate` 的 docstring 已经明确：`audit_rules.yaml` **没有药物词法轴**，而 AGENTS.md 禁止药物建议；补这一轴是新增规则，需人工审核。

**语音模式上线前必须有一个明确决定**：
- **选项 A（保守，推荐做 Demo 用）**：接受"语音通道没有药物词法拦截"，靠 ①prompt 硬边界（`intervention.py:724`「绝不提供药物、医疗相关建议」）+ ②整段审计兜底。**不新增规则**。
- **选项 B**：由人给 `audit_rules.yaml` 加一个 `medication_rejection` 轴（药物名词表 + 「吃/剂量/毫克/mg/停药」组合），并把它同时接进闸门。**需人工审核**，我不代为拟定。

**(6) 危机时的具体动作（建议，需人工确认措辞）**

```
risk_level ∈ {high, crisis}
  → 1. 立即停止 AI 语音播放（stop + 清队列 + 中止合成）
  → 2. 会话切到纯文字通道（voiceInputActive=false，禁用通话按钮）
  → 3. 走既有 escalation 流程（AGENTS.md：必须触发）
  → 4. 播出一句【固定的、预审核过的】短句（例如"我很在意你刚才说的话，我们换成文字慢慢聊，好吗？"）
       —— 用预合成 PCM 缓存播放，不经过 LLM，也就不用过闸门
  → 5. UI 立即展示 crisis_resource_card（audit_rules.yaml:227-236 的既有内容）
```

两点注意：
- 第 4 步那句固定话术是**新增的用户可见安全文案**。它不是检测规则、不是阈值，严格说不触发 AGENTS.md 的"规则变更需审核"条款；但**我建议仍然让人过一眼**，因为它出现在危机时刻。
- **不要用 TTS 朗读热线电话号码**。TTS 读数字容易出错（"400-161-9995" 可能被读成"四十万…"），而危机场景下念错号码后果严重。号码只以**文字卡片**形式呈现（既有做法）。这是我在本报告中**最强烈的一条建议**。

### 5.3 安全设计的净效应

| 风险 | 是否被结构上排除 |
|---|---|
| high/crisis 时 AI 还在语音陪伴 | ✅ 硬禁用（复用 `streaming_allowed`） |
| 说诊断性标签 / 污名化词 | ✅ 句子级闸门（既有词表）+ 整段审计 |
| 说药物建议 | ⚠️ **仅靠 prompt + 整段审计**（词法轴缺失，需人工决定是否补） |
| 已经播出的第一句踩线 | ⚠️ 概率很低但**非零**（闸门 + 首句纯共情约束缓解） |
| 审计改写了已播内容 | ⚠️ **无法收回**（架构性残余风险，必须先说后校） |
| 危机时误念热线号码 | ✅ 建议只用文字卡片，不用 TTS |

---

## 6. 可复用既有资产

以下全部是我**读过源码并核对过行号**的。行号对应当前工作区状态。

### 6.1 `client/public/worklets/pcm-capture-processor.js`（73 行）

- 注册处理器名：`registerProcessor('pcm-capture', PcmCaptureProcessor)`（第 73 行）
- `const BLOCK_SAMPLES = 1600; // 16000 Hz × 100 ms`（第 20 行）
- **消息协议**（第 15-18 行注释）：
  - 主线程 → worklet：`{ type: 'flush' }`
  - worklet → 主线程：`{ type: 'block', samples: Float32Array(1600), rms: number }`（第 39 行，`rms = sqrt(mean(x²))`）
  - worklet → 主线程：`{ type: 'flushed', samples: Float32Array }`（第 45、48 行）
- 无状态、不做重采样（假定上游 `AudioContext({sampleRate:16000})`）、不阻塞渲染线程。
- **本项目零改动可复用做通话上行**。注意它**导出 `rms`**，可以直接拿来做 §2.3 方案 D/能量门控的输入，不用另开 Analyzer。

### 6.2 `client/src/services/streamingAsr.ts`（326 行）

导出的类型与 API：

```ts
export interface StreamingAsrMessage {
  type: 'partial' | 'final' | 'status' | 'error';
  text?: string; elapsed_ms?: number; latency_ms?: number;
  reason?: string;          // 仅 final: 'endpoint' | 'finalize'
  engine?: string;
  refined?: boolean;        // 仅 final: 是否经离线复识别纠错
  available?: boolean; message?: string;
}

export interface StreamingAsrMeta {
  elapsedMs: number; latencyMs: number; reason: string;
  engine: string; refined: boolean;
}

export interface StreamingAsrCallbacks {
  onPartial?: (text: string, meta: StreamingAsrMeta) => void;
  onFinal?:   (text: string, meta: StreamingAsrMeta) => void;
  onStatus?:  (status: { available: boolean; reason?: string; engine?: string }) => void;
}

export function floatToInt16(samples: Float32Array): ArrayBuffer;   // 第 62 行

export class StreamingAsrSession {
  constructor(cb: StreamingAsrCallbacks = {});        // 第 92 行
  get active(): boolean;                              // 第 97 行
  async start(stream: MediaStream): Promise<void>;    // 第 217 行  ← 接受外部 MediaStream
  async stop(): Promise<void>;                        // 第 274 行
  resetSegment(): void;                               // 第 314 行
}

export { BLOCK_MS };   // = 100，第 55 / 326 行
```

内部常量：`WORKLET_URL = '/worklets/pcm-capture-processor.js'`（53）、`SAMPLE_RATE = 16000`（54）、`MAX_PENDING_BLOCKS = 10`（57）、`MAX_RECONNECTS = 3`（59）、`resolveWsUrl()` → `${proto}//${location.host}/algorithm/api/v1/asr/ws`（73-76）。

**对通话特性的三个关键点**：
- `start(stream)` **接受外部 `MediaStream`** → 通话页可以和现有 `useVoiceAnalysis` **共用一路麦克风**，避免 `getUserMedia` 两次（见 §2.6 失败模式 7）。
- `stop()` 会先发 `{"type":"finalize"}` 并等 600 ms（第 283-288 行）—— **通话模式下不要用它做"立刻停止"**，打断需要新的即时路径。
- 断线策略已内建：指数退避 0.5/1/1.5 s，3 次后如实上报不可用（158-171 行）。

### 6.3 `client/src/hooks/useVoiceAnalysis.ts`（375 行）

```ts
export type AudioEnergyCallback = (rmsEnergy: number) => void;

export function useVoiceAnalysis(
  ageGroup: AgeGroup | null = null,
  onAudioEnergy?: AudioEnergyCallback | null
): {
  metrics: VoiceAnalysis;
  start: () => Promise<void>;
  stop: () => void;
  reset: () => void;
  audioContextRef: React.MutableRefObject<AudioContext | null>;
  analyserRef: React.MutableRefObject<AnalyserNode | null>;
};
```

关键内部行为：
- `navigator.mediaDevices.getUserMedia({ audio: true })`（**第 233 行 —— 没有任何 audio 约束**，是加 `echoCancellation/noiseSuppression/autoGainControl` 的改动点）。
- 第 236 行 `new AudioContext()`（默认采样率，用于声学分析）；`analyser.fftSize = 2048`、`smoothingTimeConstant = 0.8`（241-242）。
- 第 274-321 行构造 `StreamingAsrSession`，把 `onPartial` 写进 `asrPartialText`、`onFinal` 写进 `speechText` / `speechTextHistory`（截 5 条）/ `asrFinalText` / `asrFinalSeq`（自增序号，用来区分"新的一段"即使文本相同）/ `asrRefined`。
- 导出给下游的 metrics 字段（通话会用到）：`isRecording`、`isSpeechRecognizing`、`asrPartialText`、`speechText`、`speechTextHistory`、`asrFinalText`、`asrFinalSeq`、`asrRefined`、`asrEngine`、`asrError`、`asrLatencyMs`。
- `stop()`（337-353）会 `stream.getTracks().forEach(t=>t.stop())` 并 `audioContext.close()`，然后 `asrSessionRef.current.stop()`。

**对通话特性的关键点**：`stop()` 会把**整个 MediaStream 关掉**。通话场景"打断后要继续听"不能调它 —— 需要新增一个"只清 ASR 段/队列、不关流"的路径（`resetSegment()` 是现成的但不含播放侧动作）。

### 6.4 `algorithm/api/asr.py`（269 行）

```python
router = APIRouter(prefix="/asr", tags=["语音识别"])

MAX_AUDIO_B64_CHARS = 4_000_000                     # 第 39 行

class TranscribeRequest(BaseModel):                 # 第 42 行
    audio_base64: str
    sample_rate: int = 16000

class TranscribeResponse(BaseModel):                # 第 53 行
    text: str; engine: str
    duration_ms: float = 0.0; latency_ms: float = 0.0; error: str = ""

class StatusResponse(BaseModel):                    # 第 62 行
    available: bool; engine: str
    model_dir: str | None = None; model_name: str | None = None
    sample_rate: int = 16000; error: str = ""
    streaming_available: bool = False
    streaming_engine: str = AsrEngine.UNAVAILABLE.value
    streaming_model_name: str | None = None
    streaming_error: str = ""
    punctuation_available: bool = False
    punctuation_error: str = ""

@router.websocket("/ws")
async def asr_stream(websocket: WebSocket) -> None:              # 第 95 行

@router.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest) -> TranscribeResponse:    # 第 205 行

@router.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:                                  # 第 241 行

@router.get("/model-dir")
def model_dir() -> dict[str, str | None]:                        # 第 266 行
```

**WS 协议（第 98-112 行）**：
- 客户端 → 服务端：二进制帧 = 16 kHz mono int16 LE PCM（建议 100 ms / 3200 B）；文本帧 = `{"type":"finalize"}` / `{"type":"reset"}`
- 服务端 → 客户端：`{"type":"status", available, engine, error}`（连接后立即）、`{"type":"partial", text, elapsed_ms, latency_ms, engine}`、`{"type":"final", text, elapsed_ms, latency_ms, reason, engine, refined}`、`{"type":"error", message}`
- 用 `run_in_threadpool(session.accept_pcm16, data)`（第 162 行）把 CPU 密集解码丢出事件循环 —— **TTS 合成必须照抄这个模式**。

### 6.5 `algorithm/asr/streaming.py`（430 行）

```python
STREAMING_MODEL_NAME = "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
RULE1_MIN_TRAILING_SILENCE = 1.6    # 第 63 行
RULE2_MIN_TRAILING_SILENCE = 0.8    # 第 64 行
RULE3_MIN_UTTERANCE_LENGTH = 15.0   # 第 65 行
MIN_REFINE_SAMPLES = int(SAMPLE_RATE * 0.5)   # 第 60 行

def resolve_streaming_model_dir(explicit: Optional[str | Path] = None) -> Optional[Path]   # 第 76 行

class StreamingRecognizer:                       # 第 108 行
    @classmethod
    def instance(cls) -> "StreamingRecognizer"   # 第 120 行
    available -> bool                            # property, 第 129 行
    load_error -> str                            # property, 第 136 行
    engine -> AsrEngine                          # property, 第 141 行
    load() -> bool                               # 第 145 行
    new_session() -> Optional["StreamingSession"]# 第 199 行
    status() -> dict[str, object]                # 第 205 行（含 endpoint_rules 三元组）

class StreamingSession:                          # 第 223 行
    accept(samples: np.ndarray) -> list[StreamingAsrUpdate]        # 第 312 行
    accept_pcm16(raw: bytes) -> list[StreamingAsrUpdate]           # 第 369 行
    finalize() -> Optional[StreamingAsrUpdate]                     # 第 379 行
    reset() -> None                                                # 第 420 行
    total_samples -> int                                           # property, 第 428 行
```

**对通话特性最重要的一点**：`accept()` / `accept_pcm16()` **在同一次调用里就可能同时返回 partial 和 final**（第 329-367 行）—— 端点检测和定稿都在服务端完成，**不需要客户端再发一次 `finalize`**。这意味着通话服务端可以在同一个 WS 循环里：收到端点 → 立刻起 LLM。这是 §3.2 投机启动能落地的基础。

`_refine()`（第 254 行）和 `_refine_asr()`（第 266 行）会把定稿交给离线 Paraformer 复识别；`_refine` 再调 `PunctuationRestorer.restore()`。

### 6.6 `algorithm/asr/engine.py`（317 行）

```python
MODEL_NAME = "sherpa-onnx-paraformer-zh-small-2024-03-09"
SAMPLE_RATE = 16000 ; FEATURE_DIM = 80
REQUIRED_FILES = ("model.int8.onnx", "tokens.txt")

def resolve_model_dir(explicit: Optional[str | Path] = None) -> Optional[Path]   # 第 64 行
def decode_wav(data: bytes) -> tuple[np.ndarray, int]                            # 第 99 行

class ParaformerRecognizer:                       # 第 163 行
    instance() -> "ParaformerRecognizer"          # 第 180 行
    available -> bool; load_error -> str; engine -> AsrEngine      # properties
    load() -> bool                                                  # 第 208 行
    transcribe(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> AsrResult   # 第 256 行
    status() -> dict[str, object]                                   # 第 307 行
```

### 6.7 `algorithm/asr/punctuation.py`（175 行）

```python
PUNCT_MODEL_NAME = "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"
PUNCT_REQUIRED_FILES = ("model.onnx",)

def resolve_punct_model_dir(explicit: Optional[str | Path] = None) -> Optional[Path]  # 第 43 行

class PunctuationRestorer:                        # 第 64 行
    instance() -> "PunctuationRestorer"           # 第 81 行
    available -> bool; load_error -> str
    load() -> bool                                # 第 100 行
    restore(text: str) -> str                     # 第 150 行（失败/缺失/过短 → 原样返回）
    status() -> dict[str, object]                 # 第 169 行
```

**[建议]**：`restore()` 的输入是 LLM 输出时，同样能用来给"没有标点"的句子切分提供边界。但**这不是它现在的用途**（它只服务 ASR 定稿），不要混用 —— LLM 输出通常自带标点。

### 6.8 `algorithm/api/intervention.py`（1943 行）—— 通话下游全在这里

| 位置 | 名称 / 签名 | 说明 |
|---|---|---|
| 第 394 行 | `class SmartChatRequest(BaseModel)`：`user_id: str`、`message: str`、`conversation_history: list[dict] = []`、`session_id: Optional[str] = None` | 对话请求体 |
| 第 404 行 | `class SmartChatResponse(BaseModel)`：`reply, dialog_state, dialogue_mode="EMPATHY", action_type, risk_level, audit_passed, requires_escalation, emotion_probs, evidence, fallback=False, scene_retrieval=None, cognitive_distortion=None` | 非流式响应 |
| 第 618 行 | `_LLM_BASE_URL = os.environ.get(...)`；第 621 行 `_LLM_MODEL = os.environ.get('DASHSCOPE_MODEL', 'qwen-turbo')` | LLM 端点与模型 |
| 第 626 行 | `_build_llm_messages(user_message, conversation_history, emotion_probs, risk_level, style=None, scene_context="", ask_question=True) -> list[dict]` | 流式/非流式**共用**的 prompt 装配；含全部安全硬边界（721-726 行）与"简短 1~2 句"风格约束（729 行） |
| 第 754 行 | `_call_llm(user_message, conversation_history, emotion_probs, risk_level, style=None, dialogue_mode="EMPATHY", scene_context="") -> str` | 非流式（`requests.post`，第 808 行） |
| 第 ~960 行 | `_iter_llm_deltas(messages, api_key, timeout) -> Iterator[str]` | **SSE 流式**（`requests.post(..., stream=True)`，第 988-1003 行），逐 `data:` 解析 `choices[0].delta.content` |
| 第 943 行 | `_SENTENCE_BOUNDARY_RE = re.compile(r'[。！？!?；;\n]')` | 切句正则 |
| 第 947 行 | `_STREAM_UNIT_MAX_CHARS = 200` | 无标点时的切分上限 |
| 第 950 行 | `_sse_pack(event_type: ChatStreamEventType, payload: dict) -> str` | SSE 打包 |
| 第 1027 行 | `class _StreamingSafetyGate` | **§5 的核心复用件**，接口见 §5.1 |
| 第 1160 行 | `_chat_stream_events(request: SmartChatRequest) -> Iterator[str]` | 事件序列 `meta → delta* → [revise] → done`；第 1181-1183 行 `streaming_allowed = prep.risk_level in ("low","medium") and not prep.crisis_detected` |
| 第 1269 行 | `@router.post("/smart-chat/stream")` `async def smart_chat_stream(request) -> StreamingResponse` | `media_type="text/event-stream"`，带 `X-Accel-Buffering: no` |

### 6.9 `client/src/context/AnxietyContext.tsx`（1145 行）

Context 接口中与语音相关的部分（第 308-312 行）：

```ts
  // 纯语音输入（只开麦克风，不开摄像头）—— 供聊天页"语音录入"用。
  voiceInputActive: boolean;
  startVoiceInput: () => Promise<void>;
  stopVoiceInput: () => void;
```

实现：

```ts
// 第 975-983 行
const startVoiceInput = useCallback(async () => {
  if (voice.metrics.isRecording) return;
  try { await voice.start(); console.log('[多模态] 语音输入已启动（仅麦克风）'); }
  catch (err) { console.error('[多模态] 语音输入启动失败:', err); }
}, [voice]);

// 第 985-988 行
const stopVoiceInput = useCallback(() => { voice.stop(); console.log('[多模态] 语音输入已停止'); }, [voice]);
```

- 暴露到 Provider value：第 1115-1117 行 `voiceInputActive: voiceMetrics.isRecording`、`startVoiceInput`、`stopVoiceInput`。
- 相关 hook：第 400 行 `const eyeTracking = useEyeTracking(ageGroup);`；第 426 行 `const [eyeMetrics, setEyeMetrics] = useState(eyeTracking.metrics);`；第 447 行同步；第 957 行 `eyeTracking.reset()`（关摄像头时）；第 1085 行 `reset()` 里也重置。
- `window.__recentUserText` 在 `analyzeTextGlobal`（第 991-994 行）里被写入 —— 通话的 ASR 定稿若要进文本分析，走这个入口是一致的。

**对通话特性的关键点**：`startVoiceInput` / `stopVoiceInput` 是"只开麦克风"的**正确入口**，通话页应该复用它（而不是自己 `getUserMedia`）。但它内部是 `voice.start()` → `getUserMedia({audio:true})`（无约束）→ 新建 `AudioContext()` → 新建 `StreamingAsrSession`。**通话需要的三处改动都在这个 hook 内部**：①加 audio 约束；②让通话和它共用同一路 stream；③加"打断但不关流"的路径。

### 6.10 `algorithm/config/audit_rules.yaml`（286 行）

可被 §5 直接复用的词表（**全部是既有内容，复用不需审核**）：

| 位置 | 键 | 内容条数 |
|---|---|---|
| 第 36-56 行 | `stigma_rejection.diagnostic_labels` (11) + `stigmatizing_terms` (6) | 17 |
| 第 175-193 行 | `teen_specific.stigma_rejection.additional_diagnostic_labels` (8) + `additional_stigmatizing_terms` (6) | 14 |
| 第 227-236 行 | `crisis_resource_card`（3 条热线 + message） | 危机资源卡 |
| 第 118-138 行 | `teen_specific.crisis_delay.direct_crisis_signals` (7) + `joke_crisis_patterns` (7) + `min_crisis_matches: 2` | 危机信号 |
| 第 10-13 行 | `crisis_delay.max_turns_without_escalation: 2`、`action_on_fail: "escalate_human"` | 升级延迟轴 |
| 第 281-286 行 | `global.max_rewrite_attempts: 2`、`enable_audit_log`、`audit_log_path` | 全局 |

配套：`algorithm/intervention/auditor.py:53 def load_audit_config(config_path: Optional[str] = None) -> dict`。

### 6.11 `algorithm/tools/download_asr_model.py`（224 行）—— TTS 模型下载可直接扩展

```python
@dataclass(frozen=True)
class Preset:
    repo: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = field(default_factory=tuple)
    subdir: str = "asr"
    note: str = ""

PRESETS: dict[str, Preset] = { "paraformer-zh-small": ..., "streaming-zipformer-zh-14m": ..., "punct-zh-en": ... }

def mirror_endpoint() -> str      # 默认 https://hf-mirror.com，可被 HF_ENDPOINT 覆盖（第 99 行）
_BROWSER_UA = "Mozilla/5.0 ... Chrome/153.0.0.0 Safari/537.36"   # 第 107 行
def download(repo: str, filename: str, dest: Path, retries: int = 3) -> None   # 第 113 行
def _download_preset(name, preset, models_root, skip_test_wavs) -> bool        # 第 160 行
def main(argv: Iterable[str] | None = None) -> int                             # 第 192 行
DEFAULT_PRESET = "paraformer-zh-small"
```

**建议新增一个 preset**（这是本报告对 TTS 落地的具体建议，未执行）：

```python
"tts-zh-fanchen-C": Preset(
    repo="csukuangfj/vits-zh-hf-fanchen-C",
    required=("vits-zh-hf-fanchen-C.onnx", "lexicon.txt", "tokens.txt"),
    optional=("date.fst", "number.fst", "phone.fst", "new_heteronym.fst"),
    subdir="tts",
    note="离线中文 TTS 121 MB —— 陪伴通话的出声端",
),
```

（`subdir="tts"` → `algorithm/models/tts/vits-zh-hf-fanchen-C/`，与 `algorithm/models/{asr,punct}/` 并列；`algorithm/models/` 已在 `.gitignore` 中。）

### 6.12 `algorithm/shared/dataclasses.py`（452 行）

```python
class AsrEngine(str, Enum):        # 第 290 行
    PARAFORMER = "sherpa-onnx-paraformer"
    ZIPFORMER_STREAMING = "sherpa-onnx-streaming-zipformer"
    UNAVAILABLE = "unavailable"

class AsrEventType(str, Enum):     # 第 302 行
    PARTIAL = "partial"; FINAL = "final"

@dataclass
class AsrResult:                   # 第 308 行
    text: str; engine: AsrEngine = AsrEngine.PARAFORMER
    duration_ms: float = 0.0; latency_ms: float = 0.0; error: str = ""
    @property is_empty -> bool ; @property ok -> bool

@dataclass
class StreamingAsrUpdate:          # 第 338 行
    event: AsrEventType; text: str
    elapsed_ms: float = 0.0; latency_ms: float = 0.0
    reason: str = ""; engine: AsrEngine = AsrEngine.ZIPFORMER_STREAMING
    refined: bool = False
    @property is_final -> bool ; @property is_empty -> bool

class ChatStreamEventType(str, Enum):   # 第 381 行
    # META / DELTA / REVISE / DONE / ERROR
```

**注意第 382-387 行的契约注释**：`ChatStreamEventType` 的取值在**三个地方**同时出现，改动必须三处同步：①`algorithm/api/intervention.py`（产出）②`server/src/services/algorithm-bridge.ts`（原样透传）③`client/src/services/index.ts`（消费）。**通话的语音事件（如果有新的事件类型）也必须遵守这条三处同步规则** —— 更保守的做法是**复用** `ChatStreamEventType`，把音频作为二进制帧走另一条 WS，不要往 SSE 里塞 base64 音频。

按 AGENTS.md，「所有模块间通信必须通过 `shared/dataclasses.py` 中的共享数据类」+「新增数据类时同步更新 `shared/__init__.py` 的导出列表」+「`shared/dataclasses.py` 的每次修改必须伴随单元测试」。所以通话若引入新的 `VoiceCallState` 之类，必须走这套流程。

---

## 7. 未验证 / 有风险的点

**这一节列举本报告中所有我没能证实的东西。请勿把其中任何一条当成已验证事实。**

### 7.1 网络与可达性

| # | 说法 | 状态 |
|---|---|---|
| 1 | `hf-mirror.com` 对**大文件**（LFS）在浏览器 UA 下可完整下载 | **[未验证]** 我只读到 `Content-Length`（HTTP 200 + 大小），**没有真实下载过任何一个 TTS 模型**。父会话已用同一镜像成功下载 ASR 模型，所以可行性高，但 TTS 仓库的所有文件（尤其 `lexicon.txt` 这种小文件）是否都能过 403 拦截，我没有逐个试。`lexicon.txt` 与 `tokens.txt` 的 HEAD 返回 `Content-Type: text/plain` 但 `Content-Length` 缺失（chunked）—— 这是**异常信号**，建议下载时校验大小。 |
| 2 | `matcha-icefall-zh-baker` 的 vocoder 完全无法获取 | **[未验证]** 我验证了 ①仓库无 vocoder ②`csukuangfj/vocos-22khz-univ`、`csukuangfj/hifigan_v2`、`csukuangfj/sherpa-onnx-vocoder-hifigan` 在 hf-mirror 均返回 **401**。但我**没有穷举**所有可能的镜像仓库名，也没有试 ModelScope 或其他国内镜像。**结论是"在我检查过的路径下不可达"，不是"绝对不存在"。** |
| 3 | `vits-zh-aishell3` 的采样率 | **[未验证]** 官方 vits.rst 概览表写 `vits-model-aishell3` 为 8000 Hz / 116 MB，而官方 RTF 表写 "aishell3" 为 30 MB / RTF 0.365 —— **两个官方表格自相矛盾**，且我实测的 HF 仓库 `csukuangfj/vits-zh-aishell3` 是 fp32 121.4 MB / int8 39.9 MB，又是第三个数字。我未能抓到该仓库的采样率说明。**不要依赖这一行做决策。** |
| 4 | `vits-piper-zh_CN-huayan-x_low`（20.6 MB）是否可用 | **[未验证]** 它只有 7 个文件、**没有 espeak-ng-data**，却有一个 `lexicon.txt`（而且里面还混进了一个本不该出现的 `export-onnx-zh-hf-fanchen-models.py`）。这个仓库看起来是**生成脚本跑歪了**的产物。**不建议使用。** |
| 5 | GitHub 上 `k2-fsa.github.io` 文档站可达，但 `raw.githubusercontent.com` 时通时断 | **[实测]**：同一个 `raw.githubusercontent.com/.../vits.rst` 第一次 timeout、第二次成功；`rtf.rst` 与 `matcha.rst` 多次失败。所以**任何依赖 raw.githubusercontent 的自动化都不可靠**。 |

### 7.2 模型能力与性能（全部未实测）

| # | 说法 | 状态 |
|---|---|---|
| 6 | fanchen-C 本机 CPU RTF ≈ 0.15–0.25 | **[估计]** 由 Pi4 的 1.600（4 线程）按"A72 单核 ≈ 现代桌面核 1/6~1/10"外推。**这是我全篇最重要的未验证数字**，直接决定 Path A 成不成立。 |
| 7 | TTS 首 callback 延迟 100–300 ms | **[估计]** 完全没有依据可查；sherpa 文档只给总 RTF，不给首块延迟。 |
| 8 | 任何模型的**中文音质** | **[未验证]** 我没听过任何一个模型。fanchen-C 优于 piper-huayan 是**架构推断**（自带中文词典 vs espeak-ng cmn 音素化），不是听感结论。**Demo 前必须试听。** |
| 9 | piper `zh_CN-huayan-medium` 的许可证 | **[未验证]** MODEL_CARD 明确写 `License: Unknown`，数据集来自 `https://github.com/PlayVoice/HuaYan_TTS`。**用于竞赛 Demo 前应确认授权**，这一条有实际风险。 |
| 10 | aishell3 音色的授权来源 | **[未验证]** README 显示来自 `jackyqs/vits-aishell3-175-chinese`。 |
| 11 | fanchen-C 的 `lexicon.txt` 在 hf-mirror 上的实际大小 | **[实测-部分]** HEAD 返回 chunked 无 `Content-Length`；官方文档说 2.4 MB。**未下载验证。** |
| 12 | `OfflineTts` 多线程/并发安全性 | **[未验证]** 我读了 `api/asr.py` 与 `punctuation.py` 里"CT-Transformer 会话不保证并发安全 → 加锁"的先例，据此**建议**单线程串行，但**没有验证** `OfflineTts` 是否真的不能并发。 |

### 7.3 DashScope / 云端（已实测的部分与未实测的部分要分清）

| # | 说法 | 状态 |
|---|---|---|
| 13 | `wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-omni-flash-realtime` 可用，且 `session.created` 内容如上 | **✅ [实测]** 握手成功，返回内容逐字引用自响应。**注意**：我只做到"收到 `session.created`"就关闭了，**没有发过一个音频字节、没有触发过一次响应、没有收到过一个 `response.audio.delta`**。 |
| 14 | `qwen-omni-turbo-realtime` 不存在 | **✅ [实测]** 它不在 `GET /compatible-mode/v1/models` 返回的列表里。**但**该列表可能分页/可能不含所有模型；阿里云文档里确实提到过 `qwen-omni-turbo-realtime` 的默认音色（`Chelsie`）与温度。「拿不到」≠「不存在」。 |
| 15 | 端到端首声延迟 ≈ 0.85 s | **[估计]** 完全没测。 |
| 16 | `semantic_vad` 的实际效果 | **[未验证]** 只从文档读到它的存在与适用范围（`qwen3.8-omni-flash-realtime` + `qwen3.5-omni-realtime` 系列）。 |
| 17 | `idle_timeout_ms` 的实际行为 | **[未验证]** 只从文档读到（`[5000, 30000]`，仅 `qwen3.5-omni-plus/flash-realtime`）。 |
| 18 | Realtime 的计费标准 | **[未验证]** 我没有查到本账号的价格，也没有做过任何计费调用。文档只说按 token（音频 token 折算）。 |
| 19 | 本报告列出的所有 `qwen3*` 模型 id 的**长期可用性** | **[未验证]** 这些 id 来自一次快照。注意到列表里带日期的快照版本（如 `-2026-03-15`）与不带日期的别名并存；**生产代码应固定快照版本**，否则别名指向的模型会漂移。 |
| 20 | 阿里云文档页面上的「Last Updated: Sep 18, 2026 / Sep 24, 2026」日期 | **[未验证]** 页面显示的最后更新日期晚于我的常识预期。我按"页面原文如此"引用，**没有独立核实该日期**。 |

### 7.4 浏览器 / AEC / 打断

| # | 说法 | 状态 |
|---|---|---|
| 21 | Chromium AEC3 收敛需 200–500 ms | **[估计]** 工程共识，我**没有在本机实测**。 |
| 22 | AEC 在外接音箱/蓝牙/大音量下失效 | **[估计]** 由公开 issue 佐证（Chromium「*echoCancellation='all' on macOS is unpredictable*」、WebKit bug 320320/320452 关于 VPIO 无条件实例化打断麦克风），但**没有一条是针对"Windows 11 + Edge/Chrome + 本机扬声器 + 本地 TTS"这个具体组合的实测**。 |
| 23 | `AudioBufferSourceNode.stop()` 端到端 < 50 ms | **[估计]** 按渲染量子推算 + 主线程调度余量，**未实测**。 |
| 24 | `speechSynthesis.cancel()` 在 Windows Edge 上的具体行为（缓冲区剩余音频、`cancel()` 后立即 `speak()` 的竞态、`pause()` 后 `cancel()` 卡死） | **[未验证]** 我知道 API 存在与语义，但 Edge 的具体缺陷行为我没有实测，也**没能抓到权威页面**（Google/MDN 在本网络不可达）。 |
| 25 | 「文本级自回声过滤」（§2.3 方案 D）的实际拦截率 | **[估计]** 我**没有实测**过"AI 的话被 ASR 识别成什么"。回声被识别出的文本与原文的编辑距离分布是未测量的 —— 阈值（≥6 字连续子串）是拍的。**这是 §2 里最需要先做实验的一个数。** |
| 26 | 400 ms 半双工窗口是否够覆盖 AEC 收敛 | **[估计]** 与第 21 条同源。 |

### 7.5 端点与轮次（§3）

| # | 说法 | 状态 |
|---|---|---|
| 27 | `rule2 = 0.40 s` 的实际过早截断率 | **[未验证]** 完全没测。**建议先在通话模式做影子测试**（不真正打断，只记录"如果按 0.4 s 切会切在哪里"），跑 20–30 轮真人语音再定。 |
| 28 | 语义端点启发式（结尾语气词 / 连接词）的有效性 | **[估计]** 纯经验设计，未验证。 |
| 29 | 本机是否有 Silero VAD 模型文件 | **[未验证]** 我只确认了 `sherpa_onnx.SileroVadConfig` 这个类**存在**（在父会话给出的已装能力清单里），**没有确认** `algorithm/models/` 下有对应的 `.onnx`。 |
| 30 | 投机启动 + 取消窗口能省 300–500 ms | **[估计]** 未实现、未测。 |
| 31 | `qwen-turbo` 首 token 延迟 300–700 ms | **[估计]** 未测。本机到 DashScope 的 RTT 我也没测。 |

### 7.6 延迟预算（§4）

**§4 两张表里的每一个 `[估计]` 都是估计。** 具体而言，**没有任何一条端到端链路被真正跑通过**：没有下载 TTS 模型、没有合成过一句话、没有让浏览器播过一个字节的 AI 语音、没有向 Realtime API 发过音频。表里的"中位 ≈ 1.6 s / ≈ 0.85 s"是**给现场期望管理用的假设值，不是预测值**。

### 7.7 安全设计（§5）

| # | 说法 | 状态 |
|---|---|---|
| 32 | 复用 `streaming_allowed` 做"语音模式硬禁用"满足 AGENTS.md 的"复用既有规则不需审核" | **[估计]** 这是我的解读。`streaming_allowed` 是**既有的、已上线的**风险判定产物，我只是把它接到新通道上。但**"把既有阈值用到新场景"算不算"新增升级条件"**，边界由人来判 —— **建议在做之前问一句**。 |
| 33 | 给 `_StreamingSafetyGate` 加 `max_unit_chars` 参数或改用独立切句器不改红线行为 | **[估计]** 同上。推荐"独立切句器 + 调用公有 `screen()`"，因为**完全不动安全组件**。 |
| 34 | 首句纯共情约束（改 prompt 文案） | **[未验证]** 这是**安全相关的 prompt 变更**，按 AGENTS.md 需人工审核。我**没有**改动它。 |
| 35 | 危机时那句固定话术的措辞 | **[未验证]** 我**没有拟**这条文案（拟一条危机时刻的用户可见文案，超出了调研报告的边界）。需要人来写并审核。 |
| 36 | 「不要用 TTS 读热线号码」 | **[估计]** TTS 读数字出错是常见现象，但我**没有实测** sherpa VITS 对 `400-161-9995` 的读法（这需要先有模型）。**不过即使测了，这个建议也应该照做 —— 危机场景下不值得赌。** |
| 37 | 药物词法轴缺口 | **✅ [实测]** `_StreamingSafetyGate` 的 docstring（`intervention.py:1042-1044`）**原文就写了**这个缺口。**这不是我的推测。** |

### 7.8 工程约束

| # | 说法 | 状态 |
|---|---|---|
| 38 | 现有环境"无 pip，安装须用 uv pip install" | **[实测-引用]** 父会话给定；我**没有**通过 `pip` 或 `uv` 安装任何东西，也没验证 `pip` 是否真的不可用。 |
| 39 | `algorithm/models/` 下确实没有 TTS / VAD 模型 | **[估计]** 我读了 `download_asr_model.py` 的 `PRESETS`（只有 asr / punct 三个）与 `.gitignore` 的说明，**没有**列过 `algorithm/models/` 的实际目录内容。若有人手工放过模型，我的"没有 TTS"结论需要更正。 |
| 40 | 通话功能会与现有模态采集争 CPU | **[未验证]** 没做压测。本机是 CPU-only，同时跑 16 kHz 流式 ASR + VITS 合成 + 声学分析（`useVoiceAnalysis` 每帧算自相关基频估计，`useVoiceAnalysis.ts:98-117` 是 O(n²) 的双层循环）+ 可能的摄像头模态，**CPU 争用是一个真实风险**。建议在 §4 的 go/no-go benchmark 里**同时**打开现有全部模态一起测。 |

---

## 附：如果只能记住五件事

1. **TTS 用 `sherpa-onnx` 的 `OfflineTts` + `vits-zh-hf-fanchen-C`（121.3 MB，hf-mirror 可达 [实测]）**，用 `generate(..., callback=...)` 边合成边推 PCM 伪造流式；**先跑 RTF benchmark，> 0.5 就换 `vits-piper-zh_CN-huayan-medium`（63.2 MB，快约 4.5 倍，但许可证 Unknown）**。matcha 因为 vocoder 不可达而**被排除**（不是因为它不好）。
2. **没有 `OnlineTts` [实测]**。低延迟靠"句子级流水线 + callback 分块"，没有别的魔法。
3. **打断不要指望浏览器 AEC**。用「400 ms 半双工窗口 + AEC + 文本级自回声过滤」这个组合，并且**打断走 partial 不走 final**。`speechSynthesis` 不是可选项：它拿不到音频样本、做不了 AEC 参考、还依赖在线语音。
4. **`wss://dashscope.aliyuncs.com/api-ws/v1/realtime` 已经握手成功 [实测]**，`qwen3-omni-flash-realtime` 支持 `interrupt_response`/`semantic_vad`/`idle_timeout_ms`，能把首声中位从 ~1.6 s 压到 ~0.85 s **[估计]**，代价是**未成年人音频出设备 + 计费** —— 这是产品决策。
5. **安全上仓库已经有 `_StreamingSafetyGate`（`intervention.py:1027`），逐句过闸直接复用；"高风险硬禁用"直接复用 `streaming_allowed`（`intervention.py:1181`）。药物词法轴是既有缺口（docstring 原文承认），补它需人工审核。危机时不要用 TTS 念热线号码。**
