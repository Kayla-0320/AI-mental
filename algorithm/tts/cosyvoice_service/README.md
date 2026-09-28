# CosyVoice3 TTS 微服务

零样本音色克隆的中文语音合成服务，供算法服务通过 HTTP 流式调用。

**为什么是独立服务**：CosyVoice3 需要 `torch==2.7.0+cu128` + `numpy<2` +
`transformers==4.51.3`，与算法服务的 `torch==2.11` + `numpy==2.5`
**直接冲突**，放不进同一个解释器。所以模型跑在这里，算法侧只当客户端
（`../../cosyvoice_engine.py`）。

---

## 一、目录约定

服务期望 `COSYVOICE_ROOT` 指向这样一个目录：

```
$COSYVOICE_ROOT/
├── .venv/                                  # 本服务专用解释器（见第二节）
├── CosyVoice/                              # 官方仓库（含 third_party/Matcha-TTS）
│   └── cosyvoice/cli/cosyvoice.py
└── pretrained_models/
    └── Fun-CosyVoice3-0.5B/                # 模型权重，约 5.05 GB
        ├── cosyvoice3.yaml
        ├── llm.pt  flow.pt  hift.pt
        ├── speech_tokenizer_v3.onnx
        ├── campplus.onnx
        └── CosyVoice-BlankEN/
```

参考音**不入这个目录** —— 它是"音色"这一交付物的定义，所以放在仓库内
（`prompt_qwen3_voice.wav`），保证可复现、可版本管理。

---

## 二、从零搭起来

### 1. 取官方仓库与模型

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE_ROOT/CosyVoice"
```

模型走 ModelScope（HuggingFace 国内不稳）：

```python
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir=r'$COSYVOICE_ROOT/pretrained_models/Fun-CosyVoice3-0.5B')
```

### 2. 建独立 venv

```bash
cd "$COSYVOICE_ROOT"
uv venv --python 3.12 .venv

# torch 必须 cu128：RTX 50 系是 sm_120，cu121 及更早没有对应 kernel，
# 会在第一次 GPU 运算时报 "no kernel image is available"
uv pip install --python .venv/Scripts/python.exe \
  "torch==2.7.0" "torchaudio==2.7.0" --index-url https://download.pytorch.org/whl/cu128

# 其余推理依赖（顺序有讲究：先 setuptools，否则 jieba 等会因 build 隔离失败）
uv pip install --python .venv/Scripts/python.exe --no-build-isolation \
  setuptools wheel cython packaging \
  numpy==1.26.4 HyperPyYAML==1.2.3 omegaconf==2.3.0 conformer==0.3.2 \
  diffusers==0.29.0 librosa==0.10.2 soundfile x-transformers \
  transformers==4.51.3 onnxruntime==1.18.0 inflect rich networkx pyworld \
  hydra-core==1.3.2 lightning==2.2.4 openai-whisper==20231117 tiktoken \
  wetext==0.0.4 gdown wget protobuf==4.25.3 fastapi uvicorn matplotlib
```

> ⚠️ `openai-whisper` 与 `hydra-core` 是**必须**的：前者被
> `cosyvoice/cli/frontend.py` 顶层 import，后者被 Matcha-TTS 的
> `matcha/utils/instantiators.py` import。缺任一个都是 ImportError。
>
> ⚠️ `grpcio==1.57.0` 在 Python 3.12 上**没有预编译 wheel**，会走源码编译并失败。
> 本服务不需要 gRPC（只用 FastAPI HTTP），所以**不装**。

### 3. 起服务

```bash
cd "AI-mental-main (2)/AI-mental-main/AI-mental-main/algorithm/tts/cosyvoice_service"
set COSYVOICE_ROOT=D:\path\to\cosyvoice
%COSYVOICE_ROOT%\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8002
```

自检：

```bash
curl http://127.0.0.1:8002/health
# {"available":true,"engine":"cosyvoice3","sample_rate":24000,"speed":1.2,
#  "prompt":"prompt_qwen3_voice.wav","error":""}
```

---

## 三、算法侧怎么接

```bash
set TTS_BACKEND=cosyvoice
set COSYVOICE_API_URL=http://127.0.0.1:8002   # 默认值，可省略
set COSYVOICE_SPEED=1.2                        # 默认值，可省略
```

然后 `GET /api/v1/tts/status` 应报出：

```json
{"available": true, "engine": "cosyvoice3", "sample_rate": 24000, "speed": 1.2, ...}
```

> ⚠️ **`TTS_BACKEND` 拼错会静默回落到 vits**（只打一条 warning）。
> 所以这个端点报的 `engine` 才是事实 —— 换引擎后务必核对它。

---

## 四、环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `COSYVOICE_ROOT` | `D:\Develop\AIC\_tts_candidates\cosyvoice` | 权重与仓库所在根目录 |
| `COSYVOICE_MODEL` | `Fun-CosyVoice3-0.5B` | `pretrained_models/` 下的目录名 |
| `COSYVOICE_PROMPT_WAV` | 仓库内 `prompt_qwen3_voice.wav` | 参考音路径 |
| `COSYVOICE_SPEED` | `1.2` | 默认语速，逐块保音高加速 |
| `COSYVOICE_API_URL` | `http://127.0.0.1:8002` | **算法侧**读，指向本服务 |

---

## 五、实测数据与设计取舍

### 首块延迟是这门技术的核心指标，它有四个独立来源

按影响从大到小（RTX 5060 Laptop 8 GB，33 字文本，目标 1.2×）：

| 来源 | 影响 | 处置 |
|---|---|---|
| **首次调用的一次性开销** | 首块 +2.8 s | **启动预热**（`_warmup`） |
| **产出块粒度**（`token_hop_len`） | 默认 25 vs 10 差 1.6 s | 临时设成 10（`STREAM_FIRST_HOP`） |
| **攒块伸缩** | 攒到 2 s 要多等 1.8 s | **不攒，逐块伸缩** |
| 网络/HTTP | ~8 ms | 忽略 |

逐步实测（同一句话）：

| 配置 | 首块 | 说明 |
|---|---|---|
| 默认 hop=25 + 逐块伸缩 + 未预热 | 4.39 s | 初版 |
| hop=25 + 攒 2 s + 未预热 | 8.09 s | 攒块最差 |
| hop=10 + 逐块伸缩 + 未预热 | 4.20 s | 只解决了粒度问题 |
| **hop=10 + 逐块伸缩 + 预热** | **1.90 s** | 采用 |

预热前的首块内部拆解（这是定位的关键证据）：:

    cvsvc|first_block|model_ms=2698|stretch_ms=1692|samples=5760

预热后：:

    cvsvc|first_block|model_ms=1875|stretch_ms=3|samples=5760

`stretch_ms` 从 1692 ms 降到 3 ms —— 第一次调用 `librosa.effects.time_stretch`
要初始化 STFT/相位声码缓存。**不做预热，每个进程的第一句话都会慢一倍。**

### 语速与音色

当前实测（预热后，端到端经算法侧引擎）：**首块 1.90 s，音频 9.33 s，
5.57 字/秒，加速比 1.264×**。同文本 1.0× 原速为 11.80 s / 4.41 字/秒。

保音高由 `pitch_safe_resample` 保证（相位声码器 + 精确长度校正）。
F0 中位在 hop=25/10/5 下恒为 **218.2 Hz**，与原速 210.5 Hz 相差 0.6 半音 ——
音色未被改变。**不要**改用 `resample_poly` 直接重采样：它快得多（0.3 ms），
但会把 F0 抬高到 1.19×（实测 328.8 Hz），听感变成"快放"。

### 为什么必须非流式才能用 CosyVoice 自己的 `speed`

`cosyvoice/cli/model.py` 有硬断言：

```python
if speed != 1.0:
    assert token_offset == 0 and finalize is True, 'speed change only support non-stream inference mode'
```

流式下 `token_offset` 已不为 0，speed 几乎不生效（实测 1.45 只快 4%）。
而非流式虽能真正提速，却要等整段合成完（首块 8.39 s）才出声 ——
所以本服务**不用**它，改用"流式出声 + 逐块保音高加速"。

### 显存

半精度加载后约 **1.65 GiB**（`fp16=True` 只开 autocast **不省显存**，
必须自己"CPU 读盘 → 转半精度 → 上卡"，见 `server.py:_install_half_loader`）。

### 两个必须一起做的代码修改

1. **半精度加载**：默认路径是 fp32 整份 `.to("cuda")`（llm 1.9 G + flow 1.3 G），
   8 GB 卡会 OOM。
2. **spk 投影层 dtype 钉住**：`flow.py` 的 `F.normalize` 在 autocast 下产出 fp32，
   而 `flow_matching.py` 用 `spks.dtype` 分配 `spks_in`，半精度下抛
   `RuntimeError: expand(torch.cuda.HalfTensor{[0, 80]}, size=[80])`。

   ⚠️ 这个异常发生在 `llm_job` **子线程**里会被 threading 吞掉，主循环永远等不到
   `llm_end_dict` —— 表现为"整机 CPU/GPU 都 0% 的假死"，**没有任何 Python 回溯**。
   排查时不要往"卡死/死锁"方向找，先看线程异常。

### 两个排查时容易走错方向的坑

1. **`token_hop_len` 是引擎单例上的状态**，改完必须还原（`synthesize_chunks`
   用 `try/finally` 保证）。不还原会污染后续请求，且下一个请求会把 hop 越改越小。
2. **客户端看到的"成批到达"未必是缓冲问题**。实测首块产出与发送时刻完全一致
   （`yield 6505 ms` → `send 6505 ms`），只是服务端一块就有 2.7 s 音频，
   被客户端的 8192 字节读取切成 15 个瞬间到达的小块。
   判断时应当**对照服务端的产出时刻**，而不是只看客户端间隔。

### 换音色

替换 `prompt_qwen3_voice.wav`，并**同步修改** `server.py` 的 `PROMPT_TEXT`
使其与实际录音逐字一致 —— 不一致会明显损害克隆质量。文本必须以
`You are a helpful assistant.<|endofprompt|>` 开头，缺这个标记 `llm.py` 会直接抛断言。

参考音取 3–6 秒、发音清晰的一段最稳。当前的参考音是从一段 Qwen3-TTS 生成音频里
裁出的前两句（0–5.6 s / 28 字），句界由 100 ms 窗口能量包络测得。
