# 心灵花园 —— 青少年 AI 心理健康支持系统（源代码 + 最小模型集）

> 第八届全球校园人工智能算法精英大赛 · 算法主题赛「AI + 心理健康」
> 团队编号：**AIC-2026-56697579**
> 本包为 2026-11-18 的工作区源代码快照，并随包附带**运行所必需的最小模型/工件集**（约 2.1 GB），
> 不含数据集、`.env` 与任何密钥。

本包对应提交材料中的「完整代码包」。作品的功能说明、实验数据与结论口径以《技术方案 v2》PDF 与答辩 PPT/视频为准；本文件只说明**代码怎么读、怎么跑、模型放在哪**。

---

## 1. 作品与技术栈

面向青少年的 AI 心理健康支持系统。平台以「感知 → 评估 → 干预 → 升级」四层为业务骨架，另设安全审计、隐私保护、联邦学习三个支撑体系：

- **多模态感知**：文本情绪、语音情绪（ASR + 声学特征）、面部表情（端侧 ONNX）、行为与节律信号、跨模态融合。
- **评估**：标准化量表、风险分级、趋势与随访预测、结构化报告。
- **干预**：CBT 式对话、支持策略路由、情境条件化的认知解读（C2D2 8 类）、持续陪伴。
- **安全**：生成后审计（Safety Loop）、危机分级与转介三出口、审计日志哈希链。
- **隐私**：原始音视频不出设备，仅上传脱敏特征摘要；敏感字段加密存储。

| 端 | 技术 | 目录 | 默认端口 |
|---|---|---|---|
| 前端 | React 18 + TypeScript + Vite + antd | `client/` | 5173 |
| 后端 | Node.js + Express + Prisma + PostgreSQL + Socket.IO | `server/` | 3000 |
| 算法 | Python 3.10+ + FastAPI + PyTorch | `algorithm/` | 8001 |
| 语音合成（可选） | VITS（随包）/ Qwen3 / CosyVoice3 | `algorithm/tts/` | 8002 |
| 数据库 | PostgreSQL（Docker 容器 `mh-postgres`） | — | 5432 |

## 2. 目录说明

```
.
├── README.md                       # 本文件
├── 00-代码包说明.md                 # 提交用：包内/包外清单、校验方式、合规声明
├── MANIFEST-SHA256.txt             # 全部文件的 SHA-256（解压后可校验）
├── start-package.cmd               # 一键启动（Windows 双击即可）
├── start-package.ps1               #   同上，设置模型路径后调用 start-all.mjs
├── package.json / server.js        # 根脚本与启动器
├── start-all.mjs                   # 五端编排（db → tts → algo → api → web）
├── AGENTS.md                       # 项目约定：数据类、随机种子、安全红线
├── .env.example                    # 环境变量模板（复制为 .env 后填写）
├── algorithm/                      # 算法服务（FastAPI）
│   ├── main.py                     #   入口，启动时加载根目录 .env
│   ├── api/                        #   17 组 REST 路由
│   ├── perception/ assessment/ intervention/ escalation/
│   ├── audit/ privacy/ federated/ shared/
│   ├── config/                     #   YAML 规则与阈值
│   ├── models/                     #   ★ 随包：ASR / 标点 / VITS 默认音色
│   ├── artifacts/                  #   ★ 随包：认知扭曲权重、场景索引、融合/质量/策略工件
│   ├── experiments/                #   研究轮次脚本（可复算）
│   ├── tests/                      #   pytest 用例（84 个文件）
│   └── requirements.txt
├── external_models/                # ★ 随包：CLIP / BGE 编码器（需由启动器设环境变量）
├── server/                         # Node 后端（routes 18 / services 27 / Prisma）
├── client/                         # React 前端
│   └── public/models/              #   ★ 随包：人脸关键点、8 类表情 ONNX、眨眼模型
├── scripts/                        # 端到端验证 + 研究复现脚本（149 个）
└── docs/                           # 工程设计与审计记录
```

## 3. 环境要求

- Node.js ≥ 18（推荐 20/22），npm ≥ 9
- **Python 3.10 或 3.11**（`requirements.txt` 的 `mediapipe==0.10.14` / `torch==2.4.1` 在 3.12+ 无可用 wheel）
- Docker Desktop（首次启动自动拉取并创建 PostgreSQL 容器）
- 磁盘：解压后约 2.1 GB，另需 ≥ 2 GB 放依赖与数据库
- 内存：Web + 算法基础功能约 4 GB；加载 VITS/CLIP/BGE 建议 ≥ 12 GB 可用内存

## 4. 快速开始

> ⚠️ **解压到纯英文路径**（如 `D:\MindGarden`）。包内已把根目录命名为 ASCII，但若上层目录含中文，
> TTS 的第三方库 `kaldifst` 打不开 `.fst` 文件，`/api/v1/tts/status` 会报 `available:false`（其余功能不受影响）。
> 已经解压到中文路径的话，把整个目录移到 `D:\MindGarden` 即可，无需重新解压。

### 4.1 一键（推荐）

```powershell
# 双击 start-package.cmd，或在包根目录执行：
powershell -NoProfile -ExecutionPolicy Bypass -File .\start-package.ps1
```

`start-package.ps1` 会自动完成四件事：

1. 若 `.env` 不存在，从 `.env.example` 复制一份（算法端读取）；
2. 若 `server/.env` 不存在，从 `server/.env.example` 复制一份（**Node 端只读这个文件**）；
3. 把 `ROUND3_VLM_ENCODER` 指向 `external_models/chinese-clip-vit-base-patch16`、`SCENE_EMBED_MODEL` 指向 `external_models/bge-base-zh-v1.5`；
4. 若包根目录有 `.venv`，设置 `AIC_PYTHON` / `ALGORITHM_PYTHON`，然后调用 `start-all.mjs`。

> ⚠️ 根 `.env` 与 `server/.env` 是**两份**配置：算法服务读根 `.env`，Node 后端（`dotenv`，cwd=`server/`）只读 `server/.env`。需要真实大模型回复时，请把**同一把** `DASHSCOPE_API_KEY` 填进这两个文件，否则聊天页会出现「暂时无法回应」的失败卡片。

### 4.2 手动

```powershell
Copy-Item .env.example .env
Copy-Item server\.env.example server\.env
notepad .env              # 填 DASHSCOPE_API_KEY（不填则算法侧走模板兜底）
notepad server\.env       # 填同一把 key；不填则 Node 桥接 401

cd server
npm install
npx prisma migrate deploy       # Prisma CLI 会自动读取 server\.env
$env:DATABASE_URL = "postgresql://postgres:postgres123@localhost:5432/mental_health?schema=public"
npx tsx prisma\seed.ts          # seed 是普通 tsx 脚本，不会自动读 .env，需显式给 DATABASE_URL
cd ..

cd client; npm install; cd ..

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r algorithm\requirements.txt

$env:ROUND3_VLM_ENCODER = "$PWD\external_models\chinese-clip-vit-base-patch16"
$env:SCENE_EMBED_MODEL  = "$PWD\external_models\bge-base-zh-v1.5"
$env:AIC_PYTHON         = "$PWD\.venv\Scripts\python.exe"
$env:ALGORITHM_PYTHON   = "$PWD\.venv\Scripts\python.exe"

node start-all.mjs
```

启动后：前端 <http://127.0.0.1:5173>，后端 <http://127.0.0.1:3000>，算法 <http://127.0.0.1:8001/api/v1>。

> `start-all.mjs --no-tts` 可跳过 8002；`--status` 只打印五端状态。
> 换机器时若解压路径不是「往上三级」，务必用上面的两个 `*_PYTHON` 环境变量指定解释器。

## 5. 包内模型清单（已附带，无需再下载）

| 路径 | 体积 | 用途 |
|---|---:|---|
| `algorithm/models/asr/sherpa-onnx-paraformer-zh-small-2024-03-09` | 78 MB | 语音定稿识别 |
| `algorithm/models/asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23` | 25 MB | 流式语音识别 |
| `algorithm/models/punct/sherpa-onnx-punct-ct-transformer-...` | 285 MB | ASR 标点恢复 |
| `algorithm/models/tts/vits-zh-hf-fanchen-C` | 131 MB | 默认 TTS 音色（`TTS_BACKEND=vits`） |
| `algorithm/artifacts/cognitive_distortion_ckpt` | 391 MB | 认知扭曲 8 分类（C2D2） |
| `algorithm/artifacts/scene_index` | 23 MB | 情境索引（场景检索） |
| `algorithm/artifacts/{innovation_fusion,quality_head,round2_fusion,round3_fusion,round3_strategy,round2_forecast}` | ~2 MB | 图文专家 / 画质头 / 融合 / 策略 / 预测工件 |
| `external_models/chinese-clip-vit-base-patch16` | 718 MB | 图文与知识 MoE 的视觉-文本编码器 |
| `external_models/bge-base-zh-v1.5` | 391 MB | 场景检索查询编码器 |
| `client/public/models/face_emotion_8cls_hsemotion.onnx` | 16 MB | 面部 8 类表情（折叠为三分类） |
| `client/public/models/face_landmarker.task` | 3.6 MB | MediaPipe 人脸关键点（Apache-2.0） |
| `client/public/models/open_closed_eye.onnx` | 45 KB | 眨眼开闭判定（Apache-2.0） |

**仍需自备（本包未含）**：训练语料（C2D2/CH-SIMS 等，按原许可自行申请）、Qwen3-TTS 音色（4.3 GB，可选）、CosyVoice3 权重（12.2 GB，可选）、`Face/` 目录下的表情训练权重（270 MB，仅训练用）、三分类旧模型 `face_emotion_3cls.onnx`（90 MB，线上已改用 8 类）。

## 6. 自检与测试

```powershell
# 按文件分批跑，避免一次性加载大权重
$env:PYTHONPATH = "algorithm"
.\.venv\Scripts\python.exe -m pytest algorithm\tests\test_perception.py -q
.\.venv\Scripts\python.exe -m pytest algorithm\tests\test_intervention.py -q

# 端到端验证（需先启动服务）
node scripts\verify_chat_send.mjs
node scripts\verify_voice_send.mjs
node scripts\verify_quality_startup.mjs
```

## 7. 阅读约定

- `docs/` 是开发过程中的设计、审计与实验记录，**保留原始状态**，个别结论可能已被后续轮次修订（例如审计轴数从五轴扩展到六轴）。对外口径以最终提交的《技术方案 v2》为准。
- `algorithm/experiments/` 保留 19 轮研究脚本与指标 JSON，便于复算；脚本中的本机绝对路径是原始记录。
- `start-package.ps1` 会覆盖 `ROUND3_VLM_ENCODER` / `SCENE_EMBED_MODEL` 两个环境变量；如你已有自己的模型目录，请改脚本或直接设置后调用 `start-all.mjs`。

## 8. 合规声明

- 包内不含 `.env`、API Key、`.pem`/`.key` 等凭据；示例文件中的值均为占位符。
- 包内不含参赛团队所在学校名称、指导教师与个人身份信息（已全量检索）。
- 包内不含真实用户原始文本、音频、图像或问卷回收记录。
- 面部与语音均为**端侧推理**；上传到服务端的只有脱敏后的特征摘要。
- 所列第三方模型仅用于本次赛事评审与复现，许可与出处见各目录内 `README`/`NOTICE` 及 `external_models/README.md`。
- 系统定位为「初步识别、即时支持、情绪疏导」，不做诊断、不提供用药建议；高风险一律走升级/转介流程。
