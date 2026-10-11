# 三分类表情模型接入多模态（happy / neutral / sad）

> 结论：**打开多模态开关 = 眨眼检测与本机表情识别同时开始工作**。
> 表情读数只展示与上报，**不参与风险评分**（`riskScore`/`riskLevel` 链路未改动）。
> 接入日期：2026-10-10 · 作者：集成改动（含验证证据）
>
> **2026-10-10 追加（模型不再写死，且已换模型）**：分类器已改成按 `FaceEmotionModelSpec` 加载，
> 换模型 = 改一个对象而不是改代码；**线上模型已换成 HSEmotion `enet_b0_8_best_vgaf`**
> （8 类 → 折叠三分类，224px，ImageNet 归一化，15.3 MB，Apache-2.0）。
> 换的理由是实测：在真实成人人脸（RAF-DB test, n=794）上，旧模型把 **56.9% 的笑脸**
> 读成悲伤、92.5% 的中性脸读成悲伤、sad AUC 仅 0.630；新模型分别是 **0.8% / 10.8% / 0.965**。
> 代价是本仓库数据集上的数字由 95.8% 降到 82.3%（该数据集的 sad = 婴幼儿嚎哭）。
> 详见 `docs/fer_external_model_integration.md`（候选清单、许可、跨域 AUC、验收记录、遗留项）。

## 1. 数据流（一条流、一帧多用）

```
getUserMedia（mediaStreamManager 唯一持有）
        │
        ▼
useFacialAnalysis  每 33ms 一拍（眨眼通道 30/15/10 自适应）
  MediaPipe FaceLandmarker.detectForVideo(video)
        │
        ├── 每拍：blendshape → blinkScore / 头姿 / EAR
        │        └─ onFrameSignal ──► useEyeTracking（眨眼状态机，主链路）
        │
        └── 每 ~100ms（指标拍，10fps）
                 ├─ AU → emotionMapping（原有启发式规则，7 维，进融合）
                 ├─ onEyePatchFrame ──► useEyeStateModel（可选第二判定，env 开关，默认关）
                 └─ onFaceFrame(video, landmarks)      ← 本次新增
                          │
                          ▼
                    useFaceEmotion（节流 ≥1s + 在飞丢帧）
                      FacePatchExtractor：脸框外扩 → 正方形 224×224 canvas
                      FaceEmotionClassifier：ONNX Runtime Web（ResNet50，3 类）
                          │ 结果写 overlayRef
                          ▼
        useFacialAnalysis.setMetrics 时并入 FacialAnalysis.faceEmotion*
                          │
                          ├─► AnxietyContext.facialMetrics ──► comprehensiveState.facial
                          │        └─► Profile 页「本机表情模型」卡片（展示）
                          └─► POST /api/profile/anxiety-report（facial 字段原样带上）
                                   └─► server perceptionTags.facial.faceEmotion（落库留痕）
```

**没有第二路摄像头、没有第二个检测器**：模型消费的是 `useFacialAnalysis` 已经拿到的那一帧。

## 2. 开关与降级

| 情况 | 行为 |
|---|---|
| 多模态开关打开 | `useFaceEmotion(enabled=true)` 懒加载模型（首次约 90 MB，之后浏览器缓存），状态 `loading → ready` |
| 多模态开关关闭 / 组件卸载 | `enabled=false` → 释放 ONNX 会话（回收约 90 MB 堆）+ 清读数与 overlay |
| 模型下载失败 / 浏览器不支持 | 状态 `unavailable`，Profile 卡片显示原因；**眨眼链路完全照常**（不抛错、不 await、不改 `FacialFrameSignal`） |
| `VITE_DISABLE_FACE_EMOTION_MODEL=1` | 彻底不加载、不下载（省流量/省 CPU） |
| 无 WebGPU | 自动回退 `wasm`，且**跑在 Web Worker 里**（`env.wasm.proxy`），不阻塞 30fps 眨眼通道 |
| 人脸出画 / 关键点退化 | 该帧不采信（`fullyInsideFrame=false` 或不产 patch），沿用眼 patch 的同一规矩 |
| 人脸离开画面 / 读数 3 秒未刷新 | 读数自动失效并清空（`FACE_EMOTION_READING_TTL_MS`）—— 不允许把旧读数当"当前表情"一直显示 |
| `webgpu` 与 `wasm-proxy` 都建不起来 | 本模态置 `unavailable`，**不回退到主线程 wasm**（免得 300~400 ms 的前向把眨眼时间戳拖歪）；确有需要可显式打开 `allowMainThreadWasm` |

## 3. 文件清单

新增：
- `client/src/services/faceEmotionClassifier.ts` —— ONNX 会话管理 + 纯函数（`packFaceImage` / `softmax` / `toReading`）
- `client/src/hooks/useFaceEmotion.ts` —— 生命周期、节流、在飞丢帧、overlay 发布
- `client/src/hooks/facePatchCanvas.ts` —— 人脸裁剪框几何 + canvas 取像素（复用画布，按帧可调）
- `client/public/models/face_emotion_3cls.onnx`（89.62 MB，**被 .gitignore 忽略**，可由脚本重建）
- `client/public/models/NOTICE-face-emotion-3cls.txt` —— 出处/SHA/口径/限制（入库）
- `scripts/export_face_emotion_onnx.py` —— 手写 ResNet50 严格加载权重 → 导出 ONNX → 数值对齐（`--eval-dir` 可扫尺寸/算混淆矩阵）
- `scripts/verify_face_emotion_onnx.ts` —— 用真 onnxruntime-web(wasm) 跑端到端验证
- `scripts/face_emotion_domain_shift_check.py` —— **上线域体检**：分辨率/光照/模糊/压缩逐个施加，找"哪个因素把开心打掉了"（§6.4）
- `scripts/verify_face_emotion_accuracy.ts` —— 用真实图像跑**真正的前端代码**（常量 + 索引→标签映射）的精度复核（§6.1）
- `client/src/hooks/faceEmotionPacing.ts` —— **采样节律与展示平滑**的纯逻辑（§6.5）
- `scripts/verify_face_emotion_pacing.ts` —— 把节律改前/改后的刷新率直接算出来的断言脚本（§6.5）
- `client/src/components/EmotionStatusDock.tsx` —— 左下角悬浮工具栏：三条概率常显 + 近 10 秒峰值 + 刷新率/后端/裁剪源分辨率

修改：
- `client/src/hooks/useFacialAnalysis.ts` —— 新增第 4 个可选回调 `onFaceFrame` 与 `overlayRef`；两条 `setMetrics` 路径并入 overlay；`stop()` 清空 overlay
- `client/src/hooks/perceptionCapabilities.ts` —— `FACE_EMOTION_MODEL_AVAILABLE`（默认开，可 env 关）
- `client/src/types/multimodal.types.ts` —— `FaceEmotionLabel` / `FaceEmotionReading` / `FaceEmotionModelStatus` / `FacialMetricsOverlay` + `FacialAnalysis.faceEmotion*` 三个可选字段
- `client/src/context/AnxietyContext.tsx` —— 人脸帧回调转发、`useFaceEmotion(cameraEnabled && AVAILABLE, facial.overlayRef)`、开关关闭/`reset()` 时复位
- `client/src/pages/patient/Profile.tsx` —— 概览卡片一行 + 面部详情卡片一块（含三类概率、耗时、输入尺寸、"仅展示不参与风险"）
- `server/src/controllers/profile.controller.ts` —— `perceptionTags.facial` 增加 `faceEmotion` / `faceEmotionStatus`（只留痕，不参与计算）
- `.gitignore` —— 说明为何该 90 MB 产物保持忽略、只放行 NOTICE

## 4. 模型口径（必须与训练一致）

- 权重：`D:\Develop\AIC\Face\ResNet101_WholeData.pth`（名字写 ResNet101，**实际 ResNet50**，`fc=(3,2048)`，23 567 352 参数）
- 类别顺序：**`neutral(0) / happy(1) / sad(2)`**
  ⚠️ 训练脚本用的是 `classes = os.listdir(data_dir)`，**不是 `sorted()`**，所以顺序推不出来，只能实测；
  2026-10-10 扫全部 6 种排列确认（详见 §6）。
- 预处理：RGB、`x/255`、NCHW —— 训练时 transform 只有 `ToTensor()`，**没有 ImageNet 归一化、没有 resize**
  （`config.py` 里虽然定义了 `MEAN/STD = [0.523,0.428,0.389] / [0.226,0.200,0.191]`，但当前训练脚本没用它）
- **输入尺寸：96×96** —— 数据集里所有图片原生就是 96×96，模型只在 96 上训过（详见 §6）
- 导出：opset 17，输入名 `input`（H/W 动态），输出名 `logits`

## 5. 验证证据（全部本机实跑）

| 检查 | 命令 | 结果 |
|---|---|---|
| 权重↔结构 | `export_face_emotion_onnx.py` | `load_state_dict(strict=True)` 通过：320 张量、23 567 352 参数、`fc.weight=(3,2048)`、stage 3/4/6/3 |
| 导出保真 | 同上（torch ↔ onnxruntime CPU） | 224px 最大差 1.155e-06；64px 3.099e-06（阈值 1e-4） |
| 产物 | 同上 | 89.62 MB（93 974 968 B），SHA-256 `074ff87c…5026` |
| 端侧运行 | `node server/node_modules/tsx/dist/cli.mjs scripts/verify_face_emotion_onnx.ts` | ORT Web 加载成功（后端 `wasm-proxy`）；224px 推理 **383.5 ms**、128px **117.1 ms**；两次推理结果一致；softmax 和为 1 |
| 类型 | `tsc -b --force`（client） | 退出码 0 |
| 构建 | `vite build` | 成功；`dist/models/face_emotion_3cls.onnx` 89.62 MB 已随构建产出 |
| 服务端类型 | `tsc --noEmit`（server） | 退出码 0 |
| 运行中的 dev server | `GET /models/face_emotion_3cls.onnx` 等 | 200，Content-Length 93 974 968；三个新模块均能被 Vite 正常转换 |
| 独立复核（两个只读子代理，各自跑 tsc + ORT 端到端 + diff 审查） | — | 9 条不变量（眨眼安全/同帧复用/overlay 正确性/不进风险/生命周期/标签与预处理/服务端兼容/gitignore/向后兼容）**全部 CONFIRMED**；提出 7 条缺陷，已全部修复（见 §5.1） |

### 5.1 复核发现与修复（均已复验）

| # | 级别 | 问题 | 修复 |
|---|---|---|---|
| 1 | 中 | **边下载边关闭会泄漏会话**：`InferenceSession.create()` 在 await 之后才赋值，此时若已 `dispose()`，那 90 MB 会话无人回收 | `dispose()` 置 `disposed`；create 完成后若已释放则立即 `session.release()` 并返回 false（已加回归用例） |
| 2 | 中 | **读数无有效期**：人脸离开画面后没有新读数，overlay 每拍仍重新发布旧值，界面把几分钟前的表情显示成"当前" | 新增 `FACE_EMOTION_READING_TTL_MS = 3000`：每次成功结果排一个过期定时器，超时未刷新即 `setLatest(null)` |
| 3 | 低 | 复位后仍在飞的推理会写回 overlay，下一段会话第一拍会闪出上一段的读数 | 加会话代号 `generationRef`：`reset()`/关闭时 +1，`.then` 里校验代号，过期结果直接丢弃 |
| 4 | 低 | 主线程 wasm 兜底会阻塞 JS 线程（一次前向 300~400 ms），把 30fps 眨眼时间戳拖歪 | 该兜底改为**默认关闭**，仅在显式 `allowMainThreadWasm` 时启用并打印警告 |
| 5 | 低 | `extract()`/推理链异常会冒泡到 `useFacialAnalysis` 的帧循环（错误被记到眨眼通道头上）或变成未处理 rejection | `onFaceFrame` 整体 try/catch + `.catch()`，只打日志、丢本帧 |
| 6 | 低 | 环境变量关掉模型时界面仍显示"未启用（多模态未开启）"，与"用户主动关闭"无法区分 | Profile 详情卡区分显示 `FACE_EMOTION_MODEL_DISABLED_REASON` |
| 7 | 低 | `package.json` 的 `^1.30.0` 与写死的 CDN wasm `@1.30.0` 有版本漂移风险（升级后 glue/wasm 不匹配会静默失效） | 改为运行时按 `ort.env.versions.web` 推导 CDN 路径（已实测推导出 1.30.0），写死值仅作兜底 |

> 另有两条**未改**（已确认属可接受）：①加载失败不自动重试（避免反复下载 90 MB，状态保持 `unavailable` 直到重新开关多模态）；②`toggleCamera` 依赖数组里 `faceEmotion.reset` 而非整个对象（已顺手收敛，避免每次渲染重建回调）。

## 6. 精度标定（2026-10-10 已完成）

数据集到手后实测（脚本 `scripts/face_emotion_accuracy_check.py`，ONNX + onnxruntime CPU，与前端 `packFaceImage` 同口径）：

| 配置 | validation 准确率 |
|---|---|
| **修前**：224px + `happy, neutral, sad` | **40.6%** |
| **修后**：96px + `neutral, happy, sad` | **95.2%** |

两处根因都是"训练时的事实没被如实记录"：

1. **输入尺寸**：数据集全部是 **96×96**，训练 `Compose([ToTensor()])` 不 resize，
   模型只看过 96 —— 此前默认 224 等于把 96 拉大再喂。尺寸扫描结果：
   `64→43.6% / 96→95.1% / 128→54.9% / 160→51.6% / 224→40.0% / 256→37.1%`（n=450 抽样）。
2. **类别顺序**：训练用 `os.listdir()` 分索引（非排序），实测顺序是 `neutral, happy, sad`。
   此前按目录名 `sort()` 假设成 `happy, neutral, sad`，导致 **happy/neutral 两个名字整体对调** ——
   模型其实全预测对了，只是标签贴错，准确率因此掉到 40.6%。

全量 validation 混淆矩阵（行=真实，列=修正后的预测）：

```
          neutral    happy      sad
  happy        37      465        2
neutral       477       23       12
    sad         3        0      596
```

**交叉验证**：`Face/ResNet101_best9517.pth` 的文件名正好是 **95.17%**，
与本次在 96px + 正确映射下独立测得的 **95.2%** 吻合 —— 说明模型本身没问题，
问题全在部署侧的这两个常量。

### 6.1 复核（两层，都必须跑）

**第 1 层：模型侧** —— 扫输入尺寸 × 全部 6 种类别排列 + 混淆矩阵

```powershell
& "D:\Develop\AIC\.venv\Scripts\python.exe" scripts/face_emotion_accuracy_check.py
```
（onnxruntime 原生，与前端 `packFaceImage` 同口径：RGB / 双线性 / x/255 / NCHW）

**第 2 层：前端侧** —— 用**真正的前端代码**跑真实图像

Node 侧没有图像解码库，所以先用 Python 导出裸 RGBA，再让 TS 跑
`FaceEmotionClassifier.classify` → `packFaceImage` → `toReading` → `FACE_EMOTION_LABELS`：

```powershell
# 1) 导出（默认每类 200 张，共 600 张）
& "D:\Develop\AIC\.venv\Scripts\python.exe" scripts/face_emotion_accuracy_check.py `
    --dump-rgba .tmp-face-rgba --dump-limit 200
# 2) 跑前端代码，测「常量 + 索引→标签映射」这一层
node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_face_emotion_accuracy.ts --data .tmp-face-rgba
```

实测（n=600）：

| 配置 | 准确率 | 脚本退出码 |
|---|---|---|
| 发布配置：96px + `neutral,happy,sad` | **95.3%** | 0 ✅ |
| 历史 bug 配置：`--force-input-size 224` | **40.7%** | 1 ❌ |

`--force-input-size` 是**回归自检开关**：故意配错，用来证明这个测试确实能发现问题（不是恒过）。正常复核不要传。

> ⚠️ 第 2 层不能省：`tsc` 只保证类型对，**保证不了索引→标签的映射语义对**。
> 2026-10-10 那次 `toReading` 把 `probs[0]` 写死成 happy，只改 `FACE_EMOTION_LABELS`
> 是修不好的 —— 这类错误只有跑真实数据才能暴露。

**重新训练或重新导出后必须重跑两层** —— `os.listdir` 的顺序换机器就可能变。

### 6.2 剩余限制

1. **`testing` 集含第 4 类 `fear`（606 张）**，而模型只有 3 类。
   实测：fear 样本 **99.3% 被归到 sad**。在 3 分类可判的 450 张上准确率 96.2%；
   若把 fear 计入分母，"总体"只有 72.2%，**上限本来就只有 75%**。
   → 对外**不要报 testing 的总准确率**，要说明 fear 不可判；
   需要 4 类（含恐惧）必须重新训练。
2. 训练 27 轮的历史（`ResNet101_history.csv`）显示 train_acc 96.5% / valid_acc 62.7%，
   与 best9517 的 95.17% 不一致 —— 该 CSV 与最终权重不是同一次运行，引用时注意。
3. 真实摄像头场景的准确率**低于** 95.2%：数据集是已裁好的 96×96 人脸，
   而前端是"从画面裁人脸框再缩放"，留白比例/光照/角度都不同。95.2% 是数据集上的上限，
   不是线上指标。**具体是哪些因素、各掉多少，见 §6.4 的上线域体检**（结论：光照几乎无影响，
   掉分集中在分辨率/模糊，且失败模式正是"微笑被判成中性"）。

### 6.3 其余待办

1. **部署需要额外一步**：`.onnx` 未入库，部署机要么跑一次导出脚本，要么把该文件随构建产物拷贝。
2. **未接入通话页**：`useCallVision`（VoiceCall）另有一条 `useFacialAnalysis(ageGroup, null, null)` 管线，本次只接多模态开关这条；通话页要接需显式传第 4 个回调。
3. **咨询师端未展示**：`perceptionTags.facial.faceEmotion` 已落库，但咨询师端 UI 还没读它。

### 6.4 上线域诊断（2026-10-10）：掉分几乎全部来自「细节丢失」，不是光照

> **用户反馈「咧嘴笑被识别成悲伤」的完整根因与修法见 [`face_fix_diagnosis.md`](face_fix_diagnosis.md)。**
> 一句话：前端没裁错（只偏 2.1px，代价 ≈1pp），是**模型对"人脸在 patch 里的纵向位置"极度敏感**
> —— 上移 4px 就有 35%、6px 有 76.2% 的"开心"被改判成"悲伤"。已通过**抗干扰微调**修复：
> 干净验证 95.23%→**95.60%**、鲁棒验证 72.08%→**94.40%**，逐干扰的"开心→悲伤"全部归零。

用户反馈「开心几乎识别不出来」。数据集上的 95.2% 说明模型本身没问题，
所以先要定位是**哪一个上线因素**把开心打掉了。做法是把 validation 图按摄像头真实会遇到的
退化**逐个单独施加**，再用同一套前端口径重测：

```powershell
& "D:\Develop\AIC\.venv\Scripts\python.exe" scripts/face_emotion_domain_shift_check.py --limit 80
```

实测（validation n=240，每格 = 总体 / 开心召回 / 开心被判成中性的比例）：

| 退化 | 总体 | 开心召回 | 开心→中性 | 结论 |
|---|---|---|---|---|
| clean | 97.5% | 96.2% | 3.8% | 基线 |
| flat / dark / bright / wb_warm / wb_cool | 97.5–98.3% | 96.2–98.8% | ≤3.8% | **光照与白平衡基本无影响** |
| noise8 | 95.8% | 93.8% | 6.2% | 噪声影响很小 |
| blur1 | 92.1% | 83.8% | 16.2% | 开始明显掉 |
| halfres（等效源 48px） | 90.0% | 77.5% | 22.5% | 分辨率减半 → 开心掉 19pp |
| thirdres（等效源 32px） | 75.4% | 47.5% | 43.8% | 分辨率 1/3 → 开心掉一半 |
| blur2 | 64.6% | 28.7% | **68.8%** | **微笑成片被判成中性** |
| jpeg30 | 68.3% | 60.0% | 10.0% | 压缩主要伤中性/悲伤 |
| webcam（叠加） | 70.4% | 65.0% | 35.0% | 典型摄像头条件 |
| webcam_hard | 59.6% | 13.8% | 57.5% | 差条件 |

三条可执行的结论：

1. **光照归一化 / 白平衡补偿不值得做** —— `dark/flat/wb_*` 全在 95% 以上，
   去加 mean/std 归一化只会偏离训练分布（训练是 `ToTensor()`，无归一化）。
2. **开心类对清晰度最敏感，且失败模式恰好是「被判成中性」**（blur2 下 68.8%）。
   这与用户描述完全一致。可操作的做法只有两条：让脸在画面里更大、别让 patch 被放大。
   所以界面新增 **`裁剪源 Xpx → 96px`** 读数：`X < 96` 就是"放大喂模型"，
   并给出非诊断性的可操作提示「离摄像头近一些」。
3. **前端的降采样质量从默认 `'low'` 改为 `'high'`**（`client/src/hooks/facePatchCanvas.ts`）——
   默认 `'low'` 在源矩形明显大于 96px 时只做 2×2 双线性采样（欠采样），
   而微笑的嘴角弧度/鼻唇沟正是高频特征。这是一次零额外成本的滤波改进。

同时 A/B 了两种补偿方案，**结论是暂不采用**（记录在此以免重复试）：

| 方案 | clean（总体/开心） | blur2 | thirdres | 判断 |
|---|---|---|---|---|
| 不加（现状） | 95.6 / 93.3% | 62.8 / 20.0% | 75.0 / 36.7% | 基线 |
| 锐化单独用 | 94.4 / 90.0% | 69.4 / 30.0% | 69.4 / 40.0% | **干净样本掉点**，不采用 |
| TTA：原始+锐化取平均 | 95.0 / 91.7% | 70.6 / 33.3% | 72.8 / 40.0% | 收益不抵 2× 推理成本 |
| TTA：水平翻转取平均 | 96.1 / 93.3% | 66.1 / 20.0% | 77.8 / 46.7% | 干净不掉点，但同样 2× 成本 |

> ⚠️ 这些都是**仿真**：退化算子是我们构造的，不是用户真实摄像头的画面。
> 它们能回答"哪个因素最可能致命"，不能替代真实采集。真要把线上精度坐实，
> 必须采一批**带标签的真实摄像头 patch** 再测（见 6.5）。

### 6.5 采样节律（2026-10-10）："一直在等结果"是节律问题，不是模型问题

旧实现把采样间隔写死成 `max(1000ms, 上次耗时 × 2)`。慢后端上一次推理 600–1500ms，
实际周期就被抬到 **1.2–3 秒**，界面几秒才跳一次。现在改为**贴住后端能力连续推理**：

```
gap = clamp(上次耗时 × 1.15, 60ms, 2000ms)      // 见 hooks/faceEmotionPacing.ts
```

人脸帧只在**指标拍**（10fps）到来，所以 60ms 下限的实际含义是"每拍都跑"。
`scripts/verify_face_emotion_pacing.ts` 把改前/改后直接算了出来（5 秒内的成功采样次数）：

| 单次推理 | 旧周期 | 新周期 | 旧次数/5s | 新次数/5s | 倍率 |
|---|---|---|---|---|---|
| 30ms（WebGPU） | 1000ms | 60ms | 6 | **51** | 8.5× |
| 200ms | 1000ms | 230ms | 6 | 17 | 2.8× |
| 600ms | 1200ms | 690ms | 5 | 8 | 1.6× |
| 1500ms（wasm 单线程） | 3000ms | 1725ms | 2 | 3 | 1.5× |

配套的三件事（都已实现并断言）：

- **读数有效期自适应** `faceEmotionReadingTtlMs(耗时)`（1.5–8s，恒 > 2 个采样周期）——
  写死 3 秒会让慢后端在两次成功之间就过期，界面闪成「暂无新读数」。
- **展示平滑**：τ=450ms 的 EMA，α = 1−exp(−Δt/τ)，首帧直接等于原始值（不制造从 0 爬升的假过渡）。
  上屏用 `smoothedProbabilities`，**判定与风险一律用未平滑的 `probabilities`**，
  卡片里两者并列显示。
- **实测刷新率 / 执行后端 / 单次耗时 / 裁剪源分辨率** 全部如实上屏（左下角工具栏），
  慢就是慢，不让用户对着不动的数字猜。

验证命令：

```powershell
node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_face_emotion_pacing.ts   # 节律断言，退出码即结论
```

## 7. 回滚

- 关功能：`VITE_DISABLE_FACE_EMOTION_MODEL=1`（不加载、不下载、界面显示"未启用"）。
- 完全移除：删除第 3 节"新增"的 4 个源码文件 + `client/public/models/face_emotion_3cls.onnx`，
  再把 `useFacialAnalysis` 的第 4 个回调与 `AnxietyContext` 的 `useFaceEmotion` 调用去掉即可
  （其余改动都是纯增量、可选字段，删掉后不影响眨眼与其它模态）。
