# 外部 FER 模型能不能接进「表情识别」这个板块？—— 实测结论与接入方案

> 2026-10-10。所有数字都是本机跑出来的，命令见 §6，可逐条复现。
> 相关：`docs/face_fix_diagnosis.md`（此前三分类模型的精度体检）、`docs/face_emotion_model_integration.md`（接入面总览）。
>
> **✅ 状态：已按选项 A 切换完成**（2026-10-10）。线上模型 = HSEmotion `enet_b0_8_best_vgaf`
> （`client/public/models/face_emotion_8cls_hsemotion.onnx`，15.3 MB，Apache-2.0），
> 规格见 `FACE_EMOTION_SPEC`，第三方声明见 `NOTICE-face-emotion-hsemotion.txt`。
> 执行记录、验收结果与遗留项见 §7。

---

## 0. 结论（三句话）

1. **能接，而且已经接好了。** 表情分类器已经从"写死一个模型"改成"按**模型规格**（`FaceEmotionModelSpec`）加载"，
   外部 8 类 FER 模型已经**在真正的前端代码路径上跑通**（`FaceEmotionClassifier` + ORT-Web wasm，见 §2）。
   默认仍是原来的自训练模型，**行为逐位不变**（frontend 路径 95.8% / 575-600，与改前一致）。

2. **而且应该接。** 在**真实成人人脸**（RAF-DB test）上，我们自训练模型的表现远不如 15 MB 的 HSEmotion：

   | | 我们（89.6 MB） | HSEmotion（15.3 MB） |
   |---|---|---|
   | 真实成人**笑脸**被显示成「**悲伤**」的比例 | **56.9%**（喂原图紧取景时 **98.3%**） | **0.8%** |
   | 真实成人中性脸被显示成「悲伤」 | 92.5% | 10.8% |
   | sad 的 AUC（P(sad) 排序能力，与阈值无关） | 0.630 | **0.965** |

   「咧嘴笑 → 识别成悲伤」这个症状，**在一份与用户无关的公开成人数据集上被完整复现了**。
   这不是用户的脸的问题，也不是取景偏差的问题，是**模型本身在成人人脸上不成立**。

3. **但不要用"我们自己数据集上的准确率"来选模型。** 在本仓库自带数据集（`数据集/Face`）上，
   外部模型**全线落后**（sad 召回 10%~33%，我们 100%）——因为那个数据集的 `sad` 大量是
   **婴幼儿嚎哭 / 满脸泪水 / 瞪眼张嘴的惊恐**，在 8 类模型眼里是 Fear/Surprise。
   两个数据集的"sad"根本不是同一个概念。**这正是"本地 95.8% 却在用户脸上不灵"的根因。**

---

## 1. 接入面：改了哪些代码

| 文件 | 改动 | 性质 |
|---|---|---|
| `client/src/services/faceEmotionClassifier.ts` | 新增 `FaceEmotionModelSpec` + `FACE_EMOTION_SPEC`；`packFaceImage` 支持灰度/单通道/BGR/mean-std；`toReading` 支持**任意类别数折叠**到展示三分类；`classify` 按规格取输入形状、按 `spec.labels.length` 校验输出 | 行为兼容（默认规格 = 原模型） |
| `client/src/types/multimodal.types.ts` | `FaceEmotionReading` 新增 `offMapProbability`（折叠不到三分类的概率总量，如实上报）；`FACE_EMOTION_LABELS` 的语义从"模型输出顺序"改成"展示口径" | 纯新增字段 |
| `client/src/hooks/useFaceEmotion.ts` | 裁剪边长跟 `FACE_EMOTION_SPEC.inputSize` 走（换 224px 模型时**直接从摄像头原帧裁 224px**，不是把 96 放大） | 行为兼容 |
| `scripts/verify_face_emotion_accuracy.ts` | 新增 `--spec <json>`：用外部模型的规格跑**同一条前端代码路径**（预处理 + 索引→标签 + 折叠），并汇报折叠出去的概率质量 | 仅测试 |
| `scripts/fer_external_eval.py`（新） | 通用「任意 FER ONNX」评估：规格化预处理、三种取景（native / live / tight）、5 点对齐、**类别顺序实测探针**、sad P/R/F1/AUC | 新增工具 |
| `scripts/fer_cross_domain_bench.py`（新） | 成人跨域对照（RAF-DB parquet 内存解码），含"标签整数含义"的两种假设对照 | 新增工具 |

**为什么下游一行都不用改**：`FaceEmotionReading.probabilities` 永远只有 neutral/happy/sad 三个键，
所以 `faceEmotionPacing`（节律/有效期/平滑）、`EmotionStatusDock`、`Profile` 全部无感。
外部模型多出来的类不丢，走 `offMapProbability`。

### 规格里三个"猜错就废"的字段（各有一个实测反例）

| 字段 | 猜错的后果（实测） |
|---|---|
| `inputSize` | 我们自己的模型 96→224：95.2% → **40.6%**（`face_emotion_accuracy_check.py`） |
| 归一化 `mean/std/scale` | FER+ 模型：`scale=1.0`（0~255）得 69.1%；误设成 `/255` 得 **33.3%**，且 7 类全部输出 neutral（模型退化成常数） |
| `labels`（类别顺序） | 用 `--probe-labels` 穷举验证：4 个候选的实测最佳排列**都等于规范顺序**；而把 5 点对齐的左右点对调，准确率 73.8% → **51.7%** |

> 类别顺序**必须实测**，不能只信 README：本项目自己就栽过两次（`os.listdir` 顺序、输入尺寸）。

### 5 点对齐值多少钱（OpenCV Zoo 那个 4.6 MB 模型）

| 输入 | 三分类准确率 | happy | neutral | sad | sad AUC |
|---|---|---|---|---|---|
| 直接缩放 112×112 | 70.6% | 80.0% | 48.3% | 31.7% | 0.6578 |
| **5 点相似变换对齐** | **73.8%** | **96.7%** | **73.3%** | 26.7% | **0.7696** |

对齐把"取景/远近/轻微转头"归一化掉了 —— 这一条对我们线上裁剪（固定产出 h_n≈0.66）特别有意义：
**要求对齐的模型，不适合直接吃我们的裁框**。

---

## 2. 真机验证：外部 8 类模型跑在真正的前端代码路径上

```
python scripts/face_emotion_accuracy_check.py --dump-rgba .tmp-face-fer20 --dump-limit 20
node server/node_modules/tsx/dist/cli.mjs scripts/verify_face_emotion_accuracy.ts \
    --data .tmp-face-fer20 \
    --model D:\Develop\AIC\_fer_candidates\hsemotion_enet_b0_8_best_vgaf.onnx \
    --spec  D:\Develop\AIC\_fer_eval_cache\spec_hsemotion.json
```

```
[规格] 模型自己的类别 → 展示类：Anger→其他, …, Happiness→happy, Neutral→neutral, Sadness→sad, …
[表情模型] ✅ HSEmotion enet_b0_8_best_vgaf（8类 → 三分类折叠） 已加载（15.3 MB，输入 224px，8 类 → 折叠为 neutral/happy/sad，后端 wasm-proxy）
准确率 = 81.7%  (49/60)
平均单张推理 47.0 ms | 总耗时 3.2 s
折叠不进三分类的平均概率质量 = 31.1%
```

结论：**不同输入尺寸、ImageNet 归一化、8 类输出 → 折叠三分类，这条链路在浏览器同款运行时（ORT Web wasm proxy）上完全可用**，
单张 47 ms，与我们自己 90 MB 模型同量级（45.4 ms）。退出码 0。

---

## 3. 候选模型清单（尺寸 / 许可 / 速度）

| 模型 | 文件大小 | 输入 | 类别 | 许可 | 单线程 CPU（Python ORT） |
|---|---|---|---|---|---|
| **HSEmotion `enet_b0_8_best_vgaf`** | **15.3 MB** | 224, RGB, ImageNet 归一化 | 8 | **Apache-2.0**（EmotiEffLib README 明示"学术与商用均无限制"） | 17 ms |
| OpenCV Zoo `facial_expression_recognition_mobilefacenet_2022july` | **4.6 MB** | 112, RGB, (x-0.5)/0.5，**需 5 点对齐** | 7 | **Apache-2.0**（OpenCV Zoo，目录内 LICENSE） | 9~17 ms |
| ONNX Model Zoo `emotion-ferplus-8` | 33.4 MB | 64, **单通道灰度, 原始 0~255** | 8 (FER+) | Apache-2.0（onnx/models 仓库）；⚠️ 模型卡另标 MIT，**待人工确认** | 17 ms |
| `dima806/vit_fer2013` (ViT-base) | 327.6 MB | 224, RGB, 0.5/0.5 | 7 | ⚠️ **未确认**（HF 页面本机不可达） | 284 ms ← 浏览器里不现实 |

> 我们用过的"下载即可跑"的模型共 4 个，全部亲测输入/输出规格与类别顺序（穷举验证），产物在 `D:\Develop\AIC\_fer_candidates\`。

---

## 4. 本地域实测（`数据集/Face/validation`，原图口径，每类 60，n=180）

| 模型 | 大小 | 三分类准确率 | sad 精确率 | sad 召回 | sad AUC | 折叠覆盖率 |
|---|---|---|---|---|---|---|
| **我们（v2 微调）** | 89.6 MB | **97.2%** | 100% | **100%** | **1.0000** | 100% |
| HSEmotion | 15.3 MB | 84.3% | 80.0% | 33.3% | 0.7901 | 67.2% |
| FER+（正确归一化） | 33.4 MB | 69.1% | 100% | 11.7% | 0.7778 | 91.7% |
| FER+（归一化猜错） | 33.4 MB | 33.3% | 0% | 0% | 0.5065 | 100% |
| OpenCV（对齐） | 4.6 MB | 73.8% | 94.1% | 26.7% | 0.7696 | 88.9% |
| OpenCV（未对齐） | 4.6 MB | 70.6% | 70.4% | 31.7% | 0.6578 | 75.6% |
| ViT dima806 | 327.6 MB | 69.4% | 46.2% | 10.0% | 0.6390 | 74.4% |

几个可读的事实：
- 本数据集 `sad` 在 HSEmotion 眼里：**Fear 35.0% / Sadness 33.3% / Neutral 15.0%** —— 它把"婴幼儿嚎哭"读成恐惧，把人读成中性。
- ViT 那 327 MB 也没救：`sad` 只召回 10%（其中 25% 判成 fear、23.3% 判成 angry）。
- **模型再大也没用，问题在数据集的"sad"定义**。

---

## 5. 成人跨域实测（RAF-DB test，真实世界成人人脸，100×100 彩色对齐图）

- 取 `elipaluma/rafdb` test parquet 794 张（每类 ≤120），**内存解码，不落中间图**。
- MediaPipe 在上面测到关键点 681/794；**h_n 中位数 0.9612** —— 即这些人脸本来几乎占满画面，而线上永远是 **0.6608**。
- 标签整数含义**不照抄任何 README**：同时跑两种假设，
  在"1 起（1=surprise…5=sad…7=neutral）"下三个外部模型都正常（77.8% / 76.9% / 73.6%），
  在"0 起"下全部退化到随机（0.8% / 12.2% / 17.5%）—— **数据自己指出了正确约定**。

### 5.1 我们自己的模型：7 类里 6 类被判成「悲伤」

按线上裁剪口径重渲染（h_n=0.6608，681 张）：

| 真实（RAF-DB） | →中性 | →开心 | →悲伤 | n |
|---|---|---|---|---|
| angry | 2.2% | 7.6% | **90.2%** | 92 |
| disgust | 5.8% | 6.7% | **87.5%** | 104 |
| fear | 0.0% | 3.2% | **96.8%** | 63 |
| **happy** | 1.0% | 42.2% | **56.9%** | 102 |
| **neutral** | 6.6% | 0.9% | **92.5%** | 106 |
| sad | 1.0% | 1.0% | 98.1% | 104 |
| surprise | 1.8% | 0.9% | **97.3%** | 110 |

**用户的原始症状（"我咧着嘴笑识别成了悲伤"）在这里被独立复现**：
每 2 张真实成人笑脸就有 1 张以上被显示成「悲伤」，中性脸更是 92.5% 显示成「悲伤」。
喂 RAF-DB 原图（紧取景）时更极端：**7 类全部 ~100% 判成 sad**（这是"脸占满画面 → sad"取景捷径的直接证据，
与 `docs/face_fix_diagnosis.md` 记录的 shift-up 敏感度是同一个病根）。

### 5.2 横向对比（同一批图，各模型吃自己的口径）

| 模型 | 大小 | 三分类准确率 | sad 精确率 | sad 召回 | sad F1 | **sad AUC** | 折叠覆盖率 |
|---|---|---|---|---|---|---|---|
| 我们（按线上口径重渲染） | 89.6 MB | 48.7% | 39.5% | 98.1% | 56.4% | 0.6301 | 100% |
| 我们（喂原图，对照） | 89.6 MB | 33.6% | 33.5% | 100% | 50.2% | 0.6411 | 100% |
| **HSEmotion** | **15.3 MB** | **77.8%** | 87.7% | 83.3% | **85.5%** | **0.9646** | 45.2% |
| FER+ | 33.4 MB | 76.9% | 97.0% | 54.2% | 69.5% | 0.9052 | 79.2% |
| OpenCV（对齐） | 4.6 MB | 73.6% | 95.2% | 50.0% | 65.6% | 0.8050 | 94.6% |

HSEmotion 的逐类表现（同一批图）：happy 85.8% 正确、neutral 64.2% 正确、sad 83.3% 正确，
而 angry/disgust/fear/surprise 有 81.7%~94.2% 被判为"其他"（**它知道自己不认识的就不硬猜**）——
这正是我们那个 3 类模型结构上做不到的事。

### 5.3 这份对比的诚实边界

1. **我们的模型吃的是"重渲染成 h_n=0.66"的图**，越界区域只能用**边缘均值填充**（真实背景是头发/房间）。
   但"喂原图"的对照组同样崩（33.6%、happy→sad 98.3%），所以结论不依赖这个填充。
2. **分辨率对我们有利**：我们输入 96（RAF-DB 是 100×100，几乎不放大），
   HSEmotion 输入 224（把 100×100 拉大 2.24 倍，是**吃亏**的一方）。它仍然大幅胜出。
3. **sad 精确率那一列对 3 类模型天然不利**：RAF-DB 的 angry/fear/disgust/surprise 对我们是"本来就没有的类"，
   只能被归进三类里，必然拉低精确率。所以**判断优劣请看 sad AUC 与"笑脸被显示成悲伤的比例"这两列**（它们与类别词汇无关）。
4. 这里测的是 rafdb 的 test 划分（794 张），不是完整 3068 张；每类 ≤120。

### 5.4 换模型之后，裁框要不要一起改？

线上裁框永远是 `h_n≈0.6608`，而 AffectNet 系模型期望的是更紧的框。把 RAF-DB **重渲染**成不同口径，
看新模型对取景有多敏感（同一批图、同一套填充方式，可比）：

| 喂给 HSEmotion 的取景 | 三分类准确率 | sad 精确率 | sad 召回 | sad AUC |
|---|---|---|---|---|
| 原图（RAF-DB 本身就是 h_n≈0.96 的紧对齐图） | 77.8% | 87.7% | 83.3% | 0.9646 |
| **现行线上口径 h_n=0.6608** | 71.8% | 80.6% | 76.0% | 0.9310 |
| AffectNet 风格紧框 h_n=0.90 | 73.1% | 81.7% | **85.6%** | **0.9681** |

**结论：这次先不动裁框**，理由是这份数据**定不了这件事**：
RAF-DB 的图本身已经是紧对齐裁图，要把它"变回 0.66 口径"只能把四周用**边缘均值填满**
（真实场景里那是头发/房间的**真实背景**）。填充面积越大，越吃亏 ——
而 0.66 恰好需要填得更多，所以表里 0.66 那一行的劣势**至少有一部分是填充伪影**，
不是模型对取景的真实敏感度。要定这件事，必须在**真实摄像头画面**上量（见 §7 遗留项 4）。
（对照组：我们自己那个模型在同样条件下 0.66 口径 48.7%、原图 33.6%，方向相反且差距大得多 ——
它是被"脸占满画面 → 悲伤"这条捷径绑死的，不能作为裁框依据。）

---

## 6. 复现命令

```powershell
cd "D:\Develop\AIC\AI-mental-main (2)\AI-mental-main\AI-mental-main"
$PY = "D:\Develop\AIC\.venv\Scripts\python.exe"

# ① 外部候选的规格核实 + 类别顺序实测 + 本地域精度（native/live/tight 三种取景）
& $PY scripts\fer_external_eval.py --specs "D:\Develop\AIC\_fer_eval_cache\specs_variants.json" `
    --limit 60 --framings native --probe-labels --latency --out "D:\Develop\AIC\_fer_eval_cache\probe2.md"

# ② 成人跨域对照（RAF-DB；含两种标签约定假设）
& $PY scripts\fer_cross_domain_bench.py `
    --parquet "D:\Develop\AIC\_fer_bench\_raw\elipaluma_rafdb\data\test-00000-of-00001.parquet" `
    --specs "D:\Develop\AIC\_fer_eval_cache\specs_variants.json" --per-class 120 `
    --out "D:\Develop\AIC\_fer_eval_cache\xdomain_rafdb.md"

# ③ 真机路径（前端代码 + ORT Web wasm）：外部 8 类模型能否端到端跑通
& $PY scripts\face_emotion_accuracy_check.py --dump-rgba .tmp-face-fer20 --dump-limit 20
node "server/node_modules/tsx/dist/cli.mjs" scripts\verify_face_emotion_accuracy.ts `
    --data .tmp-face-fer20 --model "D:\Develop\AIC\_fer_candidates\hsemotion_enet_b0_8_best_vgaf.onnx" `
    --spec "D:\Develop\AIC\_fer_eval_cache\spec_hsemotion.json"

# ④ 回归：规格化重构必须让默认模型行为一字不变（重构当时跑的，见 §7.0）
& $PY scripts\face_emotion_accuracy_check.py --dump-rgba .tmp-face-rgba --dump-limit 200
node "server/node_modules/tsx/dist/cli.mjs" scripts\verify_face_emotion_accuracy.ts --data .tmp-face-rgba
node "server/node_modules/tsx/dist/cli.mjs" scripts\verify_face_emotion_onnx.ts
node "server/node_modules/tsx/dist/cli.mjs" scripts\verify_face_emotion_pacing.ts
```

回归现状（**切换之后**，2026-10-10）：client `tsc --noEmit` 0；`vite build` ✓ exit 0；
三个 verifier 全绿（exit 0）；**线上模型（HSEmotion）经真实前端代码路径 = 82.3%（494/600）**，
平均单张 45.8 ms（ORT Web wasm-proxy）。
（规格化重构当时、还没换模型的对照：**95.8%（575/600）**，与重构前逐位一致 ——
即"把模型变成规格"这一步本身没有改变任何行为，之后的数字变化全部来自换模型。）

---

## 7. 决策与执行记录

### 7.0 ✅ 已执行：选项 A（换成 HSEmotion）

用户 2026-10-10 选定 A。实际改动：

| 改动 | 内容 |
|---|---|
| 新增资产 | `client/public/models/face_emotion_8cls_hsemotion.onnx`（15,039,595 B，SHA-256 `fa07e841…c898`） |
| 新增声明 | `client/public/models/NOTICE-face-emotion-hsemotion.txt`（Apache-2.0 + 来源 + 实测规格 + 换模型理由） |
| 切换规格 | `FACE_EMOTION_SPEC`：`inputSize` 96→224、新增 ImageNet `mean/std`、8 类 `labels` + `fold`、`bytes` |
| 界面 | `EmotionStatusDock` 新增「其他表情 X%」行；落在三类之外的占比 >50% 时标题显示「○ 其他表情」而不是硬报三类第一名 |
| 校验脚本 | 两个 verifier 的默认模型路径跟随切换；`verify_face_emotion_accuracy.ts` 的门槛由 80% 改为 **60%（= 拦规格错配）**，60~90% 区间提示"数据域差异"而非配置错误 |
| 旧模型 | `face_emotion_3cls.onnx`（89.6 MB）保留在原地但**运行时不加载**，只作为三分类审计脚本的历史产物 |

**验收（全部本机实测）**

| 验收项 | 结果 |
|---|---|
| client `tsc --noEmit` | **0** |
| `vite build` | **✓ exit 0**（14.8 s） |
| `verify_face_emotion_accuracy.ts`（真实前端路径，600 张） | **82.3%**（494/600，happy R 92.0% / neutral R 81.0% / sad R 74.0%，exit 0） |
| `verify_face_emotion_onnx.ts` | **exit 0**（预览含 ImageNet 归一化、8 类折叠 + `offMapProbability`、**固定形状下喂错尺寸优雅降级返回 null**、边加载边 dispose 竞态） |
| `verify_face_emotion_pacing.ts` | **exit 0**（节律/有效期/平滑三项断言全过） |
| 下载体积 | 90 MB → **15.3 MB** |

**代价（已确认接受）**：本仓库数据集上的数字由 95.8% 降到 82.3%，且 sad 召回 100%→74% ——
该数据集 `sad` = 婴幼儿嚎哭，新模型读成 Fear/Surprise（§4）。
衡量真实场景请用 §5.2：真实成人笑脸被判「悲伤」56.9% → **0.8%**，sad AUC 0.630 → **0.965**。

### 7.1 备选方案（未采用，留档）

**选项 B —— 双模型仲裁（折中）**
- 收益：15.3 MB（**下载量降到 1/6**）、Apache-2.0 可商用、单张 ≈17 ms（CPU）/ 47 ms（ORT-Web wasm），
  成人脸上"笑脸被显示成悲伤"从 56.9% 降到 **0.8%**，sad AUC 0.63 → **0.965**。
- 代价：本仓库数据集上的三分类准确率从 97.2% 降到 84.3%（**但那个数字衡量的是一个"sad = 婴幼儿嚎哭"的域**）；
  真实成人脸上有 45.2% 的预测会落到三分类之外 → 界面需要显示「其他表情 X%」（`offMapProbability` 已就绪）。
- 需要改：`FACE_EMOTION_SPEC` 一个对象 + 把模型放进 `client/public/models/` + 更新 `NOTICE-*.txt`。

**选项 B —— 双模型仲裁（折中）**
保留自训练模型为主，仅在它给 neutral/sad 接近时用 HSEmotion 第二意见。
代价是内存与一次前向，收益是保留"本地域 97%"的既有观感 —— 但从 §4/§5 看，**本地域那 97% 很可能是个幻觉**，我不推荐为它付复杂度。

**选项 C —— 采集 + 标注 + 微调（原本的路线）**
如果还要继续微调，**必须换评测集**：拿一份成人数据（RAF-DB 或 ExpW）当验收标准，
否则只会继续优化"婴幼儿嚎哭识别"。这条路的真正价值是"针对**你自己的脸**"，那才需要采集。

### 7.2 遗留事项

1. `dima806/vit_fer2013` 的许可证未确认（不推荐用：327 MB，浏览器里不可行）。
2. `onnx/models` 的 FER+ 许可两份声明（Apache-2.0 / 卡上 MIT），商用前需人工确认。
   （**不影响本次切换**：线上模型是 Apache-2.0 的 HSEmotion。）
3. `face_emotion_3cls.onnx`（89.6 MB）仍在 `client/public/models/` 里、会被 `vite build`
   一起拷进 `dist/`（浏览器不会下载它，但产物体积多 90 MB）。确认不再需要回溯后可以删掉，
   或移到 `D:\Develop\AIC\Face\` —— 那会把三分类审计脚本的 `DEFAULT_MODEL` 一起改掉，所以没有顺手做。
4. **裁框要不要跟着收紧到 h_n≈0.90**：§5.4 显示在 RAF-DB 上紧框比现行 0.66 略好
   （sad 召回 76.0% → 85.6%、AUC 0.931 → 0.968），但那份数据**无法排除填充伪影**，
   因此没有动 `FACE_PATCH_MARGIN`。**要定这件事，必须在真实摄像头画面上量**：
   用一次真人采集（哪怕几十张自拍），分别按 0.66 / 0.90 渲染后跑同一套比对。
5. 8 类模型的「其他表情」在真实成人脸上有 44.6%~46.3% 的占比（§5.2）——
   如果希望界面给出更具体的词（如「惊讶／恐惧／愤怒」），需要先把展示词汇从 3 类扩到 N 类，
   那是 UI 与埋点层面的改动，本次没做。
