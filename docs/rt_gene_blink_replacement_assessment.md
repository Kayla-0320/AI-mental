# RT-GENE / RT-BENE 能否替换现有眨眼检测 —— 可行性评估

> 评估对象：<https://github.com/Tobias-Fischer/rt_gene>（RT-GENE, ECCV 2018 + RT-BENE, ICCVW 2019）
> 评估对象版本：tag `5.0.0`（源码本次已实际取回核对，见 §7 与 §9 附件目录）
> 评估对象链路：`useFacialAnalysis.ts` → `onFrameSignal` → `useEyeTracking.ts` → `blinkLogic.BlinkStateMachine`
> 结论出具时间：2026-09-27

---

## 0. 结论（先给答案）

**不能作为"替换件"接入，不建议替换。** 三条硬阻断：

1. **它不是一个"眨眼检测器"，而是"每帧睁眼/闭眼概率分类器"** —— 与我们现有的 `eyeBlinkLeft/Right` blendshape **语义完全相同**，换掉它并不改变任何时间维度的判定逻辑；
2. **它不解决我们真正的瓶颈** —— 现链路的误差主要来自 `useFacialAnalysis.ts:577` 的 **100ms（10fps）采样**，不是分类器精度。换模型不改变量纲级问题（详见 §4）；
3. **权重拿不到 + 许可证是 CC BY-NC-SA 4.0（禁商用、相同方式共享）** —— 见 §2.5、§2.6。

**真正值得借用的只有两件东西**，而且都不涉及采用它的模型权重：

- **RT-BENE 数据集**（Zenodo，本机实测 HTTP 200 可达）：17 名受试者、20 万+ 眼部 patch、逐帧 0/1 标注 —— 可以直接用来**给"我们现在的 MediaPipe 眨眼"打出 precision/recall/F1**，补上 `docs/video_perception_audit.md:1004` 里那条至今 `[未验证]` 的欠采样评估；
- **它的眼 patch 裁剪口径**（60×36、按双眼中心去 roll 对齐）—— 实测**只用眼部角点、不需要头姿 PnP**，用 MediaPipe 现成的角点即可复刻（§5.2）。

---

## 1. 先纠正一个前提：RT-GENE 本体不是眨眼项目

同一个仓库里是两篇论文的代码：

| 目录 | 论文 | 任务 |
|---|---|---|
| `rt_gene/`、`rt_gene_standalone/`、`rt_gene_model_training/`、`rt_gene_inpainting/` | RT-GENE (ECCV 2018) | **注视方向估计**（gaze），附带 S3FD 人脸检测 + 3DDFA 68 关键点 + 头姿 |
| 同上目录内的 `rt_bene` 模块、`rt_bene_standalone/`、`rt_bene_model_training/` | RT-BENE (ICCVW 2019) | **眨眼（睁闭眼）估计**，即我们关心的部分 |

所以"用 RT-GENE 替换眨眼检测"在工程上实际是：**用 RT-BENE 的 `BlinkEstimator`（`rt_gene/scripts/estimate_blink.py` → `rt_gene/src/rt_bene/*`）替换 `blinkScore = min(eyeBlinkLeft, eyeBlinkRight)` 这一路信号源**。下游的状态机（迟滞 / 时长窗 / 最小间隔 / 滚动指标）**原本就要保留**，它不在 RT-BENE 里。

---

## 2. 源码事实核查（tag 5.0.0，已实际下载）

### 2.1 接口

| 项 | PyTorch 后端 | TensorFlow 后端（launch 文件里的默认） |
|---|---|---|
| 输入 | 左眼 + 右眼 patch，`cv2.resize` 到 **60×36**，ImageNet 均值方差归一化 | 左眼 + 右眼 patch，`cv2.resize` 到 **96×96**，**右眼水平翻转** |
| 输入顺序 | `predict(left_eyes, right_eyes)` | 源码注释写明模型期望的顺序是 `[right, left]` |
| 输出 | 两个分支各出 1024 维 → concat → FC(512) → FC(1) → `sigmoid`，**双眼融合成一个标量** | 多个模型输出取平均，同样**单标量** |
| 判定 | `blink = bool(p >= threshold)`，`threshold` 默认 **0.425**（`launch/estimate_blink.launch`；standalone 脚本 argparse 里写着 0.5，但构造时被硬编码成 0.425，实际生效的是 0.425） | 同左 |
| 模型规模 | `BlinkEstimationModelVGG16` = **2 × VGG16 特征分支**（去掉分类头）；`estimate_blink_pytorch.py` 支持传多个 ckpt 做 ensemble（`blink_model_pytorch_vgg16_allsubjects{1,2,3}.model`） | ensemble 默认 **2 个**：`blink_model_1.h5` + `blink_model_2.h5` |

关键点：**输出是"双眼"的单一概率**。它既不能给左右眼分别的分数，也不能给"闭眼持续时长"。我们现在用的 `min(eyeBlinkLeft, eyeBlinkRight)` 是**每眼连续 0~1 分级信号**，信息量严格多于它的单帧标量。

### 2.2 依赖栈（要落地的真实成本）

`rt_gene/README.md` §Requirements + `rt_gene/package.xml`：

- **ROS**（Kinetic/Melodic），`rospy` / `rospkg` / `cv_bridge` / `sensor_msgs`；
- 人脸检测：**S3FD**（`rt_gene/src/rt_gene/SFD`，来自 face-alignment）；
- 68 关键点：**3DDFA**（MobileNet 版，`phase1_wpdc_vdc.pth.tar`）+ 3D 模型点 `face_model_68.txt`；
- 另有 **dlib**（`pip install tensorflow-gpu numpy scipy tqdm torch torchvision Pillow dlib opencv-python`）；
- 推理设备默认 `/gpu:0`（`estimate_blink.launch`）。

`rt_bene_standalone/estimate_blink_standalone.py` **只接受"已经裁好的左右眼图片目录"** —— 仓库没有给出"整帧 → 眼 patch"的可直接复用入口；那段逻辑藏在 ROS 节点 `extract_landmarks_node.py` + `tracker_generic.py` 里（§5.2 已核对）。

### 2.3 权重来源与本机实测结果

`rt_gene/scripts/download_models.py` → `rt_gene/src/rt_gene/download_tools.py`：所有模型（含眨眼模型）都指向 imperialcollegelondon.**box.com** 的 legacy `shared/static/...` 链接。

本机实测（2026-09-27）：

```
HEAD https://imperialcollegelondon.box.com/shared/static/wwky1um443vgz9oy90zllv0s7474a5dj.model
  → HTTP 404
GET  -L 同上
  → 301 跳转到 https://imperialcollegelondon.app.box.com/public/static/…，随后连接失败（http=000）
```

**即：这份权重在当前环境里既没有镜像、也没有可用的下载地址。** 注意这与 `algorithm/requirements.txt:25` 记录的环境约束一致（GitHub release 与 huggingface 直连都不可达）。

### 2.4 数据集（这部分是可达的）

`https://zenodo.org/api/records/3685316` 实测 **HTTP 200**。内容：`rt_bene_subjects.csv` + 17 份 `s0xx_blink_labels.csv` + 17 个 `s0xx_noglasses_eyes.tar`（10–140 MB/个）。标注口径：**0.0 = 睁眼，1.0 = 眨眼，0.5 = 标注者分歧（训练时丢弃）**，并自带 3-fold 划分。

### 2.5 许可证（硬约束）

代码、权重、数据集**全部**是 **CC BY-NC-SA 4.0**：`非商业使用` + `相同方式共享`。仓库说明：*"Commercial usage is not permitted; please contact … regarding commercial licensing."*

对本平台的含义：

- 比赛/非商业演示：许可证上可用，但**必须在报告与界面注明来源与许可证**；
- 平台一旦走商业化（这是策划方案里的既定方向）：**直接禁止**；
- SA 条款对"衍生作品"的传染范围本身有解释空间，把它混进一个其余部分为 MIT/Apache 风格的代码库，是一个**长期的许可证孤岛**。平台的安全与合规文档里不应留这种含糊项。

### 2.6 与现链路的逐条对照

| 维度 | 现状（MediaPipe blendshape） | RT-BENE | 谁更好 |
|---|---|---|---|
| 信号语义 | 每眼连续 0~1 闭合度，`min(L,R)` 去单眼眨动 | 双眼融合单帧概率，阈值二值化 | **现状信息更多** |
| 粒度 | 每眼 | 只能给双眼一个数 | 现状 |
| 时间分辨率 | 由 `useFacialAnalysis` 的 100ms 定时器决定 | **同一个定时器决定** —— 换模型不改变 | 平 |
| 运行位置 | 浏览器内（WASM + GPU delegate），视频不出设备 | Python/ROS 进程；或需自己导出 ONNX | 现状（隐私） |
| 依赖 | `@mediapipe/tasks-vision`，已装、已跑 | ROS + S3FD + 3DDFA + dlib + torch/TF1 + GPU | 现状 |
| 权重可得性 | CDN 直接可取，已在跑 | **本机实测不可得** | 现状 |
| 许可 | Apache-2.0 | **CC BY-NC-SA 4.0** | 现状 |
| 单帧成本 | 与面部分析**共用同一次推理**，边际成本≈0 | 2×VGG16×若干模型/帧，需另开一路 | 现状 |
| 回归测试 | `tools/blink_tests`（24 项断言，直接编译真实 `blinkLogic.ts`） | 无 | 现状 |

---

## 3. 实施视角：如果硬要接，需要动什么

诚实列出（不是推荐，是成本盘点）：

1. 新写"整帧 → 左右眼 60×36 patch"的裁剪（§5.2 可复刻）+ 一组后端推理服务与 `/api/v1/perception/blink` 端点；
2. 改 `client/src/hooks/useFacialAnalysis.ts`（在已有 `detectForVideo` 结果之上裁眼 patch，并把**眼部图像**送到后端）或 `useEyeTracking.ts`（换信号源）；
3. 改 `blinkLogic.ts` 的阈值口径（连续度 → 概率）与 `tools/blink_tests` 的 24 项断言；
4. **新增隐私问题**：现在 `docs/perception_capabilities.md` 明确"共用同一路摄像头、同一份推理结果、不再单独开一路"，ASR 也写了"音频不出设备"。把眼部图像传给服务器与之直接冲突，且眼部图像属于生物特征数据，需要重新走知情同意（`backend/privacy` 那一层）；
5. 前端方案（ONNX + `onnxruntime-web`，该依赖已存在于 `client/package.json`）可以规避第 4 条，但要接受 NC 许可与随之而来的权重分发问题。

**结论：成本高、收益未验证、并引入许可与隐私两项新风险。**

---

## 4. 为什么"换成 RT-BENE 会更准"目前没有证据

1. **论文没有和 MediaPipe 比过。** RT-BENE（ICCVW 2019）的对比基线是 EAR、眼睛角点距离那一类传统方法；MediaPipe FaceLandmarker 的 52 个 blendshape（含 `eyeBlinkLeft/Right`）是 2021 年之后的东西。用一篇 2019 年的论文去论证"比我们现在的模型准"，不成立。
2. **训练域偏窄。** RT-BENE 标注的是 RT-GENE 数据集里 **"noglasses"（不戴眼镜）** 的部分。青少年用户里戴眼镜的比例不低，这正是这类模型的已知弱项。
3. **真正的瓶颈是采样率，不是分类器。** 这一点本仓库自己的审计已经写明：
   - `docs/video_perception_audit.md:15`：眨眼"**10fps 采样** 对 100–400ms 眨眼严重欠采样"；
   - `docs/call_implementation_steps.md:1310`：`minMs=60` 在 100ms 采样下**不可分辨**，`minGapMs=100` **等于**采样间隔，"这不是有点误差，是量纲级问题"；
   - `docs/video_perception_audit.md:1004`：欠采样的**实际比例至今未实测**，建议补一次对照实验。
   
   换任何一个"每帧分类器"都不会改变第 3 条 —— 而如果第 3 条修好了（提到 30fps），MediaPipe 这一路**本来就能测准**，就更没有理由引入一个 NC 许可的第三方权重。

---

## 5. 真正值得借用的两件事

### 5.1 RT-BENE 数据集（Zenodo，本机可达）

用途有两档，都不需要接受它的许可证以外的风险：

- **A 档（评测，零许可风险）：** 把标注当作 ground truth，给"现在这条 MediaPipe 眨眼链路"打指标。
  - 直接从 `s0xx_noglasses_eyes.tar` 里的**眼部 patch**入手，比"用整帧重跑 MediaPipe"更容易复现；
  - 产出：precision / recall / F1 + **在不同采样率下的召回衰减曲线**（10fps vs 30fps）—— 正好把 `video_perception_audit.md:1004` 的 `[未验证]` 变成实测数字。
  - 这一步的结论直接决定后面要不要做模型替换。**先做这个，别先换模型。**
- **B 档（自训 + 浏览器端，仅在 A 档证明 blendshape 不够时）：**
  - 用 60×36 输入训一个小骨干（MobileNetV3-small / ResNet18），导出 ONNX；
  - 用**项目已有依赖** `onnxruntime-web`（`client/package.json` 已列）在浏览器内推理 —— **视频依然不出设备**，绕开 §3.4 的隐私冲突；
  - 注意：用 RT-BENE 数据集训练出来的权重仍受 NC 约束，只能在比赛/非商业范围用，且报告必须标注来源；若要商用，需换许可更宽松的标注数据。

### 5.2 眼 patch 裁剪口径（可复刻，已核对源码）

`rt_gene/src/rt_gene/tracker_generic.py:28-83` 的 `get_eye_image_from_landmarks`：

- 只取 **68 关键点里的 4 个眼角**（索引 36/39/42/45）算宽度与中心；
- 按左右眼中心做**仿射对齐去掉 roll**（不涉及头部俯仰/偏航，**不需要 PnP 头姿**）；
- bbox 左右各留 `margin_ratio = 1.0 × 眼宽`，上下按宽高比 `(36/60)/2 = 0.3` 推；
- `cv2.resize` 到 **60×36**。

**这意味着用 MediaPipe 已有的角点（`useFacialAnalysis.ts:35-37` 的 `eyeOuterL/eyeInnerL/eyeOuterR/eyeInnerR` = 33/133/263/362）就能复刻同口径的眼 patch**，无需引入 3DDFA/S3FD/dlib。也就是说：技术上完全可复刻，"不推荐"的理由不是做不到，而是**许可证 + 收益未验证**。

---

## 6. 建议路线（按 ROI 排序）

| 优先级 | 路线 | 内容 | 预期工作量 |
|---|---|---|---|
| **P0** | A. 眨眼专用高频通道 | 只把**眨眼所需的那一路**从 10fps 提到 30fps（`docs/video_perception_audit.md:996` 已有同样结论：其余面部指标维持 5–10fps），并重调状态机时间参数（`minMs` 下调、`minGapMs` 拉开、时长全部按时间戳而非帧计数） | 1–2 天 |
| **P0** | B. 用 RT-BENE 标注做离线评测 | 量化现有链路在不同采样率下的 P/R/F1，补上仓库里那条 `[未验证]` | 0.5–1 天 |
| P2 | C. 自训小模型 + ONNX | 仅当 B 证明 blendshape 不够；保持浏览器内推理 | 3–5 天 |
| **不建议** | D. 直接用 RT-BENE 权重 | 权重本机不可得、ROS/3DDFA/dlib 依赖、NC 许可、眼部图像上传破坏隐私承诺 | — |

**A + B 加起来能回答"我们的眨眼到底准不准"这个真问题，成本低于 D 的十分之一。**

---

## 7. 本机环境实测记录（可复现）

| 目标 | 结果 |
|---|---|
| `github.com` / `raw.githubusercontent.com` | **实际可用**：hosts 里确实把这两个域名指到 127.0.0.1，但本机有加速器（Watt Toolkit）在 127.0.0.1:443 上做反代，实测 `Invoke-WebRequest https://github.com/...` 与 `git clone` 均返回 200（**本文档第一版误判为"不可用"，此处更正**） |
| `huggingface.co` 直连 | 仍被 hosts 屏蔽；**走 `hf-mirror.com` 可用**（HTTP 200），ASR 模型就是这么下的 |
| `cdn.jsdelivr.net` | **可用**（本次即用它取回 rt_gene@5.0.0 源码；前端加载 MediaPipe wasm 也走它） |
| `zenodo.org`（RT-BENE 数据集） | **可用**，HTTP 200 |
| `storage.openvinotoolkit.org`（open-closed-eye-0001 权重） | **可用**，HTTP 200，46 164 B，SHA-384 与官方 model.yml 完全一致 |
| `mrl.cs.vsb.cz`（MRL Eye 数据集） | **可用**，HTTP 200 |
| `imperialcollegelondon.box.com`（RT-BENE 权重） | **不可用**：HEAD 404；GET 301 → app.box.com 后连接失败（再测仍 404）。这是仓库 README 里唯一的权重来源，**官方没有镜像** |
| `D:\Develop\AIC\.venv` | 有 `torch 2.14.0+cpu` / `numpy` / `fastapi`；**无** `torchvision` / `cv2` / `mediapipe`；`pip` 原缺，已用 `python -m ensurepip` 装回，并装了 `onnxruntime 1.30.0` + `pillow` 用于离线评测 |

---

## 8. 引用

- 仓库：<https://github.com/Tobias-Fischer/rt_gene>（tag `5.0.0`）
- RT-GENE 论文：Fischer, Chang, Demiris, *RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments*, ECCV 2018
- RT-BENE 论文：Cortacero, Fischer, Demiris, *RT-BENE: A Dataset and Baselines for Real-Time Blink Estimation in Natural Environments*, ICCVW 2019（[CVF 开放版](https://openaccess.thecvf.com/content_ICCVW_2019/html/GAZE/Cortacero_RT-BENE_A_Dataset_and_Baselines_for_Real-Time_Blink_Estimation_in_ICCVW_2019_paper.html)）
- RT-BENE 数据集：<https://zenodo.org/records/3685316>（CC BY-NC-SA 4.0）
- QUT 索引页：<https://open.qcr.ai/code/rt_gene_code/>、<https://open.qcr.ai/code/rt_bene_code/>
- 本仓库相关审计：`docs/perception_capabilities.md` §六、`docs/video_perception_audit.md` §2.2 与 §996/§1004、`docs/call_implementation_steps.md:1302-1311`

---

## 9. 附件：本次核对所取回的 rt_gene@5.0.0 源码

存放于 **`D:\Develop\AIC\tools\rt_gene_probe\`**（在 git 仓库之外，**不属于本仓库**，仅作核对证据）：

```
rt_gene__README.md                                       (requirements / license / library 列表)
rt_gene__package.xml                                     (ROS 依赖)
rt_gene__scripts__estimate_blink.py                      (ROS 节点：订阅 /subjects/images → 发布 /subjects/blink)
rt_gene__scripts__download_models.py                     (权重下载入口)
rt_gene__launch__estimate_blink.launch                   (backend=tensorflow, 2 个模型, threshold=0.425)
rt_gene__src__rt_bene__estimate_blink_pytorch.py         (60x36 + ImageNet 归一化 + sigmoid)
rt_gene__src__rt_bene__estimate_blink_tensorflow.py      (96x96 + 右眼翻转 + 顺序 [right,left])
rt_gene__src__rt_bene__blink_estimation_models_pytorch.py(VGG16/ResNet18/50/VGG19/DenseNet121 双分支)
rt_gene__src__rt_bene__estimate_blink_base.py            (threshold / overlay)
rt_gene__src__rt_gene__download_tools.py                 (box.com 权重地址 + md5 白名单)
rt_gene__src__rt_gene__extract_landmarks_method_base.py  (S3FD + 3DDFA, eye_image_size=(60,36))
rt_gene__src__rt_gene__tracker_generic.py                (get_eye_image_from_landmarks：眼 patch 裁剪口径)
rt_gene__src__rt_gene__gaze_tools.py                     (get_normalised_eye_landmarks 等)
rt_bene_standalone__estimate_blink_standalone.py         (只接受裁好的左右眼图片目录)
rt_bene_model_training__README.md / rt_bene_standalone__README.md
```

⚠️ 该目录内文件受 **CC BY-NC-SA 4.0** 约束，**不得**移入或复制进本仓库；本文件（§2、§5.2）只做事实转述与路径引用，未复制其代码。
