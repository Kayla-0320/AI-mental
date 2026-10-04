# 眨眼信号源选型 —— GitHub 项目调研与「借鉴落地」报告

> 配套文档：`docs/rt_gene_blink_replacement_assessment.md`（RT-GENE / RT-BENE 为何**不**做替换件）
> 本文回答的是它的下一步：**那到底借鉴什么、用哪个模型、接了会怎样。**
> 出具时间：2026-09-27　评估人：编码 agent（所有数字均标注实测 / 声称）
>
> **2026-09-27 更新：本文的步 A/B 已实施**，见 §8「实施记录」——
> 眨眼通道已提到 30fps（自适应降档）、时长量化粒度已如实上报、
> 第二判定通道已接入（默认关闭）。风险公式**未改动**。

---

## 0. 结论摘要

1. **不采用 RT-BENE 权重**（许可证 CC BY-NC-SA + 权重链接实测失效），但**采用它的两样东西**：
   - **眼 patch 裁剪口径** → 已重写为 TypeScript 并落地：`client/src/hooks/eyePatch.ts`（+22 项断言）；
   - **RT-BENE 标注数据集**（Zenodo 实测可达）→ 已作为**离线评测集**接入：`tools/blink_eval/`。
2. **替代模型从 GitHub 另选，且都是宽松许可证**，不碰 RT-BENE 的 NC 权重。
   下表的"用途定位"是**按 §3.2 的实测结果**给的（不是按 README 自评分）：
   | 候选 | 许可证 | 体积 | 输入 | 用途定位 |
   |---|---|---|---|---|
   | [OMZ open-closed-eye-0001](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/open-closed-eye-0001) | **Apache-2.0** | **46 KB** | 32×32 **BGR** → softmax[?, ?] | **实测两个受试者都最稳 → 首选主信号** |
   | [PINTO0309/OCEC](https://github.com/PINTO0309/OCEC) | **MIT** | 115 KB – 6.4 MB | 24×40 RGB → `prob_open` | 第二意见 / 交叉校验（**阈值必须重标定**） |
   | [fodorad/BlinkLinMulT](https://github.com/fodorad/BlinkLinMulT) | **MIT** | 20–32 MB（模型） | 64×64 时序窗口 | **只借鉴其时序算法**，不引模型 |
3. **一个重要实测纠正**：`open-closed-eye-0001` 的**类别顺序与官方 model card 相反**（见 §3.3）——
   照文档接会把"睁眼"当"闭眼"，AUC 从 0.918 掉到 0.082（即完全反向）。这类事情**只能靠实测发现**。
4. **另一个重要实测纠正**：现行 `blinkRate` 的误差方向被仓库文档**高估**了。用**真实状态机**跑下来，
   10 fps 对"眨眼**事件计数**"的影响约 **-5%**（不是"量纲级"），真正的失真在**时长指标**上：
   10 fps 下闭眼时长只能取 100 ms 的整数倍，而 `avgBlinkDuration` / `longBlinkCount(>400ms)`
   正是 `anxietyIndex` 的加分项（见 §3.4）。
5. **两处仍未验证**（不要当成已完成）：视频级标注数据集（Eyeblink8 / TalkingFace / RN / CEW）
   本次**全部不可达**，因此"真实用户身上 10fps 到底漏多少"仍然没有外部真值；
   浏览器内 ONNX（onnxruntime-web）**尚未实跑**。

---

## 1. GitHub 眨眼项目调研

**许可证分诊是第一道筛子**（本平台已明确要走商业化，NC 一律出局）：

| 项目 | 许可证 | 任务/接口 | 权重 | 浏览器可行 | 结论 |
|---|---|---|---|---|---|
| PINTO0309/OCEC | **MIT**（LICENSE 实读） | 每眼二分类，**逐帧分类器，无时序逻辑**；`images[N,3,24,40]` → `prob_open[N]` sigmoid | 6 个 ONNX 全部 HTTP 200（115 097 / 176 580 / 494 914 / 875 566 / 1 693 332 / 6 372 200 B） | ✅ 115 KB 极合适 | 采用为第二意见（§3.2 跨域 AUG 偏低，阈值须重标定） |
| OMZ open-closed-eye-0001 | **Apache-2.0** | 每眼二分类逐帧；`input.1[1,3,32,32]` BGR → `19[1,2,1,1]` softmax | ONNX **46 164 B**，HTTP 200，**SHA-384 与官方 model.yml 完全一致** | ✅ 46 KB | **采用为首选主信号**（§3.2 两个受试者都最稳） |
| fodorad/BlinkLinMulT | **MIT** | 逐帧打分 **+ 时序事件抽取（迟滞）**；64×64、1.5 s 窗 | HF `fodorad/blink_detection` 可达（20–32 MB） | 可行但偏重 | **仅借鉴算法** |
| OpenFace | **非商业**（"NONCOMMERCIAL RESEARCH USE ONLY"） | AU45 眨眼强度 + 事件 | — | ❌ | **淘汰（许可证）** |
| Eye-LRCN | — | LRCN 眨眼完整性 | **无官方代码/权重** | — | 淘汰（不存在） |
| Pushtogithub23 / imprvhub 等 EAR 系 | MIT | MediaPipe/dlib EAR 阈值，**无状态机** | 无（只用关键点） | ✅ | 不采用（我们已有更完整的链路） |
| dilawar/eye-blink-detector | **无 LICENSE** | dlib EAR | 无 | — | 淘汰（无许可证 = 不可用） |

**唯一值得借鉴的"时序算法"**：BlinkLinMulT 的 `blinklinmult/train/events.py::to_intervals`
（MIT，约 20 行）：一段信号必须先**超过 high 阈值**才算候选，随后只要**不低于 low 阈值**就延续。
它的 high/low 迟滞设计与我们 `blinkLogic.BlinkStateMachine` 的 `on/offThreshold` **同构**，
但它**没有**最短/最长时长过滤与最小间隔去抖（全仓 grep 无 `min_gap`/`min_duration`/`debounce`）——
也就是说：**我们的状态机比它更完整，无需替换**；它的价值在于可作为迟滞部分的第三方交叉验证。

---

## 2. 数据集可达性（真值来源）

| 数据集 | 实测状态 | 能用来做什么 |
|---|---|---|
| **RT-BENE**（Zenodo 3685316） | **HTTP 200**；CC BY-NC-SA 4.0 | **每眼逐帧标注**，评"分类器准不准" ✅ |
| **MRL Eye** | HTTP 200，341 866 898 B；近红外灰度、静态图、**无眨眼事件** | 只适合训练/评测"睁闭眼分类"，不适合眨眼事件 |
| Eyeblink8 | **DNS 解析失败** | — |
| TalkingFace / Researcher's Night | 连接失败 / 超时 | — |
| CEW | HTTP 403 | — |
| HF `MichalMlodawski/closed-open-eyes` | HTTP 200，ODC-By（宽松） | OCEC 的训练源；如需自训可用 |
| HF `MichalMlodawski/open-closed-eye-classification-*` | 可达但 **CC BY-NC-ND 4.0** | ❌ 淘汰（NC + 禁衍生） |

> ⚠️ **RT-BENE 不能用来评"采样率够不够"**：实测受试者 s012 的帧号跨度 7 239 而只给了 935 帧
> （dense_ratio = 0.129），说明它是从视频里**抽帧**得到的，帧号不是可靠时间轴。
> 想测采样率必须要有**逐帧连续**的视频级标注 —— 而上面那四个视频级数据集本次**全部不可达**。
> 这就是 §3.4 只能用**合成序列**做机制演示、而不能给出真实人群数字的原因。

---

## 3. 本次实测

### 3.1 评测方法

- 数据：RT-BENE 眼部 patch（60×36 RGB，实测确认尺寸），标注 0.0 睁眼 / 1.0 眨眼 / 0.5 分歧（丢弃）。
- 脚本：`tools/blink_eval/eval_eye_patch_models.py`（onnxruntime CPU，批推理）。
- 指标：以**闭眼为正类**（漏检比误检更伤：漏检 → blinkRate 偏低 → 误报"解离/回避"），
  报 Accuracy / 闭眼 P·R·F1 / **AUC**（阈值无关）/ 阈值扫描的最优 F1。
- 预处理严格按各自官方定义复刻（`ovec` 用 BGR +(x−127)/255；`ocec` 用 RGB + x/255、Resize(24,40)）。

### 3.2 结果（**初步**：2 个受试者、172 个闭眼帧）

分受试者（用来暴露方差）：

| 受试者 | 模型 | Acc | 闭眼 P | 闭眼 R | 闭眼 F1 | **AUC** |
|---|---|---|---|---|---|---|
| s012（935 张，闭眼 23） | open-closed-eye-0001（46 KB, Apache-2.0） | 0.9583 | 0.289 | 0.478 | 0.361 | **0.9179** |
| s012 | OCEC-p（115 KB, MIT） | 0.6706 | 0.059 | 0.826 | 0.110 | 0.8754 |
| s012 | OCEC-n（177 KB, MIT） | 0.7102 | 0.069 | 0.870 | 0.129 | 0.9188 |
| s007（1857 张，闭眼 149） | **open-closed-eye-0001** | 0.9472 | 0.780 | 0.477 | 0.592 | **0.9738** |
| s007 | OCEC-p | 0.4949 | 0.118 | 0.819 | 0.206 | 0.7618 |
| s007 | OCEC-n | 0.4987 | 0.114 | 0.779 | 0.199 | 0.7347 |

合并两个受试者（2792 张，闭眼 172；`tools/blink_eval/result_2subjects.md`）：

| 模型 | Acc | 闭眼 P | 闭眼 R | 闭眼 F1 | **AUC** | 最优阈值 / F1 | recall≥0.9 的阈值 |
|---|---|---|---|---|---|---|---|
| **open-closed-eye-0001** | 0.9509 | 0.636 | 0.477 | 0.545 | **0.9608** | 0.02 / 0.604 | **无（达不到 0.9）** |
| OCEC-p | 0.5537 | 0.104 | 0.820 | 0.185 | 0.7953 | 0.99 / 0.358 | 0.22（precision 仅 0.097） |
| OCEC-n | 0.5695 | 0.104 | 0.791 | 0.185 | 0.7844 | 0.99 / 0.335 | 0.18（precision 仅 0.091） |

**读法（重要）：**

1. **闭眼帧只占 1.8%–2.5%**，所以阈值 0.5 下的 precision 天然很低，**F1 不是合适的比较量，AUC 才是**。
2. **受试者间方差极大**：同一个 OCEC-n 在 s012 上 AUC 0.919、在 s007 上只有 0.735；
   open-closed-eye-0001 则从 0.918 升到 0.974。**单个受试者的结论不可外推**，
   这也是本文坚持把"样本量/受试者数"写在每张表里的原因。就现有数据看，
   **open-closed-eye-0001 在两个受试者上都更稳**，且它只有 46 KB + Apache-2.0。
3. **两个模型在跨域数据上都没有可用阈值**：
   - OCEC 在 0.5 阈值下几乎把一切都判成"闭眼"（s007 上 Acc 只有 0.49）——它的 `prob_open`
     在这批数据上系统性偏低，**必须按实际摄像头数据重新标定**；
   - open-closed-eye-0001 的闭眼分数集中在 0 附近（最优阈值被扫到 0.02），
     而且**没有任何阈值能把闭眼召回做到 0.9** —— 约 15–20% 的"标注为闭眼"的帧，
     模型在任何阈值下都判为睁眼。
4. 第 3 条的第 2 个现象有一个**可能的解释**（未证实）：RT-BENE 的闭眼段常常只有 1–3 帧连续，
   标注很可能只覆盖**闭合峰值**而不是整个闭眼区间，同时这些帧本身可能因为运动模糊/半闭眼
   而难以判定。这既影响分类器评测，也影响"时长"类结论。

**因此 §5 的接入顺序里，B 步（只上报不进风险）不是可选项，而是必须项**：
在拿到实际摄像头数据上的分数分布之前，任何阈值都是猜的。



### 3.3 ⚠️ 实测纠正：`open-closed-eye-0001` 的类别顺序与官方文档相反

OpenVINO 的 model card 写输出是 softmax **[open, closed]**。实测在同一批数据上：

| 取哪个下标当"闭眼" | AUC |
|---|---|
| index 0 | **0.9179** |
| index 1（照文档） | **0.0821** |

`0.0821 = 1 − 0.9179`，即**完全反向**。也就是说在这批数据上 **index 0 才是"闭眼"侧**。
脚本已把默认值设成实测得到的 index 0，并留 `--ovec-closed-index` 开关便于复核。
**这条也说明：模型卡/README 的接口描述必须用数据验收，不能直接接。**

### 3.4 采样率影响的实测（用**真实** `BlinkStateMachine`）

脚本：`tools/blink_eval/sampling_effect.js`（直接 require `tools/blink_tests` 编译出的**真实** `blinkLogic.js`，
参数就是线上的 `BLENDSHAPE_PARAMS`）。对 30 fps 序列按 1/2/3/6 抽稀，模拟 30/15/10/5 fps：

| 闭眼信号形状 | 30 fps | 15 fps | 10 fps | 5 fps |
|---|---|---|---|---|
| **ramp**（接近真实 blendshape 过渡）· 事件召回 | 100% | 100% | **95.5%** | 85.1% |
| ramp · 平均检出时长 | 190 ms | 189 ms | 204 ms | 239 ms |
| ramp · 检出时长量化粒度 | 33 ms | 67 ms | **100 ms** | 200 ms |
| **binary**（理想二值上界）· 事件召回 | 100% | 100% | **100%** | 95.0% |

**三条结论：**

1. **事件计数**：10 fps 的损失约 **5%**（ramp 情形），远小于"量纲级问题"的描述；
   在理想二值信号下甚至为 0%。仓库文档 `video_perception_audit.md:1004` 的 `[未验证]`
   可以据此下调紧迫性 —— 但仍需真实视频级数据确认（见 §2 的警告）。
2. **时长指标才是真问题**：10 fps 下闭眼时长只能是 100 ms 的整数倍，`avgBlinkDuration`
   与 `longBlinkCount(>400ms)` 随之失真，而这两项**直接给 `anxietyIndex` 加分**
   （`blinkLogic.ts:158-161`，`>2` 次长眨眼即加最多 30 分）。
   要在 10 fps 下保住时长精度，靠换模型没用，**只能提高采样率**（或改用帧级时间戳做插值）。
3. 所以正确顺序是：**先把眨眼这一路的采样率提上去（或先砍掉它对风险的贡献），再谈换不换分类器。**

> ⚠️ 这两行数字是**合成序列上的机制演示**，不是对真实用户的测量。合成参数：眨眼 100–400 ms、
> 间隔 2.5–5.5 s、ramp 用 `sin^0.6` 起落 + 2% 噪声、固定随机种子 42（可复现）。
> 拿到逐帧连续标注（Eyeblink8/自录片段）后，用同一条命令即可得真实数字。

---

## 4. 已落地的借鉴物（文件清单）

| 文件 | 内容 | 许可证/来源 | 状态 |
|---|---|---|---|
| `client/src/hooks/eyePatch.ts` | 复刻 RT-BENE 眼 patch 口径（角点→去 roll→margin 1.0→宽高比推出框高），纯逻辑、零 DOM | 口径借鉴自 RT-GENE/RT-BENE（**代码为 TS 重写，未复制**） | **尚未接入主链路**（文件头已标注） |
| `client/src/hooks/eyePatchCanvas.ts` | 把几何真正裁成像素（canvas 仿射），供浏览器内 ONNX 使用；视频不出设备 | 同上 | 同上 |
| `tools/blink_tests/eyePatch.test.js` | **22 项断言**（口径/roll/仿射/退化输入/分辨率无关性） | 自研 | ✅ 22/22 通过 |
| `tools/blink_tests/run.ps1` | 现在同时编译并跑 `blinkLogic`（24 项）+ `eyePatch`（22 项） | 自研 | ✅ 原 24 项无回归 |
| `tools/blink_eval/fetch_assets.ps1` | 一键拉取模型 + 数据集（含 open-closed-eye 的 **SHA-384 校验**与 RT-BENE 标注） | — | ✅ 幂等，实测通过 |
| `tools/blink_eval/eval_eye_patch_models.py` | 候选分类器离线评测台（含类别顺序开关、阈值扫描、分数导出） | — | ✅ 已跑出真实数字 |
| `tools/blink_eval/sampling_effect.js` | 用**真实**状态机测不同采样率下的事件召回/时长失真 | — | ✅ 已跑出真实数字 |
| `docs/rt_gene_blink_replacement_assessment.md` | RT-GENE/RT-BENE 评估（含 §7 环境实测**更正**：GitHub 经本机加速器可用） | — | ✅ |

跑一遍全套（零新增 npm 依赖；Python 侧只需 `onnxruntime` + `pillow`，已装入 `.venv`）：

```powershell
powershell -File D:\Develop\AIC\tools\blink_tests\run.ps1          # 46 项断言
powershell -File D:\Develop\AIC\tools\blink_eval\fetch_assets.ps1 # 模型 + 标注（-WithPatches 再加眼 patch）
D:\Develop\AIC\.venv\Scripts\python.exe D:\Develop\AIC\tools\blink_eval\eval_eye_patch_models.py `
    --data-dir D:\Develop\AIC\tools\blink_eval\data `
    --models ocec_n=...\ocec_n.onnx ovec=...\open_closed_eye.onnx
node D:\Develop\AIC\tools\blink_eval\sampling_effect.js --synthetic --shape ramp
```

---

## 5. 接入方案（推荐组合与顺序）

**推荐链路（全部宽松许可证、全程浏览器内、视频不出设备）：**

```
MediaPipe FaceLandmarker（已在跑，同一帧、不新增推理）
  → eyePatch 裁左右眼（open-closed-eye 用 32×32；OCEC 用 24×40，同一次裁剪换目标尺寸即可）
  → open-closed-eye-0001（Apache-2.0, 46 KB）每眼闭眼概率
     └─ 可选：OCEC（MIT, 115 KB）做第二意见，两者不一致时该帧标记为低置信
  → blinkLogic.BlinkStateMachine（保留现有迟滞/时长窗/最小间隔，不替换）
  → blinkRate / avgBlinkDuration / longBlinkCount
```

**建议按三步走，每一步都能单独验收：**

| 步 | 动作 | 为什么先做 | 是否影响风险链路 |
|---|---|---|---|
| **A** | 先做 §3.4 的**真实数据**版测量：录一段 30 fps、带眨眼标注的自录片段（≥3 分钟、≥50 次眨眼），跑 `sampling_effect.js --csv` | 决定"要不要提高采样率"这个更便宜的问题，避免先引模型 | 否 |
| **B** | 把眨眼信号源换成 open-closed-eye-0001（`onnxruntime-web` 已是现有依赖），**先只上报、不进 riskScore**（`blinkMethod: 'onnx-ovec'`，风险权重暂时置 0） | 拿到线上真实分布与阈值，同时不动安全逻辑 | 否（只上报） |
| **C** | 依据 B 的实测分布重定阈值，再把眨眼并入 `riskScore` | 这一步会改变风险与落库内容 | **是 → 按 `AGENTS.md` 需人工审核** |

**必须写进代码的三条诚实性约束（沿用现有风格）：**

- `blinkMethod` 增加 `'onnx-ocec'` / `'onnx-ovec'` 取值，界面与 `evidence` 如实标注来源；
- ONNX 会话加载失败/推理异常时**必须降级回 blendshape**，不得静默返回 0（现有代码对"缺失即 null"已有此约定）；
- `eyePatch` 返回 `null`（关键点退化）或 patch 越界时，该帧**不计入**眨眼统计，并让 `signalQuality` 反映出来。

---

## 6. 许可证清单（合规必读）

| 资产 | 许可证 | 商用 | 备注 |
|---|---|---|---|
| OCEC 模型与代码 | **MIT** | ✅ | 保留版权声明即可 |
| open-closed-eye-0001 | **Apache-2.0** | ✅ | 保留 NOTICE/LICENSE |
| BlinkLinMulT 代码 | **MIT** | ✅ | 仅借鉴算法，若抄函数需保留声明 |
| MediaPipe tasks-vision | Apache-2.0 | ✅ | 现状 |
| RT-BENE **数据集与标注** | **CC BY-NC-SA 4.0** | ❌ | **只用于非商业评测**；数据**不得进仓库**（现存放于 `tools/blink_eval/data`，在 git 仓库之外）；报告须标注来源与论文引用 |
| RT-GENE / RT-BENE **代码与权重** | CC BY-NC-SA 4.0 | ❌ | 本次**未采用**（我们只借鉴裁剪口径并自行重写） |
| OpenFace | 非商业（CMU） | ❌ | 已淘汰 |
| HF `open-closed-eye-classification-*` | CC BY-NC-**ND** 4.0 | ❌ | 已淘汰（NC + 禁衍生） |

---

## 7. 未能验证 / 未做的事（不要当成已完成）

1. **真实视频级标注**（Eyeblink8 / TalkingFace / Researcher's Night / CEW）全部不可达
   → §3.4 只有合成机制演示，**没有真实人群的采样率损失数字**。建议自录片段补上（步 A）。
2. **浏览器内 ONNX 未实跑**：`onnxruntime-web` 的算子兼容性只是静态排查（Conv/Relu/Gemm/Sigmoid 等
   均为 ORT-Web 支持算子），**没有在浏览器里跑过**；模型也还没放进 `client/public/`。
3. **OCEC 的 F1 0.992+ 是 README 声称**，且其训练语料未公开 → **不可与 MRL/CEW 基准比较**；
   本文所有准确率数字都是在 **RT-BENE patch** 上实测的，与它的自评口径不同。
4. **受试者覆盖不足**：§3.2 目前只有 2 个受试者（s012 + s007）、172 个闭眼帧，
   而且 s012 的闭眼帧只有 23 个。**"首选 open-closed-eye-0001"这个结论受样本限制**，
   补测更多受试者（脚本已支持，见 `fetch_assets.ps1 -WithPatches`）后需要复核。
5. **本文的评测喂的是 RT-BENE 自带的 60×36 眼图**，不是我们 `eyePatch` 现场裁出来的眼图。
   也就是说 §3.2 的 AUC 是**相对乐观的上界**：线上还叠了一层"我们的裁剪口径与训练分布是否一致"
   的域偏移。要量它，得跑一次端到端（自录片段 → MediaPipe → eyePatch → ONNX → 与人工标注比对）。
6. **未做**：`useFacialAnalysis` / `useEyeTracking` 的任何改动（换信号源会改变风险输出，
   按仓库约定需人工审核）；`client/public/models/` 未添加任何模型文件（避免留下未使用的死资源，
   接入时再把 `open_closed_eye.onnx`（46 KB）放进去即可）。
7. **RT-BENE 标注语义**：s012 的闭眼段只有 1–3 帧连续，可能标注的是**闭合峰值**而非整个闭眼区间，
   这会同时影响分类器评测与"时长"结论；未与数据集论文核对。
8. **Zenodo 限流**：补下载更多受试者时被限流（响应头出现 `retry-after`，实测约 50 KB/s 且频繁中断），
   因此本次只完成 2 个受试者。`fetch_assets.ps1 -WithPatches` 已支持断点续传（`curl -C -`），
   网络好的时候重跑即可补齐。

---

## 8. 实施记录（2026-09-27）

### 8.1 改了什么（按"准确 → 实用"排序）

| # | 改动 | 文件 | 默认状态 |
|---|---|---|---|
| 1 | **眨眼通道 10fps → 30fps**：把「每拍都做的事」（blendshape/EAR/头姿 → 下发）与「每 3 拍才做的事」（AU/情绪映射 + React setState + 眼 patch）拆成两层 | `useFacialAnalysis.ts`、`blinkChannel.ts`（新） | **开启** |
| 2 | **自适应降档 30 → 15 → 10fps**：按「推理 + 下发」的实测耗时判断，连续 3 次超预算降档、连续 10 次很轻松升档；跑不动时如实上报实际帧率，而不是让 setInterval 排队把时间戳搞乱 | `blinkChannel.ts` | **开启** |
| 3 | **如实上报时长量化粒度**：`blinkChannelFps` / `blinkDurationQuantumMs` / `blinkSampleIntervalMs` 进入 `EyeMovementAnalysis` | `useEyeTracking.ts`、`multimodal.types.ts` | **开启** |
| 4 | **修正 `signalQuality` 的假满值**：原来 `EXPECTED_FPS = 5` 而实际 10fps，覆盖率恒被夹到 1；现在按实测帧率算，跑到 15fps 就如实折半 | `useEyeTracking.ts` | **开启** |
| 5 | **修掉一处潜伏 bug**：`lastEndMs === 0` 被当成"上一次眨眼结束于 t=0"，会把会话开头的第一次眨眼按"间隔不足"丢掉 | `blinkLogic.ts` | **开启** |
| 6 | **亚采样插值**（用阈值穿越点估时长，把量化误差从 ±一个采样间隔降到 ±半帧） | `blinkLogic.ts` | **关闭**（见 8.3） |
| 7 | **第二判定通道**：眼 patch → 浏览器内 ONNX（`open-closed-eye-0001`，46 KB，Apache-2.0）→ 每眼闭眼概率 + 与 blendshape 的一致率 | `eyeStateClassifier.ts`、`useEyeStateModel.ts`、`eyePatch*`、`AnxietyContext.tsx` | **关闭**（env 开关） |
| 8 | **性能**：绿通道均值画布改为复用（原来每帧 `createElement('canvas')`；10fps 时每秒 10 个垃圾对象，提到 30fps 后会变成 30 个） | `useFacialAnalysis.ts` | **开启** |

**风险公式逐行未动**（`useEyeTracking.ts` 里那 4 个 `riskScore +=` 与优化前完全一致），
本次改的是**采样密度**与**上报口径**，不是判定规则或权重 —— 这是刻意的边界。

### 8.2 实测收益（用真实状态机 + 解析真值）

`tools/blink_eval/sampling_effect.js --synthetic --shape ramp`（眨眼 100–400 ms、间隔 2.5–5.5 s、
固定种子 42；真值由 sin^0.6 波形与 0.5/0.35 阈值解析求出）：

| 配置 | 事件召回 | 平均时长 | 时长量化粒度 |
|---|---|---|---|
| 优化前：10fps，无插值 | 92.1% | 204 ms | **100 ms** |
| 优化后：**30fps**，无插值 | **95.7%** | 190 ms | **33 ms** |
| 30fps + 插值（未开启） | 95.7% | 183 ms | 33 ms |

结论：**时长分辨率提升 3 倍（100 ms → 33 ms）、事件召回提升约 3.6 个百分点**，
且这两项直接决定 `avgBlinkDuration` / `longBlinkCount(>400ms)` 的失真程度，
而它们给 `anxietyIndex` 加分（>350 ms 加 15 分、每次长眨眼最多 30 分）。

> ⚠️ 这些是**合成序列上的机制演示**，不是真实人群的实测；真实数字需要自录带标注片段
> （步 A：`node sampling_effect.js --csv <导出的分数> --fps 30`）。

### 8.3 为什么亚采样插值默认关闭（一次诚实的自我否决）

插值写完、测完之后**分桶诊断**发现它不是普遍更优（`tools/blink_eval/_diag_interp.js`）：

| 真实闭眼时长 | 30fps 原始 MAE | 30fps 插值 MAE | 谁更好 |
|---|---|---|---|
| 0–150 ms | 43.2 ms | 42.5 ms | 插值（微弱） |
| 150–250 ms | 29.0 ms | 39.1 ms | **原始** |
| 250–400 ms | 25.1 ms | 34.6 ms | **原始** |

原因：旧写法的两个端点误差**方向相反、会互相抵消**（进入时刻被推后→偏短，退出时刻被推后→偏长），
插值缩小了每个端点的误差，却打断了这个抵消；而 `sin^0.6` 型响应在粗采样下本来就不是线性的。
在**分段线性**过渡上插值确实把 MAE 压低约 25–33%（`tools/blink_tests/blinkLogic.test.js` 有断言），
但真实 blendshape 属于哪一类**必须用真实数据判定**。

因此：**代码保留、测试保留、默认关闭**（`BlinkParams.interpolateDurations`），
等步 A 的真实数据出来再决定是否打开。这比"默认打开、看起来更先进"更负责。

### 8.4 第二判定通道的现状与启用方式

- 模型：`client/public/models/open_closed_eye.onnx`（46 164 B，SHA-384 与官方 model.yml 一致，
  出处与许可证见同目录 `NOTICE-open-closed-eye-0001.txt`）。
  ⚠️ 该目录被 `.gitignore` 的 `*.onnx` / `models/` 规则覆盖，**已在 `.gitignore` 末尾显式放行**——
  否则部署后模型缺失、功能静默失效。
- 启用：`VITE_ENABLE_EYE_STATE_MODEL=1`。默认关闭的两个理由写在
  `perceptionCapabilities.EYE_STATE_MODEL_AVAILABLE` 的注释里（13.6 MB wasm 运行时 + 需先标定阈值）。
- 输出：`eyeStateModelClosedBoth/Left/Right`、`eyeStateModelAgreement`、`eyeStateModelLatencyMs`、
  `eyeStateModelStatus`。**只上报，不进 `riskScore`**。
- 类别顺序按实测取 index 0 = 闭眼（官方 model card 标注相反，见 §3.3），
  并有 11 项断言把"BGR 通道序 + (x−127)/255 + index 0"钉住（`eyeStateClassifier.test.js`）。

### 8.5 验证结果（本次全部实跑）

| 检查 | 结果 |
|---|---|
| 纯逻辑断言 `tools/blink_tests/run.ps1` | **87 项全过**（blinkLogic 37 + blinkChannel 17 + eyePatch 22 + eyeStateClassifier 11） |
| 客户端全量类型检查 `tsc --noEmit -p tsconfig.json`（非增量、整项目） | **exit 0** |
| 新增/修改模块单独用 esbuild 打包（7 个文件） | **全部 OK**（排除 react/onnxruntime 外部依赖） |
| 生产构建 `npm run build`（`tsc -b && vite build`） | **成功过一次**：exit 0、3996 modules、dist 内含 `index.html` 与 `models/open_closed_eye.onnx` |
| `git check-ignore` 模型文件 | 已被 `!` 规则放行（可入库） |

> ⚠️ **构建的环境不稳定性（与本次代码无关）**：后续重跑 `npm run build` 多次在 **chunk 渲染阶段**
> 以 `System.OutOfMemoryException` / SIGABRT(134) / esbuild 服务被杀 结束，
> 且当前机器只剩约 7.8 GB 可用内存（有其它重任务在跑）。
> 证据表明不是本次代码引起：① 同样的代码完整构建成功过一次；
> ② 全项目 `tsc --noEmit` 干净通过；③ 本次新增/修改的 7 个模块单独 esbuild 打包全部成功。
> 因此 **dist 目前是半成品（public 资源已拷、index.html 缺失），内存宽裕时需要重新构建一次**。
> 另：首次构建时报的 `useVoiceCall.ts` 类型错误属于**别人未提交的在写文件**（`?? useVoiceCall.ts`），
> 后续已由该文件的作者修好，本次未改动它。

> ⚠️ 构建产物里有 `dist/assets/ort-wasm-simd-threaded.jsep-*.wasm` 约 **28 MB** —— 这是
> **既有问题**，不是本次引入：`pages/patient/Profile.tsx` 早已 import `useOnDeviceInference`，
> 而它会 `import('onnxruntime-web')`，于是 Vite 一直把这个 wasm 打进 dist。
> 要真正瘦身需要单独处理（例如 `optimizeDeps.exclude` + 运行时 CDN，或把该 Hook 改成按需加载）。

### 8.6 步 A 的执行方式（自录 → MediaPipe → CSV，已就绪）

步 A 之前卡在"没有带标注的视频数据"，现在补上了采集端（`tools/blink_eval/`）：

```powershell
# ① 一次性准备：编译浏览器版状态机 + 拷贝 MediaPipe 运行时与模型（自托管，不依赖运行时 CDN）
powershell -File D:\Develop\AIC\tools\blink_eval\prepare.ps1

# ② 起本地服务（getUserMedia 只在安全上下文可用，file:// 会被 Chrome 拒）
powershell -File D:\Develop\AIC\tools\blink_eval\serve.ps1
#    → 浏览器打开 http://127.0.0.1:8099/capture.html

# ③ 录制 1–3 分钟（≥30 次自然眨眼，眼镜照常戴）→ 慢放复核标注 → 导出 3 个文件
#    blink_scores_*.csv（逐帧分数）、blink_marks_*.csv（人工标注）、blink_meta_*.json（采样条件）

# ④ 分析
node D:\Develop\AIC\tools\blink_eval\sampling_effect.js `
     --csv <blink_scores_*.csv> --fps 30 [--marks <blink_marks_*.csv>]
```

**采集端的两个设计要点**（都是为了"测出来的数字可信"）：

1. **时间轴用真实视频帧号**：录制循环走 `requestVideoFrameCallback`，`frame` 列记的是
   `presentedFrames`、`t` 列记的是 `mediaTime` —— 这样 `frame / fps` 是可信时间轴，
   而不是"我以为我跑了 30fps"。跟不上时如实写进 `blink_meta_*.json` 的 `droppedVideoFrames`。
   顺手也把 `sampling_effect.js` 的连续性判断改对了：**步长均匀**（即使跳号）即视为可信时间轴，
   只有**步长不规则**（如 RT-BENE 那种从视频里抽帧的数据集）才警告。
2. **状态机用的是真实源码的编译产物**：`prepare.ps1` 把 `client/src/hooks/blinkLogic.ts` 与
   `blinkChannel.ts` 编译成 ESM 供页面 import，**不在 HTML 里手抄一份判定逻辑** ——
   否则测到的是副本行为，与线上必然漂移。

**自检（不需要摄像头）**：`node _selftest_fixture.js` 生成模拟自录的 CSV，再跑上面第 ④ 步。
本次实测该链路端到端通过，并顺手发现并修掉一个**工具自身的陈旧 bug**：插值改成参数开关之后，
`sampling_effect.js` 仍然只传分数、没换参数，于是"插值"一行与"原始"完全相同（假对照）。
修好后在自检夹具（60 s / 13 次眨眼 / 110–370 ms）上：

| 配置 | 事件召回 | 时长 MAE | 量化粒度 |
|---|---|---|---|
| 10fps 原始 | 92.3% | 29.3 ms | 100 ms |
| 10fps + 插值 | 92.3% | **24.7 ms** | 100 ms |
| 15fps + 插值 | 100% | **11.0 ms** | 67 ms |
| 5fps | **53.8%** | 81.3 ms | 200 ms |

与 §8.3 的判断一致：**插值有帮助但依信号形状而定，仍不宜默认开启**；
而 5fps 这种量级的降采样会让召回直接腰斩 —— 这也是自适应降档把 10fps 设为下限的原因。

### 8.7 仍然没做的

- **步 A 的"真人数字"**：采集端已就绪，但**还没有真实录制的片段**跑过 ——
  §8.2 与 §8.6 的数字都是合成序列/自检夹具，不能当作线上误差。
- **步 C（让眨眼并入风险）**：未做，需人工审核。
- **浏览器内实跑（产品链路）**：`useEyeStateModel` 的 wasm 加载与推理没在浏览器里跑过
  （采集端的 MediaPipe 部分另说）；预处理数学有 Node 断言，但产品侧端到端仍需一次人工验证。
- **UI 展示**：新增的 `blinkChannelFps` / `blinkDurationQuantumMs` 等字段已进入指标对象，
  但界面尚未把它们呈现出来（建议在 Profile 的眨眼卡片上显示"时长精度 ±33ms"）。

---

## 9. 能不能在 RT-BENE 上训练？（**能**，但有三条硬边界）

流水线已写好并端到端验证：`tools/blink_eval/train_eye_state.py`。

**做什么**：读 RT-BENE 的标注眼图 → 按**受试者级**切分 → 训一个自写小卷积网
（157 153 参数，614 KB fp32）→ 导出 ONNX → 用**导出的 ONNX**在留出受试者上评分
→ 与现成模型（`open-closed-eye-0001` / OCEC）在同一批样本上对照。

两个工程决定值得记下：

- **输入是 0–255 的 RGB `[N,3,36,60]`，归一化用 `register_buffer` 烘进 ONNX 图**。
  浏览器端只要把像素整数直接填进张量即可 —— 少一处手写预处理，就少一处静默出错的机会
  （这一路上已经栽过 BGR/RGB 与类别顺序两次）。
- **尺寸与裁剪口径和线上对齐**：RT-BENE 的 patch 就是 60×36、按双眼中心去 roll 裁出来的，
  与 `eyePatch.ts` 复刻的口径同源。这意味着自训模型的**预处理链路可以完全复用**。

### 9.1 边界一：许可证（最关键）

RT-BENE 是 **CC BY-NC-SA 4.0**。在它上面训练的权重属于**衍生作品**：

| 用途 | 可否 |
|---|---|
| 比赛 / 研究（非商业） | ✅ 可用，须标注数据来源与许可证 |
| 商业化产品 | ❌ **不可用**。要商用必须换宽松许可的数据源重训（HF `MichalMlodawski/closed-open-eyes`，ODC-By；或 MRL Eye） |

所以正确说法是：**"为比赛训一个域匹配的模型"可行且值得做；"把它当作产品资产"不行。**

### 9.2 边界二：必须按受试者切分 —— 数据齐了之后，自训模型**确实明显更强**

先看**反例**（把教训留下）：只拿到 2 个受试者时做的双向两折，结论自相矛盾 ——

| 折 | 训练 | 留出 | 自训 614 KB | `open-closed-eye-0001` 46 KB | OCEC-n 177 KB |
|---|---|---|---|---|---|
| A | s007 | s012 | **AUC 0.969 / F1 0.723** | AUC 0.918 / 0.462 | AUC 0.919 / 0.453 |
| B | s012 | s007 | AUC 0.849 / 0.418 | **AUC 0.974 / 0.753** | AUC 0.735 / 0.319 |

1 个受试者训练时，模型更像在记那个人 —— **"训谁"就能决定谁赢**。
所以当时不能下任何结论。

数据补齐到 **17 个受试者、73 907 张可用标注眼图**（闭眼 5 216）后，重做受试者级评估：

**主结果**（两折，各自训练 14 个受试者 / 约 46 232 张；留出受试者**完全不同**）

| 折 | 留出受试者 | 留出样本 / 闭眼 | 模型 | **AUC** | 最优阈值 F1 | 保 90% 召回 |
|---|---|---|---|---|---|---|
| 1 | s000 + s002 + s008 | 27 675 / 2 179 | **自训 614 KB** | **0.9917** | **0.866**（thr 0.92） | thr 0.89 → precision **0.816** |
| 1 | 同上 | | `open-closed-eye-0001` 46 KB | 0.9022 | 0.684（thr 0.44） | **达不到 90% 召回** |
| 1 | 同上 | | OCEC-n 177 KB | 0.7973 | 0.314 | — |
| 2 | s005 + s009 + s014 | 12 691 / 370 | **自训 614 KB** | **0.9822** | **0.682**（thr 0.95） | thr 0.60 → precision 0.321 |
| 2 | 同上 | | `open-closed-eye-0001` | 0.9652 | 0.663（thr 0.48） | — |
| 2 | 同上 | | OCEC-n | 0.7408 | 0.119 | — |

训练过程稳定：10 epoch、每轮约 35 秒（CPU，4 线程），留出 AUC 升到 0.98–0.99。

**怎么解读（四条都要说清）**：

1. **两折结论一致：域匹配的自训模型都不输，AUC 两折都更高**（0.9917 vs 0.9022；0.9822 vs 0.9652）。
   把"46 KB 的 MRL 近红外模型"换成"614 KB 的 RGB 域内模型"是划算的 —— 模型仍只有 614 KB，
   浏览器内单次推理仍是亚毫秒级。
2. **但优势幅度差别很大**：折 1 是碾压（F1 +18 个点），折 2 几乎打平（+2 个点）。
   所以**不要说"自训必然大幅更好"**；只能说"在域匹配的数据上不差，通常更好，
   收益大小取决于留出的是谁"。
3. **0.99 是"主场"成绩**：留出集与训练集同属 RT-BENE（同一次采集、同一批画质），
   而线上要用的是**自录片段裁出来的 patch**。真实水平必须用 `capture.html` 自录 + 人工标注来定。
4. **数据量本身就是结论**：1 个受试者 → 结论自相矛盾；14 个受试者 → 留出从没见过的人，AUC 0.98–0.99。
   这条经验对以后任何"要不要自训"的决策都适用。

> 数据体检（`check_dataset.py`）：Zenodo 限流导致 7 个受试者的 tar 被截断，
> 但**截断都落在文件边界上**，实测 **0 张坏图**；缺失的 40 583 张里绝大部分是睁眼帧，
> 5 216 张闭眼帧**一张没少**。所以这批数据是可用的。

### 9.2.1 但它仍然**不能进产品**

再强调一次许可证：RT-BENE 是 CC BY-NC-SA 4.0 → 上面这个 614 KB 权重是**衍生作品**，
**只能用于比赛/研究**。要商用必须换数据源重训（HF `MichalMlodawski/closed-open-eyes`，ODC-By），
训练脚本与评估流程可原样复用，只是换数据加载器。因此本次**没有**把它接进
`client/public/models/`，也没有改 `eyeStateClassifier.ts`。

若将来换许可数据重训出可商用版本，接入只需三步（规格已固定，见 §9.3）：
把 ONNX 放进 `client/public/models/`、写一份 NOTICE、给 `eyeStateClassifier.ts` 加一条
"输入 0–255 RGB `[N,3,36,60]`、输出单个闭眼 logit"的分支（`eyePatch` 的 `rtBene` 目标尺寸
已经是 60×36，可直接复用）。

### 9.3 边界三：阈值必须重新标定

闭眼帧只占 4.3%–7.1%（取决于受试者），**阈值 0.5 完全不可用**：

| 模型（留出 s000+s002+s008） | 最优阈值 | 阈值 0.5 下的闭眼 F1 | 最优阈值下的 F1 |
|---|---|---|---|
| 自训 | 0.92 | 0.698 | **0.866** |
| `open-closed-eye-0001` | 0.44 | 0.681 | 0.684 |
| OCEC-n | 0.89 | 0.288 | 0.314 |

阈值必须在**留出受试者**上标定，并把结果写进模型元信息
（`metrics.json` 里已有 `best_threshold` / `recall90_threshold`）。
这也再次说明 §5 步 B「先只上报、不进风险」不是可选项。

### 9.4 还缺什么

1. **可商用的数据源**：RT-BENE 训出来的权重只能非商业用（§9.2.1）。
   换 HF `MichalMlodawski/closed-open-eyes`（ODC-By，126 560 张）重训即可，
   流水线与评估流程原样复用 —— 这是目前最值得做的下一步。
2. **真实自录片段的验证**：0.99 是"主场"成绩。必须用 `capture.html` 自录 + 人工标注，
   看两件事：(a) 跨到我们自己裁剪的 patch 上掉多少；(b) RT-BENE 疑似只标**闭合峰值**
   （闭眼段常只有 1–3 帧连续），据此训练的模型很可能**低估闭眼时长**，
   会直接影响 `avgBlinkDuration` / `longBlinkCount`。
3. **补齐被截断的 7 个受试者**：`fetch_assets.ps1 -WithPatches` 支持断点续传，
   网络宽松时重跑；补全后闭眼样本会更多（当前 5 216 张已全部拿到，缺的是睁眼帧）。
4. **导出依赖**：torch 2.14 的 ONNX 导出需要 `onnxscript`（已装入 `.venv`）。

### 9.5 命令

```powershell
# 0) 先体检：Zenodo 限流会让 tar 被截断，确认有多少可用样本、有没有坏图
D:\Develop\AIC\.venv\Scripts\python.exe tools\blink_eval\check_dataset.py --data-dir tools\blink_eval\data

# 1) 训练 + 导出 ONNX + 与现成模型同批对照（需要 onnxscript，已装）
D:\Develop\AIC\.venv\Scripts\python.exe tools\blink_eval\train_eye_state.py `
  --data-dir tools\blink_eval\data `
  --out tools\blink_eval\runs\rtbene_v1 `
  --epochs 10 --threads 4 `
  --holdout-subjects s000 s002 s008 --max-val-subjects 3 `
  --compare tools\blink_eval\models\open_closed_eye.onnx tools\blink_eval\models\ocec_n.onnx
```

- `--holdout-subjects` 不传时，脚本会读 `rt_bene_subjects.csv` 的 `validation` 标记，
  再补足到 `--max-val-subjects` 个；**切分始终按受试者**，并在 `metrics.json` 里记录留出名单，
  便于日后核对"是不是真的按人分开了"。
- 本次两折的产物：`runs/rtbene_v1/`（留出 s000+s002+s008）、`runs/rtbene_fold2/`（留出 s005+s009+s014）。
- 每次训练约 6 分钟（46 k 样本、10 epoch、CPU 4 线程）。
