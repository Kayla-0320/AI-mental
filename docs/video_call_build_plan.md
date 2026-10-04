# 视频通话：修复与建设计划

> 这份文档回答三个问题：**要修什么 / 为什么修 / 做完是什么效果。**
>
> 它不重新论证可行性（那在 [`video_in_call_evaluation.md`](./video_in_call_evaluation.md)），
> 也不重复声音侧的方案（那在 [`voice_call_plan.md`](./voice_call_plan.md)）。
> 它只做一件事：**把"能融"变成"怎么做、先做什么、做完长什么样"。**
>
> 事实来源标注：
> * **[实测]** —— 本机真实跑出来的数字（本次会话）。
> * **[实测-代码]** —— 从当前工作区读到的代码，给了 `文件:行号`。
> * **[估计]** —— 有依据的推算，未测。
> * **[未验证]** —— 没验证过。

> **执行状态（本轮）**：阶段 1（F1–F4、F10）与阶段 2（F5、F6、F11）**已完成并通过浏览器实测验收**
> —— 数据见 §8。阶段 3（出声与通话闭环）、阶段 4（视觉进通话）**未开始**。

---

## 0. 一页结论

**要修 12 项，其中 4 项是阻塞级 —— 不修完不要开始写通话页的媒体代码。**

| 组 | 数量 | 内容 | 为什么这个优先级 |
|---|---|---|---|
| **P0 阻塞** | 4 | 媒体流生命周期（F1–F4） | 现在"再开一次就漏一份"（§2.F1）；通话里"再开一次"是自动事件，不修就是在一个漏水的地基上盖楼 |
| **P0 诚信** | 2 | 两个常量冒充测量值（F5–F6） | 它们能**改写数据库里的患者风险等级**，而通话是旗舰功能、摄像头一直开着、评委会盯着看 |
| **P1 可靠性** | 3 | 通话场景专属（F7–F9） | 20–30 分钟的长会话必然遇到：标签页切后台、设备被抢走、低头误报 |
| **P2 清理** | 3 | 死代码 / 假数据 / 版本漂移（F10–F12） | 不修不阻塞，但都是"未来的事故源"，顺手做掉 |

**通话的最终形态：**

用户看到的是一个**整屏的通话页**，中间是 AI 头像（有呼吸动效），自己的画面在右下角小窗。
说出来的话**逐字上屏**，AI 的回复**边说边出字幕**，随时可以打断。
摄像头是一个**显式开关**：开了之后，视觉信号只做三件诚实的事（存在感、画面质量提示、
给自己看的生理小读数），**完全不参与风险判定、不写数据库**。

**这个决定的依据（很重要）：** 安全链路靠的是**实时文本转写**（`/api/v1/asr/ws` +
`_StreamingSafetyGate`，都已可用 **[实测]**），**不依赖视觉**。所以把视觉从风险判定里拿掉，
**安全性一点都没有降低**，但去掉了一整类误报和一类不诚信的数据。

---

## 1. 总表：改什么、不修会怎样

| 编号 | 改什么 | 位置 | 不修会怎样 | 量 |
|---|---|---|---|---|
| **F1** | 让 `start()` 幂等（开新流前先停旧流） | `useFacialAnalysis.ts:474`、`useVoiceAnalysis.ts:280` | **根因**。每次"再开一次"漏一份麦克风/摄像头，`stop()` 永远够不到旧的 | 小 |
| **F2** | 抽 `MediaStreamManager`（单例 + 引用计数） | 新建 `client/src/services/mediaStreamManager.ts` | 通话页要自己开一路，和感知 hook 各开一路 → 抢同一个设备，两套清理逻辑互不知情 | 中 |
| **F3** | 补入口守卫 + 给「启动分析」按钮一个关闭态 | `AnxietyContext.tsx:511/967`、`Profile.tsx:821` | 这个入口**只能开不能关**，点完还弹窗引导用户重试 | 小 |
| **F4** | 摄像头失败必须**可见**（现在被静默吞掉） | `useFacialAnalysis.ts:516`、`AnxietyContext.tsx:989` | 用户拒了摄像头权限、麦克风照跑，界面却显示"摄像头已开启" | 小 |
| **F5** | 删 `microExpressions.avgDuration = 200` | `useFacialAnalysis.ts:332` | 硬编码常量冒充测量值；"微表情次数"会上屏给用户看 | 极小 |
| **F6** | 删常量置信度（两处） | `useFacialAnalysis.ts:283`、`:465` | `min(0.95, 0.7+n*0.005)`：跑满 50 帧恒为 0.95，与画面内容无关 → **进融合 → 改 `riskLevel` → 落库** | 小 |
| **F7** | 通话期视觉**不进融合/不进风险/不上报** | `AnxietyContext.tsx:631-921`、`reportToServer` | 通话是看脸最不准的场景，却能改风险标签 | 中 |
| **F8** | 指标改名 `downwardGazeRatio`→`headDownRatio` 等 + 通话期阈值 | `useEyeTracking.ts` / `blinkLogic.ts` | "低头看手机"（通话常态）被判成"回避" → 持续误报 | 中 |
| **F9** | `track.onended` + 后台标签页 + 帧陈旧检测 | `useFacialAnalysis.ts`、`useVoiceAnalysis.ts` | 流断了 UI 还说"检测中"；切后台后读数冻在最后一帧被当成"检测不到" | 中 |
| **F10** | 删死代码 `useBlinkDetection.ts` / `usePerception.ts` | 两个文件 | 各自会开独立摄像头/麦克风，零导入但留着就是事故源 | 极小 |
| **F11** | 删 `algorithm-bridge.ts` 的假 `facialFeatures` | `algorithm-bridge.ts:620-623` | `{brightness:128, contrast:50}` 造出一个假的"快乐"模态，**把焦虑分往下拉** | 极小 |
| **F12** | MediaPipe 锁到本地 `node_modules` 副本 | `useFacialAnalysis.ts` 的 CDN URL | 用的是 `@latest`，会漂移；依赖 11.7 MB 外网 | 小 |

---

## 2. 逐项：现象 / 为什么 / 怎么改 / 怎么验收

### F1 —— 让 `start()` 幂等（**根因，最先做**）

**现象 [实测-代码]**：两个 `start()` 里都是覆盖写。

```ts
// useFacialAnalysis.ts:479
streamRef.current = stream;      // ← 不先 stop 旧的
// useVoiceAnalysis.ts:283
streamRef.current = stream;      // ← 同上
```

`streamRef` 只记得住最后一份。旧的轨道仍然 `live`（浏览器麦克风指示灯亮着），
但已无任何引用可达，`stop()` 永远够不到它。`intervalRef.current`、`faceLandmarkerRef`
同理被覆盖 → 定时器和 MediaPipe 实例一起漏。

**为什么这是根因**：它把"再开一次"这个**看起来无害的动作**变成了泄漏。
放大后实测 **[实测]**：5 条 live 麦克风 + 3 个 MediaPipe 实例 + 138 次/秒的定时器 + 1.7 s 冻结。

**为什么不能靠"别乱点"绕过**：通话里"再开一次"是**自动事件**，不用点鼠标 ——
切前后摄像头、断线重连、React 组件重挂载（StrictMode）、手机来电打断、切后台再回来。
**每一次都是一次泄漏。**

**改法**：

```ts
const start = useCallback(async () => {
  // 幂等：先结清上一次，再开新的
  streamRef.current?.getTracks().forEach(t => t.stop());
  streamRef.current = null;
  if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = 0; }
  faceLandmarkerRef.current?.close();   // 仅 useFacialAnalysis
  faceLandmarkerRef.current = null;
  // ...以下原样
```

`useVoiceAnalysis` 同理，另外要 `audioContextRef.current?.close()` 并把 `asrSessionRef.current?.stop()` 提前（现在在第 325 行，位置已经对了，保持）。

**验收**：连调 `start()` 5 次 → 包裹 `getUserMedia` 计数应为 **1**，`liveTrackCount` 恒为 **1**，
`setInterval` 计数恒为 **1**。

---

### F2 —— 抽 `MediaStreamManager`（单例 + 引用计数）

**为什么**：仓库里现在有 **5 个独立 `getUserMedia` 调用点 [实测-代码]**：
`useFacialAnalysis.ts:476`(video)、`useVoiceAnalysis.ts:282`(audio)、
`useBlinkDetection.ts:207`(video，死代码)、`usePerception.ts:69`(audio，死代码)、
`PatientRoom.tsx:84`(audio)、`useWebRTC.ts:98/194`(video+audio)。

通话页如果按现有接法再加两路，且用户此前开过"多模态感知"：
**2 路摄像头 + 2~3 路麦克风 + 2~3 个 AudioContext 同时存活**，两套清理逻辑互不知情。

**接口（建议）**：

```ts
export type MediaKind = 'camera' | 'microphone';
export type MediaConsumer = 'perception' | 'call' | 'blink' | 'counselor-room';

export const mediaStreamManager = {
  /** 借用；同一个 consumer 重复借 = 幂等，返回同一份 */
  acquire(kind: MediaKind, consumer: MediaConsumer, constraints?: MediaStreamConstraints): Promise<MediaStream>;
  /** 归还；引用计数归零才真的 stop 轨道 */
  release(consumer: MediaConsumer): void;
  /** 谁在用（调试面板用，也用于通话页判断"摄像头已被占用"） */
  holders(kind: MediaKind): MediaConsumer[];
  /** 一份只读的轨道视图，给"多个消费者共用同一个设备"用 */
  borrowTracks(kind: MediaKind, consumer: MediaConsumer): MediaStreamTrack[];
};
```

**关键设计**：`acquire` 对同一 `consumer` **必须幂等** —— 这是 F1 的结构性版本，
让"重复调用"在类型层面就不可能出错，而不是靠每个调用点自觉。

**验收**：
1. `perception` 和 `call` 同时借 camera → 底层 `getUserMedia` 只调 **1** 次，两方拿到**同一份**轨道。
2. `release('perception')` 后 `call` 仍持有 → 轨道**不**被 stop。
3. 全部 release → `liveTrackCount === 0`。
4. 整个测试过程浏览器麦克风/摄像头指示灯只有一次亮起。

---

### F3 —— 补入口守卫 + 给「启动分析」按钮一个关闭态

**现象 [实测-代码]**：

* `AnxietyContext.tsx:511` 的 `handleStart`（`start-multimodal-analysis` 监听器）**完全没有守卫**，
  每次调用都真的再开一路 `voice.start()` + `facial.start()`。
* `AnxietyContext.tsx:967` 的 `toggleCamera` 没有重入保护，且 `setCameraEnabled(true)`
  在 `await Promise.all(...)` **之后**才执行 —— await 期间 `cameraEnabled` 仍是 `false`，
  后续点击继续走"开"分支。
* 唯一触发点 `Profile.tsx:821` 的「启动分析」按钮**只能开、不能关**，点完还 `alert`
  "如果还是没有变化，请检查摄像头/麦克风权限" —— **等于在引导用户再点一次**。

**为什么**：让"入口"只能进不能出，是把一个代码缺陷变成用户可复现的操作。
对比：`startVoiceInput`（`:997`）**已经有** `if (voice.metrics.isRecording) return;` 守卫 ——
说明团队知道要写，只是摄像头侧漏了。

**改法**：
1. `handleStart` 开头加 `if (voice.metrics.isRecording || facial.metrics.isDetecting) return;`
2. `toggleCamera` 用 `useRef` 做重入锁；`setCameraEnabled(true)` 移到 `await` **之前**（乐观置位），失败再回滚。
3. 「启动分析」按钮改成读 `cameraEnabled` 的真正开关（文案"开启/关闭多模态感知"），或直接删掉它、统一走 `toggleCamera`。删掉 `alert`，改为页内状态提示。

**验收**：进入 Profile 页 → 连点按钮 5 次 → `getUserMedia` 计数为 **1**；
再点 1 次 → `liveTrackCount === 0` 且按钮文案回到"开启"。

---

### F4 —— 摄像头失败必须可见

**现象 [实测-代码]**：`useFacialAnalysis.ts:516-518` 把错误**自己吞了**：

```ts
} catch {
  console.warn('[面部分析] 摄像头权限被拒绝');   // ← 不往外抛
}
```

`start()` 是 `async` 但**永不 reject**。因此 `AnxietyContext.tsx:989` 的
`catch (err) { setCameraEnabled?? }` 是**死代码**，`setCameraEnabled(true)` 照常执行。

**为什么不修很糟**：用户只给麦克风、拒了摄像头（或摄像头被 Zoom 占用）时：
`voice` 在跑、界面显示"摄像头已开启"、**一个面部数据都没有**。
在通话里这个表现的等价物是"对方看不到我，但我这边一切正常"。

**改法**：
1. `start()` 返回 `Promise<boolean>`（成功/失败），把真实原因带出来（`NotAllowedError` / `NotFoundError` / `NotReadableError`）。
2. `toggleCamera` 依据返回值决定 `cameraEnabled`，失败时回滚并显示原因。
3. 通话页加"视觉不可用"状态（对应 §3.6）。

**验收**：devtools 里把摄像头权限设为 deny → 点开关 → 界面必须出现明确的"摄像头不可用（权限被拒绝）"，
且**麦克风不被打开**（或已打开则明确告知"仅麦克风"），绝不出现"已开启"。

---

### F5 —— 删 `microExpressions.avgDuration = 200`

**现象 [实测-代码]** `useFacialAnalysis.ts:332`：

```ts
avgDuration: microExprCountRef.current > 0 ? 200 : 0,   // ← 有计数就恒为 200ms
```

**为什么**：它不是测量值，是**为了填字段而写的常数**。而它会被上屏（
`AnxietyContext.tsx:196` 会生成"我们还捕捉到了 N 次微表情——那些持续时间不到半秒的……"）。

**改法**：删掉该字段的赋值（或置 `0`/`null`），把 `microExpressions.avgDuration`
从 `FacialAnalysis` 里移除；生成文案的那段改成只说次数，不说时长。

**验收**：`FacialAnalysis.microExpressions` 不再有 `avgDuration` 字段（类型定义同步删）；
开着摄像头说不同长度的话，界面上的"微表情"文案不再出现"持续时间"字样。

---

### F6 —— 删常量置信度（**两处**）

**现象 [实测-代码]**：

```ts
// useFacialAnalysis.ts:283  ← MediaPipe 路径（当前 LIVE 的那条）
const confidence = useMediaPipeRef.current ? Math.min(0.95, 0.7 + frameCountRef.current * 0.005) : 0.5;

// useFacialAnalysis.ts:465  ← 像素回退路径
confidence: Math.min(0.6, 0.3 + frameCountRef.current * 0.005),
```

**为什么这是全项目最该删的一行**：它是**帧计数**，不是信号质量。
只要摄像头开着满 50 帧（**5 秒**），`confidence` 就恒为 **0.95** ——
画面全黑、没有人在镜头里、人脸检测失败，都一样是 0.95。

而它**不是**自娱自乐的展示值，它一路往下走：

```
facialMetrics.confidence
  → AnxietyContext.tsx:633   weights.facial = min(cfg, max(cfg*0.5, confidence*cfg))
  → :670/:714                fusionWeights.facial / normalizedWeights.facial
  → :769-770                 fusedProbs += facialProbs5 * weight
  → :903                     modalityRiskScores.facial
  → riskLevel
  → reportToServer           → 落库 patientProfile.riskLevel
```

**所以它是一个能改写数据库里患者风险等级的常数。**

**改法**（二选一，我推荐第一个）：
1. **如实上报信号质量**：用 MediaPipe 真实可得的量，例如
   `faceLandmarks.length > 0` 的比例、blendshape 输出是否有效、画面亮度方差。
   拿不到就报 `null`，让下游把该模态剔除，**而不是拿常数顶上**。
2. 直接删掉该字段，`weights.facial` 改为按"是否有有效帧"取固定权重。

同时 `AnxietyContext.tsx:633` 依赖 `confidence` 的那行要一并改。

**验收**：
1. 用一张全黑画面（或无人的画面）跑面部分析 → `confidence` 必须显著低于有人脸时，或为 `null`。**不允许恒为 0.95。**
2. `grep "0.005"` 在 `client/src` 无命中。
3. 新增单测：无人脸帧 → 不产生 facial 权重。

---

### F7 —— 通话期视觉**不进**融合/风险/上报

**为什么**（这是本计划最重要的产品判断）：

1. **通话是看脸最不准的场景**：低头看手机、记笔记、走动离开画面、说话带动下半脸 ——
   全都是通话常态，却会被现有阈值判成"回避/注意力涣散"。
2. **视觉对安全没有贡献**：危机判定的文本链路（实时转写 + `_StreamingSafetyGate`）**已经独立可用 [实测]**。
   拿掉视觉不降低安全性。
3. **误报的代价不对称**：一次"检测到你注意力涣散/情绪回避"的误报，出现在一个心理健康产品上，
   比"少一个模态"严重得多。

**改法**：在 `AnxietyContext` 的融合入口加一个"通话模式"标志位：

```ts
// 通话期：视觉只上屏，不进 weights、不进 riskLevel、不上报
const visualInFusion = !isCallActive;
```

具体要改的点：`AnxietyContext.tsx:631-633`（weights）、`:903`（modalityRiskScores）、
`reportToServer`（`:1066-1097`）里剔除 `facial` / `eye` 两项。

**验收**：
1. 通话中打开摄像头 → 界面能看到视觉读数，但 `reportToServer` 的 payload 里
   `facial`/`eye` 字段为空或不存在。
2. 同一段对话，开/关摄像头两次 → 落库的 `riskLevel` **完全一致**（这是最硬的验收）。

---

### F8 —— 指标改名 + 通话期阈值

**现象 [实测-代码]**：`useEyeTracking.ts` 输出的 `downwardGazeRatio`、`attentionScatter`
实际计算的是**头部角度**（`atan2(matrix[6], matrix[10])` 得到 pitch），
不是眼睛的注视方向 —— **构念错误**：名字说的是"目光向下"，量的是"头低下来"。

**为什么在通话里必须改**：打电话时低头看手机/记笔记是**正常行为**，
而现有规则（`pitch > 10°` 判为回避）会**持续误报**。一个 20 分钟的通话能刷出大量假信号。

**改法**：
1. 改名：`downwardGazeRatio` → `headDownRatio`，`attentionScatter` → `headYawStd`。（纯改名，语义变诚实）
2. 通话模式下用独立阈值（不改全局默认）：`pitch > 25°` 且持续 > 3 s 才算一帧"低头"，
   并**只上屏、不告警**。
3. **[未验证]** `facialTransformationMatrixes` 的行/列主序与 pitch 符号需要一次抬头/低头对照实测 ——
   如果符号反了，"抬头"会被统计成"低头"。

**验收**：一次 5 分钟通话里，正常低头看笔记 20 次 → `headDownRatio` 上升，
但**不产生任何风险告警**。

---

### F9 —— `track.onended` + 后台标签页 + 帧陈旧检测

**现象 [实测-代码]**：全仓没有任何 `track.onended` / `onmute` 监听。摄像头被别的 App 抢走、
被拔掉、权限被改 → 轨道 `ended`，但 `isDetecting` 仍为 `true`，UI 继续显示"检测中"。

**为什么"一直开着"会放大它**：通话一开就是 20–30 分钟（对比现在的 30 秒试用），
设备被抢/睡醒/切后台在这些时长里几乎是必然事件。

**三个具体场景**：

| 场景 | 现在的表现 | 应该的表现 |
|---|---|---|
| 流被系统收回 | UI 说"检测中"，零数据 | 立刻显示"摄像头已断开"，停止上报 |
| 切到后台标签页 | 定时器被节流，帧不进来，读数**冻在最后一帧** | 停表：显示"通话已暂停采集"，回来时重新计时 |
| 帧陈旧（>`3×` 间隔没新帧） | 无检测 | 视为无数据，不参与任何计算 |

**改法**：
1. `track.addEventListener('ended', ...)` → 置 `isDetecting=false` 并显示原因。
2. `document.visibilitychange` → 隐藏时暂停分析循环并标记；可见时恢复。
3. 分析循环里加 `now - lastFrameTs > 3 * intervalMs` 的陈旧检查。

**验收**：通话中把摄像头从设备管理器禁用 → 3 秒内界面出现"摄像头已断开"；
切到别的标签页 30 秒回来 → 不产生任何"检测不到/注意力涣散"的结论。

---

### F10 / F11 / F12 —— 清理

| 编号 | 动作 | 依据 |
|---|---|---|
| **F10** | 删 `useBlinkDetection.ts`（10 KB）与 `usePerception.ts` | 全仓导入点为 **0 [实测-代码]**；两者各自会开一路摄像头/麦克风 + 自己的循环/MediaRecorder。留着 = 留两个未来的事故源 |
| **F11** | 删 `algorithm-bridge.ts:620-623` 的 `facialFeatures = { brightness: 128, contrast: 50 }` | 硬编码常量造出一个**假的"快乐"模态**（`emotionMapping` 由它推出），会把焦虑分**往下拉** |
| **F12** | MediaPipe wasm 改为引用本地 `node_modules` 副本（已有 `@mediapipe/tasks-vision@1.0.1`） | 现在用 `@latest`，版本会漂移；且依赖 11.7 MB 外网 **[实测]**（`cdn.jsdelivr.net` / `storage.googleapis.com` 当前可达，但答辩现场不保证） |

---

## 3. 做完之后：视频通话长什么样

### 3.1 屏幕

```
┌────────────────────────────────────────────────┐
│  ← 返回                          00:42         │  ← 通话计时
│                                                │
│                                                │
│                    ◕                           │  ← AI 头像
│              小 安 · 心理陪伴                   │     （呼吸动效，说话时轻微律动）
│           ◉ 正在回应…                          │  ← 状态：正在聆听 / 正在思考 / 正在回应
│                                                │
│                                                │
│   ┌──────────────────────┐                     │
│   │ 我：最近老是睡不着…   │                     │  ← 用户语音转写（边说边逐字上屏）
│   └──────────────────────┘                     │
│   ┌────────────────────────────────┐           │
│   │ 小安：听起来这段时间挺难熬的，  │           │  ← AI 回复（边说边出字幕，
│   │       是从什么时候开始的？      │           │     光标闪烁；可随时打断）
│   └────────────────────────────────┘           │
│                                                │
│  ┌──────────┐  本机分析中 · 画面不上传          │  ← 仅"摄像头开着"时出现
│  │ 自己的画面 │  ▓▓▓░░ 光线偏暗                 │  ← 诚实的画面质量提示（可关）
│  └──────────┘                                  │
│                                                │
│      [ 🎤 ]      [ 📷 ]      [ 挂断 ]          │  ← 三个按钮；📷 有明确开/关态
└────────────────────────────────────────────────┘
```

**三个按钮的行为**：

| 按钮 | 默认 | 点了之后 |
|---|---|---|
| 麦克风 | 开 | 静音（自己听得到提示，AI 侧显示"对方已静音"）。**静音 != 挂断**，会话与安全监测继续（无音频则无文本，此时不做风险推断） |
| 摄像头 | **关** | 开：借摄像头、开始本机分析、右下角出现自拍小窗；再点：还回去、小窗消失、分析停止 |
| 挂断 | —— | 冲刷最后一段定稿 → 落库 → 回到会话列表 |

### 3.2 一次通话的时间轴（一轮对话）

| t | 发生什么 | 用户感知 |
|---|---|---|
| 0 ms | 用户开口 | —— |
| ~330 ms | 中间结果上屏 | 字幕开始逐字出现 **[实测]** |
| 用户停口 + 400–500 ms | 句末判定 → 定稿（复识别 + 补标点，51–61 ms） | 字幕定稿、不再抖动 |
| +75–108 ms | META 完成（感知 + 风险 + 场景检索） | 无感知（后端） |
| +370–390 ms | LLM 首 token | AI 状态变"正在回应" |
| +442–497 ms | 首句可上屏 + 开始合成 | 字幕出现 + 声音开始 |
| ≈1.1–2.4 s | 首声（Path A） | AI 开口 **[估计]** |

> **诚实的预期管理**：Path A 的首声中位是 **1.6 s [估计]**，真人通话是 200–500 ms。
> 所以会有**轻微的"对讲机感"**，靠"半双工 + AEC + 文本级自回声过滤"缓解，
> 不会做到真人的自然重叠。要做到 1 s 内只能走 Path B（音频出设备，与项目"原始数据不出设备"原则冲突）。

### 3.3 用户能看到的实时反馈（这些是"像不像在打电话"的关键）

1. **AI 状态词**随轮次真实变化（聆听 → 思考 → 回应），不是装饰动画。
2. **用户字幕逐字上屏**：这是"它在听我说话"的最强证据 **[实测]** 首个 partial ~770 ms。
3. **AI 字幕逐句上屏 + 光标闪烁**：让 1.6 s 的等待有东西可看。
4. **说话时头像轻微律动**、聆听时呼吸动效 —— 纯装饰，但成本为零。
5. **打断**：用户一开口，AI 立刻停下（服务端 `interrupt_response: true`，或本地停播 + 取消）。

### 3.4 开了摄像头之后，多出什么

**只多三样，都是诚实的：**

| 视觉做什么 | 具体 | 为什么这个可以进通话 |
|---|---|---|
| **① 存在感** | `faceLandmarks.length > 0` → "你在画面里 / 暂时看不到你" | 输入真、无中间量可编造；低头、走动、侧脸都不影响；产品含义真实（"对方还在听吗"） |
| **② 画面质量提示** | 亮度/清晰度不足 → "光线有点暗，看不清" | 这是**对用户有用的诊断信息**，而且它天然诚实（就是画面统计量） |
| **③ 给自己看的小读数** | 眨眼频率、头部角度 —— 像运动手环，**只上屏给自己看** | 满足"视觉功能存在"的产品叙事，且**不写库、不进风险** |

**成本优化 [估计]**：如果只做 ①，帧率可以降到 **2–5 fps**（存在感不需要 10 fps）。
这比现在 10 fps 跑整条 Face Mesh 管线省得多。只有做 ③ 才需要 ≥30 fps（否则眨眼欠采样，见下）。

### 3.5 视觉在通话里**不做**什么（这一节要写进答辩材料，避免吹牛）

* ❌ **不做口型同步**（lip-sync / viseme 对齐）—— 不在本计划范围。
* ❌ **不用视觉判断情绪，也不让视觉影响 AI 的回复内容**。
* ❌ **视觉不进风险判定、不写 `patientProfile.riskLevel`**（F7）。
* ❌ 不做心率/HRV（rPPG 在本机条件下物理不成立，保持 `RPPG_AVAILABLE = false`）。
* ❌ 不做呼吸频率（摄像头路径不可达；改由 ASR 停顿/气口节奏推断 —— 那才是通话的天然产物）。
* ❌ 不做全双工同时说话（AEC 不可靠，明确降级为"半双工 + 可打断"并如实告知）。

### 3.6 失败场景下用户看到什么（**这一节决定产品是否可信**）

| 场景 | 用户看到 | 关键要求 |
|---|---|---|
| 用户拒了摄像头权限 | "摄像头不可用（权限被拒绝）· 改为纯语音通话" | **麦克风照常可用**，通话不中断；绝不显示"已开启"（F4） |
| 摄像头被别的 App 抢走 | "摄像头已断开"（<3 s 内） | 视觉读数停止更新，**不产生任何"检测不到"的结论**（F9） |
| 切到后台标签页 | "采集已暂停（页面在后台）" | 回来后重新计时；**不把暂停期算成"注意力涣散"**（F9） |
| 断网 | "网络不稳定，正在重连…" + 字幕保留 | 已说出的内容不丢；重连后继续 |
| 光线太暗 | "光线偏暗"（不阻断通话） | 不因为看不清就说"情绪回避" |
| **检测到高风险** | 通话**立即停止生成**，全屏显示热线卡片 + "需要我帮你联系咨询师吗" | 复用既有升级流程；热线**只出现一次**（去重已实现） |

### 3.7 一句话描述这个产品的样子

> 一个**能打电话的 AI 陪伴**：说出来的话立刻变成字幕，AI 边说边出字幕、可以随时打断；
> 摄像头开着的时候，它知道**你还在不在**，会提醒你**光线太暗**，
> 也会把眨眼和头部的小读数安静地摆在角落给你自己看 ——
> 但它**不会**因为你在低头看手机，就断定你在回避。

---

## 4. 实施顺序（4 个阶段，每阶段有闸门）

### 阶段 1：地基（F1–F4 + F10）—— **不做完不写通话代码**

| 步骤 | 内容 | 闸门 |
|---|---|---|
| 1.1 | F1 让两个 `start()` 幂等 | 连调 5 次 → `getUserMedia` 计数 = 1 |
| 1.2 | F2 抽 `MediaStreamManager` | 双消费者共用一份轨道；全 release 后 `liveTrackCount = 0` |
| 1.3 | F3 入口守卫 + 按钮关闭态 | 连点 5 次 → 计数 = 1；再点 1 次 → 计数 = 0 |
| 1.4 | F4 摄像头失败可见 | 拒权限 → 明确报错且不误报"已开启" |
| 1.5 | F10 删死代码 | 全仓构建通过、测试不减少 |

**这一阶段可以完全独立完成、独立验收，不依赖 TTS。**

### 阶段 2：诚信（F5 + F6 + F11）—— 小改动，高价值

| 步骤 | 内容 | 闸门 |
|---|---|---|
| 2.1 | F5 删 `avgDuration = 200` | 文案不再提"持续时间" |
| 2.2 | F6 删两处常量置信度，改为如实上报或 `null` | 全黑画面 ≠ 0.95 |
| 2.3 | F11 删假 `facialFeatures` | 无视觉数据时不再产生假"快乐"模态 |

### 阶段 3：出声与通话闭环（沿用 `voice_call_plan.md` 的阶段 0–2）

| 步骤 | 内容 | 闸门 |
|---|---|---|
| 3.0 | **TTS RTF benchmark** | **RTF < 1.0 才继续**（这是 `voice_call_plan.md` 的 go/no-go 闸门，至今未跑） |
| 3.1 | 单句出声闭环 | 用户说一句 → AI 出声 |
| 3.2 | 通话页 UI（§3.1）+ 轮次切换 + 打断 | 首声 < 2.4 s **[估计]**；能打断 |
| 3.3 | 安全：逐句过闸 → 逐句合成 → 逐句播；高风险硬禁用语音 | 危机路径回归通过 |

### 阶段 4：视觉进通话（F7 + F8 + F9 + F12）—— 可选，默认不融合

| 步骤 | 内容 | 闸门 |
|---|---|---|
| 4.1 | F7 通话期视觉不进融合/风险/上报 | **开/关摄像头，落库 `riskLevel` 完全一致** |
| 4.2 | F8 改名 + 通话期阈值 | 正常低头 20 次 → 0 告警 |
| 4.3 | F9 `onended` + 后台 + 帧陈旧 | 抢走摄像头 → 3 s 内可见；切后台不产生结论 |
| 4.4 | 视觉三件事（存在感 / 画面质量 / 自看读数），**F12** 锁版本 | 存在感可降到 2–5 fps |
| 4.5 | 失败场景 UI（§3.6） | 六种场景逐一走查 |

> **注意**：阶段 4 是**唯一可砍**的阶段。砍掉之后，通话功能仍然完整、
> 产品叙事仍然成立（"AI 陪伴通话"），只是没有摄像头。**阶段 1、2 不可砍。**

---

## 5. 验收清单（可直接勾）

**地基**
- [ ] 连调 `facial.start()` 5 次 → `getUserMedia` 计数 = 1
- [ ] 连调 `voice.start()` 5 次 → `getUserMedia` 计数 = 1
- [ ] 连点「启动分析」5 次 → `getUserMedia` 计数 = 1
- [ ] 所有消费者 release 后 → `liveTrackCount === 0`、浏览器指示灯熄灭
- [ ] `perception` + `call` 同时借 camera → 底层只调 1 次
- [ ] 拒绝摄像头权限 → 明确报错，且不显示"已开启"

**诚信**
- [ ] `grep "0.005" client/src` 无命中
- [ ] 全黑画面 / 无人脸 → `confidence` 显著降低或为 `null`，**不恒为 0.95**
- [ ] 无视觉数据时不再产生假"快乐"模态
- [ ] `grep "avgDuration"` 在面部分析字段上无命中

**通话**
- [ ] 开摄像头 / 关摄像头两次 → 落库 `riskLevel` **完全一致**
- [ ] 通话中 `reportToServer` payload 不含 `facial`/`eye`
- [ ] 正常低头 20 次 → 0 条风险告警
- [ ] 抢走摄像头 → 3 s 内显示"已断开"，无"检测不到"结论
- [ ] 切后台 30 s 回来 → 无"注意力涣散"结论
- [ ] 危机文本 → 立刻停止生成、显示热线、**热线只出现一次**、`requires_escalation=True`

**回归**
- [ ] `algorithm/tests/` 测试数不减少（当前 **1100 passed / 9 failed**：8 个 `matplotlib` 缺失 + 1 个在修）
- [ ] 三端构建通过（client / server / algorithm）
- [ ] 流式聊天（`/smart-chat/stream`）行为不变：META 75–108 ms、首 delta 442–497 ms **[实测]**

---

## 6. 未验证 / 风险

| # | 事项 | 状态 |
|---|---|---|
| 1 | TTS 总 RTF < 1.0 | **[未验证]** `voice_call_plan.md` 阶段 0 的闸门，**至今未跑，是整个计划最大的未知** |
| 2 | 视频在真实摄像头 30 fps 下的成本（现测 13.9 ms/tick 是 canvas 15 fps 供流） | **[未验证]** |
| 3 | `facialTransformationMatrixes` 的 pitch **符号** | **[未验证]** 若反了，"抬头"会被算成"低头" |
| 4 | 10 fps 对眨眼率的实际欠采样比例 | **[推断]** 由"眨眼 100–400 ms vs 采样 100 ms"推算，无 30 fps 对照 |
| 5 | 4–8 核普通笔记本上的表现 | **[未验证]** 本机 20 核，不代表用户机器 |
| 6 | "视觉只展示不决策"对通话观感的实际影响 | **[估计]** 无可用原型，属产品判断 |
| 7 | 存在感检测降到 2–5 fps 的省电收益 | **[估计]** 未实测 |

**已知的架构级风险**：`voice_call_plan.md` §6 指出 —— AEC（`echoCancellation: true`）
是必要但不充分的，真正的全双工需要额外的残留回声抑制。本计划选择**明确降级为半双工并告知用户**，
而不是假装做到了全双工。

---

## 7. 一页速览

1. **先修地基，再盖楼。** 根因是一行覆盖写（`streamRef.current = stream`），
   它让"再开一次"必然漏一份；而通话里的"再开一次"是**自动事件**（切摄像头、重连、重挂载）。
2. **12 项修改，4 项阻塞、2 项诚信、3 项可靠性、3 项清理。** 全部有可机器验证的验收标准。
3. **最该删的两行是诚信问题**：`avgDuration = 200` 和 `Math.min(0.95, 0.7 + n*0.005)`。
   后者能把一个 5 秒后恒定的常数一路送进数据库的 `patientProfile.riskLevel`。
4. **视觉进通话，但不进风险判定。** 依据是：安全链路本来就走文本，
   而通话是看脸最不准的场景 —— 拿掉视觉**不降低安全性**，但去掉了一整类误报。
5. **视觉在通话里只做三件诚实的事**：存在感、画面质量提示、给自己看的小读数。
   最值得做的是"人还在不在画面里"这个布尔值，不是微表情。
6. **效果**：一个能打电话的 AI 陪伴 —— 边说边出字幕、可以随时打断、
   摄像头开着时它知道你在不在、会提醒你光线太暗，
   但**不会**因为你低头看手机就断定你在回避。
7. **诚实的天花板**：Path A 首声中位 1.6 s **[估计]**，会有轻微对讲机感；
   进 1 s 只能走 Path B（音频出设备），与"原始数据不出设备"原则冲突。能做，但有代价。

---

## 8. 验收结果（**已执行**，2026 本轮）

阶段 1（F1–F4、F10）与阶段 2（F5、F6、F11）**已全部完成并通过验收**。
阶段 3、4 未开始。

### 8.1 改动的文件

| 文件 | 改动 |
|---|---|
| `client/src/services/mediaStreamManager.ts` | **新建**：设备唯一拥有者（幂等 acquire + 引用计数 + `onEnded` + 错误分类） |
| `client/src/hooks/useFacialAnalysis.ts` | `start()` 幂等 / 走管理器 / 失败抛错；真实置信度；`facePresent`；删 `avgDuration=200` |
| `client/src/hooks/useVoiceAnalysis.ts` | `start()` 幂等 / 走管理器 / 失败抛错；`stop` 改交还引用 |
| `client/src/types/multimodal.types.ts` | `confidence: number \| null`；新增 `facePresent` / `unavailableReason`；删 `avgDuration` |
| `client/src/context/AnxietyContext.tsx` | 重入锁 `startingMediaRef`；`handleStart` 加守卫并 await；`toggleCamera` 返回结果 + 失败可见；`confidence` 为 null 时剔除面部模态；表情文案不再断言时长 |
| `client/src/pages/patient/Profile.tsx` | 「启动分析」单向按钮 → 真正的采集开关；显示失败原因 |
| `server/src/services/algorithm-bridge.ts` | 删写死的 `{brightness:128, contrast:50}`，并删掉 `facialFeatures` 形参槽位 |
| 删除 | `client/src/hooks/useBlinkDetection.ts`、`client/src/hooks/usePerception.ts`（零导入） |

### 8.2 浏览器实测（CDP + Edge，假设备，真实登录 + 真实 Profile 页）

验收脚本：`D:\Develop\AIC\_probe_media_fix.js`（工作区根目录，不进仓库）。
两种权限场景各跑一遍，**全部到达预期状态**：

**场景一：正常授权（假设备）**

| 检查 | 点击次数 | `getUserMedia` | live 轨道 | 100ms 分析循环 | 标签 |
|---|---|---|---|---|---|
| 冷启动（含 MediaPipe 下载） | 1 | 2（audio, video） | 2 | 1 | 关闭采集 |
| 关掉后立刻重开（设备竞态探针） | 1 | 2 | 2 | 1 | 关闭采集 |
| 关掉后等 2.5s 重开 | 1 | 2 | 2 | 1 | 关闭采集 |
| **A. UI 路径连点** | **5** | **2** | **2** | **1** | 关闭采集 |
| **B. 同步调用 onClick（绕开 antd）** | **5** | **2** | **2** | **1** | 关闭采集 |
| **D. 关闭路径同步调用 onClick** | **5** | **0** | **0** | **0** | 开启采集 |
| C. 关闭后重开 | 1 | 2 | 2 | 1 | 关闭采集 |

* **未处理的 Promise rejection：0**（说明 `start()` 改为会 reject 之后，所有调用方都接住了）。
* 关闭后 live 轨道在三个检查点上都归零。

**场景二：摄像头被拒（只授予麦克风）**

| 检查 | 点击次数 | `getUserMedia` | live 轨道 | 可见提示 |
|---|---|---|---|---|
| 冷启动 | 1 | 2（video → `NotAllowedError`） | **1（麦克风仍可用）** | ✅「摄像头不可用（可能没有授权或被其他程序占用）· 已仅启用麦克风」 |
| A. UI 路径连点 | 5 | 2（同上） | 1 | ✅ 同上 |
| B. 同步调用 onClick | 5 | 2（同上） | 1 | ✅ 同上 |
| D. 关闭路径同步调用 | 5 | 0 | 0 | 「面部读数不可用：摄像头不可用（权限被拒绝）」 |

### 8.3 与改造前的对比（同一个测试口径）

| 指标 | 改造前 | 改造后 |
|---|---|---|
| 连点 5 次触发的 `getUserMedia` | 5 次以上（每点一次真的再开一路） | **2 次**（音频 1 + 视频 1，每个设备一次） |
| 连点 5 次后的 live 麦克风轨道 | **5 条**（浏览器指示灯关不掉） | **1 条** |
| 视频分析循环定时器 | **约 14 个**同时在跑（138 次/秒） | **1 个**（100 ms） |
| MediaPipe 实例 | 3 个（只有最后一个能 `close()`） | 1 个 |
| 摄像头被拒时的界面 | 显示"摄像头已开启"，实际零数据（异常被静默吞掉） | 明确提示原因，麦克风照常可用 |
| 面部置信度 | `min(0.95, 0.7 + 帧数×0.005)` → 5 秒后恒为 0.95 | 由「模型输出 × 人脸尺度 × 头部姿态」相乘；没人脸时为 `null` |
| 微表情平均时长 | 恒定 200 ms（硬编码） | 字段已删除，文案不再断言时长 |

### 8.4 构建与测试

| 项 | 结果 |
|---|---|
| `client` 类型检查 `tsc -b` | ✅ 通过 |
| `client` 生产构建 `vite build` | ✅ 3987 模块，8.31 s |
| `server` 类型检查 `tsc --noEmit` | ✅ 通过 |
| `server` 构建 `npm run build` | ✅ 通过 |
| Python 测试 `pytest algorithm/tests` | **1132 passed / 8 failed / 1 skipped** |
| 流式聊天回归 `test_chat_stream.py` + `test_intervention_prompt.py` | ✅ 138 passed |

* 8 个失败**全部**是 `ModuleNotFoundError: No module named 'matplotlib'`（预先存在，与本次改动无关；
  基线为 9 个，少掉的那个是并行 agent 正在修的那条）。

### 8.5 环境备注（复现时要知道）

* 客户端生产构建在本机需要放大 Node 堆并限制 esbuild 并行度，否则会崩：
  `NODE_OPTIONS=--max-old-space-size=3072`、`GOMAXPROCS=2`、`GOGC=40`。
  这是**进程级内存上限**导致的，与代码无关 —— 未经改动的 `useKeyboardDynamics.ts`
  同样会让 esbuild 报 `runtime: cannot allocate memory`。
* Postgres 跑在 Docker 容器 `mh-postgres`；Docker Desktop 未启动时登录会 500（Prisma `P1001`）。
* **算法服务（8001）必须用项目自己的 venv 启动，端口是 8001**（不是 `algorithm/README.md`
  原来写的 8000）。用错解释器的后果**极其隐蔽**：服务照常起、`/health` 照常 200，
  但 `/api/v1/asr/ws` 返回 **404**，前端「语音输入」静默失效 —— 本项目已因此坏过。
  根因是它原本没有固定启动方式。现在固定走守卫脚本（启动前校验解释器 / 依赖 / 端口）：

  ```bash
  npm run dev:algorithm          # 启动
  npm run dev:algorithm:check    # 只检查，不启动
  ```

  守卫**刻意不看进程的 `ExecutablePath` / `CommandLine`** 来判断"是否用错解释器"：
  Windows 上 uv 建的 venv 里 `Scripts\python.exe` 是**硬链接**，WMI 报出来的是链接目标的
  路径，据此判断会误报（已踩，见下）。它改为直接问 `/api/v1/asr/status` 的
  `streaming_available` —— **服务的自述状态才是事实**。
* 环境里存在一个"有 fastapi 但没 sherpa-onnx"的 venv（`D:\Develop\AIC\_verify_venv`）。
  用它启 8001 可以**完整复现**上面那个病态（`streaming_available: false` + ws 404），
  也是这次事故的真实成因。复现命令（在 `algorithm/` 下执行）：
  `& 'D:\Develop\AIC\_verify_venv\Scripts\python.exe' -m uvicorn main:app --port 8001`
* 验收脚本踩过的三个坑（避免重踩）：Profile 页 `loading` 时整页被 `<Spin>` 替换导致开关消失；
  `AnxietyFloatingWidget` 进页 3 秒后弹权限引导会盖住页面（预写 `multimodal_permission_shown=1` 关掉）；
  用 `ant-btn-loading` 判断"操作是否在跑"会误判，必须直接对验收不变量轮询。

### 8.6 下一步

* **阶段 3 的施工细节已另起一份文档**：[`docs/call_implementation_steps.md`](./call_implementation_steps.md)
  —— 逐步施工单（S0–S9）。其中三条**读代码读出来的**结论直接简化了原计划：
  * 通话的文本必须走 Node 的 SSE 端点（不能直连算法服务），否则**不落库、不补热线**；
  * SSE 的 `delta` **已经是过闸的完整句子**，它天然就是 TTS 的合成单位
    → **不需要新写切句器，也不碰 `_StreamingSafetyGate`**；
  * `meta.streaming === false` 就是"高风险禁用语音"的现成开关 → **零新增阈值**；
  * ⚠️ 另外发现一条反直觉的：**打断时不能 abort SSE** —— 现有服务端在被 abort 后会
    走非流式回退（`consultation.service.ts:360-405`），**再发一次完整 LLM 请求并把一份
    用户从未听过的回复落库**。正确语义是「停播 + 排队下一轮」。
* **阶段 3** 的 go/no-go 闸门仍是 `voice_call_plan.md` 阶段 0 的 **TTS 总 RTF < 1.0 benchmark**
  —— **至今未跑**，是整个计划最大的未知（施工单的 S0）。
* **阶段 4**（视觉进通话）建议在阶段 3 之后再动，其中 F7（通话期视觉不进融合）的验收标准是
  "开/关摄像头两次，落库 `riskLevel` 完全一致"。
