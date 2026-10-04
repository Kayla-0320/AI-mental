# 感知能力口径：哪些指标我们没有测量

> 结论先行：**心率、呼吸频率、睡眠这三项，本平台目前无法采集，因此任何界面、报告、融合结果里都不允许出现它们的数值。**
> 该口径的代码落点是 `client/src/hooks/perceptionCapabilities.ts`。

## 一、为什么必须写这份说明

演示版曾经出现下面这些"看起来很真"的数字：

| 界面位置 | 曾经显示的数值 | 实际来源 |
| --- | --- | --- |
| 生理行为层 · rPPG 心率 | 180 bpm / HRV 5 ms | 摄像头画面绿色通道均值的极简峰值计数，被自动曝光和头部晃动带偏 |
| 生理行为层 · 呼吸模式 | 20 次/分（不停跳动） | 麦克风 RMS 能量的过零率，说话/风扇/键盘声都会被当成呼吸 |
| 多模态报告正文 | "你的心率偏高（180bpm）" | 直接引用上面那个假读数 |
| 睡眠监测 | 7 条 09-10~09-16 的睡眠记录 | 前端写死的数组 |
| 今日动态情绪画像 · 睡眠 | 44 分 | 由当天文字的情绪概率反推 |
| 同龄雷达图 · 睡眠 | 44 分 | 同上 |

这些都不是测量结果，却会被用户和咨询师当成真实生理数据判读，属于必须修掉的信任风险。

## 二、当前的真实能力清单

### 真实采集（9 项）

1. 键盘动力学（`useKeyboardDynamics`）
2. 文字语义（`useTextAnalysis`）
3. 语音声学（`useVoiceAnalysis`，Web Audio 的基频/能量/语速/停顿/频谱）
4. 面部微表情（`useFacialAnalysis`，MediaPipe Face Mesh + AU）
5. 语音深层语义（`useVoiceSemantics`，依赖语音识别文本）
6. 认知扭曲（`useCognitiveDistortion`，依赖文字/语音文本）
7. 行为互动（`useBehavioralActivation`，仅统计平台内操作次数）
8. 眨眼 / 头姿（`useEyeTracking`，复用 `useFacialAnalysis` 同一帧的 MediaPipe 输出，见第六节）
9. 昼夜活动记录（`useCircadianRhythm`，仅记录 App 内活动时间点）

### 无法采集（3 项，已关闭）

| 模态 | 开关 | 说明 |
| --- | --- | --- |
| rPPG 心率 / HRV | `RPPG_AVAILABLE = false` | 普通摄像头 + 环境光下信噪比不足；需固定曝光 + 带通滤波 + 频域判定才有意义 |
| 呼吸频率 | `BREATHING_AVAILABLE = false` | 语音识别稳定后，改为由语音停顿/气口节奏推断，当前不输出数值 |
| 睡眠 | `SLEEP_AUTO_DETECTION_AVAILABLE = false` | 无夜间传感器与可穿戴设备；睡眠只能由用户手动记录 |

需要重新启用时设置环境变量（默认均为关闭）：

```
VITE_ENABLE_RPPG=true
VITE_ENABLE_BREATHING=true
VITE_ENABLE_SLEEP_AUTO=true
```

## 三、关闭后的行为约定

- **Hook 层**：`useRPPG.updateFromFrame` / `useBreathing.updateFromAudioEnergy` 在开关关闭时直接 return，指标保持 `isMeasuring: false`、数值为 0。
- **融合层**：`AnxietyContext.performFusion` 不把 `hrv` / `breathing` 计入 `activeModalities` 与权重，`evidence` 也不写入，因此**焦虑指数与风险等级不受影响**。
- **个人基线**：不再为 `heartRate` / `breathingRate` 建立基线。
- **报告层**：叙事报告不生成任何心率/呼吸/睡眠段落，并在结尾附上一行数据来源说明。
- **界面层**：卡片保留但置灰，数值显示 `--`，附「暂不支持检测」标签与原因；融合条件说明里列出未接入的传感器。
- **咨询师端**：同样显示「未采集（无心率传感器 / 算法未启用）」，避免咨询师把占位值当读数。

## 四、睡眠数据的唯一合法来源

睡眠数据只允许来自用户主动记录，链路如下：

1. 前端 `client/src/pages/patient/Sleep.tsx`、`Profile.tsx` 调用 `extraApi.recordSleep`
2. 后端 `POST /api/extra/sleep` → `SleepService.record` → `SleepRecord` 表
3. `ProfileService.calculateRealSleepQuality` / `getTodayEmotionalState` 读取真实记录，取近 30 天主观质量均值 × 10 作为 0-100 分
4. **没有任何记录时返回 0，语义是「未采集」**：前端据此显示「未记录」，雷达图显示「睡眠(未记录)」且不参与对比

同时，后端 AI 画像生成（`aiService.generateDynamicProfile`）的提示词中已删除 `sleepQuality` 字段，并在解析后主动 `delete parsed.sleepQuality` 兜底，防止模型自作主张返回睡眠评分。

## 五、仍然保留的诚实性约束

- 昼夜活动 ≠ 睡眠检测：只提示「深夜仍在使用」和「使用时间是否规律」，不推断入睡/起床时间。活动样本少于 10 条时不给"规律性"百分比。
- 行为互动的"趋势"必须有 ≥4 天跨天记录才判定，避免同一会话内的累计值伪造成下降趋势。
- 行为激活在互动次数为 0 时不显示"采集中"。
- 「注视方向」不是眼球注视：平台未接入虹膜跟踪，`downwardGazeRatio` / `attentionScatter` 由头部姿态推算，界面上按「低头帧占比」「头姿波动」展示，报告里写「较多处于低头姿态」而不是「视线向下」。
- 报告结尾固定附上数据来源说明。

## 六、眨眼检测链路（主链路已切到 blendshape 方案）

眨眼有三个可能的数据来源，当前主链路使用哪一个必须能从数据本身看出来。

### 主路径：MediaPipe 眼睑闭合度

1. `useFacialAnalysis` 本来就用 `FaceLandmarker`（`outputFaceBlendshapes: true`）逐帧推理，只是以前没读 `eyeBlinkLeft/Right`；
2. 现在每帧通过 `onFrameSignal({ greenMean, eyeAspect, headPitch, headYaw, blinkScore })` 一并传出，`blinkScore = min(eyeBlinkLeft, eyeBlinkRight)`（双眼取小，单眼眨动/眯眼不算眨眼）；**模型没输出 blendshape 时必须是 `null`，不能默认 0**（否则等价于「永远睁眼」）；
3. `useEyeTracking` 把它交给 `blinkLogic.BlinkStateMachine`：
   - 迟滞判定 `on 0.5 / off 0.35`，避免分数在阈值附近抖动把一次眨眼切成多次；
   - 时长窗 `60ms ≤ 闭眼 < 800ms`，过短当噪声、过长当休息性闭眼或检测丢失；
   - 最小间隔 `100ms` 去抖；
4. 与面部分析共用同一路摄像头和同一份推理结果，**不再单独开第二路 video + FaceLandmarker**。

### 降级路径：关键点纵横比（EAR）

仅当 blendshape 缺失时使用：`eyeAspect < 0.15` 视为闭眼，判定与过滤仍走同一个状态机。此时 `metrics.blinkMethod === 'ear'`，界面显示「EAR 降级」标签，`evidence` 里也会写明来源。

### 已删除的历史实现

`pages/patient/UnifiedAssessment.tsx`（独立测评页，内置亮度法眨眼检测）此前未在 `App.tsx` 注册路由，用户进不去，其结果也从未进入多模态融合，**已删除**。

- 量表测评能力本身不受影响：`Profile.tsx` 有独立的 PHQ-9 / GAD-7 实现（`/api/profile/assessments`）。
- 该页面曾是 `useBlinkDetection` / `useKeyboardAnxiety` / `usePerception` 的唯一引用方，三者现已无引用。`useBlinkDetection` 里的亮度降级路径（`method: 'brightness'`）不再是主链路的任何一环。
- `blinkLogic.BlinkStateMachine` **仍在主链路使用**（`useEyeTracking` 依赖它），不可删除；回归测试 `tools/blink_tests` 测的就是它。

### 指标口径

- `blinkRate`：60 秒滚动窗口内的眨眼数按实际统计窗口折算（`min(会话时长, 60s)` 为分母）。旧实现拿「会话总时长」当分母，会话超过 60 秒后频率会被系统性摊薄（10 分钟会话里稳定的 15 次/分会显示成 1.5 次/分），已修复并有回归测试。
- `signalQuality`：`来源基准分（blendshape 0.9 / EAR 0.5）× 近 60 秒面部帧覆盖率（期望 ≥5fps）`。旧实现是 `0.7 / 0.2` 两档常量，界面上按百分比展示时会被误读成测量精度。
- 回归测试：`tools/blink_tests/run.ps1`（直接编译真实源码 `client/src/hooks/blinkLogic.ts` 后跑 24 项断言）。
