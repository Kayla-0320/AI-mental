# 视频 / 摄像头感知栈审计

> 审计对象：`AI-mental-main` 前端 React Hook（`client/src/hooks/`）+ Python 算法服务（`algorithm/perception/`）
> 审计方式：**只读代码**（未启动服务、未运行页面）。所有结论标注：
> **[实测]** = 在当前磁盘文件中读到并原文引用；**[推断]** = 由读到的代码推导；**[未验证]** = 本次无法确认。
> 审计时间点：工作区处于「另一 agent 正在改写」的状态 —— `git status` 显示 `blinkLogic.ts` / `useEyeTracking.ts` / `useFacialAnalysis.ts` / `useRPPG.ts` / `useBreathing.ts` / `useCircadianRhythm.ts` / `useBehavioralActivation.ts` / `AnxietyContext.tsx` 均有**未提交的改动**。本文所有引用均以**工作区当前内容**为准。

---

## 1. 结论速览

| 模态 | 真 / 空 | 进 UI | 影响 safety | 能否融入通话 |
| --- | --- | --- | --- | --- |
| 面部 AU / 主导表情 / 微表情（MediaPipe blendshape） | **半真**：输入是真摄像头信号，AU 是模型真输出；但 AU→情绪映射是人工线性组合，`微表情平均时长` 是硬编码 200ms | ✅ Profile / 患者端 / 咨询师端 | ✅ **是** → `weights.facial` → 融合 → `riskLevel` → 上报落库 `patientProfile.riskLevel` | ⚠️ 可保留，但必须删掉 200ms 常量、放弃 `confidence=0.7+n*0.005` 常量置信度、通话期降到 5–10fps |
| 眨眼频率 / 平均眨眼时长 / 长眨眼 | **真**（MediaPipe `eyeBlinkLeft/Right` 真值），但 **10fps 采样** 对 100–400ms 眨眼严重欠采样 | ✅ Profile 卡 + 融合证据 + 咨询师端 | ✅ **是**（eye 权重 0.02–0.03，另进个人基线） | ⚠️ 可保留，但通话期应把面部分析提到 ≥30fps，否则 blinkRate 系统性偏低 |
| 低头帧占比 / 头姿波动 | **底层真、构念空**：角度确由 `facialTransformationMatrixes` 真算，但它是**头部姿态**不是**注视方向**；字段名 `downwardGazeRatio/attentionScatter` 误导 | ✅ 三处 UI（已加"头姿代理"标注） | ✅ **是**（>0.5 / >0.6 直接加 riskScore） | ⚠️ 可保留但应改名 `headDownRatio` / `headYawStd` |
| rPPG 心率 / HRV | **空**：绿通道峰值计数 + `hrvRmssd = 50 - |HR-baseline|*0.5` 纯拟合公式；现已被开关整体关闭 | ❌ 只渲染灰色 `-- bpm` 占位卡 | ❌ 已被 `RPPG_AVAILABLE &&` 挡住（AnxietyContext:644） | ❌ 不可。网络摄像头在普通室内光下测 HRV 不成立 |
| 呼吸频率 / 叹气 / 规律性 | **空**：麦克风 RMS 过零率（跟摄像头无关），读数在 5–40 次/分乱跳；现已被开关关闭 | ❌ 灰色 `--` 占位卡 | ❌ 已被 `BREATHING_AVAILABLE &&` 挡住（AnxietyContext:645） | ❌ 不可。应改用通话音频的 ASR 停顿/气口节奏 |
| 昼夜活动（深夜使用 / 使用规律性） | **非摄像头**：时钟 + App 内活动时间戳；数值真实但**语义不是睡眠** | ✅ | ✅ **是**（权重 0.06，另进叙事报告） | ✅ 与通话正交，可继续用 |
| 行为激活（互动次数 / 探索率 / 趋势） | **非摄像头**：点击次数 + 路由访问；数值真实 | ✅ | ✅ **是**（权重 0.03–0.04） | ✅ 与通话正交，可继续用 |
| 像素回退面部分析（MediaPipe 不可用时） | **半空**：像素差是真算的，但区域坐标写死、AU 映射任意；**且从不发帧信号** → 眨眼/眼动整条链静默失效 | ⚠️ 仅当 MediaPipe 失败时 | ✅ 是（`facialMetrics.confidence=0.5` 仍进融合） | ❌ 不建议带入通话；失败时应显示"视觉能力不可用"而不是静默降级 |
| Python 侧 `eye_tracking.py` / `rppg_hrv.py` / `breathing.py` / `circadian_rhythm.py` / `behavioral_activation.py` / `facial_expression.py` / `hierarchical_fusion.py` | **不可达（实质为空）**：`AnalyzeRequest` 根本没有 `eye_features / hrv_features / breathing_features / circadian_features` 字段，前端也从不发 `facial_features` | ❌ | ❌ | ❌ 直接删掉或明确标注为离线实验代码 |

**一句话结论**：通话功能可以复用的"真"东西只有三样 —— **MediaPipe 面部 AU/blendshape、由 blendshape 推的眨眼、由 transformation matrix 推的头部姿态**。心率、呼吸、注视方向都是空壳；Python 感知层与摄像头**没有任何连接**。

---

## 2. 逐项证据

### 2.1 面部微表情 / AU / 主导表情 —— `useFacialAnalysis.ts`

**位置**：核心计算 `client/src/hooks/useFacialAnalysis.ts:141-347`（`analyzeFrameMediaPipe`）；AU 提取 `:161-194`；情绪映射 `:254-267`。

**输入 [实测]**：真·摄像头帧。`:476-478` 取 `getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:480}}})`，`:481-485` 建 `<video>`，`:513` 用 `setInterval` 按 100ms 喂给 `faceLandmarker.detectForVideo(video, timestamp)`（`:148`）。同时开启 `outputFaceBlendshapes: true` 与 `outputFacialTransformationMatrixes: true`（`:125-126`）。

**输出 [实测]**（`:302-343`）：`actionUnits.au1_browRaise … au26_jawDrop`、`actionUnits.eyeAspect`、`actionUnits.headPitch/headYaw/headRoll`、`geometry.{browDistance,mouthWidth,eyeAspect,mouthOpen}`、`dominantExpression`、`expressionIntensity`、`emotionMapping.{happiness,sadness,anger,fear,surprise,disgust,distress}`、`microExpressions.count/avgDuration/recentEmotions`、`confidence`、`frameCount`。

**真 / 空判定：输入真、映射未验证（半真）。**

① AU 值是 MediaPipe 模型的直接产物，是真测量 [实测]：

```ts
// useFacialAnalysis.ts:166-176
au1 = bsMap['browInnerUp'] || 0;
au2 = bsMap['browOuterUpLeft'] || bsMap['browOuterUpRight'] || 0;
au4 = bsMap['browDownLeft'] || bsMap['browDownRight'] || 0;
au5 = bsMap['eyeWideLeft'] || bsMap['eyeWideRight'] || 0;
au6 = bsMap['cheekSquintLeft'] || bsMap['cheekSquintRight'] || 0;
au7 = bsMap['eyeSquintLeft'] || bsMap['eyeSquintRight'] || 0;
au9 = bsMap['noseSneerLeft'] || bsMap['noseSneerRight'] || 0;
au10 = bsMap['upperLipRaiserLeft'] || bsMap['upperLipRaiserRight'] || 0;
...
au26 = bsMap['jawOpen'] || 0;
```

② 但 AU → 情绪分数是**手写线性组合 + 手写放大系数**，没有任何训练/验证依据 [实测]：

```ts
// useFacialAnalysis.ts:254-262
const emotionMapping = {
  happiness: Math.min(1, (au6 * 0.5 + au12 * 0.5) * (au6 > 0.3 && au12 > 0.3 ? 1.5 : 0.4)),
  sadness: Math.min(1, (au1 * 0.4 + au4 * 0.4 + au15 * 0.5) * (au15 > 0.3 ? 1.3 : 1)),
  anger: Math.min(1, (au4 * 0.5 + au5 * 0.4 + au7 * 0.4) * (au4 > 0.4 ? 1.3 : 1)),
  fear: Math.min(1, (au1 * 0.4 + au2 * 0.4 + au26 * 0.5)),
  surprise: Math.min(1, (au1 * 0.35 + au2 * 0.35 + au5 * 0.35 + au26 * 0.35) * (au1 > 0.3 && au2 > 0.3 && au5 > 0.3 ? 1.4 : 1)),
  disgust: Math.min(1, (au9 * 0.6 + au10 * 0.6) * (au9 > 0.3 && au10 > 0.3 ? 1.4 : 1)),
  distress: Math.min(1, (au1 * 0.35 + au4 * 0.35 + au7 * 0.35 + au15 * 0.35)),
};
```

`1.5 / 0.4 / 1.3 / 1.4` 这些"协同放大"系数是拍的；`expressionIntensity = Σ/2` 也是拍的 [实测]：

```ts
// useFacialAnalysis.ts:267
const expressionIntensity = Math.min(1, Object.values(emotionMapping).reduce((s, v) => s + v, 0) / 2);
```

③ **硬编码常量冒充测量值** —— `微表情平均时长` 只要有过一次计数就恒为 `200ms` [实测]：

```ts
// useFacialAnalysis.ts:330-334
microExpressions: {
  count: microExprCountRef.current,
  avgDuration: microExprCountRef.current > 0 ? 200 : 0,   // ← 硬编码
  recentEmotions: microExprCountRef.current > 0 ? [dominantExpression] : [],
},
```

④ **微表情计数规则不是 FACS 微表情检测**：它只统计"主导表情在一次计时窗内发生变化"，且只有非"平静"时才更新计时器 [实测]：

```ts
// useFacialAnalysis.ts:270-280
const microThreshold = faceConfig.microExpressionThreshold * 1000; // 转换为 ms
const now = Date.now();
if (dominantExpression !== prevExprRef.current && dominantExpression !== '平静') {
  if (prevExprTimeRef.current > 0 && now - prevExprTimeRef.current < microThreshold) {
    microExprCountRef.current++;
  }
}
if (dominantExpression !== '平静') {
  prevExprRef.current = dominantExpression;
  prevExprTimeRef.current = now;
}
```

注意：它的判据是"两次**主表达式切换**间隔短"，不是"AU 强度瞬态 < 500ms"；且因为按 10fps 采样，最小区分粒度就是 100ms —— 与文件头宣称的 `FACS / 微表情 <500ms (Ekman, 2001)` 不是一回事。

⑤ **置信度是帧计数常量，不是信号质量** [实测]：

```ts
// useFacialAnalysis.ts:283
const confidence = useMediaPipeRef.current ? Math.min(0.95, 0.7 + frameCountRef.current * 0.005) : 0.5;
```

约 50 帧（5 秒）后 confidence 就饱和到 0.95，之后**与画面质量、是否真的检测到人脸完全无关**。而它直接决定面部模态权重：

```ts
// AnxietyContext.tsx:631-633
if (facialMetrics.isDetecting) {
  activeModalities.push('facial');
  weights.facial = Math.min(fusionConfig.facial, Math.max(fusionConfig.facial * 0.5, facialMetrics.confidence * fusionConfig.facial));
}
```

**Gating [实测]**：`perceptionCapabilities.ts` **完全没有提到面部/摄像头**（该文件只声明 rPPG / 呼吸 / 睡眠三项）。摄像头可用性是**运行期**在 hook 内部 try/catch 探测的，且失败后只 `console.warn`，**不产生任何 UI 提示**：

```ts
// useFacialAnalysis.ts:133-137
} catch (err) {
  console.warn('[面部分析] ⚠️ MediaPipe 初始化失败，回退到像素分析:', err);
  useMediaPipeRef.current = false;
  return false;
}
```

**进 UI [实测]**：
- `client/src/pages/patient/Profile.tsx:897-902`（面部卡：检测状态 / 帧数 / 表情）
- `client/src/pages/patient/Profile.tsx:1207-1243`（主导表情、微表情次数、AU12/AU6/AU4/AU1/AU15/AU9/AU7/AU26 标签）
- `client/src/pages/consultant/ConsultantRoom.tsx:1627-1629`（咨询师端看患者表情/微表情）
- `AnxietyContext.tsx:181-203` 写进 `narrativeAnalysis`；`:830` 写进 `evidence`

**影响 safety [实测]**：**是**。面部 → `weights.facial`（0.08/0.12/0.20，按年龄段，`ageConfig.ts:236-250`）→ `fusedProbs` → `anxietyProb = normalizedProbs[2]` → `riskLevel`：

```ts
// AnxietyContext.tsx:808-811
const anxietyProb = normalizedProbs[2];
const riskOffset = fusionConfig.riskThresholdOffset;
const adjustedAnxiety = Math.max(0, Math.min(1, anxietyProb - riskOffset));
const riskLevel = adjustedAnxiety > 0.7 ? 'crisis' : adjustedAnxiety > 0.5 ? 'high' : adjustedAnxiety > 0.3 ? 'medium' : 'low';
```

再经 `reportToServer`（`AnxietyContext.tsx:1055` `level: comprehensiveState.riskLevel`）→ `POST /api/profile/anxiety-report` → **写库**：

```ts
// server/src/controllers/profile.controller.ts:191-197
// 高风险时更新风险等级
if (level === 'high' || level === 'crisis') {
  await prisma.patientProfile.update({
    where: { userId: req.userId! },
    data: { riskLevel: level === 'crisis' ? 'HIGH' : 'MEDIUM' },
  });
}
```

⚠️ 也就是说：一个由"帧计数置信度 + 手写 AU 线性组合"驱动的数字，可以改写数据库里的 `patientProfile.riskLevel`。它**不**触发客户端的危机弹窗（弹窗只认服务端字段，见 `Chat.tsx:231` / `:292`），但它会改变咨询师/管理员看到的患者风险标签。**这是本次审计里最需要复核的一条链路。**

另外 `AnxietyContext.tsx:1034-1040` 用 `anxietyIndex > 60` 自动下发"现实任务"，`anxietyIndex = comprehensiveState.emotionProbs[2] * 100`（`:1021`）同样包含面部/眨眼贡献。

---

### 2.2 眨眼频率 / 平均眨眼时长 / 长眨眼 —— `useEyeTracking.ts` + `blinkLogic.ts`

（两条路径的甄别见 §3。）

**位置**：判定状态机 `blinkLogic.ts:66-172`；驱动与风险计算 `useEyeTracking.ts:55-126`。

**输入 [实测]**：**真实 blendshape 信号**，来自 `useFacialAnalysis` 同一帧：

```ts
// useFacialAnalysis.ts:188-194
// 眼睑闭合度（眨眼主路径）：双眼取较小值，单眼眨动/眯眼不计为眨眼
const blinkL = bsMap['eyeBlinkLeft'];
const blinkR = bsMap['eyeBlinkRight'];
blinkScore =
  blinkL !== undefined && blinkR !== undefined
    ? Math.min(blinkL, blinkR)
    : null;
```

```ts
// useFacialAnalysis.ts:298
onFrameSignal({ greenMean, eyeAspect, headPitch, headYaw, blinkScore });
```

**这不是 EAR 阈值法**，是模型直接输出的眼睑闭合度；EAR（`eyeAspect`）只在 `blinkScore === null` 时降级使用 [实测]：

```ts
// useEyeTracking.ts:62-67
const method: BlinkMethod = blinkScore !== null ? 'blendshape' : 'ear';
const closed =
  blinkScore !== null
    ? machineRef.current.isClosed(blinkScore)
    : eyeAspect < EAR_CLOSED_THRESHOLD;   // 0.15
machineRef.current.update(closed, now);
```

**输出 [实测]**（`useEyeTracking.ts:112-125`）：`blinkRate`（次/分）、`avgBlinkDuration`（ms）、`longBlinkCount`、`blinkSampleCount`、`blinkMethod`、`signalQuality`、`riskScore`。

**真 / 空判定：真测量，但有采样率偏差。**

公式本身是对的 [实测]：

```ts
// blinkLogic.ts:145-155
stats(nowMs: number, sessionStartMs: number): BlinkStats {
  const elapsed = (nowMs - sessionStartMs) / 1000;
  const recent = this.events.filter((e) => nowMs - e.time < 60000);
  const windowSec = Math.min(Math.max(elapsed, 1), 60);

  const blinkRate =
    windowSec > 0 ? Math.round((recent.length / windowSec) * 60) : 0;
```

分子是"最近 60 秒的眨眼数"、分母是"min(已过时长, 60) 秒" —— 量纲一致 ✓。**旧版曾经错**（见 §3 的 diff）。

判据链也是真的、可解释的 [实测]：

```ts
// blinkLogic.ts:79-83  迟滞
isClosed(score: number): boolean {
  return this.closed
    ? score > this.params.offThreshold      // 0.35
    : score > this.params.onThreshold;      // 0.5
}
// blinkLogic.ts:101-107  时长窗 + 最小间隔
const duration = nowMs - this.startMs;
if (duration < this.params.minMs || duration >= this.params.maxMs) {   // [60, 800)
  this.lastEndMs = nowMs;
  return null;
}
if (this.startMs - this.lastEndMs < this.params.minGapMs) return null; // 100ms
```

**必须指出的科学限制 [推断]**：帧源是 `setInterval(..., 100)`（10fps）。人一次眨眼闭眼期约 100–400ms，10fps 下**大部分眨眼会整段落在两帧之间**，根本不会被采样到；而采到的眨眼 duration 分辨率也只有 100ms。因此 `blinkRate` 是**系统性偏低的下界估计**，不是"测得准的频率"。正常 15–20 次/分可能被显示成个位数 —— 这会直接触发 `blinkRate < config.lowBlinkThreshold(=8)` 的"眨眼过少 → 解离"风险分支：

```ts
// useEyeTracking.ts:104-108
if (s.blinkRate > 0 && s.blinkRate < config.lowBlinkThreshold) {   // 8
  riskScore += (config.lowBlinkThreshold - s.blinkRate) / config.lowBlinkThreshold * 0.35;
}
```

**这是一个方向性错误的风险信号**：采样不足造成的低读数被解释成"注意力过度集中/解离"。⚠️

**Gating [实测]**：`perceptionCapabilities.ts` **不管理眨眼**。可用性完全隐式：只有 MediaPipe 成功 → `analyzeFrameMediaPipe` → 才发帧信号；否则（见 §2.9）整条链静默失效。`signalQuality` 是算出来的、不是常量 [实测]：

```ts
// useEyeTracking.ts:74-78
const elapsedSec = Math.max(0, (now - sessionStartRef.current) / 1000);
const expectedFrames = Math.max(1, Math.min(elapsedSec, WINDOW_MS / 1000) * EXPECTED_FPS);  // 5fps
const coverage = Math.min(1, frameTimes.length / expectedFrames);
const qualityBase = method === 'blendshape' ? QUALITY_BASE.blendshape : QUALITY_BASE.ear;
const signalQuality = Math.round(qualityBase * coverage * 100) / 100;
```

（注：`EXPECTED_FPS = 5` 而实际投喂 10fps，故 `coverage` 一上来就饱和到 1，`signalQuality` 基本恒为 0.9 —— **这个"质量分"实际上没有区分能力** [推断]。）

**进 UI [实测]**：
- `client/src/pages/patient/Profile.tsx:974-988`（生理行为层小卡）
- `client/src/pages/patient/Profile.tsx:1433-1453`（眨眼/头姿大卡，含来源标签 `眼睑闭合度` / `EAR 降级`）
- `client/src/pages/consultant/ConsultantRoom.tsx:1794-1816`
- `AnxietyContext.tsx:845-848` 写进融合证据

**影响 safety [实测]**：**是**。eye 模态活跃即进融合（权重 0.02–0.03）：

```ts
// AnxietyContext.tsx:647
if (eyeMetrics.isMeasuring) { activeModalities.push('eye'); weights.eye = fusionConfig.eye; }
// AnxietyContext.tsx:758
const eyeProbs5 = [0.1, eyeMetrics.riskScore * 0.2, eyeMetrics.riskScore * 0.5, 0.05, Math.max(0.1, 1 - eyeMetrics.riskScore * 0.3)];
```

并且进个人基线（`:893` `blinkRate: eyeMetrics.blinkRate || undefined`）和自适应权重相关记录（`:910`）。

---

### 2.3 低头帧占比 / 头姿波动 —— `useEyeTracking.ts:80-98`

**输入 [实测]**：头部 **pitch/yaw**，来自 MediaPipe 面部变换矩阵，**不是眼球注视**：

```ts
// useFacialAnalysis.ts:230-236
if (result.facialTransformationMatrixes && result.facialTransformationMatrixes.length > 0) {
  const matrix = result.facialTransformationMatrixes[0].data;
  if (matrix && matrix.length >= 12) {
    headPitch = Math.atan2(matrix[6], matrix[10]) * (180 / Math.PI);
    headYaw = Math.atan2(-matrix[2], Math.sqrt(matrix[6] ** 2 + matrix[10] ** 2)) * (180 / Math.PI);
    headRoll = Math.atan2(matrix[1], matrix[5]) * (180 / Math.PI);
  }
}
```

**输出 / 真空判定：底层角度真、构念空** [实测]：

```ts
// useEyeTracking.ts:87-98
// 低头帧占比
const downwardCount = poseHistory.filter(p => p.pitch > 10).length;
const downwardRatio = poseHistory.length > 0 ? downwardCount / poseHistory.length : 0;

// 头姿波动（偏航标准差）；不足 30 帧时不给值，避免把噪声当分散
let scatter = 0;
if (poseHistory.length > 30) {
  const yaws = poseHistory.map(p => p.yaw);
  const mean = yaws.reduce((a, b) => a + b, 0) / yaws.length;
  const variance = yaws.reduce((s, y) => s + (y - mean) ** 2, 0) / yaws.length;
  scatter = Math.min(1, Math.sqrt(variance) / 15);
}
```

`pitch > 10` 里的 `10°` 和 `/15` 都是拍的阈值/归一化常数 [推断]。**没有虹膜注视跟踪** → "向下注视占比"这一构念不成立；代码自己承认了这一点（工作区新增）：

```ts
// useEyeTracking.ts:18-20（文件头）
 * ⚠️ 口径说明：downwardGazeRatio / attentionScatter 由 **头部姿态** 推算
 *   （本平台未接入虹膜注视跟踪），分别是「低头帧占比」和「头部偏航波动」，
 *   是注视方向的代理指标而非眼球注视方向。字段名沿用历史接口以保证兼容。
```

以及 `poseProxy: true`（`:122`）。UI 也加了说明（`Profile.tsx:1451-1452`、`ConsultantRoom.tsx:1810-1811`）。**这是本栈里"诚实标注"做得最好的一处**：数值是真的，名字是旧的。

**Gating**：无 capability 开关；依赖面部分析是否在跑。**进 UI [实测]**：`Profile.tsx:981`、`Profile.tsx:1445-1446`、`ConsultantRoom.tsx:1807-1808`。

**影响 safety [实测]**：**是**，且权重比看起来更大：

```ts
// useEyeTracking.ts:109-110
if (downwardRatio > 0.5) riskScore += Math.min(0.35, (downwardRatio - 0.5) / 0.3 * 0.35);
if (scatter > 0.6) riskScore += Math.min(0.3, (scatter - 0.6) / 0.3 * 0.3);
```

---

### 2.4 rPPG 心率 / HRV —— `useRPPG.ts`（**空，现已关闭**）

**输入 [实测]**：**不是**人脸区域，是全帧缩到 32×32 后一个**写死的 16×12 矩形**的绿通道均值：

```ts
// useFacialAnalysis.ts:286-299
if (onFrameSignal) {
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = 32; tmpCanvas.height = 32;
  const tmpCtx = tmpCanvas.getContext('2d', { willReadFrequently: true });
  if (tmpCtx) {
    tmpCtx.drawImage(video, 0, 0, 32, 32);
    const faceRegion = tmpCtx.getImageData(8, 4, 16, 12);   // ← 固定矩形，未随人脸移动
    ...
    const greenMean = gCount > 0 ? gSum / gCount : 0;
    onFrameSignal({ greenMean, eyeAspect, headPitch, headYaw, blinkScore });
  }
}
```

注释里叫 `faceRegion`，但它只是 32×32 图里的 `(8,4)-(24,16)`；人脸一动，这个矩形就落在背景/额头/头发上 [实测]。

**输出（设计上）**：`heartRate`、`hrvRmssd`、`signalQuality`、`riskScore`。

**真 / 空判定：空（假公式），且现在被整体关闭。**

假公式一：所谓"峰值检测"就是三点局部极大计数，没有任何滤波/频域判定 [实测]：

```ts
// useRPPG.ts:34-45
const signal = hrHistoryRef.current;
// 简单峰值检测估计心率
let peaks = 0;
for (let i = 2; i < signal.length - 2; i++) {
  if (signal[i] > signal[i-1] && signal[i] > signal[i-2] &&
      signal[i] > signal[i+1] && signal[i] > signal[i+2]) {
    peaks++;
  }
}
// 估算心率（假设采样率约 10fps，30 个采样点 = 3 秒）
const durationSec = 30 / 10;
const estimatedHR = (peaks / durationSec) * 60;
```

`durationSec = 30/10 = 3` 是**硬编码**，且分母永远是 3 秒 —— 而 `hrHistoryRef` 会累积到 180 点（`:29`），所以实际统计窗口和分母根本不匹配 [实测]。

假公式二：HRV 不是从峰间间隔算的，是从心率与年龄基线的差值**反推**的 [实测]：

```ts
// useRPPG.ts:49-50
// HRV 简化估计（峰间间隔标准差）
const hrvRmssd = Math.max(5, 50 - Math.abs(estimatedHR - config.restingHRBaseline) * 0.5);
```

**科学上不成立**：**网络摄像头 rPPG 无法可靠恢复 HRV**。即使 HR 在受控条件下可估，RMSSD 需要逐搏间隔的毫秒级精度，摄像头 30fps 采样（33ms 量化）+ 自动曝光/压缩抖动，误差远大于 RMSSD 本身的量级（数十 ms）。代码里这个"HRV"完全绕开逐搏间隔，因此它不是测量，是拟合 [实测+推断]。

**已被关闭 [实测]**：

```ts
// useRPPG.ts:22-25
const updateFromFrame = useCallback((greenChannelMean: number, timestamp: number) => {
  // 不具备心率采集能力：不采样、不估算、不更新任何指标
  if (!RPPG_AVAILABLE) return;
```

```ts
// perceptionCapabilities.ts:31
export const RPPG_AVAILABLE = envFlag(env.VITE_ENABLE_RPPG);
// :16-18
function envFlag(value: unknown): boolean {
  return value === 'true' || value === '1';
}
```

**可用性是"编译期环境变量"而不是"运行期测量" [实测]**：`RPPG_AVAILABLE` 来自 `import.meta.env`，默认 false。实测工作区 `.env` 内容只有 `DASHSCOPE_API_KEY=...`，**没有** `VITE_ENABLE_RPPG` → 恒为 `false`。所以它**永远**在同一个二进制里取 false，不会因为"用户换了台带传感器的设备"而变 true。

**进 UI [实测]**：数值路径实际不可达。`Profile.tsx:944-954` 渲染灰底 `心率：-- bpm` 占位；`ConsultantRoom.tsx:1728` 用 `hrv?.isMeasuring` 门控，false 时走 `:1720-1724` 的"未采集（无心率传感器）"。**没有任何地方会渲染 `heartRate` 的真实数字。**

**影响 safety [实测]**：**否**，双重拦截：

```ts
// AnxietyContext.tsx:640-645
// ⚠️ 本设备不具备心率/呼吸采集能力，这两个模态永不参与融合，
if (RPPG_AVAILABLE && hrvMetrics.isMeasuring) { activeModalities.push('hrv'); weights.hrv = fusionConfig.hrv; }
if (BREATHING_AVAILABLE && breathingMetrics.isMeasuring) { activeModalities.push('breathing'); weights.breathing = fusionConfig.breathing; }
```

证据链里也刻意不写（`:841` 注释：`心率/呼吸不具备采集能力：不写入 evidence`），叙事报告里同样删过（`:230-232`）。这部分修复是**真实且完整**的 [实测]。

---

### 2.5 呼吸频率 / 叹气 / 规律性 —— `useBreathing.ts`（**空，现已关闭**）

**输入 [实测]**：**麦克风**，不是摄像头。`useVoiceAnalysis.ts:122` 每个 rAF 帧回调 `if (onAudioEnergy) onAudioEnergy(rms);`（rms 来自 `getFloatTimeDomainData`，`:108-116`）。

**输出**：`breathingRate`、`regularityCV`、`sighCount`、`signalQuality`、`riskScore`。

**真 / 空判定：空。** 核心公式是"能量序列过零率 ÷ 2 ÷ 秒数"，把一切能量起伏都算成呼吸 [实测]：

```ts
// useBreathing.ts:40-48
// 计算过零率（呼吸周期指标）
const mean = smoothed.reduce((a, b) => a + b, 0) / smoothed.length;
let crossings = 0;
for (let i = 1; i < smoothed.length; i++) {
  if ((smoothed[i] - mean) * (smoothed[i-1] - mean) < 0) crossings++;
}

// 呼吸频率估计（假设 10fps 采样）
const breathsPerMinute = (crossings / 2) / (smoothed.length / 10) * 60;
```

`/ 10` 这个"假设 10fps"也是硬编码，而实际调用频率是 rAF 的 ~60fps [实测+推断] —— 即使不看噪声，量纲就已经错了 6 倍。

**已关闭 [实测]**：

```ts
// useBreathing.ts:23-25
const updateFromAudioEnergy = useCallback((rmsEnergy: number) => {
  // 不具备呼吸检测能力：不采样、不估算、不更新任何指标
  if (!BREATHING_AVAILABLE) return;
```

`BREATHING_AVAILABLE = envFlag(env.VITE_ENABLE_BREATHING)`（`perceptionCapabilities.ts:41`），`.env` 未设置 → 恒 false [实测]。

**进 UI**：`Profile.tsx:955-965` 灰卡 `频率：-- 次/分`；`ConsultantRoom.tsx:1748-1758` "未采集（算法未启用）"。数字不可达。

**影响 safety**：否（`AnxietyContext.tsx:645` 拦截，`:841` 不写 evidence）。

---

### 2.6 昼夜活动 —— `useCircadianRhythm.ts`（**非摄像头**）

**输入 [实测]**：**系统时钟 + App 内活动时间戳**，与摄像头无关：

```ts
// useCircadianRhythm.ts:22-29
const recordActivity = useCallback(() => {
  const now = Date.now();
  activityTimestampsRef.current.push(now);
  ...
```
调用点：`AnxietyContext.tsx:569`（全局 `input` 事件里 `circadian.recordActivity()`）。

**输出 / 真空判定：真测量，但被"睡眠"语义借用过。** 数值含义是"App 使用时间的深夜程度与离散度"：

```ts
// useCircadianRhythm.ts:36-43
let lateNightRisk = 0;
if (hour >= config.lateActivityRiskHour || (hour < 5 && hour < 6.5)) {
  const hoursPast = hour >= config.lateActivityRiskHour
    ? hour - config.lateActivityRiskHour
    : (24 - config.lateActivityRiskHour) + hour;
  lateNightRisk = Math.min(1, 0.3 + hoursPast * 0.1);
}
```
⚠️ `hour < 5 && hour < 6.5` 是**退化条件**（等价于 `hour < 5`）[实测] —— 无害但是写错了，原意应是 `hour < 6.5`。

`MIN_SAMPLES_FOR_REGULARITY = 10` 门槛是真的（`:47-57`），样本不足时不再输出伪"规律性百分比"。

文件头已明确否认这是睡眠检测（`:4-6`）。

**Gating**：无开关（也不需要）。
**进 UI [实测]**：`Profile.tsx` 模态面板；叙事报告 `AnxietyContext.tsx:208-214`；证据 `:836`。
**影响 safety [实测]**：**是**，权重 0.06：

```ts
// AnxietyContext.tsx:642
if (circadianMetrics.isActive) { activeModalities.push('circadian'); weights.circadian = fusionConfig.circadian; }
// AnxietyContext.tsx:753
const circProbs5 = [0.05, circadianMetrics.riskScore * 0.4, circadianMetrics.riskScore * 0.5, 0.05, Math.max(0.1, 1 - circadianMetrics.riskScore * 0.5)];
```

---

### 2.7 行为激活 —— `useBehavioralActivation.ts`（**非摄像头**）

**输入 [实测]**：平台内互动计数 + 路由访问。`AnxietyContext.tsx:568` `behavioralActivation.recordInteraction()`；`:942` `behavioralActivation.recordModuleVisit(moduleName)`。

**输出 / 真空判定：真测量（计数是真的）。**

```ts
// useBehavioralActivation.ts:31-37
const count = interactionCountRef.current;
const explorationRate = visitedModulesRef.current.size / 10; // 假设 10 个功能模块

let withdrawalScore = 0;
if (count < config.minDailyInteractions) {
  withdrawalScore = Math.min(1, (1 - count / config.minDailyInteractions) * 0.7);
}
```
`/10` 的分母"假设 10 个功能模块"是硬编码 [实测]（模块数变了就失真）；`isActive: count > 0`（`:73`）与"至少 4 天才判趋势"（`:49`）是刚加的真实性修复。

**Gating**：无开关。**进 UI**：`Profile.tsx:966-972`、`ConsultantRoom.tsx:1780-1791`、叙事 `:235-239`、证据 `:843`。
**影响 safety**：**是**，权重 0.03–0.04（`:646`、`:757`、`:788`）。

---

### 2.8 睡眠

**不是模态**，只是显式声明"不可自动检测" [实测]：`AnxietyContext.tsx:61` `'睡眠（无夜间传感器）'` 进入 `UNAVAILABLE_MODALITIES`，`perceptionCapabilities.ts:50` `SLEEP_AUTO_DETECTION_AVAILABLE = envFlag(env.VITE_ENABLE_SLEEP_AUTO)` → false。不进融合。✓ 处理正确。

---

### 2.9 像素回退面部分析 —— `useFacialAnalysis.ts:352-472`（**半空，且会静默掐断眨眼链**）

MediaPipe 初始化失败时被选中 [实测]：

```ts
// useFacialAnalysis.ts:510-515
// 选择分析函数：MediaPipe 用 100ms (10fps)，像素分析用 200ms (5fps)
const analyzeFn = useMediaPipeRef.current ? analyzeFrameMediaPipe : analyzeFramePixel;
const fps = useMediaPipeRef.current ? 100 : 200;
intervalRef.current = window.setInterval(analyzeFn, fps);
```

**区域坐标写死** [实测]：

```ts
// useFacialAnalysis.ts:367-371
const regions = {
  brow: { x: 40, y: 25, w: 80, h: 20 },
  eye: { x: 40, y: 40, w: 80, h: 15 },
  mouth: { x: 50, y: 75, w: 60, h: 20 },
};
```

**AU 映射任意** [实测]：

```ts
// useFacialAnalysis.ts:428-433
const au1 = Math.min(1, browMotion / 5);
const au4 = Math.min(1, Math.max(0, (10 - (regionBrightness.brow || 0)) / 20));
const au6 = Math.min(1, (regionBrightness.eye || 0) / 100);
const au12 = Math.min(1, mouthMotion / 4);
const au15 = Math.min(1, Math.max(0, (100 - (regionBrightness.mouth || 0)) / 30));
const au26 = Math.min(1, mouthMotion / 6);
```

**关键缺陷：这条路径从不调用 `onFrameSignal`** [实测]。全文件 `onFrameSignal` 只出现在两处（`useFacialAnalysis.ts:286` 与 `:298`），**都在 `analyzeFrameMediaPipe` 内部**：

```
grep onFrameSignal in useFacialAnalysis.ts:
  Line  90: export function useFacialAnalysis(... onFrameSignal? ...)
  Line 286:       if (onFrameSignal) {
  Line 298:           onFrameSignal({ greenMean, eyeAspect, headPitch, headYaw, blinkScore });
```

后果 [推断]：MediaPipe 不可用 → `analyzeFramePixel` 跑 5fps → `eyeTracking.updateFromFacialFrame` **一次也不会被调用** → `eyeMetrics.isMeasuring` 永远是 `false`（不是 0 次/分，而是"未测量"）→ 眨眼模态整体不活跃。**它不会输出"0 次眨眼"这种假数据，而是静默消失**，且 UI 上没有任何"视觉能力已降级"的提示（只有 `console.warn`）。这比输出错值好，但对"通话功能是否可信"是个盲区。

---

### 2.10 Python 侧：`algorithm/perception/**`（**结论：与摄像头没有任何连接**）

**API 面 [实测]**：`algorithm/api/perception.py:33-44`：

```python
class AnalyzeRequest(BaseModel):
    """感知分析请求体"""
    text: str = Field(..., description="待分析的文本内容")
    wav_path: Optional[str] = Field(None, description="WAV 音频文件路径（可选）")
    facial_features: Optional[dict] = Field(None, description="面部特征字典（可选，含 AU 强度等）")
    behavior_features: Optional[dict] = Field(None, description="行为特征字典（可选，含活跃时段等）")
    age_group: Optional[str] = Field(...)
    age: Optional[int] = Field(None, description="用户实际年龄（可选，自动推断年龄分组）")
    user_id: Optional[str] = Field(None, description="用户 ID（可选，用于更新心理数字孪生）")
```

**没有** `eye_features` / `hrv_features` / `breathing_features` / `circadian_features` / `behavioral_activation_features` / `voice_semantics_features`。而 `PerceptionService.analyze_multimodal` 虽然接受这些参数（`perception_service.py:850-866`），**没有任何入口能传进来**；grep 全仓的 `eye_features|hrv_features|breathing_features|circadian_features` 命中**只**在 `perception_service.py` 自身的形参与分支里。→ `eye/eye_tracking.py`、`physiological/rppg_hrv.py`、`physiological/breathing.py`、`circadian/circadian_rhythm.py`、`behavioral_activation/behavioral_activation.py` 这 5 个文件在线上**不可达** [实测]。

**前端是否发 `facial_features`？否** [实测]。三处真实调用：

```ts
// client/src/pages/patient/Chat.tsx:176-180
const res = await fetch(`${PERCEPTION_API}/api/v1/perception/analyze`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text }),
});
```
```ts
// client/src/pages/consultant/ConsultantRoom.tsx:399-403
const res = await fetch(`${PERCEPTION_API}/api/v1/perception/analyze`, {
  ...
  body: JSON.stringify({ text, behavior_features: behaviorFeatures }),
});
```
（`behaviorFeatures` 在这里是空对象，`ConsultantRoom.tsx:395-398`：`if (data?.anxiety) { behaviorFeatures = {}; }`）

第三处 `client/src/hooks/usePerception.ts` —— **这个 hook 没有任何页面 import**（grep `usePerception` 只命中它自己），属于死代码；它的 `analyzeMultimodal`（`:178-209`）虽然能带 `facial_features`，但没有调用方。

**唯一一处"想传面部特征"的代码在 Node 服务端，而且是硬编码假值** [实测]：

```ts
// server/src/services/algorithm-bridge.ts:614-627
// 从帧提取面部特征（简化版：使用图像亮度/对比度作为代理特征）
if (frameBase64) {
  try {
    const imgBuf = Buffer.from(frameBase64, 'base64');
    // 简单面部特征代理：计算图像平均亮度和对比度
    // 真实场景应使用 face-api.js 或 OpenCV
    facialFeatures = {
      brightness: 128, // 默认值，实际应从图像计算
      contrast: 50,
    };
  } catch {
    // 帧处理失败，忽略
  }
}
```

后果链 [实测+推断]：`FacialFeatures`（`face/facial_expression.py:43-81`）**没有** `brightness`/`contrast` 字段 → `perception_service.py:214` 的 `hasattr` 过滤把它们丢掉 → 得到一个**全默认的 `FacialFeatures()`** → 所有 AU = 0 → `emotion_scores` 全 0 → `features_to_emotion_risk` 返回 `risk_score = 0.0`（不是 `None`）：

```python
# face/facial_expression.py:511-523
total = sum(emotion_scores.values())
if total > 0:
    emotion_probs = {k: v / total for k, v in emotion_scores.items()}
else:
    emotion_probs = {"neutral": 1.0}
    for name in AU_EMOTION_RULES:
        emotion_probs[name] = 0.0
negative_risk = sum(
    emotion_probs.get(name, 0.0)
    for name in ["sadness", "fear", "anger", "distress"]
)
```
而这个 `facial_risk = 0.0` 会被当成**有效模态**塞进融合（`perception_service.py:1006-1010`，`confidence=0.7`），`huaf_risk_to_probs(0.0)` 产出 `[0.4, 0, 0, 0, 0.6]` —— 一个"快乐+中性"的 L1 层模态，**会往下拉融合焦虑分** [推断]。好在这条路目前只有 `algorithm/routes` 里的 `/algorithm/perception/upload` 能到（`server/src/routes/algorithm.routes.ts:160-166`），而**没有任何前端页面调用它** [实测] → 当前是"潜伏缺陷"，不是活跃缺陷。

**`fusion/fusion.py`** [实测]：只做文本+音频加权/Stacking/动态加权（`:82-90` `fuse_weighted_average(text_probs, audio_probs, ...)`），**完全没有摄像头/面部输入**。而默认策略是 HUAF：

```python
# perception_service.py:344
DEFAULT_FUSION_STRATEGY = "hierarchical"
```
→ 走 `:994-1055` 的 HUAF-TC 分支，`fusion.py` 只在显式改用其他策略时才被调用，实际是备用代码。

**`fusion/hierarchical_fusion.py`** [实测]：`LAYER_MODALITIES`（`:135-139`）把 `"hrv"/"breathing"/"eye"` 放在 **L3 生理背景层**，权重先验 `AGE_PRIORS`（`:127-132`）在大学档给 L3 只有 `0.27`；`_fuse_layer` 用熵+置信度做不确定性加权（`:254-257`）。算法本身是自洽的，**但输入永远是空**（因为上面那些特征传不进来）→ 这层"11 模态层级融合"在线上实际只跑 `text`（+ 可能 `cognitive`，`:933-940` 从文本自动算）。

`algorithm/api/perception.py` 里另有两个与摄像头无关的端点：`/perception/debias`、`/perception/debias/batch`、`/perception/age-groups`。

---

## 3. 眨眼检测的两条路径

### 3.1 哪条在跑？—— `useEyeTracking` + `blinkLogic` 是活跃链路；`useBlinkDetection` 是死代码

**挂载点 [实测]**（`client/src/context/AnxietyContext.tsx`）：

```ts
// :22
import { useFacialAnalysis, type FacialFrameSignal } from '../hooks/useFacialAnalysis';
// :29
import { useEyeTracking } from '../hooks/useEyeTracking';
// :413
const facial = useFacialAnalysis(ageGroup, onFrameSignal);
// :420
const eyeTracking = useEyeTracking(ageGroup);
// :381-385
const onFrameSignalRef = useRef<(signal: FacialFrameSignal) => void>();
onFrameSignalRef.current = (signal) => {
  rppg.updateFromFrame(signal.greenMean, Date.now());
  eyeTracking.updateFromFacialFrame(signal);
};
```

`useBlinkDetection` **的导入点为零**。全仓 grep `useBlinkDetection`：

```
client/src/hooks/blinkLogic.ts:4         * 从 useBlinkDetection 中抽出，...
client/src/hooks/useBlinkDetection.ts:20 * 判定规则见 blinkLogic.ts
client/src/hooks/useBlinkDetection.ts:28 } from './blinkLogic';
client/src/hooks/useBlinkDetection.ts:30 export type { BlinkMethod } from './blinkLogic';
client/src/hooks/useBlinkDetection.ts:55 export function useBlinkDetection() {
client/src/hooks/useEyeTracking.ts:6     *  ... blinkLogic.BlinkStateMachine ...
client/src/hooks/useEyeTracking.ts:29    import { BLENDSHAPE_PARAMS, BlinkStateMachine, ... } from './blinkLogic';
```

→ `useBlinkDetection.ts` **是死代码** [实测]：它自己会 `getUserMedia` 开第二路摄像头（`:207-209`）、自己再建一个 `FaceLandmarker`（`:79-88`）、自己跑 `requestAnimationFrame` 循环（`:250`），但没有任何组件挂载它。它的亮度兜底路径（`getEyeRegionBrightness` `:98-117`，固定 `width*0.35 × height*0.12` 矩形）因此**永不执行**。

### 3.2 眨眼频率公式，以及那个窗口/分母 bug

**现状（工作区已修）** [实测]：

```ts
// blinkLogic.ts:145-151
stats(nowMs: number, sessionStartMs: number): BlinkStats {
  const elapsed = (nowMs - sessionStartMs) / 1000;
  const recent = this.events.filter((e) => nowMs - e.time < 60000);
  const windowSec = Math.min(Math.max(elapsed, 1), 60);

  const blinkRate =
    windowSec > 0 ? Math.round((recent.length / windowSec) * 60) : 0;
```

**修复前的版本（HEAD 里的）** [实测，来自 `git diff`]：

```diff
-    const blinkRate =
-      elapsed > 0 ? Math.round((recent.length / Math.max(elapsed, 1)) * 60) : 0;
+    const windowSec = Math.min(Math.max(elapsed, 1), 60);
+    const blinkRate =
+      windowSec > 0 ? Math.round((recent.length / windowSec) * 60) : 0;
```

**这就是你怀疑的那个 bug，而且它确实存在过**：分子 `recent` 是"最近 60 秒的眨眼数"，分母却是"会话总时长"。10 分钟会话里稳定的 15 次/分会被摊薄成 `15/10 = 1.5 次/分`，然后触发 `< lowBlinkThreshold(8)` 的"眨眼过少/解离"风险分支（`useEyeTracking.ts:104-105`）→ **长时间使用的用户会被系统性地判为解离风险**。当前工作区版本已把它改成 `min(max(elapsed,1),60)` 秒，量纲一致，60 秒以内与旧版等价。**修复是真的。** [实测]

小瑕疵 [推断]：
- 会话头 1 秒内 `windowSec` 被 clamp 到 1，若此时恰好记到 1 次眨眼会瞬时显示 60 次/分（下一帧就回落），属于无害瞬态。
- `BLENDSHAPE_PARAMS.minGapMs = 100`（`blinkLogic.ts:45`），但文件头注释写"最小间隔 100ms"、commit message 写"120ms"，二者不一致（代码以 100 为准）。
- `useBlinkDetection.ts` 里还有一处**未被修复**的同类问题：它 `start()` 时把 `lastPublishRef.current = 0`、`sessionStartRef.current = Date.now()`（`:239-240`），但 `publishMetrics` 用 `METRIC_INTERVAL_MS = 3000` 节流（`:53`、`:120-121`）——由于该 hook 已死代码，不影响线上。

### 3.3 其它已确认的修复（工作区未提交改动）

`git status` 显示这些文件有未提交改动，`git diff` 显示的具体修改 [实测]：

| 文件 | 改了什么 |
| --- | --- |
| `blinkLogic.ts` | ① `BlinkMethod` 新增 `'ear'` 取值；② `stats()` 分母修正（上节） |
| `useEyeTracking.ts` | 整文件重写：从"单帧 EAR 穿越计数（`eyeAspect<0.15 && prev>0.2`）"改为"blendshape 迟滞 + 60~800ms 时长窗 + 100ms 最小间隔"状态机；`signalQuality` 从 `0.7/0.2` 两档常量改为"来源基准分 × 帧覆盖率"；新增 `blinkMethod` 如实上报来源；新增 `poseProxy` 标记；`gazeHistoryRef` 里 `{x: headPitch, y: headPitch}` 的复制粘贴错误被删除 |
| `useFacialAnalysis.ts` | `FrameSignalCallback` 从位置参数 `(greenMean, eyeAspect, headPitch)` 改为对象 `FacialFrameSignal{greenMean, eyeAspect, headPitch, headYaw, blinkScore}`；新增 `blinkScore = Math.min(eyeBlinkLeft, eyeBlinkRight)`，**缺失时保持 `null` 而不是默认 0** |
| `useRPPG.ts` / `useBreathing.ts` | 开头加 `if (!RPPG_AVAILABLE) return;` / `if (!BREATHING_AVAILABLE) return;` |
| `useCircadianRhythm.ts` | 加 `MIN_SAMPLES_FOR_REGULARITY = 10`；样本不足时 `riskScore` 只算 `lateNightRisk*0.6`；新增 `activitySampleCount` |
| `useBehavioralActivation.ts` | 加 `lastDayRef` 跨天判据（修掉"同一会话内累计值伪造下降趋势"）；趋势需 ≥4 天；`isActive: count > 0` |
| `AnxietyContext.tsx` | `toggleCamera` 关闭时显式 `eyeTracking.reset()/rppg.reset()/breathingHook.reset()`（修掉"关摄像头后冻结读数仍在活跃模态里"）；`UNAVAILABLE_MODALITIES` + `AVAILABILITY_NOTE` 诚实披露；叙事报告删除心率/呼吸段落 |

其中"`blinkScore` 缺失时必须是 `null` 而不是 0"这一条改动很关键 —— 工作区代码里留了理由 [实测]：

```ts
// useFacialAnalysis.ts:158-159
// 眼睑闭合度：blendshape 缺失时必须保持 null，不能默认为 0（否则等价于「永远睁眼」）
let blinkScore: number | null = null;
```

若默认为 0，则 `isClosed(0)` 恒 false → 永远"睁眼" → `blinkRate` 恒为 0 → 命中"眨眼过少"分支。这个坑现在被堵住了。

**遗留问题 [实测]**：`blinkLogic.ts` 的 docstring 声称"让判定规则可被独立测试"，但 `client/package.json` 里**没有任何测试框架**（无 `vitest`/`jest`），`client/**/*.{test,spec}.{ts,tsx}` glob 结果为空 → `BlinkStateMachine` 目前**零测试覆盖**，"可独立测试"只是设计意图，未落地。

---

## 4. MediaPipe 可用性

**依赖声明 [实测]** —— `client/package.json:13`：

```json
"@mediapipe/tasks-vision": "^1.0.1",
```

**实际安装 [实测]**：`client/node_modules/@mediapipe/tasks-vision` 存在，`package.json` 的 `version` 为 `1.0.1`；`wasm/` 目录里有本地二进制：

```
vision_wasm_internal.wasm            11756954
vision_wasm_module_internal.wasm     11756972
vision_wasm_nosimd_internal.wasm     10960242
```

**但代码不使用本地 wasm** —— 它从公网 CDN 拉 wasm、从 Google 存储拉模型 [实测]：

```ts
// useFacialAnalysis.ts:111-127
const vision = await import('@mediapipe/tasks-vision');
const { FaceLandmarker, FilesetResolver } = vision;

const filesetResolver = await FilesetResolver.forVisionTasks(
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
);

const landmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
  baseOptions: {
    modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    delegate: 'GPU',
  },
  runningMode: 'VIDEO',
  numFaces: 1,
  outputFaceBlendshapes: true,
  outputFacialTransformationMatrixes: true,
});
```

`useBlinkDetection.ts:74-88` 是**同一份 CDN 地址的复制**（但该文件是死代码）。

**`client/public/` 下没有模型资产 [实测]**：整个目录只有一个文件 —— `client/public/worklets/pcm-capture-processor.js`（2654 B，语音用）。**没有任何 `.task`、`.tflite`、`wasm` 本地副本**。

→ **风险点 [实测+推断]**：
1. `@latest` **未锁版本**：CDN 上的 tasks-vision 升级后 wasm 与已安装的 npm 包（1.0.1）可能 ABI 不匹配，而本地明明有 1.0.1 的 wasm 却不用。
2. **首帧依赖外网**：离线/内网/答辩现场断网时，MediaPipe 初始化失败 → 静默降级到像素法。
3. 模型 `face_landmarker.task`（float16）每次页面会话都要从 `storage.googleapis.com` 下载一次（浏览器缓存可能命中，但无 Service Worker 预缓存 [未验证]）。

**不可用时的处理 [实测]**：有 try/catch，但只是 `console.warn` + 切到像素分析（见 §2.9）。**没有**：UI 提示、`metrics` 里的 `visionAvailable: false` 标记、`perceptionCapabilities.ts` 里的开关。用户在 Profile 页看到的是"面部卡不亮"，但不知道自己到底是"没开摄像头"还是"模型没加载起来"。

`useEyeTracking` 另有一层降级：`blinkScore === null` → `method='ear'`，用关键点 EAR 阈值 0.15（`useEyeTracking.ts:36`、`:62-66`）——这条降级**是**真实可达的（MediaPipe 加载成功但未输出 blendshape 时），并且 UI 会用橙色 `EAR 降级` 标签披露（`Profile.tsx:1438-1442`、`ConsultantRoom.tsx:1810`）。✓

---

## 5. 摄像头生命周期与资源占用

### 5.1 谁拥有摄像头？

**感知用摄像头：单一所有者 = `AnxietyContext` → `useFacialAnalysis`** [实测]。

启动的**两个**入口（都在 Context 内）：

```ts
// 入口 A：全局自定义事件（Profile 页"启动分析"按钮，Profile.tsx:821）
// AnxietyContext.tsx:540-547
try {
  facial.start();
  console.log('[多模态] 面部启动完成');
} catch (err) { ... }
setCameraEnabled(true);
```
```ts
// 入口 B：Chat 页 / 浮动组件的摄像头开关（Chat.tsx:597、AnxietyFloatingWidget.tsx:198）
// AnxietyContext.tsx:983-991
} else {
  console.log('[多模态] 开启摄像头/麦克风...');
  try {
    await Promise.all([voice.start(), facial.start()]);
    setCameraEnabled(true);
  } catch (err) { ... }
}
```

另有自动重启 [实测]：

```ts
// AnxietyContext.tsx:946-964
useEffect(() => {
  if (!cameraEnabled) return;
  if (facial.metrics.isDetecting) return;
  if (facial.metrics.frameCount === 0) return;
  const restartTimer = setTimeout(async () => {
    try { await facial.start(); ... } catch { ... }
  }, 2000);
  return () => clearTimeout(restartTimer);
}, [cameraEnabled, facial.metrics.isDetecting, facial.metrics.frameCount, facial.start]);
```

`useFacialAnalysis` 返回 `videoRef/streamRef`（`:544`），但 **`AnxietyContext` 只解构 `metrics/start/stop/reset`**（`:413` 使用 `facial.metrics/start/stop/reset`）→ 摄像头画面**没有任何页面渲染**。用户唯一的"摄像头开着"的反馈是 `AnxietyFloatingWidget.tsx:130-139` 的顶部 3px 光条和 `:155` 的小红点。**通话功能如果要复用这条流，必须先把它提升成一个带预览的可共享 media 层** —— 现在它是一路"看不见的流"。

### 5.2 双 `getUserMedia` / 流泄漏风险（**真问题**）

**风险 1：`toggleCamera` 无重入保护** [实测]

```ts
// AnxietyContext.tsx:967-993
const toggleCamera = useCallback(async () => {
  if (cameraEnabled) { ... } else {
    await Promise.all([voice.start(), facial.start()]);
    setCameraEnabled(true);
  }
}, [cameraEnabled, voice, facial, eyeTracking, rppg, breathingHook]);
```

`cameraEnabled` 只在 `await` **之后**才置 true。快速双击（或"摄像头按钮 + 浮动组件开关"同时点）会进入两次 else 分支 → **两次 `getUserMedia(video)`**（`useFacialAnalysis.ts:476`）和**两次 `getUserMedia(audio)`**（`useVoiceAnalysis.ts:282`）。`facial.streamRef.current = stream`（`:479`）/ `voice.streamRef.current = stream`（`:283`）都是**覆盖写**，第一次拿到的流无人引用 → `stop()` 只停得掉第二次的（`:524`），**第一条视频/音频轨道永久泄漏**（摄像头指示灯不灭）。`useFacialAnalysis.start()` 内部也没有 `if (streamRef.current) return` 之类的守卫 [实测]。

事件入口 A 同样无守卫（`AnxietyContext.tsx:541` 直接 `facial.start()`）。

**风险 2：麦克风有守卫、摄像头没有** [实测]

```ts
// AnxietyContext.tsx:997-1005
const startVoiceInput = useCallback(async () => {
  if (voice.metrics.isRecording) return;   // ← 有守卫
  try { await voice.start(); ... }
}, [voice]);
```
`useVoiceAnalysis.start()` 自身**没有**守卫 [实测]（`:280-283` 直接 `getUserMedia`）。

**风险 3：通话功能会引入第二个摄像头/麦克风所有者 —— 与感知栈冲突** [实测]

如果按现有 WebRTC 代码接通话：

```ts
// client/src/hooks/useWebRTC.ts:96-104
const startLocalStream = useCallback(async (video = true, audio = true) => {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: video ? { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' } : false,
    audio: audio ? { echoCancellation: true, noiseSuppression: true } : false,
  });
```
```ts
// client/src/hooks/useWebRTC.ts:193-200（收到 Offer 自动接听时再开一路）
if (!localStreamRef.current) {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
    audio: { echoCancellation: true, noiseSuppression: true },
  });
```

加上 `PatientRoom.tsx:82-91` 自己还有一路**独立 AudioContext + 麦克风**：

```ts
// PatientRoom.tsx:82-91
const startAudioCapture = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;
    const ctx = new AudioContext();
    audioContextRef.current = ctx;
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
```

**当前仓库里 `getUserMedia` 共有 5 个独立调用点** [实测]（grep `getUserMedia`）：
`useFacialAnalysis.ts:476`（video）、`useVoiceAnalysis.ts:282`（audio）、`useBlinkDetection.ts:207`（video，死代码）、`usePerception.ts:69`（audio，死代码）、`PatientRoom.tsx:84`（audio）、`useWebRTC.ts:98` + `:194`（video+audio）。

**结论 [推断]**：如果通话页按现有 `useWebRTC` 接法，而用户此前已经在 Chat 页开过"多模态感知"，会出现 **2 路摄像头 + 2~3 路麦克风 + 2~3 个 AudioContext 同时存活**；`useWebRTC.stopLocalStream` 只清自己那一路，感知那一路不受影响（反之亦然）。这是把视频特征折进通话功能时**第一个必须先解决的结构问题**：需要一个 `MediaStreamManager`（单例，引用计数，统一 `getUserMedia`/`stop`），让感知 hook 与 WebRTC 共享同一条轨道。

另外 `PatientRoom.tsx:139-148` 的 `stopAllCapture` 只停 `streamRef`（那一路麦克风）和 `audioContextRef`，**不停 WebRTC 的 `localStream`**；`useWebRTC.hangUp()`（`:140-154`）停的是自己的 `localStreamRef`。两套清理互不知情 → 只挂断通话但没点"结束采集"时，`PatientRoom` 的麦克风轨道仍在跑 [推断，需运行验证]。

### 5.3 帧率与单帧成本（用于估算 CPU）

**视频侧定时器 [实测]**：

```ts
// useFacialAnalysis.ts:510-513
// 选择分析函数：MediaPipe 用 100ms (10fps)，像素分析用 200ms (5fps)
const analyzeFn = useMediaPipeRef.current ? analyzeFrameMediaPipe : analyzeFramePixel;
const fps = useMediaPipeRef.current ? 100 : 200;
intervalRef.current = window.setInterval(analyzeFn, fps);
```
（变量名叫 `fps`，实际是**毫秒**。）

- 目标帧率：**10 fps**（MediaPipe 路径）/ **5 fps**（像素回退）。
- 死代码 `useBlinkDetection` 若被挂载，用 `requestAnimationFrame`（`useBlinkDetection.ts:250`）→ ~60fps + 另一次 `detectForVideo`。

**每帧工作（MediaPipe 路径）[实测]**：

1. `landmarker.detectForVideo(video, performance.now())`（`:148`）—— GPU delegate 的 FaceLandmarker 全模型推理（478 关键点 + 52 blendshape + 4x4 矩阵）。这是绝对主成本。
2. **每帧新建一个 canvas 并做一次 `getImageData`**（`:287-292`）：
   ```ts
   const tmpCanvas = document.createElement('canvas');
   tmpCanvas.width = 32; tmpCanvas.height = 32;
   const tmpCtx = tmpCanvas.getContext('2d', { willReadFrequently: true });
   tmpCtx.drawImage(video, 0, 0, 32, 32);
   const faceRegion = tmpCtx.getImageData(8, 4, 16, 12);
   ```
   每帧一次 DOM 创建 + `getContext` + `drawImage` + `getImageData`（192 px）。10fps 下不致命，但 `document.createElement` 每帧一次是无谓的分配（rPPG 已关闭，这段本可整段删掉）。
3. `setMetrics(prev => ({...prev, actionUnits:{...}, geometry:{...}, ...}))`（`:302-343`）→ 每帧一次 React 状态更新，并且**每次都新建 7 个嵌套对象**。
4. 通过 `onFrameSignal` 触发 `rppg.updateFromFrame`（已 early-return）+ `eyeTracking.updateFromFacialFrame`（`:383-384`）→ 后者又 `setMetrics` 一次（`useEyeTracking.ts:112`）→ **每帧两次 React 更新**。
5. `AnxietyContext` 里 12 个"同步 hook metrics 到本地 state"的 effect 用 `JSON.stringify(metrics)` 做依赖比较（`AnxietyContext.tsx:450-468`），例如：
   ```ts
   useEffect(() => { setFacialMetrics(facial.metrics); }, [JSON.stringify(facial.metrics)]);
   useEffect(() => { setEyeMetrics(eyeTracking.metrics); }, [JSON.stringify(eyeTracking.metrics)]);
   ```
   每帧的两次 setMetrics → 每次 render 都要把 12 个 metrics 对象序列化一遍做浅比较 [推断]。`facial.metrics` 含 `actionUnits`(18 字段) + `geometry`(4) + `emotionMapping`(7) 等，估算每帧 JSON.stringify 约 1–3 KB 字符串；10fps 下 ≈ 10–30 KB/s 的纯 GC 压力，外加 12 次 effect 调度。
6. 融合循环：`setInterval(..., 3000)`（`AnxietyContext.tsx:929-933`）→ 每 3 秒一次 11 模态加权 + 叙事文本生成（`generateNarrativeAnalysis`，几百行字符串拼接）。

**音频侧（呼吸/语音）[实测]**：`useVoiceAnalysis.analyzeFrame` 由 rAF 自调度（`:102`、`:277`），约 60fps；每帧分配 `new Float32Array(analyser.fftSize /* 2048 */)`（`:107`，8 KB/帧 ≈ **480 KB/s 分配**）并 `frameDataRef.current.push(dataArray)`（`:109`，**持续累积，未见清理** [未验证是否有上限]），再做 2048 次乘加的 RMS 循环，然后 `onAudioEnergy(rms)`（`:122`）。呼吸 hook 已 early-return，所以目前成本 = 一次空函数调用 ×60/s。**若打开 `VITE_ENABLE_BREATHING`，`useBreathing.updateFromAudioEnergy` 会在 60fps 下跑 600 元素数组的 map+slice+reduce 平滑（`:34-38`，约 O(600×6)）+ 每次 `slice(-600)` 复制 → 这是一个必须重算频率（应降到 10–20Hz）的点** [推断]。
另外每 60 帧（约 1s，`:136` `frameCountRef.current % 60 === 0`）做一次全数组统计（reduce ×N），`energies` 数组无长度上限 [未验证]。

**成本量级小结 [推断，未实测 CPU]**：稳态下 ≈ 10 次/秒 MediaPipe 推理（GPU）+ 10 次/秒 ×2 React 更新 + 60 次/秒 音频分析（8 KB 分配 + 2048 浮点 RMS）+ 每 3 秒一次融合。对笔记本而言，**MediaPipe 推理是唯一的重项**；如果通话功能要把面部提到 30fps，推理成本直接 ×3，而 10fps→30fps 主要收益只体现在"眨眼能测准"上（见 §2.2），所以更划算的做法是：**通话期只把眨眼所需的 blendshape 读取提到 30fps（或改由 WebRTC 轨道 + 更轻的 landmark-only 模型），其余面部指标维持 5–10fps**。

---

## 6. 未验证 / 不确定

1. **[未验证] MediaPipe 现场是否真的初始化成功。** 本次未运行浏览器。代码路径上它是 `await import()` + CDN wasm + Google 模型；断网/被墙/CDN 版本漂移都会失败。建议在答辩机上实测一次 `console` 是否出现 `[面部分析] ✅ MediaPipe Face Mesh 初始化成功`（`useFacialAnalysis.ts:131`）。若看到 `⚠️ MediaPipe 初始化失败，回退到像素分析`，则**整个眨眼/头姿模态都是死的**（§2.9）。
2. **[未验证] `facialTransformationMatrixes.data` 的行/列主序与 pitch 符号。** 代码用 `atan2(matrix[6], matrix[10])` 当 pitch、`pitch > 10` 判"低头"。按列主序（`data[c*4+r]`）解读为 `atan2(m21, m22)` 恰好是标准俯仰角提取，但这依赖具体序列化约定；**若符号相反，`downwardGazeRatio` 会把"抬头"统计成"低头"**。需要一次真实会话对照（抬头/低头各 10 秒看读数方向）才能定论。
3. **[未验证] 10fps 对眨眼率的实际欠采样比例。** §2.2 的"系统性偏低"是基于"眨眼 100–400ms、采样 100ms"的推算 [推断]，没有实测对照（例如同一用户同时用 30fps 参考录屏比对）。这直接决定"眨眼过少→解离"分支的误报率，**建议在集成通话前补一次对照实验**。
4. **[未验证] `AnxietyContext.tsx:381-385` 的 TDZ 正确性。** `onFrameSignalRef.current = (signal) => { rppg.updateFromFrame(...); eyeTracking.updateFromFacialFrame(...) }` 出现在 `const rppg`（`:417`）和 `const eyeTracking`（`:420`）**之前**。因为赋值的是闭包、执行在渲染之后，逻辑上不会 TDZ 崩溃；但这是"在声明前引用 const"的写法，靠的是调用时序而非语言保证。若未来有人把回调改成渲染期同步执行，会立刻 `ReferenceError`。标记为**结构脆弱**而非当前 bug [推断]。
5. **[未验证] `frameDataRef`（`useVoiceAnalysis.ts:109`）是否有上限。** 每帧 push 一个 `Float32Array(2048)`，本次只读到 push、未读到清理/上限逻辑；如果无上限，长时间通话会持续增长（与摄像头无关，但属同一通话成本账）。
6. **[未验证] 服务端 `/algorithm/perception/upload` 是否被其他入口（如咨询师端、管理端、定时任务）调用。** 本次 grep 范围是 `client/src` + `server/src`；仅确认**前端页面**没有调用它。若它被任何脚本/后台调用，§2.10 的"硬编码 `brightness:128` 造出 happy 模态"就是活跃缺陷。
7. **[未验证] `docs/` 下既有文档（`perception_capabilities.md`、`project_audit_v2.md`、`feature_inventory.md`）的结论是否与当前代码一致。** 抽查到的片段（"眨眼/头姿=真实采集"、"rPPG/呼吸/睡眠=无法采集"）与本次代码审计**一致**，但 `feature_inventory.md:83` 仍写着 `/perception/analyze` 支持"生理+认知+眼动"，与 §2.10 的"这些字段根本不在请求模型里"**不一致** —— 该文档已过期。
8. **[未验证] 未提交改动是否已通过 `tsc -b`。** `client/package.json:8` 的 `build` 是 `tsc -b && vite build`；工作区有 34 个文件被改动。`useEyeTracking.ts` 新增的 `blinkMethod`/`avgBlinkDuration`/`poseProxy` 字段已同步进 `multimodal.types.ts:344-353`，看起来自洽，但本次**未运行类型检查**。

---

## 附：把视频特征折进"实时语音通话"的建议取舍（基于以上证据）

| 保留 | 理由 |
| --- | --- |
| MediaPipe `eyeBlinkLeft/Right` → 眨眼（提到 ≥30fps） | 唯一"输入真、公式真"的摄像头指标；10fps 会让它变成"系统性误报解离" |
| `facialTransformationMatrixes` → headPitch/headYaw | 真信号；但**必须改名**并放弃"注视/回避"叙事 |
| MediaPipe blendshape → AU 原值（上屏，不下结论） | 模型输出可信；**不要**再用 `emotionMapping` 的 0.5/1.5/1.3 系数去驱动风险 |
| `blinkMethod` / `signalQuality` / `poseProxy` 这类来源标注 | 这是本栈最有价值的工程实践，通话功能应继承 |

| 丢弃 | 理由 |
| --- | --- |
| rPPG 心率 / HRV | 假公式 + 摄像头物理上测不了 HRV（`perceptionCapabilities.ts` 已关，保持关闭） |
| 呼吸频率（麦克风过零率） | 假公式 + 采样率量纲错（`/10` vs 实际 60fps）；改由 ASR 停顿/气口节奏 |
| `microExpressions.avgDuration = 200` / `confidence = 0.7 + n*0.005` | 硬编码常量冒充测量 |
| 像素回退面部分析 | 固定矩形 + 任意映射 + 不发帧信号（静默掐断眨眼链） |
| Python `eye/` `physiological/` `circadian/` `behavioral_activation/` `face/` 模块 | HTTP 请求模型里没有对应字段，前端也从不发 `facial_features` → 线上不可达 |
| `server/src/services/algorithm-bridge.ts:620-623` 的 `{brightness:128, contrast:50}` | 造出一个 risk=0 的"快乐"L1 模态；即使今天不可达，也应删除 |

| 集成前必须先做 | 理由 |
| --- | --- |
| 抽 `MediaStreamManager`（单例 + 引用计数），让感知 hook 与 WebRTC 共享轨道 | 现在有 5 个独立 `getUserMedia` 调用点；`toggleCamera` 无重入守卫会泄漏轨道（§5.2） |
| 把摄像头流提升为可渲染的共享对象 + 显式"感知采集进行中"UI | 当前视频流完全不可见，用户/评委无法判断画面是否真的在采（§5.1） |
| 明确"视觉能力不可用"的 UI 状态 | 现在 MediaPipe 失败只 `console.warn`，静默降级到像素法（§4） |
