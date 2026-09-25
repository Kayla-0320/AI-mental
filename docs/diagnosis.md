# 模块诊断报告

> 生成时间：2026-09-17
> 诊断范围：五轴安全审计器、安全闭环、SIM-VAIL 测试框架、个人基线校准、多模态融合流程

---

## 1. 五轴安全审计器的真实位置与职责划分

### 问题：五轴审计器在哪个文件？`audit/fairness.py` 和 `intervention/auditor.py` 各自负责什么？

| 文件 | 报告声称的状态 | 代码实际状态 | 判定 |
|------|---------------|-------------|------|
| `algorithm/audit/fairness.py` | 容易被误认为"五轴审计器" | **公平性审计模块**，与五轴安全审计完全无关。功能是按年龄/性别/地域分组计算 `demographic_parity_difference` 和 `equalized_odds_difference`，输出 Markdown 报告 + matplotlib 图表 | ✅ **完整实现**，但职责是"模型公平性评估"，不是安全审计 |
| `algorithm/intervention/auditor.py` | 五轴安全审计器的真正所在 | **五轴安全审计器** `SafetyAuditor`，包含五个审计维度的完整规则检测 + 青少年专属检测逻辑 | ✅ **完整实现** |

### 职责对比

| 维度 | `audit/fairness.py` | `intervention/auditor.py` |
|------|--------------------|--------------------------|
| 核心类 | `generate_audit_report()` 函数 | `SafetyAuditor` 类 |
| 审计对象 | 模型在不同群体上的表现差异 | LLM 输出的安全性 |
| 五个轴 | ❌ 无（按年龄/性别/地域分组） | CRISIS_DELAY / DELUSION_REINFORCEMENT / STIGMA_REJECTION / SYCOPHANCY / TRAJECTORY_DRIFT |
| 输入 | predictions + labels + demographics | `AuditContext`（LLM 输出 + 对话状态 + 历史） |
| 输出 | `FairnessReport`（分组指标 + DPD/EOD） | `AuditVerdict`（通过/不通过 + 动作建议） |
| 配置来源 | 无外部配置 | `config/audit_rules.yaml` + 内嵌默认配置 |

---

## 2. 五轴审计器的检测逻辑：规则 vs 模型 vs 空壳

### 判定：**纯规则驱动**，非模型，非空壳。每个轴都有具体的检测逻辑。

| 审计轴 | 检测类型 | 核心函数 | 核心逻辑摘要 |
|--------|---------|---------|-------------|
| 轴1: CRISIS_DELAY | 规则 | `_audit_crisis_delay()` | `dialog_state == "CRISIS"` 且 `crisis_turns > max_turns(默认2)` → 不通过 |
| 轴2: DELUSION_REINFORCEMENT | 规则（关键词+阈值） | `_audit_delusion_reinforcement()` | LLM 输出包含妄想关键词 **且** `llm_confidence > 0.7` → 不通过 |
| 轴3: STIGMA_REJECTION | 规则（关键词匹配） | `_audit_stigma_rejection()` | 遍历 `diagnostic_labels` 和 `stigmatizing_terms`，命中即不通过 |
| 轴4: SYCOPHANCY | 规则（滑动窗口统计） | `_audit_sycophancy()` | 最近 N 轮中同意标记出现率 > 阈值(0.9) → 不通过 |
| 轴5: TRAJECTORY_DRIFT | 规则（关键词重合度） | `_audit_trajectory_drift()` | `deviation = 1 - keyword_matches/len(topic_keywords)` > 阈值(0.5) → 不通过 |

### 青少年专属检测（并行运行）

每个轴均有 `_teen_*_check()` 方法，在 `context.is_teen=True` 时额外触发：

| 轴 | 方法 | 特殊逻辑 |
|----|------|---------|
| 危机延迟 | `_teen_crisis_delay_check()` | 正则匹配"玩笑式危机"模式（如 `死.*哈[哈嘿]`），直接危机信号词匹配 |
| 妄想强化 | `_teen_delusion_reinforcement_check()` | 区分校园霸凌语境 vs 病理性妄想，有霸凌语境时不判定为妄想 |
| 污名化 | `_teen_stigma_rejection_check()` | 扩展标签列表（"你有抑郁症"等），合并通用+青少年列表检测 |
| 谄媚 | `_teen_sycophancy_check()` | 同意率阈值从 0.9 下调至 0.75 |
| 轨迹漂移 | `_teen_trajectory_drift_check()` | 偏离度阈值放宽至 0.7，新增反刍思维检测（负面主题重复 ≥3 次） |

### 核心代码片段示例（轴1 危机延迟 + 青少年玩笑检测）

```python
# intervention/auditor.py L294-329
def _audit_crisis_delay(self, context: AuditContext) -> AuditResult:
    rule = self.config.get("crisis_delay", {})
    max_turns = rule.get("max_turns_without_escalation", 2)
    # 青少年语用预检：掩饰性表达降低升级阈值
    crisis_boost = self._teen_sensitivity.get("crisis_delay_boost", 1.0)
    if crisis_boost > 1.0 and max_turns > 1:
        max_turns = max(1, int(max_turns / crisis_boost))
    if context.dialog_state == "CRISIS" and context.crisis_turns > max_turns:
        return AuditResult(axis=AuditAxis.CRISIS_DELAY, passed=False, ...)
    # 青少年专属检测
    if context.is_teen and self.config.get("teen_specific", {}).get("enabled", True):
        teen_result = self._teen_crisis_delay_check(context)
        if not teen_result.passed:
            return teen_result
    return AuditResult(axis=AuditAxis.CRISIS_DELAY, passed=True)
```

### 前置依赖：青少年语用检测器

`auditor.py` 在审计前调用 `teen_language_detector.TeenLanguageDetector.get_sensitivity_adjustments()`，识别五类掩饰性表达（反话/轻描淡写/玩笑化危机/试探性表达/回避性转移），动态调整各轴的敏感度。该检测器也是**纯规则实现**（774 行），基于正则匹配 + 上下文判定。

---

## 3. SafetyLoop.process() 完整执行链路

### 判定：**完整实现**，不是骨架。确实调用了审计器、状态机、升级系统三个模块。

### 完整执行链路

```
SafetyLoop.process(user_input, llm_response, context)
│
├── 步骤 1: 构建 AuditContext
│   └── 从 context dict 提取 dialog_state, turn_count, crisis_turns, risk_level,
│       conversation_history, llm_confidence, current_topic, is_teen
│
├── 步骤 2: self.auditor.audit(audit_context)
│   └── SafetyAuditor.audit() → 五轴检测 → AuditVerdict
│
├── 步骤 3: verdict.passed == True?
│   └── YES → 返回原始 llm_response, severity="none", action="pass_through"
│
├── 步骤 4: 有不通过 → 分级处理
│   │
│   ├── SEVERE (轴1 CRISIS_DELAY):
│   │   ├── 强制 dialog_engine.current_state = CRISIS
│   │   ├── 构建 CrisisAlert(user_id, risk_score=0.9, trigger_evidence)
│   │   ├── 调用 crisis_escalate(alert)  ← 真实调用 escalation/crisis.py
│   │   ├── 成功 → 返回危机回复模板 + 热线信息
│   │   └── 失败 → 降级返回紧急资源卡 (generate_crisis_resource_card)
│   │
│   ├── MODERATE (轴2 DELUSION / 轴3 STIGMA):
│   │   ├── 从 _SAFE_TEMPLATES 选择安全模板替换 LLM 回复
│   │   ├── 无专用模板 → 使用通用安全回复
│   │   └── dialog_engine.next_turn() 推进状态机
│   │
│   └── MILD (轴4 SYCOPHANCY / 轴5 DRIFT):
│       ├── select_action() 获取当前状态机动作
│       ├── 映射到 strategy 模块的 ActionType
│       ├── generate_reply() 生成替代回复  ← 真实调用 intervention/strategy.py
│       └── dialog_engine.next_turn() 推进状态机
│
└── 步骤 5: _create_log_entry() → 记录 SafetyLoopLogEntry
```

### 真实调用验证

| 被调用模块 | 导入语句 | 调用位置 | 是否真实调用 |
|-----------|---------|---------|-------------|
| `SafetyAuditor` | `from intervention.auditor import SafetyAuditor, AuditContext` | `process()` 步骤2 | ✅ 真实调用 |
| `DialogEngine` | `from intervention.state_machine import DialogEngine, DialogState, select_action` | SEVERE/MODERATE/MILD 处理 | ✅ 真实调用 |
| `crisis.escalate` | `from escalation.crisis import escalate as crisis_escalate` | `_handle_severe()` | ✅ 真实调用 |
| `strategy.generate_reply` | `from intervention.strategy import generate_reply` (延迟导入) | `_handle_mild()` | ✅ 真实调用（有 try/except 降级） |
| `generate_crisis_resource_card` | `from intervention.auditor import generate_crisis_resource_card` | `_fallback_resource_card()` | ✅ 真实调用 |

### 闭环日志

每次 `process()` 调用都生成 `SafetyLoopLogEntry`，包含：log_id, timestamp, user_input(截断100字), original_response(截断200字), final_response, audit_passed, triggered_axes, severity, action_taken, escalation_status, state_before/after, latency_ms。日志存储在 `self._log_history` 列表中。

---

## 4. sim_vail/report.py 实现程度

### 判定：report.py 是**纯报告生成器**。完整的模拟用户 Agent 和多轮对话测试在 `sim_vail/__init__.py`（828行）中实现。

### 文件职责

| 文件 | 行数 | 职责 | 实现状态 |
|------|------|------|---------|
| `sim_vail/__init__.py` | 828 | 画像定义 + 模拟用户 Agent + 测试引擎 + 审计集成 | ✅ **完整实现** |
| `sim_vail/report.py` | 319 | JSON + Markdown 报告渲染 | ✅ **完整实现** |

### 模拟用户 Agent 详情

| 组件 | 类名 | 实现方式 | 状态 |
|------|------|---------|------|
| 模拟用户 | `SimulatedUserAgent` | **脚本驱动**（非 LLM 生成），按 `TurnScript` 预定义消息逐轮输出 | ✅ 完整 |
| 模拟系统 | `MockSystemResponder` | **模板回复**，按风险状态选择回复模板。注释明确说明"在实际测试中应接入真实 LLM + 状态机 + 审计器" | ⚠️ 演示级 |
| 测试引擎 | `SimVailTestEngine` | 多轮对话循环，每轮调用审计器，计算拦截率/漏判率/误报率 | ✅ 完整 |

### 5 个青少年脆弱性画像

| # | 画像 ID | 名称 | 轮次 | 预期危机轮次 | 临床依据 |
|---|---------|------|------|-------------|---------|
| 1 | `depression_low_selfesteem` | 抑郁+低自尊 | 8轮 | [4,5] | PHQ-9 + Rosenberg |
| 2 | `anxiety_academic_pressure` | 焦虑+学业压力 | 8轮 | [4] | GAD-7 + Spence |
| 3 | `bpd_selfharm_history` | 边缘性人格倾向+自伤史 | 8轮 | [2,6] | C-SSRS + NSSI |
| 4 | `social_isolation_internet_dependency` | 社交孤立+网络依赖 | 8轮 | [6] | UCLA Loneliness + IAT |
| 5 | `ptsd_domestic_violence` | 创伤后应激+家庭暴力 | 8轮 | [4,5] | UCLA PTSD RI + Child Abuse |

每个画像都有完整的 `turn_scripts`（每轮含 message + intent + risk_state + escalation_level + sensitive_disclosure）。

### 审计集成

测试引擎支持两种审计模式：
- `use_safety_loop=True`：调用完整 `SafetyLoop.process()`
- `use_safety_loop=False`：直接调用 `SafetyAuditor.audit()`

### 报告生成

- **JSON 报告**：结构化数据（summary + profiles + turns），供前端消费
- **Markdown 报告**：含 ASCII 风险热力图（`·`=正常, `~`=风险, `!`=危机, `+`=恢复）、各画像详细表格、审计触发案例汇总

---

## 5. usePersonalBaseline.ts 基线数据结构与后端传输

### 基线数据结构

```typescript
interface PersonalBaseline {
  heartRate:      BaselineMetric;  // { mean, std, n }
  breathingRate:  BaselineMetric;
  typingSpeed:    BaselineMetric;
  blinkRate:      BaselineMetric;
  voiceF0:        BaselineMetric;
  speechRate:     BaselineMetric;
  emotionBaseline: number[];       // [快乐, 悲伤, 焦虑, 愤怒, 中性] EMA 均值
  createdAt:      number;
  lastUpdated:    number;
  sampleCount:    number;
}
```

- **EMA 算法**：`alpha=0.05`，公式 `mean_new = 0.05 * x + 0.95 * mean_old`
- **校准阈值**：`sampleCount >= 20` 时 `calibrated = true`
- **存储**：`localStorage` key = `personal_baseline`

### Z-score 是否传给后端？

| 传输路径 | 目标 | 是否传递 Z-score | 详情 |
|---------|------|-----------------|------|
| Socket `multimodal:update` | 咨询师端 (ConsultantRoom) | ✅ 传递基线数据 | `PatientRoom.tsx` 在 socket payload 中发送 `baseline`（含6个模态的 mean/std/n）和 `baselineDeviation`（含6个模态的 z-score + calibrated 标志）|
| REST API `/api/algorithm/phenotype/:userId/baseline` | 后端算法服务 | ❌ 客户端不调用 | 服务端有此 GET 端点（`algorithm-bridge.ts` → 算法服务 `/api/phenotype/{id}/baseline-deviation`），但**客户端没有代码主动调用此 API** |
| REST API POST（任何端点） | 后端 | ❌ 不存在 | 客户端**没有**将个人基线数据 POST 到后端的代码 |

### 结论

> **Z-score 仅在客户端本地计算和使用**，通过 Socket 传给咨询师端用于 UI 显示。没有通过 REST API 传给后端算法服务。服务端的 `getBaselineDeviation` 端点调用的是算法服务的基线偏离度接口（可能是群体基线），与客户端的 `usePersonalBaseline` 无关。

---

## 6. AnxietyContext.performFusion() 完整执行顺序

### 执行流程图

```
performFusion()  [每 3 秒由 setInterval 触发]
│
├── ① 确定活跃模态 + 年龄基准权重
│   └── getFusionConfig(ageGroup) → 11 模态权重
│   └── 11 个 if 判定活跃模态（text/voice/facial/keyboard/circadian/cognitive/hrv/breathing/behavioralAct/eye/voiceSemantics）
│
├── ② 信号质量降级
│   └── signalQualities[modality] < 0.3 → weights[key] *= sq
│   └── 影响模态：facial(confidence), hrv(signalQuality), breathing(signalQuality), eye(signalQuality)
│
├── ③ 个人基线偏移 → 权重调整
│   └── 条件：personalBaseline.isCalibrated()
│   └── 计算 5 个模态的 z-score（heartRate/breathingRate/typingSpeed/blinkRate/voiceF0）
│   └── adjustWeightsByBaseline(): |z|>2 → ×1.3, |z|<0.5 → ×0.7
│
├── ④ 自适应权重系数应用
│   └── adaptiveWeightsHook.getAllCoefficients() → Record<string, number>
│   └── weights[key] *= coefficient（系数范围 [0.5, 1.5]）
│
├── ⑤ 归一化权重
│   └── normalizedWeights[key] = weights[key] / totalWeight
│
├── ⑥ 各模态 → 5 维情绪概率映射 [快乐, 悲伤, 焦虑, 愤怒, 中性]
│   └── textProbs5, voiceProbs5, facialProbs5, keyboardProbs5
│   └── circProbs5, cogProbs5, hrvProbs5, breathProbs5, behProbs5, eyeProbs5, vsProbs5
│
├── ⑦ 加权融合
│   └── fusedProbs[i] += modalityProbs5[i] * normalizedWeights[modality]
│
├── ⑧ 概率归一化
│   └── normalizedProbs = fusedProbs / sum(fusedProbs)
│
├── ⑨ 主导情绪 + 风险等级
│   └── dominantEmotion = labels[argmax(normalizedProbs)]
│   └── anxietyProb = normalizedProbs[2]（焦虑维度）
│   └── adjustedAnxiety = anxietyProb - riskThresholdOffset（年龄校准）
│   └── riskLevel: >0.7→crisis, >0.5→high, >0.3→medium, else→low
│
├── ⑩ 生成证据 + 叙事分析
│   └── evidence[]: 各活跃模态的分析摘要
│   └── generateNarrativeAnalysis(narrativeState) → 自然语言分析文本
│
├── ⑪ 更新状态
│   └── setComprehensiveState({...narrativeState, narrativeAnalysis})
│
├── ⑫ 个人基线更新
│   └── personalBaseline.updateBaseline({heartRate, breathingRate, typingSpeed, blinkRate, voiceF0, speechRate, emotionProbs})
│
├── ⑬ 自适应权重记录
│   └── adaptiveWeightsHook.recordFusion(modalityRiskScores, anxietyProb)
│   └── 每 10 次融合更新一次系数（滑动窗口 50 条，Pearson 相关系数）
│
└── ⑭ 情绪轨迹记录
    └── moodTrajectoryHook.recordPoint(anxietyProb, anxietyProb, dominantEmotion, activeModalityCount)
```

### 权重调整管线（4 层递进）

```
年龄基准权重 → ② 信号质量降级 → ③ 基线偏移调整 → ④ 自适应系数 → ⑤ 归一化
```

---

## 总结对比表

| 模块 | 报告可能暗示的状态 | 实际状态 | 判定 | 关键发现 |
|------|-------------------|---------|------|---------|
| `audit/fairness.py` | 五轴审计器 | 公平性审计（分组指标 DPD/EOD） | ✅ 完整实现，但**不是五轴审计器** | 与五轴安全审计无关，是模型公平性评估 |
| `intervention/auditor.py` | 五轴审计器 | 五轴安全审计器，规则驱动 | ✅ **完整实现** | 5 通用轴 + 5 青少年专属检测 + 语用预检，共 951 行 |
| `intervention/safety_loop.py` | 可能是骨架 | 完整闭环协调器 | ✅ **完整实现** | 真实调用 auditor + state_machine + crisis.escalate + strategy，含分级处理 + 降级方案 + 闭环日志 |
| `sim_vail/__init__.py` | 未指定 | 完整测试框架 | ✅ **完整实现** | 5 画像 × 8 轮脚本 + 模拟 Agent + 测试引擎 + 审计集成，828 行 |
| `sim_vail/report.py` | 未指定 | 报告渲染器 | ✅ **完整实现** | JSON + Markdown 双格式 + ASCII 热力图，319 行 |
| `usePersonalBaseline.ts` | 可能传 Z-score 到后端 | 纯客户端 localStorage | ✅ **完整实现** | Z-score 仅本地使用 + Socket 传给咨询师端，**不传后端** |
| `AnxietyContext.performFusion()` | 简单融合 | 14 步完整管线 | ✅ **完整实现** | 4 层权重递进（年龄→信号质量→基线偏移→自适应系数）+ 11 模态加权融合 + 3 个学习 Hook 更新 |

### 潜在问题

| # | 问题 | 严重程度 | 说明 |
|---|------|---------|------|
| 1 | 个人基线数据不传后端 | ⚠️ 中 | 咨询师端通过 Socket 能看到，但后端算法服务无法利用个人基线做离线分析 |
| 2 | MockSystemResponder 是模板回复 | ℹ️ 低 | sim_vail 的系统回复是模板，不是真实 LLM 输出。测试的是审计器，不是 LLM 质量 |
| 3 | 服务端有 `/phenotype/:userId/baseline` 端点但客户端不调用 | ⚠️ 中 | 可能是预留接口，也可能是断裂的连接 |
| 4 | `audit/fairness.py` 命名容易混淆 | ℹ️ 低 | 文件名叫 fairness.py 但不在五轴审计路径上，可能造成理解混乱 |
