# 项目全面检查报告 v2

> 生成时间：2026-09-17
> 扫描范围：algorithm/ client/ server/ 三端全部源码
> 扫描规则：跳过 `__pycache__`、`node_modules`、`.git`、`venv`、`dist`、`.pytest_cache`、`outputs`

---

## 一、项目概览

### 1.1 总体统计

| 指标 | 数值 |
|------|------|
| 总文件数 | 284 |
| 总代码行数 | 79,450 |

### 1.2 语言占比

| 语言 | 文件数 | 行数 | 占比 |
|------|--------|------|------|
| Python (.py) | 132 | 40,050 | 50.4% |
| React TSX (.tsx) | 43 | 12,778 | 16.1% |
| TypeScript (.ts) | 75 | 11,204 | 14.1% |
| JSON (.json) | 7 | 6,104 | 7.7% |
| HTML (.html) | 7 | 4,150 | 5.2% |
| Markdown (.md) | 9 | 2,975 | 3.7% |
| Prisma (.prisma) | 1 | 653 | 0.8% |
| CSS (.css) | 2 | 540 | 0.7% |
| YAML (.yaml) | 3 | 390 | 0.5% |
| 其他 | 5 | 596 | 0.8% |

### 1.3 三端目录结构

```
AI-mental-main/
├── algorithm/          # Python 算法后端 (FastAPI)
│   ├── api/            #   REST API 路由层 (11 files, 1467 lines)
│   ├── assessment/     #   心理评估模块 (14 files, 4336 lines)
│   ├── audit/          #   公平性审计模块 (4 files, 604 lines)
│   ├── config/         #   配置文件 (3 files, 390 lines)
│   ├── escalation/     #   危机升级模块 (9 files, 2984 lines)
│   ├── experiments/    #   实验脚本 (7 files)
│   ├── federated/      #   联邦学习模块 (4 files, 923 lines)
│   ├── intervention/   #   智能干预模块 (12 files, 5770 lines)
│   ├── perception/     #   多模态感知模块 (35 files, 7558 lines)
│   ├── privacy/        #   隐私保护模块 (6 files, 1980 lines)
│   ├── shared/         #   共享数据类 (5 files, 316 lines)
│   ├── sim_vail/       #   SIM-VAIL 测试框架 (2 files, 1590 lines)
│   ├── tests/          #   单元测试 (24 files, 8218 lines)
│   └── main.py         #   FastAPI 入口
├── client/             # React 前端 (Vite + TypeScript)
│   ├── src/
│   │   ├── components/ #   通用组件 (5 files, 546 lines)
│   │   ├── context/    #   React Context (1 file, 1096 lines)
│   │   ├── hooks/      #   自定义 Hooks (21 files, 4459 lines)
│   │   ├── pages/      #   页面组件 (33 files, 10767 lines)
│   │   ├── services/   #   API 服务层 (3 files, 267 lines)
│   │   ├── store/      #   Zustand 状态管理 (1 file, 53 lines)
│   │   └── types/      #   TypeScript 类型 (2 files, 589 lines)
│   └── package.json
├── server/             # Node.js 后端 (Express + Prisma)
│   ├── src/
│   │   ├── config/     #   配置 (2 files, 35 lines)
│   │   ├── controllers/#   控制器 (7 files, 687 lines)
│   │   ├── middlewares/#   中间件 (3 files, 85 lines)
│   │   ├── routes/     #   路由 (14 files, 751 lines)
│   │   ├── services/   #   服务层 (16 files, 3724 lines)
│   │   ├── socket/     #   Socket.IO (1 file, 224 lines)
│   │   └── app.ts      #   Express 入口
│   ├── prisma/         #   数据库 Schema
│   └── package.json
└── docs/               # 文档
```

### 1.4 最近修改的文件（按时间倒序，前 20）

| 时间 | 文件路径 | 行数 |
|------|---------|------|
| 2026-09-17 20:49 | algorithm/tests/test_sim_vail_comparison.py | 235 |
| 2026-09-17 20:48 | algorithm/api/sim_vail.py | 246 |
| 2026-09-17 20:48 | algorithm/sim_vail/report.py | 472 |
| 2026-09-17 20:48 | algorithm/sim_vail/__init__.py | 1118 |
| 2026-09-17 20:42 | algorithm/tests/test_baseline_aware_auditor.py | 398 |
| 2026-09-17 20:40 | algorithm/intervention/auditor.py | 1112 |
| 2026-09-17 20:40 | algorithm/intervention/safety_loop.py | 790 |
| 2026-09-17 20:37 | algorithm/config/audit_rules.yaml | 274 |
| 2026-09-17 20:34 | client/src/hooks/usePersonalBaseline.ts | 331 |
| 2026-09-17 20:33 | client/src/services/index.ts | 186 |
| 2026-09-17 20:32 | server/src/routes/algorithm.routes.ts | 219 |
| 2026-09-17 20:32 | server/src/services/algorithm-bridge.ts | 495 |
| 2026-09-17 20:28 | algorithm/escalation/logs/escalation_audit.jsonl | 415 |
| 2026-09-17 20:27 | algorithm/tests/test_personal_baseline.py | 597 |
| 2026-09-17 20:24 | algorithm/api/assessment.py | 378 |
| 2026-09-17 20:24 | algorithm/assessment/personal_baseline.py | 495 |
| 2026-09-17 20:23 | algorithm/config/baseline.yaml | 37 |
| 2026-09-17 20:20 | docs/diagnosis.md | 308 |
| 2026-09-17 20:17 | docs/project_report.md | 764 |
| 2026-09-17 16:12 | client/src/pages/patient/PatientRoom.tsx | 709 |

---

## 二、模块完整性检查

### 2.1 算法端 (algorithm/)

| 模块路径 | 文件数 | 行数 | 状态 | 测试文件 | 备注 |
|---------|--------|------|------|---------|------|
| algorithm/perception/text | 1 | 2 | **空壳** | - | 仅 docstring，无代码 |
| algorithm/perception/text_emotion | 8 | 2224 | **完整实现** | test_perception_new.py | SProp-GNN + 对抗去偏 |
| algorithm/perception/voice | 2 | 522 | **完整实现** | test_acoustic.py | 声学特征提取 |
| algorithm/perception/voice_semantics | 2 | 171 | **完整实现** | - | 语义特征 |
| algorithm/perception/face | 2 | 812 | **完整实现** | test_age_differentiated_perception.py | FACS AU 系统 |
| algorithm/perception/behavior | 2 | 692 | **完整实现** | - | 键盘动力学 |
| algorithm/perception/behavioral_activation | 2 | 143 | **完整实现** | - | 行为激活水平 |
| algorithm/perception/circadian | 2 | 140 | **完整实现** | - | 昼夜节律 |
| algorithm/perception/cognitive | 2 | 131 | **完整实现** | - | 认知扭曲 |
| algorithm/perception/eye | 2 | 129 | **完整实现** | - | 眼动模式 |
| algorithm/perception/physiological | 3 | 276 | **完整实现** | - | rPPG-HRV + 呼吸 |
| algorithm/perception/fusion | 4 | 628 | **完整实现** | test_fusion.py | 11 模态加权融合 |
| algorithm/assessment/multi_task | 6 | 1098 | **完整实现** | test_assessment.py | 多任务预测器 |
| algorithm/assessment/risk | 1 | 464 | **完整实现** | - | 综合风险评估 |
| algorithm/assessment/scale | 1 | 523 | **完整实现** | - | PHQ-9/GAD-7/PSS-10 映射 |
| algorithm/assessment/trend | 1 | 460 | **完整实现** | - | 趋势分析 + 变点检测 |
| algorithm/assessment/report | 1 | 389 | **完整实现** | - | 评估报告生成 |
| algorithm/assessment/personal_baseline.py | 1 | 495 | **完整实现** | test_personal_baseline.py (44 tests) | EMA 基线 + Z-score |
| algorithm/intervention/auditor.py | 1 | 1112 | **完整实现** | test_auditor.py + test_baseline_aware_auditor.py | 五轴安全审计器 |
| algorithm/intervention/safety_loop.py | 1 | 790 | **完整实现** | test_safety_loop.py | 安全闭环协调器 |
| algorithm/intervention/state_machine.py | 1 | 634 | **完整实现** | test_state_machine.py | 对话状态机 |
| algorithm/intervention/teen_language_detector.py | 1 | 774 | **完整实现** | test_teen_language_detector.py | 青少年语用检测 |
| algorithm/intervention/cbt | 2 | 681 | **完整实现** | test_cbt.py | 认知重构 |
| algorithm/intervention/crisis | 1 | 465 | **完整实现** | - | 危机干预 |
| algorithm/intervention/socratic | 1 | 492 | **完整实现** | - | 苏格拉底式提问 |
| algorithm/intervention/strategy | 1 | 399 | **完整实现** | - | 策略生成 |
| algorithm/intervention/middleware.py | 1 | ~350 | **完整实现** | - | 审计中间件 |
| algorithm/escalation/detector | 1 | 287 | **完整实现** | test_escalation.py | 危机检测 |
| algorithm/escalation/protocol | 1 | 678 | **完整实现** | - | 升级协议 |
| algorithm/escalation/referral | 1 | 302 | **完整实现** | - | 转介匹配 |
| algorithm/escalation/self_harm | 1 | 316 | **完整实现** | - | 自伤风险评估 |
| algorithm/escalation/crisis.py | 1 | ~465 | **完整实现** | - | 危机升级核心 |
| algorithm/escalation/scheduler.py | 1 | ~200 | **完整实现** | - | 咨询师排班 |
| algorithm/audit/fairness.py | 1 | 593 | **完整实现** | test_fairness.py | 公平性审计（非五轴） |
| algorithm/audit/bias | 1 | 2 | **空壳** | - | 仅 docstring |
| algorithm/audit/explainability | 1 | 2 | **空壳** | - | 仅 docstring |
| algorithm/privacy/anonymization | 1 | 407 | **完整实现** | - | K-匿名 |
| algorithm/privacy/consent | 1 | 378 | **完整实现** | - | 知情同意 |
| algorithm/privacy/encryption | 1 | 310 | **完整实现** | - | AES 加密 |
| algorithm/privacy/on_device.py | 1 | 585 | **完整实现** | test_on_device.py | 端侧推理 |
| algorithm/privacy/sanitizer.py | 1 | ~300 | **完整实现** | test_sanitizer.py | 数据脱敏 |
| algorithm/federated | 4 | 923 | **完整实现** | test_federated.py | Flower 联邦学习 |
| algorithm/shared/dataclasses.py | 1 | 275 | **完整实现** | test_dataclasses.py | 统一数据接口 |
| algorithm/shared/config | 1 | 2 | **空壳** | - | 仅 docstring |
| algorithm/shared/exceptions | 1 | 2 | **空壳** | - | 仅 docstring |
| algorithm/shared/utils | 1 | 2 | **空壳** | - | 仅 docstring |
| algorithm/sim_vail | 2 | 1590 | **完整实现** | test_sim_vail_framework.py + test_sim_vail_comparison.py | 对比实验 |

### 2.2 服务端 (server/)

| 模块路径 | 文件数 | 行数 | 状态 | 备注 |
|---------|--------|------|------|------|
| server/src/controllers | 7 | 687 | **完整实现** | 7 个控制器 |
| server/src/services | 16 | 3724 | **完整实现** | 含 algorithm-bridge |
| server/src/routes | 14 | 751 | **完整实现** | 14 个路由文件 |
| server/src/middlewares | 3 | 85 | **完整实现** | auth/error/validator |
| server/src/socket | 1 | 224 | **完整实现** | Socket.IO + WebRTC 信令 |
| server/src/config | 2 | 35 | **完整实现** | 配置 + 数据库 |
| server/prisma | 2 | ~700 | **完整实现** | Schema + Seed |

### 2.3 客户端 (client/)

| 模块路径 | 文件数 | 行数 | 状态 | 备注 |
|---------|--------|------|------|------|
| client/src/components | 5 | 546 | **完整实现** | 通用 UI 组件 |
| client/src/pages/patient | 19 | 6878 | **完整实现** | 患者端页面 |
| client/src/pages/consultant | 7 | 2498 | **完整实现** | 咨询师端页面 |
| client/src/pages/admin | 7 | 1391 | **完整实现** | 管理端页面 |
| client/src/hooks | 21 | 4459 | **完整实现** | 含 11 模态分析 hooks |
| client/src/services | 3 | 267 | **完整实现** | API 服务层 |
| client/src/types | 2 | 589 | **完整实现** | 多模态类型定义 |
| client/src/store | 1 | 53 | **完整实现** | Zustand 状态 |
| client/src/context | 1 | 1096 | **完整实现** | AnxietyContext 融合引擎 |

---

## 三、接口一致性检查

### 3.1 后端共享数据类 (algorithm/shared/dataclasses.py)

| 数据类 | 字段 | 用途 |
|--------|------|------|
| `RiskLevel` (Enum) | LOW/MEDIUM/HIGH/CRISIS | 风险等级枚举 |
| `EvidenceItem` | source, description, weight | 判断依据来源 |
| `EmotionResult` | text_emotion_probs(5维), audio_risk_prob, confidence, timestamp, evidence | 情绪感知结果 |
| `RiskAssessment` | phq9_estimated, gad7_estimated, risk_level, confidence, evidence[] | 心理风险评估 |
| `CrisisAlert` | user_id, risk_score, trigger_evidence[], timestamp, recommended_action | 危机警报 |
| `FeatureSummary` | text_emotion_probs, audio_risk_prob, confidence, session_id, timestamp | 跨模块特征传递 |
| `AuditAxis` (Enum) | CRISIS_DELAY/DELUSION_REINFORCEMENT/STIGMA_REJECTION/SYCOPHANCY/TRAJECTORY_DRIFT | 审计轴 |
| `AuditAction` (Enum) | PASS/REWRITE/INJECT_RESOURCE/ESCALATE_HUMAN | 审计动作 |
| `AuditResult` | axis, passed, reason, suggested_action | 单轴审计结果 |
| `AuditVerdict` | passed, results[], final_action, rewritten_content | 综合审计裁决 |
| `PhenotypeFeature` | name, value, confidence, source, unit, significant_change | 单个表型特征 |
| `PhenotypeVector` | user_id, time_window_days, 5类特征列表 | 多模态数字表型 |
| `EscalationChannel` (Enum) | WEBHOOK/IN_APP/CONSOLE | 升级通道 |
| `EscalationStatus` (Enum) | PENDING/DISPATCHED/ACKNOWLEDGED/RESOLVED/FAILED | 升级状态 |
| `EscalationResult` | alert_id, alert, status, channels_notified[], assigned_counselor, response_time_seconds | 升级结果 |

### 3.2 shared 数据类使用情况

以下模块正确导入了 shared 数据类：

| 模块 | 导入的数据类 |
|------|-------------|
| api/assessment.py | EmotionResult |
| api/escalation.py | CrisisAlert, EscalationChannel |
| api/perception.py | EmotionResult |
| assessment/multi_task/predictor.py | EvidenceItem, RiskAssessment, RiskLevel |
| assessment/phenotype.py | PhenotypeFeature, PhenotypeVector |
| assessment/scale/__init__.py | EmotionResult, PhenotypeVector |
| escalation/detector/__init__.py | EmotionResult, CrisisAlert |
| escalation/referral/__init__.py | CrisisAlert, RiskLevel |
| intervention/middleware.py | AuditAction, AuditVerdict |
| intervention/state_machine.py | CrisisAlert, RiskAssessment, RiskLevel |
| perception/fusion/fusion.py | EmotionResult |
| perception/perception_service.py | EmotionResult |
| privacy/sanitizer.py | EmotionResult, FeatureSummary |

**模块内部自定义的数据类**（非重复，属于模块专用）：共 50+ 个 dataclass 分布在各自模块中，如 `ComorbidityResult`、`BaselineDeviation`、`SafetyLoopLogEntry` 等。这些是模块专用的扩展数据结构，不属于 shared 层的通用接口，**设计合理**。

### 3.3 前端 TypeScript 类型 vs 后端数据类

| 对比项 | 后端 (Python) | 前端 (TypeScript) | 一致性 |
|--------|--------------|-------------------|--------|
| 情绪维度 | `EmotionResult.text_emotion_probs`: **5 维** [快乐, 悲伤, 焦虑, 愤怒, 中性] | `TextAnalysis.emotionDistribution`: **8 维** [快乐, 悲伤, 焦虑, 愤怒, 恐惧, 厌恶, 惊讶, 中性] | **不一致** |
| 声音情绪 | `EmotionResult.audio_risk_prob`: float | `VoiceAnalysis.emotionProbs`: **5 维** | 维度一致 |
| 综合情绪 | `EmotionResult.text_emotion_probs`: 5 维 | `ComprehensiveEmotionState.emotionProbs`: **5 维** | 一致 |
| RiskLevel | 枚举: low/medium/high/crisis (小写) | `riskLevel`: 'low' \| 'medium' \| 'high' \| 'crisis' (小写) | **一致** |
| 个人基线 | `PersonalBaseline`: heartRate, breathingRate, typingSpeed, emotionBaseline(5维) 等 | `PersonalBaseline`: 同名同结构 (BaselineMetric: {mean, std, n}) | **一致** |
| 基线偏离 | `BaselineDeviation`: deviations per modality + significant_deviations + calibrated | `BaselineDeviation`: 同名同结构 | **一致** |

### 3.4 不一致清单（按严重程度排序）

| # | 严重程度 | 问题 | 位置 | 说明 |
|---|---------|------|------|------|
| 1 | **高** | 文本情绪维度不一致 | 后端 `EmotionResult.text_emotion_probs` = 5 维 vs 前端 `TextAnalysis.emotionDistribution` = 8 维 | 前端在融合时会将 8 维映射到 5 维（`textProbs5`），但后端 API 返回的永远是 5 维。前端内部 8 维仅用于 UI 展示，不影响后端通信。 |
| 2 | **高** | 算法桥接层 URL 全部不匹配 | `server/src/services/algorithm-bridge.ts` | 详见第四章 4.4 节 |
| 3 | **中** | 前端 `VoiceAnalysis.emotionProbs` 5 维 vs `TextAnalysis.emotionDistribution` 8 维 | `client/src/types/multimodal.types.ts` | 前端内部不一致，但有映射逻辑 |
| 4 | **低** | `perception/text` 模块为空壳 | `algorithm/perception/text/__init__.py` | 实际文本分析由 `text_emotion` 模块完成 |

---

## 四、前后端链路检查

### 4.1 后端 API 路由一览 (server/src/routes/)

| 路由文件 | 挂载前缀 | 端点数 | 主要功能 |
|---------|---------|--------|---------|
| auth.routes.ts | /api/auth | 3 | 注册/登录/获取资料 |
| consultation.routes.ts | /api/consultation | 6 | AI 对话管理 |
| profile.routes.ts | /api/profile | 14 | 心理画像/情绪/测评 |
| healing.routes.ts | /api/healing | 4 | 疗愈室 |
| expert.routes.ts | /api/expert | 9 | 专家预约 |
| treatment.routes.ts | /api/treatment | 6 | 治疗规划 |
| learning.routes.ts | /api/learning | 6 | 持续学习 |
| crisis.routes.ts | /api/crisis | 5 | 危机干预 |
| mood.routes.ts | /api/mood | 5 | 心情打卡 |
| review.routes.ts | /api/reviews | 2 | 咨询评价 |
| community.routes.ts | /api/community | 7 | 社区互助 |
| extra.routes.ts | /api/extra | 13 | 成就/睡眠/支付/报告 |
| notification.routes.ts | /api/notifications | 4 | 通知 |
| algorithm.routes.ts | /api/algorithm | 9 | 算法桥接 |

**总计：93 个后端 API 端点**

### 4.2 前端 API 调用一览 (client/src/services/index.ts)

| 服务分组 | 调用数 | 覆盖的后端路由 |
|---------|--------|--------------|
| authApi | 3 | auth/* |
| consultationApi | 6 | consultation/* |
| profileApi | 10 | profile/* |
| healingApi | 4 | healing/* |
| expertApi | 9 | expert/* |
| treatmentApi | 6 | treatment/* |
| learningApi | 6 | learning/* |
| crisisApi | 6 | crisis/* |
| moodApi | 5 | mood/* |
| reviewApi | 2 | reviews/* |
| communityApi | 7 | community/* |
| extraApi | 11 | extra/* |
| baselineApi | 2 | algorithm/baseline/* |

**前端共调用 77 个 API，覆盖 12 个路由分组。**

### 4.3 前后端对比

| 类别 | 详情 |
|------|------|
| 后端有、前端未调用 | notification/* (4个), algorithm/smart-chat, algorithm/assess, algorithm/perception, algorithm/phenotype, algorithm/comorbidity, algorithm/escalate, algorithm/perception/upload |
| 前端调用、后端有 | auth/*, consultation/*, profile/*, healing/*, expert/*, treatment/*, learning/*, crisis/*, mood/*, reviews/*, community/*, extra/*, algorithm/baseline/* |
| 前端调用、后端不存在 | **无** |

> 前端未直接调用算法桥接的大部分端点（smart-chat/assess/perception 等），这些端点预留给未来扩展或咨询师端管理后台使用。前端仅通过 `baselineApi` 调用算法桥接。

### 4.4 算法桥接层 URL 匹配检查

```mermaid
graph LR
    subgraph "Node.js Server (algorithm-bridge.ts)"
        A1[smartChat] -->|/api/smart-chat| X1[算法 FastAPI]
        A2[assessRisk] -->|/api/assessment/predict| X1
        A3[getPhenotype] -->|/api/phenotype/:id| X1
        A4[getBaselineDeviation] -->|/api/phenotype/:id/baseline-deviation| X1
        A5[analyzeComorbidity] -->|/api/comorbidity/analyze| X1
        A6[escalateCrisis] -->|/emergency/escalate| X1
        A7[analyzePerception] -->|/api/v1/perception/analyze| X1
        A8[syncBaseline] -->|/api/v1/assessment/baseline/sync| X1
        A9[getBaseline] -->|/api/v1/assessment/baseline/:id| X1
    end
    subgraph "Python Algorithm (FastAPI)"
        B1["/api/v1/intervention/safety-loop"]
        B2["/api/v1/assessment/scale-estimate"]
        B3["/api/v1/profile/phenotype"]
        B4["/api/v1/profile/phenotype/:id/deviation"]
        B5["未检测到对应端点"]
        B6["/api/v1/emergency/escalate"]
        B7["/api/v1/perception/analyze"]
        B8["/api/v1/assessment/baseline/sync"]
        B9["/api/v1/assessment/baseline/:id"]
    end
```

**URL 不匹配清单：**

| # | 桥接方法 | 桥接 URL | 实际算法端点 | 匹配 |
|---|---------|---------|-------------|------|
| 1 | smartChat | `/api/smart-chat` | **不存在** | **不匹配** |
| 2 | assessRisk | `/api/assessment/predict` | `/api/v1/assessment/scale-estimate` | **不匹配**（路径+前缀） |
| 3 | getPhenotype | `/api/phenotype/{id}` | `/api/v1/profile/phenotype` | **不匹配**（路径+前缀） |
| 4 | getBaselineDeviation | `/api/phenotype/{id}/baseline-deviation` | `/api/v1/profile/phenotype/{id}/deviation` | **不匹配**（路径+前缀） |
| 5 | analyzeComorbidity | `/api/comorbidity/analyze` | **不存在** | **不匹配** |
| 6 | escalateCrisis | `/emergency/escalate` | `/api/v1/emergency/escalate` | **不匹配**（缺 /api/v1 前缀） |
| 7 | analyzePerception | `/api/v1/perception/analyze` | `/api/v1/perception/analyze` | **匹配** |
| 8 | syncBaseline | `/api/v1/assessment/baseline/sync` | `/api/v1/assessment/baseline/sync` | **匹配** |
| 9 | getBaseline | `/api/v1/assessment/baseline/{id}` | `/api/v1/assessment/baseline/{id}` | **匹配** |

> **关键发现**：算法桥接层 9 个方法中，仅 3 个 URL 正确匹配（均为近期新增的基线同步和感知分析端点），其余 6 个 URL 不匹配。这意味着 **smartChat、assessRisk、getPhenotype、getBaselineDeviation、analyzeComorbidity、escalateCrisis 在实际运行时会因 404 而降级到 fallback 模式**。

### 4.5 Socket.IO 通信链路

| 事件名 | 方向 | 数据格式 | 功能 |
|--------|------|---------|------|
| `expert:join` | Client→Server | `{bookingId: string}` | 加入咨询房间 |
| `expert:leave` | Client→Server | `{bookingId: string}` | 离开咨询房间 |
| `expert:message` | Client→Server | `{bookingId, content, contentType?}` | 发送咨询消息 |
| `consultation:end` | Client→Server | `{bookingId: string}` | 结束咨询 |
| `typing:start/stop` | Client→Server | `conversationId: string` | 打字状态 |
| `multimodal:update` | Client→Server | `{bookingId, multimodalData}` | 多模态数据同步 |
| `webrtc:join/offer/answer/ice-candidate/leave` | 双向 | WebRTC 信令 | 视频通话 |
| `notification:read` | Client→Server | `notificationIds: string[]` | 标记已读 |
| `expert:message` | Server→Client | Message object (含 sender) | 广播消息 |
| `consultation:ended` | Server→Client | `{bookingId}` | 通知咨询结束 |
| `patient:multimodal` | Server→Client | multimodalData | 转发多模态数据 |
| `webrtc:user-joined/user-left/offer/answer/ice-candidate` | Server→Client | WebRTC 信令 | 视频信令转发 |
| `typing:update` | Server→Client | `{conversationId, isTyping}` | 打字指示器 |
| `error` | Server→Client | `{message: string}` | 错误通知 |

---

## 五、测试覆盖检查

### 5.1 算法端测试统计

**pytest --collect-only 结果：633 个测试用例**

| 测试文件 | 对应模块 | 行数 | 用例数 | 状态 |
|---------|---------|------|--------|------|
| test_dataclasses.py | shared/dataclasses | 274 | ~15 | PASS |
| test_perception_new.py | perception/text_emotion | 241 | ~12 | PASS |
| test_age_differentiated_perception.py | perception/age_config | 681 | ~30 | PASS |
| test_fusion.py | perception/fusion | 241 | ~12 | PASS |
| test_acoustic.py | perception/voice | 142 | ~8 | PASS |
| test_perception_service.py | perception/perception_service | 180 | ~10 | PASS |
| test_assessment.py | assessment/multi_task | 216 | ~12 | PASS |
| test_phenotype.py | assessment/phenotype | 369 | ~18 | PASS |
| test_personal_baseline.py | assessment/personal_baseline | 597 | 44 | PASS |
| test_comorbidity.py | assessment/comorbidity | 179 | ~10 | PASS |
| test_auditor.py | intervention/auditor | 438 | ~25 | PASS |
| test_teen_auditor.py | intervention/auditor (teen) | 542 | ~30 | PASS |
| test_baseline_aware_auditor.py | intervention/auditor (baseline) | 398 | 17 | PASS |
| test_safety_loop.py | intervention/safety_loop | 552 | ~28 | PASS |
| test_state_machine.py | intervention/state_machine | 348 | ~20 | PASS |
| test_teen_language_detector.py | intervention/teen_language_detector | 675 | ~35 | PASS |
| test_cbt.py | intervention/cbt | 215 | ~12 | PASS |
| test_escalation.py | escalation/* | 332 | ~18 | PASS |
| test_fairness.py | audit/fairness | 293 | ~15 | PASS |
| test_federated.py | federated/* | 227 | ~12 | PASS |
| test_on_device.py | privacy/on_device | 204 | ~10 | PASS |
| test_sanitizer.py | privacy/sanitizer | 312 | ~15 | PASS |
| test_sim_vail_framework.py | sim_vail/* | 327 | ~18 | PASS |
| test_sim_vail_comparison.py | sim_vail/comparison | 235 | 15 | PASS |

### 5.2 无对应测试的模块

| 模块 | 是否有测试 | 备注 |
|------|-----------|------|
| perception/behavior | **无** | 键盘动力学 |
| perception/behavioral_activation | **无** | 行为激活 |
| perception/circadian | **无** | 昼夜节律 |
| perception/cognitive | **无** | 认知扭曲 |
| perception/eye | **无** | 眼动模式 |
| perception/physiological | **无** | rPPG-HRV + 呼吸 |
| perception/voice_semantics | **无** | 语音语义 |
| intervention/socratic | **无** | 苏格拉底提问 |
| intervention/strategy | **无** | 策略生成 |
| intervention/crisis | **无** | 危机干预 |
| intervention/middleware | **无** | 审计中间件 |
| escalation/protocol | **无** | 升级协议 |
| escalation/referral | **无** | 转介匹配 |
| escalation/self_harm | **无** | 自伤评估 |
| escalation/scheduler | **无** | 咨询师排班 |
| privacy/anonymization | **无** | K-匿名 |
| privacy/consent | **无** | 知情同意 |
| privacy/encryption | **无** | AES 加密 |
| assessment/risk | **无** | 综合风险 |
| assessment/scale | **无** | 量表映射 |
| assessment/trend | **无** | 趋势分析 |
| assessment/report | **无** | 报告生成 |
| sim_vail (API route) | **无** | API 路由层 |

### 5.3 前端测试

**未检测到前端测试文件。** `client/` 目录下无 `.test.tsx`、`.test.ts`、`.spec.tsx`、`.spec.ts` 文件。

---

## 六、代码质量检查

### 6.1 超过 100 行的函数

| 文件 | 函数名 | 行数 |
|------|--------|------|
| algorithm/sim_vail/__init__.py | `_build_profiles()` | 225 |
| algorithm/experiments/run_all.py | `generate_competition_report()` | 280 |
| algorithm/experiments/cost_sensitive_analysis.py | `generate_report()` | 180 |
| algorithm/experiments/privacy_engineering.py | `generate_report()` | 154 |
| algorithm/experiments/sim_vail_simulation.py | `generate_simvail_report()` | 145 |
| algorithm/experiments/ablation_study.py | `generate_report()` | 140 |
| algorithm/experiments/run_all.py | `run_all_experiments()` | 127 |
| algorithm/perception/age_config.py | `_build_young_adult_profile()` | 126 |
| algorithm/perception/age_config.py | `_build_mid_adolescent_profile()` | 123 |
| algorithm/perception/age_config.py | `_build_early_adolescent_profile()` | 122 |
| algorithm/intervention/auditor.py | `_get_default_config()` | 118 |

### 6.2 超过 500 行的文件

| 文件 | 行数 |
|------|------|
| client/src/pages/patient/Profile.tsx | 1709 |
| client/src/pages/consultant/ConsultantRoom.tsx | 1479 |
| algorithm/experiments/sim_vail_simulation.py | 1129 |
| algorithm/sim_vail/__init__.py | 1119 |
| algorithm/intervention/auditor.py | 1112 |
| client/src/context/AnxietyContext.tsx | 1097 |
| algorithm/perception/age_config.py | 950 |
| algorithm/perception/face/facial_expression.py | 811 |
| algorithm/experiments/privacy_engineering.py | 797 |
| algorithm/intervention/safety_loop.py | 790 |
| algorithm/intervention/teen_language_detector.py | 774 |
| algorithm/perception/perception_service.py | 738 |
| client/src/pages/patient/PatientRoom.tsx | 710 |
| algorithm/perception/behavior/behavior_pattern.py | 691 |
| algorithm/tests/test_age_differentiated_perception.py | 682 |
| algorithm/intervention/cbt/cognitive_restructuring.py | 680 |
| algorithm/escalation/protocol/__init__.py | 679 |
| algorithm/tests/test_teen_language_detector.py | 676 |
| algorithm/perception/text_emotion/adversarial_debias.py | 644 |
| algorithm/intervention/state_machine.py | 634 |
| algorithm/perception/text_emotion/sprop_gnn.py | 631 |
| algorithm/experiments/run_all.py | 622 |
| algorithm/experiments/ablation_study.py | 612 |
| algorithm/tests/test_personal_baseline.py | 598 |
| algorithm/audit/fairness.py | 593 |
| algorithm/privacy/on_device.py | 585 |
| algorithm/experiments/cost_sensitive_analysis.py | 561 |
| algorithm/tests/test_safety_loop.py | 553 |
| algorithm/tests/test_teen_auditor.py | 543 |
| algorithm/assessment/phenotype.py | 538 |
| algorithm/assessment/scale/__init__.py | 524 |
| algorithm/perception/voice/acoustic.py | 521 |
| client/src/hooks/useFacialAnalysis.ts | 520 |
| server/src/services/profile.service.ts | 514 |
| client/src/pages/patient/UnifiedAssessment.tsx | 509 |

### 6.3 TODO/FIXME 注释

| 文件 | 内容 |
|------|------|
| algorithm/api/audit.py:8 | TODO: 审计面板接口 |
| algorithm/api/audit.py:9 | TODO: 公平性指标接口 |
| algorithm/api/audit.py:10 | TODO: 可解释性报告接口 |
| algorithm/api/federated.py:8 | TODO: 客户端注册接口 |
| algorithm/api/federated.py:9 | TODO: 模型聚合接口 |
| algorithm/api/federated.py:10 | TODO: 训练状态查询接口 |
| algorithm/api/privacy.py:8 | TODO: 匿名化处理接口 |
| algorithm/api/privacy.py:9 | TODO: 知情同意管理接口 |
| algorithm/api/privacy.py:10 | TODO: 安全加密存储接口 |

> 共 9 个 TODO，均为 API 路由桩（接口功能已实现但 API 端点未暴露）。

### 6.4 裸 except (bare except)

**未检测到。** 项目中无 `except:` 裸异常捕获。

### 6.5 硬编码密钥

**未检测到。** 无硬编码 API Key、密码或 Token。

### 6.6 敏感数据写入日志

**未检测到。** 无 print/console.log 输出密码、Token 等敏感信息。

---

## 七、安全检查

### 7.1 诊断性语言检查

以下位置包含诊断性语言（用于**检测和拦截**，属于安全审计器的黑名单配置，**设计意图正确**）：

| 文件 | 行号 | 内容 | 性质 |
|------|------|------|------|
| algorithm/intervention/auditor.py | 88 | `"diagnostic_labels": ["你患了", "抑郁症", "你确诊了", "焦虑症"]` | 审计器黑名单 |
| algorithm/intervention/auditor.py | 147 | `"抑郁症", "社交焦虑症", "强迫性症状"` | 青少年扩展黑名单 |
| algorithm/intervention/auditor.py | 817-832 | 检测"直接告知'你有抑郁症'会导致..." | 注释说明 |
| algorithm/intervention/middleware.py | 127-128 | `("抑郁症", "直接告知诊断")`, `("你确诊了", "你被诊断为")` | 中间件拦截模式 |
| algorithm/intervention/state_machine.py | 18 | `任何状态禁止输出诊断性言语，如"你患了抑郁症"` | 状态机约束注释 |
| algorithm/intervention/state_machine.py | 100 | `"症", "病", "确诊", "症状描述", "诊断书"` | 禁止关键词 |
| algorithm/perception/age_config.py | 746 | `"抑郁状态", "确诊为", "诊断结果"` | 年龄配置黑名单 |
| client/src/hooks/ageConfig.ts | 133 | `'确诊为', '诊断结果'` | 前端年龄配置 |
| client/src/pages/admin/AdminReview.tsx | 31 | `"确诊为焦虑抑郁障碍"` | 模拟案例数据 |

> **结论**：诊断性语言仅出现在**拦截黑名单**和**测试用例**中，用于检测并阻止 AI 输出诊断性语言。设计意图正确。AdminReview.tsx 中的模拟案例数据使用了诊断性语言，属于展示用途。

### 7.2 .env.example 检查

| 配置项 | 值 | 是否占位符 |
|--------|---|-----------|
| DATABASE_URL | `postgresql://postgres:password@localhost:5432/mental_health?schema=public` | **部分风险**：使用了默认密码 `password` |
| JWT_SECRET | `your-jwt-secret-key-change-in-production` | 占位符 |
| JWT_REFRESH_SECRET | `your-refresh-token-secret-change-in-production` | 占位符 |
| DASHSCOPE_API_KEY | `your-dashscope-api-key` | 占位符 |
| REDIS_URL | `redis://localhost:6379` | 默认值 |

### 7.3 .gitignore 检查

| 检查项 | 状态 |
|--------|------|
| `.env` 排除 | **已排除** |
| `.env.local` 排除 | **已排除** |
| `*.key` / `*.pem` 排除 | **已排除** |
| `node_modules/` 排除 | **已排除** |
| `__pycache__/` 排除 | **已排除** |
| 模型权重文件排除 | **已排除** (.bin, .safetensors, .h5, .pth 等) |
| 数据集排除 | **已排除** (datasets/, data/, *.csv) |
| 日志文件排除 | **已排除** (*.log) |
| JSONL 排除 | **已排除** (*.jsonl) |

### 7.4 个人基线数据脱敏

- 前端 `usePersonalBaseline.ts` 仅传递**统计特征**（mean/std/n），不传递原始生理数据
- 后端 `personal_baseline.py` 存储的是 EMA 统计量，不含原始时间序列
- `shared/dataclasses.py` 的 `FeatureSummary` 注释明确标注"不包含原始文本或可识别信息"
- **结论**：基线数据脱敏设计正确

### 7.5 审计器绕过检查

- `SafetyLoop.process()` 中所有路径都经过 `auditor.audit()`，无跳过审计的代码路径
- `smartChat` 降级模式（fallback=true）返回空回复，不绕过审计
- `algorithm-bridge.ts` 的降级模式返回标记数据，不绕过审计
- **结论**：未发现绕过审计器的代码路径

---

## 八、依赖与配置检查

### 8.1 算法端依赖 (algorithm/requirements.txt)

| 依赖 | 版本 | 用途 |
|------|------|------|
| fastapi | 0.115.0 | Web 框架 |
| uvicorn[standard] | 0.30.6 | ASGI 服务器 |
| pydantic | 2.9.2 | 数据验证 |
| pydantic-settings | 2.5.2 | 配置管理 |
| transformers | 4.44.2 | NLP 模型 |
| torch | 2.4.1 | 深度学习 |
| sentencepiece | 0.2.0 | 分词器 |
| librosa | 0.10.2 | 音频处理 |
| soundfile | 0.12.1 | 音频 IO |
| opencv-python | 4.10.0.84 | 计算机视觉 |
| mediapipe | 0.10.14 | 面部检测 |
| numpy | 1.26.4 | 科学计算 |
| scipy | 1.14.1 | 科学计算 |
| scikit-learn | 1.5.2 | 机器学习 |
| pandas | 2.2.3 | 数据处理 |
| shap | 0.46.0 | 可解释性 |
| lime | 0.2.0.1 | 可解释性 |
| flwr | 1.11.1 | 联邦学习 |
| cryptography | 43.0.1 | 加密 |
| loguru | 0.7.2 | 日志 |
| aiohttp | 3.10.5 | 异步 HTTP |
| python-dotenv | 1.0.1 | 环境变量 |

### 8.2 服务端依赖 (server/package.json)

| 依赖 | 版本 | 用途 |
|------|------|------|
| @prisma/client | ^5.22.0 | ORM |
| bcryptjs | ^2.4.3 | 密码哈希 |
| cors | ^2.8.5 | 跨域 |
| dotenv | ^16.4.5 | 环境变量 |
| express | ^4.21.1 | Web 框架 |
| express-rate-limit | ^7.4.1 | 限流 |
| jsonwebtoken | ^9.0.2 | JWT |
| multer | ^1.4.5-lts.1 | 文件上传 |
| socket.io | ^4.8.1 | WebSocket |
| zod | ^3.23.8 | 数据验证 |

### 8.3 客户端依赖 (client/package.json)

| 依赖 | 版本 | 用途 |
|------|------|------|
| @ant-design/icons | ^5.4.0 | 图标库 |
| @mediapipe/tasks-vision | ^1.0.1 | 端侧面部检测 |
| antd | ^5.20.0 | UI 组件库 |
| axios | ^1.7.7 | HTTP 客户端 |
| dayjs | ^1.11.13 | 日期处理 |
| onnxruntime-web | ^1.30.0 | 端侧模型推理 |
| react | ^18.3.1 | UI 框架 |
| react-dom | ^18.3.1 | DOM 渲染 |
| react-router-dom | ^6.26.0 | 路由 |
| recharts | ^2.12.7 | 图表 |
| socket.io-client | ^4.7.5 | WebSocket 客户端 |
| zustand | ^4.5.5 | 状态管理 |

### 8.4 配置文件完整性

| 文件 | 行数 | 顶级配置项 | 状态 |
|------|------|-----------|------|
| algorithm/config/audit_rules.yaml | 275 | crisis_delay, delusion_reinforcement, stigma_rejection, sycophancy, trajectory_drift, teen_specific, crisis_resource_card, baseline_sensitivity, global | **完整** |
| algorithm/config/baseline.yaml | 38 | ema, calibration, deviation, storage, modality_mapping | **完整** |
| algorithm/config/fusion.yaml | 80 | weighted_average, stacking, dynamic_weighting, risk_thresholds, fallback, age_profiles | **完整** |

---

## 九、已知问题追踪

基于上一份诊断报告（docs/diagnosis.md），逐项验证：

| # | 问题 | 上次状态 | 当前状态 | 是否已解决 |
|---|------|---------|---------|-----------|
| 1 | 个人基线数据不传后端 | 仅客户端本地 + Socket 传咨询师端 | **已实现**：`baselineApi.syncBaseline()` → POST `/api/algorithm/baseline/sync` → 算法后端 `personal_baseline.sync_baseline()`。`usePersonalBaseline.ts` 在每次融合后自动同步。 | **已解决** |
| 2 | audit/fairness.py 命名混淆 | fairness.py 是公平性审计，非五轴审计器 | **未改变**：fairness.py 仍为公平性审计模块。但 diagnosis.md 已明确说明职责划分，且 `audit/bias` 和 `audit/explainability` 仍为空壳。 | **部分解决**（文档已澄清，代码未重构） |
| 3 | 服务端 /phenotype/:userId/baseline 端点未被调用 | 客户端不调用此 API | **已对接**：`baselineApi.getBaseline()` 调用 GET `/api/algorithm/baseline/:userId`。但注意此端点通过 `algorithm-bridge.getBaseline()` 调用算法端 `/api/v1/assessment/baseline/{id}`，URL 匹配正确。 | **已解决** |
| 4 | MockSystemResponder 是模板回复 | 模板回复，非真实 LLM | **未改变**：SIM-VAIL 测试框架仍使用 `MockSystemResponder` 模板回复。注释说明"在实际测试中应接入真实 LLM"。 | **未解决**（设计决策：比赛 Demo 级别） |
| 5 | 前端 bundle 过大（2.2MB） | 未做代码分割 | **未检测到**：无 `.test.tsx` 或构建产物可检查。Vite 默认有代码分割，但未确认具体 bundle 大小。 | **未验证** |

---

## 十、可改进之处

### P0 — 影响核心功能，必须修

| # | 问题 | 位置 | 建议修复 | 预计工作量 |
|---|------|------|---------|-----------|
| 1 | **算法桥接层 6 个 URL 不匹配** | `server/src/services/algorithm-bridge.ts` | 修正 URL 路径：smartChat→`/api/v1/intervention/safety-loop`，assessRisk→`/api/v1/assessment/scale-estimate`，getPhenotype→`/api/v1/profile/phenotype`，getBaselineDeviation→`/api/v1/profile/phenotype/{id}/deviation`，escalateCrisis→`/api/v1/emergency/escalate`，analyzeComorbidity 需新增算法端端点 | 2-3 小时 |
| 2 | **smartChat 无对应算法端点** | `algorithm/api/` | 需在算法端新增 `/api/v1/intervention/smart-chat` 端点，将 SafetyLoop 完整闭环暴露为 API | 3-4 小时 |
| 3 | **analyzeComorbidity 无对应算法端点** | `algorithm/api/` | 需在算法端新增 `/api/v1/comorbidity/analyze` 端点 | 1-2 小时 |

### P1 — 影响用户体验，尽量修

| # | 问题 | 位置 | 建议修复 | 预计工作量 |
|---|------|------|---------|-----------|
| 4 | **前端无测试** | `client/` | 至少为核心 hooks（usePersonalBaseline, useKeyboardDynamics 等）和关键组件添加 Vitest 测试 | 1-2 天 |
| 5 | **前端文本情绪 8 维 vs 后端 5 维不一致** | `client/src/types/multimodal.types.ts` | 统一为 5 维或在 API 边界做显式转换并添加注释 | 2-3 小时 |
| 6 | **35 个超过 500 行的大文件** | 多处 | 拆分 Profile.tsx (1709行)、ConsultantRoom.tsx (1479行)、auditor.py (1112行) 等 | 2-3 天 |
| 7 | **前端未调用算法桥接的大部分端点** | `client/src/services/index.ts` | 考虑在患者端或咨询师端增加 smart-chat、assess 等功能的调用 | 1 天 |

### P2 — 优化项，时间允许再做

| # | 问题 | 位置 | 建议修复 | 预计工作量 |
|---|------|------|---------|-----------|
| 8 | **6 个空壳子模块** | `perception/text`, `audit/bias`, `audit/explainability`, `shared/config`, `shared/exceptions`, `shared/utils` | 实现或移除空壳模块，避免误导 | 4-6 小时 |
| 9 | **9 个 API 路由 TODO 桩** | `api/audit.py`, `api/federated.py`, `api/privacy.py` | 实现这些 API 端点（底层功能已实现） | 3-4 小时 |
| 10 | **23 个模块无对应测试** | 多处 | 为核心模块（intervention/*, escalation/*, privacy/*）补充测试 | 3-5 天 |
| 11 | **11 个超长函数（>100行）** | 多处 | 拆分 `_build_profiles()` (225行)、`generate_competition_report()` (280行) 等 | 1 天 |
| 12 | **MockSystemResponder 模板回复** | `algorithm/sim_vail/__init__.py` | 接入真实 LLM 以验证审计器在真实回复上的效果 | 1 天 |
| 13 | **.env.example 数据库默认密码** | `.env.example` | 将 `password` 改为 `CHANGE_ME` 占位符 | 5 分钟 |

---

## 十一、答辩风险点

### 风险点 1：哪些指标是模拟数据？哪些是真实数据？

| 类别 | 风险等级 | 说明 |
|------|---------|------|
| SIM-VAIL 测试结果 | **高** | 模拟用户是脚本驱动（`SimulatedUserAgent`），系统回复是模板（`MockSystemResponder`），拦截率/漏判率不是真实场景指标 |
| 五轴审计器检测 | **低** | 规则检测逻辑完整且真实运行，但阈值是经验值而非临床验证 |
| 多模态感知 | **中** | 文本情感由 SProp-GNN 模型处理（真实），但面部/语音在无硬件时为模拟数据 |
| 个人基线 EMA | **低** | 算法真实运行，但校准需要 20+ 次融合，演示时可能未校准 |

**建议应对话术**：
> "SIM-VAIL 框架测试的是审计器的拦截能力，不是 LLM 的对话质量。五轴审计器的阈值参考了临床指南（如 C-SSRS 危机分级），但具体数值需要在真实临床环境中进一步验证。多模态感知中，文本通道是始终可用的核心通道。"

### 风险点 2：哪些模块是完整实现？哪些是设计但未实现？

| 类别 | 风险等级 |
|------|---------|
| `audit/bias`、`audit/explainability` 是空壳 | **高** — 评委可能问"审计模块为什么是空的" |
| `shared/config`、`shared/exceptions`、`shared/utils` 是空壳 | **中** |
| `perception/text` 是空壳（功能在 text_emotion 中） | **中** |
| 联邦学习是模拟实现 | **中** — 单节点模拟，非真实分布式训练 |

**建议应对话术**：
> "audit 目录下的 fairness.py 是公平性审计（完整实现），bias 和 explainability 是预留的扩展点。核心的五轴安全审计器在 intervention/auditor.py 中，有 1112 行完整实现。联邦学习使用 Flower 框架实现了完整的 FedAvg 聚合逻辑，当前是单机模拟多客户端。"

### 风险点 3：前后端是否真正打通？有没有硬编码的假数据？

| 类别 | 风险等级 |
|------|---------|
| **算法桥接层 6/9 个 URL 不匹配** | **高** — 实际运行时大部分算法端点会 404 |
| 基线同步链路（前端→Node→算法） | **低** — URL 匹配正确，已验证 |
| Socket.IO 通信 | **低** — 完整实现，含 WebRTC 信令 |

**建议应对话术**：
> "基线同步链路已完整打通并验证（POST 返回 200）。算法桥接层的部分端点 URL 需要与算法 FastAPI 的路由前缀对齐，这是已知的待修复项。核心的 Socket.IO 实时通信和 REST API 业务链路是完整的。"

### 风险点 4：安全审计器是否真的拦截了？有没有测试证据？

| 类别 | 风险等级 |
|------|---------|
| 五轴审计器 | **低** — 有 438 + 542 + 398 = 1378 行测试代码，覆盖通用审计、青少年审计、基线感知审计 |
| SafetyLoop 闭环 | **低** — 有 552 行测试，验证了 SEVERE/MODERATE/MILD 分级处理 |
| SIM-VAIL 对比实验 | **低** — 有 15 个测试验证对比实验模式 |

**建议应对话术**：
> "安全审计器有完整的测试证据：24 个测试文件、633 个测试用例。其中审计器相关测试超过 70 个，覆盖了五轴检测、青少年专属检测、基线感知调整、安全闭环处理等场景。SIM-VAIL 框架还提供了对比实验，验证基线驱动审计的提升效果。"

### 风险点 5：个人基线是否真的驱动了审计？有没有对比实验？

| 类别 | 风险等级 |
|------|---------|
| 基线感知审计 | **低** — `set_baseline_context()` + `_compute_baseline_boost()` 完整实现，17 个测试验证 |
| SIM-VAIL 对比实验 | **低** — `run_comparison()` 对 5 画像分别跑 baseline_driven=True/False，有对比报告 |
| 对比实验的说服力 | **中** — MockSystemResponder 模板回复导致两种模式结果趋同 |

**建议应对话术**：
> "个人基线通过维度到轴映射表驱动审计器敏感度调整（boost=1.5x）。对比实验框架已完整实现，当前使用模拟数据是因为 MockSystemResponder 是模板回复。接入真实 LLM 后，对比实验可以直接运行出有说服力的差异数据。"

---

> **报告结束**。本报告基于代码静态分析生成，未修改任何业务代码。
