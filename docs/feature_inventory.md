# 已实现功能清单

> 生成时间：2026-09-18 | 基于代码扫描，所有功能均已在代码中实现  
> 数据真实性标注：🟢 真实 | 🟡 部分真实（规则引擎/降级） | 🔴 模拟  
> 状态标注：✅ 完整可用 | ⚠️ 部分可用 | ❌ 不可用

---

## 一、患者端功能（React 前端）

> 源码目录：`client/src/pages/patient/`

| 页面 | 路由 | 核心功能 | 依赖后端 API | 数据真实性 | 状态 |
|---|---|---|---|---|---|
| **Home.tsx** | `/` | 首页仪表盘：情绪卡片、快捷入口、今日推荐 | `/api/profile/psychological`, `/api/consultation/today` | 🟡 调用真实 API，画像由 AI 生成 | ✅ |
| **Chat.tsx** | `/chat`, `/chat/:id` | AI 倾诉：多轮对话、长期记忆、风格匹配、情绪感知 | `/api/consultation/chat` → 桥接 `algorithm-bridge.ts` → `/api/v1/intervention/smart-chat` | 🟢 全链路真实：感知→状态机→审计→LLM | ✅ |
| **Companions.tsx** | `/companions` | 同伴社区：发帖、评论、点赞、AI 审核 | `/api/community/*` (CRUD) | 🟢 真实 CRUD + AI 内容审核 | ✅ |
| **Healing.tsx** | `/healing` | 疗愈空间：冥想引导、呼吸练习、白噪音 | `/api/healing/*` | 🟢 真实会话记录（Prisma） | ✅ |
| **MyGrowth.tsx** | `/growth` | 我的成长：成就系统、成长轨迹、数据统计 | `/api/profile/assessments`, `/api/mood` | 🟡 调用真实 API，部分统计前端计算 | ✅ |
| **Profile.tsx** | `/profile` | 个人画像：心理评分、情绪趋势、会员管理、测评 | `/api/profile/psychological`, `/api/profile/mood-trend`, `/api/profile/emotional-state`, `/api/profile/coping-strategies`, `/api/profile/membership` | 🟢 真实 API（路由修复后） | ✅ |
| **Experts.tsx** | `/experts` | 找帮手：咨询师列表、预约、筛选 | `/api/expert/*` | 🟢 真实 CRUD | ✅ |
| **PatientRoom.tsx** | `/experts/room/:bookingId` | 咨询室：实时聊天、多模态采集、视频通话 | `/api/expert/room/*`, WebSocket, 算法桥接 | 🟢 真实消息 + 算法分析 | ✅ |
| **Settings.tsx** | `/settings` | 设置：个人信息、隐私、通知偏好 | `/api/auth/profile`, `/api/notification/*` | 🟢 真实 CRUD | ✅ |
| **UnifiedAssessment.tsx** | （内嵌 Profile） | 心理测评：PHQ-9/GAD-7 量表、自动评分 | `/api/profile/assessments` (POST) | 🟢 真实评分 + 持久化 | ✅ |
| Achievements.tsx | → `/growth` 重定向 | 成就展示（已合并到 MyGrowth） | — | — | ⚠️ 已重定向 |
| Assessment.tsx | → `/profile` 重定向 | 旧版测评（已合并到 UnifiedAssessment） | — | — | ⚠️ 已重定向 |
| Sleep.tsx | → `/healing` 重定向 | 睡眠监测（已合并到 Healing） | — | — | ⚠️ 已重定向 |
| Feedback.tsx | → `/settings` 重定向 | 意见反馈（已合并到 Settings） | — | — | ⚠️ 已重定向 |
| Treatment.tsx | → `/chat` 重定向 | 治疗规划（已合并到 Chat） | — | — | ⚠️ 已重定向 |
| Learning.tsx | → `/healing` 重定向 | 学习中心（已合并到 Healing） | — | — | ⚠️ 已重定向 |

**活跃页面：10 个**（Home, Chat, Companions, Healing, MyGrowth, Profile, Experts, PatientRoom, Settings, UnifiedAssessment）  
**重定向页面：6 个**（已整合到核心页面）

---

## 二、咨询师端功能

> 源码目录：`client/src/pages/consultant/`

| 页面 | 路由 | 核心功能 | 依赖后端 API | 数据真实性 | 状态 |
|---|---|---|---|---|---|
| **ConsultantDashboard.tsx** | `/consultant` | 工作台：今日预约、统计概览 | `/api/consultation/dashboard` | 🟢 真实数据 | ✅ |
| **ConsultantRoom.tsx** | `/consultant/consultations/:bookingId` | 咨询室：实时聊天、患者画像侧栏、多模态面板 | `/api/expert/room/*`, `/api/profile/consultation-data` | 🟢 真实消息 + 实时画像 | ✅ |
| **ConsultantAppointments.tsx** | `/consultant/appointments` | 预约管理：排班、确认/拒绝预约 | `/api/expert/appointments` | 🟢 真实 CRUD | ✅ |
| **ConsultantConsultations.tsx** | `/consultant/consultations` | 咨询记录：历史咨询列表 | `/api/expert/consultations` | 🟢 真实记录 | ✅ |
| **ConsultantProfiles.tsx** | `/consultant/profiles` | 来访者画像：患者列表、焦虑数据、画像摘要 | `/api/profile/consultation-data`, `/api/profile/patient-anxiety` | 🟢 真实数据 | ✅ |
| **ConsultantSettings.tsx** | `/consultant/settings` | 咨询师设置：个人信息、资质编辑 | `/api/auth/profile` | 🟢 真实 CRUD | ✅ |
| ConsultantLayout.tsx | — | 咨询师端布局（导航框架） | — | — | ✅ |

**活跃页面：6 个**

---

## 三、管理端功能

> 源码目录：`client/src/pages/admin/`

| 页面 | 路由 | 核心功能 | 依赖后端 API | 数据真实性 | 状态 |
|---|---|---|---|---|---|
| **Dashboard.tsx** | `/admin` | 仪表盘：用户统计、危机统计、平台概览 | `/api/crisis/stats`, `/api/admin/stats` | 🟢 真实统计 | ✅ |
| **AdminUsers.tsx** | `/admin/users` | 用户管理：列表、搜索、禁用/启用 | `/api/admin/users` | 🟢 真实 CRUD | ✅ |
| **AdminConsultants.tsx** | `/admin/consultants` | 咨询师管理：审核资质、启用/禁用 | `/api/admin/consultants` | 🟢 真实 CRUD | ✅ |
| **AdminReview.tsx** | `/admin/review` | 内容审核：社区帖子 AI + 人工审核 | `/api/review/*` | 🟢 真实审核流程 | ✅ |
| **AdminCrisis.tsx** | `/admin/crisis` | 危机管理：危机记录列表、统计、处理 | `/api/crisis/records`, `/api/crisis/stats` | 🟢 真实危机记录 | ✅ |
| **AdminFeedback.tsx** | `/admin/feedback` | 反馈管理：用户反馈列表、回复 | `/api/admin/feedback` | 🟢 真实 CRUD | ✅ |
| AdminLayout.tsx | — | 管理端布局（导航框架） | — | — | ✅ |

**活跃页面：6 个**

---

## 四、算法端功能（Python FastAPI）

> 源码目录：`algorithm/api/` | 基础路径：`/api/v1`

### 4.1 感知层（perception）

> 路由文件：`algorithm/api/perception.py`

| 端点 | 方法 | 功能 | 依赖模块 | 数据真实性 |
|---|---|---|---|---|
| `/perception/analyze` | POST | 多模态情感分析（文本+面部+行为+语音+生理+认知+眼动） | `perception/perception_service.py` | 🟡 文本为 jieba+词典，其他模态为模拟输入 |
| `/perception/debias` | POST | SProp GNN 语义盲法去偏分析 | `perception/debias/` | 🟡 GNN 模型已实现，输入为模拟 |
| `/perception/debias/batch` | POST | 批量 SProp GNN 去偏 | `perception/debias/` | 🟡 同上 |
| `/perception/debias/stats` | GET | SProp GNN 统计信息 | `perception/debias/` | 🟢 真实统计 |
| `/perception/age-groups` | GET | 年龄分组列表 | `perception/age_config.py` | 🟢 真实配置 |

### 4.2 评估层（assessment）

> 路由文件：`algorithm/api/assessment.py`

| 端点 | 方法 | 功能 | 依赖模块 | 数据真实性 |
|---|---|---|---|---|
| `/assessment/scale-estimate` | POST | 多模态→量表分值自动映射（PHQ-9/GAD-7） | `assessment/multi_task/` | 🟡 规则映射，未训练端到端模型 |
| `/assessment/risk-trend` | POST | 风险趋势分析（时序建模） | `assessment/trend/` | 🟢 真实趋势计算 |
| `/assessment/baseline/sync` | POST | 同步个人基线（Welford 累积模式） | `assessment/personal_baseline.py` | 🟢 真实累积算法 |
| `/assessment/baseline/deviation` | POST | 计算基线偏离度（Z-score） | `assessment/personal_baseline.py` | 🟢 真实 Z-score |
| `/assessment/baseline/{user_id}` | GET | 获取个人基线 | `assessment/personal_baseline.py` | 🟢 真实持久化 |
| `/assessment/comorbidity/analyze` | POST | 共病模式分析（抑郁+焦虑+睡眠） | `assessment/comorbidity.py` | 🟡 规则引擎 |
| `/assessment/predict` | POST | 文本风险快速评估 | `assessment/phenotype.py` | 🟡 规则映射 |

### 4.3 干预层（intervention）

> 路由文件：`algorithm/api/intervention.py`

| 端点 | 方法 | 功能 | 依赖模块 | 数据真实性 |
|---|---|---|---|---|
| `/intervention/smart-chat` | POST | 智能对话（完整闭环：感知→状态机→审计→LLM→记忆） | 全链路 | 🟢 真实闭环 |
| `/intervention/audit-events` | GET | 获取安全审计事件列表 | `InMemoryAuditStore` | 🟢 真实内存存储 |
| `/intervention/audit-event` | POST | 添加审计事件 | `InMemoryAuditStore` | 🟢 真实 |
| `/intervention/strategy` | POST | 结构化策略生成（Action→模板→安全→回复） | `intervention/strategy/` | 🟡 模板引擎 |
| `/intervention/strategy/stats` | GET | 策略生成统计 | `intervention/strategy/` | 🟢 真实统计 |
| `/intervention/safety-loop` | POST | SafetyLoop 闭环处理（审计→状态切换→升级） | `intervention/safety_loop.py` | 🟢 真实闭环 |
| `/intervention/safety-loop/log` | GET | 获取闭环日志 | `intervention/safety_loop.py` | 🟢 真实 |
| `/intervention/safety-loop/state` | GET | 获取闭环状态 | `intervention/safety_loop.py` | 🟢 真实 |
| `/intervention/cbt-record` | POST | 获取 CBT 认知重构记录 | `intervention/cbt/` | 🟢 真实 |
| `/intervention/cbt-records` | POST | 获取用户所有 CBT 记录 | `intervention/cbt/` | 🟢 真实 |
| `/intervention/memory/query` | POST | 查询用户记忆（语义匹配） | `assessment/user_memory.py` | 🟢 真实持久化 |
| `/intervention/memory/list` | POST | 获取用户所有记忆 | `assessment/user_memory.py` | 🟢 真实 |
| `/intervention/memory/delete` | POST | 删除单条记忆 | `assessment/user_memory.py` | 🟢 真实 |
| `/intervention/memory/clear` | POST | 清空用户所有记忆 | `assessment/user_memory.py` | 🟢 真实 |
| `/intervention/memory/decay` | POST | 执行记忆降权（遗忘曲线） | `assessment/user_memory.py` | 🟢 真实 |

### 4.4 升级层（escalation）

> 路由文件：`algorithm/api/escalation.py`

| 端点 | 方法 | 功能 | 依赖模块 | 数据真实性 |
|---|---|---|---|---|
| `/emergency/escalate` | POST | 危机升级（三通道通知） | `escalation/crisis.py` | 🟢 真实 |
| `/emergency/report` | GET | 危机报告 | `escalation/` | 🟢 真实 |
| `/emergency/proactive-care/consent` | POST | 主动关怀授权 | `escalation/proactive_care.py` | 🟢 真实 |
| `/emergency/proactive-care/consent/{user_id}` | GET | 查询主动关怀状态 | `escalation/proactive_care.py` | 🟢 真实 |
| `/emergency/proactive-care/evaluate` | POST | 评估主动关怀触发 | `escalation/proactive_care.py` | 🟢 真实 |

### 4.5 审计模块（audit）

> 路由文件：`algorithm/api/audit.py`

| 端点 | 方法 | 功能 | 依赖模块 | 数据真实性 |
|---|---|---|---|---|
| `/audit/fairness` | POST | 公平性审计（偏见检测 + 图表生成） | `audit/fairness.py` | 🟢 真实 |
| `/audit/report/{report_id}` | GET | 获取审计报告 | `audit/` | 🟢 真实 |
| `/audit/chart/{report_id}` | GET | 获取审计图表 | `audit/` | 🟢 真实 |

### 4.6 隐私模块（privacy）

> 路由文件：`algorithm/api/privacy.py`

| 端点 | 方法 | 功能 | 依赖模块 | 数据真实性 |
|---|---|---|---|---|
| `/privacy/anonymize` | POST | 文本匿名化（PII 脱敏） | `privacy/anonymization/` | 🟢 真实 |
| `/privacy/consent` | POST | 创建知情同意 | `privacy/consent/` | 🟢 真实 |
| `/privacy/consent/{user_id}` | GET | 查询同意状态 | `privacy/consent/` | 🟢 真实 |
| `/privacy/encrypt` | POST | 数据加密 | `privacy/encryption/` | 🟢 真实 |

### 4.7 联邦学习（federated）

> 路由文件：`algorithm/api/federated.py`

| 端点 | 方法 | 功能 | 依赖模块 | 数据真实性 |
|---|---|---|---|---|
| `/federated/status` | GET | 联邦学习状态 | `federated/` | 🟢 真实 |
| `/federated/aggregate` | POST | 模型聚合（FedAvg） | `federated/server.py` | 🟡 模拟客户端 |
| `/federated/register` | POST | 客户端注册 | `federated/client.py` | 🟡 模拟 |

### 4.8 数字表型（phenotype）

> 路由文件：`algorithm/api/phenotype.py`

| 端点 | 方法 | 功能 | 依赖模块 | 数据真实性 |
|---|---|---|---|---|
| `/profile/phenotype` | POST | 数字表型提取 | `assessment/phenotype.py` | 🟡 规则计算 |
| `/profile/phenotype/{user_id}/deviation` | GET | 表型基线偏离 | `assessment/phenotype.py` | 🟡 规则计算 |

### 4.9 SIM-VAIL 仿真验证

> 路由文件：`algorithm/api/sim_vail.py`

| 端点 | 方法 | 功能 | 依赖模块 | 数据真实性 |
|---|---|---|---|---|
| `/sim-vail/run` | POST | 一键运行 SIM-VAIL 测试 | `experiments/sim_vail_simulation.py` | 🟢 真实仿真 |
| `/sim-vail/report` | GET | 获取最近测试报告 | `experiments/outputs/` | 🟢 真实报告 |
| `/sim-vail/profiles` | GET | 获取画像列表 | `experiments/` | 🟢 真实 |
| `/sim-vail/comparison` | POST | 运行基线驱动对比实验 | `experiments/` | 🟢 真实对比 |

**算法端点总计：48 个**

---

## 五、核心安全链路功能

> 感知 → 评估 → 干预 → 升级 完整闭环

| 环节 | 功能 | 实现模块 | 数据真实性 |
|---|---|---|---|
| 感知 | 文本情感分析（jieba+词典，120+ 情感词） | `perception/text_emotion/predict.py` | 🟢 jieba+词典 |
| 感知 | 危机关键词检测（89 词库，覆盖直接/间接/隐晦/青少年表达） | `perception/perception_service.py` | 🟢 真实 |
| 感知 | 年龄分层感知（3 年龄段：儿童/青少年/青年，差异化关键词库） | `perception/age_config.py` | 🟢 真实 |
| 感知 | 多模态融合（文本+语音加权，支持 3 种融合策略） | `perception/fusion/fusion.py` | 🟡 仅文本通道实测 |
| 感知 | SProp GNN 语义盲法去偏 | `perception/debias/` | 🟡 算法已实现，输入模拟 |
| 感知 | 昼夜节律特征 | `perception/circadian/` | 🟡 特征已实现 |
| 感知 | 认知扭曲检测 | `perception/cognitive/` | 🟡 特征已实现 |
| 感知 | 呼吸模式分析 | `perception/breathing/` | 🔴 模拟输入 |
| 感知 | HRV 风险分析 | `perception/physiological/` | 🔴 模拟输入 |
| 感知 | 行为激活分析 | `perception/behavioral_activation/` | 🔴 模拟输入 |
| 感知 | 眼动追踪分析 | `perception/eye/` | 🔴 模拟输入 |
| 感知 | 语音语义深度分析 | `perception/voice_semantics/` | 🔴 模拟输入 |
| 评估 | 个人基线（Welford 在线累积算法） | `assessment/personal_baseline.py` | 🟢 真实累积 |
| 评估 | 基线偏离计算（Z-score，阈值 |Z|>2） | `assessment/personal_baseline.py` | 🟢 真实 |
| 评估 | 多任务风险预测（PHQ-9/GAD-7 估计） | `assessment/multi_task/` | 🟡 规则映射 |
| 评估 | 共病模式分析 | `assessment/comorbidity.py` | 🟡 规则引擎 |
| 评估 | 风险趋势分析 | `assessment/trend/` | 🟢 真实 |
| 评估 | 数字表型提取 | `assessment/phenotype.py` | 🟡 规则计算 |
| 干预 | 对话状态机（5 状态 7 转移：INIT→EMPATHY→EXPLORATION→ACTION→CRISIS） | `intervention/state_machine.py` | 🟢 真实 |
| 干预 | 五轴安全审计器（危机延迟/自伤检测/污名化/谄媚/漂移） | `intervention/auditor.py` | 🟢 786 测试通过 |
| 干预 | 青少年语用检测（网络俚语/缩略语/表情密码） | `intervention/teen_language_detector.py` | 🟢 真实 |
| 干预 | 风格检测 + 回复节奏匹配（4 风格：内向/外向/激动/平静） | `intervention/style_detector.py` | 🟢 真实 |
| 干预 | CBT 认知重构引导 | `intervention/cbt/` | 🟢 真实 |
| 干预 | 苏格拉底式对话引导 | `intervention/socratic/` | ⚠️ 部分验证 |
| 干预 | SafetyLoop 闭环（审计→状态切换→自动升级） | `intervention/safety_loop.py` | 🟢 真实 |
| 干预 | 用户长期记忆（提取/存储/检索/降权/遗忘曲线） | `assessment/user_memory.py` | 🟢 JSON 持久化 |
| 升级 | 危机升级协议（三通道：平台通知+咨询师+紧急联系人） | `escalation/crisis.py` | 🟢 真实 |
| 升级 | 自伤检测与即时干预 | `escalation/self_harm/` | 🟢 真实 |
| 升级 | 咨询师排班调度 | `escalation/scheduler.py` | 🟢 真实 |
| 升级 | 审计日志（SHA-256 哈希链，防篡改） | `escalation/audit_log.py` | 🟢 真实 |
| 升级 | 主动关怀机制（触发条件+授权+每天限 1 条+危机转人工） | `escalation/proactive_care.py` | 🟢 真实 |
| 审计 | 公平性审计（偏见检测 + 可视化图表） | `audit/fairness.py` | 🟢 真实 |
| 审计 | 可解释性报告 | `audit/explainability/` | 🟢 真实 |
| 隐私 | PII 匿名化 | `privacy/anonymization/` | 🟢 真实 |
| 隐私 | 知情同意管理 | `privacy/consent/` | 🟢 真实 |
| 隐私 | AES 加密 | `privacy/encryption/` | 🟢 真实 |
| 隐私 | 端侧推理 | `privacy/on_device.py` | 🟡 已实现 |
| 联邦 | FedAvg 聚合 | `federated/server.py` | 🟡 模拟客户端 |

---

## 六、数据持久化功能

| 存储位置 | 存储内容 | 持久化方式 | 跨会话 |
|---|---|---|---|
| `algorithm/assessment/outputs/baselines.json` | 个人基线（PHQ-9/GAD-7 均值/标准差/样本数） | JSON 文件 | ✅ |
| `algorithm/assessment/outputs/user_memories.json` | 用户长期记忆（提取/检索/降权） | JSON 文件 | ✅ |
| `algorithm/escalation/outputs/audit_logs.json` | 审计日志（SHA-256 哈希链） | JSON 文件 | ✅ |
| PostgreSQL (Prisma) | 用户/咨询师/对话/消息/心理画像/测评/心情/预约/社区/危机/治疗/学习 | Prisma ORM | ✅ |
| 前端 localStorage | Token、基线缓存 | 浏览器本地 | ⚠️ 仅本机 |
| 内存 | 审计事件（InMemoryAuditStore） | 进程内存 | ❌ 重启丢失 |

### Prisma 数据模型（15 个）

| 模型 | 用途 |
|---|---|
| User | 用户账号（患者/咨询师/管理员） |
| RefreshToken | JWT 刷新令牌 |
| PatientProfile | 患者画像（风险等级/会员） |
| ConsultantProfile | 咨询师画像（资质/专长） |
| Conversation | 对话会话 |
| Message | 对话消息 |
| PsychologicalProfile | 心理画像记录 |
| Assessment | 心理测评记录 |
| HealingSession | 疗愈会话 |
| MoodRecord | 心情记录 |
| ExpertBooking | 专家预约 |
| ExpertMessage | 咨询室消息 |
| TreatmentPlan / TreatmentTask | 治疗计划/任务 |
| LearningRecord | 学习记录 |
| CommunityPost / CommunityComment / CommunityLike | 社区帖子/评论/点赞 |
| CrisisRecord | 危机记录 |
| Notification | 通知 |

---

## 七、前后端桥接功能

> 源码：`server/src/services/algorithm-bridge.ts`

| 桥接方法 | 算法端点 | 前端调用位置 | 功能 |
|---|---|---|---|
| `smartChat()` | `/api/v1/intervention/smart-chat` | `Chat.tsx` → `consultation.service.ts` | 智能对话（完整闭环） |
| `assessRisk()` | `/api/v1/assessment/predict` | `consultation.service.ts` | 文本风险快速评估 |
| `analyzePerception()` | `/api/v1/perception/analyze` | `ConsultantRoom.tsx` | 多模态情感分析 |
| `analyzeMultimodalUpload()` | `/api/v1/perception/analyze` | `PatientRoom.tsx` | 多模态上传分析 |
| `getPhenotype()` | `/api/v1/profile/phenotype` | — | 数字表型提取 |
| `getBaselineDeviation()` | `/api/v1/profile/phenotype/{id}/deviation` | — | 表型基线偏离 |
| `analyzeComorbidity()` | `/api/v1/assessment/comorbidity/analyze` | — | 共病模式分析 |
| `escalateCrisis()` | `/api/v1/emergency/escalate` | `consultation.service.ts` | 危机升级 |
| `syncBaseline()` | `/api/v1/assessment/baseline/sync` | `AnxietyContext.tsx` | 基线同步 |
| `getBaseline()` | `/api/v1/assessment/baseline/{user_id}` | `AnxietyContext.tsx` | 获取基线 |

---

## 八、Node 后端路由

> 源码目录：`server/src/routes/`

| 路由文件 | 前缀 | 端点数 | 功能 |
|---|---|---|---|
| `auth.routes.ts` | `/api/auth` | 4 | 注册/登录/刷新令牌/获取资料 |
| `profile.routes.ts` | `/api/profile` | 15 | 心理画像/情绪趋势/测评/心情/会员 |
| `consultation.routes.ts` | `/api/consultation` | 5 | 对话/今日咨询/仪表盘 |
| `community.routes.ts` | `/api/community` | 8 | 社区 CRUD + 点赞 |
| `expert.routes.ts` | `/api/expert` | 6 | 咨询师列表/预约/咨询室 |
| `healing.routes.ts` | `/api/healing` | 3 | 疗愈会话 |
| `crisis.routes.ts` | `/api/crisis` | 4 | 危机记录/统计 |
| `mood.routes.ts` | `/api/mood` | 4 | 心情记录 CRUD |
| `review.routes.ts` | `/api/review` | 3 | 内容审核 |
| `notification.routes.ts` | `/api/notification` | 4 | 通知管理 |
| `treatment.routes.ts` | `/api/treatment` | 3 | 治疗计划 |
| `learning.routes.ts` | `/api/learning` | 3 | 学习记录 |
| `algorithm.routes.ts` | `/api/algorithm` | 5 | 算法桥接代理 |
| `extra.routes.ts` | `/api/*` | 8 | 反馈/统计/其他 |

---

## 九、功能统计汇总

| 指标 | 数值 |
|---|---|
| 患者端活跃页面 | 10 个（完整可用 10 个） |
| 咨询师端页面 | 6 个（完整可用 6 个） |
| 管理端页面 | 6 个（完整可用 6 个） |
| 算法端端点 | 48 个 |
| Node 后端端点 | ~73 个 |
| 桥接层方法 | 10 个 |
| Prisma 数据模型 | 15+ 个 |
| 核心闭环功能 | 12 个（感知 3 + 评估 4 + 干预 6 + 升级 5 + 审计 2 + 隐私 4） |
| 全量测试 | 786 passed, 1 skipped, 0 failures |

### 数据真实性分布

| 等级 | 数量 | 说明 |
|---|---|---|
| 🟢 真实 | ~35 个端点 | 调用真实算法/数据库，返回真实数据 |
| 🟡 部分真实 | ~10 个端点 | 调用真实 API 但后端使用规则引擎/降级 |
| 🔴 模拟 | ~3 个端点 | 模拟输入或纯模拟数据 |

### 已实现但待深度验证的功能

| 功能 | 文件 | 说明 |
|---|---|---|
| SProp GNN 去偏 | `perception/debias/` | 算法已实现，仅模拟输入测试 |
| 联邦学习聚合 | `federated/server.py` | FedAvg 已实现，单节点模拟 |
| 多模态融合（语音通道） | `perception/fusion/` | 加权融合已实现，语音未接入真实传感器 |
| 苏格拉底式对话 | `intervention/socratic/` | 状态机已实现，端到端验证有限 |
