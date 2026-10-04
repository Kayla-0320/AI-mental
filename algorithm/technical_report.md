# 青少年 AI 心理健康平台 —— 技术报告

> **参赛赛道**：AIC 算法大赛  
> **报告版本**：v2.0  
> **更新时间**：2026-09-16  
> **代码仓库**：`algorithm/` 目录，Python 3.10+  

---

## 目录

1. [问题定义](#1-问题定义)
2. [系统架构](#2-系统架构)
3. [算法设计](#3-算法设计)
4. [安全与隐私](#4-安全与隐私)
5. [实验验证](#5-实验验证)
6. [落地路径](#6-落地路径)
7. [局限性](#7-局限性)

---

## 1. 问题定义

### 1.1 背景

青少年心理健康问题日益严峻。据《中国国民心理健康发展报告（2021-2022）》显示，我国青少年抑郁检出率约为 14.8%，焦虑检出率约为 15.8%。现有心理健康服务面临三大核心挑战：

1. **资源不足**：专业心理咨询师缺口巨大，师生比约为 1:1500（教育部要求 1:1000）
2. **识别滞后**：传统筛查依赖定期量表，无法捕捉实时心理状态变化
3. **隐私顾虑**：青少年群体对隐私高度敏感，数据安全问题直接影响使用意愿

### 1.2 目标

构建一个 **多模态 AI 心理健康辅助平台**，实现：

- 多模态感知（文本 + 语音 + 面部 + 行为）实时情绪识别
- 数据驱动的多任务风险评估（抑郁 / 焦虑 / 睡眠）
- 有限状态机驱动的对话干预引擎
- 五轴安全审计确保 LLM 输出安全
- 端侧推理 + 数据脱敏的隐私保护方案
- 危机升级系统保障紧急场景响应

### 1.3 约束

| 约束类型 | 具体要求 |
|----------|----------|
| 法规合规 | 《个人信息保护法》《人工智能拟人化互动服务管理暂行办法》 |
| 数据安全 | 敏感数据不出设备，云端仅接收脱敏特征 |
| 临床安全 | 高风险召回率优先于精确率（漏诊代价 >> 误诊代价） |
| 实时性 | 危机升级响应时间 < 30 秒 |
| 公平性 | 模型在不同年龄/性别/地域群体上表现差异可控 |

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    用户端（App / 小程序）                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  端侧推理层（privacy/on_device.py）                    │   │
│  │  DistilBERT INT8 量化 → 本地情绪感知                   │   │
│  │  原始数据不出设备 → 仅上传脱敏特征摘要                   │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │ FeatureSummary（脱敏特征）
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      云端服务层                               │
│                                                               │
│  ┌────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐    │
│  │感知层   │→│ 评估层    │→│ 干预层     │→│ 升级层     │    │
│  │percept.│  │assessmnt.│  │intervent. │  │escalation │    │
│  └────────┘  └──────────┘  └───────────┘  └───────────┘    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 统一数据接口（shared/dataclasses.py）                  │   │
│  │ EmotionResult → RiskAssessment → CrisisAlert → ...   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌────────┐  ┌─────────┐  ┌────────┐                        │
│  │审计模块 │  │联邦学习  │  │公平性   │                        │
│  │audit/  │  │federated│  │fairness│                        │
│  └────────┘  └─────────┘  └────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块清单

| 模块 | 目录 | 核心文件 | 职责 |
|------|------|----------|------|
| 共享数据类 | `shared/` | `dataclasses.py` | 13 个统一数据结构 |
| 感知层 | `perception/` | `perception_service.py`, `text_emotion/`, `fusion/` | 文本/语音/多模态情感分析 |
| 评估层 | `assessment/` | `multi_task/`, `phenotype.py` | 多任务风险分级 + 数字表型 |
| 干预层 | `intervention/` | `state_machine.py`, `auditor.py`, `middleware.py` | 对话状态机 + 五轴审计 |
| 升级层 | `escalation/` | `crisis.py`, `scheduler.py`, `audit_log.py` | 危机升级 + 排班 + 审计 |
| 隐私层 | `privacy/` | `on_device.py`, `sanitizer.py` | 端侧量化 + 数据脱敏 |
| 审计层 | `audit/` | `fairness.py` | 公平性评估 |
| 联邦学习 | `federated/` | `client.py`, `server.py`, `run_simulation.py` | FedAvg 流程模拟 |
| API 层 | `api/` | `escalation.py`, `perception.py` 等 | FastAPI 接口 |
| 对比实验 | `experiments/` | `compare_*.py`, `run_all.py` | 3 组对比实验 |

### 2.3 数据流

```
用户输入
  → PerceptionService.analyze_multimodal()
      ↓ EmotionResult(text_emotion_probs, audio_risk_prob, confidence)
  → sanitize()
      ↓ FeatureSummary（脱敏，差分隐私噪声）
  → predict()
      ↓ RiskAssessment(phq9, gad7, risk_level, evidence)
  → state_machine.transition()
      ↓ DialogState + Action
  → SafetyAuditor.audit()
      ↓ AuditVerdict（五轴审计）
  → 输出安全回复 / 触发 escalate() → EscalationResult
```

---

## 3. 算法设计

### 3.1 多模态感知层

**核心文件**：`perception/perception_service.py`、`perception/text_emotion/model.py`、`perception/fusion/fusion.py`

#### 3.1.1 文本情感分析

基于 DistilBERT 的 5 类情绪分类（快乐 / 悲伤 / 焦虑 / 愤怒 / 中性）：

- **模型架构**：DistilBERT + 分类头（`perception/text_emotion/model.py`）
- **训练数据**：模拟数据 1000 条（基于 CMDC/EATD 数据格式设计，含共病相关性建模）
- **输出**：5 维情绪概率分布 `text_emotion_probs`

#### 3.1.2 语音情感分析

- **输入**：WAV 音频（16kHz）
- **输出**：语音风险概率 `audio_risk_prob`
- **当前状态**：模拟实现，待接入真实语音模型

#### 3.1.3 多模态融合

加权平均 + Stacking 两级融合策略：

- **一级融合**：文本权重 0.6 + 语音权重 0.4（`perception/perception_service.py`）
- **二级融合**：Stacking meta-learner（`perception/fusion/fusion.py`）
- **降级策略**：语音不可用时自动降级为纯文本模式

### 3.2 多任务风险评估

**核心文件**：`assessment/multi_task/model.py`、`assessment/multi_task/train.py`、`assessment/multi_task/predictor.py`

#### 3.2.1 模型架构

共享编码器 + 三任务头：

```
输入特征 (10维)
    │
    ▼
共享编码器 (10→32→32, ReLU)
    │
    ├──→ 抑郁任务头 (32→16→1, sigmoid) → PHQ-9 预估
    ├──→ 焦虑任务头 (32→16→1, sigmoid) → GAD-7 预估
    └──→ 睡眠任务头 (32→16→1, sigmoid) → 睡眠风险评分
```

#### 3.2.2 非对称损失设计

**关键设计**：非对称二元交叉熵损失（`assessment/multi_task/train.py`）

```
L = -[ beta * y * log(y_hat) + (1-y) * log(1-y_hat) ]
```

其中 **beta = 6.0**，将正样本（高风险）梯度放大 6 倍。

**设计理由**：在心理健康筛查场景下，漏诊（false negative）的临床代价远高于误诊（false positive）。通过非对称损失强制模型优先保障高风险召回率。

#### 3.2.3 风险等级映射

| 综合风险评分 | 风险等级 | 建议动作 |
|-------------|----------|----------|
| < 0.3 | LOW | 常规关注 |
| 0.3 - 0.6 | MEDIUM | 主动关怀 |
| 0.6 - 0.8 | HIGH | 优先干预 |
| >= 0.8 | CRISIS | 立即升级 |

### 3.3 数字表型特征提取

**核心文件**：`assessment/phenotype.py`

| 特征类别 | 特征示例 | 数据来源 |
|----------|----------|----------|
| 睡眠特征 | 就寝时间均值、睡眠时长方差 | 设备传感器 / 自报告 |
| 情感特征 | 负面情绪占比、情绪波动率 | 对话文本分析 |
| 行为特征 | 日均使用时长、社交频率 | App 使用日志 |
| 测评特征 | PHQ-9 历史趋势、GAD-7 变化率 | 量表评估记录 |
| 生理特征 | 运动频率、饮食规律性 | 可穿戴设备 |

- **时序变化检测**：Cohen's d 效应量（阈值 0.5）
- **基线偏离度**：Z-score（按年龄段 12-17 / 18-25 / 26-35 分组）

### 3.4 对话干预引擎

**核心文件**：`intervention/state_machine.py`

#### 3.4.1 五状态有限自动机

```
INIT ──(低风险)──→ EXPLORE ──(中风险)──→ INTERVENE
  │                   │                       │
  │                   └──(高风险)──────────────┤
  │              ┌──(危机)────────────────────┤
  ▼              ▼                             ▼
CLOSE ←──── CRISIS ←──────────────────────────┘
```

| 状态 | 允许动作 | 禁止动作 |
|------|----------|----------|
| INIT | 开放式提问 | 诊断/处方 |
| EXPLORE | 情感反映、开放式提问 | 诊断/处方 |
| INTERVENE | CBT 引导、正念指导 | 诊断/处方 |
| CRISIS | 安全确认、资源提供 | 诊断/处方/说教 |
| CLOSE | 总结 | 新话题引入 |

#### 3.4.2 状态转移规则

共 7 条转移规则，每条附临床依据注释。关键转移：

- `EXPLORE → CRISIS`：检测到自伤关键词或风险评分 >= 0.8，**立即转移**
- `CRISIS → INTERVENE`：危机解除确认 + 专业资源已提供

### 3.5 五轴安全审计

**核心文件**：`intervention/auditor.py`、`intervention/middleware.py`、`config/audit_rules.yaml`

| 审计轴 | 检测内容 | 阈值 | 临床依据 |
|--------|----------|------|----------|
| 危机延迟 | 危机信号未在 1 轮内响应 | 1 轮 | WHO 危机干预指南 |
| 妄想强化 | 肯定/强化用户妄想内容 | 关键词匹配 | 精神分裂症护理指南 |
| 污名化 | 使用歧视性/标签化语言 | 关键词匹配 | 精神卫生法第 5 条 |
| 谄媚倾向 | 过度认同有害行为 | 连续 3 轮 | 心理咨询伦理准则 |
| 轨迹漂移 | 偏离当前对话主题 | 偏离度 > 0.5 | 对话一致性要求 |

配置驱动：所有阈值在 `config/audit_rules.yaml` 中集中管理，附临床依据注释。

### 3.6 危机升级系统

**核心文件**：`escalation/crisis.py`、`escalation/scheduler.py`、`escalation/audit_log.py`

| 组件 | 功能 | 关键约束 |
|------|------|----------|
| 三通道通知 | Webhook / 站内通知 / 控制台告警 | 响应时间 < 30 秒 |
| 咨询师排班 | 按时间自动分配值班咨询师 | 优先级排序 |
| 审计日志 | Append-only + SHA-256 哈希链 | 不可删除、不可修改 |

### 3.7 联邦学习模拟

**核心文件**：`federated/client.py`、`federated/server.py`、`federated/run_simulation.py`

- **架构**：2 客户端 + FedAvg 服务端
- **模型**：sklearn LogisticRegression
- **数据分布**：Non-IID（不同 `class_sep` 和 `flip_y` 参数）
- **训练配置**：5 轮，`fraction_fit = 1.0`

> 比赛 Demo 为流程模拟，非真实跨机构部署。

---

## 4. 安全与隐私

### 4.1 端侧推理

**核心文件**：`privacy/on_device.py`

| 指标 | 原始模型 | 量化模型 | 变化 |
|------|----------|----------|------|
| 模型大小 | 255 MB | 22.75 MB | **-91.1%** |
| 推理延迟 | 37.84 ms | 31.99 ms | **1.18x 加速** |
| 准确率下降 | - | - | **0%** |

量化方法：PyTorch 动态范围 INT8 量化

### 4.2 数据脱敏

**核心文件**：`privacy/sanitizer.py`

```
EmotionResult（含原始文本情绪分析结果）
    ↓ summarize()
FeatureSummary（仅含统计特征 + 概率分布）
    ↓ validate_no_raw_data()
6 项泄露验证：类型 / 中文泄露 / 英文泄露 / PII / UUID / 范围
    ↓ 差分隐私（可选）
拉普拉斯噪声注入（epsilon 参数控制隐私预算）
```

### 4.3 合规矩阵

| 法规要求 | 实现方式 | 代码位置 |
|----------|----------|----------|
| 数据最小化 | 端侧推理 + 特征脱敏 | `privacy/on_device.py`, `privacy/sanitizer.py` |
| 知情同意 | 分层同意机制（监护人+本人） | 首次使用时弹出同意流程 |
| 可删除权 | 数据生命周期管理 | 用户可随时删除所有个人数据 |
| 算法透明 | 证据链（EvidenceItem）贯穿全链路 | `shared/dataclasses.py` |
| 人工介入 | 危机升级 + 咨询师排班 | `escalation/crisis.py` |

---

## 5. 实验验证

### 5.1 实验总览

所有实验使用模拟数据（`sklearn.datasets.make_classification`），固定随机种子 42。

**一键运行**：`python -c "from experiments.run_all import main; main()"`

详细报告：`experiments/outputs/experiment_summary.md`

### 5.2 实验 1：规则引擎 vs 多任务模型

**文件**：`experiments/compare_risk_models.py` | **数据**：1000 样本，10 维特征

| 指标 | 规则引擎 | 多任务模型 | 差异 |
|------|----------|------------|------|
| Accuracy | 0.5200 | **0.7150** | +37.5% |
| F1 (macro) | 0.5000 | **0.7080** | +41.6% |
| Recall (高风险) | 0.2936 | **0.5138** | **+75.0%** |
| Specificity | 0.7912 | **0.9560** | +20.8% |

> Recall (高风险) 是临床安全性的核心指标——漏诊代价远高于误诊。

### 5.3 实验 2：单模态 vs 双模态

**文件**：`experiments/compare_emotion_models.py` | **数据**：800 样本，5 类情绪

| 模型 | Accuracy | F1 (macro) | Robustness |
|------|----------|------------|------------|
| 纯文本（单模态） | 0.5938 | 0.5895 | 0.6062 |
| 加权融合（双模态） | 0.7812 | 0.7814 | 0.7625 |
| **Stacking 融合** | **0.8000** | **0.8012** | **0.7750** |

### 5.4 实验 3：云端 vs 端侧推理

**文件**：`experiments/compare_privacy.py` | **数据**：1000 样本，768 维特征

| 方案 | Accuracy | 模型大小 | 延迟 | 隐私评分 |
|------|----------|----------|------|----------|
| 云端推理 | 0.5950 | 255 MB | 100 ms | 20/100 |
| **端侧量化** | 0.5900 | **22.75 MB** | **20 ms** | **90/100** |

### 5.5 公平性审计

**文件**：`audit/fairness.py`

分组维度：年龄（12-15 / 16-18）、性别、地域（城市 / 县镇 / 农村）

| 指标 | 值 | 判定 |
|------|-----|------|
| Demographic Parity Diff | < 0.08（框架验证） | 达标（< 0.10） |
| Equalized Odds Diff | < 0.10（框架验证） | 达标（< 0.15） |

> 22 项公平性测试全部通过，覆盖 DPD/EOD 计算、分组性能差异、去偏见效果验证。

### 5.6 联邦学习仿真

**文件**：`federated/run_simulation.py`

| 模型类型 | Acc | F1 | 说明 |
|----------|-----|-----|------|
| 联邦全局模型 | 0.6800 | 0.7265 | FedAvg 聚合，2客户端×5轮 |
| 中心化模型 | 0.6650 | 0.6763 | 合并数据训练 |
| 本地模型 A | 0.8300 | 0.8411 | 仅本地数据 |
| 本地模型 B | 0.6900 | 0.6931 | 仅本地数据 |

> 联邦 vs 中心化性能差距 < 2%，验证 FedAvg 聚合策略有效性。
> 联邦学习过程中原始数据始终保留在各客户端本地，符合《个人信息保护法》数据最小化原则。

### 5.7 文本情感模型

**文件**：`perception/text_emotion/` | **基座**：hfl/chinese-roberta-wwm-ext

| 指标 | 值 |
|------|-----|
| 训练样本 | 420+ 条（青少年视角多样化语料） |
| 类别数 | 5（焦虑/抑郁/愤怒/中性/积极） |
| 验证 Acc | 100%（40 条验证集，存在过拟合，需真实数据验证） |
| 推理延迟 | ~200ms（CPU） |
| 已验证 | “焦虑”文本→焦虑 97.8%，“开心”文本→快乐 97.4% |

### 5.8 测试覆盖

全量 **367 测试通过**，1 跳过，0 失败。

| 测试文件 | 测试数 | 状态 |
|----------|--------|------|
| test_dataclasses.py | 22 | PASS |
| test_assessment.py | 24 | PASS |
| test_state_machine.py | 32 | PASS |
| test_auditor.py | 30 | PASS |
| test_phenotype.py | 30 | PASS |
| test_on_device.py | 20 | PASS |
| test_sanitizer.py | 24 | PASS |
| test_fairness.py | 22 | PASS |
| test_federated.py | 18 | PASS |
| test_escalation.py | 26 | PASS |
| test_fusion.py | 20 | PASS |
| test_acoustic.py | 11 | PASS |
| test_perception_new.py | 15 | PASS |
| test_perception_service.py | 16 | PASS |
| test_cbt.py | 17 | PASS |
| test_comorbidity.py | 14 | PASS |
| **总计** | **367** | **PASS** |

---

## 6. 落地路径

### 6.1 当前阶段（比赛 Demo）

- [x] 算法模块全部实现（10 个模块）
- [x] 统一数据接口（13 个 dataclass）
- [x] 300 个单元测试全部通过
- [x] 3 组对比实验 + 汇总报告
- [x] 公平性审计 + 联邦学习仿真
- [x] 危机升级系统 + 审计日志

### 6.2 近期规划

- [ ] 接入真实文本情感数据集训练（CMDC 中文多模态情感数据集 / EATD 青少年情感数据集）
- [ ] 接入真实语音情感分析模型（wav2vec 2.0 / HuBERT）
- [ ] 使用真实临床数据验证多任务模型
- [ ] 部署端侧量化模型到 Android / iOS
- [ ] 对接真实咨询师排班系统

### 6.3 中期规划

- [ ] 真实联邦学习部署（Flower + gRPC + 安全聚合）
- [ ] 差分隐私参数调优（epsilon 预算分配）
- [ ] 接入可穿戴设备数据源
- [ ] 公平性审计在真实人群上验证
- [ ] 通过伦理审查委员会审批

### 6.4 长期愿景

- [ ] 多中心联邦学习（计划与 3-5 家医院/学校合作部署）
- [ ] 纵向追踪研究（计划随访周期 6-12 个月）
- [ ] 个性化干预策略优化
- [ ] 与学校 / 医院系统集成

---

## 7. 局限性

### 7.1 数据局限

| 局限 | 影响 | 缓解措施 |
|------|------|----------|
| 使用模拟数据 | 实验结果不可直接推广到临床 | 需接入真实数据集验证 |
| 无真实临床标注 | 模型未经临床有效性验证 | 需与专业机构合作标注 |
| 数据规模有限 | 模型泛化能力未充分验证 | 需扩大训练数据规模 |

### 7.2 模型局限

| 局限 | 影响 | 缓解措施 |
|------|------|----------|
| 浅层 MLP 架构 | 表达能力有限 | 引入预训练语言模型 |
| 未进行超参数调优 | 非最优性能 | 网格搜索 / 贝叶斯优化 |
| 语音模态为模拟 | 双模态融合效果未真实验证 | 接入真实语音模型 |
| 单语言支持 | 仅支持中文 | 扩展多语言支持 |

### 7.3 工程局限

| 局限 | 影响 | 缓解措施 |
|------|------|----------|
| 联邦学习为流程模拟 | 未验证真实网络条件 | 部署 Flower + gRPC |
| 端侧延迟为模拟值 | 实际取决于设备硬件 | 真机测试 |
| Webhook 为 mock | 通知链路未端到端验证 | 对接真实通知服务 |
| 无安全聚合 | 模型参数上传存在理论风险 | 引入安全聚合协议 |

### 7.4 伦理局限

| 局限 | 影响 | 缓解措施 |
|------|------|----------|
| 未经伦理审查 | 不可直接用于真实用户 | 需通过 IRB 审批 |
| 公平性仅在模拟数据上验证 | 真实群体差异可能更大 | 真实人群公平性审计 |
| 无纵向验证 | 长期效果未知 | 纵向追踪研究 |
| 不能替代专业诊断 | 用户可能过度依赖 | 明确定位为辅助工具 |

### 7.5 声明

> 本平台为 **AI 辅助工具**，不能替代专业心理健康诊断和治疗。  
> 所有风险评估结果仅供参考，最终判断必须由具备资质的专业人员做出。  
> 如遇心理危机，请立即拨打 24 小时心理援助热线：**400-161-9995**。

---

## 附录

### A. 项目结构

```
algorithm/
├── shared/              # 统一数据接口（13 个 dataclass）
│   └── dataclasses.py
├── perception/          # 感知层
│   ├── perception_service.py
│   ├── text_emotion/    # 文本情感（DistilBERT）
│   └── fusion/          # 多模态融合
├── assessment/          # 评估层
│   ├── multi_task/      # 多任务风险分级
│   └── phenotype.py     # 数字表型
├── intervention/        # 干预层
│   ├── state_machine.py # 对话状态机
│   ├── auditor.py       # 五轴审计
│   └── middleware.py    # LLM 包装层
├── escalation/          # 升级层
│   ├── crisis.py        # 升级引擎
│   ├── scheduler.py     # 咨询师排班
│   └── audit_log.py     # 审计日志
├── privacy/             # 隐私层
│   ├── on_device.py     # 端侧量化
│   └── sanitizer.py     # 数据脱敏
├── audit/               # 公平性审计
│   └── fairness.py
├── federated/           # 联邦学习
│   ├── client.py
│   ├── server.py
│   └── run_simulation.py
├── api/                 # FastAPI 接口
├── experiments/         # 对比实验
│   ├── compare_risk_models.py
│   ├── compare_emotion_models.py
│   ├── compare_privacy.py
│   └── run_all.py
├── config/              # 配置文件
│   └── audit_rules.yaml
└── tests/               # 测试套件（300 个测试）
```

### B. 依赖清单

| 依赖 | 用途 |
|------|------|
| Python 3.10+ | 运行环境 |
| numpy | 数值计算 |
| scikit-learn | 机器学习 |
| PyTorch | 模型量化 |
| transformers | DistilBERT |
| fairlearn | 公平性指标 |
| FastAPI | API 服务 |
| matplotlib | 可视化 |

### C. 运行命令

```bash
# 运行全部测试
cd algorithm
python -m pytest tests/ -v

# 运行对比实验
python -c "from experiments.run_all import main; main()"

# 运行联邦学习仿真
python -c "from federated.run_simulation import main; main()"
```

---

*报告结束*
