# 第二轮方向二：真实配对数据接入与质量核验

日期：2026-11-15。范围为可复现的数据来源、标签域、分组、质量掩码与受控压力评测。没有改动既有危机规则或升级阈值。

## 已完成

### 作者来源的真实三模态数据

从 [THUIAR/MMSA 官方 README](https://github.com/thuiar/MMSA#2-datasets) 指向的公开 Google Drive 获取 `SIMS/Processed/unaligned_39.pkl`。完整文件 1,228,189,774 字节，SHA-256 为 `c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f`，与作者 README 完全匹配。匿名下载，未申请或接受任何受限数据协议；保留上游研究使用条件。

原文件保存在 `D:/Develop/AIC/data/public/ch_sims_v1/unaligned_39.pkl`，同目录保存作者 README 与下载回执。只有与固定作者 SHA 匹配才允许 pickle 加载，未知摘要在反序列化前拒绝。结果目录为 `D:/Develop/AIC/_国赛优化/round2/data/ch_sims_v1_strict/`。

真实配对样本总计 2,281 条，文本/语音/视觉原始特征尺寸分别为 `[N,39,768]`、`[N,400,33]`、`[N,55,709]`，各模态按作者有效长度均池化；padding 不进入均值。文本长度取 BERT attention mask。保存原话、样本 ID、来源组、原始切分、三模态及融合的分类/回归标签。

这是作者预提取特征上的研究输入，**不是 VLM 输出，也不是已验证的心理诊断标签**。三分类定义为 negative/neutral/positive 情感极性，不能替代平台的五类情绪或认知扭曲标签。

### 来源独立的主实验切分

原始作者切分为 train 1,368 / valid 456 / test 457。按样本 ID 的原视频前缀检查，60 个视频来源全部跨原切分；另有一个完整三模态池化表示的摘要跨切分。后者只能说明表示完全一致，不直接证明原媒体文件重复。

主实验使用 `GroupShuffleSplit(random_state=42)`：先留出 20% 来源组作 test，再在剩余组中留出 25% 作 valid。

| 切分 | 样本 | 来源视频组 | negative / neutral / positive |
|---|---:|---:|---|
| train | 1,273 | 36 | 702 / 184 / 387 |
| valid | 536 | 12 | 284 / 84 / 168 |
| test | 472 | 12 | 252 / 77 / 143 |

新切分的来源组交叉为 0，成对池化特征摘要交叉为 0。原始切分映射保存在 NPZ 的 `original_split` 字段；manifest 为逐样本 JSONL，元数据含源文件和各输出 NPZ 的 SHA、特征来源指纹及类别顺序。

**边界：** video ID 是目前可用的来源代理；这不保证演员、原影视作品或制作环境完全独立。不能把这次重新分组表述为所有语义和身份泄漏都已排除。后续成绩也不能直接与原作者切分的榜单分数比较。

### 保留真实模态差异

2,281 条数据的有效范围未发现非有限特征或非法长度，三模态可用率均为 100%，没有通过删除样本获得质量提升。

独立单模态情感标签与融合标签的不同率为：文本 **30.91%**、语音 **22.10%**、视觉 **21.88%**。这些不同保留原样，不触发改标签、删样本或“不信任”推断，也不能把它们解释成真实的掩饰比例。

### 可观察故障的质量契约

`algorithm/data_quality/quality.py` 检查缺失标记、非有限数、无效质量分、过期/未来观测、无效时间范围和片段错配，返回共享的 `ModalityQualityReport`：可用性、可靠度、理由及清零后的失效模态。没有基于“情绪不一致”执行纠偏的规则。

离线特征缺乏时间戳时，明确返回 `freshness_unverified` / `alignment_unverified`；不会伪称已经实测对齐。字符串、布尔、NaN 等不合法输入稳定失效或抛出 ValueError，不会直接泄漏 `math.isfinite` 的 TypeError。

`algorithm/data_quality/audit.py` 检查样本身份、重复 ID、标签域、类别范围、空切分、跨切分来源组和完整观测内容摘要。来源键必须在共享上游数据之间规范化；例如 MMEVerse 与 CH-SIMS 使用同一个原片时必须共用来源键。

### 受控压力实验

在同一真实来源分组数据上，每个模态单独训练 `StandardScaler + LogisticRegression(C=1,max_iter=1000,random_state=42)`，均仅拟合 train。固定概率等权融合，对比忽略/使用故障元数据；测试集不选模型、不调阈值、不拟合校准。所有压力条件固定 seed 42。

| 条件 | 忽略故障元数据 macro-F1 | 使用质量掩码 macro-F1 | 质量掩码后的覆盖率 |
|---|---:|---:|---:|
| 原始干净特征 | 0.5040 | 0.5040 | 100% |
| 50% 音频缺失 | 0.4356 | 0.5108 | 100% |
| 50% 视觉过期 | 0.4869 | 0.5039 | 100% |
| 50% 音频时间错配 | 0.5178 | 0.5108 | 100% |
| 50% 音频非有限值 | 0.4593 | 0.5108 | 100% |
| 每模态独立约 50% 缺失 | 0.3919 | 0.4578 | 88.35% |
| 三模态全部缺失 | 无有效情绪证据 | 拒绝情绪推断 | 0% |

这些是**真实特征上的模拟故障注入，并提供了已知故障元数据**。实验验证质量处理和失败行为，不证明现实中能自动检测全部噪声，也不能宣称各条件统一改善：时间错配条件实际没有提升。88.35% 覆盖率之外的样本未删除；JSON 同时报全样本和覆盖样本指标，全缺失时 train prior 仅作诊断评分，不能作为正常情绪推断交付。

详细结果：`D:/Develop/AIC/_国赛优化/round2/data/quality_benchmark.json`。

### 真实视频前向样例

从 [THUIAR CH-SIMS v2 公开镜像](https://huggingface.co/datasets/THUIAR/MMLA-Datasets/tree/main/CH-SIMSv2.0) 的归档流读取 4 个真实原片及对应 TSV 原话，共约 12MB，保存成员名、字节数、SHA、原始 ID 与标注。未全量下载 11.6GB 视频归档。

样例目录 `D:/Develop/AIC/_国赛优化/round2/data/vlm_samples/`，用途仅为方向一冻结 VLM 编码器的真实前向验证；与 CH-SIMS v1 训练严格分开。这 4 个示例不构成 VLM 分类准确率评估。

## 测试与复现

方向二的 **60 项测试通过**。覆盖缺失、过期、未来时间、错配、NaN、布尔/字符串输入、未核验时间、真实模态差异不被抹除、跨数据集来源重合、重复身份、空白来源键、标签域混用、padding 污染及未知 pickle 拒绝。单元测试使用自编工程输入，不是新的情绪数据标注。

PowerShell 中先设置：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
```

在仓库目录运行：

```powershell
D:\Develop\AIC\.venv\Scripts\python.exe -X utf8 scripts/round2_data_download.py --output-dir D:\Develop\AIC\data\public\ch_sims_v1
D:\Develop\AIC\.venv\Scripts\python.exe -X utf8 scripts/round2_data_prepare.py --source D:\Develop\AIC\data\public\ch_sims_v1\unaligned_39.pkl --output-dir D:\Develop\AIC\_国赛优化\round2\data\ch_sims_v1_strict
D:\Develop\AIC\.venv\Scripts\python.exe -X utf8 scripts/round2_data_benchmark.py --data-dir D:\Develop\AIC\_国赛优化\round2\data\ch_sims_v1_strict --output-dir D:\Develop\AIC\_国赛优化\round2\data
D:\Develop\AIC\.venv\Scripts\python.exe -X utf8 -m pytest algorithm/tests/test_data_quality_round2.py algorithm/tests/test_data_quality_prepare_round2.py -q
```

## 尚未完成

1. CH-SIMS v2 全量三模态特征、VLM 全量编码与多数据集来源去重接入。MER2024 等受限数据尚未取得授权，也没有绕过访问条件。
2. 原始音视频级的时间对齐、活跃说话人、噪声强度和遮挡质量人工金标；当前样本对应关系和有效长度来自作者文件，未独立重查每个原片。
3. 由原始信号学习真实噪声、错配和质量估计的模型；本轮压力实验是故障元数据已知条件。
4. 标签错误的专家复核、可能需要的再标注及独立一致性评测。本轮保留作者单模态和融合标签，没有自动将差异标签“修正”为一致。
5. 中国青少年现场数据与年龄、性别、录音设备等分层泛化/公平性验证；当前来源独立测试不能代替人群适配证据。
6. 多随机种子/多来源留出重复实验、外部独立鲁棒性验证及基于原媒体内容的近重复检查。当前单次固定划分结果属于首轮研究基线。
