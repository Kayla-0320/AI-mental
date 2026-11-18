# 第三轮方向二：原始信号质量与中国青少年两波数据接入审核

日期：2026-11-15。代码范围为 `algorithm/data_quality/raw_signals.py`、第三轮数据脚本及测试。本轮没有招募或联系真人，没有提交数据申请、注册账号或接受受限数据协议。已有标记为 Synthetic 的问卷样例没有进入训练或效果评估。

## 1. 中国青少年两波数据：已下载不等于已可训练

来源为 [Xiao 与 Tian 的 Mendeley Data V1](https://data.mendeley.com/datasets/d8h6gbs4jk/1)，DOI `10.17632/d8h6gbs4jk.1`，公开元数据标记 CC BY 4.0。原表 `D:/Develop/AIC/data/public/chinese_adolescent_longitudinal_d8h6gbs4jk/green exposure data.xlsx` 的 SHA-256 为 `eeb680f188860ed54a1163c2f655886166c6c6e3241ee4bba267ccc14163555f`，与来源 metadata 完全匹配。

已实测的结构：

| 项目 | 结果 | 可支持的含义 |
|---|---:|---|
| 行、列 | 7,742 × 533 | 原表记录及字段数量，不能当独立纵向观测次数 |
| 年龄字段 | 9–19 | 表内年龄范围，不自动等同于平台12–17岁目标人群 |
| cohort | 两个取值 | 数值代码，具体招募/时间含义待字典核验 |
| 候选 `(cohort, code)` 唯一键 | 7,741 | 有一个键重复，两行内容不完全相同 |
| `wave2_followup=1` | 5,241 行 | 候选随访标记计数，编码语义尚未由字典独立确认 |
| `_1` / `_2` 成对字段 | 208 对 | 字段名称提供两波结构线索，不能据此自动命名量表 |
| `date2` 可解析范围 | 2024-09-18 至 2024-12-20 | 只有该日期列，不能推出基线/随访间隔 |

重复键应同时隔离两行，不能以 `keep_first` 随意保留一条作为确定事实。后续若确认身份语义，切分必须按参与者键进行；cohort、code、随访标记和随访日期不能直接成为预测基线特征。

### 词典与时间定义核查

已核查原表 ZIP 元数据：只有一个 Sheet1，没有 comments、问卷释义或 defined names；`docProps` 只有 WPS、创建/修改信息等技术属性。Mendeley API 只有一个 XLSX 文件，`articles` / `related_links` 为空。[DataCite DOI 元数据](https://api.datacite.org/dois/10.17632/d8h6gbs4jk.1) 只有版本关系，没有可确认本表编码的论文关系。

进一步按 DOI、完整文件名、作者与自然接触/绿地/心理和躯体症状等组合检索，截至本轮未找到能确认题号、计分或间隔的 primary codebook。没有向作者发邮件或提交申请。

`g1_1`、`g9_1`、`h1_1` 等抽查值为 1–4 的编码，不能仅凭“九题/七题”就命名为 PHQ-9/GAD-7，更不能擅自减1、求和或套临床阈值。作者其他论文采用某量表，也不能自动充当这个文件的逐列字典。

### 机器阻止未定义训练

已实现 `scripts/round3_data_longitudinal.py`，保持原表只读，输出逐列清单、来源摘要、重复、日期和编码示例，以及 JSON 任务契约。执行 `--require-trainable` 后已真实得到非零退出和 `training_permitted=false`。阻止原因包括：

1. 参与者身份语义未核实。
2. 原始问卷字典未核实。
3. 基线特征未定义。
4. 随访目标未定义。
5. 随访时间间隔未核实。
6. 缺失编码未核实。
7. 目标计分及单位未核实。

即便未来提供词典配置，审核器仍拒绝把 `_2`、`date2`、`wave2_followup`、参与者ID列作为基线特征，并要求每个特征的含义、基线可得性和出处位置。配置中的“已核实”是需要有原始来源支持的审核声明，不能由该软件自身证明题义正确。

结果在 `D:/Develop/AIC/_国赛优化/round3/data/longitudinal_audit.json` 和 `longitudinal_task_pending.json`。**本轮没有基于此表训练模型，也没有产生 PHQ、焦虑或临床风险预测结论；两波数据不称为长序列。**

## 2. 从特征故障处理推进到真实原始信号质量

新增原始质量函数返回统一的 `shared.RawSignalQuality`，含 `usable`、有界质量分、可观察指标、理由和限制。质量分为工程启发式分数，不是校准概率，也不是情绪强度或信任分数。

| 函数 | 输入 | 已实现的可观察证据 |
|---|---|---|
| `assess_audio_signal` | 归一化 PCM 或整型 PCM、采样率 | 时长、RMS、峰值、近静音帧比例、削波比例、DC偏移、频谱平坦度代理 |
| `assess_visual_signal` | RGB帧，可选逐帧时间/人脸存在标记 | 曝光、对比度、128px尺度的拉普拉斯方差、精确重复帧、时间单调性 |
| `assess_av_alignment` | 音视频时长，可选起点 | 容器时长差、起点偏差和元数据缺失 |

没有独立人脸存在标记时明确 `target_face_presence_unverified`；没有起点时明确 `av_start_offset_unverified`。容器时间一致不证明口型和说话内容同步，频谱平坦度也不等于真实 SNR。重复帧可能来自正常静止场景，不能直接当作摄像头故障或欺骗。

函数已交给模型进程用于全量 v2 原片编码时记录质量、mask 和理由，不重复下载其全量媒体。

## 3. 真实原片的受控故障回归

从第二轮保留的四个公开 CH-SIMS v2 原片中，使用 PyAV 实际解码 16kHz mono 音频、RGB 帧和时间元数据。原片均重新核对 SHA。以固定 seed 42 注入预先定义的严重故障：静音、强削波、白噪替换、全黑帧、全白帧、高斯模糊、重复帧、1秒音视频起点偏差。

八种条件中，每一种均有 **4/4** 原片触发预期质量代理；逐项二项分布 Clopper–Pearson 95%参考区间为 **[0.3976, 1.0000]**。不能把八种条件合并成32个独立参与者或32个独立来源；四个原片也不是总体随机样本。原始四片的工程质量分均为1，但仍保留目标人脸/起点未独立核验的标记。

结果只说明这些受控故障的代码链路可工作，**不证明现实中噪声检出率100%、低误报、主观视频质量准确率或真实语音口型同步识别**。原片没有独立质量金标，因此没有把原片上的标记统计称为误报率。阈值没有在这四个样例上调优。

详细记录含每种注入后的指标与理由：`D:/Develop/AIC/_国赛优化/round3/data/raw_signal_benchmark.json`。脚本为 `scripts/round3_data_signal_benchmark.py`。PyAV依赖由模型进程安装并共享，本进程没有重复安装或下载媒体。

## 4. EATD 三分类的标签口径审计

EATD 的三类标签来自**情绪诱发场景**（`negative/neutral/positive` 三个录音与转写文件），不是人工标注的真实情绪。本轮对语料做了只读审计（不训练、不改数据）：

| 项目 | 实测 |
|---|---|
| 被试 | 162 人（train 83 / validation 79） |
| 预期片段 | 486（每人三场景各一段） |
| 训练清单实际 | 484；缺少 `validation/25|positive`、`validation/79|positive` |
| `label.txt` | 是 20–66 的数值分（与 SDS 字段一致），**不是三类标签**；三类标签来自场景文件 |
| 同被试三场景文本重合 | 字符 3-gram Jaccard 均值 **0.0073**，即三段内容差异很大（不同问题引出不同话题） |
| 场景模板重合 | 三场景均无 ≥90% 被试共享的模板 n-gram |
| 音频时长 | 均值 16.6 s；1 条 <1 s、31 条 1–3 s、12 条 <2 s、275 条 ≥10 s |
| 被试目录名跨切分复用 | 58 个名字在两个切分中重复；实际索引用 `train/1` 形式的主键，**历史分组评测没有因人名重复而泄漏** |

进一步做"场景可分性"对照：只用字符 2–4 gram TF-IDF + 逻辑回归、按被试分组 5 折，484 条上 **accuracy 0.702 / macro-F1 0.614**。也就是说，**仅靠词面/话题差异就能达到约 0.70**；历史 RoBERTa 的 0.8196 高于它，但在这样的构造下**不能把 0.82 直接解释为真实情感识别能力**——它至少部分反映"这段录音属于哪个诱发话题"。

因此后续任何在此语料上的提升都应：先固定场景与被试分组，报告"同场景内"与"跨场景"两种评测，并避免把诱发场景准确率写成情绪识别准确率。证据：`data/eatd_label_audit.json`；脚本：`scripts/round3_eatd_label_audit.py`。
## 5. 测试与独立接口复核

本进程新增 **38 项**原始信号与两波数据契约测试通过。覆盖整型PCM/float PCM一致性、缺失/静音/削波/白噪代理、非法采样率、非有限信号、RGB尺寸/类型、曝光/模糊/重复帧、真实时间元数据、未知人脸状态、AV起点与时长，以及未来变量/ID/缺失规则/未定义题义的训练阻止。

另完成根节点第三轮接口的独立复核。发现并由根节点修复：角色前缀 `user:` 被误当英语内容证据而对空内容发布策略；空白消息ID；`use_context` 未验证；坏产物错误被当客户端422；将完整T/A/V校准状态直接套给仅文字输入。

独立测试 `algorithm/tests/test_round3_research_review.py` **17 项通过**，保留一条现有 Starlette/httpx 依赖弃用提示。测试验证：空/空白/标点/纯数字单条无英语证据时不可用；坏JSON、模式和类别等产物问题503；未知语言422；伪SHA和空白ID提前拒绝；只有与 text-only 缺失条件、独立校准分割、样本量和合法温度均匹配时才标记对应源域校准。测试中的编码器和分布为契约替身，不是新增模型效果或实际校准改善证据。

跨数据来源复核提醒已发送模型进程：v2 含 v1 原片，必须使用一致的 canonical source ID，不可仅加不同数据集前缀就认定无重合；MMEVerse 标注同样要按上游原片去重。来源视频独立也不保证演员或影视作品完全独立。

## 6. 尚未完成

1. 中国青少年两波数据的原始问卷逐列字典、计分方向、缺失编码与基线日期。缺这些前提不能做具心理学含义的任务级训练；机器阻止保持生效。**结构核验与"条目合计层"的随访预测可行性已在 [round3-longitudinal.md](<D:/Develop/AIC/AI-mental-main (2)/AI-mental-main/AI-mental-main/docs/round3-longitudinal.md>) 完成**：5,206 人两波完整，基线自评中等可预测（r≈0.51–0.56），但不优于"保持不变"。
2. 原始音视频的独立专家/人工质量金标、真实环境敏感度与误报率；本轮仅有受控故障回归。
3. 主动说话人与人脸目标绑定、口型-语音语义同步、精细 ASR 时间对齐、背景噪声和混响等学习型质量模型。
4. 质量分的学习与校准、外部设备/人群验证，以及证明质量处理带来稳定模型增益的完整消融。不能从故障代理触发直接推出分类改善。
5. 真正的中国青少年长期多模态随访、临床结果与干预因果效果。暂缓真人研究期间不开展新增招募；已有 Synthetic 示例不能替代这些证据。
6. EATD 的真实情绪人工标注与"同场景内"评测协议；当前只有诱发场景标签，且仅用词面基线量化了场景可分性，尚未做人工复核或跨场景迁移实验。

## 7. 复现命令

在仓库目录、既有项目虚拟环境中先设置 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。

```powershell
D:\Develop\AIC\.venv\Scripts\python.exe -X utf8 scripts/round3_data_longitudinal.py --source 'D:\Develop\AIC\data\public\chinese_adolescent_longitudinal_d8h6gbs4jk\green exposure data.xlsx' --output-dir D:\Develop\AIC\_国赛优化\round3\data --require-trainable
D:\Develop\AIC\.venv\Scripts\python.exe -X utf8 scripts/round3_data_signal_benchmark.py --samples D:\Develop\AIC\_国赛优化\round2\data\vlm_samples\samples.json --output-dir D:\Develop\AIC\_国赛优化\round3\data
D:\Develop\AIC\.venv\Scripts\python.exe -X utf8 scripts/round3_eatd_label_audit.py --corpus D:\Develop\AIC\data\eatd\EATD-Corpus --dataset-csv D:\Develop\AIC\data\training_runs\textcnn_sentiment\sentiment_dataset.csv --output D:\Develop\AIC\_国赛优化\round3\data\eatd_label_audit.json
D:\Develop\AIC\.venv\Scripts\python.exe -X utf8 -m pytest algorithm/tests/test_raw_signal_quality_round3.py algorithm/tests/test_longitudinal_contract_round3.py algorithm/tests/test_round3_research_review.py -q
```

第一条命令非零退出是符合当前研究定义的预期结果，不是训练失败后遗漏指标。
