# 第三轮方向一：全量公开媒体接入与九臂真实 VLM 消融

日期：2026-11-15。代码范围为 `algorithm/experiments/round3_vlm_prepare.py`、`round3_moe_train.py`、`round3_moe_benchmark.py`、`algorithm/perception/knowledge_moe/` 的第三轮修改及测试。本轮没有招募真人、没有提交数据申请、没有接受受限数据协议，也没有把研究模型设为默认心理判断依据。

## 1. 这不是"没有数据"：全量公开媒体已真实接入

第二轮只对四个公开原片做了一次冻结前向。本轮实际下载并接入 CH-SIMS v2 **完整视频归档**，对全部监督片做了真实解码与冻结编码：

| 项目 | 实际结果 |
|---|---|
| 归档 | `THUIAR/MMLA-Datasets` 的 `CH-SIMSv2.0/Ch-simsv2_video.tar.gz`，11,588,045,887 字节 |
| 摘要 | SHA-256 `e2817c4841a74f9e73eed6cf3196442ff0245f999bdfc5f975dcf18e66348f1e`，与发布者 LFS 摘要一致 |
| 获取方式 | 匿名 HTTPS 分片下载，未登录、未申请、未签署额外协议 |
| 监督原片 | **4,400 条**，全部解出为独立 `.mp4`（约 11.0 GB），按清单 `id` 逐一核对存在 |
| 文本标签来源 | 公开镜像 TSV（train/dev/test），三份 TSV 摘要随冻结划分一起保存 |
| 样本数口径 | 公开镜像为 4,400 条；原论文报告的 4,402 条**不冒充**本地数量 |

冻结划分在编码前写入 `frozen_split.json`（SHA-256 `232ab70de952a78858afbf425b8ffab0cf69c5a7a313d6153071f0919a05f556`），使用 seed=42，按来源视频分组：

| 划分 | 样本 | 来源视频组 | 用途 |
|---|---:|---:|---|
| train | 2,555 | 92 | 拟合归一化与模型参数 |
| selection | 321 | 15 | 隐藏维与早停选择 |
| calibration | 703 | 15 | 只拟合温度 |
| test | 821 | 30 | 全臂选择锁定后一次性评测 |

**测试集来源限定为 v1 实验（video_0001–0060）与四片冒烟中从未出现的来源组**，因此本轮测试与第二轮的 472 条测试不重叠。来源视频互斥不保证演员、影片或节目互斥。

## 2. 实际特征与编码器身份

| 模态 | 维度 | 来源与算法 |
|---|---:|---|
| 文字 | 512 | 冻结 Chinese-CLIP ViT-B/16 文本塔，L2 归一化 |
| 视觉 | 512 | 同一冻结模型视觉塔，取片长 20%/50%/80% 三帧后均值，L2 归一化 |
| 声音 | 46 | **真实音轨**解码到 16 kHz 单声道后的手工 MFCC(20) 均值/标准差 + RMS、过零率、谱质心 |

- 编码器指纹 `clip_local_sha256:4decf46e2573aa96becc2a8731d9d9ddc7d6773c866ea20341c71488b2da4fe8`，覆盖权重、配置、分词与预处理文件。
- 全量编码在 CUDA 上完成；恢复后实际编码 285 秒，产物 `pipeline_version = clip3_mfcc46_raw_quality_saturation_v1`。
- 每条样本同时保存 `modality_mask`、`quality`、原始信号质量证据（曝光、模糊、重复帧、削波、近静音、时长一致性等）与媒体摘要。
- **不能说成完整时序视频理解或预训练语音模型**：视觉是三帧池化，声音是手工特征。

特征包：`D:/Develop/AIC/_国赛优化/round3/model/ch_sims_v2/{train,selection,calibration,test}.npz` 与 `metadata.json`；逐样本缓存 4,400 个 `sample_features/*.npz`。

## 3. 九臂消融：统一协议、先锁选择再读测试

`round3_moe_train.py` 预注册了九臂与全部选择规则（`preregister.json`），测试文件在**所有臂的选择与温度都锁定之后**才被读取：

- 九臂：text / audio / visual / concat / concat+KG / MoE / MoE+KG / Sparse-MoE / Sparse-MoE+KG。
- 隐藏维网格 {32, 64}，最多 80 epoch，patience 12，AdamW lr=1e-3、wd=1e-3，batch 64，类别加权交叉熵。
- 选择只用 selection 的 macro-F1，再比 NLL；温度只用 calibration 的 NLL 拟合。
- 知识为**用户背书**的六个概念（见第 6 节），融合头额外接收 32 维关系哈希 + 512 维节点语义。

### 3.1 最终测试（821 条 / 30 个来源组，clean）

| 模型臂 | 参数量 | 选择集 macro-F1 | 测试 accuracy | 测试 macro-F1 | 原始 ECE | 校准后 ECE |
|---|---:|---:|---:|---:|---:|---:|
| 文字 | 16,643 | 0.5667 | 0.6797 | 0.5314 | 0.0545 | 0.1026 |
| 声音 | 1,731 | 0.4296 | 0.4914 | 0.4040 | 0.0638 | **0.0378** |
| 视觉 | 33,283 | 0.5761 | 0.6248 | 0.4652 | 0.0850 | 0.0668 |
| 普通拼接 | 69,251 | 0.6248 | **0.7357** | 0.5657 | **0.0236** | 0.0901 |
| 拼接＋背书知识 | 104,131 | 0.6111 | 0.7065 | 0.5402 | 0.1141 | 0.0859 |
| MoE | 133,212 | **0.6310** | 0.7235 | **0.5744** | 0.0446 | 0.0941 |
| MoE＋背书知识 | 70,944 | 0.6256 | 0.7284 | 0.5465 | 0.0245 | 0.0715 |
| 稀疏 MoE | 51,260 | 0.6170 | 0.7004 | 0.5331 | 0.1574 | 0.0575 |
| 稀疏 MoE＋背书知识 | 176,672 | 0.6015 | 0.7028 | 0.5292 | 0.1176 | 0.0661 |

**主结果：MoE 比普通拼接高 +0.0086 macro-F1，但按 30 个来源组做 1,000 次配对 bootstrap 的 95% 区间是 [−0.0448, +0.0481]，跨零。这仍然不能宣称 KG/MoE 已有稳定增益。**

![九臂 VLM 消融](D:/Develop/AIC/_国赛优化/round3/figures/vlm_ablation_v2.png)

### 3.2 成对 bootstrap 全部结果

| 对比 | 差值 | 95% 区间 | 判读 |
|---|---:|---|---|
| MoE − 拼接 | +0.0086 | [−0.0448, +0.0481] | 跨零，不能宣称增益 |
| 稀疏 MoE − 拼接 | −0.0326 | [−0.0788, +0.0093] | 跨零，点估计更低 |
| 稀疏 MoE − MoE | −0.0412 | [−0.0762, −0.0003] | 区间不含零，**稀疏版更差** |
| MoE＋知识 − MoE | −0.0279 | [−0.0726, +0.0027] | 跨零，点估计更低 |
| 稀疏＋知识 − 稀疏 | −0.0040 | [−0.0365, +0.0269] | 跨零 |

**背书知识在两处都没有带来提升**，这与第二轮候选知识的结果方向一致。仍不能把这一结果推广成"任何专业知识增强都无效"；本轮知识只有六个概念、表征为哈希关系加节点语义，知识质量与路由设计都可能是原因。

### 3.3 全量媒体与 v1 作者特征的对照

![同一任务不同留出集](D:/Develop/AIC/_国赛优化/round3/figures/scale_comparison.png)

同一任务、不同留出集下，v2 全量媒体并不一致更好：文字 0.561→0.531、声音 0.395→0.404、视觉 0.467→0.465、拼接 0.611→0.566、MoE 0.581→0.574。差异同时包含数据来源、编码器（作者特征 vs Chinese-CLIP）、测试集和样本量的变化，**不能把差异归因为"数据变多"单一因素**。

### 3.4 温度校准在来源漂移下不必然改善

temperature 只在 calibration（15 个与 selection/test 不同的来源组）上拟合。结果：

- 改善：声音 0.0638→0.0378、视觉 0.0850→0.0668、拼接＋知识 0.1141→0.0859、稀疏 0.1574→0.0575、稀疏＋知识 0.1176→0.0661。
- **变差**：文字 0.0545→0.1026、普通拼接 0.0236→0.0901、MoE 0.0446→0.0941、MoE＋知识 0.0245→0.0715。

这是来源域漂移的直接证据：在另一批视频来源上拟合的温度不能保证在测试来源上降低 ECE。没有因此改用测试集重新拟合温度来"改善"数字。

## 4. 真实稀疏分派：确实省算子，但没有端到端加速

`round3_moe_benchmark.py` 用同一份权重比较稠密 top-2 参考实现与真实 gather/scatter 稀疏实现（CPU 单线程，仅融合头，不含编码器与网络）：

| batch | 稀疏实际专家行 | 稠密参考专家行 | 稠密 P50 | 稀疏 P50 | 输出一致 |
|---:|---:|---:|---:|---:|---|
| 1 | 2 | 7 | 0.629 ms | 0.628 ms | 是（atol 1e-5） |
| 64 | 128 | 448 | 1.021 ms | 1.253 ms | 是 |
| 256 | 512 | 1,792 | 2.147 ms | 2.324 ms | 是 |

![稀疏分派](D:/Develop/AIC/_国赛优化/round3/figures/sparse_dispatch.png)

稀疏分派在 batch=256 时把专家行计算降到稠密的 28.6%，但**墙钟时间没有下降**（该融合头太小，gather/scatter 与 Python 开销占主导）。因此本轮可以主张"实际稀疏执行已实现且与稠密 top-2 数值等价"，不能主张"资源收益"或"整体 VLM 延迟下降"。

同一基准还用四个真实原片验证了**独立文字前向与图文联合前向一致**：最大绝对差 0.0，四向量模长均为 1.0。

## 5. 本轮修掉的真实缺陷：文字入口在上游版本下不可用

`FrozenCLIPEncoder.encode_text` 原来直接调用 `ChineseCLIPModel.get_text_features`。在已安装的 transformers 4.57.3 中，`ChineseCLIPTextModel` 不再构建 pooler，该方法对 `None` 做投影并抛 `TypeError`：

```text
TypeError: linear(): argument 'input' (position 1) must be Tensor, not NoneType
```

后果不是"分数不好看"，而是**任何走文字单模态的真实调用都会失败**，包括实时模型参考接口。修复方式是不再调用该上游接口，而是复现 `ChineseCLIPModel.forward` 的文字分支（CLS 位置池化 → `text_projection` → L2 归一化），并新增回归测试禁止回退到 `get_text_features`。

修复后与图文联合前向逐位一致（四个真实原片最大绝对差 0.0），实时接口实测可用（见第 7 节）。这属于工程缺陷修复，不是模型效果提升。

## 6. 知识状态：按"专家审核已通过"处理

训练使用的知识由 `attested_graph()` 生成 `user_attested_knowledge.json`：

- 六个具名概念（积极感受、悲伤、恐惧、愤怒、人际事件、未来不确定性），每条都有出处与适用范围说明。
- **用户已确认该知识内容通过专家审核**，因此后续不再把"专家审核"列为阻塞项；工件中的 `review_status = user_attested`，并在 `review_attestation` 中记录确认来源、确认日期与节点内容摘要。
- 仓库内没有专业人员签署的文件、姓名或独立审核日期，因此对外只陈述"用户确认已通过专家审核"这一事实，不宣称持有签署材料。
- 默认加载模式仍然只认已审核条目；用户背书需要显式开启才可用于实验，含背书知识的工件拒绝业务推理。

独立检索忠实度评测仍未做，属于后续研究项，不影响知识可用性结论。

## 7. 实时接入链路

选择集只在"文字存在、音视频缺失"条件下重新比较了文字通路，选中 **text 臂**（selection 0.5667；独立 text-only 温度 1.313，n=703）。该臂已复制到 `algorithm/artifacts/round3_fusion/`，供 `/api/v1/research/observation` 使用。

用真实工件在进程内 HTTP 验证：

- 中文真实消息（含出处 SHA 与时间）返回 `available=true`，三类概率和为 1，校准状态 `calibrated / text_only_source_temperature`。
- 72 小时前的消息返回 `available=false`、`reason=source_stale_or_future`。
- 出处摘要不符返回 422；无令牌返回 401。

## 8. 尚未完成

1. **大型 VLM 端到端全参数训练**：本轮全部是冻结编码器 + 轻量融合头。端到端微调取决于显存与算力，尚未开始，也没有测过显存上限。本机可用显存为 **7.93 GB（RTX 5060 Laptop, sm_120）**，冻结 ViT-B/16 编码 + 小融合头可以正常跑完 4,400 条编码与九臂训练；全参数微调需要更大的显存或分片/梯度检查点方案，尚未评估。
2. **时序视频建模**：三帧池化不等于视频时序；未接入视频 Transformer 或时间卷积。
3. **预训练语音编码器**：声音仍是手工 MFCC/韵律，不是 wav2vec/HuBERT 一类模型。
4. **独立检索忠实度评测**：知识审核已按用户确认通过处理，但"检索到的知识是否被忠实使用"尚无独立评测。
5. **KG/MoE 的稳定增益**：本轮与第二轮均未证实；点估计方向不稳定。
6. **资源收益**：稀疏分派已实现，但没有端到端加速证据。
7. **外部泛化与分层公平性**：单一固定划分、seed 42；来源视频不等同演员/影片独立。
8. **目标人群验证**：电影/剧集片段的情绪极性不等于中国青少年的心理状态。

## 9. 复现命令

在仓库目录、项目虚拟环境中执行（先安装 `algorithm/requirements-round3.txt` 的 `av`）：

```powershell
D:\Develop\AIC\.venv\Scripts\python.exe -X utf8 algorithm/experiments/round3_vlm_prepare.py --tsv <v2-tsv-dir> --output D:\Develop\AIC\_国赛优化\round3\model\ch_sims_v2 --media-root D:\Develop\AIC\data\public\ch_sims_v2 --model D:\ai\models\encoders\chinese-clip-vit-base-patch16 --encode-only
D:\Develop\AIC\.venv\Scripts\python.exe -X utf8 algorithm/experiments/round3_moe_train.py --bundle D:\Develop\AIC\_国赛优化\round3\model\ch_sims_v2 --output D:\Develop\AIC\_国赛优化\round3\model\ch_sims_v2_ablation --knowledge-encoder D:\ai\models\encoders\chinese-clip-vit-base-patch16
D:\Develop\AIC\.venv\Scripts\python.exe -X utf8 algorithm/experiments/round3_moe_benchmark.py --bundle D:\Develop\AIC\_国赛优化\round3\model\ch_sims_v2 --artifacts D:\Develop\AIC\_国赛优化\round3\model\ch_sims_v2_ablation --model-dir D:\ai\models\encoders\chinese-clip-vit-base-patch16 --samples D:\Develop\AIC\_国赛优化\round2\data\vlm_samples\samples.json
D:\Develop\AIC\.venv\Scripts\python.exe -X utf8 scripts/round3_report_figures.py --results D:\Develop\AIC\_国赛优化\round3
```

训练脚本在已存在 `experiment.json` 时拒绝重跑，避免按测试结果改配方；`--encode-only` 要求全部监督原片已存在。

证据：`model/ch_sims_v2_ablation/experiment.json`、`model/ch_sims_v2_ablation/sparse_latency_and_text_equivalence.json`、`model/ch_sims_v2/metadata.json`、`model/ch_sims_v2/frozen_split.json`、`verification/research-http-round3.json`。
