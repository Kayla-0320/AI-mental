# 新一轮算法产出接线说明（2026-11-17）

本轮把两件此前只在离线归档里的产出接进了算法服务：**round18 画质头**、**round19 知识图谱**。
两者都遵循同一条原则：先接通、默认不改变线上结论、证据不足的部分不冒充已验证。

## 一、画质头（round18 quality head）

| 项 | 内容 |
|---|---|
| 工件 | `algorithm/artifacts/quality_head/model.pkl`（17 KB）+ `manifest.json` |
| 模型 | `StandardScaler + Ridge(alpha=100)`，522 维 = 冻结 ChineseCLIP 视觉 512 + 10 个像素描述 |
| 训练/验证 | KonIQ-10k 公开 MOS（1–5），768 训练 / 192 验证选型 / 256 新图像测试 |
| 留出指标 | Spearman 0.7486、MAE 0.2676、低画质（MOS<2.5）AUROC 0.9022 |
| 代码 | `perception/quality_head.py`（加载、摘要校验、逐帧 MOS、阈值） |
| 接入点 | `perception/innovation.py` 的 `frame_quality()`，在拿到三帧视觉向量后打分 |
| 默认行为 | **shadow**：只把 MOS 写进本轮质量记录，不改变专家选择 |
| 强制模式 | `INNOVATION_QUALITY_GUARD=enforce` 时，最差帧 MOS<2.5 会退回文字专家 |
| 展示 | 创新抽屉「02 感知质量与自动纠偏」新增「画质模型 MOS（1–5）」 |
| 边界 | 公开图像人工评分，不是项目真人盲评；不用于情绪判断、诊断或安全阈值 |

实测（随机三帧 JPED，真实加载模型）：`available=True`、`quality_mos_worst=3.3963`、`quality_mos_mean=3.4122`、`quality_mos_low=0`，视觉照常参与。

## 二、知识图谱（round19）

| 项 | 内容 |
|---|---|
| 工件 | `algorithm/artifacts/support_knowledge/graph.json`（63 节点 / 56 边）+ `manifest.json` |
| schema | `aic.support_concepts.v2`（来源：WHO2020 / UNICEF×2 / ACL2021 / PROJECT / SKOS） |
| 审核状态 | 119 项全部 `pending`，专家审核 0 条 |
| 代码 | `perception/support_knowledge.py`（别名检索、否定/引述过滤、场景隔离、冲突隔离、预算整块截取） |
| 接口 | `GET /api/v1/knowledge/capabilities`、`POST /api/v1/knowledge/evidence` |
| 默认行为 | `SUPPORT_KNOWLEDGE_ENABLED` 未设为 `1` 时，`/evidence` 返回 503；`/capabilities` 只报规模与待审数 |
| 返回条件 | 默认只允许 `approved`（当前为 0 条）；研究使用须同时 `research=true` 且 `include_unreviewed=true` |
| 场景隔离 | 亲子来源（`parent_child`）不在默认一般语境返回；`pause` 直接弃权 |
| 边界 | 只返回概念与出处，不写用户事实、不参与风险/升级/安全阈值；`production_connected=false` |

## 三、故意没有接的部分

1. **round18 中性优先策略**：`round18_neutral.apply_policy` 需要“每专家概率矩阵”，当前 `FusionRuntime` 只导出融合后 logits 与路由权重；并且 round18 自己的 `promotion_decisions.json` 写的是 `no promotion`（整体区间跨零）。要接必须先扩 runtime 导出专家概率，再做新一轮验证，不能顺手改线上中性分布。
2. **round18 下一研究日误差带**：`round18_runtime.predict_day` 是 Beck 1–5 口径，平台 `personal-forecast.ts` 是 1–10 自评口径；半径 `0.2533` 是归一化后的 Beck 半径，直接搬会混量表。平台侧目前仍是无区间的 EMA 0.4，并标注 `validatedForProject:false`。
3. **图谱默认返回**：专家审核 0 条，默认关闭检索是 round19 协议的要求；接上的是“可显式开启的只读证据服务”，不是默认注入对话。
4. **`observe()` 的 HTTP 入口**：带视觉帧与知识命中的主路径仍只由 `api/intervention.py` 进程内调用，未开放外部 HTTP。

## 四、开关与回滚

| 开关 | 默认 | 作用 |
|---|---|---|
| `INNOVATION_QUALITY_GUARD` | `shadow` | `off` 不打分、`shadow` 只记录、`enforce` 可退回文字 |
| `SUPPORT_KNOWLEDGE_ENABLED` | `0` | 是否开放 `/knowledge/evidence` |
| `QUALITY_HEAD_ARTIFACT` | `artifacts/quality_head` | 画质头工件目录（测试用） |

回滚：删除 `algorithm/artifacts/quality_head` 并设 `INNOVATION_QUALITY_GUARD=off`；知识接口保持默认 `0` 即完全不暴露。

## 五、验证

- `pytest algorithm/tests/test_quality_head.py test_support_knowledge.py test_innovation_integration.py test_knowledge_moe.py` → **76 passed**
- `tsc --noEmit`（client）→ 退出码 0
- OpenAPI 已注册 `/api/v1/knowledge/capabilities`、`/api/v1/knowledge/evidence`、`/api/v1/innovation/capabilities`、`/api/v1/innovation/observation`
