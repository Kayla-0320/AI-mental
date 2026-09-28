# 文本流式对话（逐句上屏）设计说明

> 对应改造：`/smart-chat/stream`（Python）→ `/api/consultation/conversations/:id/messages/stream`（Node）→ 聊天页增量渲染。
> 语音侧的同类设计见 [`voice_call_plan.md`](./voice_call_plan.md) §4。

---

## 1. 一句话概括

**逐句上屏是为了体感，安全审计仍然拥有最终否决权。** 两者用一个 `revise`（整体替换）事件缝合。

---

## 2. 为什么要「先说、后校」

本项目的安全审计（`SafetyLoop` 五轴）是**整段回复级**的判据：它需要完整回复 + 对话历史 +
状态机上下文才能给结论（轴 4 谄媚看的是最近 10 轮的同意率，轴 5 轨迹漂移看的是对话走势）。

也就是说：**不存在"在吐第一个字之前就拿到审计结论"的可能。**

于是只有两个选择：

| 方案 | 后果 |
|---|---|
| A. 等审计完再整段显示 | 安全，但没有流式，首字延迟等于全文生成时间 |
| B. 逐字上屏 | 快，但**等于在审计之前就把未检查的内容交付给了用户** |
| **C. 先说、后校（本方案）** | 增量上屏保证体感；审计一旦否掉已生成内容就**整体改稿** |

方案 C 的代价必须写清楚：**已经显示出去的文字收不回来**（浏览器里可以替换，用户的眼睛不行）。
缓解靠三层叠加：

1. **句子级缓冲** —— 只有完整句子才上屏，闸门手上至少有一个语义单位可判。
   拿到"你"或"你可"这样的前缀无法判断任何事。
2. **增量闸门**（`_StreamingSafetyGate`）—— 复用 `audit_rules.yaml` 既有词法红线，命中即**停止生成**。
3. **定稿改稿** —— 整段审计的结论通过 `revise` 生效。

---

## 3. 事件契约（⚠️ 三处必须同步）

这组取值同时出现在三个地方，改动必须三处同步：

| 端 | 文件 |
|---|---|
| 产出 | `algorithm/api/intervention.py`（`ChatStreamEventType` 定义在 `algorithm/shared/dataclasses.py`） |
| 透传 | `server/src/services/algorithm-bridge.ts`（`ChatStreamEventName`，只做哑管道） |
| 消费 | `client/src/services/chatStream.ts`（`ChatStreamEventName`） |

`algorithm/shared/dataclasses.py` 里有一条单测护栏（`test_exactly_five_members`）会在新增事件时变红。

### 事件序列

```
user → meta → delta* → [revise] → done      # 正常（user 仅 Node 层产生）
user → revise → done                        # 算法层不可用，服务端回退到非流式
user → error                                # 生成失败，前端渲染可重试的失败态
```

| 事件 | 语义 | 消费端动作 |
|---|---|---|
| `user` | 服务端落库后的用户消息 | 替换本地乐观占位气泡。**只在 Node 层产生**，不属于算法层契约 |
| `meta` | 风险等级 / 对话模式 / 情绪概率 / `streaming` | 记录。⚠️ **不保证出现**（回退路径没有它），**调用方不得依赖它创建气泡** |
| `delta` | 回复增量 | **追加** |
| `revise` | 权威改稿 | **整体替换**。把改稿当增量拼上去会得到「不安全原文 + 安全改稿」，是最坏结果 |
| `done` | 本轮结束，载荷与非流式接口同构，另有 `text` | 用 `text` 作为最终权威文本 |
| `error` | 失败 | `failed` 为真时按可重试失败态渲染 |

---

## 4. 三条设计约束

### 4.1 只有一份前处理和一份定稿

`smart_chat` 被拆成两段共享管线：

* `_prepare_chat_turn(request)` —— 步骤 1~2.9：感知 → 风险 → 危机快检 → 记忆召回 → 风格 → 状态机 → 场景检索
* `_audit_and_finalize(prep, request, llm_reply)` —— 步骤 3.5~5：记忆提取 → SafetyLoop 五轴审计 → 高风险强制升级

两个端点唯一的差别只剩「第 3 步怎么拿到 `llm_reply`」：一次拿全，还是边流边拼。
**定稿永远由 `_audit_and_finalize` 完成**，所以流式路径的最终文本与审计结论与非流式逐字段同源。

> 这个重构做了一次**逐字节等价性验证**：清空 `DASHSCOPE_API_KEY` 让两条路径都走确定性模板兜底，
> 对 10 条覆盖中性/难过/焦虑/回避/激动/高风险/危机/认知扭曲的输入比对完整响应 JSON，
> 结果完全一致（需固定 `PYTHONHASHSEED`，原因见 §6 已知问题 1）。

### 4.2 `high` / `crisis` 完全关闭增量流式

```python
streaming_allowed = prep.risk_level in ("low", "medium") and not prep.crisis_detected
```

高风险/危机轮次**一个 delta 都不发**。理由：升级流程必须**原子下发**，
逐字吐出的安全确认与热线会看起来像普通闲聊。

实测行为（危机输入）：
```
[ 61.2ms] META  streaming=False risk=crisis
[641.8ms] REVISE(整体替换) '你说的这些我很重视。…'
[642.0ms] DONE  requires_escalation=True
无 delta
```

### 4.3 闸门不新增任何安全规则

`_StreamingSafetyGate` 只复用 `audit_rules.yaml` 既有的 `stigma_rejection` 词表
（含 `teen_specific` 扩展词表），把原本在整段审计时才生效的词法红线**提前到句子级**。
按 AGENTS.md「自动化流程不得自行修改安全阈值或升级条件」，改变的是**生效时机**，不是规则本身。

单测 `test_terms_are_exactly_the_existing_yaml_union` 是这件事的机器化护栏：
任何人往闸门里硬编码新词，测试都会红。

**已知缺口**（docstring 原文承认）：`audit_rules.yaml` **没有药物词法轴**，而 AGENTS.md 明令禁止
药物建议。补这一轴属于**新增安全规则**，需人工审核后方可合入，因此代码里没有擅自添加。

---

## 5. 降级路径

| 失败位置 | 行为 |
|---|---|
| 浏览器 → Node（网络/HTTP） | 客户端回退到非流式接口。**但仅当服务端尚未回传 `user` 事件时**才重发 —— 否则会把用户消息重复写一遍 |
| Node → Python（流式） | 服务端回退到非流式 `smartChat` → 再不行退到 `aiService.counselingChat`，用 `revise` 把已上屏内容整体替换 |
| Python 内部（LLM 流中断） | 用已生成的部分照常走整段审计，由审计决定最终文本，不让用户停在半句话上 |
| LLM 一个字都没拿到 | 与 `_call_llm` 完全同源的模板兜底 |
| 浏览器断开 | `AbortSignal` 一路传到上游 `fetch`，中止 LLM 调用 |

---

## 6. 已知问题（改造过程中发现，未擅自修改）

1. **`_generate_empathy_reply` 用 `hash(risk_level) % len(templates)` 选模板。**
   Python 的字符串 hash 逐进程随机化，所以**同一个风险等级在不同进程会选到不同模板**
   —— 与 AGENTS.md「固定种子 42，确保结果可复现」相冲突。
   影响范围仅限模板兜底（有 API key 时走不到）。修法是换成稳定哈希（如 `zlib.crc32`），
   但那会改变每个风险等级当前选到的模板，**属于行为变更，未擅自做**。

2. **危机热线曾经重复显示两遍。** 算法层 `SafetyLoop` 的危机资源卡
   （`audit_rules.yaml` 的 `crisis_resource_card`）本身已含三条热线，而
   `consultation.service.ts` 又无条件追加了一次。已改为幂等追加（`appendCrisisHotlines`）。

3. **冷启动 6.3 s。** 首次请求要加载感知模型 + 认知扭曲分类器 + 向量检索。
   生产/Demo 前必须**预热**（启动后先打一次 `/smart-chat` 或专门的 warmup 端点）。

---

## 7. 实测数字（真实千问，本机 127.0.0.1）

| 事件 | 冷启动 | 热 |
|---|---|---|
| `meta` | 6311 ms | **75 – 108 ms** |
| 第一个 `delta` | 6797 ms | **442 – 497 ms** |
| `done` | 6920 ms | **469 – 613 ms** |

首字 440–500 ms 的构成：`META` ~80 ms，**LLM 首 token ~370–390 ms**。

⚠️ 如实说明一个限制：项目 prompt 强制"每次回复控制在 1~2 句话"，所以 `qwen-turbo` 的流式
分块通常只有 **1–2 个 delta**（它按句而不是按字吐）。**对短回复，流式的观感收益有限**
（省下的是"等第二句生成完"的那段时间，实测 ~60–150 ms）；
真正拉开差距的是长回复，以及它为语音通话铺平的道路（语音必须逐句合成才能低延迟起播）。
