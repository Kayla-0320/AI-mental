# 第三轮方向三：后台持续追踪、退出语义与模型联动

日期：2026-11-15。代码范围为 `server/src/services/support-agent.service.ts`、`support-agent-worker.ts`、`support-agent-context.ts`、`server/src/routes/support-agent.routes.ts` 的第三轮修改，以及 `scripts/verify_support_agent_round3*.ts`、`verify_support_agent_lifecycle.ts`、`verify_support_agent_model_bridge.ts`。本轮没有真实用户参与，**后台追踪与模型联动都是工程实现与可复核验证，不是疗效证据**。

## 1. 后台追踪：从"页面内提议"到"有租约的后台进程"

第二轮只在本人打开疗愈空间时生成提议。本轮新增独立后台进程，但仍受本人开关与时段约束：

| 机制 | 实际实现 |
|---|---|
| 总开关 | 后台建议默认关闭，且与使用本人确认内容的开关**相互独立**；关闭时不读取自评历史 |
| 存储 | 仍为每用户一条保留类型 `HealingSession.metadata`，`background` 子结构含 `enabled/intervalHours/quietStart/quietEnd/nextRunAt/lease/events` |
| 频率 | 只允许 6 / 12 / 24 小时三档 |
| 安静时段 | 默认 22:00–08:00（北京时间小时）；跨午夜区间同样生效；命中安静时段写入 `quiet_hours_skipped` 并把 `nextRunAt` 推到时段结束后 |
| 错过窗口 | 进程停机导致错过预定时刻时**跳过积压而不补发**，写 `missed_window_skipped` |
| 并发 | 每用户数据库租约 `{owner, token, until(120s), scheduledAt}`；只有拿到租约的实例能产出建议；12 个实例抢同一到期任务只有 1 份租约 |
| 幂等 | 同一租约重放只留一份站内通知；旧租约不能回写建议 |
| 失败退避 | 数据库不可用时按 `min(300000, 60s × 2^failures)` 退避，单实例 tick 不重叠，停止时不再认领新任务 |
| 单用户故障隔离 | 单个用户状态损坏只跳过该用户，不阻断其他用户的后台任务 |
| 产出边界 | 后台只产生**站内**、可接受/跳过的建议，**不会自动播放工具或执行动作** |
| 撤回 | 退出后台、拒绝/撤销出处、删除账号都会取消待处理建议并令旧提醒失效；删除账号级联移除后台状态与个人证据 |

`SupportAgentWorker` 的一次 tick 只做三件事：扫描候选用户、按用户尝试认领租约、执行已认领任务；`timer.unref()` 保证进程可退出，`stop()` 等待当前批结束。

## 2. 个人证据重算：不用浏览器自报的确认状态

`support-agent-context.ts` 每次都从**数据库中最近一条本人助手状态消息**重算证据，而不是相信前端：

- 只读取本人活跃会话、时间不晚于当前时刻的消息，`take 32`。
- 处于 `awaiting_reply`（最后一条不是助手消息）时状态为 `awaiting_reply`，不猜。
- 既有 `risk_level` 为 `high`/`crisis` 时返回 `existing_escalation` 与 `goal=listen`，**只读守卫，不改阈值、不替代原有升级链路**。
- 本人已拒绝（`declined`）时直接返回，不再提议。
- 每条证据必须满足其中之一：来源用户消息真实存在且 SHA-256 与字符区间（`start/end`）完全对齐；或来源为 `user_correction` 且能在 `feedback_log` 中找到 `correct` 记录并逐字匹配。伪造摘要、越界区间、错误来源一律丢弃并计入 `revoked`。
- 目标 `listen/understand/act` 只有在 `goal_source ∈ {user_feedback, explicit_message}` 时才被采纳。
- 本人负面反馈会改变默认选项，但显式偏好仍然优先。

## 3. 模型联动：三类情感参考，不等于五类情绪与风险

第二轮遗留的问题正是"不同量纲不能直接混用"。本轮的做法是**分变量、带出处、可核验**：

- Node 只在 `SUPPORT_AGENT_MODEL_REFERENCE=1` 且配置研究令牌时才调用 Python `/api/v1/research/observation`。
- 只发送**已确认的原话片段**（≤240 字）、其 SHA-256、来源消息 ID 与该消息的 `observedAt`；不发送真实用户 ID、不发送原始音视频。
- 超时 6 秒，响应体上限 16 KiB。
- 返回必须通过严格校验：`task_id = sentiment_polarity_3`、标签顺序恰为 `[negative, neutral, positive]`、`evidence_source` 与本地证据逐位一致、`observed_at` 与本地证据时间一致、48 小时内、概率和为 1、`calibration.status = calibrated`、`artifact_sha256` 与部署固定值一致。
- 任何一项不符即视为不可用，**不保存伪造概率、不据此自动启动工具**；模型参考只作为解释，不进入风险分级，也不进入五类情绪工作点。

### 3.1 本轮新增的端到端验证（真实 PG → Node → 真实 Python 模型）

此前这一链路只有契约替身。本轮实际跑通：

1. 隔离 PostgreSQL 16（`127.0.0.1:55432/aic_round2`）中创建自编用户与一条助手状态消息，证据来自本人修正记录。
2. Node `SupportAgentService.refreshModelReference` 通过真实 HTTP 调用本地 Python 服务，Python 端加载真实 Chinese-CLIP 与研究融合头并前向。
3. 返回 `available=true`、`task_id=sentiment_polarity_3`、`calibration=calibrated/text_only_source_temperature`、`artifact_sha256=0bf751e7…`，概率和为 1。
4. 把部署固定值改成错误摘要后，同一调用返回 `modelReference=null`（拒绝未被静默接受）。
5. 关闭 `SUPPORT_AGENT_MODEL_REFERENCE` 后返回 `null`，**没有伪造概率**。
6. 本人显式目标为 `understand` 时 `suggestionEligible=false`，**模型参考不会把"想被理解"变成自动工具执行**。

证据：[http-model-verification.json](D:/Develop/AIC/_国赛优化/round3/agent/http-model-verification.json)。探针文本为自编内容，不是准确率、也不是疗效。

## 4. 验证结果与边界

| 验证 | 范围与结果 |
|---|---|
| 业务契约 32 项 | 事务替身库；沿用第二轮用例并在第三轮重跑，包含默认关闭、幂等、越权、撤回、损坏状态、HTTP 认证 |
| 真实 PostgreSQL 19 项 | 隔离 PG16；12 实例抢占只有一份租约、退出后台后旧租约不能回写、重启后等租约过期恢复、后台建议只入站内、错过窗口跳过、安静时段跳过、本人修正按反馈日志与 SHA 核验、撤回实时失效、最新 listen 目标覆盖后台活动、既有 high/crisis 只读守卫、错误任务/未校准/来源/概率/工件拒绝、异步模型期间撤回后旧结果不回写、伪造修正日志不采信、负面反馈改变默认选项、worker 无数据库退避、损坏状态隔离、删除账号级联 |
| 生命周期 | 隔离 PG16（127.0.0.1:58431）、应用实际启动、worker 注册并运行、进程内 SIGTERM 处理器触发、HTTP 关闭；**外部 OS 信号投递未验证** |
| 独立复核 4 项 | 24 小时静默排程从 23 点顺延到次日 8 点而不永久困在夜间、白天与跨午夜区间边界、无旧 context 标识的提议也必须尊重新增 `understand` 目标、同一租约重放只留一份通知且证据撤回同步取消提议。该复核使用内存业务夹具，不主张 PG 并发 |
| 真实浏览器 | React → 鉴权 Express → PG → 后台 worker → 呼吸执行器；后台开关独立、离开页面后仍生成提议、返回后提议仍在、不自动执行工具、显式接受/取消、撤回后引用与模型参考同时移除、listen 目标生效、总退出持久化、390px 视口左 32/右 358 无横向溢出、无异常控制台错误 |
| 模型联动 | 见第 3.1 节，真实 PG + 真实 Python 模型 + 真实工件摘要校验 |

浏览器截图为 [离开页面后的持久提议](D:/Develop/AIC/_国赛优化/round3/agent/01-background-persisted-offer.png)、[已确认证据与模型参考](D:/Develop/AIC/_国赛优化/round3/agent/02-confirmed-context-model-reference.png)、[手机端后台控件](D:/Develop/AIC/_国赛优化/round3/agent/03-mobile-background-controls.png)。

## 5. 尚未完成

1. **真实纵向数据与长期运行**：后台进程在隔离库和测试用户上通过；没有真实用户连续数周运行，也没有生产环境的调度与保留期实测。
2. **提前预测心理变化**：后台只做"按本人自评与已确认证据的可跳过提议"，没有训练未来恶化预测，也没有证明预警提前量。
3. **通知渠道**：只有站内通知；没有推送、短信、邮件，也没有"未读多久升级"的语义。
4. **外部系统信号**：进程内 SIGTERM 已处理，操作系统级信号投递与容器编排重启未验证。
5. **模型参考的作用范围**：三类极性只作解释性参考；仍不是五类情绪、不是量表分数、不是风险等级，也没有进入任何自动决策。
6. **策略模型与工具调用**：ESConv 策略模型为英文描述任务，未接入工具选择；没有训练 ToolACE 一类工具调用。
7. **真人可接受性与帮助程度**：本轮无真实参与者，不能给出效果结论。

## 6. 复现命令

先启动隔离测试容器并确认其绑定 `127.0.0.1:55432/aic_round2`（脚本会拒绝其他地址）：

```powershell
docker start aic-round2-postgres-20261115
$env:DATABASE_URL='postgresql://aic_round2:round2_local_test_only_20261115@127.0.0.1:55432/aic_round2'
$env:ROUND2_RESEARCH_TOKEN='<本地研究令牌>'
$env:ROUND3_FUSION_ARTIFACT='<仓库>\algorithm\artifacts\round3_fusion'
$env:ROUND3_VLM_ENCODER='D:/ai/models/encoders/chinese-clip-vit-base-patch16'
$env:ROUND3_ENCODER_DEVICE='cpu'
$env:ROUND3_STRATEGY_ARTIFACT='<仓库>\algorithm\artifacts\round3_strategy'
Push-Location '<仓库>\algorithm'; D:\Develop\AIC\.venv\Scripts\python.exe -X utf8 -m uvicorn main:app --host 127.0.0.1 --port 8001; Pop-Location
```

另一个终端：

```powershell
$env:SUPPORT_AGENT_MODEL_REFERENCE='1'
$env:SUPPORT_AGENT_MODEL_SHA256='0bf751e75576f489c702304d35437397e2ac6b345cb90f3fc8dd7adff5321c55'
$env:SUPPORT_AGENT_MODEL_VERSION='ch_sims_v2_sentiment_polarity_3'
$env:ALGORITHM_API_URL='http://127.0.0.1:8001'
$env:ROUND3_AGENT_OUTPUT='D:\Develop\AIC\_国赛优化\round3\agent'
<仓库>\server\node_modules\.bin\tsx.cmd scripts/verify_support_agent_model_bridge.ts
<仓库>\server\node_modules\.bin\tsx.cmd scripts/verify_support_agent_round3.ts
<仓库>\server\node_modules\.bin\tsx.cmd scripts/verify_support_agent_lifecycle.ts
```

`verify_support_agent_round3.ts` 与浏览器脚本会自建并清理自编用户；验证结束后停止容器可保留数据用于复算。
