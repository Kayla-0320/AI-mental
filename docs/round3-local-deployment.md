# 第三轮本地接入与验证

这是本地研发部署说明，未发布公网。先安装项目原有依赖及 `algorithm/requirements-round2.txt`，保留现有数据库配置。

## 模型接入

模型和标签必须匹配，不将三类极性、五类情绪、八类认知与量表分数互相替代。完整训练产物放在 `D:/Develop/AIC/_国赛优化/round3/model/`，运行用选择集选定的文字模型复制到 `algorithm/artifacts/round3_fusion/`。

Python配置：

```text
ROUND2_RESEARCH_TOKEN=<专用随机令牌>
ROUND3_FUSION_ARTIFACT=<选定文字模型工件目录>
ROUND3_VLM_ENCODER=D:/ai/models/encoders/chinese-clip-vit-base-patch16
ROUND3_ENCODER_DEVICE=cpu
```

Node配置相同研究令牌，并显式启用：

```text
SUPPORT_AGENT_MODEL_REFERENCE=1
ALGORITHM_API_URL=http://127.0.0.1:8000
ROUND2_RESEARCH_TOKEN=<与Python一致的令牌>
SUPPORT_AGENT_MODEL_SHA256=<选定模型的checkpoint_sha256>
SUPPORT_AGENT_MODEL_VERSION=<选定模型的task>
```

未配置研究令牌或未启用模型参考时，业务维持本人确认与目标规则，不将缺失结果当模型成功。先实际调用模型完成预热，再检验跨服务延迟。CPU首次加载时间不等于热推理时间。

## 后台与退出

进入疗愈空间，在本人总开关之外单独开启后台建议，并设置界面提供的频率与安静时段。worker只产生可接受/跳过的站内建议，不自动播放工具。关闭、撤销出处、删除账号或服务重启后，应按组件报告中的恢复与取消用例复核。

## 隔离测试库

真实数据库验证仅使用 `aic-round2-postgres-20261115` 容器，绑定 `127.0.0.1:55432/aic_round2`。脚本显式检查隔离库地址，不连接用户业务数据库。测试使用自编用户，结束清理测试用户并停止容器。

原始问卷、公开数据和专家背书记录的属性不因部署改变。带不明题号的中国两波表格必须先通过任务定义审核器，不能为了获得预测结果自动猜量表和随访间隔。

## 本轮实际跑通的本地验证

- 选定文字模型已复制到 `algorithm/artifacts/round3_fusion/`（`checkpoint_sha256 = 0bf751e7…`，`text_only_calibration` n=703、温度 1.3126）。
- Python 服务以 `uvicorn main:app --host 127.0.0.1 --port 8001` 实际启动，`/health` 与 `/api/v1/research/capabilities` 返回 200。
- `scripts/verify_round3_research.py`：真实工件下文字参考可用且标记校准，过期消息 `source_stale_or_future`，摘要不符 422，中文策略 `available=false`，无令牌 401。
- `scripts/verify_support_agent_model_bridge.ts`：隔离 PG16 中的本人修正证据 → Node provider → 真实 Python 模型，返回 `available=true`、任务与标签匹配、`artifact_sha256` 与部署固定值一致；改错固定值后返回 `null`；关闭 provider 后返回 `null`；`understand` 目标不触发工具。
- `scripts/verify_support_agent_round3.ts` 19 项真实 PG 用例、`verify_support_agent_lifecycle.ts` 生命周期用例已通过；`server` 的 `tsc --noEmit` 通过，`SupportAgentWorker` 已在 `app.ts` 的启动与关闭流程注册。
- 验证用容器 `aic-round2-postgres-20261115` 保持运行以便复算；不再需要时可停止，数据保留。
