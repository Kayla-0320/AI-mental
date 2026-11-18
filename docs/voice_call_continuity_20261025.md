# AI电话连续性与首声等待修复（2026-10-25）

用户反馈：声音中途停顿、断断续续，希望缩短延迟。本次修改集中在传输、排程与撤回语音，没有改动判停参数、音色、语速、声码器块长或安全判定规则。

## 已确认并修复的问题

| 问题 | 修改后的行为 |
| --- | --- |
| 首块耗尽后第二块才到，预缓冲资格还未消费，又多等300ms | 首块排程时就消费资格，每轮严格只生效一次 |
| 欠载条件 `base > now && max(now, base-overlap) > base` 永远不成立 | 保留本轮上一块的预计结束时刻，记录句内和跨句真实断档；换轮和主动停止重置 |
| 整轮打断传当前最大seq，只取消最后一段 | 整轮打断取消全部在途片段，服务端待合成队列不再保留前面的旧段 |
| 改稿撤回尾段时播放器调用stopAll，正确的前缀一起被掐掉 | 播放器按seq撤回尾段，前缀排程和后续PCM保持有效 |
| 取消多段只把水位线设到最小seq，较大seq的迟到start能复活旧段 | 水位线覆盖全部撤回序号，未知/撤回的start不能重建段；下一轮可以恢复 |
| HTTPResponse.read(8192)跨HTTP块等待凑满，短块被扣住 | 用read1交付已到达数据；跨块的半个int16样本保留到下一块 |
| 已过闸的完整句子仍等待350ms的计划宽限 | 足够长、以句末标点结束的完整句子立即送合成；短语按原宽限继续积累 |
| 定稿短句各起一次合成，重复起声和停顿 | 合并相邻定稿片段，每批不超过100字；文字、顺序和号码过滤保留 |
| 回复结束后宽限定时器仍能触发，跨轮读到新的缓冲区 | 正常结束前兑现剩余文本；结束/失败清计时器和缓冲；迟到回调不能驱动其他轮次 |

这里消除的是额外等待：完整句子已有时，可以省掉原本的350ms宽限；首块之后断档时，不再额外叠加300ms预缓冲。这两个数字不是整通电话的实测提速，也不代表模型能在零延迟内出声。

## 验证

- 前后端 `npm run build` 通过。
- 后端10个测试文件，共490项回归通过，包含对话流、TTS、ASR、危机分级、安全闭环、TAMER和面部状态。
- 6个前端回归脚本通过：连续性、预缓冲、音文对账、短单元守卫、撤回后错误分类、通话记录。
- 新增连续性脚本直接运行实际TtsPlayer/TtsStreamSession/TtsChannel，使用可控音频时钟。改动前4种缺陷均能复现失败，改动后通过。
- 新增HTTP测试运行本地真实HTTP分块服务器：服务端等待客户端消费第一块后才发送下一块。旧read(n)失败，新read1通过；不规则字节边界下PCM样本逐个一致。

复测入口：

```powershell
node server/node_modules/tsx/dist/cli.mjs scripts/verify_call_audio_continuity.ts
node server/node_modules/tsx/dist/cli.mjs scripts/verify_prebuffer_gate.ts
node server/node_modules/tsx/dist/cli.mjs scripts/verify_tts_text_ledger.ts
node server/node_modules/tsx/dist/cli.mjs scripts/verify_short_unit_guard.ts
node server/node_modules/tsx/dist/cli.mjs scripts/verify_tts_stop_no_audio.ts
node server/node_modules/tsx/dist/cli.mjs scripts/verify_call_transcript.ts
```

## 当前验证边界

本次检查时本机模型服务未运行。`node scripts/run-tts.mjs --check` 报可提交内存余量7557MB，低于项目现有启动下限12288MB，因此没有强行加载模型，没有重启用户服务或修改系统内存配置，也没有测出新的整通首声延迟或完成真人试听。

已完整取消浏览器与算法端排队的旧片段；CosyVoice微服务当前没有可靠的底层即时中止接口，已经进入模型计算的旧请求仍可能继续占用合成锁。该项以及真实GPU供音速度仍需在可运行环境下复测，不能据本次回归声称所有模型侧卡顿都已解决。
