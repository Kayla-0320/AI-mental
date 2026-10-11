# 文本 ↔ 语音对账链：为什么"说的和听到的不是一回事"

> 用户报告（2026-10-11）：**"AI 的回复和语音内容完全不匹配，而且语音还输出不全。"**
>
> 这两句落在**两条独立的链路**上，靠听感分不开。这份文档先给一条能自己指出
> "第一处不一致在哪一层"的对账链，再按它查出来的问题逐个修。

---

## 0. 先纠正三处前提（有代码依据，别按错的时序去改）

| 你的前提 | 实际情况 | 依据 |
|---|---|---|
| `speak` 在**第一个 delta 之前**下发 | **相反**：先流式 `delta*`，等整段生成完才定稿并下发 `speak` | `algorithm/api/intervention.py` 的 `_chat_stream_events`：delta 在 LLM 循环里逐条 `yield`，循环结束后才 `_plan_streamed_reply` + `yield SPEAK`。真实顺序 `meta → delta* → speak{text,segments} → [revise] → done` |
| `start` / PCM / `end` 都带 seq 可关联 | **二进制 PCM 帧不带 seq**：客户端按"最近一个 `start`"归属；服务端每条连接是**严格串行的单 worker**（一个 `jobs` 队列 + 单循环） | `client/src/services/ttsStream.ts` 的 `onMessage` 用 `lastStartedSeq`；`algorithm/api/tts.py` 的 worker 逐条取任务、逐块推送 |
| 旧音频可能混入新一轮 | **成立，而且真实存在** —— 取消是异步的，取消前已上路的 PCM 会和新一轮的 `start` 撞车 | 已修，见 §3 |

由此得到的结论：**`speak` 不可能"先到"**，所以 350 ms 的宽限期只覆盖"短回复"；
长回复必然先走 delta 合成。真正要守的不变量不是"计划优先"，而是
**"同一轮只有一个合成来源"** —— 否则同一段文本会被送两遍。

### 顺带查出的一个真 bug：同一轮被合成两遍

改之前的代码路径（`useVoiceCall`）：

```
第一个 delta 到达 → 扣住 350ms 等计划 → 350ms 内计划没来（长回复必然如此）
                  → flushByDelta 把扣住的文本送合成
后续每个 delta     → 直接送合成（宽限定时器已置 0，走的是"边到边送"分支）
……
整段生成完        → 服务端下发 speak{text,segments} → 客户端**再无条件地**把整份计划
                    的每一段又入队一次（没有任何"本轮已经用 delta 合成过"的检查）
```

结果：**同一段回复被合成并播放两遍**（第二遍是定稿版）。这与"文字和语音不匹配、
语音输出不全"的听感完全对得上（听到的是两份略有差异的副本，字幕只有一份）。
现在由 `SpeakPlanMode` 状态机从结构上堵死（§2）。

---

## 1. 对账链（先让数据指出是哪一层）

四个检查点，代码里都落成了数据（`client/src/services/voiceLedger.ts`）：

```
A shown   页面最终显示的文本          useVoiceCall 的 aiLines / subtree.full
B plan    SSE speak 的 text/segments  onSpeak
C said    实际发出的每个 say（按序）    speakUnit（含被有意跳过的片段）
D spoken  实际出声的 spoken_text       TtsChannel.onStart → 播放器真的出声
```

逐层比较，**第一处不一致就是第一优先修复点**：

| 比较 | 说明什么 | 去哪改 |
|---|---|---|
| A ≠ B | 字幕 / `revise` 把已播出的说法悄悄换掉了 | 字幕与 revise 策略（§4） |
| B ≠ C | 分段、过滤（号码/标点）、重复入队、双路合成 | §2、§5 |
| C ≠ D | seq 串单 / 回包错配（音频串到别的片段） | §3 |
| D 正确但仍听着断 | 播放队列与调度 | `TtsPlayer.underrun`（见 `voice_call_audio_stutter.md`） |

**有意跳过的片段（号码只上屏、纯标点不合成）与术语规范化单列为"预期例外"** ——
否则对账链天天误报，最后没人看。

看到的地方：

* `?debug=1` 面板新增「对账·轮次 / A 显示 / B 计划 / C 发出 / D 出声 / 跳过 / 不一致」；
* 每轮结束时若有不一致，控制台打一条 `[通话][对账] 第 N 轮：…` 并附四份原文。

回归（不需要浏览器）：`scripts/verify_tts_text_ledger.ts`。

---

## 2. P0：`speak` 成为唯一的正常合成来源

`SpeakPlanMode = 'pending' | 'speak-plan' | 'delta-fallback'`（**每轮一份，不可反复切换**）：

| 状态 | 谁驱动合成 | 转移条件 |
|---|---|---|
| `pending` | 谁都不驱动，delta 只扣在缓冲里 | 收到合法计划 → `speak-plan`；宽限期到点 → `delta-fallback` |
| `speak-plan` | `speak.segments`（唯一来源），扣住的 delta 直接作废 | 本轮结束 |
| `delta-fallback` | delta 逐条 | 本轮结束 |

三条附加规则（都落成代码 + 注释）：

1. **计划必须自洽**：`segments` 拼接（按空白归一化）必须等于 `text`，
   否则**拒收**并记录原因（`planJoinMatchesText`）。不静默丢弃 —— 静默会让
   350 ms 回退掩盖真正的问题。
2. **已进入 `delta-fallback` 就不再把计划入队**（记录"计划迟到"）。
   但可以做一件有价值的事：若定稿文本是"已发文本的前缀"（**删尾**），
   用 `unitsToCancelForPlan` 算出越界的**第一段 seq**，把多余尾段撤回来
   （`stopFrom`）—— 于是"听到的"回到定稿文本，而不是多念一句。
   跨边界那一段**保留**（宁可多念半句，也不吞字）。
3. **delta 在 `speak-plan` 之后永久不再送**（既有 `spokeFromPlanRef` 保留）。

---

## 3. P0：取消后的**残留隔离**（旧音频混入新一轮）

取消是异步的：`cancel` 帧发出去之后，服务端队列里/正在发的 PCM 还会到达，
而**二进制帧不带 seq**。一旦旧帧与新一轮的 `start` 撞车，旧音频就会被当作
新一轮的语音播出来 —— 这正是"字幕与声音对不上、还缺一块"。

两道闸（`TtsStreamSession`）：

```
cancelledThrough   取消水位线：seq ≤ 水位线的 start/帧一律不再接收
awaitingFreshStart 排空等待：取消后必须先看到一个**水位线之上**的 start，才恢复接收二进制帧
```

同时新增读数（面板/回调 `onDiagnostics`）：`drainedFrames`（丢弃的残留帧数）、
`drainedBytes`、`staleStarts`（陈旧 start 数）。

### 串单检测（C ≠ D 的本地探针）

`start.spoken_text` 与**本 seq 请求的文本**不一致时：

* 若该文本恰好等于**另一个待播 seq** 的请求文本 ⇒ **疑似串单**：删掉本段
  （它的 PCM 会被 `!seg` 那一关丢掉），记 `crossTalk` 计数并打警告 ——
  宁可这一句没声音，也不要让用户听到"另一句话的声音配这几句字幕"；
* 否则按**术语规范化**处理（`CBT → 认知行为疗法` 是服务端有意为之），
  记 `normalizedText` 计数，照常播放。

> ⚠️ 术语规范化是有意的，所以这里**不能**一律判违约丢弃 —— 那会把正常回复全弄哑。

### 为什么**不**把客户端 `say` 也改成严格串行

服务端每条连接本来就是**串行 worker**（一个队列、一条循环），客户端再串行
只会把"下一句的合成"推迟到上一句结束之后，直接拖慢整轮出声 —— 而首声预算是
这个项目的硬闸门（S4）。所以这里守的是"取消后的那一小段窗口"，不是整条链路。

---

## 4. P1：`revise` 的语义 —— 已播报 vs 最终显示

**2026-10-11 用户拍板**：口径是「**每一次输出的内容都需要语音合成**」——
屏幕上有的字，耳朵里必须有。所以原先"只改字幕、不重念"的保守半边被替换成
**前缀对齐后补念剩余部分**（`ttsSplit.ts::planReviseSpeech`）：

1. 从第一段起，只要这一段接上去仍是新稿的前缀，就**保留**（它念的内容本来就是新稿的
   一部分，不必重念 —— **不要求"已经播完"**，见下面的回归）；
2. 遇到第一段"跨过分歧点"的，**从它开始撤回**；纯空白段直接跳过；
3. 补念 = 新稿去掉已保留前缀的剩余部分，走同一条 `speakUnit` 路径。

⚠️ **真机回归（2026-10-11，用户反馈"延迟比以前高"）**：早先版本要求保留段必须
"**已经完整播完**"，而 `revise` 在真机上紧跟在最后一个 delta 后面到达（实测 1.3 s，
此刻一句都还没播完）→ 于是**每一轮都退化成"全撤 + 整段重念"**，首声被推后一整个
合成周期。按前缀判断后，内容没变的一轮 `cancelFrom = null` 且 `tail = ''`，
**什么都不做**。

### 4.1 根因：服务端把"只差空白"判成了"改稿"

上面那条客户端回归之所以每轮都被触发，是因为**服务端每一轮都发 `revise`**。
实测（`/smart-chat/stream`，修复前）：

```
83ms meta → 1247ms delta…(6条) → 1826ms revise → 1826ms done
91ms meta →  773ms delta…(4条) → 1324ms revise → 1324ms done
```

两处判据都用了 `.strip()`（只去首尾），而模型输出里常带 `"  \n"`（换行 + 缩进），
定稿把**中间**的空白抹掉后 `.strip()` 仍然判"不一致"：

| 位置 | 旧判据 | 每轮的实际后果 |
|---|---|---|
| `_chat_stream_events` | `final.reply.strip() != shown.strip()` | **每轮都发 `revise`** → 客户端整体替换字幕 + 撤回在途合成 + 补念 |
| `_plan_streamed_reply` | `final_text != planned` | **每轮都拒收 `speak` 计划** → 客户端永远退回 delta 驱动 |

修法：新增 `_normalize_for_compare()`（去掉**所有**空白）并在这两处替换判据
（`algorithm/api/intervention.py`）。修后实测三次不同输入：
`meta → delta* → speak → done`，**无 `revise`**，且"delta 拼接 vs 终稿"归一化后
完全相同。回归测试：`tests/test_chat_stream.py::test_revise_is_not_emitted_for_whitespace_only_difference`
与 `::test_speak_plan_survives_whitespace_only_finalize`。

> 记住这条判别法：**"屏幕文本 vs 语音文本"的相等性判断一律按归一化比较**，
> 不能用 `.strip()`／`!=`。差一个换行就足够让整条合成链路重来一遍。

### 4.2 revise 语义（客户端）

台账口径：改稿轮的 `A≠B` 与 `B≠C` 都是**必然**（屏幕是新稿、语音来源是改稿前的计划、
C 里多了补念的尾段），两份对账都标成 `预期例外：…系审计改稿`，不冒充"对账失败"。
`unspokenTail` 仍为 null —— 屏幕上每个字都有语音了。

被打断 / 危机 / 挂断时，**未朗读的那一段**单独记账（`unspokenTail`）并在页面上
明确标注「这一段没有说话（本轮语音已停），只有文字：…」——
字一个不丢（仍是既有"内容不藏"策略），但界面不再假装它出过声。

⚠️ 已知残留：`revise` 若在**一句正播到一半**时到达，那半句收不回来 ——
用户会听到"半句旧说法 + 新说法"，这是"先说后校"的固有代价
（`voice_call_plan.md` §4.4），不是本轮能消除的。

### 4.1 改稿/打断掐断后弹的「这一句没有合成出音频」是**假错误**（2026-10-11 修）

真机截图里，改稿那一刻除了「安全审计已改稿」标签，还弹了一条

> ⚠️ 语音合成出错：这一句没有合成出音频

查下去：这条**不是服务端报的**，是客户端自己误判出来的。

```
1. 一轮回复是**一次把所有句都 say 出去**的（`speakUnit` 逐句发），服务端串行合成；
2. 收到 revise（或用户打断）→ `TtsChannel.stopFrom()` 先 `player.stopAll()`
   —— 播放器的 `open` 表被清空；
3. 那些**已经 start 过、end 还没到**的句子（`seq < minSeq`，不在取消范围内）
   稍后走 `end` 分支：`endSegment()` 查不到记录 → 返回 0；
4. 旧代码把 0 一律当成"合成成功但没音频" → `onError('这一句没有合成出音频')`。
```

于是**每次改稿/打断都会弹一条假错误**（多句计划里必然有前一句在途）。

修法（不改任何时间常量、不改协议）：

* `TtsPlayer` 新增 `openSeqs()`：把"此刻还挂在播放队列里的句号"取出来；
* `TtsChannel.stopFrom()` / `stopAll()` 在 `stopAll()` **之前**取这份清单，
  交给会话层 `markPlayerStopped()` 登记；
* `end` 分支先判"这一句是不是我们主动掐断的"（登记过 / `seq ≤ cancelledThrough`
  / 正在排空残留）：是 → 计入新读数 `stoppedWithoutAudio`，**不报错**；
  否 → 照旧如实报「这一句没有合成出音频」。

读数落在通话页调试面板新增的一行 **「本轮掐断未播 N 句」** 上：有数字代表
"这几句是我们有意丢的"；真的合成不出音频时仍会弹错误 —— 两者不再混为一谈。

离线回归（不需要浏览器）：`scripts/verify_tts_stop_no_audio.ts`，3 组断言。
其中第 2 组专门盯着"别把这条修成一律不报"：空音频必须继续如实上报。

---

## 5. P1：`PHONE_LIKE` 的**粒度**（"语音输出不全"的确定真凶之一）

旧实现：在**整段（最多 100 字、按逗号打包）**上测号码正则，命中就 `return` ——
一句话里只要出现号码，**整句都不出声**，而字幕是完整的。

现在 `splitForSpeech` 只摘出号码本身，前后文照常合成，并且：

* 返回**有序片段**（`fragments`），保证"送合成 + 有意跳过"按序拼接**等于原文**；
* 每个跳过片段带原因（`phone` / `unspeakable` / `blank`），进台账当**预期例外**；
* 一个文本单元被切成的**每一段都有自己的 seq**（同 seq 复用会让两段音频互相覆盖）。

---

## 6. 复测（按你给的顺序，一次只动一处）

```powershell
# ① 纯函数层（不需要浏览器）：分段完整性、号码过滤、删尾撤回、A/B/C/D 对账
node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_tts_text_ledger.ts

# ①b 掐断后不得误报"没合成出音频"（会话层，替身播放器驱动，见 §4.1）
node "server/node_modules/tsx/dist/cli.mjs" scripts/verify_tts_stop_no_audio.ts

# ② 服务端：speak 计划的完整性契约（segments 拼接 == text）
& '.venv\Scripts\python.exe' -m pytest algorithm/tests/test_intervention_prompt.py -q

# ③ 客户端类型检查 / 构建
cd client; npx tsc -b; npm run build
```

真机四组（**不要同时调 `BARGE_IN_CONFIRM_MS` / `HALF_DUPLEX_WINDOW_MS` /
`SPEAK_PLAN_GRACE_MS` / 播放缓冲**，否则分不清是哪一项起的作用）：

| 组 | 操作 | 看什么 |
|---|---|---|
| 1 固定短句 | 不打断、不触发改稿、不含号码 | 「对账·不一致」应为"无"；若已不一致 ⇒ 问题在单次 say/解码/播放 |
| 2 长回复 | >100 字、含逗号句号术语 | 「对账·C 发出」应覆盖全文；「音频空洞」占比 |
| 3 人为打断 | AI 正在说话时立刻发下一条 | `drainedFrames` 是否 >0（残留被截住）；「对账·不一致」是否只剩"未朗读"；**不应再弹「这一句没有合成出音频」**（改稿/打断后出现即回归，见 §4.1） |
| 4 多轮连续 | 连续 5+ 轮 | 「对账·轮次」递增且每轮四份文本一致；无"念两遍" |

**排查口径**（把这三组数字发我即可定位）：
`对账·不一致` 的内容 · `打断次数` / `手动打断并发送` · `音频空洞` 与
`D 出声` 的字数。

---

## 7. 本轮**没有**改的东西（以及为什么）

| 项 | 原因 |
|---|---|
| `BARGE_IN_CONFIRM_MS` / `HALF_DUPLEX_WINDOW_MS` / `SPEAK_PLAN_GRACE_MS` | 按你的要求一次只动一处；先用对账链拿到证据 |
| `TTS_SPEED` / 播放缓冲（预缓冲、AudioWorklet 环形缓冲） | 属"产能 vs 首声"的产品取舍，见 `voice_call_audio_stutter.md` §5；空洞读数说话再定 |
| 客户端 `say` 严格串行 | 服务端已是串行 worker，客户端再串行只增延迟（§3） |
| `revise` 后重念新稿 | **2026-10-11 已拍板并实现**（口径："每一次输出的内容都需要语音合成"）：不是"整段重说"，而是前缀对齐后只补念剩余部分，见 §4 |
| 掐断后"没合成出音频"的**报错路径** | 只修**误报**（§4.1），真空音频仍然照报 —— 不能为了少弹一条错误就把真实故障一起静音 |
