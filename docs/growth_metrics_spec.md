# 我的成长 · 指标板块规范（v1，2026-10-10）

> 目标：把 `/growth` 从「展示初始值」变成「展示用户自己产生的真实指标」。
> 本文分两部分：① 现状体检（问题 + 代码证据）；② 指标板块定义（公式 / 数据源 / 展示 / 空值策略）。
> 实现见 `server/src/services/growth.service.ts`、`server/src/services/achievement.service.ts`、
> `client/src/pages/patient/MyGrowth.tsx`。

---

## 一、现状体检：为什么「用不起来」

页面截图（`docs/screenshots/13_growth_page.png`）：打卡 1、放松练习 0、心情检测 0、连续天数 0、徽章 0/7、徽章墙空。
`docs/ui_verification_report.md:35,205` 已把它记为「数据为初始值 / 新用户无数据反馈」。逐条根因：

| # | 现象 | 根因（代码证据） | 影响 |
|---|---|---|---|
| G1 | 徽章墙上永远 0 枚 | `checkAchievements()` 只在**打卡**里调用、且只查 `category:'checkin'`（`mood.service.ts:42,82-86`）。疗愈完成（`healing.service.ts:24`）与测评提交（`profile.service.ts:146`）**没有任何解锁触发** | 「探索」「疗愈」类徽章永不点亮 |
| G2 | 「坚持一周」名不副实 | 连击条件被写成累计数：`if (condition.type === 'streak' && totalCheckIns >= condition.days)`（`mood.service.ts:92`） | 断续打卡 7 次也能拿「连续 7 天」徽章 —— 指标失真 |
| G3 | 等级最高 Lv.3，Lv.4–10 永远不可达 | `level = Math.floor(unlocked / 3) + 1`（`MyGrowth.tsx:91`），而徽章总数只有 7（`achievement.service.ts:43-51`） | 等级名 `['',…,'彩虹']` 中的后 7 个是死代码 |
| G4 | 心情曲线贴在底部，读不出起伏 | Y 轴写死 `domain={[0,100]}`（`MyGrowth.tsx:217`），但打卡分值是 **1–10**（`Home.tsx:15-19` 写 9/7/5/3/1；`MoodCheckIn.score`） | 主图不可读 |
| G5 | 「情绪关键词」是编造的 | 取 `r.mood`（打卡写入的是「超好/不错/很难过」这类中文标签，`Home.tsx:83`），不足 3 个就用硬编码 `defaultKeywords = ['平静','专注','温暖','好奇','放松','勇敢']` 补（`MyGrowth.tsx:17,56-63`） | 展示用户从未表达过的词 |
| G6 | 首页「今日已打卡」「连续打卡」永远 0 | 前端读 `res.data?.checkIns`（`Home.tsx:50,66`），但 `/api/mood/checkin/history` 的 `data` **本身就是数组**（`mood.routes.ts:21`；`api.ts:20` 已把 body 解包） | 打过卡仍显示未打卡，再点一次报 400「今天已经打卡过了」 |
| G7 | 接口失败时静默显示 0 | 4 个请求各自 `.catch(() => null)`（`MyGrowth.tsx:35-38`），失败与「真是 0」不可区分 | 故障被伪装成「你什么都没做」 |
| G8 | 徽章无进度 | `/extra/achievements` 只回「已解锁列表」+ 三个总数，没有 `当前/目标` | 用户不知道离解锁还差多少，激励断裂 |

**结论**：不是「没数据」，而是 ①解锁链路断（G1/G2）、②指标口径错（G2/G4）、③展示靠兜底编造（G5/G7）、④等级与徽章不自洽（G3/G8）。

---

## 二、指标板块定义

### 2.0 设计原则

1. **一切可溯源**：每个指标都能指到一张表、一个字段、一条公式。前端不再自算、不再兜底造数。
2. **单接口聚合**：一个 `GET /api/extra/growth` 返回全部指标，避免 G7 式的静默降级。
3. **空值有语义**：没采集到 → `null`，UI 显示「待采集 + 怎么采集」，**不显示 0**（0 是「做了但为零」）。
4. **去临床化**：只用「心情、稳定、坚持、投入」等非诊断词汇，不出量表结论、不做人群比较（遵守 `AGENTS.md` 安全红线）。
5. **口径固定**：分值一律 1–10（与打卡一致）；百分比一律 0–100；天数按**本地日**归并。

### 2.1 指标分组（4 组 · 9 项 · 1 个综合指数）

#### A. 坚持（行为持续性，数据源 `mood_checkins`）

| 指标 key | 名称 | 公式 | 展示 | 空值策略 |
|---|---|---|---|---|
| `streakDays` | 连续打卡 | 从今天（今天未打卡则从昨天）向前逐日回溯，遇到断日停止 | 大字 + 🔥 | 0 天时显示「今天打个卡就开始了」 |
| `longestStreak` | 最长连续 | 全部打卡日去重排序后求最长连续段 | 副文本 | — |
| `totalCheckIns` | 累计打卡 | `count(mood_checkins)` | 计数 | 0 |
| `checkInRate7` | 近 7 天打卡率 | `近7日去重打卡天数 / 7 × 100` | 进度条 % | 无打卡 → 0% + 引导 |
| `activeDays30` | 近 30 天活跃 | `count(distinct 日(打卡 ∪ 疗愈完成))` | 计数 /30 | 0 |

> 口径说明：`streakDays` 采用「今天未打卡不算断」的宽容口径（今天还没打卡时从昨天起算），避免早上打开页面看到归零 —— 与 `Home.tsx` 现有语义一致。

#### B. 情绪（自评状态，数据源 `mood_checkins.score` 1–10）

| 指标 key | 名称 | 公式 | 展示 | 空值策略 |
|---|---|---|---|---|
| `avgMood7` | 近 7 天心情 | `mean(score)`，保留 1 位 | 值 /10 | 无记录 → `null`「待采集」 |
| `moodStability7` | 情绪稳定度 | `clamp(100 - stdev(score)/4.5×100, 0, 100)`（1–10 分理论最大标准差 4.5） | 0–100 环/条 | 记录 <2 条 → `null`（单点无法谈波动） |
| `moodDelta` | 近 7 天变化 | `mean(近7天) - mean(前7天)`，保留 1 位 | +0.8 ↑ / -0.5 ↓ | 前 7 天无记录 → `null`「再记几天就能看到趋势」 |
| `keywords` | 情绪关键词 | 近 14 天记录中：`mood` 标签 + `tags` + `note` 高频词去重，取前 6 | 标签云 | 不足则**如实留白**（不再用固定词补） |

> B 组不出「焦虑/抑郁指数」等临床结论；`keywords` 只做中性描述词，前端附「来自你自己的记录」说明。

#### C. 投入（行动量，数据源 `healing_sessions` / `assessments`）

| 指标 key | 名称 | 公式 | 展示 |
|---|---|---|---|
| `healingSessions` | 完成练习 | `count(healing_sessions where completed=true)` | 计数 |
| `healingMinutes` | 练习时长 | `sum(duration)/60` 取整（分钟） | 分钟 |
| `assessments` | 量表测评 | `count(assessments)` | 计数 |
| `lastAssessment` | 最近测评 | 最新一条的 `{type,title,totalScore,level,createdAt}` | 卡片；无 → 「去探索」入口 |

#### D. 综合成长指数（0–100，前端只展示、不重算）

```
坚持分 = 30天打卡率×0.6 + min(streakDays/21,1)×0.4        ×100
情绪分 = 归一化(avgMood14) ×0.6 + moodStability×0.4
投入分 = min(healingSessions/20,1)×50 + min(healingMinutes/180,1)×50
探索分 = min(assessments/5,1)×100

growthIndex = Σ(维度分 × 权重) / Σ(可用维度权重)     权重 = 坚持0.35 / 情绪0.30 / 投入0.20 / 探索0.15
```

* **权重按可用维度归一化**：新用户只有打卡时，指数由「坚持分」决定，而不是被 4 个 0 拉成个位数 —— 这直接解决 G7 的体验问题。
* 各维度分原样随接口返回（`dimensions[]`），前端画雷达图/进度条，用户能看到分数是怎么来的。

#### E. 等级与成长值（替代 `unlocked/3`）

```
xp = 打卡×10 + 疗愈完成×8 + 测评×15 + min(streak,30)×2 + 活跃天数30×2
Lv.1…Lv.10 门槛：0 / 50 / 140 / 280 / 480 / 760 / 1140 / 1620 / 2220 / 3000
等级名：萌芽 新芽 花苞 初绽 盛放 向阳 绽放 璀璨 星光 彩虹
```

等级由**行为量**决定（不再是「徽章数/3」），Lv.10 可达；接口回 `{level, name, xp, nextXp, progress}`。

### 2.2 徽章引擎（修 G1/G2/G8）

条件类型统一在 `achievement.service.ts` 里求值，支持：

| type | 参数 | 语义 |
|---|---|---|
| `total_checkins` | count | 累计打卡天数 |
| `streak` | days | **真实连续**打卡天数（修 G2） |
| `total_sessions` | count | 完成疗愈练习次数 |
| `healing_minutes` | minutes | 疗愈累计时长 |
| `total_assessments` | count | 测评次数 |
| `active_days` | days | 活跃天数（打卡 ∪ 疗愈去重） |

* 求值入口 `evaluateAchievements(userId)`，在**打卡成功 / 疗愈完成 / 测评提交**三处调用（修 G1），幂等 upsert。
* 接口为每枚徽章返回 `{ unlocked, unlockedAt, progress: { current, target, percent } }`（修 G8）。
* 保留原有 7 个徽章 id（不清库、不破坏已有记录），新增 5 枚使阶段覆盖更密：
  `ach_checkin_streak_3`（连续3天）、`ach_checkin_total_7`（累计7天）、
  `ach_healing_minutes_60`（静心60分钟）、`ach_active_days_30`（活跃30天）、`ach_assessment_three`（测评3次）。

### 2.3 展示形态（`/growth` 一屏）

1. **顶部**：成长指数环图 + `Lv.n 等级名` + 距离下一级还差多少 XP + 一句话总结（如「这周你来了 5 天」）。
2. **指标板块**：4 组卡片（坚持 / 情绪 / 投入 / 探索），每张卡 = 数值 + 单位 + 目标进度 + 「ⓘ 口径」说明 + 空态引导按钮。
3. **趋势区**：近 14 天心情折线（Y 轴 1–10，虚线为 7 日均线）+ 近 30 天打卡小方格热力图。
4. **徽章墙**：按分类分组，未解锁显示 `current/target` 进度，已解锁显示解锁日期。
5. **空态**：全无数据时不是「一片 0」，而是三步入坑引导（打卡 → 疗愈 → 测评），并说明每个动作会点亮什么。

---

## 三、验收口径

* `GET /api/extra/growth` 对同一用户返回的指标与直接 SQL 聚合一致（脚本 `scripts/verify-growth-metrics.mjs` 交叉核对）。
* 新注册用户：所有计数为 0、比率指标为 `null` 或 0%、`growthIndex` 有值（由可用维度归一化）、徽章墙显示 `0/12` 且每枚有进度。
* 打卡 3 天 → `ach_checkin_streak_3` 解锁；发布疗愈完成 → `ach_healing_first` 解锁；提交测评 → `ach_assessment_first` 解锁。
* 首页打卡后：立即显示「今天已经打过卡啦」与正确的连续天数（修 G6）。
* 心情折线在 1–10 量程内可读（修 G4）。

---

## 四、验收结果（2026-10-10 实测）

环境：Postgres 容器 `mh-postgres` + 后端 3000 + vite 5173（均为本机实跑）。

| 验收项 | 命令 | 结果 |
|---|---|---|
| 接口值与独立重算逐项一致 | `node scripts/verify-growth-metrics.mjs` | **24/24 通过**（演示账号） |
| 真实账号 `patient@mental.com`（稀疏数据） | `node scripts/verify-growth-metrics.mjs --email=patient@mental.com` | **24/24 通过**，空值语义正确（均分/稳定度/变化为 `null` 而非 0） |
| 全新注册用户（零数据） | 注册后同名脚本 | **24/24 通过**，无 `NaN`、指数由可用维度归一化 |
| 页面真的渲染出指标板块 | `node scripts/probe-growth-page.mjs` | **13/13 通过**，无 console 报错 |
| 首页打卡状态与成长接口同口径 | 同上 `--home=1` | **15/15 通过**（演示账号 `已打卡/连续 7`；真实账号 `未打卡/0`） |
| 打卡 / 疗愈完成 / 测评提交即时解锁 | 真实 HTTP 调用 | 分别解锁 `初次打卡`、`疗愈新手`、`测评先锋`；重复打卡仍正确返回 400 |
| 前端生产构建 | `npm run build`（client） | 退出码 0，4007 modules |

证据截图：

* `docs/screenshots/26_growth_metrics.png` —— 演示账号完整指标板块（演示账号 `growth-demo@mental.com` / `123456`，由 `scripts/seed-growth-demo.mjs` 生成）
* `docs/screenshots/27_growth_real_account.png` —— 真实账号 `patient@mental.com` 的稀疏/空态表现
* `docs/screenshots/28_home_checkin_state.png` —— 首页「今天已经打过卡啦 + 连续 7 天」（修 G6 后）

### 实现中发现但**未**改动的相邻问题

1. **全局卡片背景被 `!important` 覆盖**：`client/src/styles/healing-theme.css:324`
   ```css
   .ant-card { background: linear-gradient(135deg, #ffffff 0%, #fff5f7 100%) !important; }
   ```
   这条全局规则让**任何** Card 上的行内渐变失效。历史截图 `13_growth_page.png` 里成长页顶部卡片「白字白底、只剩一个 emoji」就是这个原因；首页 `Home.tsx` 的紫色打卡卡同样被压成白底（心情标签只剩 emoji）。
   本次把成长页与首页打卡卡的渐变改画到 `.ant-card-body` 上，**没有触碰全局规则**（避免影响其它页面）。
   若要让全站恢复设计稿：删掉该行的 `!important`（其余声明不动），然后回归一遍用到大面积渐变卡片的页面。
2. **旧成就页 `client/src/pages/patient/Achievements.tsx` 已是孤儿**：`App.tsx:81` 把 `/achievements` 重定向到 `/growth`，该文件不再被路由使用（接口形状仍兼容，未删）。

---

## 五、签到（每日心情打卡）链路修复

打卡是整条成长指标的数据源头，本轮把它从「点了没反应」修到可用可验证。

| # | 问题 | 根因 | 修法 |
|---|---|---|---|
| S1 | 首页「今天已打卡 / 连续天数」恒为 0，再点一次报 400 | 前端读 `res.data.checkIns`，而 `/mood/checkin/history` 的 `data` 就是数组 | `Home.tsx` 兼容两种形状；连续天数与后端指标同口径 |
| S2 | **「昨晚睡了多久」点了没反应，`sleep_records` 全表 0 行** | 传 `bedTime:''` / `wakeTime:''` → `new Date('')` 是 Invalid Date → 接口 500，前端 `.catch` 静默吞掉 | `SleepService.record` 支持只给 `hours`；缺时间就报可读的 400；前端不再静默，改为提示「睡眠没记上」 |
| S3 | 睡眠质量量纲冲突（会写出「800 分」） | 前端按 0-100 传 80/60/40，字段与消费方（`×10` 换算）按 1-10 | 前端按 1-10 传；后端对 >10 的值兜底收敛（`80 → 8`），历史脏数据不再产生 |
| S4 | 打卡卡被全局 `!important` 压成白底，白字隐形 | `healing-theme.css:324`（见上文「未改动」第 1 条） | 渐变改画到 `.ant-card-body`；心情标签补边框，选中表情用白色描边高亮 |
| S5 | 点错表情当天无法更改 | `POST /mood/checkin` 对同一自然日直接 400，没有更正入口 | 新增 `PUT /api/mood/checkin`：今天已有记录就原地更新（仍是每天一条，不影响连续天数），同步更正当天趋势记录 |
| S6 | 打卡解锁的徽章没有提示 | 接口一直返回 `newlyUnlocked`，旧页面丢掉了 | 打卡后弹出「🎉 解锁新徽章 …」+「去「我的成长」看看」入口；多次操作累加不覆盖 |

### 第二轮：「连续签到数据不更新」的追查（2026-10-10 下午）

现象：`patient@mental.com`（小明）的成长页显示「连续打卡 0 天 / 累计打卡 1 天」，用户认为签到没有更新。
结论：**页面数字是准的** —— 该账号当时确实只有 2026-09-25 那一次打卡，之后没有任何打卡行。
但这次追查顺带找出并修掉了 3 个真问题、2 个探针自身的误报：

| # | 问题 | 根因 | 修法 |
|---|---|---|---|
| S7 | 「连续打卡 0 天」却挂着绿色「已记录」标签（截图里的误会来源） | 指标卡状态恒为 `ok` | 0 天时状态改为 `empty`，文案改为「今天还没打卡」，并给出去打卡的引导 |
| S8 | 打卡后成长页仍停在旧数字（同一标签页内切回来、或另一个标签页打的卡） | 页面只在挂载时取一次数 | 顶部加「刷新」按钮；`window.focus` / `visibilitychange` 时静默重新取数（不闪 loading） |
| S9 | 快速连点两个表情，第二次被静默丢弃 | `handleMoodCheckIn` 有并发守卫，但按钮没有禁用反馈 | 请求进行中整排表情按钮 `disabled`，用户能看见「正在提交」而不是「点了没反应」 |

签到口径（避免再次误判）：

* `连续打卡` = 从**今天**往前逐日回溯；今天还没打卡时从**昨天**起算（不清零），断一天即归零。
* `近 7 天打卡率` = 近 7 个自然日里打过卡的天数 ÷ 7。所以「昨天打过、今天还没打」时它是 14% 而不是 0%。
* `累计打卡` = 按自然日去重后的打卡总天数。

验证（全部真浏览器 + 真接口）：

| 场景 | 命令 | 结果 |
|---|---|---|
| 全新账号走完整签到流程 | `node scripts/probe-checkin-flow.mjs` | **22/22 通过**（含「打卡后成长页 DOM 数字 = 接口值」） |
| 真实账号（已打卡→更正分支） | 同上 `--email=patient@mental.com --skip-sleep=1` | **18/18 通过** |
| 成长页 + 首页口径一致 | `node scripts/probe-growth-page.mjs --email=patient@mental.com --home=1` | **15/15 通过** |
| 指标接口 vs 独立 SQL 重算 | `node scripts/verify-growth-metrics.mjs [--email=…]` | 两个账号各 **24/24 通过** |

截图：`30_growth_after_checkin.png`（打卡后的成长页）、`27_growth_real_account.png`（小明账号修好后的样子）、`28_home_checkin_state.png`（首页已打卡状态）。

自查某个账号到底有没有当天打卡（不改数据）：

```bash
'select u.email, m.mood, m.score, m."checkInDate" from mood_checkins m
   join users u on u.id = m."userId" order by m."checkInDate" desc limit 5;' \
  | docker exec -i mh-postgres psql -U postgres -d mental_health
```

> 提醒：登录态存在 `localStorage`（同源共享）。如果同时开着两个标签页分别登录了不同账号，后登录的那个会覆盖 token，**打卡可能落到另一个账号上** —— 这也是「打了卡但成长页不动」的一类真实成因。

原始验证（第一轮）：`node scripts/probe-checkin-flow.mjs` 曾 **15/15 通过**；接口边界另测：`quality=80 → 8`、`hours=7 → duration=7`、缺时间缺 hours → 400 可读报错、起床早于入睡 → 400。
探针用临时账号时会在结束时级联删除；用 `--email=` 指定真实账号时不做任何清理（截图与断言会如实反映该账号）。
