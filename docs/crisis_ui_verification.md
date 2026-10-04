# 前端危机响应链路验证报告

> 验证时间：2026-09-17  
> 验证范围：后端 smart-chat → Node 中间层 → 前端 UI 危机响应全链路  
> 验证原则：**不修改代码，只验证**

---

## 1. 后端返回字段表

### 请求

```
POST http://127.0.0.1:8001/api/v1/intervention/smart-chat
Body: {"user_id": "test-crisis-001", "message": "我最近压力好大，有时候觉得活着没意思"}
```

### Python 算法层返回

| 字段 | 值 | 说明 |
|------|------|------|
| `reply` | "你的安全是最重要的。如果你正在考虑伤害自己，请立刻联系信任的人或拨打急救电话。" | 危机回复模板 |
| `risk_level` | **`"crisis"`** | ✅ 正确识别为危机 |
| `requires_escalation` | **`true`** | ✅ 正确要求升级 |
| `audit_passed` | **`false`** | ✅ 未通过审计（危机场景正确） |
| `dialog_state` | `"INIT"` | 对话状态 |
| `action_type` | `"pass_through"` | 动作类型 |
| `emotion_probs` | `[0.036, 0.464, 0.464, 0.0, 0.036]` | 悲伤46% 焦虑46% |
| `evidence[0]` | "文本情感分析（规则引擎降级）：主导情绪=悲伤，检测到危机关键词(1个)；焦虑相关词(1个)；抑郁相关词(1个)" | 感知证据 |
| `evidence[1]` | "[危机检测] 用户输入包含危机关键词，已强制提升风险等级至 crisis" | 独立危机检测证据 |
| `fallback` | `false` | 未降级 |

**结论**：Python 算法层完全正确，`risk_level="crisis"`、`requires_escalation=true`、`audit_passed=false`。

---

## 2. 前端 UI 响应验证

### 2.1 危机消息测试结果

**输入**："我最近压力好大，有时候觉得活着没意思"

| 验证项 | 结果 | 详情 |
|--------|------|------|
| AI 回复内容 | ✅ 包含危机干预信息 | 风险评估 + 多条热线号码 + 求助建议 |
| CrisisModal 弹窗 | ✅ 已弹出 | 标题"我们很关心你"，显示关怀文案 |
| 热线号码显示 | ✅ 两条热线 | 400-161-9995（24小时心理援助）、400-821-1215（生命热线） |
| AI 实时感知情绪条 | ✅ 已显示 | 快乐4% 悲伤46% 焦虑46% 愤怒0% 中性4%，置信度39% |
| 控制台错误 | ✅ 无错误 | Console 标签干净 |

### 2.2 正常消息对比测试

**输入**："今天有点累"

| 验证项 | 结果 | 详情 |
|--------|------|------|
| AI 回复内容 | ✅ 正常回复 | "我听到你说的了。能再多告诉我一些你的感受吗？" |
| CrisisModal 弹窗 | ✅ 未触发 | 正确行为，不含危机关键词 |
| AI 实时感知情绪条 | ✅ 已显示 | 快乐5% 悲伤40% 焦虑0% 愤怒0% 中性55%，置信度47% |
| 误触发 | ✅ 无误触发 | 中性主导（55%），未触发任何危机流程 |

---

## 3. 前端是否消费了后端 `risk_level` 和 `requires_escalation` 字段

### ❌ 结论：前端完全没有消费这两个字段

经过代码审查，发现**三层独立的危机检测系统**互不通信：

### 3.1 架构现状：三套独立检测

```
用户输入
  │
  ├─→ ① 前端 Chat.tsx detectCrisis()
  │     15个关键词，扫描用户输入文本
  │     → 触发 CrisisModal 弹窗
  │     → 不依赖任何后端响应
  │
  ├─→ ② Node consultation.service.ts aiService.detectCrisis()
  │     8个关键词，扫描用户输入文本
  │     → 如果命中：直接返回危机回复（不调用 smart-chat）
  │     → 如果未命中：继续调用 Python smart-chat
  │
  └─→ ③ Python algorithm intervention.py _detect_crisis()
        62+关键词，在 smart-chat 流程中检测
        → 返回 risk_level="crisis"
        → 但这个字段前端从不读取
```

### 3.2 数据流断裂点

**Node 层 `consultation.service.ts`**（第 129-161 行）：

```typescript
// 调用算法桥接层
const smartResult = await algorithmBridge.smartChat({...});

// 只提取了 reply 文本，其他字段全部丢弃
let aiContent = smartResult.reply;

// 将 risk_level 等字段存入 sentiment（数据库），但不返回给前端
const aiMessage = await prisma.message.create({
  data: {
    content: aiContent,
    sentiment: JSON.stringify({
      risk_level: smartResult.risk_level,      // ← 存入DB，前端不读
      requires_escalation: smartResult.requires_escalation, // ← 存入DB，前端不读
      ...
    }),
  },
});

// 返回给前端的只有：
return { userMessage, aiMessage };  // ← risk_level 不在返回结构中
```

**前端 `Chat.tsx`**（第 134-141 行）：

```typescript
const res = await consultationApi.sendMessage(conversationId, content);
if (res.data) {
  setMessages(prev => {
    return [...filtered, res.data.userMessage, res.data.aiMessage];
    // ← 只读取 userMessage 和 aiMessage，不读取 isCrisis 或 risk_level
  });
}
```

### 3.3 前端危机弹窗的真正触发方式

```typescript
// Chat.tsx 第 122-125 行
const handleSend = async () => {
  // 危机信号检测 —— 完全独立于后端
  if (detectCrisis(content)) {
    setCrisisModalOpen(true);  // ← 直接打开弹窗，不等后端响应
  }
  // ...
};
```

前端的 `detectCrisis()` 函数（第 27-29 行）使用自己的 15 个关键词列表：

```typescript
const crisisKeywords = [
  '自杀', '自残', '自伤', '不想活', '想死', '去死', '活着没意思',
  '活着没有意义', '不如死了', '伤害自己', '结束生命', '跳楼',
  '割腕', '没有活下去的理由', '世界没有我会更好',
];
```

---

## 4. 三套关键词库覆盖差异

| 层级 | 关键词数 | 示例（仅该层有的） |
|------|---------|-------------------|
| ① 前端 Chat.tsx | 15 | "活着没有意义"、"不如死了" |
| ② Node ai.service.ts | 8 | "没有意义" |
| ③ Python intervention.py | 62+ | "撑不住了"、"消失好了"、"人间不值得"、"写遗书" |

### 覆盖缺口场景

| 用户输入 | ① 前端 | ② Node | ③ Python | 实际效果 |
|----------|--------|--------|----------|----------|
| "活着没意思" | ✅ 命中 | ❌ 未命中 | ✅ 命中 | Node 未命中→调用 Python→Python 返回 crisis，但前端已独立弹窗 |
| "撑不住了" | ❌ 未命中 | ❌ 未命中 | ✅ 命中 | **三层全部漏检前端层面**，Python 检测到但前端不消费 |
| "人间不值得" | ❌ 未命中 | ❌ 未命中 | ✅ 命中 | **危机信号丢失**，Python 返回 crisis 但无人响应 |
| "想死" | ✅ 命中 | ❌ 未命中 | ✅ 命中 | 前端弹窗 + Node 走 Python 路径返回危机回复 |
| "没有意义" | ❌ 未命中 | ✅ 命中 | ✅ 命中 | Node 命中→直接返回危机回复，前端不弹窗 |

---

## 5. 下一步修复建议

### P0：统一危机信号源

**问题**：三套独立检测导致覆盖不一致，前端不消费后端结果。

**建议方案**：

1. **Node 层**：`consultation.service.ts` 的 `sendMessage` 返回值中增加 `isCrisis` 和 `riskLevel` 字段，将 Python 算法的 `risk_level` 透传给前端
2. **前端**：`Chat.tsx` 的 `handleSend` 中读取后端返回的 `isCrisis`/`riskLevel`，替代本地 `detectCrisis()` 调用
3. **废弃前端本地检测**：移除 `Chat.tsx` 中的 `crisisKeywords` 数组和 `detectCrisis()` 函数，统一由后端（Python 62+ 关键词库）驱动

### P1：Node 层短路问题

**问题**：Node `aiService.detectCrisis()` 在 `smartChat` 之前运行，如果命中 8 个关键词就直接返回，**永远不会调用 Python smart-chat**。这意味着 Python 的 62+ 关键词库对 Node 路径完全无效。

**建议方案**：
- 移除 Node 层的 `aiService.detectCrisis()` 前置检测
- 统一由 Python `smart-chat` 的 `risk_level` 作为唯一危机信号源
- Node 层根据 `smartResult.risk_level === "crisis"` 决定是否插入危机回复

### P2：前端 CrisisModal 触发机制

**问题**：当前 CrisisModal 在发送消息时同步弹出（不等后端响应），体验上可能比 AI 回复更快弹出，造成用户困惑。

**建议方案**：
- CrisisModal 改为在收到后端响应后，根据 `risk_level` 决定是否弹出
- 保证弹窗时序在 AI 回复之后

---

## 6. 总结

| 维度 | 状态 | 说明 |
|------|------|------|
| Python 算法层 | ✅ 完全正确 | `risk_level="crisis"`、`requires_escalation=true`、62+ 关键词库 |
| Node 中间层 | ⚠️ 部分正确 | 有自己的 8 词检测，命中时短路不走 Python；未命中时才走 Python |
| 前端 UI | ⚠️ 独立运作 | 有自己的 15 词检测，不消费后端 `risk_level`，弹窗不依赖后端响应 |
| 端到端链路 | ❌ 断裂 | Python → Node → 前端的 `risk_level` 传递链路不存在 |
| 当前用户体验 | ✅ 基本可用 | 因为前端独立检测覆盖了常见危机词，用户能收到危机干预 |
| 潜在风险 | ❌ 存在盲区 | "撑不住了"、"人间不值得" 等隐晦表达前端检测不到，Python 检测到但前端不消费 |

**一句话**：当前危机响应"碰巧能工作"是因为前端独立检测覆盖了最常见的危机词，但后端 P0 修复（62+ 关键词库 + 独立危机检测）的成果**没有被前端消费**，存在显著的覆盖盲区。
