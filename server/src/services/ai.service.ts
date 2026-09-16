import { config } from '../config';
import { AppError } from '../middlewares/errorHandler';

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface ChatResponse {
  content: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

class AiService {
  // 调用通义千问 (OpenAI 兼容格式)
  async chat(messages: ChatMessage[], options?: { temperature?: number; top_p?: number; model?: string }): Promise<ChatResponse> {
    const apiKey = config.dashscope.apiKey;
    if (!apiKey) {
      throw new AppError('通义千问 API 密钥未配置，请在 .env 中设置 DASHSCOPE_API_KEY', 500);
    }

    const url = `${config.dashscope.baseUrl}/chat/completions`;

    const body = {
      model: options?.model || config.dashscope.model,
      messages,
      temperature: options?.temperature || 0.8,
      top_p: options?.top_p || 0.8,
      stream: false,
    };

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 秒超时
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
    
      if (!response.ok) {
        const errorData: any = await response.json().catch(() => ({}));
        throw new AppError(`通义千问 API 错误：${errorData.error?.message || response.statusText}`, 500);
      }
    
      const data: any = await response.json();
    
      return {
        content: data.choices?.[0]?.message?.content || '',
        usage: data.usage,
      };
    } catch (error: any) {
      if (error.name === 'AbortError') {
        throw new AppError('AI 响应超时，请稍后重试', 504);
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // 心理咨询专用对话
  async counselingChat(
    userMessage: string,
    conversationHistory: ChatMessage[],
    context?: { mood?: string; riskLevel?: string; patientInfo?: string }
  ): Promise<ChatResponse> {
    const systemPrompt = this.buildCounselingPrompt(context);

    const messages: ChatMessage[] = [
      { role: 'system', content: systemPrompt },
      { role: 'assistant', content: '好的，我理解了。我会以温暖、共情的方式与你交流，关注你的感受。请告诉我你现在的想法。' },
      ...conversationHistory.slice(-10).map(m => ({ role: m.role as 'user' | 'assistant', content: m.content })),
      { role: 'user', content: userMessage },
    ];

    return this.chat(messages, { temperature: 0.85 });
  }

  // 苏格拉底式反问对话（话痨树洞）
  async socraticChat(
    userMessage: string,
    conversationHistory: ChatMessage[],
  ): Promise<ChatResponse> {
    const systemPrompt = `你是一个"话痨树洞"——一个温暖但不给建议的倾听者。你的核心原则：

1. **绝不给建议**：不说"你应该"、"我建议你"、"你可以试试"
2. **苏格拉底式反问**：用反问引导用户自我剖析，例如：
   - 用户说"我很失败" → 你反问："这个结论，是你自己得出的，还是别人替你盖章的？"
   - 用户说"没人理解我" → 你反问："你曾经尝试过让他们理解吗？还是你已经开始替他们做决定了？"
   - 用户说"我做不到" → 你反问："是'做不到'，还是'害怕做不到'？"
3. **共情但不沉溺**：先简短认可感受，然后立刻抛出反问
4. **保持对话感**：像朋友聊天，不像心理咨询，语气轻松自然
5. **适度话痨**：可以多说几句，但每段话最后都要落脚到一个反问
6. **危机意识**：如果检测到自伤/自杀倾向，温柔但直接地引导寻求专业帮助

回复格式：先共情（1-2句），再反问（1个核心问题）。总长度控制在100字以内。`;

    const messages: ChatMessage[] = [
      { role: 'system', content: systemPrompt },
      { role: 'assistant', content: '嗨，我在呢。今天想聊点什么？不用顾虑，说什么都行。' },
      ...conversationHistory.slice(-12).map(m => ({ role: m.role as 'user' | 'assistant', content: m.content })),
      { role: 'user', content: userMessage },
    ];

    return this.chat(messages, { temperature: 0.9 });
  }

  // 生成 5 分钟现实任务
  async generateRealityTask(anxietyLevel: number, context?: string): Promise<{
    title: string;
    description: string;
    duration: number;
    category: string;
    steps: string[];
  }> {
    const levelDesc = anxietyLevel > 70 ? '高度焦虑' : anxietyLevel > 40 ? '中度焦虑' : '轻度焦虑';
    const messages: ChatMessage[] = [
      {
        role: 'user',
        content: `用户当前处于${levelDesc}状态${context ? '，背景：' + context : ''}。

请生成一个"5分钟现实任务"——一个具体的、需要在真实世界中完成的小任务，帮助用户从焦虑思维中抽离出来。

要求：
1. 任务必须在5分钟内完成
2. 需要调动感官（视觉/听觉/触觉/嗅觉）
3. 不需要任何特殊工具
4. 具体到动作级别，不要抽象

返回JSON：
{
  "title": "任务名称（简短有趣）",
  "description": "任务描述（2-3句话）",
  "duration": 3-5的分钟数,
  "category": "感官类/运动类/创造类/观察类/社交类",
  "steps": ["步骤1", "步骤2", "步骤3"]
}

示例：
- "找出房间里3个绿色的东西，描述它们的质感"
- "录一段15秒窗外车流声，闭上眼睛听3遍"
- "用非惯用手写自己的名字10遍"
- "站起来做5个深蹲，感受脚底与地面的接触"`
      }
    ];

    try {
      const response = await this.chat(messages, { temperature: 0.8 });
      let cleaned = response.content.replace(/```json\n?/g, '').replace(/```/g, '').trim();
      return JSON.parse(cleaned);
    } catch {
      return {
        title: '感官锚定练习',
        description: '环顾四周，找出5个你能看到的东西、4个你能触摸到的东西、3个你能听到的声音、2个你能闻到的气味、1个你能尝到的味道。',
        duration: 5,
        category: '感官类',
        steps: ['找出5个你能看到的东西', '触摸4个不同质感的物品', '闭上眼睛听3种声音', '深呼吸，感受1种气味'],
      };
    }
  }

  // 深度情绪分析
  async analyzeEmotionDeep(text: string): Promise<{
    primaryEmotion: string;
    secondaryEmotions: string[];
    intensity: number;
    tone: string;
    urgency: number;
    keywords: string[];
    sentimentScore: number;
    summary: string;
  }> {
    const messages: ChatMessage[] = [
      {
        role: 'user',
        content: `请对以下文本进行深度心理情绪分析，严格返回JSON格式（不要包含markdown代码块标记）：
{
  "primaryEmotion": "主要情绪（如：焦虑/悲伤/愤怒/恐惧/喜悦/平静/困惑/孤独/压力/无助）",
  "secondaryEmotions": ["次要情绪1", "次要情绪2"],
  "intensity": 0-100的情绪强度,
  "tone": "语气类型（如：绝望/求助/抱怨/平静/兴奋/犹豫/愤怒/疲惫）",
  "urgency": 0-100的紧急程度,
  "keywords": ["关键情绪词1", "关键情绪词2", "关键情绪词3"],
  "sentimentScore": -100到100的情感分数（负值=负面，正值=正面）,
  "summary": "一句话概括当前情绪状态"
}

分析文本：${text}`
      }
    ];

    try {
      const response = await this.chat(messages, { temperature: 0.2 });
      let cleaned = response.content.replace(/```json\n?/g, '').replace(/```/g, '').trim();
      return JSON.parse(cleaned);
    } catch {
      return {
        primaryEmotion: '平静', secondaryEmotions: [], intensity: 30,
        tone: '平静', urgency: 10, keywords: [], sentimentScore: 0, summary: '情绪状态平稳',
      };
    }
  }

  // 基于当天所有文字动态生成心理画像
  async generateDynamicProfile(texts: string[]): Promise<{
    anxiety: number; depression: number; stress: number;
    sleepQuality: number; socialActivity: number; emotionalStability: number;
    overallScore: number; riskLevel: string;
    emotionalSummary: string;
    dominantEmotions: string[];
    emotionalTrend: string;
  }> {
    const combinedText = texts.join('\n---\n');
    const messages: ChatMessage[] = [
      {
        role: 'user',
        content: `基于以下患者今天的所有文字记录（咨询对话、心情日记等），分析其心理状态并返回JSON：
{
  "anxiety": 0-100焦虑分数,
  "depression": 0-100抑郁分数,
  "stress": 0-100压力分数,
  "sleepQuality": 0-100睡眠质量（越高越好）,
  "socialActivity": 0-100社交活跃度,
  "emotionalStability": 0-100情绪稳定性,
  "overallScore": 0-100综合心理健康分,
  "riskLevel": "LOW/MEDIUM/HIGH/CRISIS",
  "emotionalSummary": "200字以内的今日情绪总结",
  "dominantEmotions": ["今日主要情绪1", "主要情绪2", "主要情绪3"],
  "emotionalTrend": "情绪变化趋势描述（如：从焦虑逐渐平复/持续低落/波动较大）"
}

今日文字记录：
${combinedText}`
      }
    ];

    try {
      const response = await this.chat(messages, { temperature: 0.3 });
      let cleaned = response.content.replace(/```json\n?/g, '').replace(/```/g, '').trim();
      return JSON.parse(cleaned);
    } catch {
      return {
        anxiety: 50, depression: 50, stress: 50, sleepQuality: 50,
        socialActivity: 50, emotionalStability: 50, overallScore: 50,
        riskLevel: 'LOW', emotionalSummary: '数据不足，无法生成分析',
        dominantEmotions: [], emotionalTrend: '平稳',
      };
    }
  }

  // 生成个性化应对策略
  async generateCopingStrategies(profile: {
    primaryEmotion: string;
    dominantEmotions: string[];
    anxiety: number; depression: number; stress: number;
    emotionalSummary: string;
  }): Promise<{
    immediateActions: string[];
    shortTermStrategies: string[];
    longTermSuggestions: string[];
    recommendedModules: string[];
    encouragingMessage: string;
  }> {
    const messages: ChatMessage[] = [
      {
        role: 'user',
        content: `基于以下心理状态分析，生成个性化的应对策略，返回JSON：
{
  "immediateActions": ["当下可以做的3-5件具体小事"],
  "shortTermStrategies": ["本周可以尝试的3-5个策略"],
  "longTermSuggestions": ["长期建议的3-5个方向"],
  "recommendedModules": ["推荐使用的平台功能，如：冥想引导/呼吸练习/疗愈音乐/情绪日记/正念练习"],
  "encouragingMessage": "一句温暖的鼓励话语（50字以内）"
}

当前状态：
- 主要情绪：${profile.primaryEmotion}
- 情绪列表：${profile.dominantEmotions.join('、')}
- 焦虑指数：${profile.anxiety}/100
- 抑郁指数：${profile.depression}/100
- 压力指数：${profile.stress}/100
- 情绪总结：${profile.emotionalSummary}`
      }
    ];

    try {
      const response = await this.chat(messages, { temperature: 0.6 });
      let cleaned = response.content.replace(/```json\n?/g, '').replace(/```/g, '').trim();
      return JSON.parse(cleaned);
    } catch {
      return {
        immediateActions: ['深呼吸放松 5 分钟', '喝一杯温水', '到窗边看看远处'],
        shortTermStrategies: ['尝试一次冥想练习', '记录今日心情', '与信任的人聊聊'],
        longTermSuggestions: ['建立规律作息', '培养运动习惯', '学习情绪管理技巧'],
        recommendedModules: ['冥想引导', '呼吸练习', '情绪日记'],
        encouragingMessage: '每一个小小的改变都值得肯定，你正在变得更好。',
      };
    }
  }

  // 对话中的逐句情绪标注
  async analyzeSingleMessageEmotion(text: string): Promise<{
    emotion: string; intensity: number; tone: string;
  }> {
    const messages: ChatMessage[] = [
      {
        role: 'user',
        content: `分析以下文本的情绪，返回JSON：{"emotion": "主要情绪", "intensity": 0-100, "tone": "语气"}。文本：${text}`
      }
    ];

    try {
      const response = await this.chat(messages, { temperature: 0.2 });
      let cleaned = response.content.replace(/```json\n?/g, '').replace(/```/g, '').trim();
      return JSON.parse(cleaned);
    } catch {
      return { emotion: '平静', intensity: 30, tone: '平静' };
    }
  }

  // 生成心理画像报告
  async generateProfileReport(conversationSummaries: string[], assessmentResults: any): Promise<string> {
    const messages: ChatMessage[] = [
      {
        role: 'user',
        content: `基于以下心理咨询对话摘要和测评结果，生成一份专业的心理画像报告。
        
对话摘要：
${conversationSummaries.join('\n')}

测评结果：
${JSON.stringify(assessmentResults)}

请从以下维度分析：
1. 情绪状态评估
2. 主要心理困扰
3. 人格特质倾向
4. 风险等级评估
5. 建议干预方向

请用专业但易懂的语言撰写报告。`
      }
    ];

    const response = await this.chat(messages, { temperature: 0.5 });
    return response.content;
  }

  // 生成治疗规划
  async generateTreatmentPlan(profile: any, history: any): Promise<string> {
    const messages: ChatMessage[] = [
      {
        role: 'user',
        content: `基于患者心理画像生成治疗规划 JSON。画像：${JSON.stringify(profile)}。咨询历史：${JSON.stringify(history)}。

返回 JSON 格式：
{"goals":{"shortTerm":["短期目标1"],"longTerm":["长期目标1"]},"phases":[{"phase":1,"name":"阶段名","duration":"1-2周","focus":"核心焦点","consultationTopics":["主题1"],"therapistRole":"角色"}],"dailyTasks":[{"task":"任务名","description":"描述","frequency":"频率","tools":["工具"]}],"assessmentNodes":[{"week":4,"metrics":["指标"],"purpose":"目的"}]}`
      }
    ];

    const response = await this.chat(messages, { temperature: 0.5 });
    return response.content;
  }

  // 危机识别
  async detectCrisis(text: string): Promise<{ isCrisis: boolean; level: string; suggestion: string }> {
    const crisisKeywords = ['自杀', '不想活', '结束生命', '自残', '割腕', '跳楼', '去死', '没有意义'];
    const hasKeyword = crisisKeywords.some(kw => text.includes(kw));

    if (hasKeyword) {
      return {
        isCrisis: true,
        level: 'CRISIS',
        suggestion: '检测到可能的危机信号。建议立即联系专业心理危机干预热线：400-161-9995 或拨打 120。',
      };
    }

    const messages: ChatMessage[] = [
      {
        role: 'user',
        content: `请判断以下文本是否包含心理危机信号（自伤/自杀倾向），返回JSON：{"isCrisis": boolean, "level": "LOW/MEDIUM/HIGH/CRISIS", "suggestion": "建议"}。文本：${text}`
      }
    ];

    try {
      const response = await this.chat(messages, { temperature: 0.2 });
      return JSON.parse(response.content);
    } catch {
      return { isCrisis: false, level: 'LOW', suggestion: '' };
    }
  }

  // 构建心理咨询系统提示
  private buildCounselingPrompt(context?: { mood?: string; riskLevel?: string; patientInfo?: string }): string {
    let prompt = `你是一位专业、温暖的AI心理咨询师助手。你的工作原则：

1. 共情倾听：认真倾听来访者的表达，给予情感上的理解和支持
2. 认知行为疗法(CBT)：帮助识别和挑战不合理的思维模式
3. 非评判态度：不批评、不说教，尊重来访者的感受
4. 循序渐进：不急于给建议，先充分理解问题
5. 危机意识：如果发现自伤/自杀风险，及时引导寻求专业帮助
6. 边界意识：明确自己是AI辅助角色，严重情况建议寻求真人咨询师

请注意语言温暖亲切，像一位关心你的朋友。`;

    if (context?.mood) {
      prompt += `\n\n来访者当前情绪状态：${context.mood}`;
    }
    if (context?.riskLevel && context.riskLevel !== 'LOW') {
      prompt += `\n\n️ 注意：该来访者风险等级为 ${context.riskLevel}，请特别关注安全信号。`;
    }
    if (context?.patientInfo) {
      prompt += `\n\n来访者背景信息：${context.patientInfo}`;
    }

    return prompt;
  }
}

export default new AiService();
