import { Response } from 'express';
import { AuthRequest } from '../middlewares/auth';
import consultationService from '../services/consultation.service';

class ConsultationController {
  async createConversation(req: AuthRequest, res: Response) {
    try {
      const result = await consultationService.createConversation(req.userId!, req.body.title);
      res.status(201).json({ code: 0, message: '创建成功', data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getConversations(req: AuthRequest, res: Response) {
    try {
      const page = parseInt(req.query.page as string) || 1;
      const pageSize = parseInt(req.query.pageSize as string) || 20;
      const result = await consultationService.getConversations(req.userId!, page, pageSize);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async getMessages(req: AuthRequest, res: Response) {
    try {
      const conversationId = req.params.conversationId as string;
      const page = parseInt(req.query.page as string) || 1;
      const pageSize = parseInt(req.query.pageSize as string) || 50;
      const result = await consultationService.getConversationMessages(conversationId, req.userId!, page, pageSize);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async sendMessage(req: AuthRequest, res: Response) {
    try {
      const conversationId = req.params.conversationId as string;
      const { content, contentType } = req.body;
      const result = await consultationService.sendMessage(conversationId, req.userId!, content, contentType);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  /**
   * 流式发送消息（SSE）—— 逐句上屏，安全审计仍是最终权威。
   *
   * 为什么不用 `EventSource`：它只支持 GET、无法携带 Authorization 头，而本接口是
   * POST + Bearer。所以这里就是一个普通 POST，只是响应体是 `text/event-stream`；
   * 浏览器端用 `fetch` + `ReadableStream` 手工分帧（见 client/src/services/index.ts）。
   *
   * 事件名与语义见 `algorithm-bridge.ts` 的 `ChatStreamEventName`：
   * `user` → `meta` → `delta`* → [`revise`] → `done`，失败为 `error`。
   *
   * ⚠️ SSE 一旦写出响应头就无法再改 HTTP 状态码，因此**所有可预检的错误都必须在
   * `writeHead` 之前返回**；进入流之后的服务层错误只能通过 `error` 事件上报。
   */
  async sendMessageStream(req: AuthRequest, res: Response) {
    const conversationId = req.params.conversationId as string;
    const { content, contentType = 'text' } = req.body;

    if (typeof content !== 'string' || !content.trim()) {
      res.status(400).json({ code: 400, message: '消息内容不能为空' });
      return;
    }

    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      // no-transform：禁止中间层压缩/改写，否则流式会退化成一次性返回
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
      'X-Accel-Buffering': 'no',
    });
    if (typeof res.flushHeaders === 'function') res.flushHeaders();

    let closed = false;
    const abort = new AbortController();

    const emit = (event: string, data: Record<string, unknown>) => {
      if (closed || res.writableEnded) return;
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    // 浏览器断开（关标签页 / 路由切走）：中止上游 LLM 调用。
    // 这既省 token，也避免把回复写进一个已经没人看的会话。
    req.on('close', () => {
      closed = true;
      abort.abort();
    });

    try {
      await consultationService.sendMessageStream(
        conversationId,
        req.userId!,
        content,
        contentType,
        emit,
        abort.signal,
      );
    } catch (error: any) {
      // 能走到这里，说明服务层连一个 error 事件都没能发出（例如会话不存在）。
      // 绝不能改成 res.status(...).json()：响应头已经发出去了。
      emit('error', {
        failed: true,
        errorCode: error?.statusCode === 404 ? 'NOT_FOUND' : 'STREAM_FAILED',
        errorMessage: error?.message || '对话服务异常，请稍后再试',
      });
    } finally {
      if (!res.writableEnded) res.end();
    }
  }

  async deleteConversation(req: AuthRequest, res: Response) {
    try {
      const conversationId = req.params.conversationId as string;
      await consultationService.deleteConversation(conversationId, req.userId!);
      res.json({ code: 0, message: '删除成功' });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  /**
   * 批量删除对话。请求体 `{ conversationIds: string[] }`。
   *
   * 挂在集合路径上（DELETE /consultation/conversations）而不是
   * /conversations/batch —— 后者会和 /conversations/:conversationId 抢路由，
   * 将来若新增一个 id 恰好叫 batch 的会话就会静默删错东西。
   */
  async deleteConversations(req: AuthRequest, res: Response) {
    try {
      const { conversationIds } = req.body as { conversationIds?: unknown };
      if (!Array.isArray(conversationIds)) {
        res.status(400).json({ code: 400, message: 'conversationIds 必须为数组' });
        return;
      }
      const result = await consultationService.deleteConversations(
        conversationIds as string[],
        req.userId!,
      );
      res.json({ code: 0, message: '删除成功', data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }

  async sendSocraticMessage(req: AuthRequest, res: Response) {
    try {
      const conversationId = req.params.conversationId as string;
      const { content } = req.body;
      const result = await consultationService.sendSocraticMessage(conversationId, req.userId!, content);
      res.json({ code: 0, data: result });
    } catch (error: any) {
      res.status(error.statusCode || 500).json({ code: error.code || 500, message: error.message });
    }
  }
}

export default new ConsultationController();
