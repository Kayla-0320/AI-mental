import { randomUUID } from 'node:crypto';
import service, { SupportAgentService } from './support-agent.service';
import { AppError } from '../middlewares/errorHandler';

/** 站内建议后台进程：数据库租约防重，错误退避，停止时不再认领新任务。 */
export class SupportAgentWorker {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private running = false;
  private stopped = true;
  private afterId: string | undefined;
  private failures = 0;
  private shutdownRequested = false;
  readonly owner = randomUUID();
  constructor(private agent: SupportAgentService = service, private intervalMs = 60000) {}

  /** 返回不含用户信息的进程状态，供退避与关停验收。 */
  status(): { running: boolean; stopped: boolean; failures: number; backoffMs: number } {
    return { running: this.running, stopped: this.stopped, failures: this.failures, backoffMs: Math.min(300000, this.intervalMs * 2 ** Math.min(this.failures, 5)) };
  }

  /** 启动后自动恢复数据库中到期/租约失效的任务，不补发错过的历史时段。 */
  start(): void { if (!this.stopped) return; this.stopped = false; this.shutdownRequested = false; this.schedule(1000); }

  /** 等待当前批次收束；已认领而未执行的租约自然到期，可由其他实例恢复。 */
  async stop(): Promise<void> {
    this.stopped = true; this.shutdownRequested = true; if (this.timer) clearTimeout(this.timer); this.timer = null;
    while (this.running) await new Promise<void>(resolve => setTimeout(resolve, 25));
  }

  private schedule(delay: number): void {
    if (this.stopped) return;
    this.timer = setTimeout(() => { void this.tick().finally(() => this.schedule(Math.min(300000, this.intervalMs * 2 ** Math.min(this.failures, 5)))); }, delay);
    this.timer.unref();
  }

  /** 单次批处理可显式调用以验收租约语义；同实例不会重叠运行。 */
  async tick(): Promise<{ scanned: number; claimed: number; completed: number; error: boolean }> {
    if (this.running) return { scanned: 0, claimed: 0, completed: 0, error: false };
    this.running = true;
    const result = { scanned: 0, claimed: 0, completed: 0, error: false };
    try {
      const users = await this.agent.scanBackgroundUsers(this.afterId); result.scanned = users.length;
      this.afterId = users.length === 25 ? users[users.length - 1].id : undefined;
      for (const row of users) {
        if (this.shutdownRequested) break;
        try {
          const lease = await this.agent.claimBackground(row.userId, this.owner);
          if (lease) { result.claimed++; await this.agent.executeBackground(lease); result.completed++; }
        } catch (failure: unknown) {
          if (failure instanceof AppError) {
            console.warn('[SupportAgentWorker] 一条陪伴记录暂不可用，已跳过且未覆盖原记录');
            continue;
          }
          throw failure;
        }
      }
      this.failures = 0;
    } catch (failure: unknown) {
      this.failures++; result.error = true;
      // 不打印原话、连接串或异常对象；故障不生成虚假的成功状态。
      console.warn('[SupportAgentWorker] 数据服务暂不可用，本轮停止并退避');
    } finally { this.running = false; }
    return result;
  }
}
