/** 正式Node提供器 -> 实际Python HTTP -> 本地冻结模型；仅自编探针，无真人数据。 */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fetchPlatformModelReference, PersonalEvidence } from '../server/src/services/support-agent-context';

async function main(): Promise<void> {
  const repo = path.resolve(__dirname, '..');
  const python = process.env.ALGORITHM_PYTHON || 'D:/Develop/AIC/.venv/Scripts/python.exe';
  const program = [
    'from fastapi import FastAPI',
    'from api.innovation import router',
    'from perception.knowledge_moe.observation import infer_observation',
    'from datetime import datetime, timezone',
    'import hashlib, uvicorn',
    'text="模型预热"',
    'infer_observation(text,"warmup",hashlib.sha256(text.encode()).hexdigest(),datetime.now(timezone.utc).isoformat())',
    'app=FastAPI(); app.include_router(router,prefix="/api/v1")',
    'uvicorn.run(app,host="127.0.0.1",port=18279,log_level="warning")',
  ].join('\n');
  process.env.ALGORITHM_API_URL = 'http://127.0.0.1:18279';
  delete process.env.ROUND2_RESEARCH_TOKEN; delete process.env.SUPPORT_AGENT_MODEL_REFERENCE;
  const child = spawn(python, ['-X', 'utf8', '-c', program], { cwd: path.join(repo, 'algorithm'), windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
  let errors = ''; child.stderr.on('data', data => { errors += data.toString(); });
  try {
    let ready = false;
    for (let i = 0; i < 200; i++) {
      try { if ((await fetch(`${process.env.ALGORITHM_API_URL}/api/v1/innovation/capabilities`)).ok) { ready = true; break; } } catch { /* 等待本地加载 */ }
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    assert.ok(ready, `Python test service unavailable: ${errors.slice(-800)}`);
    const quote = '我想把刚才的意思说清楚，先不需要建议。';
    const evidence: PersonalEvidence = { id: 'test-evidence', quote, sourceId: 'test-source', sourceSha256: createHash('sha256').update(quote).digest('hex'),
      origin: 'user_correction', conversationId: 'synthetic-test', stateMessageId: 'test-state', revision: 1, start: 0, end: [...quote].length, observedAt: new Date().toISOString() };
    const result = await fetchPlatformModelReference(evidence);
    assert.equal(result?.available, true); assert.equal(result!.calibration.status, 'calibrated');
    assert.equal(result!.evidence_source.source_sha256, evidence.sourceSha256);
    assert.equal(result!.observed_at, evidence.observedAt);
    process.env.SUPPORT_AGENT_MODEL_REFERENCE = '0';
    assert.equal(await fetchPlatformModelReference(evidence), null);
    const report = { status: 'passed', real_local_model: true, research_token_required: false, source_and_time_verified: true,
      disabled_provider_returns_null: true, task: result!.task_id, model_id: result!.artifact_sha256, human_participants: 0 };
    const output = path.join(repo, 'output/innovation-integration'); fs.mkdirSync(output, { recursive: true });
    fs.writeFileSync(path.join(output, 'production-bridge.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  } finally { child.kill(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
