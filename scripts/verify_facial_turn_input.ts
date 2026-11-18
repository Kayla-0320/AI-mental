import assert from 'node:assert/strict';
import { buildFacialTurnInput } from '../client/src/services/facialLiveState';
import { defaultFacial } from '../client/src/types/multimodal.types';

const now = Date.now();
const facial = { ...defaultFacial, isDetecting: true, facePresent: true, timestamp: now };
const frames = [now - 2200, now - 1100, now - 10].map(sampledAt => ({ sampledAt, jpeg: 'data:image/jpeg;base64,fixture' }));
for (const status of ['loading', 'unavailable', 'disabled'] as const) {
  const result = buildFacialTurnInput({ ...facial, faceEmotionStatus: status }, frames, now);
  assert.equal(result?.visualFrames?.length, 3, '表情模型状态不应阻断原始像素检查');
  assert.equal('probabilities' in result!, false, '不得伪造表情概率');
}
assert.equal(buildFacialTurnInput({ ...facial, facePresent: false }, frames, now), undefined);
assert.equal(buildFacialTurnInput({ ...facial, isDetecting: false }, frames, now), undefined);
assert.equal(buildFacialTurnInput({ ...facial, timestamp: now - 1001 }, frames, now), undefined);
assert.equal(buildFacialTurnInput(facial, frames.slice(0, 2), now), undefined);
assert.equal(buildFacialTurnInput(facial, frames.map(f => ({...f, sampledAt: now - 11000})), now), undefined);
assert.equal(buildFacialTurnInput(facial, [...frames].reverse(), now), undefined);
const snapshot = buildFacialTurnInput(facial, frames, now)!;
snapshot.visualFrames![0].jpeg = 'changed';
assert.notEqual(frames[0].jpeg, 'changed');
console.log('PASS: independent pixel evidence, no invented emotions, freshness, presence, three-frame contract, snapshot isolation');
