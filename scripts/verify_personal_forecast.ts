import assert from 'node:assert/strict';
import { forecastSelfReports } from '../server/src/services/personal-forecast';

const now = new Date('2026-11-17T18:00:00+08:00');
const point = (day: number, score: number) => ({ score, recordedAt: new Date(`2026-11-${day}T12:00:00+08:00`) });
assert.equal(forecastSelfReports([], now).status, 'abstain');
assert.equal(forecastSelfReports([point(17, 5), point(17, 6), point(17, 7)], now).reason, 'insufficient_history');
assert.equal(forecastSelfReports([point(14, 5), point(15, 6), point(16, 7)], now).reason, 'latest_day_missing');
assert.equal(forecastSelfReports([point(10, 5), point(11, 6), point(17, 7)], now).reason, 'insufficient_history');
const result = forecastSelfReports([point(17, 9), point(15, 5), point(16, 7), point(17, 7), point(18, 1)], now);
assert.equal(result.status, 'experimental');
assert.equal(result.observedDays, 3); // 同日9和7先聚合为8。
assert.equal(result.score, 6.68); assert.equal(result.targetDate, '2026-11-18');
assert.equal(result.validatedForProject, false);
// 上海零点属于新一天，避免执行主机时区导致跨日错位。
assert.equal(forecastSelfReports([point(17, 5)], new Date('2026-11-17T16:01:00Z')).reason, 'latest_day_missing');
console.log('personal forecast: calendar aggregation, future exclusion, gap reset, cold start and EMA passed');
