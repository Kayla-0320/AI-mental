export interface SignalQuality {
  quality_score: number; reason_codes: string[]; metrics: Record<string, number>;
}
export interface InnovationReference {
  available: boolean; model_id?: string; observed_modalities: string[];
  expert_names: string[]; expert_weights: number[]; warnings: string[];
  quality: { visual?: SignalQuality }; reason?: string;
  knowledge_matches: Array<{ id: string; text: string; source_url: string; review_status: string; relations: Array<{ predicate: string; target: string }> }>;
}
const object = (value: unknown): Record<string, unknown> | null => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null;
const strings = (value: unknown): string[] => Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string').slice(0, 32) : [];

/** 验证历史JSON里的展示字段，损坏记录保持无结果，不用默认中性值补齐。 */
export function parseInnovation(value: unknown): InnovationReference | null {
  const row = object(value);
  if (!row || typeof row.available !== 'boolean') return null;
  const names = strings(row.expert_names);
  const weights = Array.isArray(row.expert_weights) && row.expert_weights.every(v => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1) ? row.expert_weights as number[] : [];
  const quality = object(object(row.quality)?.visual);
  const metrics = object(quality?.metrics);
  const visual = quality && typeof quality.quality_score === 'number' && Number.isFinite(quality.quality_score) && quality.quality_score >= 0 && quality.quality_score <= 1 ? {
    quality_score: quality.quality_score, reason_codes: strings(quality.reason_codes),
    metrics: Object.fromEntries(Object.entries(metrics || {}).filter(([, v]) => typeof v === 'number' && Number.isFinite(v))) as Record<string, number>,
  } : undefined;
  const knowledge = Array.isArray(row.knowledge_matches) ? row.knowledge_matches.map(object).filter((entry): entry is Record<string, unknown> => !!entry)
    .filter(entry => ['id', 'text', 'source_url', 'review_status'].every(key => typeof entry[key] === 'string'))
    .slice(0, 4).map(entry => ({ id: String(entry.id), text: String(entry.text), source_url: String(entry.source_url), review_status: String(entry.review_status),
      relations: Array.isArray(entry.relations) ? entry.relations.map(object).filter((relation): relation is Record<string, unknown> => !!relation && typeof relation.predicate === 'string' && typeof relation.target === 'string')
        .slice(0, 8).map(relation => ({ predicate: String(relation.predicate), target: String(relation.target) })) : [],
    })) : [];
  return { available: row.available, model_id: typeof row.model_id === 'string' ? row.model_id : undefined,
    observed_modalities: strings(row.observed_modalities), expert_names: names.length === weights.length ? names : [],
    expert_weights: names.length === weights.length ? weights : [], warnings: strings(row.warnings), quality: { visual },
    knowledge_matches: knowledge, reason: typeof row.reason === 'string' ? row.reason : undefined };
}

export function innovationFromSentiment(sentiment?: string): InnovationReference | null {
  try { return parseInnovation(JSON.parse(sentiment || '{}').innovation); } catch { return null; }
}

export const expertLabel = (name: string): string => ({ text: '文字专家', visual: '画面专家', audio: '声音专家',
  'text+visual': '图文联合专家', 'text+audio': '文字与声音专家', 'audio+visual': '声音与画面专家',
  'text+audio+visual': '多模态联合专家', candidate_knowledge: '知识概念专家' }[name] || name);

export const qualityReason = (reason: string): string => ({ underexposure_proxy: '画面过暗', overexposure_proxy: '画面过亮',
  low_contrast_proxy: '画面对比度偏低', blur_proxy: '画面清晰度偏低', uniform_center_occlusion_proxy: '中心存在大块均匀遮挡',
  repeated_frames_proxy: '采样画面重复', frame_timing_unverified: '未单独核验帧间时序', target_face_presence_unverified: '未在服务端独立核验人脸',
  missing_video_frames: '缺少画面帧', no_confirmed_target_face: '未确认目标人脸' }[reason] || reason);
