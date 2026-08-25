import type { AnalysisRecord } from '../types';
import { authedFetch, buildApiUrl } from './api';

const ANALYSES_STORAGE_KEY = 'auto_analyst_analyses_history';

function getStoredAnalyses(): AnalysisRecord[] {
  try {
    const raw = localStorage.getItem(ANALYSES_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveStoredAnalyses(analyses: AnalysisRecord[]): void {
  try {
    localStorage.setItem(ANALYSES_STORAGE_KEY, JSON.stringify(analyses));
  } catch {
    // Ignore storage quota
  }
}

export async function executeAnalysis(
  command: string,
  dataset?: Array<Record<string, unknown>>,
  sessionId: string = 'default_session',
  datasetName?: string,
): Promise<AnalysisRecord> {
  const res = await authedFetch(buildApiUrl('/api/v1/analyze'), {
    method: 'POST',
    body: JSON.stringify({
      command,
      dataset: dataset && dataset.length > 0 ? dataset : undefined,
      session_id: sessionId,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Analysis execution failed');
  }

  const data = await res.json();
  const record: AnalysisRecord = {
    id: `an_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
    command: data.command,
    user_intent: data.user_intent,
    required_operations: data.required_operations || [],
    final_explanation: data.final_explanation,
    execution_graph: data.execution_graph,
    evidence: data.evidence || [],
    dataset_summary: data.dataset_summary,
    validation_summary: data.validation_summary,
    duration_ms: data.duration_ms,
    dataset_name: datasetName || 'Dataset',
    created_at: new Date().toISOString(),
    status: 'completed',
  };

  const current = getStoredAnalyses();
  current.unshift(record);
  saveStoredAnalyses(current);
  return record;
}

export function listAnalyses(): AnalysisRecord[] {
  return getStoredAnalyses();
}

export function getAnalysisById(id: string): AnalysisRecord | null {
  const all = getStoredAnalyses();
  return all.find((a) => a.id === id) || null;
}

