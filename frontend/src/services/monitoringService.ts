import type {
  MonitoringOverviewData,
  MonitoringResult,
} from '../types';
import { authedFetch, buildApiUrl } from './api';

export type RunMonitoringParams = {
  model_id: string;
  current_dataset: Array<Record<string, unknown>>;
  reference_dataset?: Array<Record<string, unknown>>;
  feature_columns?: string[];
  target_column?: string;
  thresholds?: Record<string, number>;
};

export async function runMonitoring(params: RunMonitoringParams): Promise<MonitoringResult> {
  const res = await authedFetch(buildApiUrl('/api/v1/monitoring/run'), {
    method: 'POST',
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Monitoring evaluation failed');
  }

  return res.json();
}

export async function getMonitoringHistory(modelId?: string): Promise<MonitoringResult[]> {
  const query = modelId ? `?model_id=${encodeURIComponent(modelId)}` : '';
  const res = await authedFetch(buildApiUrl(`/api/v1/monitoring/history${query}`));

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to retrieve monitoring history');
  }

  return res.json();
}

export async function getMonitoringOverview(): Promise<MonitoringOverviewData> {
  const res = await authedFetch(buildApiUrl('/api/v1/monitoring/overview'));

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to retrieve monitoring overview');
  }

  return res.json();
}
