import type { DatasetProfile } from '../types';
import { buildApiUrl } from './api';

type FastApiUploadResponse = {
  dataset?: { name?: string; rows?: number; columns?: number; file_type?: string };
  dataset_name?: string;
  rows?: number;
  columns?: number;
  column_names?: string[];
  missing_values?: number;
  duplicate_rows?: number;
  duplicates?: number;
  preview?: Array<Record<string, unknown>>;
  data_types?: Record<string, string>;
  memory_usage?: string;
};

function normalizeUploadResponse(payload: FastApiUploadResponse): DatasetProfile {
  const datasetName = payload.dataset_name ?? payload.dataset?.name ?? 'Uploaded Dataset';

  return {
    dataset_name: datasetName,
    rows: payload.rows ?? payload.dataset?.rows ?? 0,
    columns: payload.columns ?? payload.dataset?.columns ?? 0,
    column_names: payload.column_names ?? [],
    missing_values: payload.missing_values ?? 0,
    duplicates: payload.duplicates ?? payload.duplicate_rows ?? 0,
    preview: payload.preview ?? [],
    data_types: payload.data_types,
    memory_usage: payload.memory_usage,
  };
}

export async function uploadDataset(file: File): Promise<DatasetProfile> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(buildApiUrl('/api/v1/datasets/upload'), {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || errorData.message || 'Upload failed');
  }

  const payload = (await response.json()) as FastApiUploadResponse;
  return normalizeUploadResponse(payload);
}
