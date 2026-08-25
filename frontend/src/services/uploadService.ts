import type { DatasetProfile } from '../types';
import { buildApiUrl } from './api';
import { registerDatasetProfile } from './datasetService';

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
  quality_score?: number;
  column_analysis?: Record<string, any>;
  numeric_analysis?: Record<string, any>;
  categorical_analysis?: Record<string, any>;
  recommendations?: Array<any>;
  insights?: Array<any>;
  workspace_id?: string | null;
  project_id?: string | null;
  workspace_dataset_id?: string | null;
};

type UploadContext = {
  workspaceId?: string;
  projectId?: string;
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
    quality_score: payload.quality_score,
    column_analysis: payload.column_analysis,
    numeric_analysis: payload.numeric_analysis,
    categorical_analysis: payload.categorical_analysis,
    recommendations: payload.recommendations,
    insights: payload.insights,
    workspace_id: payload.workspace_id ?? undefined,
    project_id: payload.project_id ?? undefined,
    workspace_dataset_id: payload.workspace_dataset_id ?? undefined,
  };
}

export async function uploadDataset(file: File, context?: UploadContext): Promise<DatasetProfile> {
  const formData = new FormData();
  formData.append('file', file);
  if (context?.workspaceId) {
    formData.append('workspace_id', context.workspaceId);
  }
  if (context?.projectId) {
    formData.append('project_id', context.projectId);
  }

  const response = await fetch(buildApiUrl('/api/v1/datasets/upload'), {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || errorData.message || 'Upload failed');
  }

  const payload = (await response.json()) as FastApiUploadResponse;
  const profile = normalizeUploadResponse(payload);
  registerDatasetProfile(profile, file.name);
  return profile;
}

