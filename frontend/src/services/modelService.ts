import type { InferenceResponse, ModelMetadata, ModelStatus } from '../types';
import { authedFetch, buildApiUrl } from './api';

export type ListModelsParams = {
  family?: string;
  problem_type?: string;
  status?: string;
};

export async function listModels(params?: ListModelsParams): Promise<ModelMetadata[]> {
  const query = new URLSearchParams();
  if (params?.family) query.set('family', params.family);
  if (params?.problem_type) query.set('problem_type', params.problem_type);
  if (params?.status) query.set('status', params.status);

  const url = buildApiUrl(`/api/v1/models${query.toString() ? `?${query.toString()}` : ''}`);
  const res = await authedFetch(url);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to retrieve models from registry');
  }

  return res.json();
}

export async function getModelMetadata(modelId: string): Promise<ModelMetadata> {
  const res = await authedFetch(buildApiUrl(`/api/v1/models/${encodeURIComponent(modelId)}`));

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to retrieve model '${modelId}'`);
  }

  return res.json();
}

export async function runModelInference(
  modelId: string,
  data: Array<Record<string, unknown>> | Record<string, unknown>,
): Promise<InferenceResponse> {
  const res = await authedFetch(buildApiUrl(`/api/v1/models/${encodeURIComponent(modelId)}/predict`), {
    method: 'POST',
    body: JSON.stringify({ data }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Inference execution failed');
  }

  return res.json();
}

export async function updateModelStatus(
  modelId: string,
  status: ModelStatus | string,
): Promise<{ status: string; model_id: string; new_status: string }> {
  const res = await authedFetch(buildApiUrl(`/api/v1/models/${encodeURIComponent(modelId)}/status`), {
    method: 'PATCH',
    body: JSON.stringify({ status }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to update model status');
  }

  return res.json();
}

export async function deleteModel(modelId: string): Promise<{ status: string; message: string }> {
  const res = await authedFetch(buildApiUrl(`/api/v1/models/${encodeURIComponent(modelId)}`), {
    method: 'DELETE',
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to delete model');
  }

  return res.json();
}

