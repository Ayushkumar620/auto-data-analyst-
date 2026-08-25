import type { DatasetItem, DatasetProfile } from '../types';
import { authedFetch, buildApiUrl } from './api';

const DATASETS_STORAGE_KEY = 'auto_analyst_datasets_registry';

function getStoredDatasets(): DatasetItem[] {
  try {
    const raw = localStorage.getItem(DATASETS_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveStoredDatasets(datasets: DatasetItem[]): void {
  try {
    localStorage.setItem(DATASETS_STORAGE_KEY, JSON.stringify(datasets));
  } catch {
    // Ignore storage quota errors
  }
}

export function registerDatasetProfile(profile: DatasetProfile, fileName?: string): DatasetItem {
  const id = profile.id || profile.workspace_dataset_id || `ds_${Date.now()}`;
  const name = fileName || profile.dataset_name || 'Dataset';
  const extension = name.includes('.') ? name.split('.').pop()?.toLowerCase() || 'csv' : 'csv';

  const item: DatasetItem = {
    id,
    name,
    file_type: profile.file_type || extension,
    rows: profile.rows || 0,
    columns: profile.columns || (profile.column_names ? profile.column_names.length : 0),
    created_at: profile.created_at || new Date().toISOString(),
    status: profile.status || 'ready',
    project_id: profile.project_id,
    workspace_id: profile.workspace_id,
    profile,
  };

  const current = getStoredDatasets();
  const existingIndex = current.findIndex((d) => d.id === id || d.name === name);
  if (existingIndex >= 0) {
    current[existingIndex] = item;
  } else {
    current.unshift(item);
  }
  saveStoredDatasets(current);
  return item;
}

export async function listDatasets(): Promise<DatasetItem[]> {
  const localDatasets = getStoredDatasets();

  try {
    // 1. Fetch from datasets endpoint
    const res = await authedFetch(buildApiUrl('/api/v1/datasets/'));
    let apiDatasets: DatasetItem[] = [];
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data.datasets)) {
        apiDatasets = data.datasets.map((d: any) => ({
          id: String(d.id || d.name),
          name: d.name || 'Dataset',
          file_type: d.file_type || 'csv',
          rows: d.rows || 0,
          columns: d.columns || 0,
          created_at: d.created_at || new Date().toISOString(),
          status: 'ready',
        }));
      }
    }

    // 2. Fetch project datasets from projects endpoint
    const projRes = await authedFetch(buildApiUrl('/api/v1/projects'));
    if (projRes.ok) {
      const projData = await projRes.json();
      if (Array.isArray(projData.projects)) {
        for (const p of projData.projects) {
          if (Array.isArray(p.datasets)) {
            for (const d of p.datasets) {
              const item: DatasetItem = {
                id: `proj_${p.id}_${d.id || d.name}`,
                name: d.name || 'Project Dataset',
                file_type: d.file_type || 'csv',
                rows: d.rows || 0,
                columns: d.columns || 0,
                created_at: d.created_at || p.created_at,
                status: 'ready',
                project_id: p.id,
              };
              if (!apiDatasets.some((ad) => ad.name === item.name)) {
                apiDatasets.push(item);
              }
            }
          }
        }
      }
    }

    // Merge API & local datasets by ID / Name
    const map = new Map<string, DatasetItem>();
    for (const d of localDatasets) {
      map.set(d.id, d);
      map.set(d.name, d);
    }
    for (const d of apiDatasets) {
      if (!map.has(d.id) && !map.has(d.name)) {
        map.set(d.id, d);
      }
    }

    return Array.from(new Set(map.values()));
  } catch {
    return localDatasets;
  }
}

export async function getDatasetById(id: string): Promise<DatasetItem | null> {
  const all = await listDatasets();
  return all.find((d) => d.id === id || d.name === id) || null;
}

export function deleteDataset(id: string): void {
  const current = getStoredDatasets();
  const updated = current.filter((d) => d.id !== id && d.name !== id);
  saveStoredDatasets(updated);
}

export async function cleanDataset(file: File): Promise<Record<string, unknown>> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await authedFetch(buildApiUrl('/api/v1/datasets/clean'), {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Cleaning failed');
  }

  return res.json();
}
