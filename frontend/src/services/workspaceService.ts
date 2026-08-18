import type { Project, Workspace } from '../types';
import { buildApiUrl } from './api';

type CreateWorkspaceInput = {
  name: string;
  owner: string;
};

type CreateProjectInput = {
  workspaceId: string;
  name: string;
};

async function parseError(response: Response): Promise<never> {
  const errorData = await response.json().catch(() => ({}));
  throw new Error(errorData.detail || errorData.message || 'Workspace request failed');
}

export async function createWorkspace(input: CreateWorkspaceInput): Promise<Workspace> {
  const response = await fetch(buildApiUrl('/api/v1/workspaces'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });

  if (!response.ok) {
    return parseError(response);
  }

  return response.json();
}

export async function createProject(input: CreateProjectInput): Promise<Project> {
  const response = await fetch(buildApiUrl(`/api/v1/workspaces/${input.workspaceId}/projects`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: input.name }),
  });

  if (!response.ok) {
    return parseError(response);
  }

  return response.json();
}
