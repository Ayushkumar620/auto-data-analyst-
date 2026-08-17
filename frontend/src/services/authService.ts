import { buildApiUrl, authedFetch, parseApiError } from './api';

export type UserOut = {
  id: number;
  email: string;
  username: string;
  is_active: boolean;
};

export type TokenResponse = {
  access_token: string;
  token_type: string;
  user: UserOut;
};

export type LoginRequest = { email: string; password: string };
export type UserCreate = { email: string; username: string; password: string };

export async function login(payload: LoginRequest): Promise<TokenResponse> {
  const res = await authedFetch(buildApiUrl('/api/v1/auth/login'), {
    method: 'POST',
    body: JSON.stringify(payload),
  });
  if (!res.ok) return parseApiError(res);
  return res.json();
}

export async function register(payload: UserCreate): Promise<TokenResponse> {
  const res = await authedFetch(buildApiUrl('/api/v1/auth/register'), {
    method: 'POST',
    body: JSON.stringify(payload),
  });
  if (!res.ok) return parseApiError(res);
  return res.json();
}

export async function fetchCurrentUser(): Promise<UserOut> {
  const res = await authedFetch(buildApiUrl('/api/v1/auth/me'));
  if (!res.ok) return parseApiError(res);
  return res.json();
}

export function logout(): void {
  localStorage.removeItem('auth_token');
}

export async function listProjects(): Promise<{ id: number; name: string; description: string | null; datasets: unknown[] }[]> {
  const res = await authedFetch(buildApiUrl('/api/v1/projects'));
  if (!res.ok) return parseApiError(res);
  const data = await res.json();
  return data.projects ?? [];
}

export async function createProject(name: string, description?: string) {
  const res = await authedFetch(buildApiUrl('/api/v1/projects'), {
    method: 'POST',
    body: JSON.stringify({ name, description }),
  });
  if (!res.ok) return parseApiError(res);
  return res.json();
}