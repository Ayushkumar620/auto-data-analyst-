const API_BASE_URL = (import.meta as ImportMeta & { env?: { VITE_API_BASE_URL?: string } }).env?.VITE_API_BASE_URL ?? '';

export function buildApiUrl(path: string): string {
  if (!path.startsWith('/')) {
    return `${API_BASE_URL}/${path}`;
  }
  return `${API_BASE_URL}${path}`;
}

const TOKEN_KEY = 'auth_token';

export function setAuthToken(token: string | null): void {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export async function authedFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers || {});
  const token = getAuthToken();
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  if (!headers.has('Content-Type') && !(init.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }
  const response = await fetch(input, { ...init, headers });
  // Auto-logout on auth failure
  if (response.status === 401) {
    setAuthToken(null);
  }
  return response;
}

export function apiJson<T>(response: Response): Promise<T> {
  return response.json();
}

export async function parseApiError(response: Response): Promise<never> {
  const errorData = await response.json().catch(() => ({}));
  throw new Error(errorData.detail || errorData.message || `Request failed with status ${response.status}`);
}
