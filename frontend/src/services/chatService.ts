import { buildApiUrl, authedFetch, parseApiError } from './api';

export type ChatResponse = {
  message: string;
  intent: string;
  status: string;
  evidence: Record<string, unknown>;
  visualization: { data: unknown[]; layout?: Record<string, unknown> } | null;
  suggested_questions: string[];
};

export async function sendChatMessage(file: File, message: string, sessionId = 'default'): Promise<ChatResponse> {
  const form = new FormData();
  form.append('file', file);
  form.append('message', message);
  form.append('session_id', sessionId);

  const res = await authedFetch(buildApiUrl('/api/v1/chat'), {
    method: 'POST',
    body: form,
  });
  if (!res.ok) return parseApiError(res);
  const payload = await res.json();
  return {
    message: payload.message ?? '',
    intent: payload.intent ?? '',
    status: payload.status ?? '',
    evidence: payload.evidence ?? {},
    visualization: payload.visualization ?? null,
    suggested_questions: payload.suggested_questions ?? [],
  };
}
