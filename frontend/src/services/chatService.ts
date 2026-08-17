import { buildApiUrl, authedFetch, parseApiError } from './api';

export type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
};

export type ChatRequest = {
  message: string;
  project_id?: number;
};

export type ChatResponse = {
  response: string;
  tool_calls?: unknown[];
};

export async function sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
  const res = await authedFetch(buildApiUrl('/api/v1/chat/message'), {
    method: 'POST',
    body: JSON.stringify(payload),
  });
  if (!res.ok) return parseApiError(res);
  return res.json();
}