import { buildApiUrl, authedFetch, parseApiError } from './api';
import type { ChatSessionApiResponse } from '../types';

export type ChatResponse = {
  message: string;
  intent: string;
  status: string;
  evidence: Record<string, unknown>;
  visualization: { data: unknown[]; layout?: Record<string, unknown> } | null;
  suggested_questions: string[];
  command_result?: Record<string, unknown> | null;
};

export type CommandExecutionResponse = {
  command: string;
  user_intent: string;
  required_operations: string[];
  selected_agents: string[];
  model_selection_summary: Record<string, unknown> | null;
  execution_steps: Array<Record<string, unknown>>;
  validation_summary: {
    status: string;
    critical_issues: number;
    warnings: number;
    diagnostics: Record<string, unknown>;
  };
  final_explanation: string;
  evidence: Array<Record<string, unknown>>;
  visualization: { data?: unknown[]; layout?: Record<string, unknown>; chart_type?: string; x?: string; y?: string; title?: string } | null;
  dataset_summary: Record<string, unknown>;
  duration_ms: number;
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
    command_result: payload.command_result ?? null,
  };
}

export async function executeCommand(file?: File | null, command: string = '', datasetId?: string): Promise<CommandExecutionResponse> {
  const form = new FormData();
  if (file) form.append('file', file);
  form.append('command', command);
  if (datasetId) form.append('dataset_id', datasetId);

  const res = await authedFetch(buildApiUrl('/api/v1/chat/command'), {
    method: 'POST',
    body: form,
  });
  if (!res.ok) return parseApiError(res);
  return await res.json();
}

export async function sendConversationalMessage(
  message: string,
  sessionId: string = 'default_session',
  dataset?: Array<Record<string, unknown>>,
  datasetName?: string,
): Promise<ChatSessionApiResponse> {
  const res = await authedFetch(buildApiUrl('/api/v1/chat/session'), {
    method: 'POST',
    body: JSON.stringify({
      message,
      session_id: sessionId,
      dataset: dataset || null,
      dataset_name: datasetName || 'dataset',
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Conversational analyst request failed');
  }

  return res.json();
}

export async function getConversationalSession(sessionId: string): Promise<Record<string, unknown>> {
  const res = await authedFetch(buildApiUrl(`/api/v1/chat/session/${encodeURIComponent(sessionId)}`));
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to retrieve session');
  }
  return res.json();
}
