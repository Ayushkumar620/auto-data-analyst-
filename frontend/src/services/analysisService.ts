import { buildApiUrl } from './api';

type AnalysisContext = {
  workspaceId?: string;
  projectId?: string | number;
};

function appendContext(formData: FormData, context?: AnalysisContext): void {
  if (context?.workspaceId) {
    formData.append('workspace_id', context.workspaceId);
  }
  if (context?.projectId !== undefined && context?.projectId !== null) {
    formData.append('project_id', String(context.projectId));
  }
}

export type InsightItem = {
  type: string;
  title: string;
  description: string;
  severity?: string;
  confidence?: string;
  recommendation?: string;
};

export type InsightsResponse = {
  facts?: Record<string, unknown>;
  insights: InsightItem[];
};

export type ForecastPoint = {
  date: string;
  prediction: number;
  lower: number;
  upper: number;
};

export type ForecastResponse = {
  status: string;
  target: string;
  date_column: string;
  frequency: string;
  horizon: number;
  model: string;
  metrics: Record<string, number | null>;
  forecast: ForecastPoint[];
  visualization?: Record<string, unknown>;
};

export type ReportResponse = {
  status: string;
  report_id: string;
  download_url: string;
  report?: Record<string, unknown>;
};

async function parseError(response: Response): Promise<never> {
  const errorData = await response.json().catch(() => ({}));
  throw new Error(errorData.detail || errorData.message || 'Request failed');
}

export async function generateEda(file: File, context?: AnalysisContext): Promise<Record<string, unknown>> {
  const formData = new FormData();
  formData.append('file', file);
  appendContext(formData, context);

  const response = await fetch(buildApiUrl('/api/v1/datasets/eda'), {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    return parseError(response);
  }

  return response.json();
}

export async function generateInsights(file: File, context?: AnalysisContext): Promise<InsightsResponse> {
  const formData = new FormData();
  formData.append('file', file);
  appendContext(formData, context);

  const response = await fetch(buildApiUrl('/api/v1/insights/generate'), {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    return parseError(response);
  }

  const payload = (await response.json()) as InsightsResponse;
  return {
    facts: payload.facts,
    insights: payload.insights ?? [],
  };
}

export async function generateForecast(file: File, horizon = 6, context?: AnalysisContext): Promise<ForecastResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('horizon', String(horizon));
  appendContext(formData, context);

  const response = await fetch(buildApiUrl('/api/v1/forecast'), {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    return parseError(response);
  }

  return response.json();
}

export async function generateReport(
  file: File,
  outputFormat: 'pdf' | 'excel' | 'powerpoint' = 'pdf',
  context?: AnalysisContext,
): Promise<ReportResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('output_format', outputFormat);
  appendContext(formData, context);

  const response = await fetch(buildApiUrl('/api/v1/reports/generate'), {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    return parseError(response);
  }

  const payload = (await response.json()) as ReportResponse;
  return {
    ...payload,
    download_url: buildApiUrl(payload.download_url),
  };
}
