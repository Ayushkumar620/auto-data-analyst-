import type {
  ReportDetail,
  ReportSummary,
} from '../types';
import { authedFetch, buildApiUrl } from './api';

export type CreateReportParams = {
  title: string;
  dataset_name?: string;
  report_type?: string;
  executive_summary: string;
  dataset_overview?: Record<string, unknown>;
  data_quality?: Record<string, unknown>;
  kpis?: Array<Record<string, unknown>>;
  charts?: Array<Record<string, unknown>>;
  insights?: Array<Record<string, unknown>>;
  evidence?: Array<Record<string, unknown>>;
  recommendations?: string[];
  forecast?: Record<string, unknown>;
  model_results?: Record<string, unknown>;
  monitoring?: Record<string, unknown>;
};

export async function listReports(): Promise<ReportSummary[]> {
  const res = await authedFetch(buildApiUrl('/api/v1/reports'));
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to list reports');
  }
  return res.json();
}

export async function getReportDetail(reportId: string): Promise<ReportDetail> {
  const res = await authedFetch(buildApiUrl(`/api/v1/reports/detail/${encodeURIComponent(reportId)}`));
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to retrieve report details');
  }
  return res.json();
}

export async function createReport(params: CreateReportParams): Promise<ReportDetail> {
  const res = await authedFetch(buildApiUrl('/api/v1/reports/create'), {
    method: 'POST',
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to create report');
  }
  return res.json();
}

export async function deleteReport(reportId: string): Promise<void> {
  const res = await authedFetch(buildApiUrl(`/api/v1/reports/${encodeURIComponent(reportId)}`), {
    method: 'DELETE',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to delete report');
  }
}

export async function downloadExecutivePdf(params: {
  title: string;
  command: string;
  explanation: string;
  kpis?: Record<string, unknown>;
  evidence_list?: Array<Record<string, unknown>>;
  dataset_summary?: Record<string, unknown>;
}): Promise<Blob> {
  const res = await authedFetch(buildApiUrl('/api/v1/reports/executive-pdf'), {
    method: 'POST',
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to generate Executive PDF');
  }
  return res.blob();
}
