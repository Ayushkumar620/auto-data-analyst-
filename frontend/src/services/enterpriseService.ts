import { authedFetch, buildApiUrl, parseApiError } from './api';

export type DBConnection = {
  connection_id: string;
  name: string;
  db_type: string;
  host?: string;
  port?: number;
  database: string;
  username?: string;
  created_at: string;
};

export type TableSchema = {
  table_name: string;
  row_count_estimate?: number;
  columns: Array<{ name: string; type: string }>;
};

export type QueryResult = {
  columns: string[];
  rows: Array<Record<string, any>>;
  total_rows: number;
  execution_time_ms: number;
  query: string;
};

export type PIIDetection = {
  column: string;
  pii_type: string;
  confidence: number;
  sample_matches_count: number;
  total_samples_tested: number;
  suggested_strategy: 'mask' | 'hash' | 'anonymize';
};

export type PIIScanReport = {
  has_pii: boolean;
  total_pii_columns: number;
  detections: PIIDetection[];
  scanned_columns: number;
  scanned_rows: number;
};

export type AuditEvent = {
  event_id: string;
  timestamp: string;
  user_id: string;
  action: string;
  resource_type: string;
  resource_id: string;
  details: Record<string, any>;
  signature_hash: string;
};

export type AlertRule = {
  rule_id: string;
  name: string;
  channel_type: string;
  webhook_url?: string;
  email_recipient?: string;
  trigger_on_drift: boolean;
  trigger_on_degradation: boolean;
  enabled: boolean;
};

export type ModelDeployment = {
  deployment_id: string;
  model_id: string;
  endpoint_name: string;
  endpoint_path: string;
  status: string;
  target_column: string;
  features: string[];
  total_invocations: number;
  avg_latency_ms: number;
  deployed_at: string;
};

// Database Connectors APIs
export async function listDBConnections(): Promise<DBConnection[]> {
  const res = await authedFetch(buildApiUrl('/api/v1/connectors'));
  if (!res.ok) await parseApiError(res);
  return res.json();
}

export async function testDBConnection(data: any): Promise<{ success: boolean; message: string }> {
  const res = await authedFetch(buildApiUrl('/api/v1/connectors/test'), {
    method: 'POST',
    body: JSON.stringify(data),
  });
  if (!res.ok) await parseApiError(res);
  return res.json();
}

export async function createDBConnection(data: any): Promise<DBConnection> {
  const res = await authedFetch(buildApiUrl('/api/v1/connectors/create'), {
    method: 'POST',
    body: JSON.stringify(data),
  });
  if (!res.ok) await parseApiError(res);
  return res.json();
}

export async function inspectConnectionTables(connectionId: string): Promise<TableSchema[]> {
  const res = await authedFetch(buildApiUrl(`/api/v1/connectors/${connectionId}/tables`));
  if (!res.ok) await parseApiError(res);
  return res.json();
}

export async function executeDBQuery(connectionId: string, query: string, limit = 500): Promise<QueryResult> {
  const res = await authedFetch(buildApiUrl(`/api/v1/connectors/${connectionId}/query`), {
    method: 'POST',
    body: JSON.stringify({ query, limit }),
  });
  if (!res.ok) await parseApiError(res);
  return res.json();
}

// Governance & PII APIs
export async function scanDatasetPII(data: Array<Record<string, any>>): Promise<PIIScanReport> {
  const res = await authedFetch(buildApiUrl('/api/v1/governance/scan-pii'), {
    method: 'POST',
    body: JSON.stringify({ data }),
  });
  if (!res.ok) await parseApiError(res);
  return res.json();
}

export async function redactDataset(data: Array<Record<string, any>>, redactions?: Record<string, string>): Promise<any> {
  const res = await authedFetch(buildApiUrl('/api/v1/governance/redact-dataset'), {
    method: 'POST',
    body: JSON.stringify({ data, redactions }),
  });
  if (!res.ok) await parseApiError(res);
  return res.json();
}

export async function getAuditLogs(limit = 50): Promise<AuditEvent[]> {
  const res = await authedFetch(buildApiUrl(`/api/v1/governance/audit-logs?limit=${limit}`));
  if (!res.ok) await parseApiError(res);
  return res.json();
}

// Alerts APIs
export async function listAlertRules(): Promise<AlertRule[]> {
  const res = await authedFetch(buildApiUrl('/api/v1/alerts/rules'));
  if (!res.ok) await parseApiError(res);
  return res.json();
}

export async function saveAlertRule(data: Partial<AlertRule>): Promise<AlertRule> {
  const res = await authedFetch(buildApiUrl('/api/v1/alerts/rules'), {
    method: 'POST',
    body: JSON.stringify(data),
  });
  if (!res.ok) await parseApiError(res);
  return res.json();
}

export async function dispatchTestAlert(data: { rule_id?: string; title: string; message: string; severity?: string }): Promise<any> {
  const res = await authedFetch(buildApiUrl('/api/v1/alerts/test'), {
    method: 'POST',
    body: JSON.stringify(data),
  });
  if (!res.ok) await parseApiError(res);
  return res.json();
}

// Model Serving APIs
export async function deployModelEndpoint(modelId: string, endpointName?: string): Promise<ModelDeployment> {
  const res = await authedFetch(buildApiUrl('/api/v1/models/served/deploy'), {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId, endpoint_name: endpointName }),
  });
  if (!res.ok) await parseApiError(res);
  return res.json();
}

export async function listModelDeployments(): Promise<ModelDeployment[]> {
  const res = await authedFetch(buildApiUrl('/api/v1/models/served/endpoints'));
  if (!res.ok) await parseApiError(res);
  return res.json();
}

export async function predictViaEndpoint(deploymentId: string, records: Array<Record<string, any>>): Promise<any> {
  const res = await authedFetch(buildApiUrl(`/api/v1/models/served/${deploymentId}/predict`), {
    method: 'POST',
    body: JSON.stringify({ records }),
  });
  if (!res.ok) await parseApiError(res);
  return res.json();
}
