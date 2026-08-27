export type DatasetProfile = {
  id?: string;
  dataset_name: string;
  rows: number;
  columns: number;
  column_names: string[];
  missing_values: number;
  duplicates: number;
  preview: Array<Record<string, unknown>>;
  data_types?: Record<string, string>;
  memory_usage?: string;
  quality_score?: number;
  workspace_id?: string;
  project_id?: string;
  workspace_dataset_id?: string;
  file_type?: string;
  created_at?: string;
  status?: string;
  column_analysis?: Record<string, ColumnMetadata>;
  numeric_analysis?: Record<string, NumericAnalysis>;
  categorical_analysis?: Record<string, CategoricalAnalysis>;
  recommendations?: Array<string | Record<string, unknown>>;
  insights?: Array<Record<string, unknown>>;
};

export type ColumnMetadata = {
  type?: string;
  dtype?: string;
  missing?: number;
  missing_percentage?: number;
  unique?: number;
  sample_values?: unknown[];
};

export type NumericAnalysis = {
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  median?: number;
  q25?: number;
  q75?: number;
};

export type CategoricalAnalysis = {
  top_categories?: Record<string, number>;
  cardinality?: number;
  mode?: string;
};

export type DatasetItem = {
  id: string;
  name: string;
  file_type: string;
  rows: number;
  columns: number;
  created_at?: string;
  status?: string;
  project_id?: string | number;
  workspace_id?: string;
  profile?: DatasetProfile;
};

export type Workspace = {
  id: string;
  name: string;
  owner: string;
  projects?: Project[];
};

export type Project = {
  id: string | number;
  workspace_id?: string;
  name: string;
  description?: string | null;
  created_at?: string;
  datasets?: Array<Record<string, unknown>>;
  sessions?: Array<Record<string, unknown>>;
};

export type AnalysisRecord = {
  id: string;
  command: string;
  user_intent?: string;
  required_operations?: string[];
  final_explanation?: string;
  execution_graph?: unknown;
  evidence?: Array<Record<string, unknown>>;
  dataset_summary?: Record<string, unknown>;
  validation_summary?: Record<string, unknown>;
  duration_ms?: number;
  dataset_name?: string;
  created_at: string;
  status: 'completed' | 'failed' | 'running';
};

// ==========================================
// Phase 3: Model Registry Types
// ==========================================
export type ModelStatus = 'active' | 'staging' | 'archived';
export type ModelFamily = 'traditional_ml' | 'ann' | 'cnn' | 'forecasting';
export type ProblemType =
  | 'binary_classification'
  | 'multiclass_classification'
  | 'regression'
  | 'time_series_forecast';

export type ModelMetadata = {
  model_id: string;
  name: string;
  version: number;
  model_family: ModelFamily | string;
  algorithm: string;
  problem_type: ProblemType | string;
  target_column: string;
  feature_columns: string[];
  feature_dtypes: Record<string, string>;
  hyperparameters: Record<string, unknown>;
  training_metrics: Record<string, number>;
  validation_metrics: Record<string, number>;
  primary_metric_name: string;
  primary_metric_value: number;
  loss_curve?: number[];
  created_at: string;
  status: ModelStatus | string;
  tags?: string[];
  preprocessor_meta?: Record<string, unknown>;
  reference_profile?: Record<string, unknown>;
  feature_importances?: Record<string, number>;
};

export type InferenceResponse = {
  model_id: string;
  predictions: Array<number | string>;
  probabilities?: Array<number[] | Record<string, number>>;
  duration_ms?: number;
  batch_size?: number;
};

// ==========================================
// Phase 4: Conversational AI Analyst Types
// ==========================================
export type ChatRole = 'user' | 'assistant' | 'system';

export type EvidenceItem = {
  dataset_id?: string;
  dataset_name?: string;
  columns?: string[];
  operation?: string;
  calculation?: string;
  source_reference?: string;
  result?: unknown;
  confidence?: number;
  claim_type?: string;
  claim?: string;
  raw_value?: unknown;
  metadata?: Record<string, unknown>;
};

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  evidence?: EvidenceItem[];
  metadata?: Record<string, unknown>;
  timestamp: string;
};

export type AnalystSession = {
  session_id: string;
  dataset_name?: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
};

export type ChatSessionApiResponse = {
  session_id: string;
  message: string;
  response: string;
  evidence: EvidenceItem[];
  dataset_context?: {
    dataset_name?: string;
    row_count?: number;
    column_count?: number;
    numeric_columns?: string[];
    categorical_columns?: string[];
  };
  metadata?: Record<string, unknown>;
  created_at: string;
};

// ==========================================
// Phase 5: Forecasting & What-If Types
// ==========================================
export type ForecastPoint = {
  timestamp: string;
  prediction: number;
  lower_bound: number;
  upper_bound: number;
};

export type ForecastResult = {
  model_id: string;
  model_name: string;
  model_family: string;
  target: string;
  time_column: string;
  frequency: string;
  forecast_horizon: number;
  predictions: ForecastPoint[];
  confidence_level: number;
  validation_metrics: Record<string, number>;
  baseline_metrics: Record<string, number>;
  assumptions: string[];
  warnings: string[];
  limitations: string[];
  evidence: EvidenceItem[];
  confidence: number;
  status: string;
};

export type ScenarioResult = {
  scenario_name: string;
  target_metric: string;
  baseline_value: number;
  scenario_value: number;
  absolute_difference: number;
  percentage_difference: number;
  assumptions: string[];
  limitations: string[];
  evidence: EvidenceItem[];
  confidence: number;
};

export type ScenarioComparison = {
  target_metric: string;
  baseline_value: number;
  scenarios: ScenarioResult[];
  ranked_scenarios: ScenarioResult[];
  summary: string;
  evidence: EvidenceItem[];
};

// ==========================================
// Phase 6: Model Monitoring & Drift Types
// ==========================================
export type DriftSeverityLevel = 'NONE' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | 'WARNING' | 'HEALTHY';

export type FeatureDriftResult = {
  feature_name: string;
  drift_detected: boolean;
  drift_score: number;
  statistical_test: string;
  p_value?: number | null;
  reference_statistics?: Record<string, unknown>;
  current_statistics?: Record<string, unknown>;
  threshold: number;
  severity: DriftSeverityLevel;
  confidence?: number;
};

export type DatasetDriftReport = {
  dataset_id: string;
  reference_dataset_id: string;
  features_checked: string[];
  drifted_features: string[];
  drift_percentage: number;
  overall_drift: boolean;
  schema_drift_detected: boolean;
  schema_changes?: Record<string, unknown>;
  data_quality_changes?: Record<string, unknown>;
  feature_results: Record<string, FeatureDriftResult>;
  severity: DriftSeverityLevel;
  warnings?: string[];
  confidence?: number;
};

export type ModelPerformanceReport = {
  model_id: string;
  reference_metrics: Record<string, number>;
  current_metrics: Record<string, number>;
  metric_changes: Record<string, number>;
  degradation_detected: boolean;
  target_monitoring_status: string;
  evaluation_dataset_rows: number;
  confidence?: number;
};

export type PredictionDriftReport = {
  model_id: string;
  prediction_drift_detected: boolean;
  statistical_test: string;
  drift_score: number;
  p_value?: number | null;
  reference_prediction_stats?: Record<string, unknown>;
  current_prediction_stats?: Record<string, unknown>;
  confidence?: number;
};

export type MonitoringResult = {
  run_id?: string;
  model_id: string;
  status: string;
  overall_severity: DriftSeverityLevel;
  data_drift?: DatasetDriftReport;
  prediction_drift?: PredictionDriftReport;
  performance_drift?: ModelPerformanceReport;
  data_quality?: Record<string, unknown>;
  recommendations: string[];
  warnings: string[];
  evidence?: EvidenceItem[];
  confidence?: number;
  timestamp?: string;
  executed_at?: string;
};

export type MonitoringOverviewData = {
  total_models: number;
  monitored_models: number;
  healthy_models: number;
  warning_models: number;
  critical_models: number;
  total_runs: number;
  last_run_timestamp?: string | null;
};

// ==========================================
// Phase 7: Reports & Decision Deliverables
// ==========================================
export type ReportKPI = {
  name: string;
  value: number | string;
  unit?: string;
  change?: number;
  formatted?: string;
};

export type ReportInsightItem = {
  title: string;
  narrative: string;
  metric?: string;
  evidence?: string;
  impact?: string;
};

export type ReportSummary = {
  report_id: string;
  title: string;
  dataset_name: string;
  report_type: string;
  created_at: string;
  status: string;
  executive_summary: string;
  kpi_count: number;
  insight_count: number;
  recommendation_count: number;
  has_forecast: boolean;
  has_model: boolean;
  has_monitoring: boolean;
};

export type ReportDetail = {
  report_id: string;
  title: string;
  dataset_name: string;
  report_type: string;
  created_at: string;
  status: string;
  executive_summary: string;
  dataset_overview?: Record<string, unknown>;
  data_quality?: Record<string, unknown>;
  kpis?: ReportKPI[];
  charts?: Array<{ id: string; type: string; title: string; data: unknown }>;
  insights?: ReportInsightItem[];
  evidence?: EvidenceItem[];
  recommendations?: string[];
  forecast?: Record<string, unknown>;
  model_results?: Record<string, unknown>;
  monitoring?: Record<string, unknown>;
  download_url?: string;
};

// ==========================================
// Milestone 7 Task 2: Synthesized Insights
// ==========================================
export type SynthesizedInsightItem = {
  insight_id: string;
  category: string;
  title: string;
  statement: string;
  evidence_refs?: Array<Record<string, unknown>>;
  supporting_metrics?: Record<string, unknown>;
  confidence: number;
  importance: number;
  assumptions?: string[];
  limitations?: string[];
  provenance?: Record<string, unknown>;
};

export type ContradictionItem = {
  contradiction_id: string;
  involved_insights: string[];
  conflicting_evidence: Array<Record<string, unknown>>;
  explanation: string;
  confidence: number;
  resolution: string;
};

export type SynthesisReportData = {
  executive_summary: string;
  key_insights: SynthesizedInsightItem[];
  important_findings: string[];
  data_quality_findings: SynthesizedInsightItem[];
  model_findings: SynthesizedInsightItem[];
  forecast_findings: SynthesizedInsightItem[];
  anomalies: SynthesizedInsightItem[];
  segments: SynthesizedInsightItem[];
  relationships: SynthesizedInsightItem[];
  cross_analysis_findings: SynthesizedInsightItem[];
  contradictions: ContradictionItem[];
  limitations: string[];
  recommended_next_questions: string[];
  overall_confidence: number;
  evidence?: Array<Record<string, unknown>>;
  metadata?: Record<string, unknown>;
};


// ==========================================
// Milestone 7 Task 3: Conversational Analytical Context
// ==========================================
export type ExecutionRecordItem = {
  execution_id: string;
  turn_id: number;
  timestamp: string;
  user_command: string;
  resolved_command: string;
  task_type: string;
  intent: string;
  target?: string | null;
  features: string[];
  time_column?: string | null;
  model_selected?: string | null;
  metrics: Record<string, unknown>;
  confidence: number;
  status: string;
  summary: string;
  top_findings: string[];
  resolved_references: Record<string, string>;
};

export type DatasetSnapshotItem = {
  dataset_id: string;
  dataset_name: string;
  columns: string[];
  numeric_columns: string[];
  categorical_columns: string[];
  datetime_columns: string[];
  identifier_columns: string[];
  constant_columns: string[];
  original_rows: number;
  current_rows: number;
  preview_sample: Array<Record<string, unknown>>;
  quality_score: number;
};

export type AnalyticalContextData = {
  session_id: string;
  created_at: string;
  last_active_at: string;
  active_dataset_id?: string | null;
  datasets: Record<string, DatasetSnapshotItem>;
  active_target?: string | null;
  active_features: string[];
  active_time_column?: string | null;
  active_task?: string | null;
  previous_task?: string | null;
  current_intent?: string | null;
  previous_intent?: string | null;
  last_execution_id?: string | null;
  latest_metrics: Record<string, unknown>;
  latest_confidence: number;
  latest_model_name?: string | null;
  latest_forecast_horizon?: number | null;
  latest_cluster_count?: number | null;
  latest_anomaly_count?: number | null;
  execution_history: ExecutionRecordItem[];
  pending_clarification?: Record<string, unknown> | null;
  assumptions: string[];
  limitations: string[];
  warnings: string[];
};
