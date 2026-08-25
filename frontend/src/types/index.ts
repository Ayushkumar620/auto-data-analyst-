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

