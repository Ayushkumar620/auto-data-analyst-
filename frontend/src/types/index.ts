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
