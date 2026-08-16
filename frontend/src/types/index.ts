export type DatasetProfile = {
  dataset_name: string;
  rows: number;
  columns: number;
  column_names: string[];
  missing_values: number;
  duplicates: number;
  preview: Array<Record<string, unknown>>;
  data_types?: Record<string, string>;
  memory_usage?: string;
};

export type Workspace = {
  id: string;
  name: string;
  owner: string;
  projects?: Project[];
};

export type Project = {
  id: string;
  workspace_id: string;
  name: string;
  datasets?: Array<Record<string, unknown>>;
  sessions?: Array<Record<string, unknown>>;
};
