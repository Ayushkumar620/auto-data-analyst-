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
