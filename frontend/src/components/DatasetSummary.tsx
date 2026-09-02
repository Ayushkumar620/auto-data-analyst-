import React from 'react';
import SummaryData from './SummaryData';
import type { DatasetProfile } from '../types';

type DatasetSummaryProps = {
  profile: DatasetProfile | {
    dataset_name: string;
    rows: number;
    columns: number;
    missing_values: number;
    duplicates?: number;
    memory_usage?: string;
    quality_score?: number;
    column_names?: string[];
    data_types?: Record<string, string>;
    preview?: Array<Record<string, unknown>>;
  };
  className?: string;
};

export default function DatasetSummary({ profile, className }: DatasetSummaryProps) {
  return <SummaryData profile={profile as any} className={className} />;
}
