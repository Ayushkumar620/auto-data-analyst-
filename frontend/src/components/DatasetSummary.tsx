import React from 'react';

type DatasetSummaryProps = {
  profile: {
    dataset_name: string;
    rows: number;
    columns: number;
    missing_values: number;
    duplicates: number;
  };
};

export default function DatasetSummary({ profile }: DatasetSummaryProps) {
  return (
    <div>
      <h3>{profile.dataset_name}</h3>
      <p>Rows: {profile.rows}</p>
      <p>Columns: {profile.columns}</p>
      <p>Missing Values: {profile.missing_values}</p>
      <p>Duplicate Rows: {profile.duplicates}</p>
    </div>
  );
}
