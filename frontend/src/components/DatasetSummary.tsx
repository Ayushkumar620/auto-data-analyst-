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
    <section className="summary-card">
      <h3>{profile.dataset_name}</h3>
      <div className="metric-grid">
        <article className="metric-tile">
          <p className="metric-label">Rows</p>
          <p className="metric-value">{profile.rows}</p>
        </article>
        <article className="metric-tile">
          <p className="metric-label">Columns</p>
          <p className="metric-value">{profile.columns}</p>
        </article>
        <article className="metric-tile">
          <p className="metric-label">Missing values</p>
          <p className="metric-value">{profile.missing_values}</p>
        </article>
        <article className="metric-tile">
          <p className="metric-label">Duplicate rows</p>
          <p className="metric-value">{profile.duplicates}</p>
        </article>
      </div>
    </section>
  );
}
