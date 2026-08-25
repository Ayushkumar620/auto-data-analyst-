import React from 'react';

type DatasetSummaryProps = {
  profile: {
    dataset_name: string;
    rows: number;
    columns: number;
    missing_values: number;
    duplicates: number;
    memory_usage?: string;
    quality_score?: number;
    column_names?: string[];
    data_types?: Record<string, string>;
  };
};

export default function DatasetSummary({ profile }: DatasetSummaryProps) {
  const numericCount = profile.data_types
    ? Object.values(profile.data_types).filter((t) =>
        t.toLowerCase().includes('int') || t.toLowerCase().includes('float') || t.toLowerCase().includes('double'),
      ).length
    : null;

  const categoricalCount = profile.data_types && numericCount !== null
    ? Object.keys(profile.data_types).length - numericCount
    : null;

  return (
    <section className="summary-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>{profile.dataset_name}</h3>
          {profile.memory_usage && (
            <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.8rem' }}>
              Memory usage: {profile.memory_usage}
            </p>
          )}
        </div>
        {profile.quality_score !== undefined && (
          <div
            style={{
              padding: '0.35rem 0.75rem',
              borderRadius: '9999px',
              fontSize: '0.8rem',
              fontWeight: 700,
              backgroundColor: profile.quality_score >= 80 ? 'var(--success-bg)' : 'var(--warning-bg)',
              color: profile.quality_score >= 80 ? 'var(--success)' : 'var(--warning)',
              border: `1px solid ${profile.quality_score >= 80 ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`,
            }}
          >
            Quality: {profile.quality_score}%
          </div>
        )}
      </div>

      <div className="metric-grid">
        <article className="metric-tile">
          <p className="metric-label">Rows</p>
          <p className="metric-value">{profile.rows.toLocaleString()}</p>
        </article>
        <article className="metric-tile">
          <p className="metric-label">Columns</p>
          <p className="metric-value">{profile.columns}</p>
        </article>
        <article className="metric-tile">
          <p className="metric-label">Missing values</p>
          <p className="metric-value">{profile.missing_values.toLocaleString()}</p>
        </article>
        <article className="metric-tile">
          <p className="metric-label">Duplicate rows</p>
          <p className="metric-value">{profile.duplicates.toLocaleString()}</p>
        </article>
      </div>

      {numericCount !== null && categoricalCount !== null && (
        <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.75rem', flexWrap: 'wrap' }}>
          <span className="sidebar-soon-badge" style={{ backgroundColor: '#eff6ff', color: '#1d4ed8', borderColor: '#bfdbfe' }}>
            {numericCount} Numeric Columns
          </span>
          <span className="sidebar-soon-badge" style={{ backgroundColor: '#f5f3ff', color: '#6d28d9', borderColor: '#ddd6fe' }}>
            {categoricalCount} Categorical Columns
          </span>
        </div>
      )}
    </section>
  );
}
