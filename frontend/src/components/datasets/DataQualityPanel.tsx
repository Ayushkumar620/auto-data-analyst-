import React from 'react';
import type { DatasetProfile } from '../../types';
import { IconCheck, IconAlertTriangle, IconInfo } from '../ui/Icons';

type DataQualityPanelProps = {
  profile: DatasetProfile;
};

export default function DataQualityPanel({ profile }: DataQualityPanelProps) {
  const missingRate = profile.rows > 0 && profile.columns > 0
    ? (profile.missing_values / (profile.rows * profile.columns)) * 100
    : 0;

  const duplicateRate = profile.rows > 0
    ? (profile.duplicates / profile.rows) * 100
    : 0;

  // Determine overall status
  const score = profile.quality_score !== undefined
    ? profile.quality_score
    : Math.max(0, 100 - (missingRate * 2 + duplicateRate * 3));

  const statusType: 'good' | 'warning' | 'critical' =
    score >= 80 ? 'good' : score >= 50 ? 'warning' : 'critical';

  return (
    <div style={{ display: 'grid', gap: '1.25rem' }}>
      {/* Quality banner */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          padding: '1.25rem',
          borderRadius: '14px',
          border: '1px solid',
          borderColor:
            statusType === 'good'
              ? 'rgba(16, 185, 129, 0.3)'
              : statusType === 'warning'
              ? 'rgba(245, 158, 11, 0.3)'
              : 'rgba(239, 68, 68, 0.3)',
          backgroundColor:
            statusType === 'good'
              ? 'rgba(16, 185, 129, 0.05)'
              : statusType === 'warning'
              ? 'rgba(245, 158, 11, 0.05)'
              : 'rgba(239, 68, 68, 0.05)',
        }}
      >
        <div
          style={{
            width: '48px',
            height: '48px',
            borderRadius: '12px',
            backgroundColor:
              statusType === 'good'
                ? 'var(--success)'
                : statusType === 'warning'
                ? 'var(--warning)'
                : 'var(--alert)',
            color: '#ffffff',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.25rem',
            fontWeight: 700,
            flexShrink: 0,
          }}
        >
          {statusType === 'good' ? <IconCheck size={24} /> : <IconAlertTriangle size={24} />}
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600, color: 'var(--ink)' }}>
              Data Quality: {statusType === 'good' ? 'Healthy' : statusType === 'warning' ? 'Moderate Quality' : 'Action Needed'}
            </h3>
            <span
              style={{
                fontSize: '0.78rem',
                fontWeight: 700,
                padding: '0.2rem 0.6rem',
                borderRadius: '9999px',
                backgroundColor: statusType === 'good' ? 'var(--success-bg)' : statusType === 'warning' ? 'var(--warning-bg)' : 'var(--alert-bg)',
                color: statusType === 'good' ? 'var(--success)' : statusType === 'warning' ? 'var(--warning)' : 'var(--alert)',
              }}
            >
              {Math.round(score)} / 100
            </span>
          </div>
          <p className="muted" style={{ margin: '0.25rem 0 0', fontSize: '0.86rem' }}>
            {statusType === 'good'
              ? 'Dataset structure is clean with minimal missing values and no critical integrity issues.'
              : statusType === 'warning'
              ? 'Dataset has some missing fields or duplicate rows that may require cleaning before modeling.'
              : 'High percentage of missing or duplicate values detected. Consider cleaning the dataset first.'}
          </p>
        </div>
      </div>

      {/* Metric Breakdown */}
      <div className="metric-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <article className="metric-tile">
          <p className="metric-label">Missing Values</p>
          <p className="metric-value">{profile.missing_values.toLocaleString()}</p>
          <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.78rem' }}>
            {missingRate.toFixed(2)}% of total cells
          </p>
        </article>

        <article className="metric-tile">
          <p className="metric-label">Duplicate Rows</p>
          <p className="metric-value">{profile.duplicates.toLocaleString()}</p>
          <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.78rem' }}>
            {duplicateRate.toFixed(2)}% of total rows
          </p>
        </article>

        <article className="metric-tile">
          <p className="metric-label">Memory Footprint</p>
          <p className="metric-value" style={{ fontSize: '1.2rem' }}>
            {profile.memory_usage || '—'}
          </p>
          <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.78rem' }}>
            {profile.rows.toLocaleString()} × {profile.columns} matrix
          </p>
        </article>
      </div>

      {/* Quality Recommendations */}
      {profile.recommendations && profile.recommendations.length > 0 && (
        <div className="glass-card glass-card--padded" style={{ border: '1px solid rgba(226, 232, 240, 0.9)' }}>
          <h4 style={{ margin: '0 0 0.75rem', fontSize: '0.95rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <IconInfo size={16} color="var(--primary)" />
            Automated Quality Recommendations
          </h4>
          <ul style={{ margin: 0, paddingLeft: '1.25rem', display: 'grid', gap: '0.4rem' }}>
            {profile.recommendations.map((rec, idx) => (
              <li key={idx} style={{ fontSize: '0.88rem', color: 'var(--ink-secondary)' }}>
                {typeof rec === 'string' ? rec : (rec as any).message || JSON.stringify(rec)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

