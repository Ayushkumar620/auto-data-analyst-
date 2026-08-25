import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import type { ReportSummary } from '../../types';
import { IconFileText, IconDatabase } from '../ui/Icons';

type ReportCardProps = {
  report: ReportSummary;
  onDelete?: (reportId: string) => void;
};

export default function ReportCard({ report, onDelete }: ReportCardProps) {
  const [confirmDelete, setConfirmDelete] = useState(false);

  const formattedDate = new Date(report.created_at).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });

  const typeLabels: Record<string, string> = {
    comprehensive: 'Comprehensive',
    forecast: 'Forecast Deliverable',
    model: 'Model Review',
    monitoring: 'Drift & Health Audit',
    analysis: 'Analysis Synthesis',
    eda_comprehensive: 'EDA Deliverable',
    file_export: 'Exported Report',
  };

  const typeLabel = typeLabels[report.report_type] || report.report_type || 'Report';

  return (
    <div
      className="glass-card glass-card--padded"
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        height: '100%',
        transition: 'transform 0.15s ease, box-shadow 0.15s ease',
      }}
    >
      <div>
        {/* Top header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem', marginBottom: '0.65rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <span
              style={{
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                color: 'var(--primary)',
                padding: '0.25rem',
                borderRadius: '6px',
                display: 'inline-flex',
              }}
            >
              <IconFileText size={16} aria-hidden />
            </span>
            <span
              style={{
                fontSize: '0.72rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                color: 'var(--muted)',
                letterSpacing: '0.04em',
              }}
            >
              {typeLabel}
            </span>
          </div>

          <span
            style={{
              fontSize: '0.7rem',
              fontWeight: 600,
              backgroundColor: '#ecfdf5',
              color: '#059669',
              padding: '0.15rem 0.45rem',
              borderRadius: '999px',
            }}
          >
            {report.status}
          </span>
        </div>

        {/* Title */}
        <h3 style={{ margin: '0 0 0.4rem', fontSize: '1rem', fontWeight: 600, color: 'var(--ink)' }}>
          <Link
            to={`/reports/${encodeURIComponent(report.report_id)}`}
            style={{ textDecoration: 'none', color: 'inherit' }}
          >
            {report.title}
          </Link>
        </h3>

        {/* Dataset tag & date */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.78rem', color: 'var(--muted)', marginBottom: '0.75rem' }}>
          <IconDatabase size={13} aria-hidden />
          <span>{report.dataset_name}</span>
          <span>·</span>
          <span>{formattedDate}</span>
        </div>

        {/* Executive summary preview */}
        {report.executive_summary && (
          <p
            className="muted"
            style={{
              fontSize: '0.82rem',
              margin: '0 0 0.85rem',
              lineHeight: '1.45',
              display: '-webkit-box',
              WebkitLineClamp: 3,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
          >
            {report.executive_summary}
          </p>
        )}
      </div>

      {/* Footer / Meta & Actions */}
      <div
        style={{
          borderTop: '1px solid #f1f5f9',
          paddingTop: '0.75rem',
          marginTop: '0.5rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.5rem',
        }}
      >
        <div style={{ display: 'flex', gap: '0.5rem', fontSize: '0.76rem', color: 'var(--muted)' }}>
          {report.kpi_count > 0 && <span>{report.kpi_count} KPIs</span>}
          {report.insight_count > 0 && <span>· {report.insight_count} Insights</span>}
          {report.recommendation_count > 0 && <span>· {report.recommendation_count} Actions</span>}
        </div>

        <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
          {onDelete && (
            confirmDelete ? (
              <div style={{ display: 'flex', gap: '0.25rem' }}>
                <button
                  type="button"
                  onClick={() => onDelete(report.report_id)}
                  className="action-btn"
                  style={{ color: '#dc2626', borderColor: '#fecaca', padding: '0.2rem 0.45rem', fontSize: '0.72rem' }}
                >
                  Confirm
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmDelete(false)}
                  className="ghost-text-btn"
                  style={{ fontSize: '0.72rem', padding: '0.2rem' }}
                >
                  Cancel
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setConfirmDelete(true)}
                className="ghost-text-btn"
                style={{ fontSize: '0.74rem', color: 'var(--muted)' }}
                title="Delete report"
              >
                ✕
              </button>
            )
          )}

          <Link
            to={`/reports/${encodeURIComponent(report.report_id)}`}
            className="primary-btn"
            style={{ padding: '0.28rem 0.75rem', fontSize: '0.78rem', textDecoration: 'none' }}
          >
            Open Report →
          </Link>
        </div>
      </div>
    </div>
  );
}
