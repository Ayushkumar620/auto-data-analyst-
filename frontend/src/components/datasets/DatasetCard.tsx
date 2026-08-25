import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import type { DatasetItem } from '../../types';
import { useDataset } from '../../context/DatasetContext';
import { useNotification } from '../../context/NotificationContext';
import { IconDatabase, IconTrendUp, IconAnalyst, IconCheck } from '../ui/Icons';

type DatasetCardProps = {
  dataset: DatasetItem;
  onDelete?: (id: string) => void;
};

export default function DatasetCard({ dataset, onDelete }: DatasetCardProps) {
  const { profile, fileName, setDataset } = useDataset();
  const { notify } = useNotification();
  const navigate = useNavigate();

  const isActive = (profile && profile.dataset_name === dataset.name) || fileName === dataset.name;

  const handleSetActive = (e: React.MouseEvent) => {
    e.stopPropagation();
    const datasetProfile = dataset.profile || {
      dataset_name: dataset.name,
      rows: dataset.rows,
      columns: dataset.columns,
      column_names: [],
      missing_values: 0,
      duplicates: 0,
      preview: [],
      file_type: dataset.file_type,
    };
    setDataset(datasetProfile, dataset.name);
    notify(`"${dataset.name}" is now the active dataset.`, 'success');
  };

  const handleAnalyze = (e: React.MouseEvent) => {
    e.stopPropagation();
    const datasetProfile = dataset.profile || {
      dataset_name: dataset.name,
      rows: dataset.rows,
      columns: dataset.columns,
      column_names: [],
      missing_values: 0,
      duplicates: 0,
      preview: [],
      file_type: dataset.file_type,
    };
    setDataset(datasetProfile, dataset.name);
    navigate('/analyst');
  };

  return (
    <div
      className={`glass-card glass-card--padded dataset-card${isActive ? ' dataset-card--active' : ''}`}
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        position: 'relative',
        cursor: 'pointer',
        transition: 'all 200ms ease',
      }}
      onClick={() => navigate(`/datasets/${encodeURIComponent(dataset.id)}`)}
    >
      {/* Top section */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem', marginBottom: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', minWidth: 0 }}>
            <div
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '10px',
                backgroundColor: isActive ? 'var(--primary)' : 'var(--primary-light)',
                color: isActive ? '#ffffff' : 'var(--primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}
            >
              <IconDatabase size={18} aria-hidden />
            </div>
            <div style={{ minWidth: 0 }}>
              <h3
                style={{
                  margin: 0,
                  fontSize: '0.96rem',
                  fontWeight: 600,
                  color: 'var(--ink)',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
                title={dataset.name}
              >
                {dataset.name}
              </h3>
              <p className="muted" style={{ margin: 0, fontSize: '0.75rem', textTransform: 'uppercase' }}>
                {dataset.file_type || 'CSV'}
              </p>
            </div>
          </div>

          {isActive && (
            <span
              className="sidebar-soon-badge"
              style={{
                backgroundColor: '#ecfdf5',
                color: '#059669',
                borderColor: '#a7f3d0',
                display: 'flex',
                alignItems: 'center',
                gap: '0.2rem',
              }}
            >
              <IconCheck size={12} aria-hidden /> Active
            </span>
          )}
        </div>

        {/* Metrics */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '0.5rem',
            padding: '0.65rem 0.75rem',
            backgroundColor: 'rgba(248, 250, 252, 0.8)',
            borderRadius: '10px',
            marginBottom: '1rem',
          }}
        >
          <div>
            <p className="metric-label" style={{ fontSize: '0.7rem' }}>Rows</p>
            <p style={{ margin: '0.1rem 0 0', fontWeight: 700, fontSize: '1.1rem', fontFamily: 'var(--font-heading)' }}>
              {dataset.rows ? dataset.rows.toLocaleString() : '—'}
            </p>
          </div>
          <div>
            <p className="metric-label" style={{ fontSize: '0.7rem' }}>Columns</p>
            <p style={{ margin: '0.1rem 0 0', fontWeight: 700, fontSize: '1.1rem', fontFamily: 'var(--font-heading)' }}>
              {dataset.columns || '—'}
            </p>
          </div>
        </div>
      </div>

      {/* Action Footer */}
      <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', borderTop: '1px solid rgba(226, 232, 240, 0.7)', paddingTop: '0.75rem' }}>
        <button
          type="button"
          className="primary-btn"
          style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
          onClick={handleAnalyze}
        >
          <IconAnalyst size={13} aria-hidden /> Analyze
        </button>

        {!isActive && (
          <button
            type="button"
            className="action-btn"
            style={{ padding: '0.35rem 0.65rem', fontSize: '0.78rem' }}
            onClick={handleSetActive}
            title="Set as active working dataset"
          >
            Set Active
          </button>
        )}

        <Link
          to={`/datasets/${encodeURIComponent(dataset.id)}`}
          className="action-btn"
          style={{ padding: '0.35rem 0.65rem', fontSize: '0.78rem', textDecoration: 'none', marginLeft: 'auto' }}
          onClick={(e) => e.stopPropagation()}
        >
          Open →
        </Link>
      </div>
    </div>
  );
}
