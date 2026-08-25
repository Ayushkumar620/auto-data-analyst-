import React, { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { PageContainer } from '../components/layout/PageContainer';
import DatasetSummary from '../components/DatasetSummary';
import PreviewTable from '../components/PreviewTable';
import SchemaExplorer from '../components/datasets/SchemaExplorer';
import DataQualityPanel from '../components/datasets/DataQualityPanel';
import DatasetActions from '../components/datasets/DatasetActions';
import ErrorState from '../components/ui/ErrorState';
import { LoadingSpinner } from '../components/ui/LoadingState';
import { getDatasetById } from '../services/datasetService';
import { useDataset } from '../context/DatasetContext';
import { useNotification } from '../context/NotificationContext';
import type { DatasetItem, DatasetProfile } from '../types';
import { IconDatabase, IconChevronRight } from '../components/ui/Icons';

type TabKey = 'overview' | 'schema' | 'preview' | 'quality';

export default function DatasetWorkspacePage() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const [dataset, setDatasetItem] = useState<DatasetItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState<TabKey>('overview');
  const [query, setQuery] = useState('');

  const { setDataset: setActiveDataset } = useDataset();
  const { notify } = useNotification();
  const navigate = useNavigate();

  useEffect(() => {
    if (!datasetId) return;

    setLoading(true);
    setError('');

    getDatasetById(decodeURIComponent(datasetId))
      .then((item) => {
        if (!item) {
          setError(`Dataset "${datasetId}" was not found in your workspace.`);
        } else {
          setDatasetItem(item);
        }
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : 'Failed to load dataset');
      })
      .finally(() => setLoading(false));
  }, [datasetId]);

  if (loading) {
    return (
      <PageContainer>
        <LoadingSpinner label="Loading dataset workspace…" size={36} />
      </PageContainer>
    );
  }

  if (error || !dataset) {
    return (
      <PageContainer>
        <ErrorState
          message={error || 'Dataset not found.'}
          onRetry={() => navigate('/datasets')}
        />
      </PageContainer>
    );
  }

  const profile: DatasetProfile = dataset.profile || {
    dataset_name: dataset.name,
    rows: dataset.rows,
    columns: dataset.columns,
    column_names: [],
    missing_values: 0,
    duplicates: 0,
    preview: [],
    file_type: dataset.file_type,
    status: dataset.status,
  };

  const handleAskAnalyst = (e: React.FormEvent) => {
    e.preventDefault();
    setActiveDataset(profile, dataset.name);
    navigate('/analyst');
  };

  return (
    <PageContainer>
      {/* Breadcrumb Header */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem', fontSize: '0.84rem' }}>
          <Link to="/datasets" className="muted" style={{ textDecoration: 'none' }}>
            Datasets
          </Link>
          <IconChevronRight size={14} className="muted" aria-hidden />
          <span style={{ fontWeight: 600, color: 'var(--ink)' }}>{dataset.name}</span>
        </div>

        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            gap: '1.25rem',
            flexWrap: 'wrap',
          }}
        >
          <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <h1 className="page-title" style={{ margin: 0 }}>{dataset.name}</h1>
              <span
                className="sidebar-soon-badge"
                style={{ backgroundColor: '#eff6ff', color: '#1d4ed8', borderColor: '#bfdbfe', textTransform: 'uppercase' }}
              >
                {dataset.file_type || 'CSV'}
              </span>
            </div>
            <p className="page-subtitle" style={{ marginTop: '0.3rem' }}>
              {dataset.rows.toLocaleString()} rows · {dataset.columns} columns
            </p>
          </div>

          <DatasetActions profile={profile} fileName={dataset.name} />
        </div>
      </div>

      {/* Tabs */}
      <div
        style={{
          display: 'flex',
          gap: '0.5rem',
          borderBottom: '1px solid rgba(226, 232, 240, 0.9)',
          paddingBottom: '0.2rem',
        }}
      >
        {(
          [
            { key: 'overview', label: 'Overview' },
            { key: 'schema', label: `Schema (${dataset.columns})` },
            { key: 'preview', label: 'Data Preview' },
            { key: 'quality', label: 'Data Quality' },
          ] as const
        ).map((tab) => (
          <button
            key={tab.key}
            type="button"
            className={activeTab === tab.key ? 'primary-btn' : 'action-btn'}
            style={{
              padding: '0.45rem 0.9rem',
              fontSize: '0.86rem',
              borderRadius: '8px',
            }}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div style={{ display: 'grid', gap: '1.25rem' }}>
          <DatasetSummary profile={profile} />

          {/* Prompt AI Analyst Box */}
          <div className="glass-card glass-card--padded">
            <h3 style={{ margin: '0 0 0.5rem', fontSize: '1rem', fontWeight: 600 }}>
              Ask your data anything
            </h3>
            <p className="muted" style={{ margin: '0 0 1rem', fontSize: '0.88rem' }}>
              Launch an autonomous multi-agent analysis with this dataset pre-loaded.
            </p>
            <form onSubmit={handleAskAnalyst}>
              <div className="analyst-input-wrap">
                <textarea
                  className="analyst-textarea"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder={`e.g. Find key trends and anomalies in ${dataset.name}...`}
                  rows={2}
                  aria-label="Analysis query"
                />
                <button type="submit" className="primary-btn analyst-submit-btn">
                  Analyze →
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {activeTab === 'schema' && (
        <div className="glass-card glass-card--padded">
          <h3 style={{ margin: '0 0 1rem', fontSize: '1.05rem', fontWeight: 600 }}>Dataset Schema</h3>
          <SchemaExplorer profile={profile} />
        </div>
      )}

      {activeTab === 'preview' && (
        <div className="glass-card glass-card--padded">
          <h3 style={{ margin: '0 0 1rem', fontSize: '1.05rem', fontWeight: 600 }}>Data Preview</h3>
          <PreviewTable preview={profile.preview} />
        </div>
      )}

      {activeTab === 'quality' && (
        <div className="glass-card glass-card--padded">
          <h3 style={{ margin: '0 0 1rem', fontSize: '1.05rem', fontWeight: 600 }}>Data Quality & Profiling</h3>
          <DataQualityPanel profile={profile} />
        </div>
      )}
    </PageContainer>
  );
}
