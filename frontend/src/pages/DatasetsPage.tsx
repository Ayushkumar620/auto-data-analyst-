import React, { useEffect, useState, useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { PageContainer, PageHeader } from '../components/layout/PageContainer';
import DatasetCard from '../components/datasets/DatasetCard';
import EmptyState from '../components/ui/EmptyState';
import ErrorState from '../components/ui/ErrorState';
import { SkeletonCard } from '../components/ui/LoadingState';
import { IconDatabase, IconUpload } from '../components/ui/Icons';
import { listDatasets, deleteDataset } from '../services/datasetService';
import type { DatasetItem } from '../types';

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<DatasetItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'recent' | 'name' | 'rows'>('recent');
  const navigate = useNavigate();

  const loadData = async () => {
    setLoading(true);
    setError('');
    try {
      const items = await listDatasets();
      setDatasets(items);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch datasets');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleDelete = (id: string) => {
    if (window.confirm('Are you sure you want to remove this dataset from your registry?')) {
      deleteDataset(id);
      setDatasets((prev) => prev.filter((d) => d.id !== id && d.name !== id));
    }
  };

  const filteredDatasets = useMemo(() => {
    let result = datasets.filter((d) => {
      const q = searchQuery.toLowerCase();
      return (
        d.name.toLowerCase().includes(q) ||
        (d.file_type && d.file_type.toLowerCase().includes(q))
      );
    });

    result.sort((a, b) => {
      if (sortBy === 'name') {
        return a.name.localeCompare(b.name);
      }
      if (sortBy === 'rows') {
        return (b.rows || 0) - (a.rows || 0);
      }
      // 'recent'
      const dateA = a.created_at ? new Date(a.created_at).getTime() : 0;
      const dateB = b.created_at ? new Date(b.created_at).getTime() : 0;
      return dateB - dateA;
    });

    return result;
  }, [datasets, searchQuery, sortBy]);

  return (
    <PageContainer>
      <PageHeader
        eyebrow="Data Workspace"
        title="Datasets"
        subtitle="Manage the data powering your analyses."
        actions={
          <div className="page-header-action-row">
            <Link to="/upload" className="primary-btn" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <IconUpload size={16} aria-hidden />
              Upload Dataset
            </Link>
          </div>
        }
      />

      {/* Filter / Search Bar */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: 1, minWidth: '240px', maxWidth: '420px' }}>
          <input
            type="text"
            placeholder="Search datasets by name or type..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="horizon-input"
            style={{ width: '100%', padding: '0.5rem 0.85rem' }}
            aria-label="Search datasets"
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <label htmlFor="dataset-sort" className="muted" style={{ fontSize: '0.84rem' }}>
            Sort by:
          </label>
          <select
            id="dataset-sort"
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="horizon-input"
            style={{ padding: '0.45rem 0.75rem', fontSize: '0.84rem' }}
          >
            <option value="recent">Most Recent</option>
            <option value="name">Name (A–Z)</option>
            <option value="rows">Row Count</option>
          </select>
        </div>
      </div>

      {/* Content */}
      {loading ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1rem' }}>
          <SkeletonCard lines={4} />
          <SkeletonCard lines={4} />
          <SkeletonCard lines={4} />
        </div>
      ) : error ? (
        <ErrorState message={error} onRetry={loadData} />
      ) : filteredDatasets.length === 0 ? (
        searchQuery ? (
          <EmptyState
            icon={<IconDatabase size={40} />}
            title="No matching datasets"
            description={`No datasets match "${searchQuery}".`}
            action={
              <button type="button" className="action-btn" onClick={() => setSearchQuery('')}>
                Clear Search
              </button>
            }
          />
        ) : (
          <EmptyState
            icon={<IconDatabase size={48} />}
            title="No datasets yet"
            description="Upload your first dataset to start generating automated insights, forecasts, and reports."
            action={
              <Link to="/upload" className="primary-btn">
                Upload your first dataset
              </Link>
            }
          />
        )
      ) : (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
            gap: '1.25rem',
          }}
        >
          {filteredDatasets.map((dataset) => (
            <DatasetCard
              key={dataset.id}
              dataset={dataset}
              onDelete={() => handleDelete(dataset.id)}
            />
          ))}
        </div>
      )}
    </PageContainer>
  );
}

