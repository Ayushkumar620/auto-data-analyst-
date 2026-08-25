import React, { useEffect, useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { PageContainer, PageHeader, Card } from '../components/layout/PageContainer';
import ModelCard from '../components/models/ModelCard';
import EmptyState from '../components/ui/EmptyState';
import ErrorState from '../components/ui/ErrorState';
import { SkeletonCard } from '../components/ui/LoadingState';
import { IconBrain, IconTrendUp, IconCheck } from '../components/ui/Icons';
import { listModels } from '../services/modelService';
import type { ModelMetadata } from '../types';

export default function ModelRegistryPage() {
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [familyFilter, setFamilyFilter] = useState<string>('ALL');
  const [statusFilter, setStatusFilter] = useState<string>('ALL');

  const loadModels = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await listModels();
      setModels(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to retrieve model registry');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const filteredModels = useMemo(() => {
    return models.filter((m) => {
      const q = searchQuery.toLowerCase();
      const matchesSearch =
        m.name.toLowerCase().includes(q) ||
        m.algorithm.toLowerCase().includes(q) ||
        m.target_column.toLowerCase().includes(q);

      if (!matchesSearch) return false;
      if (familyFilter !== 'ALL' && m.model_family !== familyFilter) return false;
      if (statusFilter !== 'ALL' && m.status.toLowerCase() !== statusFilter.toLowerCase()) return false;
      return true;
    });
  }, [models, searchQuery, familyFilter, statusFilter]);

  const activeCount = models.filter((m) => m.status.toLowerCase() === 'active').length;
  const stagingCount = models.filter((m) => m.status.toLowerCase() === 'staging').length;

  return (
    <PageContainer>
      <PageHeader
        eyebrow="Intelligence Center"
        title="Model Registry & Leaderboard"
        subtitle="Manage, benchmark, and deploy machine learning models trained by the autonomous orchestrator."
      />

      {/* KPI summary strip */}
      <div className="kpi-strip">
        <div className="kpi-tile">
          <span className="kpi-icon">
            <IconBrain size={20} />
          </span>
          <div>
            <p className="kpi-value">{loading ? '—' : models.length}</p>
            <p className="kpi-label">Registered Models</p>
          </div>
        </div>

        <div className="kpi-tile">
          <span className="kpi-icon" style={{ backgroundColor: '#ecfdf5', color: '#059669' }}>
            <IconCheck size={20} />
          </span>
          <div>
            <p className="kpi-value">{loading ? '—' : activeCount}</p>
            <p className="kpi-label">Active Deployed</p>
          </div>
        </div>

        <div className="kpi-tile">
          <span className="kpi-icon" style={{ backgroundColor: '#eff6ff', color: '#1d4ed8' }}>
            <IconTrendUp size={20} />
          </span>
          <div>
            <p className="kpi-value">{loading ? '—' : stagingCount}</p>
            <p className="kpi-label">Staging Models</p>
          </div>
        </div>

        <div className="kpi-tile">
          <span className="kpi-icon">⚡</span>
          <div>
            <p className="kpi-value">{models.length > 0 ? `${(Math.max(...models.map(m => m.primary_metric_value || 0)) * 100).toFixed(1)}%` : '—'}</p>
            <p className="kpi-label">Top Score</p>
          </div>
        </div>
      </div>

      {/* Search & Filters */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
        }}
      >
        <input
          type="text"
          placeholder="Search models by name, algorithm, or target..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="horizon-input"
          style={{ width: '320px', padding: '0.45rem 0.85rem', fontSize: '0.86rem' }}
          aria-label="Search models"
        />

        {/* Filter buttons */}
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <select
            value={familyFilter}
            onChange={(e) => setFamilyFilter(e.target.value)}
            className="horizon-input"
            style={{ padding: '0.4rem 0.75rem', fontSize: '0.84rem' }}
            aria-label="Filter by model family"
          >
            <option value="ALL">All Families</option>
            <option value="traditional_ml">Traditional ML</option>
            <option value="ann">Neural Network (ANN)</option>
            <option value="cnn">Convolutional (CNN)</option>
            <option value="forecasting">Forecasting</option>
          </select>

          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="horizon-input"
            style={{ padding: '0.4rem 0.75rem', fontSize: '0.84rem' }}
            aria-label="Filter by lifecycle status"
          >
            <option value="ALL">All Statuses</option>
            <option value="active">Active</option>
            <option value="staging">Staging</option>
            <option value="archived">Archived</option>
          </select>
        </div>
      </div>

      {/* Content */}
      {loading ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1rem' }}>
          <SkeletonCard lines={4} />
          <SkeletonCard lines={4} />
          <SkeletonCard lines={4} />
        </div>
      ) : error ? (
        <ErrorState message={error} onRetry={loadModels} />
      ) : filteredModels.length === 0 ? (
        searchQuery || familyFilter !== 'ALL' || statusFilter !== 'ALL' ? (
          <EmptyState
            icon={<IconBrain size={40} />}
            title="No matching models found"
            description="Try changing your search keywords or filter criteria."
            action={
              <button
                type="button"
                className="action-btn"
                onClick={() => {
                  setSearchQuery('');
                  setFamilyFilter('ALL');
                  setStatusFilter('ALL');
                }}
              >
                Reset Filters
              </button>
            }
          />
        ) : (
          <EmptyState
            icon={<IconBrain size={48} />}
            title="No models registered yet"
            description="When you run analytical commands requiring modeling in the AI Analyst or Command Studio, models will be automatically evaluated, registered, and tracked here."
            action={
              <Link to="/analyst" className="primary-btn">
                Run Model Analysis
              </Link>
            }
          />
        )
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.25rem' }}>
          {filteredModels.map((model) => (
            <ModelCard key={model.model_id} model={model} />
          ))}
        </div>
      )}
    </PageContainer>
  );
}

