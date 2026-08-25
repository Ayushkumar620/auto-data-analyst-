import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { PageContainer } from '../components/layout/PageContainer';
import DriftPanel from '../components/monitoring/DriftPanel';
import PerformanceMonitoringPanel from '../components/monitoring/PerformanceMonitoringPanel';
import MonitoringStatusBadge from '../components/monitoring/MonitoringStatusBadge';
import ErrorState from '../components/ui/ErrorState';
import { LoadingSpinner } from '../components/ui/LoadingState';
import { getMonitoringHistory } from '../services/monitoringService';
import type { MonitoringResult } from '../types';
import { IconChevronRight, IconActivity } from '../components/ui/Icons';

export default function MonitoringDetailPage() {
  const { modelId } = useParams<{ modelId: string }>();
  const [history, setHistory] = useState<MonitoringResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!modelId) return;
    setLoading(true);
    setError('');

    getMonitoringHistory(modelId)
      .then((data) => setHistory(data))
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load model monitoring details'))
      .finally(() => setLoading(false));
  }, [modelId]);

  if (loading) {
    return (
      <PageContainer>
        <LoadingSpinner label="Loading model monitoring profile…" size={36} />
      </PageContainer>
    );
  }

  const latestRun = history.length > 0 ? history[0] : null;

  return (
    <PageContainer>
      {/* Breadcrumb */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem', fontSize: '0.84rem' }}>
          <Link to="/monitoring" className="muted" style={{ textDecoration: 'none' }}>
            Monitoring
          </Link>
          <IconChevronRight size={14} className="muted" aria-hidden />
          <span style={{ fontWeight: 600, color: 'var(--ink)' }}>{modelId}</span>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.5rem' }}>
          <div>
            <h1 className="page-title" style={{ margin: 0 }}>Model Monitoring Profile: {modelId}</h1>
            <p className="page-subtitle" style={{ margin: '0.2rem 0 0' }}>
              Historical drift tests and performance degradation audits for registered artifact.
            </p>
          </div>

          {latestRun && (
            <MonitoringStatusBadge status={latestRun.overall_severity} />
          )}
        </div>
      </div>

      {error && <ErrorState message={error} />}

      {latestRun ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {latestRun.data_drift && (
            <DriftPanel dataDrift={latestRun.data_drift} />
          )}

          {latestRun.performance_drift && (
            <PerformanceMonitoringPanel performanceDrift={latestRun.performance_drift} />
          )}
        </div>
      ) : (
        <div className="glass-card glass-card--padded">
          <p className="muted" style={{ margin: 0 }}>
            No monitoring evaluations have been performed for model '{modelId}' yet. Visit the main Monitoring dashboard to trigger an evaluation.
          </p>
        </div>
      )}
    </PageContainer>
  );
}

