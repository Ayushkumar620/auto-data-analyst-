import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { PageContainer, PageHeader, Card } from '../components/layout/PageContainer';
import MonitoringOverview from '../components/monitoring/MonitoringOverview';
import DriftPanel from '../components/monitoring/DriftPanel';
import PerformanceMonitoringPanel from '../components/monitoring/PerformanceMonitoringPanel';
import MonitoringHistory from '../components/monitoring/MonitoringHistory';
import MonitoringStatusBadge from '../components/monitoring/MonitoringStatusBadge';
import EmptyState from '../components/ui/EmptyState';
import ErrorState from '../components/ui/ErrorState';
import { LoadingSpinner } from '../components/ui/LoadingState';
import { useDataset } from '../context/DatasetContext';
import { useNotification } from '../context/NotificationContext';
import { listModels } from '../services/modelService';
import {
  runMonitoring,
  getMonitoringHistory,
  getMonitoringOverview,
} from '../services/monitoringService';
import type {
  ModelMetadata,
  MonitoringOverviewData,
  MonitoringResult,
} from '../types';
import { IconActivity, IconBrain, IconDatabase, IconCheck, IconAlertTriangle } from '../components/ui/Icons';

export default function MonitoringPage() {
  const { profile, fileName } = useDataset();
  const { notify } = useNotification();

  const datasetName = fileName || profile?.dataset_name;

  // Overview data & models
  const [overviewData, setOverviewData] = useState<MonitoringOverviewData | null>(null);
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [history, setHistory] = useState<MonitoringResult[]>([]);

  // Execution state
  const [currentResult, setCurrentResult] = useState<MonitoringResult | null>(null);
  const [running, setRunning] = useState(false);
  const [loadingInitial, setLoadingInitial] = useState(true);
  const [error, setError] = useState('');

  const loadInitialData = async () => {
    setLoadingInitial(true);
    setError('');
    try {
      const [ov, mdls, hist] = await Promise.all([
        getMonitoringOverview().catch(() => null),
        listModels().catch(() => []),
        getMonitoringHistory().catch(() => []),
      ]);

      setOverviewData(ov);
      setModels(mdls);
      setHistory(hist);

      if (mdls.length > 0) {
        setSelectedModelId(mdls[0].model_id);
      }

      if (hist.length > 0) {
        setCurrentResult(hist[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to initialize monitoring dashboard');
    } finally {
      setLoadingInitial(false);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  const handleRunMonitoring = async () => {
    if (!selectedModelId) {
      setError('Please select a registered model to monitor.');
      return;
    }

    if (!profile || !profile.preview || profile.preview.length === 0) {
      setError('Please select or upload an evaluation dataset first.');
      return;
    }

    setRunning(true);
    setError('');

    try {
      const result = await runMonitoring({
        model_id: selectedModelId,
        current_dataset: profile.preview,
      });

      setCurrentResult(result);
      notify(`Monitoring completed for model "${selectedModelId}".`, 'success');

      // Refresh overview & history
      const [ov, hist] = await Promise.all([
        getMonitoringOverview().catch(() => null),
        getMonitoringHistory().catch(() => []),
      ]);
      setOverviewData(ov);
      setHistory(hist);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Monitoring run failed');
    } finally {
      setRunning(false);
    }
  };

  if (loadingInitial) {
    return (
      <PageContainer>
        <LoadingSpinner label="Loading model monitoring center…" size={36} />
      </PageContainer>
    );
  }

  const selectedModel = models.find((m) => m.model_id === selectedModelId);

  return (
    <PageContainer>
      <PageHeader
        eyebrow="MLOps & Observability"
        title="Model Monitoring & Data Drift"
        subtitle="Automated statistical data drift detection, schema verification, and performance degradation tracking."
      />

      {/* KPI Overview Strip */}
      <MonitoringOverview data={overviewData} loading={false} />

      {/* Control & Context Bar */}
      <div className="glass-card glass-card--padded" style={{ margin: '1rem 0 1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          {/* Model Selection */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
            <div className="field" style={{ margin: 0 }}>
              <label htmlFor="model-select" style={{ fontSize: '0.8rem', fontWeight: 600 }}>
                Monitored Model
              </label>
              <select
                id="model-select"
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                className="horizon-input"
                style={{ padding: '0.45rem 0.75rem', fontSize: '0.86rem', minWidth: '220px' }}
                disabled={running}
              >
                {models.length === 0 ? (
                  <option value="">No models registered</option>
                ) : (
                  models.map((m) => (
                    <option key={m.model_id} value={m.model_id}>
                      {m.name} ({m.algorithm})
                    </option>
                  ))
                )}
              </select>
            </div>

            {/* Dataset Context Pill */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: '#f8fafc', padding: '0.4rem 0.75rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
              <IconDatabase size={16} color="var(--primary)" aria-hidden />
              <div>
                <span style={{ fontSize: '0.74rem', color: 'var(--muted)', display: 'block' }}>Evaluation Batch:</span>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--ink)' }}>
                  {profile ? datasetName : 'No dataset loaded'}
                </span>
              </div>
            </div>
          </div>

          {/* Action */}
          <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
            <button
              type="button"
              onClick={handleRunMonitoring}
              className="primary-btn"
              disabled={running || !selectedModelId || !profile}
              style={{ padding: '0.45rem 1rem', fontSize: '0.86rem' }}
            >
              {running ? 'Running Statistical Audit…' : '⚡ Run Monitoring'}
            </button>
          </div>
        </div>
      </div>

      {error && <ErrorState message={error} />}

      {/* Selected Model Status Card if active */}
      {selectedModel && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            backgroundColor: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: '12px',
            padding: '0.75rem 1rem',
            marginBottom: '1.25rem',
            boxShadow: '0 1px 4px rgba(15, 23, 42, 0.04)',
          }}
        >
          <div>
            <span style={{ fontSize: '0.76rem', color: 'var(--muted)', fontWeight: 600 }}>Active Monitored Baseline:</span>
            <p style={{ margin: '0.1rem 0 0', fontSize: '0.94rem', fontWeight: 600, color: 'var(--ink)' }}>
              {selectedModel.name} · Target: <strong>{selectedModel.target_column}</strong> ({selectedModel.feature_columns.length} features)
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
            <Link
              to={`/models/${encodeURIComponent(selectedModel.model_id)}`}
              className="action-btn"
              style={{ padding: '0.3rem 0.65rem', fontSize: '0.78rem', textDecoration: 'none' }}
            >
              Model Details →
            </Link>
          </div>
        </div>
      )}

      {/* Current Monitoring Assessment View */}
      {currentResult ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginBottom: '2rem' }}>
          {/* Header Banner */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              backgroundColor: 'rgba(255, 255, 255, 0.9)',
              padding: '0.85rem 1.25rem',
              borderRadius: '12px',
              border: '1px solid #e2e8f0',
            }}
          >
            <div>
              <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600 }}>
                Monitoring Audit Report for {currentResult.model_id}
              </h3>
              <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.78rem' }}>
                Run ID: <code>{currentResult.run_id || 'latest'}</code> · Executed: {currentResult.executed_at ? new Date(currentResult.executed_at).toLocaleString() : currentResult.timestamp || 'Just now'}
              </p>
            </div>
            <MonitoringStatusBadge status={currentResult.overall_severity} />
          </div>

          {/* Recommendations & Warnings */}
          {(currentResult.recommendations?.length > 0 || currentResult.warnings?.length > 0) && (
            <div className="glass-card glass-card--padded" style={{ borderLeft: '4px solid var(--primary)' }}>
              <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.94rem', fontWeight: 600 }}>
                Automated Remediation & Insights
              </h4>
              {currentResult.recommendations?.map((rec, i) => (
                <p key={i} style={{ margin: '0.25rem 0', fontSize: '0.84rem', color: 'var(--ink)' }}>
                  💡 <strong>Recommended Action:</strong> {rec}
                </p>
              ))}
              {currentResult.warnings?.map((warn, i) => (
                <p key={i} style={{ margin: '0.25rem 0', fontSize: '0.84rem', color: '#b45309' }}>
                  ⚠️ {warn}
                </p>
              ))}
            </div>
          )}

          {/* Data Drift Section */}
          {currentResult.data_drift && (
            <div>
              <h3 className="section-title" style={{ margin: '0 0 0.85rem' }}>
                Feature & Dataset Drift Analysis
              </h3>
              <DriftPanel dataDrift={currentResult.data_drift} />
            </div>
          )}

          {/* Performance Degradation Section */}
          {currentResult.performance_drift && (
            <div>
              <h3 className="section-title" style={{ margin: '0 0 0.85rem' }}>
                Performance Degradation Tracking
              </h3>
              <PerformanceMonitoringPanel performanceDrift={currentResult.performance_drift} />
            </div>
          )}
        </div>
      ) : (
        !running && (
          <EmptyState
            icon={<IconActivity size={48} />}
            title="No monitoring reports generated yet"
            description="Select a model and an evaluation dataset above, then click 'Run Monitoring' to execute statistical hypothesis tests."
          />
        )
      )}

      {/* History Table */}
      <MonitoringHistory history={history} onSelectRun={(run) => setCurrentResult(run)} />
    </PageContainer>
  );
}
