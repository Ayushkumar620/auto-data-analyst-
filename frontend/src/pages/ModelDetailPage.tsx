import React, { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { PageContainer } from '../components/layout/PageContainer';
import StatusChip from '../components/models/StatusChip';
import MetricBadge from '../components/models/MetricBadge';
import PlotlyChart from '../components/PlotlyChart';
import ErrorState from '../components/ui/ErrorState';
import { LoadingSpinner } from '../components/ui/LoadingState';
import {
  getModelMetadata,
  updateModelStatus,
  deleteModel,
  runModelInference,
} from '../services/modelService';
import { useNotification } from '../context/NotificationContext';
import { deployModelEndpoint } from '../services/enterpriseService';
import type { InferenceResponse, ModelMetadata, ModelStatus } from '../types';
import { IconChevronRight, IconBrain, IconCheck, IconAlertTriangle } from '../components/ui/Icons';

export default function ModelDetailPage() {
  const { modelId } = useParams<{ modelId: string }>();
  const [model, setModel] = useState<ModelMetadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [statusUpdating, setStatusUpdating] = useState(false);

  // Inference state
  const [inferenceInput, setInferenceInput] = useState<Record<string, any>>({});
  const [inferenceResult, setInferenceResult] = useState<InferenceResponse | null>(null);
  const [inferring, setInferring] = useState(false);
  const [inferenceError, setInferenceError] = useState('');

  const { notify } = useNotification();
  const navigate = useNavigate();

  const loadModel = async () => {
    if (!modelId) return;
    setLoading(true);
    setError('');
    try {
      const data = await getModelMetadata(modelId);
      setModel(data);

      // Initialize default inference input with zeros/empty
      const initial: Record<string, any> = {};
      data.feature_columns.forEach((col) => {
        const dtype = data.feature_dtypes?.[col] || '';
        initial[col] = dtype.includes('int') || dtype.includes('float') ? 0 : '';
      });
      setInferenceInput(initial);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to retrieve model');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadModel();
  }, [modelId]);

  const handleStatusChange = async (newStatus: ModelStatus) => {
    if (!model) return;
    setStatusUpdating(true);
    try {
      await updateModelStatus(model.model_id, newStatus);
      setModel({ ...model, status: newStatus });
      notify(`Model status updated to "${newStatus}".`, 'success');
    } catch (e) {
      notify(e instanceof Error ? e.message : 'Failed to update status', 'error');
    } finally {
      setStatusUpdating(false);
    }
  };

  const handleDelete = async () => {
    if (!model) return;
    if (!window.confirm(`Are you sure you want to delete model "${model.name}"?`)) return;

    try {
      await deleteModel(model.model_id);
      notify(`Model "${model.name}" was deleted.`, 'info');
      navigate('/models');
    } catch (e) {
      notify(e instanceof Error ? e.message : 'Failed to delete model', 'error');
    }
  };

  const handleRunInference = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!model) return;

    setInferring(true);
    setInferenceError('');
    try {
      const res = await runModelInference(model.model_id, [inferenceInput]);
      setInferenceResult(res);
      notify('Inference computed successfully!', 'success');
    } catch (err) {
      setInferenceError(err instanceof Error ? err.message : 'Inference execution failed');
    } finally {
      setInferring(false);
    }
  };

  if (loading) {
    return (
      <PageContainer>
        <LoadingSpinner label="Loading model details…" size={36} />
      </PageContainer>
    );
  }

  if (error || !model) {
    return (
      <PageContainer>
        <ErrorState message={error || 'Model not found.'} onRetry={() => navigate('/models')} />
      </PageContainer>
    );
  }

  // Prepare loss curve chart data
  const lossCurvePlotData =
    model.loss_curve && model.loss_curve.length > 0
      ? [
          {
            x: model.loss_curve.map((_, i) => i + 1),
            y: model.loss_curve,
            type: 'scatter' as const,
            mode: 'lines+markers' as const,
            name: 'Training Loss',
            line: { color: '#4f46e5', width: 2 },
          },
        ]
      : null;

  // Prepare feature importance chart data
  const featureImpEntries = model.feature_importances ? Object.entries(model.feature_importances) : [];
  featureImpEntries.sort((a, b) => b[1] - a[1]);

  const featureImportancePlotData =
    featureImpEntries.length > 0
      ? [
          {
            x: featureImpEntries.slice(0, 10).map((e) => e[1]),
            y: featureImpEntries.slice(0, 10).map((e) => e[0]),
            type: 'bar' as const,
            orientation: 'h' as const,
            marker: { color: '#06b6d4' },
          },
        ]
      : null;

  return (
    <PageContainer>
      {/* Breadcrumbs */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem', fontSize: '0.84rem' }}>
          <Link to="/models" className="muted" style={{ textDecoration: 'none' }}>
            Models
          </Link>
          <IconChevronRight size={14} className="muted" aria-hidden />
          <span style={{ fontWeight: 600, color: 'var(--ink)' }}>{model.name}</span>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem', flexWrap: 'wrap' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <h1 className="page-title" style={{ margin: 0 }}>{model.name}</h1>
              <StatusChip status={model.status} />
              <span className="muted" style={{ fontSize: '0.82rem' }}>v{model.version}</span>
            </div>
            <p className="page-subtitle" style={{ marginTop: '0.3rem' }}>
              {model.algorithm} · Target: <strong>{model.target_column}</strong> · {model.problem_type.replace(/_/g, ' ')}
            </p>
          </div>

          {/* Status actions */}
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              type="button"
              className="primary-btn"
              onClick={async () => {
                try {
                  const dep = await deployModelEndpoint(model.model_id, `${model.name}_endpoint`);
                  notify(`Model live REST endpoint active at ${dep.endpoint_path}!`, 'success');
                } catch (e: any) {
                  notify(e.message || 'Deployment failed', 'error');
                }
              }}
              style={{ padding: '0.4rem 0.85rem', fontSize: '0.82rem', background: '#4f46e5' }}
            >
              🚀 Deploy Live REST Endpoint
            </button>

            <select
              value={model.status.toLowerCase()}
              onChange={(e) => handleStatusChange(e.target.value as ModelStatus)}
              disabled={statusUpdating}
              className="horizon-input"
              style={{ padding: '0.4rem 0.75rem', fontSize: '0.84rem' }}
              aria-label="Change model deployment status"
            >
              <option value="active">Promote to Active</option>
              <option value="staging">Set as Staging</option>
              <option value="archived">Archive Model</option>
            </select>

            <button
              type="button"
              className="ghost-text-btn"
              onClick={handleDelete}
              style={{ color: 'var(--alert)', fontSize: '0.84rem' }}
            >
              Delete Model
            </button>
          </div>
        </div>
      </div>

      {/* Metrics Section */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
        {/* Validation Metrics */}
        <div className="glass-card glass-card--padded">
          <h3 style={{ margin: '0 0 0.75rem', fontSize: '0.96rem', fontWeight: 600 }}>Validation Performance</h3>
          <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
            {Object.entries(model.validation_metrics || {}).map(([k, v]) => (
              <MetricBadge key={k} name={k} value={v} />
            ))}
          </div>
        </div>

        {/* Training Metrics */}
        <div className="glass-card glass-card--padded">
          <h3 style={{ margin: '0 0 0.75rem', fontSize: '0.96rem', fontWeight: 600 }}>Training Metrics</h3>
          <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
            {Object.entries(model.training_metrics || {}).map(([k, v]) => (
              <MetricBadge key={k} name={k} value={v} />
            ))}
          </div>
        </div>
      </div>

      {/* Visualizations: Loss Curve & Feature Importances */}
      {(lossCurvePlotData || featureImportancePlotData) && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.25rem' }}>
          {lossCurvePlotData && (
            <div className="glass-card glass-card--padded">
              <h3 style={{ margin: '0 0 0.75rem', fontSize: '0.96rem', fontWeight: 600 }}>Training Loss Curve</h3>
              <PlotlyChart
                data={lossCurvePlotData}
                layout={{
                  title: 'Loss per Epoch / Iteration',
                  xaxis: { title: 'Epoch / Step' },
                  yaxis: { title: 'Loss' },
                  height: 260,
                }}
              />
            </div>
          )}

          {featureImportancePlotData && (
            <div className="glass-card glass-card--padded">
              <h3 style={{ margin: '0 0 0.75rem', fontSize: '0.96rem', fontWeight: 600 }}>Top Feature Importances</h3>
              <PlotlyChart
                data={featureImportancePlotData}
                layout={{
                  title: 'Feature Importance Score',
                  yaxis: { autorange: 'reversed' },
                  height: 260,
                }}
              />
            </div>
          )}
        </div>
      )}

      {/* Schema & Hyperparameters */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.25rem' }}>
        {/* Feature Schema */}
        <div className="glass-card glass-card--padded">
          <h3 style={{ margin: '0 0 0.75rem', fontSize: '0.96rem', fontWeight: 600 }}>
            Input Feature Schema ({model.feature_columns.length} features)
          </h3>
          <div style={{ maxHeight: '240px', overflowY: 'auto' }}>
            <table className="result-table" style={{ width: '100%' }}>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Expected Type</th>
                </tr>
              </thead>
              <tbody>
                {model.feature_columns.map((col) => (
                  <tr key={col}>
                    <td style={{ fontWeight: 500 }}>{col}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: 'var(--primary)' }}>
                      {model.feature_dtypes?.[col] || 'numeric'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Hyperparameters */}
        <div className="glass-card glass-card--padded">
          <h3 style={{ margin: '0 0 0.75rem', fontSize: '0.96rem', fontWeight: 600 }}>Hyperparameters</h3>
          <div style={{ maxHeight: '240px', overflowY: 'auto' }}>
            <table className="result-table" style={{ width: '100%' }}>
              <thead>
                <tr>
                  <th>Parameter</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(model.hyperparameters || {}).map(([param, val]) => (
                  <tr key={param}>
                    <td style={{ fontWeight: 500 }}>{param}</td>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>
                      {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Interactive Live Inference Form */}
      <div className="glass-card glass-card--padded">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600 }}>Live Model Inference</h3>
            <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.84rem' }}>
              Test real-time predictions by entering feature values.
            </p>
          </div>
        </div>

        <form onSubmit={handleRunInference}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0.85rem', marginBottom: '1.25rem' }}>
            {model.feature_columns.map((col) => (
              <div key={col} className="field">
                <label htmlFor={`infer-${col}`} style={{ fontSize: '0.8rem', fontWeight: 600 }}>
                  {col}
                </label>
                <input
                  id={`infer-${col}`}
                  value={inferenceInput[col] ?? ''}
                  onChange={(e) => setInferenceInput({ ...inferenceInput, [col]: e.target.value })}
                  className="horizon-input"
                  style={{ width: '100%', padding: '0.4rem 0.65rem' }}
                />
              </div>
            ))}
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
            <button type="submit" className="primary-btn" disabled={inferring}>
              {inferring ? 'Running Prediction…' : '⚡ Run Prediction'}
            </button>
          </div>
        </form>

        {inferenceError && <p className="status-error" style={{ marginTop: '1rem' }}>{inferenceError}</p>}

        {/* Prediction Results */}
        {inferenceResult && (
          <div
            style={{
              marginTop: '1.25rem',
              padding: '1rem',
              borderRadius: '12px',
              backgroundColor: '#ecfdf5',
              border: '1px solid #a7f3d0',
            }}
          >
            <h4 style={{ margin: '0 0 0.5rem', color: '#065f46', fontSize: '0.95rem' }}>
              Prediction Output
            </h4>
            <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
              <div>
                <p className="metric-label" style={{ color: '#047857' }}>Predicted {model.target_column}</p>
                <p style={{ margin: '0.2rem 0 0', fontSize: '1.4rem', fontWeight: 700, fontFamily: 'var(--font-heading)', color: '#065f46' }}>
                  {inferenceResult.predictions.join(', ')}
                </p>
              </div>
              {inferenceResult.probabilities && (
                <div>
                  <p className="metric-label" style={{ color: '#047857' }}>Probabilities</p>
                  <p style={{ margin: '0.2rem 0 0', fontSize: '0.88rem', fontFamily: 'var(--font-mono)' }}>
                    {JSON.stringify(inferenceResult.probabilities)}
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </PageContainer>
  );
}

