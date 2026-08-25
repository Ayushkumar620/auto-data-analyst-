import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { PageContainer, PageHeader, Card } from '../components/layout/PageContainer';
import ForecastBuilder from '../components/forecasting/ForecastBuilder';
import ForecastResultView from '../components/forecasting/ForecastResultView';
import ScenarioBuilder from '../components/forecasting/ScenarioBuilder';
import ScenarioResultView from '../components/forecasting/ScenarioResultView';
import EmptyState from '../components/ui/EmptyState';
import ErrorState from '../components/ui/ErrorState';
import { useDataset } from '../context/DatasetContext';
import { useNotification } from '../context/NotificationContext';
import { runForecast, runWhatIfScenario } from '../services/forecastService';
import type { ForecastResult, ScenarioResult } from '../types';
import { IconTrendUp, IconDatabase, IconBrain, IconCheck } from '../components/ui/Icons';

type TabKey = 'forecast' | 'whatif';

export default function ForecastsPage() {
  const { profile, fileName } = useDataset();
  const { notify } = useNotification();
  const navigate = useNavigate();

  const datasetName = fileName || profile?.dataset_name;

  const [activeTab, setActiveTab] = useState<TabKey>('forecast');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Results
  const [forecastResult, setForecastResult] = useState<ForecastResult | null>(null);
  const [scenarioResult, setScenarioResult] = useState<ScenarioResult | null>(null);

  // Natural language prompt input
  const [nlPrompt, setNlPrompt] = useState('');

  const handleRunForecast = async (config: {
    targetColumn: string;
    timeColumn?: string;
    horizon: number;
    confidenceLevel: number;
  }) => {
    if (!profile || !profile.preview || profile.preview.length === 0) {
      setError('Please select or upload a dataset with sample data first.');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const res = await runForecast({
        dataset: profile.preview,
        target_column: config.targetColumn,
        time_column: config.timeColumn,
        forecast_horizon: config.horizon,
        confidence_level: config.confidenceLevel,
      });

      if (res.status === 'NOT_SUPPORTED') {
        setError(res.warnings?.[0] || 'Dataset is not suitable for time-series forecasting.');
        setForecastResult(null);
      } else {
        setForecastResult(res);
        notify('Time-series forecast computed successfully!', 'success');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate forecast');
    } finally {
      setLoading(false);
    }
  };

  const handleRunWhatIf = async (config: {
    target: string;
    scenarioName: string;
    pctChange: number;
  }) => {
    if (!profile || !profile.preview || profile.preview.length === 0) {
      setError('Please select or upload a dataset with sample data first.');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const res = await runWhatIfScenario({
        dataset: profile.preview,
        target: config.target,
        scenario_name: config.scenarioName,
        changed_variables: { pct: config.pctChange },
        assumptions: [`Target ${config.target} shifted by ${config.pctChange > 0 ? `+${(config.pctChange * 100).toFixed(0)}%` : `${(config.pctChange * 100).toFixed(0)}%`}`],
      });

      if ('scenario_name' in res) {
        setScenarioResult(res as ScenarioResult);
        notify('What-If scenario simulation completed!', 'success');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to simulate scenario');
    } finally {
      setLoading(false);
    }
  };

  const handleNlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!nlPrompt.trim()) return;
    navigate('/analyst');
  };

  return (
    <PageContainer>
      <PageHeader
        eyebrow="Predictive Intelligence"
        title="Forecasting & What-If Scenario Analysis"
        subtitle="Autonomous candidate model benchmarking, probabilistic intervals, and counterfactual scenario modeling."
      />

      {/* Dataset Context Bar */}
      {profile ? (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            backgroundColor: 'rgba(99, 102, 241, 0.08)',
            border: '1px solid rgba(99, 102, 241, 0.25)',
            borderRadius: '12px',
            padding: '0.75rem 1rem',
            marginBottom: '1.25rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <IconDatabase size={18} color="var(--primary)" aria-hidden />
            <div>
              <p style={{ margin: 0, fontSize: '0.88rem', fontWeight: 600, color: 'var(--ink)' }}>
                Active Dataset: {datasetName}
              </p>
              <p className="muted" style={{ margin: '0.1rem 0 0', fontSize: '0.78rem' }}>
                {profile.rows.toLocaleString()} rows · {profile.columns} columns available
              </p>
            </div>
          </div>
          <Link
            to="/datasets"
            className="action-btn"
            style={{ padding: '0.3rem 0.65rem', fontSize: '0.78rem', textDecoration: 'none' }}
          >
            Switch Dataset
          </Link>
        </div>
      ) : (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            backgroundColor: '#fffbeb',
            border: '1px solid #fde68a',
            borderRadius: '12px',
            padding: '0.75rem 1rem',
            marginBottom: '1.25rem',
          }}
        >
          <p style={{ margin: 0, fontSize: '0.86rem', color: '#92400e' }}>
            No dataset selected. Choose an active dataset to enable time-series forecasting.
          </p>
          <Link
            to="/datasets"
            className="primary-btn"
            style={{ padding: '0.3rem 0.65rem', fontSize: '0.78rem', textDecoration: 'none' }}
          >
            Select Dataset →
          </Link>
        </div>
      )}

      {/* Natural Language Forecasting Quick Box */}
      <div className="glass-card glass-card--padded" style={{ marginBottom: '1.25rem' }}>
        <h3 style={{ margin: '0 0 0.5rem', fontSize: '0.94rem', fontWeight: 600 }}>
          ⚡ Describe a Forecast or Scenario in Natural Language
        </h3>
        <form onSubmit={handleNlSubmit} style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
          <input
            type="text"
            placeholder='e.g. "Forecast sales for the next 6 months with 90% confidence" or "What if revenue drops 10%?"'
            value={nlPrompt}
            onChange={(e) => setNlPrompt(e.target.value)}
            className="horizon-input"
            style={{ flex: 1, minWidth: '280px', padding: '0.45rem 0.75rem' }}
          />
          <button type="submit" className="action-btn" disabled={!nlPrompt.trim()}>
            Ask AI Analyst →
          </button>
        </form>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: '0.5rem', borderBottom: '1px solid #e2e8f0', marginBottom: '1.25rem' }}>
        <button
          type="button"
          onClick={() => setActiveTab('forecast')}
          style={{
            padding: '0.6rem 1rem',
            background: 'none',
            border: 'none',
            borderBottom: activeTab === 'forecast' ? '2px solid var(--primary)' : '2px solid transparent',
            color: activeTab === 'forecast' ? 'var(--primary)' : 'var(--muted)',
            fontWeight: activeTab === 'forecast' ? 700 : 500,
            cursor: 'pointer',
            fontSize: '0.92rem',
          }}
        >
          📈 Time-Series Forecasting
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('whatif')}
          style={{
            padding: '0.6rem 1rem',
            background: 'none',
            border: 'none',
            borderBottom: activeTab === 'whatif' ? '2px solid var(--primary)' : '2px solid transparent',
            color: activeTab === 'whatif' ? 'var(--primary)' : 'var(--muted)',
            fontWeight: activeTab === 'whatif' ? 700 : 500,
            cursor: 'pointer',
            fontSize: '0.92rem',
          }}
        >
          🔮 What-If Scenario Simulation
        </button>
      </div>

      {error && <ErrorState message={error} />}

      {/* Tab Contents */}
      {activeTab === 'forecast' && (
        <div>
          <ForecastBuilder profile={profile} onRun={handleRunForecast} loading={loading} />

          {forecastResult ? (
            <ForecastResultView result={forecastResult} historicalData={profile?.preview} />
          ) : (
            !loading && (
              <EmptyState
                icon={<IconTrendUp size={40} />}
                title="No forecast generated yet"
                description="Configure target metric and horizon above, then click 'Generate Forecast'."
              />
            )
          )}
        </div>
      )}

      {activeTab === 'whatif' && (
        <div>
          <ScenarioBuilder profile={profile} onRun={handleRunWhatIf} loading={loading} />

          {scenarioResult ? (
            <ScenarioResultView result={scenarioResult} />
          ) : (
            !loading && (
              <EmptyState
                icon={<IconBrain size={40} />}
                title="No What-If scenarios created yet"
                description="Select target metric and adjust the shift slider above to test alternative scenarios."
              />
            )
          )}
        </div>
      )}
    </PageContainer>
  );
}

