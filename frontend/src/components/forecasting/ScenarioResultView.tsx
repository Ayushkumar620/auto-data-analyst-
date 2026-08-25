import React from 'react';
import { Link } from 'react-router-dom';
import type { ScenarioResult } from '../../types';
import PlotlyChart from '../PlotlyChart';

type ScenarioResultViewProps = {
  result: ScenarioResult;
};

export default function ScenarioResultView({ result }: ScenarioResultViewProps) {
  const isPositive = result.percentage_difference >= 0;

  const comparisonPlotData = [
    {
      x: ['Baseline', result.scenario_name],
      y: [result.baseline_value, result.scenario_value],
      type: 'bar' as const,
      marker: {
        color: ['#64748b', isPositive ? '#059669' : '#dc2626'],
      },
      text: [
        result.baseline_value.toLocaleString(),
        result.scenario_value.toLocaleString(),
      ],
      textposition: 'auto' as const,
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '1.5rem' }}>
      {/* Metric Comparison Strip */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '0.85rem',
        }}
      >
        <div className="kpi-tile">
          <span className="kpi-icon" style={{ backgroundColor: '#f1f5f9', color: '#475569' }}>
            📊
          </span>
          <div>
            <p className="kpi-value">{result.baseline_value.toLocaleString()}</p>
            <p className="kpi-label">Baseline {result.target_metric}</p>
          </div>
        </div>

        <div className="kpi-tile">
          <span className="kpi-icon" style={{ backgroundColor: isPositive ? '#ecfdf5' : '#fef2f2', color: isPositive ? '#059669' : '#dc2626' }}>
            🔮
          </span>
          <div>
            <p className="kpi-value" style={{ color: isPositive ? '#059669' : '#dc2626' }}>
              {result.scenario_value.toLocaleString()}
            </p>
            <p className="kpi-label">Simulated Outcome</p>
          </div>
        </div>

        <div className="kpi-tile">
          <span className="kpi-icon" style={{ backgroundColor: '#eff6ff', color: '#1d4ed8' }}>
            ⚡
          </span>
          <div>
            <p className="kpi-value" style={{ color: isPositive ? '#059669' : '#dc2626' }}>
              {result.percentage_difference > 0 ? `+${result.percentage_difference.toFixed(1)}%` : `${result.percentage_difference.toFixed(1)}%`}
            </p>
            <p className="kpi-label">Relative Impact</p>
          </div>
        </div>

        <div className="kpi-tile">
          <span className="kpi-icon" style={{ backgroundColor: '#fdf4ff', color: '#9333ea' }}>
            Δ
          </span>
          <div>
            <p className="kpi-value">
              {result.absolute_difference > 0 ? `+${result.absolute_difference.toLocaleString()}` : result.absolute_difference.toLocaleString()}
            </p>
            <p className="kpi-label">Absolute Delta</p>
          </div>
        </div>
      </div>

      {/* Visual Comparison Chart */}
      <div className="glass-card glass-card--padded">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600 }}>
              Scenario Comparison: Baseline vs {result.scenario_name}
            </h3>
            <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.8rem' }}>
              Target: <strong>{result.target_metric}</strong> · Confidence: {Math.round(result.confidence * 100)}%
            </p>
          </div>

          <Link
            to="/analyst"
            className="action-btn"
            style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem', textDecoration: 'none' }}
          >
            ⚡ Ask Analyst about this scenario →
          </Link>
        </div>

        <PlotlyChart
          data={comparisonPlotData}
          layout={{
            title: `Simulated ${result.target_metric} Impact`,
            yaxis: { title: result.target_metric },
            height: 280,
          }}
        />
      </div>

      {/* Epistemic Non-Causal Attribution Warning Banner */}
      <div
        style={{
          padding: '0.85rem 1rem',
          borderRadius: '10px',
          backgroundColor: '#eff6ff',
          borderLeft: '4px solid #3b82f6',
          fontSize: '0.84rem',
          color: '#1e40af',
          lineHeight: '1.45',
        }}
      >
        <strong>Model-Based Simulation Note:</strong>
        <p style={{ margin: '0.25rem 0 0' }}>
          This counterfactual scenario represents an associative model projection holding other variables constant. It is an analytical decision-support estimate and does not constitute a guaranteed real-world causal outcome.
        </p>
      </div>

      {/* Assumptions List */}
      {result.assumptions && result.assumptions.length > 0 && (
        <div className="glass-card glass-card--padded">
          <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.9rem', fontWeight: 600 }}>
            Scenario Assumptions
          </h4>
          <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.82rem', color: 'var(--muted)' }}>
            {result.assumptions.map((assump, i) => (
              <li key={i}>{assump}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

