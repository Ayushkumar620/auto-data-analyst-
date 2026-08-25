import React from 'react';
import { Link } from 'react-router-dom';
import type { ForecastResult } from '../../types';
import ForecastSummary from './ForecastSummary';
import PlotlyChart from '../PlotlyChart';

type ForecastResultViewProps = {
  result: ForecastResult;
  historicalData?: Array<Record<string, unknown>>;
};

export default function ForecastResultView({ result, historicalData }: ForecastResultViewProps) {
  const points = result.predictions || [];

  // Prepare plot traces
  const timestamps = points.map((p) => p.timestamp);
  const predictions = points.map((p) => p.prediction);
  const lowerBounds = points.map((p) => p.lower_bound);
  const upperBounds = points.map((p) => p.upper_bound);

  const traces: any[] = [];

  // Historical trace if available
  if (historicalData && historicalData.length > 0 && result.target) {
    const histY = historicalData.map((d) => Number(d[result.target]) || 0).slice(-20);
    const histX = historicalData.map((d, i) => String(d[result.time_column] || `T-${histY.length - i}`)).slice(-20);

    traces.push({
      x: histX,
      y: histY,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Historical Actuals',
      line: { color: '#64748b', width: 2 },
    });
  }

  // Upper bound (for shaded area)
  traces.push({
    x: timestamps,
    y: upperBounds,
    type: 'scatter',
    mode: 'lines',
    line: { width: 0 },
    showlegend: false,
    name: 'Upper Bound',
  });

  // Lower bound (fill to upper)
  traces.push({
    x: timestamps,
    y: lowerBounds,
    type: 'scatter',
    mode: 'lines',
    fill: 'tonexty',
    fillcolor: 'rgba(99, 102, 241, 0.15)',
    line: { width: 0 },
    name: `${Math.round(result.confidence_level * 100)}% Confidence Interval`,
  });

  // Forecast trajectory
  traces.push({
    x: timestamps,
    y: predictions,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Forecast Trajectory',
    line: { color: '#4f46e5', width: 3, dash: 'dash' },
    marker: { size: 6, color: '#4f46e5' },
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', marginTop: '1.5rem' }}>
      {/* Top summary strip */}
      <ForecastSummary result={result} />

      {/* Main Forecast Chart */}
      <div className="glass-card glass-card--padded">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600 }}>
              Forecast Trajectory for {result.target}
            </h3>
            <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.8rem' }}>
              Model: <code>{result.model_name}</code> ({result.model_family}) · Horizon: {result.forecast_horizon} periods
            </p>
          </div>

          <Link
            to="/analyst"
            className="action-btn"
            style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem', textDecoration: 'none' }}
          >
            ⚡ Ask Analyst about this forecast →
          </Link>
        </div>

        <PlotlyChart
          data={traces}
          layout={{
            title: `Projected ${result.target} (${result.forecast_horizon} Periods Ahead)`,
            xaxis: { title: 'Time Period / Sequence' },
            yaxis: { title: result.target },
            height: 360,
            showlegend: true,
            legend: { orientation: 'h', y: -0.2 },
          }}
        />
      </div>

      {/* Predictions Table & Model Metadata */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}>
        {/* Trajectory Table */}
        <div className="glass-card glass-card--padded">
          <h4 style={{ margin: '0 0 0.65rem', fontSize: '0.92rem', fontWeight: 600 }}>
            Projected Values
          </h4>
          <div style={{ maxHeight: '240px', overflowY: 'auto' }}>
            <table className="result-table" style={{ width: '100%' }}>
              <thead>
                <tr>
                  <th>Period</th>
                  <th>Forecast</th>
                  <th>Lower Bound</th>
                  <th>Upper Bound</th>
                </tr>
              </thead>
              <tbody>
                {points.map((p) => (
                  <tr key={p.timestamp}>
                    <td style={{ fontWeight: 500 }}>{p.timestamp}</td>
                    <td style={{ fontWeight: 700, color: 'var(--primary)' }}>{p.prediction.toLocaleString()}</td>
                    <td className="muted">{p.lower_bound.toLocaleString()}</td>
                    <td className="muted">{p.upper_bound.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Validation & Epistemic Assumptions */}
        <div className="glass-card glass-card--padded">
          <h4 style={{ margin: '0 0 0.65rem', fontSize: '0.92rem', fontWeight: 600 }}>
            Model Evaluation & Epistemic Limitations
          </h4>

          {result.validation_metrics && Object.keys(result.validation_metrics).length > 0 && (
            <div style={{ marginBottom: '0.75rem' }}>
              <span className="muted" style={{ fontSize: '0.74rem', fontWeight: 600, textTransform: 'uppercase' }}>Validation Metrics:</span>
              <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.2rem' }}>
                {Object.entries(result.validation_metrics).map(([k, v]) => (
                  <span key={k} style={{ fontSize: '0.78rem', background: '#f1f5f9', padding: '0.15rem 0.45rem', borderRadius: '4px' }}>
                    <strong>{k}:</strong> {v}
                  </span>
                ))}
              </div>
            </div>
          )}

          {result.limitations && result.limitations.length > 0 && (
            <div style={{ marginTop: '0.5rem' }}>
              <span className="muted" style={{ fontSize: '0.74rem', fontWeight: 600, textTransform: 'uppercase' }}>Limitations & Warnings:</span>
              <ul style={{ margin: '0.3rem 0 0', paddingLeft: '1.2rem', fontSize: '0.8rem', color: 'var(--muted)' }}>
                {result.limitations.map((lim, i) => (
                  <li key={i}>{lim}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
