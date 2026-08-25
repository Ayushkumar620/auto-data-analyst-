import React from 'react';
import type { ModelPerformanceReport } from '../../types';
import MonitoringStatusBadge from './MonitoringStatusBadge';

type PerformanceMonitoringPanelProps = {
  performanceDrift: ModelPerformanceReport;
};

export default function PerformanceMonitoringPanel({ performanceDrift }: PerformanceMonitoringPanelProps) {
  const refMetrics = performanceDrift.reference_metrics || {};
  const currMetrics = performanceDrift.current_metrics || {};
  const metricChanges = performanceDrift.metric_changes || {};

  const allKeys = Array.from(new Set([...Object.keys(refMetrics), ...Object.keys(currMetrics)]));

  if (performanceDrift.target_monitoring_status === 'unavailable' || allKeys.length === 0) {
    return (
      <div className="glass-card glass-card--padded">
        <h4 style={{ margin: '0 0 0.4rem', fontSize: '0.94rem', fontWeight: 600 }}>
          Ground-Truth Performance Evaluation
        </h4>
        <p className="muted" style={{ margin: 0, fontSize: '0.84rem' }}>
          Ground-truth target labels were not present in the current evaluation batch. Performance metrics cannot be directly evaluated without verified labels.
        </p>
      </div>
    );
  }

  return (
    <div className="glass-card glass-card--padded" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div>
          <h4 style={{ margin: 0, fontSize: '0.94rem', fontWeight: 600 }}>
            Model Performance vs Baseline
          </h4>
          <p className="muted" style={{ margin: '0.15rem 0 0', fontSize: '0.78rem' }}>
            Evaluated on {performanceDrift.evaluation_dataset_rows} verified labeled records
          </p>
        </div>

        <MonitoringStatusBadge
          status={performanceDrift.degradation_detected ? 'CRITICAL' : 'HEALTHY'}
        />
      </div>

      <div style={{ overflowX: 'auto', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
        <table className="result-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Metric</th>
              <th>Reference (Baseline)</th>
              <th>Current (Monitored)</th>
              <th>Absolute Delta</th>
            </tr>
          </thead>
          <tbody>
            {allKeys.map((key) => {
              const refVal = refMetrics[key] !== undefined ? refMetrics[key] : '—';
              const currVal = currMetrics[key] !== undefined ? currMetrics[key] : '—';
              const change = metricChanges[key];

              const isDrop = typeof change === 'number' && change < 0;

              return (
                <tr key={key}>
                  <td style={{ fontWeight: 600 }}>{key}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {typeof refVal === 'number' ? refVal.toFixed(4) : refVal}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                    {typeof currVal === 'number' ? currVal.toFixed(4) : currVal}
                  </td>
                  <td
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 600,
                      color: isDrop ? '#dc2626' : change > 0 ? '#059669' : 'inherit',
                    }}
                  >
                    {change !== undefined
                      ? change > 0
                        ? `+${change.toFixed(4)}`
                        : change.toFixed(4)
                      : '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

