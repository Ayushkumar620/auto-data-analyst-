import React from 'react';
import type { MonitoringResult } from '../../types';
import MonitoringStatusBadge from './MonitoringStatusBadge';

type MonitoringHistoryProps = {
  history: MonitoringResult[];
  onSelectRun?: (run: MonitoringResult) => void;
};

export default function MonitoringHistory({ history, onSelectRun }: MonitoringHistoryProps) {
  if (!history || history.length === 0) {
    return (
      <div className="glass-card glass-card--padded">
        <h4 style={{ margin: '0 0 0.4rem', fontSize: '0.94rem', fontWeight: 600 }}>
          Monitoring Run History
        </h4>
        <p className="muted" style={{ margin: 0, fontSize: '0.84rem' }}>
          No monitoring runs recorded yet. Execute a monitoring run above to record baseline audits.
        </p>
      </div>
    );
  }

  return (
    <div className="glass-card glass-card--padded">
      <h4 style={{ margin: '0 0 0.75rem', fontSize: '0.94rem', fontWeight: 600 }}>
        Monitoring Run Audit History ({history.length})
      </h4>

      <div style={{ overflowX: 'auto', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
        <table className="result-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Run ID</th>
              <th>Model</th>
              <th>Timestamp</th>
              <th>Severity</th>
              <th>Drifted Features</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {history.map((run, i) => (
              <tr key={run.run_id || i}>
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem' }}>
                  {run.run_id || `run_${i + 1}`}
                </td>
                <td style={{ fontWeight: 600 }}>{run.model_id}</td>
                <td className="muted" style={{ fontSize: '0.78rem' }}>
                  {run.executed_at ? new Date(run.executed_at).toLocaleString() : run.timestamp || '—'}
                </td>
                <td>
                  <MonitoringStatusBadge status={run.overall_severity} />
                </td>
                <td>
                  {run.data_drift?.drifted_features?.length || 0} / {run.data_drift?.features_checked?.length || 0}
                </td>
                <td>
                  {onSelectRun && (
                    <button
                      type="button"
                      className="action-btn"
                      onClick={() => onSelectRun(run)}
                      style={{ padding: '0.2rem 0.55rem', fontSize: '0.74rem' }}
                    >
                      View Report
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

