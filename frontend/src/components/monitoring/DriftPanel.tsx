import React from 'react';
import type { DatasetDriftReport } from '../../types';
import DriftTable from './DriftTable';
import PlotlyChart from '../PlotlyChart';
import MonitoringStatusBadge from './MonitoringStatusBadge';

type DriftPanelProps = {
  dataDrift: DatasetDriftReport;
};

export default function DriftPanel({ dataDrift }: DriftPanelProps) {
  const featureEntries = Object.entries(dataDrift.feature_results || {});

  // Prepare horizontal bar chart for drift scores
  const sortedEntries = [...featureEntries].sort((a, b) => b[1].drift_score - a[1].drift_score).slice(0, 12);
  const featNames = sortedEntries.map((e) => e[0]);
  const driftScores = sortedEntries.map((e) => e[1].drift_score);
  const colors = sortedEntries.map((e) => (e[1].drift_detected ? '#ef4444' : '#10b981'));

  const chartData = [
    {
      x: driftScores,
      y: featNames,
      type: 'bar' as const,
      orientation: 'h' as const,
      marker: { color: colors },
      text: driftScores.map((s) => s.toFixed(3)),
      textposition: 'auto' as const,
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Top summary row */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
          gap: '0.85rem',
        }}
      >
        <div className="kpi-tile">
          <span className="kpi-icon" style={{ backgroundColor: '#eff6ff', color: '#1d4ed8' }}>
            🔍
          </span>
          <div>
            <p className="kpi-value">{dataDrift.features_checked?.length || 0}</p>
            <p className="kpi-label">Features Checked</p>
          </div>
        </div>

        <div className="kpi-tile">
          <span
            className="kpi-icon"
            style={{
              backgroundColor: dataDrift.overall_drift ? '#fef2f2' : '#ecfdf5',
              color: dataDrift.overall_drift ? '#dc2626' : '#059669',
            }}
          >
            {dataDrift.overall_drift ? '⚠️' : '✅'}
          </span>
          <div>
            <p className="kpi-value" style={{ color: dataDrift.overall_drift ? '#dc2626' : '#059669' }}>
              {dataDrift.drifted_features?.length || 0}
            </p>
            <p className="kpi-label">Drifted Features</p>
          </div>
        </div>

        <div className="kpi-tile">
          <span className="kpi-icon" style={{ backgroundColor: '#fdf4ff', color: '#9333ea' }}>
            %
          </span>
          <div>
            <p className="kpi-value">{dataDrift.drift_percentage.toFixed(1)}%</p>
            <p className="kpi-label">Dataset Drift Share</p>
          </div>
        </div>

        <div className="kpi-tile">
          <span className="kpi-icon" style={{ backgroundColor: '#f8fafc', color: '#475569' }}>
            🛡️
          </span>
          <div>
            <div style={{ marginTop: '0.2rem' }}>
              <MonitoringStatusBadge status={dataDrift.severity || (dataDrift.overall_drift ? 'HIGH' : 'HEALTHY')} />
            </div>
            <p className="kpi-label" style={{ marginTop: '0.2rem' }}>Overall Severity</p>
          </div>
        </div>
      </div>

      {/* Drift Scores Bar Chart */}
      {sortedEntries.length > 0 && (
        <div className="glass-card glass-card--padded">
          <h4 style={{ margin: '0 0 0.5rem', fontSize: '0.94rem', fontWeight: 600 }}>
            Feature Drift Divergence Scores
          </h4>
          <p className="muted" style={{ margin: '0 0 0.75rem', fontSize: '0.78rem' }}>
            Red bars indicate features exceeding the statistical drift significance threshold.
          </p>
          <PlotlyChart
            data={chartData}
            layout={{
              title: 'Statistical Drift Score by Feature',
              xaxis: { title: 'Drift Score (KS Statistic / PSI)' },
              yaxis: { autorange: 'reversed' },
              height: 280,
            }}
          />
        </div>
      )}

      {/* Feature Drift Detailed Table */}
      <div className="glass-card glass-card--padded">
        <h4 style={{ margin: '0 0 0.75rem', fontSize: '0.94rem', fontWeight: 600 }}>
          Individual Feature Statistical Results
        </h4>
        <DriftTable dataDrift={dataDrift} />
      </div>
    </div>
  );
}
