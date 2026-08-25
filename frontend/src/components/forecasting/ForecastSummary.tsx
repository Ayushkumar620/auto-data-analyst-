import React from 'react';
import type { ForecastResult } from '../../types';

type ForecastSummaryProps = {
  result: ForecastResult;
};

export default function ForecastSummary({ result }: ForecastSummaryProps) {
  const points = result.predictions || [];
  const firstPred = points.length > 0 ? points[0].prediction : 0;
  const lastPred = points.length > 0 ? points[points.length - 1].prediction : 0;
  const delta = lastPred - firstPred;
  const pctChange = firstPred !== 0 ? (delta / Math.abs(firstPred)) * 100 : 0;

  const isPositive = pctChange >= 0;

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
        gap: '0.85rem',
        margin: '1rem 0',
      }}
    >
      <div className="kpi-tile">
        <span className="kpi-icon" style={{ backgroundColor: '#eff6ff', color: '#1d4ed8' }}>
          🎯
        </span>
        <div>
          <p className="kpi-value">{result.target}</p>
          <p className="kpi-label">Target Metric</p>
        </div>
      </div>

      <div className="kpi-tile">
        <span className="kpi-icon" style={{ backgroundColor: '#f5f3ff', color: '#6d28d9' }}>
          🤖
        </span>
        <div>
          <p className="kpi-value" style={{ fontSize: '1.05rem' }}>{result.model_name}</p>
          <p className="kpi-label">Algorithm</p>
        </div>
      </div>

      <div className="kpi-tile">
        <span className="kpi-icon" style={{ backgroundColor: isPositive ? '#ecfdf5' : '#fef2f2', color: isPositive ? '#059669' : '#dc2626' }}>
          {isPositive ? '📈' : '📉'}
        </span>
        <div>
          <p className="kpi-value" style={{ color: isPositive ? '#059669' : '#dc2626' }}>
            {pctChange > 0 ? `+${pctChange.toFixed(1)}%` : `${pctChange.toFixed(1)}%`}
          </p>
          <p className="kpi-label">Trajectory Delta</p>
        </div>
      </div>

      <div className="kpi-tile">
        <span className="kpi-icon" style={{ backgroundColor: '#fffbeb', color: '#d97706' }}>
          🛡️
        </span>
        <div>
          <p className="kpi-value">{Math.round(result.confidence_level * 100)}%</p>
          <p className="kpi-label">Confidence Interval</p>
        </div>
      </div>
    </div>
  );
}
