import React, { useState } from 'react';
import type { DatasetProfile } from '../../types';

type ForecastBuilderProps = {
  profile: DatasetProfile | null;
  onRun: (config: { targetColumn: string; timeColumn?: string; horizon: number; confidenceLevel: number }) => void;
  loading: boolean;
};

export default function ForecastBuilder({ profile, onRun, loading }: ForecastBuilderProps) {
  const numericColumns = profile?.column_names?.filter((col) => {
    const dtype = profile.data_types?.[col]?.toLowerCase() || '';
    return dtype.includes('int') || dtype.includes('float') || dtype.includes('num');
  }) || profile?.column_names || [];

  const dateColumns = profile?.column_names?.filter((col) => {
    const name = col.toLowerCase();
    const dtype = profile.data_types?.[col]?.toLowerCase() || '';
    return name.includes('date') || name.includes('time') || name.includes('year') || name.includes('month') || dtype.includes('datetime');
  }) || [];

  const [targetColumn, setTargetColumn] = useState<string>(numericColumns[0] || '');
  const [timeColumn, setTimeColumn] = useState<string>(dateColumns[0] || '');
  const [horizon, setHorizon] = useState<number>(6);
  const [confidenceLevel, setConfidenceLevel] = useState<number>(0.8);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!targetColumn) return;
    onRun({
      targetColumn,
      timeColumn: timeColumn || undefined,
      horizon,
      confidenceLevel,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="glass-card glass-card--padded">
      <h3 style={{ margin: '0 0 0.85rem', fontSize: '1rem', fontWeight: 600 }}>
        Configure Forecast Model
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
        {/* Target Metric Selection */}
        <div className="field">
          <label htmlFor="target-col" style={{ fontSize: '0.8rem', fontWeight: 600 }}>
            Target Metric *
          </label>
          <select
            id="target-col"
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            className="horizon-input"
            style={{ width: '100%', padding: '0.45rem 0.65rem' }}
            required
            disabled={loading}
          >
            {numericColumns.length === 0 ? (
              <option value="">No numeric columns available</option>
            ) : (
              numericColumns.map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))
            )}
          </select>
        </div>

        {/* Time / Date Column */}
        <div className="field">
          <label htmlFor="time-col" style={{ fontSize: '0.8rem', fontWeight: 600 }}>
            Time / Date Column (Optional)
          </label>
          <select
            id="time-col"
            value={timeColumn}
            onChange={(e) => setTimeColumn(e.target.value)}
            className="horizon-input"
            style={{ width: '100%', padding: '0.45rem 0.65rem' }}
            disabled={loading}
          >
            <option value="">Auto-Detect / Index Sequence</option>
            {dateColumns.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </div>

        {/* Forecast Horizon */}
        <div className="field">
          <label htmlFor="fc-horizon" style={{ fontSize: '0.8rem', fontWeight: 600 }}>
            Forecast Horizon ({horizon} periods)
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <input
              id="fc-horizon"
              type="range"
              min={1}
              max={24}
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              disabled={loading}
              style={{ flex: 1 }}
            />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.86rem', fontWeight: 600 }}>
              {horizon}
            </span>
          </div>
        </div>

        {/* Confidence Interval */}
        <div className="field">
          <label htmlFor="fc-conf" style={{ fontSize: '0.8rem', fontWeight: 600 }}>
            Confidence Interval
          </label>
          <select
            id="fc-conf"
            value={confidenceLevel}
            onChange={(e) => setConfidenceLevel(Number(e.target.value))}
            className="horizon-input"
            style={{ width: '100%', padding: '0.45rem 0.65rem' }}
            disabled={loading}
          >
            <option value={0.8}>80% Interval (Standard)</option>
            <option value={0.9}>90% Interval (Conservative)</option>
            <option value={0.95}>95% Interval (Strict)</option>
          </select>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button
          type="submit"
          className="primary-btn"
          disabled={loading || !targetColumn}
          style={{ padding: '0.45rem 1.25rem', fontSize: '0.88rem' }}
        >
          {loading ? 'Running Autonomous Forecaster…' : '⚡ Generate Forecast'}
        </button>
      </div>
    </form>
  );
}

