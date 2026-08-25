import React, { useState } from 'react';
import type { DatasetProfile } from '../../types';

type ScenarioBuilderProps = {
  profile: DatasetProfile | null;
  onRun: (config: { target: string; scenarioName: string; pctChange: number; segment?: string }) => void;
  loading: boolean;
};

export default function ScenarioBuilder({ profile, onRun, loading }: ScenarioBuilderProps) {
  const numericColumns = profile?.column_names?.filter((col) => {
    const dtype = profile.data_types?.[col]?.toLowerCase() || '';
    return dtype.includes('int') || dtype.includes('float') || dtype.includes('num');
  }) || profile?.column_names || [];

  const [target, setTarget] = useState<string>(numericColumns[0] || '');
  const [scenarioName, setScenarioName] = useState<string>('Optimistic Growth (+15%)');
  const [pctChange, setPctChange] = useState<number>(15);

  const handlePreset = (name: string, pct: number) => {
    setScenarioName(name);
    setPctChange(pct);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!target) return;
    onRun({
      target,
      scenarioName,
      pctChange: pctChange / 100,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="glass-card glass-card--padded">
      <h3 style={{ margin: '0 0 0.85rem', fontSize: '1rem', fontWeight: 600 }}>
        Simulate Counterfactual Scenario
      </h3>

      {/* Preset Quick Chips */}
      <div style={{ marginBottom: '1rem' }}>
        <span className="muted" style={{ fontSize: '0.74rem', fontWeight: 600, textTransform: 'uppercase' }}>
          Quick Scenario Presets:
        </span>
        <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.35rem', flexWrap: 'wrap' }}>
          <button
            type="button"
            className="analyst-chip"
            onClick={() => handlePreset('Aggressive Expansion (+25%)', 25)}
          >
            🚀 Aggressive (+25%)
          </button>
          <button
            type="button"
            className="analyst-chip"
            onClick={() => handlePreset('Moderate Growth (+10%)', 10)}
          >
            📈 Moderate (+10%)
          </button>
          <button
            type="button"
            className="analyst-chip"
            onClick={() => handlePreset('Mild Contraction (-5%)', -5)}
          >
            📉 Contraction (-5%)
          </button>
          <button
            type="button"
            className="analyst-chip"
            onClick={() => handlePreset('Downside Shock (-20%)', -20)}
          >
            ⚠️ Downside (-20%)
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
        {/* Target Metric */}
        <div className="field">
          <label htmlFor="whatif-target" style={{ fontSize: '0.8rem', fontWeight: 600 }}>
            Target Metric *
          </label>
          <select
            id="whatif-target"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            className="horizon-input"
            style={{ width: '100%', padding: '0.45rem 0.65rem' }}
            required
            disabled={loading}
          >
            {numericColumns.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </div>

        {/* Scenario Name */}
        <div className="field">
          <label htmlFor="whatif-name" style={{ fontSize: '0.8rem', fontWeight: 600 }}>
            Scenario Name
          </label>
          <input
            id="whatif-name"
            type="text"
            value={scenarioName}
            onChange={(e) => setScenarioName(e.target.value)}
            className="horizon-input"
            style={{ width: '100%', padding: '0.45rem 0.65rem' }}
            required
            disabled={loading}
          />
        </div>

        {/* Perturbation Shift Slider */}
        <div className="field">
          <label htmlFor="whatif-pct" style={{ fontSize: '0.8rem', fontWeight: 600 }}>
            Hypothetical Shift ({pctChange > 0 ? `+${pctChange}%` : `${pctChange}%`})
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <input
              id="whatif-pct"
              type="range"
              min={-50}
              max={50}
              step={1}
              value={pctChange}
              onChange={(e) => setPctChange(Number(e.target.value))}
              disabled={loading}
              style={{ flex: 1 }}
            />
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.86rem',
                fontWeight: 700,
                color: pctChange >= 0 ? '#059669' : '#dc2626',
              }}
            >
              {pctChange > 0 ? `+${pctChange}%` : `${pctChange}%`}
            </span>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button
          type="submit"
          className="primary-btn"
          disabled={loading || !target}
          style={{ padding: '0.45rem 1.25rem', fontSize: '0.88rem' }}
        >
          {loading ? 'Running What-If Simulation…' : '🔮 Run What-If Simulation'}
        </button>
      </div>
    </form>
  );
}
