import React, { useState, useMemo } from 'react';
import type { DatasetDriftReport, FeatureDriftResult } from '../../types';
import MonitoringStatusBadge from './MonitoringStatusBadge';

type DriftTableProps = {
  dataDrift: DatasetDriftReport;
};

export default function DriftTable({ dataDrift }: DriftTableProps) {
  const [filterSeverity, setFilterSeverity] = useState<string>('ALL');
  const [searchTerm, setSearchTerm] = useState<string>('');

  const featureEntries = Object.entries(dataDrift.feature_results || {});

  const filteredFeatures = useMemo(() => {
    return featureEntries.filter(([featName, res]) => {
      const matchesSearch = featName.toLowerCase().includes(searchTerm.toLowerCase());
      if (!matchesSearch) return false;
      if (filterSeverity !== 'ALL') {
        if (filterSeverity === 'DRIFTED' && !res.drift_detected) return false;
        if (filterSeverity === 'HEALTHY' && res.drift_detected) return false;
      }
      return true;
    });
  }, [featureEntries, searchTerm, filterSeverity]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
      {/* Search & Severity Filter */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.6rem' }}>
        <input
          type="text"
          placeholder="Filter features..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="horizon-input"
          style={{ width: '220px', padding: '0.35rem 0.65rem', fontSize: '0.82rem' }}
          aria-label="Filter feature drift"
        />

        <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
          <span className="muted" style={{ fontSize: '0.76rem' }}>Show:</span>
          <select
            value={filterSeverity}
            onChange={(e) => setFilterSeverity(e.target.value)}
            className="horizon-input"
            style={{ padding: '0.35rem 0.65rem', fontSize: '0.82rem' }}
          >
            <option value="ALL">All Features ({featureEntries.length})</option>
            <option value="DRIFTED">Drifted Only ({dataDrift.drifted_features?.length || 0})</option>
            <option value="HEALTHY">Stable Features</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
        <table className="result-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Feature</th>
              <th>Status</th>
              <th>Drift Score</th>
              <th>p-Value</th>
              <th>Statistical Test</th>
              <th>Threshold</th>
            </tr>
          </thead>
          <tbody>
            {filteredFeatures.length === 0 ? (
              <tr>
                <td colSpan={6} style={{ textAlign: 'center', padding: '1.5rem', color: 'var(--muted)' }}>
                  No features match the selected filter.
                </td>
              </tr>
            ) : (
              filteredFeatures.map(([featName, res]) => (
                <tr key={featName}>
                  <td style={{ fontWeight: 600, color: 'var(--ink)' }}>{featName}</td>
                  <td>
                    <MonitoringStatusBadge status={res.severity || (res.drift_detected ? 'HIGH' : 'HEALTHY')} />
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                    {res.drift_score.toFixed(4)}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }} className="muted">
                    {res.p_value !== undefined && res.p_value !== null ? res.p_value.toFixed(4) : '—'}
                  </td>
                  <td style={{ fontSize: '0.78rem' }}>
                    <code style={{ background: '#f1f5f9', padding: '0.1rem 0.35rem', borderRadius: '4px' }}>
                      {res.statistical_test}
                    </code>
                  </td>
                  <td className="muted" style={{ fontFamily: 'var(--font-mono)' }}>
                    {res.threshold}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
