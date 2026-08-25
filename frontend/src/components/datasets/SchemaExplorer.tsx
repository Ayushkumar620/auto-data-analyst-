import React, { useState, useMemo } from 'react';
import type { DatasetProfile } from '../../types';

type SchemaExplorerProps = {
  profile: DatasetProfile;
};

export default function SchemaExplorer({ profile }: SchemaExplorerProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState('ALL');

  const columns = useMemo(() => {
    const names = profile.column_names && profile.column_names.length > 0
      ? profile.column_names
      : profile.preview && profile.preview.length > 0
      ? Object.keys(profile.preview[0])
      : [];

    return names.map((colName) => {
      const colMeta = profile.column_analysis?.[colName];
      const dtype = profile.data_types?.[colName] || colMeta?.dtype || colMeta?.type || 'unknown';
      const missingCount = colMeta?.missing ?? (colMeta?.missing_percentage !== undefined ? Math.round((colMeta.missing_percentage * profile.rows) / 100) : 0);
      const missingPct = colMeta?.missing_percentage ?? (profile.rows > 0 ? (missingCount / profile.rows) * 100 : 0);
      const uniqueCount = colMeta?.unique ?? profile.categorical_analysis?.[colName]?.cardinality ?? null;

      // Extract sample from preview
      const samples: unknown[] = [];
      if (profile.preview && Array.isArray(profile.preview)) {
        for (const row of profile.preview) {
          const val = row[colName];
          if (val !== null && val !== undefined && !samples.includes(val)) {
            samples.push(val);
            if (samples.length >= 3) break;
          }
        }
      }

      return {
        name: colName,
        dtype,
        missingCount,
        missingPct,
        uniqueCount,
        samples,
      };
    });
  }, [profile]);

  const availableTypes = useMemo(() => {
    const types = new Set<string>();
    columns.forEach((c) => {
      if (c.dtype.includes('int') || c.dtype.includes('float') || c.dtype.includes('double')) {
        types.add('Numeric');
      } else if (c.dtype.includes('date') || c.dtype.includes('time')) {
        types.add('Datetime');
      } else if (c.dtype.includes('bool')) {
        types.add('Boolean');
      } else {
        types.add('Categorical');
      }
    });
    return Array.from(types);
  }, [columns]);

  const filteredColumns = useMemo(() => {
    return columns.filter((col) => {
      const matchesSearch = col.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        col.dtype.toLowerCase().includes(searchQuery.toLowerCase());

      if (!matchesSearch) return false;
      if (typeFilter === 'ALL') return true;

      const isNumeric = col.dtype.includes('int') || col.dtype.includes('float') || col.dtype.includes('double');
      const isDatetime = col.dtype.includes('date') || col.dtype.includes('time');
      const isBool = col.dtype.includes('bool');

      if (typeFilter === 'Numeric') return isNumeric;
      if (typeFilter === 'Datetime') return isDatetime;
      if (typeFilter === 'Boolean') return isBool;
      if (typeFilter === 'Categorical') return !isNumeric && !isDatetime && !isBool;
      return true;
    });
  }, [columns, searchQuery, typeFilter]);

  if (!columns.length) {
    return <p className="muted" style={{ padding: '1rem', margin: 0 }}>No schema details available for this dataset.</p>;
  }

  return (
    <div style={{ display: 'grid', gap: '0.85rem' }}>
      {/* Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <input
          type="text"
          placeholder="Filter columns by name or type..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="horizon-input"
          style={{ width: '260px', padding: '0.4rem 0.75rem', fontSize: '0.84rem' }}
          aria-label="Filter columns"
        />

        <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
          <button
            type="button"
            className={typeFilter === 'ALL' ? 'primary-btn' : 'action-btn'}
            style={{ padding: '0.3rem 0.65rem', fontSize: '0.78rem' }}
            onClick={() => setTypeFilter('ALL')}
          >
            All ({columns.length})
          </button>
          {availableTypes.map((type) => (
            <button
              key={type}
              type="button"
              className={typeFilter === type ? 'primary-btn' : 'action-btn'}
              style={{ padding: '0.3rem 0.65rem', fontSize: '0.78rem' }}
              onClick={() => setTypeFilter(type)}
            >
              {type}
            </button>
          ))}
        </div>
      </div>

      {/* Schema Table */}
      <div
        className="table-shell"
        style={{
          border: '1px solid rgba(226, 232, 240, 0.9)',
          borderRadius: '12px',
          overflowX: 'auto',
          backgroundColor: '#ffffff',
        }}
      >
        <table className="result-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead style={{ backgroundColor: '#f8fafc' }}>
            <tr>
              <th style={{ padding: '0.65rem 0.85rem', width: '22%' }}>Column Name</th>
              <th style={{ padding: '0.65rem 0.85rem', width: '16%' }}>Data Type</th>
              <th style={{ padding: '0.65rem 0.85rem', width: '18%' }}>Missing</th>
              <th style={{ padding: '0.65rem 0.85rem', width: '14%' }}>Unique Count</th>
              <th style={{ padding: '0.65rem 0.85rem', width: '30%' }}>Sample Values</th>
            </tr>
          </thead>
          <tbody>
            {filteredColumns.map((col) => (
              <tr key={col.name}>
                <td style={{ fontWeight: 600, color: 'var(--ink)', padding: '0.6rem 0.85rem' }}>
                  {col.name}
                </td>
                <td style={{ padding: '0.6rem 0.85rem' }}>
                  <span
                    style={{
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.78rem',
                      padding: '0.2rem 0.5rem',
                      borderRadius: '6px',
                      backgroundColor: 'rgba(99, 102, 241, 0.08)',
                      color: 'var(--primary)',
                    }}
                  >
                    {col.dtype}
                  </span>
                </td>
                <td style={{ padding: '0.6rem 0.85rem' }}>
                  {col.missingCount > 0 ? (
                    <span style={{ color: 'var(--warning)', fontWeight: 600, fontSize: '0.84rem' }}>
                      {col.missingCount} ({col.missingPct.toFixed(1)}%)
                    </span>
                  ) : (
                    <span style={{ color: 'var(--success)', fontSize: '0.84rem' }}>0 (0%)</span>
                  )}
                </td>
                <td style={{ padding: '0.6rem 0.85rem', fontFamily: 'var(--font-mono)', fontSize: '0.84rem' }}>
                  {col.uniqueCount !== null ? col.uniqueCount.toLocaleString() : '—'}
                </td>
                <td style={{ padding: '0.6rem 0.85rem', color: 'var(--muted)', fontSize: '0.82rem' }}>
                  {col.samples.length > 0 ? (
                    <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap' }}>
                      {col.samples.map((s, idx) => (
                        <span
                          key={idx}
                          style={{
                            padding: '0.15rem 0.45rem',
                            borderRadius: '4px',
                            backgroundColor: '#f1f5f9',
                            maxWidth: '120px',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {String(s)}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <span style={{ fontStyle: 'italic' }}>—</span>
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
