import React from 'react';
import type { DatasetProfile } from '../types';

export type SummaryDataProps = {
  data?: {
    rows?: number;
    columns?: number;
    nulls?: number;
    missing_values?: number;
    shape?: { rows: number; columns: number };
    total_null_cells?: number;
    dtypes?: Record<string, string>;
    data_types?: Record<string, string>;
    column_names?: string[];
    columns_list?: string[];
    preview?: Array<Record<string, unknown>>;
    head?: Array<Record<string, unknown>>;
    dataset_name?: string;
    name?: string;
  };
  profile?: DatasetProfile | {
    dataset_name?: string;
    rows?: number;
    columns?: number;
    missing_values?: number;
    duplicates?: number;
    column_names?: string[];
    data_types?: Record<string, string>;
    dtypes?: Record<string, string>;
    preview?: Array<Record<string, unknown>>;
    head?: Array<Record<string, unknown>>;
    memory_usage?: string;
    quality_score?: number;
  };
  className?: string;
};

type ColumnTypeItem = {
  name: string;
  type: string;
};

/**
 * Normalizes raw datatype strings into clean, readable labels.
 * e.g. 'int64' -> 'integer', 'float64' -> 'float', 'object' -> 'string'
 */
function normalizeTypeName(rawType: string = 'string'): string {
  const lower = rawType.toLowerCase();
  if (lower.includes('int')) return 'integer';
  if (lower.includes('float') || lower.includes('double')) return 'float';
  if (lower.includes('date') || lower.includes('time')) return 'datetime';
  if (lower.includes('bool')) return 'boolean';
  if (lower.includes('object') || lower.includes('str') || lower.includes('category') || lower.includes('text')) {
    return 'string';
  }
  return rawType;
}

/**
 * Clean Badge Component for Column Types
 */
function TypeBadge({ type }: { type: string }) {
  const t = type.toLowerCase();
  let bg = '#eff6ff';
  let color = '#1d4ed8';
  let border = '#bfdbfe';

  if (t === 'integer' || t === 'float' || t === 'number') {
    bg = '#f5f3ff';
    color = '#6d28d9';
    border = '#ddd6fe';
  } else if (t === 'datetime') {
    bg = '#ecfdf5';
    color = '#047857';
    border = '#a7f3d0';
  } else if (t === 'boolean') {
    bg = '#fffbeb';
    color = '#b45309';
    border = '#fde68a';
  }

  return (
    <span
      style={{
        display: 'inline-block',
        padding: '0.15rem 0.5rem',
        borderRadius: '6px',
        fontSize: '0.75rem',
        fontWeight: 600,
        fontFamily: 'var(--font-mono, monospace)',
        backgroundColor: bg,
        color: color,
        border: `1px solid ${border}`,
        textTransform: 'lowercase',
      }}
    >
      {type}
    </span>
  );
}

/**
 * SummaryHeader Sub-component
 */
export function SummaryHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
      <div>
        <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 700, color: 'var(--ink, #0f172a)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span>📋</span> {title}
        </h3>
        {subtitle && (
          <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.8rem' }}>
            {subtitle}
          </p>
        )}
      </div>
    </div>
  );
}

/**
 * MetricCards Sub-component (Rows, Columns, Null Values)
 */
export function MetricCards({ rows, columns, nulls }: { rows: number; columns: number; nulls: number }) {
  return (
    <div
      className="summary-metric-cards"
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
        gap: '1rem',
        marginBottom: '1.25rem',
      }}
    >
      {/* Rows Card */}
      <div
        className="summary-metric-card"
        style={{
          backgroundColor: '#ffffff',
          border: '1px solid var(--border, #e2e8f0)',
          borderRadius: '12px',
          padding: '1rem 1.25rem',
          boxShadow: '0 1px 3px rgba(15, 23, 42, 0.04)',
          transition: 'transform 180ms ease, box-shadow 180ms ease',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
          <span
            style={{
              fontSize: '0.74rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              color: 'var(--muted, #64748b)',
            }}
          >
            ROWS
          </span>
          <span style={{ fontSize: '1rem', opacity: 0.7 }} aria-hidden="true">↕</span>
        </div>
        <p
          style={{
            margin: 0,
            fontSize: '1.85rem',
            fontWeight: 700,
            fontFamily: 'var(--font-heading, sans-serif)',
            color: 'var(--ink, #0f172a)',
            lineHeight: 1.1,
          }}
          data-testid="summary-rows-value"
        >
          {rows.toLocaleString()}
        </p>
      </div>

      {/* Columns Card */}
      <div
        className="summary-metric-card"
        style={{
          backgroundColor: '#ffffff',
          border: '1px solid var(--border, #e2e8f0)',
          borderRadius: '12px',
          padding: '1rem 1.25rem',
          boxShadow: '0 1px 3px rgba(15, 23, 42, 0.04)',
          transition: 'transform 180ms ease, box-shadow 180ms ease',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
          <span
            style={{
              fontSize: '0.74rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              color: 'var(--muted, #64748b)',
            }}
          >
            COLUMNS
          </span>
          <span style={{ fontSize: '1rem', opacity: 0.7 }} aria-hidden="true">↔</span>
        </div>
        <p
          style={{
            margin: 0,
            fontSize: '1.85rem',
            fontWeight: 700,
            fontFamily: 'var(--font-heading, sans-serif)',
            color: 'var(--ink, #0f172a)',
            lineHeight: 1.1,
          }}
          data-testid="summary-columns-value"
        >
          {columns.toLocaleString()}
        </p>
      </div>

      {/* Null Values Card */}
      <div
        className="summary-metric-card"
        style={{
          backgroundColor: '#ffffff',
          border: '1px solid var(--border, #e2e8f0)',
          borderRadius: '12px',
          padding: '1rem 1.25rem',
          boxShadow: '0 1px 3px rgba(15, 23, 42, 0.04)',
          transition: 'transform 180ms ease, box-shadow 180ms ease',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
          <span
            style={{
              fontSize: '0.74rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              color: 'var(--muted, #64748b)',
            }}
          >
            NULL VALUES
          </span>
          <span style={{ fontSize: '1rem', opacity: 0.7 }} aria-hidden="true">⊘</span>
        </div>
        <p
          style={{
            margin: 0,
            fontSize: '1.85rem',
            fontWeight: 700,
            fontFamily: 'var(--font-heading, sans-serif)',
            color: nulls > 0 ? '#d97706' : '#059669',
            lineHeight: 1.1,
          }}
          data-testid="summary-nulls-value"
        >
          {nulls.toLocaleString()}
        </p>
      </div>
    </div>
  );
}

/**
 * ColumnTypesTable Sub-component
 */
export function ColumnTypesTable({ columns }: { columns: ColumnTypeItem[] }) {
  if (!columns || columns.length === 0) return null;

  return (
    <div
      className="column-types-card"
      style={{
        backgroundColor: '#ffffff',
        border: '1px solid var(--border, #e2e8f0)',
        borderRadius: '14px',
        padding: '1.25rem',
        marginBottom: '1.25rem',
        boxShadow: '0 1px 3px rgba(15, 23, 42, 0.04)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
        <h4 style={{ margin: 0, fontSize: '0.98rem', fontWeight: 700, color: 'var(--ink, #0f172a)', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
          <span>🧩</span> Columns &amp; Types
        </h4>
        <span
          style={{
            fontSize: '0.75rem',
            fontWeight: 600,
            color: 'var(--muted, #64748b)',
            backgroundColor: '#f8fafc',
            padding: '0.2rem 0.55rem',
            borderRadius: '6px',
            border: '1px solid #e2e8f0',
          }}
        >
          {columns.length} columns
        </span>
      </div>

      <div
        className="table-responsive-container"
        style={{
          overflowX: 'auto',
          maxWidth: '100%',
          borderRadius: '8px',
          border: '1px solid var(--border, #e2e8f0)',
          margin: 0,
        }}
      >
        <table className="result-table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.84rem' }}>
          <thead>
            <tr style={{ backgroundColor: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
              <th style={{ padding: '0.55rem 0.85rem', width: '50%', textAlign: 'left', fontWeight: 600, color: 'var(--ink-secondary, #334155)' }}>
                Column
              </th>
              <th style={{ padding: '0.55rem 0.85rem', width: '50%', textAlign: 'left', fontWeight: 600, color: 'var(--ink-secondary, #334155)' }}>
                Type
              </th>
            </tr>
          </thead>
          <tbody>
            {columns.map((col, idx) => (
              <tr
                key={col.name}
                style={{
                  backgroundColor: idx % 2 === 0 ? '#ffffff' : '#fcfdfe',
                  borderBottom: '1px solid #f1f5f9',
                }}
              >
                <td style={{ padding: '0.5rem 0.85rem', fontFamily: 'var(--font-mono, monospace)', fontWeight: 600, color: 'var(--ink, #0f172a)' }}>
                  {col.name}
                </td>
                <td style={{ padding: '0.5rem 0.85rem' }}>
                  <TypeBadge type={col.type} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/**
 * DataPreviewTable Sub-component
 */
export function DataPreviewTable({
  columns,
  rows,
}: {
  columns: string[];
  rows: Array<Record<string, unknown>>;
}) {
  if (!rows || rows.length === 0) return null;

  return (
    <div
      className="data-preview-card"
      style={{
        backgroundColor: '#ffffff',
        border: '1px solid var(--border, #e2e8f0)',
        borderRadius: '14px',
        padding: '1.25rem',
        boxShadow: '0 1px 3px rgba(15, 23, 42, 0.04)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <h4 style={{ margin: 0, fontSize: '0.98rem', fontWeight: 700, color: 'var(--ink, #0f172a)', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
          <span>👀</span> Data Preview
        </h4>
        <span
          style={{
            fontSize: '0.75rem',
            fontWeight: 600,
            color: 'var(--muted, #64748b)',
            backgroundColor: '#f8fafc',
            padding: '0.2rem 0.55rem',
            borderRadius: '6px',
            border: '1px solid #e2e8f0',
          }}
        >
          First {rows.length} rows
        </span>
      </div>

      <div
        className="table-responsive-container"
        style={{
          overflowX: 'auto',
          maxWidth: '100%',
          borderRadius: '8px',
          border: '1px solid var(--border, #e2e8f0)',
          margin: 0,
        }}
      >
        <table className="result-table" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.84rem' }}>
          <thead>
            <tr style={{ backgroundColor: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
              {columns.map((col, idx) => (
                <th
                  key={col}
                  style={{
                    padding: '0.55rem 0.85rem',
                    textAlign: 'left',
                    fontWeight: 600,
                    color: 'var(--ink-secondary, #334155)',
                    whiteSpace: 'nowrap',
                    borderRight: idx < columns.length - 1 ? '1px solid #f1f5f9' : undefined,
                  }}
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rIdx) => (
              <tr
                key={rIdx}
                style={{
                  backgroundColor: rIdx % 2 === 0 ? '#ffffff' : '#fcfdfe',
                  borderBottom: '1px solid #f1f5f9',
                }}
              >
                {columns.map((col, cIdx) => {
                  const val = row[col];
                  const isNull =
                    val === null ||
                    val === undefined ||
                    (typeof val === 'number' && isNaN(val)) ||
                    (typeof val === 'string' && val.trim() === '');
                  const isNum = typeof val === 'number';

                  return (
                    <td
                      key={col}
                      style={{
                        padding: '0.5rem 0.85rem',
                        textAlign: isNum ? 'right' : 'left',
                        whiteSpace: 'nowrap',
                        color: isNull ? 'var(--muted, #94a3b8)' : 'var(--ink, #0f172a)',
                        borderRight: cIdx < columns.length - 1 ? '1px solid #f8fafc' : undefined,
                      }}
                    >
                      {isNull ? (
                        <span style={{ fontStyle: 'italic', fontWeight: 500 }}>—</span>
                      ) : isNum ? (
                        val.toLocaleString()
                      ) : (
                        String(val)
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/**
 * Main SummaryData Component
 * Clean, professional boxed dashboard structure for dataset summaries.
 */
export default function SummaryData({ data, profile, className }: SummaryDataProps) {
  // 1. Extract Metrics
  const rows = data?.rows ?? data?.shape?.rows ?? profile?.rows ?? 0;
  const columnsCount = data?.columns ?? data?.shape?.columns ?? profile?.columns ?? 0;
  const nulls =
    data?.nulls ??
    data?.total_null_cells ??
    data?.missing_values ??
    profile?.missing_values ??
    0;

  const datasetName =
    data?.dataset_name ??
    data?.name ??
    profile?.dataset_name ??
    'Dataset Summary';

  // 2. Extract Columns & Types
  const rawTypes =
    data?.data_types ??
    data?.dtypes ??
    profile?.data_types ??
    (profile as any)?.dtypes ??
    {};

  const previewList =
    data?.preview ??
    data?.head ??
    profile?.preview ??
    (profile as any)?.head ??
    [];

  const columnNames =
    data?.column_names ??
    data?.columns_list ??
    profile?.column_names ??
    (Object.keys(rawTypes).length > 0
      ? Object.keys(rawTypes)
      : previewList.length > 0
      ? Object.keys(previewList[0])
      : []);

  const columnTypes: ColumnTypeItem[] = columnNames.map((name) => {
    const raw = rawTypes[name] || 'string';
    return {
      name,
      type: normalizeTypeName(raw),
    };
  });

  // 3. Extract Preview
  const previewRows = previewList.slice(0, 10);
  const previewColumns =
    previewRows.length > 0
      ? Object.keys(previewRows[0])
      : columnNames;

  return (
    <div
      className={`summary-dashboard-container${className ? ` ${className}` : ''}`}
      style={{ display: 'grid', gap: '1.25rem', width: '100%', maxWidth: '100%' }}
      data-testid="summary-data-dashboard"
    >
      {/* Box 1: Dataset Summary & Metric Cards */}
      <div
        className="summary-box-card"
        style={{
          backgroundColor: '#ffffff',
          border: '1px solid var(--border, #e2e8f0)',
          borderRadius: '14px',
          padding: '1.25rem',
          boxShadow: '0 1px 3px rgba(15, 23, 42, 0.04)',
        }}
      >
        <SummaryHeader title={datasetName ? `Dataset Summary: ${datasetName}` : 'Dataset Summary'} />
        <MetricCards rows={rows} columns={columnsCount || columnNames.length} nulls={nulls} />
      </div>

      {/* Box 2: Columns & Types Table */}
      {columnTypes.length > 0 && <ColumnTypesTable columns={columnTypes} />}

      {/* Box 3: Data Preview Table */}
      {previewRows.length > 0 && (
        <DataPreviewTable columns={previewColumns} rows={previewRows} />
      )}
    </div>
  );
}
