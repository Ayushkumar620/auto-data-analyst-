import React, { useState, useMemo } from 'react';

type PreviewTableProps = {
  preview: Array<Record<string, unknown>>;
  pageSizeOptions?: number[];
  initialPageSize?: number;
};

export default function PreviewTable({
  preview,
  pageSizeOptions = [10, 25, 50],
  initialPageSize = 10,
}: PreviewTableProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(initialPageSize);
  const [filterQuery, setFilterQuery] = useState('');

  const columns = useMemo(() => {
    if (!preview || preview.length === 0) return [];
    return Object.keys(preview[0]);
  }, [preview]);

  const filteredRows = useMemo(() => {
    if (!filterQuery.trim()) return preview;
    const query = filterQuery.toLowerCase();
    return preview.filter((row) =>
      columns.some((col) => {
        const val = row[col];
        return val !== null && val !== undefined && String(val).toLowerCase().includes(query);
      }),
    );
  }, [preview, columns, filterQuery]);

  const totalPages = Math.ceil(filteredRows.length / pageSize) || 1;
  const safePage = Math.min(currentPage, totalPages);

  const paginatedRows = useMemo(() => {
    const start = (safePage - 1) * pageSize;
    return filteredRows.slice(start, start + pageSize);
  }, [filteredRows, safePage, pageSize]);

  if (!preview || !preview.length) {
    return <p className="muted" style={{ padding: '1rem', margin: 0 }}>No preview data available.</p>;
  }

  return (
    <div className="preview-table-container" style={{ display: 'grid', gap: '0.75rem' }}>
      {/* Controls */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.5rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <input
            type="text"
            placeholder="Search preview rows..."
            value={filterQuery}
            onChange={(e) => {
              setFilterQuery(e.target.value);
              setCurrentPage(1);
            }}
            className="horizon-input"
            style={{ width: '220px', padding: '0.4rem 0.65rem', fontSize: '0.84rem' }}
            aria-label="Filter preview rows"
          />
          <span className="muted" style={{ fontSize: '0.8rem' }}>
            Showing {filteredRows.length} of {preview.length} rows
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <label htmlFor="preview-page-size" className="muted" style={{ fontSize: '0.8rem' }}>
            Per page:
          </label>
          <select
            id="preview-page-size"
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setCurrentPage(1);
            }}
            className="horizon-input"
            style={{ padding: '0.35rem 0.5rem', fontSize: '0.82rem' }}
          >
            {pageSizeOptions.map((size) => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Table shell */}
      <div
        className="table-shell"
        style={{
          maxHeight: '440px',
          overflowY: 'auto',
          overflowX: 'auto',
          border: '1px solid rgba(226, 232, 240, 0.9)',
          borderRadius: '12px',
          backgroundColor: '#ffffff',
        }}
      >
        <table className="result-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead style={{ position: 'sticky', top: 0, zIndex: 2, backgroundColor: '#f8fafc' }}>
            <tr>
              <th style={{ width: '48px', color: 'var(--muted)', fontSize: '0.75rem', fontWeight: 600 }}>#</th>
              {columns.map((column) => (
                <th key={column} style={{ whiteSpace: 'nowrap', padding: '0.65rem 0.85rem' }}>
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paginatedRows.length === 0 ? (
              <tr>
                <td colSpan={columns.length + 1} style={{ textAlign: 'center', padding: '1.5rem' }} className="muted">
                  No matching rows found.
                </td>
              </tr>
            ) : (
              paginatedRows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  <td style={{ color: 'var(--muted)', fontSize: '0.75rem', padding: '0.55rem 0.75rem' }}>
                    {(safePage - 1) * pageSize + rowIndex + 1}
                  </td>
                  {columns.map((column) => {
                    const value = row[column];
                    const isNull = value === null || value === undefined || value === '';
                    return (
                      <td
                        key={column}
                        style={{
                          padding: '0.55rem 0.85rem',
                          fontFamily: typeof value === 'number' ? 'var(--font-mono)' : 'inherit',
                          fontSize: '0.86rem',
                        }}
                      >
                        {isNull ? (
                          <span style={{ color: 'var(--muted)', fontStyle: 'italic', opacity: 0.6 }}>null</span>
                        ) : typeof value === 'boolean' ? (
                          value ? 'true' : 'false'
                        ) : typeof value === 'object' ? (
                          JSON.stringify(value)
                        ) : (
                          String(value)
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.25rem' }}>
          <span className="muted" style={{ fontSize: '0.8rem' }}>
            Page {safePage} of {totalPages}
          </span>
          <div style={{ display: 'flex', gap: '0.35rem' }}>
            <button
              type="button"
              className="action-btn"
              style={{ padding: '0.3rem 0.65rem', fontSize: '0.8rem' }}
              disabled={safePage <= 1}
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            >
              Previous
            </button>
            <button
              type="button"
              className="action-btn"
              style={{ padding: '0.3rem 0.65rem', fontSize: '0.8rem' }}
              disabled={safePage >= totalPages}
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
