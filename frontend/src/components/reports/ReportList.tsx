import React, { useState, useMemo } from 'react';
import type { ReportSummary } from '../../types';
import ReportCard from './ReportCard';
import EmptyState from '../ui/EmptyState';
import { IconFileText } from '../ui/Icons';

type ReportListProps = {
  reports: ReportSummary[];
  onDeleteReport?: (reportId: string) => void;
  onCreateClick?: () => void;
};

export default function ReportList({ reports, onDeleteReport, onCreateClick }: ReportListProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState('ALL');
  const [sortBy, setSortBy] = useState<'newest' | 'oldest' | 'title'>('newest');

  const filteredReports = useMemo(() => {
    return reports
      .filter((r) => {
        const matchesSearch =
          r.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
          r.dataset_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          r.executive_summary.toLowerCase().includes(searchTerm.toLowerCase());

        if (!matchesSearch) return false;
        if (typeFilter !== 'ALL' && r.report_type !== typeFilter) return false;

        return true;
      })
      .sort((a, b) => {
        if (sortBy === 'newest') {
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        }
        if (sortBy === 'oldest') {
          return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
        }
        return a.title.localeCompare(b.title);
      });
  }, [reports, searchTerm, typeFilter, sortBy]);

  if (reports.length === 0) {
    return (
      <EmptyState
        icon={<IconFileText size={48} />}
        title="No reports generated yet"
        description="Generate comprehensive analytical reports from your analyses, forecasts, models, or datasets."
        action={
          onCreateClick ? (
            <button type="button" onClick={onCreateClick} className="primary-btn">
              ⚡ Create Report
            </button>
          ) : undefined
        }
      />
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Controls */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
        <input
          type="text"
          placeholder="Search reports by title, dataset..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="horizon-input"
          style={{ minWidth: '240px', flex: 1, padding: '0.4rem 0.75rem', fontSize: '0.86rem' }}
          aria-label="Search reports"
        />

        <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            className="horizon-input"
            style={{ padding: '0.4rem 0.65rem', fontSize: '0.84rem' }}
          >
            <option value="ALL">All Types ({reports.length})</option>
            <option value="comprehensive">Comprehensive</option>
            <option value="forecast">Forecasting</option>
            <option value="model">Model Review</option>
            <option value="monitoring">Monitoring</option>
            <option value="analysis">Analysis</option>
          </select>

          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="horizon-input"
            style={{ padding: '0.4rem 0.65rem', fontSize: '0.84rem' }}
          >
            <option value="newest">Sort: Newest First</option>
            <option value="oldest">Sort: Oldest First</option>
            <option value="title">Sort: Title A-Z</option>
          </select>
        </div>
      </div>

      {/* Grid */}
      {filteredReports.length === 0 ? (
        <p className="muted" style={{ textAlign: 'center', padding: '2rem 0' }}>
          No reports match the selected search query.
        </p>
      ) : (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
            gap: '1.25rem',
          }}
        >
          {filteredReports.map((report) => (
            <ReportCard
              key={report.report_id}
              report={report}
              onDelete={onDeleteReport}
            />
          ))}
        </div>
      )}
    </div>
  );
}

