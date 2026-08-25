import React, { useEffect, useState } from 'react';
import { PageContainer, PageHeader } from '../components/layout/PageContainer';
import ReportList from '../components/reports/ReportList';
import ReportBuilder from '../components/reports/ReportBuilder';
import ErrorState from '../components/ui/ErrorState';
import { LoadingSpinner } from '../components/ui/LoadingState';
import { useDataset } from '../context/DatasetContext';
import { useNotification } from '../context/NotificationContext';
import {
  listReports,
  createReport,
  deleteReport,
  type CreateReportParams,
} from '../services/reportService';
import type { ReportSummary } from '../types';

export default function ReportsPage() {
  const { profile } = useDataset();
  const { notify } = useNotification();

  const [reports, setReports] = useState<ReportSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const [showBuilder, setShowBuilder] = useState(false);
  const [creating, setCreating] = useState(false);

  const loadReports = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await listReports();
      setReports(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load reports');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadReports();
  }, []);

  const handleCreateReport = async (params: CreateReportParams) => {
    setCreating(true);
    setError('');
    try {
      const newRep = await createReport(params);
      notify(`Report "${newRep.title}" created successfully!`, 'success');
      setShowBuilder(false);
      loadReports();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create report');
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteReport = async (reportId: string) => {
    try {
      await deleteReport(reportId);
      notify('Report deleted successfully.', 'info');
      setReports((prev) => prev.filter((r) => r.report_id !== reportId));
    } catch (err) {
      notify(err instanceof Error ? err.message : 'Failed to delete report', 'error');
    }
  };

  return (
    <PageContainer>
      <PageHeader
        eyebrow="Deliverables & Outputs"
        title="Analytical Reports"
        subtitle="Review, export, and share comprehensive findings synthesized across your workspace."
        actions={
          !showBuilder && (
            <button
              type="button"
              onClick={() => setShowBuilder(true)}
              className="primary-btn"
            >
              ⚡ Create Report
            </button>
          )
        }
      />

      {error && <ErrorState message={error} />}

      {/* Report Builder Form (if active) */}
      {showBuilder && (
        <div style={{ marginBottom: '1.5rem' }}>
          <ReportBuilder
            profile={profile}
            onSubmit={handleCreateReport}
            onCancel={() => setShowBuilder(false)}
            loading={creating}
          />
        </div>
      )}

      {/* Main List */}
      {loading ? (
        <LoadingSpinner label="Loading analytical reports…" size={36} />
      ) : (
        <ReportList
          reports={reports}
          onDeleteReport={handleDeleteReport}
          onCreateClick={() => setShowBuilder(true)}
        />
      )}
    </PageContainer>
  );
}

