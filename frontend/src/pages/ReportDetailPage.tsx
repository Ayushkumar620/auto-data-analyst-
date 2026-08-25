import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { PageContainer } from '../components/layout/PageContainer';
import ExecutiveSummary from '../components/reports/ExecutiveSummary';
import ReportMetrics from '../components/reports/ReportMetrics';
import ReportInsights from '../components/reports/ReportInsights';
import ReportEvidence from '../components/reports/ReportEvidence';
import ErrorState from '../components/ui/ErrorState';
import { LoadingSpinner } from '../components/ui/LoadingState';
import { useNotification } from '../context/NotificationContext';
import { getReportDetail, downloadExecutivePdf } from '../services/reportService';
import type { ReportDetail } from '../types';
import { IconChevronRight, IconDatabase, IconFileText } from '../components/ui/Icons';

export default function ReportDetailPage() {
  const { reportId } = useParams<{ reportId: string }>();
  const { notify } = useNotification();

  const [report, setReport] = useState<ReportDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!reportId) return;
    setLoading(true);
    setError('');

    getReportDetail(reportId)
      .then(setReport)
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load report details'))
      .finally(() => setLoading(false));
  }, [reportId]);

  const handleDownloadPdf = async () => {
    if (!report) return;
    setDownloading(true);
    try {
      const blob = await downloadExecutivePdf({
        title: report.title,
        command: `Executive Analysis for ${report.dataset_name}`,
        explanation: report.executive_summary,
        kpis: report.kpis?.reduce((acc, k) => ({ ...acc, [k.name]: k.value }), {}) || {},
        evidence_list: report.evidence as any[] || [],
        dataset_summary: report.dataset_overview || {},
      });

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${report.report_id}_executive_report.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      notify('Executive PDF report downloaded successfully!', 'success');
    } catch (err) {
      notify(err instanceof Error ? err.message : 'Failed to download PDF report', 'error');
    } finally {
      setDownloading(false);
    }
  };

  if (loading) {
    return (
      <PageContainer>
        <LoadingSpinner label="Loading analytical report deliverable…" size={36} />
      </PageContainer>
    );
  }

  if (error || !report) {
    return (
      <PageContainer>
        <ErrorState message={error || 'Report not found.'} />
        <Link to="/reports" className="action-btn" style={{ marginTop: '1rem', display: 'inline-block', textDecoration: 'none' }}>
          ← Back to Reports
        </Link>
      </PageContainer>
    );
  }

  const formattedDate = new Date(report.created_at).toLocaleDateString('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });

  return (
    <PageContainer>
      {/* Breadcrumb */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem', fontSize: '0.84rem' }}>
          <Link to="/reports" className="muted" style={{ textDecoration: 'none' }}>
            Reports
          </Link>
          <IconChevronRight size={14} className="muted" aria-hidden />
          <span style={{ fontWeight: 600, color: 'var(--ink)' }}>{report.title}</span>
        </div>

        {/* Report Header Card */}
        <div
          className="glass-card glass-card--padded"
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            flexWrap: 'wrap',
            gap: '1rem',
            marginBottom: '1.5rem',
          }}
        >
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.35rem' }}>
              <span
                style={{
                  fontSize: '0.72rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  backgroundColor: 'rgba(99, 102, 241, 0.1)',
                  color: 'var(--primary)',
                  padding: '0.15rem 0.5rem',
                  borderRadius: '4px',
                }}
              >
                {report.report_type}
              </span>
              <span
                style={{
                  fontSize: '0.72rem',
                  fontWeight: 600,
                  backgroundColor: '#ecfdf5',
                  color: '#059669',
                  padding: '0.15rem 0.5rem',
                  borderRadius: '4px',
                }}
              >
                {report.status}
              </span>
            </div>

            <h1 className="page-title" style={{ margin: '0 0 0.4rem', fontSize: '1.45rem' }}>
              {report.title}
            </h1>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.82rem', color: 'var(--muted)' }}>
              <IconDatabase size={14} aria-hidden />
              <span>Dataset: <strong>{report.dataset_name}</strong></span>
              <span>·</span>
              <span>Published: {formattedDate}</span>
            </div>
          </div>

          {/* Header Actions */}
          <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center', flexWrap: 'wrap' }}>
            <Link
              to="/analyst"
              className="action-btn"
              style={{ padding: '0.4rem 0.85rem', fontSize: '0.82rem', textDecoration: 'none' }}
            >
              ⚡ Ask Analyst
            </Link>

            <button
              type="button"
              onClick={handleDownloadPdf}
              className="primary-btn"
              disabled={downloading}
              style={{ padding: '0.4rem 1rem', fontSize: '0.82rem' }}
            >
              {downloading ? 'Compiling PDF…' : '📥 Download Executive PDF'}
            </button>
          </div>
        </div>
      </div>

      {/* Report Body Structure */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
        {/* Executive Summary */}
        <ExecutiveSummary summary={report.executive_summary} />

        {/* KPIs */}
        {report.kpis && report.kpis.length > 0 && (
          <ReportMetrics kpis={report.kpis} />
        )}

        {/* Insights */}
        {report.insights && report.insights.length > 0 && (
          <ReportInsights insights={report.insights} />
        )}

        {/* Evidence */}
        {report.evidence && report.evidence.length > 0 && (
          <ReportEvidence evidence={report.evidence} />
        )}

        {/* Recommendations */}
        {report.recommendations && report.recommendations.length > 0 && (
          <div className="glass-card glass-card--padded" style={{ borderLeft: '4px solid #10b981' }}>
            <h3 style={{ margin: '0 0 0.65rem', fontSize: '1rem', fontWeight: 600, color: 'var(--ink)' }}>
              🎯 Recommended Strategic Actions
            </h3>
            <ul style={{ margin: 0, paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
              {report.recommendations.map((rec, i) => (
                <li key={i} style={{ fontSize: '0.88rem', color: '#334155', lineHeight: '1.45' }}>
                  {rec}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </PageContainer>
  );
}
