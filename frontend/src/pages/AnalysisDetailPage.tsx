import React, { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { PageContainer } from '../components/layout/PageContainer';
import ErrorState from '../components/ui/ErrorState';
import { getAnalysisById } from '../services/analysisHistoryService';
import type { AnalysisRecord } from '../types';
import { IconBarChart, IconChevronRight, IconCheck, IconAnalyst } from '../components/ui/Icons';

export default function AnalysisDetailPage() {
  const { analysisId } = useParams<{ analysisId: string }>();
  const [analysis, setAnalysis] = useState<AnalysisRecord | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!analysisId) return;
    const found = getAnalysisById(analysisId);
    setAnalysis(found);
  }, [analysisId]);

  if (!analysis) {
    return (
      <PageContainer>
        <ErrorState
          message="Analysis record not found."
          onRetry={() => navigate('/analyses')}
        />
      </PageContainer>
    );
  }

  return (
    <PageContainer>
      {/* Breadcrumb Header */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.5rem', fontSize: '0.84rem' }}>
          <Link to="/analyses" className="muted" style={{ textDecoration: 'none' }}>
            Analyses
          </Link>
          <IconChevronRight size={14} className="muted" aria-hidden />
          <span style={{ fontWeight: 600, color: 'var(--ink)' }}>Analysis Detail</span>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem', flexWrap: 'wrap' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <h1 className="page-title" style={{ margin: 0 }}>"{analysis.command}"</h1>
              <span
                className="sidebar-soon-badge"
                style={{
                  backgroundColor: '#ecfdf5',
                  color: '#059669',
                  borderColor: '#a7f3d0',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.2rem',
                }}
              >
                <IconCheck size={12} aria-hidden /> Completed
              </span>
            </div>
            <p className="page-subtitle" style={{ marginTop: '0.3rem' }}>
              {analysis.dataset_name ? `Dataset: ${analysis.dataset_name} · ` : ''}
              {analysis.duration_ms ? `Executed in ${Math.round(analysis.duration_ms)}ms · ` : ''}
              {new Date(analysis.created_at).toLocaleString()}
            </p>
          </div>

          <Link to="/analyst" className="primary-btn" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <IconAnalyst size={16} aria-hidden />
            New Query
          </Link>
        </div>
      </div>

      {/* Main explanation card */}
      {analysis.final_explanation && (
        <div className="glass-card glass-card--padded" style={{ border: '1px solid rgba(99, 102, 241, 0.25)', backgroundColor: 'rgba(255, 255, 255, 0.95)' }}>
          <h2 className="section-title" style={{ margin: '0 0 0.5rem', color: 'var(--primary)' }}>
            Executive Synthesis & Findings
          </h2>
          <p style={{ margin: 0, fontSize: '0.96rem', lineHeight: 1.65, color: 'var(--ink)', whiteSpace: 'pre-wrap' }}>
            {analysis.final_explanation}
          </p>
        </div>
      )}

      {/* Operations / Intent summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
        {analysis.user_intent && (
          <div className="glass-card glass-card--padded">
            <h3 style={{ margin: '0 0 0.4rem', fontSize: '0.9rem', fontWeight: 600 }} className="muted">
              Detected User Intent
            </h3>
            <p style={{ margin: 0, fontSize: '0.92rem', fontWeight: 500, color: 'var(--ink)' }}>
              {analysis.user_intent}
            </p>
          </div>
        )}

        {analysis.required_operations && analysis.required_operations.length > 0 && (
          <div className="glass-card glass-card--padded">
            <h3 style={{ margin: '0 0 0.4rem', fontSize: '0.9rem', fontWeight: 600 }} className="muted">
              Executed Pipeline Operations
            </h3>
            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
              {analysis.required_operations.map((op, idx) => (
                <span
                  key={idx}
                  style={{
                    padding: '0.2rem 0.6rem',
                    borderRadius: '6px',
                    backgroundColor: '#eef2ff',
                    color: '#3730a3',
                    fontSize: '0.78rem',
                    fontWeight: 600,
                  }}
                >
                  {op}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Evidence Section */}
      <div className="glass-card glass-card--padded">
        <h2 className="section-title" style={{ margin: '0 0 1rem' }}>
          Evidence Chain ({analysis.evidence?.length || 0} findings)
        </h2>

        {!analysis.evidence || analysis.evidence.length === 0 ? (
          <p className="muted">No individual evidence items recorded for this execution.</p>
        ) : (
          <div style={{ display: 'grid', gap: '0.75rem' }}>
            {analysis.evidence.map((ev, idx) => (
              <div
                key={idx}
                style={{
                  padding: '0.85rem 1rem',
                  borderRadius: '10px',
                  backgroundColor: '#f8fafc',
                  border: '1px solid #e2e8f0',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                  <span
                    style={{
                      fontSize: '0.74rem',
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      padding: '0.15rem 0.5rem',
                      borderRadius: '4px',
                      backgroundColor: 'rgba(99, 102, 241, 0.1)',
                      color: 'var(--primary)',
                    }}
                  >
                    {String(ev.claim_type || 'FACT')}
                  </span>
                  {ev.confidence !== undefined && (
                    <span className="muted" style={{ fontSize: '0.78rem' }}>
                      Confidence: {Math.round(Number(ev.confidence) * 100)}%
                    </span>
                  )}
                </div>
                <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--ink-secondary)' }}>
                  {String(ev.claim || ev.statement || JSON.stringify(ev))}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </PageContainer>
  );
}
