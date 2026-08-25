import React, { useEffect, useState, useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { PageContainer, PageHeader } from '../components/layout/PageContainer';
import EmptyState from '../components/ui/EmptyState';
import { listAnalyses } from '../services/analysisHistoryService';
import type { AnalysisRecord } from '../types';
import { IconBarChart, IconAnalyst, IconCheck } from '../components/ui/Icons';

export default function AnalysesPage() {
  const [analyses, setAnalyses] = useState<AnalysisRecord[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    setAnalyses(listAnalyses());
  }, []);

  const filteredAnalyses = useMemo(() => {
    if (!searchQuery.trim()) return analyses;
    const q = searchQuery.toLowerCase();
    return analyses.filter(
      (a) =>
        a.command.toLowerCase().includes(q) ||
        (a.user_intent && a.user_intent.toLowerCase().includes(q)) ||
        (a.dataset_name && a.dataset_name.toLowerCase().includes(q)),
    );
  }, [analyses, searchQuery]);

  return (
    <PageContainer>
      <PageHeader
        eyebrow="Intelligence History"
        title="Analyses"
        subtitle="Review past multi-agent analyses, evidence chains, and execution graphs."
        actions={
          <div className="page-header-action-row">
            <Link to="/analyst" className="primary-btn" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <IconAnalyst size={16} aria-hidden />
              New Analysis
            </Link>
          </div>
        }
      />

      {/* Search Filter */}
      {analyses.length > 0 && (
        <div>
          <input
            type="text"
            placeholder="Search analyses by command or dataset..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="horizon-input"
            style={{ width: '320px', padding: '0.5rem 0.85rem' }}
            aria-label="Search analyses"
          />
        </div>
      )}

      {/* Content */}
      {analyses.length === 0 ? (
        <EmptyState
          icon={<IconBarChart size={48} />}
          title="No analyses yet"
          description="Ask questions about your data in the AI Analyst to generate automated findings and evidence chains."
          action={
            <Link to="/analyst" className="primary-btn">
              Start your first analysis
            </Link>
          }
        />
      ) : filteredAnalyses.length === 0 ? (
        <EmptyState
          icon={<IconBarChart size={40} />}
          title="No matching analyses"
          description={`No recorded analyses match "${searchQuery}".`}
          action={
            <button type="button" className="action-btn" onClick={() => setSearchQuery('')}>
              Clear Search
            </button>
          }
        />
      ) : (
        <div style={{ display: 'grid', gap: '1rem' }}>
          {filteredAnalyses.map((item) => (
            <div
              key={item.id}
              className="glass-card glass-card--padded"
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                flexWrap: 'wrap',
                gap: '1rem',
                cursor: 'pointer',
              }}
              onClick={() => navigate(`/analyses/${item.id}`)}
            >
              <div style={{ flex: 1, minWidth: '260px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.35rem' }}>
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
                  {item.dataset_name && (
                    <span className="muted" style={{ fontSize: '0.78rem' }}>
                      Dataset: <strong>{item.dataset_name}</strong>
                    </span>
                  )}
                  {item.duration_ms && (
                    <span className="muted" style={{ fontSize: '0.78rem' }}>
                      · {Math.round(item.duration_ms)} ms
                    </span>
                  )}
                </div>

                <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600, color: 'var(--ink)' }}>
                  "{item.command}"
                </h3>

                {item.user_intent && (
                  <p className="muted" style={{ margin: '0.25rem 0 0', fontSize: '0.84rem' }}>
                    Intent: {item.user_intent}
                  </p>
                )}
              </div>

              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                <span
                  style={{
                    fontSize: '0.8rem',
                    color: 'var(--muted)',
                    backgroundColor: '#f8fafc',
                    padding: '0.3rem 0.6rem',
                    borderRadius: '6px',
                    border: '1px solid #e2e8f0',
                  }}
                >
                  {item.evidence?.length || 0} Evidence Items
                </span>
                <Link
                  to={`/analyses/${item.id}`}
                  className="primary-btn"
                  style={{ padding: '0.4rem 0.8rem', fontSize: '0.82rem', textDecoration: 'none' }}
                  onClick={(e) => e.stopPropagation()}
                >
                  View Details →
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </PageContainer>
  );
}

