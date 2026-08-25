import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { PageContainer, PageHeader } from '../components/layout/PageContainer';
import { useDataset } from '../context/DatasetContext';
import { useNotification } from '../context/NotificationContext';
import { executeAnalysis } from '../services/analysisHistoryService';
import ErrorState from '../components/ui/ErrorState';
import { LoadingSpinner } from '../components/ui/LoadingState';
import { IconDatabase, IconAnalyst, IconCheck } from '../components/ui/Icons';

const EXAMPLE_PROMPTS = [
  'Analyze my sales data and find top revenue drivers',
  'Why did profit margin decline last quarter?',
  'Find unusual patterns in transactions',
  'Compare category performance and return key findings',
  'Identify anomalies and correlate them with regional factors',
  'Summarize data distribution and provide executive insights',
];

export default function AnalystPage() {
  const { profile, fileName } = useDataset();
  const { notify } = useNotification();
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const datasetName = fileName || profile?.dataset_name;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError('');

    try {
      // Execute multi-agent analysis with real API
      const record = await executeAnalysis(
        query.trim(),
        profile?.preview,
        'analyst_session',
        datasetName,
      );
      notify('Analysis completed successfully!', 'success');
      navigate(`/analyses/${record.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed to execute');
    } finally {
      setLoading(false);
    }
  };

  const handlePromptClick = (prompt: string) => {
    setQuery(prompt);
  };

  return (
    <PageContainer className="analyst-page">
      <div className="analyst-hero">
        <PageHeader
          eyebrow="AI Analyst"
          title="Ask your data anything"
          subtitle="Natural language commands powered by multi-agent AI reasoning."
        />

        {/* Active Dataset Context Badge */}
        {profile ? (
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              backgroundColor: 'rgba(99, 102, 241, 0.08)',
              border: '1px solid rgba(99, 102, 241, 0.25)',
              borderRadius: '12px',
              padding: '0.75rem 1rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <IconDatabase size={18} color="var(--primary)" aria-hidden />
              <div>
                <p style={{ margin: 0, fontSize: '0.88rem', fontWeight: 600, color: 'var(--ink)' }}>
                  Active Context: {datasetName}
                </p>
                <p className="muted" style={{ margin: '0.1rem 0 0', fontSize: '0.78rem' }}>
                  {profile.rows.toLocaleString()} rows · {profile.columns} columns loaded
                </p>
              </div>
            </div>
            <Link to="/datasets" className="action-btn" style={{ padding: '0.3rem 0.65rem', fontSize: '0.78rem', textDecoration: 'none' }}>
              Switch Dataset
            </Link>
          </div>
        ) : (
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              backgroundColor: '#fffbeb',
              border: '1px solid #fde68a',
              borderRadius: '12px',
              padding: '0.75rem 1rem',
            }}
          >
            <p style={{ margin: 0, fontSize: '0.86rem', color: '#92400e' }}>
              No dataset selected. The AI will analyze default sample metrics.
            </p>
            <Link to="/datasets" className="action-btn" style={{ padding: '0.3rem 0.65rem', fontSize: '0.78rem', textDecoration: 'none' }}>
              Select a Dataset
            </Link>
          </div>
        )}

        {error && <ErrorState message={error} />}

        {loading ? (
          <div className="glass-card glass-card--padded">
            <LoadingSpinner label="Autonomous agents decomposing intent and analyzing data…" size={36} />
          </div>
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="analyst-input-wrap">
              <textarea
                className="analyst-textarea"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={
                  profile
                    ? `e.g. Analyze ${datasetName} and identify key growth drivers and correlations...`
                    : 'e.g. Analyze sales performance and find what is driving revenue growth...'
                }
                rows={3}
                aria-label="Analysis command"
              />
              <button
                className="primary-btn analyst-submit-btn"
                type="submit"
                disabled={!query.trim() || loading}
              >
                Analyze →
              </button>
            </div>
          </form>
        )}

        <div>
          <p className="analyst-examples-label">Try an example command:</p>
          <div className="analyst-chips">
            {EXAMPLE_PROMPTS.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="analyst-chip"
                onClick={() => handlePromptClick(prompt)}
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="analyst-info-grid">
        <div className="analyst-info-card">
          <h3>Multi-Agent Reasoning</h3>
          <p>
            Intent detection, dynamic planning, and execution DAGs coordinated across specialized AI agents.
          </p>
        </div>
        <div className="analyst-info-card">
          <h3>Evidence-Backed Insights</h3>
          <p>
            Every finding is bound to verifiable statistical computations — never hallucinated observations.
          </p>
        </div>
        <div className="analyst-info-card">
          <h3>Autonomous Context</h3>
          <p>
            The selected dataset is automatically formatted, checked for quality, and passed to analytical routines.
          </p>
        </div>
      </div>
    </PageContainer>
  );
}
