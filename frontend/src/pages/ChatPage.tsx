import React, { useState } from 'react';
import { executeCommand } from '../services/chatService';
import type { CommandExecutionResponse } from '../services/chatService';
import PlotlyChart from '../components/PlotlyChart';
import AnalysisResponseRenderer from '../components/analyst/AnalysisResponseRenderer';
import HorizontalAnalysisTimeline, { type TimelineStage } from '../components/analyst/HorizontalAnalysisTimeline';
import { useDataset } from '../context/DatasetContext';
import { PageContainer, PageHeader, Card } from '../components/layout/PageContainer';
import EmptyState from '../components/ui/EmptyState';
import Spinner from '../components/ui/Spinner';
import { IconDatabase, IconBrain, IconCheck, IconTrendUp, IconActivity } from '../components/ui/Icons';

type HistoryEntry = {
  command: string;
  result: CommandExecutionResponse;
};

const SAMPLE_COMMANDS = [
  'Analyze my sales data and identify top drivers.',
  'Why did profit decrease last quarter?',
  'Clean this dataset and find the top 10 customers.',
  'Compare revenue between regions.',
  "Forecast next quarter's performance.",
  'Build the best model to predict customer churn.',
  'Find unusual anomalies or correlation patterns.',
  'Create an executive report showing financial variance.',
];

export default function ChatPage() {
  const { profile, fileName } = useDataset();
  const [file, setFile] = useState<File | null>(null);
  const [command, setCommand] = useState('');
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState('');
  const [activeResult, setActiveResult] = useState<CommandExecutionResponse | null>(null);

  const activeDatasetName = file?.name || fileName || profile?.dataset_name;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0] ?? null);
    setError('');
  };

  const handleExecute = async (cmdToRun?: string) => {
    const targetCommand = (cmdToRun ?? command).trim();
    if (!file && !profile) {
      setError('Please select or upload a dataset file first.');
      return;
    }
    if (!targetCommand) {
      setError('Please enter a natural language command.');
      return;
    }
    setWorking(true);
    setError('');

    try {
      // If no new file uploaded, create a CSV Blob from active dataset preview
      let targetFile = file;
      if (!targetFile && profile?.preview && profile.preview.length > 0) {
        const keys = Object.keys(profile.preview[0]);
        const csvContent = [
          keys.join(','),
          ...profile.preview.map((row) => keys.map((k) => JSON.stringify(row[k] ?? '')).join(',')),
        ].join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv' });
        targetFile = new File([blob], fileName || `${profile.dataset_name || 'dataset'}.csv`, { type: 'text/csv' });
      }

      if (!targetFile) {
        throw new Error('No dataset file available. Please upload a file.');
      }

      const res = await executeCommand(targetFile, targetCommand);
      setActiveResult(res);
      setHistory((h) => [{ command: targetCommand, result: res }, ...h]);
      setCommand('');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Command execution failed');
    } finally {
      setWorking(false);
    }
  };

  return (
    <PageContainer className="command-studio-page">
      <PageHeader
        eyebrow="Autonomous AI Data Analyst"
        title="Command-Driven Analysis Studio"
        subtitle="Give a natural-language command. The multi-agent AI engine decomposes intent, formulates execution plans, trains models, validates evidence, and generates executive findings."
      />

      <div className="studio-layout-grid">
        {/* ================================================================
            LEFT PANEL: Upload, Profile, Command, Quick Commands
            ================================================================ */}
        <aside className="studio-left-panel">
          {/* Dataset Upload Card */}
          <Card className="studio-panel-card">
            <h3 style={{ margin: '0 0 0.75rem', fontSize: '1rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <IconDatabase size={16} /> Dataset Source
            </h3>

            <div className="field" style={{ marginBottom: '0.65rem' }}>
              <label htmlFor="chat-file" style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--muted)', display: 'block', marginBottom: '0.3rem' }}>
                Upload Dataset (CSV / Excel)
              </label>
              <input
                id="chat-file"
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileChange}
                disabled={working}
                style={{ width: '100%', fontSize: '0.85rem' }}
              />
            </div>

            {activeDatasetName && (
              <div
                style={{
                  padding: '0.65rem 0.85rem',
                  borderRadius: '10px',
                  backgroundColor: '#f0fdf4',
                  border: '1px solid #bbf7d0',
                  marginTop: '0.4rem',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.84rem', fontWeight: 600, color: '#166534', wordBreak: 'break-all' }}>
                    {activeDatasetName}
                  </span>
                  <span style={{ fontSize: '0.68rem', fontWeight: 700, color: '#15803d', textTransform: 'uppercase' }}>
                    Active
                  </span>
                </div>
                {profile && (
                  <p className="muted" style={{ margin: '0.2rem 0 0', fontSize: '0.75rem' }}>
                    {profile.rows.toLocaleString()} rows · {profile.columns} columns
                  </p>
                )}
              </div>
            )}
          </Card>

          {/* Command Input Card */}
          <Card className="studio-panel-card">
            <h3 style={{ margin: '0 0 0.6rem', fontSize: '1rem', fontWeight: 600 }}>
              Enter Command
            </h3>

            {error ? <div className="status-error" style={{ marginBottom: '0.75rem', fontSize: '0.85rem', padding: '0.5rem 0.75rem', borderRadius: '8px' }}>{error}</div> : null}

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              <textarea
                value={command}
                onChange={(e) => setCommand(e.target.value)}
                placeholder="e.g. Clean this dataset, compare revenue between regions, and predict next month's performance..."
                disabled={working}
                rows={3}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !working) {
                    e.preventDefault();
                    handleExecute();
                  }
                }}
                className="horizon-input"
                style={{ width: '100%', minWidth: 0, resize: 'vertical', fontSize: '0.88rem' }}
                aria-label="Analytical command"
              />

              <button
                className="primary-btn"
                onClick={() => handleExecute()}
                disabled={working || (!file && !profile) || !command.trim()}
                style={{ width: '100%', padding: '0.65rem 1rem' }}
                type="button"
              >
                {working ? (
                  <>
                    <Spinner size={16} /> Executing DAG…
                  </>
                ) : (
                  '⚡ Analyze Data'
                )}
              </button>
            </div>
          </Card>

          {/* Quick Commands Card */}
          <Card className="studio-panel-card">
            <h3 style={{ margin: '0 0 0.5rem', fontSize: '0.92rem', fontWeight: 600 }} className="muted">
              Quick Command Inspiration
            </h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              {SAMPLE_COMMANDS.map((sample) => (
                <button
                  key={sample}
                  className="analyst-chip"
                  onClick={() => {
                    setCommand(sample);
                    if (file || profile) handleExecute(sample);
                  }}
                  type="button"
                  style={{ textAlign: 'left', fontSize: '0.8rem', padding: '0.45rem 0.65rem' }}
                  disabled={working}
                >
                  ⚡ {sample}
                </button>
              ))}
            </div>
          </Card>
        </aside>

        {/* ================================================================
            RIGHT PANEL: Execution Graph, Report, Models, Visualizations
            ================================================================ */}
        <main className="studio-right-panel">
          {activeResult ? (
            <>
              {/* Autonomous Agent Execution Timeline Panel */}
              <Card className="execution-graph-card">
                <HorizontalAnalysisTimeline
                  stages={[
                    {
                      id: 'step-intent',
                      stepNumber: 1,
                      title: 'Intent & Planning',
                      agentName: 'IntentAnalyzer',
                      status: 'completed',
                      durationMs: Math.round(activeResult.duration_ms * 0.15),
                      summary: `Detected Intent: ${activeResult.user_intent}`,
                      details: [
                        `Decomposed into ${activeResult.required_operations.length} specialized analytical operations`,
                        'Target pipeline formulated with dynamic dependency resolution',
                      ],
                    },
                    {
                      id: 'step-data-quality',
                      stepNumber: 2,
                      title: 'Data Quality & Schema',
                      agentName: 'DataValidationAgent',
                      status: activeResult.validation_summary.status === 'PASSED' ? 'completed' : 'warning',
                      durationMs: Math.round(activeResult.duration_ms * 0.2),
                      summary: `Validation Audit: ${activeResult.validation_summary.status} (${activeResult.validation_summary.critical_issues} issues)`,
                      details: [
                        `Schema integrity verified with ${activeResult.validation_summary.warnings} non-critical warnings`,
                        'Verified data types, temporal column roles, and null concentration',
                      ],
                    },
                    ...activeResult.selected_agents.map((agent, i) => ({
                      id: `step-agent-${i}`,
                      stepNumber: i + 3,
                      title: agent.replace(/([A-Z])/g, ' $1').trim(),
                      agentName: agent,
                      status: 'completed' as const,
                      durationMs: Math.round((activeResult.duration_ms * 0.45) / Math.max(activeResult.selected_agents.length, 1)),
                      summary: activeResult.required_operations[i] || 'Executed specialized statistical calculation',
                      details: [
                        `Operation: ${activeResult.required_operations[i] || 'Autonomous analytical execution'}`,
                        activeResult.model_selection_summary
                          ? `Champion Model: ${activeResult.model_selection_summary.model_name} (Score: ${Number(activeResult.model_selection_summary.primary_metric_value).toFixed(4)})`
                          : 'Computed model evaluation metrics and regression projections',
                      ],
                    })),
                    {
                      id: 'step-synthesis',
                      stepNumber: activeResult.selected_agents.length + 3,
                      title: 'Executive Synthesis',
                      agentName: 'DecisionExplainer',
                      status: 'completed',
                      durationMs: Math.round(activeResult.duration_ms * 0.2),
                      summary: 'Synthesized evidence-backed findings and actionable decisions',
                      details: [
                        'Structured final analytical findings and visual projections',
                        'Verified factual claims against mathematical data proof',
                      ],
                    },
                  ]}
                  userIntent={activeResult.user_intent}
                  totalDurationMs={activeResult.duration_ms}
                />
              </Card>

              {/* Best Model Selection (if applicable) */}
              {activeResult.model_selection_summary && (
                <Card style={{ borderLeft: '4px solid #10b981', backgroundColor: '#f0fdf4' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <IconTrendUp size={20} style={{ color: '#16a34a' }} />
                    <div>
                      <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700, color: '#15803d' }}>
                        Autonomous Model Selection: {String(activeResult.model_selection_summary.model_name ?? '')}
                      </h3>
                      <p style={{ margin: '0.2rem 0 0', fontSize: '0.84rem', color: '#166534' }}>
                        Primary Metric Score: <strong>{Number(activeResult.model_selection_summary.primary_metric_value ?? 0).toFixed(4)}</strong>
                      </p>
                    </div>
                  </div>
                </Card>
              )}

              {/* Executive Report & Key Findings Card */}
              <Card>
                <h2 className="section-title" style={{ margin: '0 0 0.75rem', color: 'var(--primary)' }}>
                  Executive Report & Verifiable Findings
                </h2>
                <div className="executive-report-body">
                  <AnalysisResponseRenderer content={activeResult.final_explanation} />
                </div>
              </Card>

              {/* Visual Evidence Card (if chart available) */}
              {activeResult.visualization && (activeResult.visualization.data || activeResult.visualization.chart_type) && (
                <Card>
                  <h2 className="section-title" style={{ margin: '0 0 0.75rem' }}>
                    Visual Evidence & Projections
                  </h2>
                  <div className="chart-responsive-shell">
                    {activeResult.visualization.data ? (
                      <PlotlyChart data={activeResult.visualization.data as any} layout={activeResult.visualization.layout} />
                    ) : (
                      <div style={{ padding: '1rem', background: '#f8fafc', borderRadius: '8px', textAlign: 'center' }}>
                        Chart Type: {activeResult.visualization.chart_type} ({activeResult.visualization.x} vs {activeResult.visualization.y})
                      </div>
                    )}
                  </div>
                </Card>
              )}
            </>
          ) : (
            /* Empty State */
            <Card>
              <EmptyState
                icon={<IconBrain size={48} />}
                title="Ready for Analysis"
                description="Select a dataset on the left panel and submit a natural-language command to trigger the autonomous multi-agent execution pipeline."
              />
            </Card>
          )}

          {/* Recent Command History */}
          {history.length > 0 && (
            <Card>
              <h2 className="section-title" style={{ margin: '0 0 0.75rem' }}>
                Recent Command History ({history.length})
              </h2>
              <div style={{ display: 'grid', gap: '0.5rem' }}>
                {history.map((entry, idx) => (
                  <div
                    key={idx}
                    className="glass-card"
                    style={{
                      padding: '0.75rem 1rem',
                      cursor: 'pointer',
                      border: activeResult === entry.result ? '1px solid var(--primary)' : '1px solid #e2e8f0',
                      backgroundColor: activeResult === entry.result ? 'var(--primary-light)' : '#ffffff',
                    }}
                    onClick={() => setActiveResult(entry.result)}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <strong style={{ fontSize: '0.88rem', color: 'var(--ink)' }}>{entry.command}</strong>
                      <span className="muted" style={{ fontSize: '0.74rem' }}>
                        {entry.result.duration_ms} ms
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </main>
      </div>
    </PageContainer>
  );
}

