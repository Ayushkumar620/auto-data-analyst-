import React, { useState } from 'react';
import { executeCommand } from '../services/chatService';
import type { CommandExecutionResponse } from '../services/chatService';
import PlotlyChart from '../components/PlotlyChart';

type HistoryEntry = {
  command: string;
  result: CommandExecutionResponse;
};

const SAMPLE_COMMANDS = [
  'Analyze my sales data.',
  'Why did profit decrease last year?',
  'Clean this dataset and find the top 10 customers.',
  'Compare revenue between India and the US.',
  "Predict next month's sales.",
  'Build the best model to predict customer churn.',
  'Find unusual transactions and explain them.',
  'Create a report showing the financial performance.',
];

export default function ChatPage() {
  const [file, setFile] = useState<File | null>(null);
  const [command, setCommand] = useState('');
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState('');
  const [activeResult, setActiveResult] = useState<CommandExecutionResponse | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0] ?? null);
    setError('');
  };

  const handleExecute = async (cmdToRun?: string) => {
    const targetCommand = (cmdToRun ?? command).trim();
    if (!file) {
      setError('Please select a dataset file first.');
      return;
    }
    if (!targetCommand) {
      setError('Please enter a natural language command.');
      return;
    }
    setWorking(true);
    setError('');

    try {
      const res = await executeCommand(file, targetCommand);
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
    <div className="page-stack chat-page">
      <header>
        <p className="hero-eyebrow">Autonomous AI Data Analyst</p>
        <h1>Command-Driven Analysis Studio</h1>
        <p className="muted">
          Give a natural-language command describing your desired outcome. The multi-agent AI engine
          will automatically inspect data, compose required tools, train models, validate results, and explain the findings.
        </p>
      </header>

      <section className="card">
        <div className="field">
          <label htmlFor="chat-file">Dataset file (CSV / Excel)</label>
          <input
            id="chat-file"
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileChange}
            disabled={working}
          />
        </div>

        {error ? <div className="status-error">{error}</div> : null}

        <div className="chat-input-row">
          <input
            type="text"
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            placeholder="e.g. Clean this dataset, compare revenue between regions, and predict next month's sales..."
            disabled={working}
            onKeyDown={(e) => e.key === 'Enter' && !working && handleExecute()}
            style={{ flex: 1, minWidth: 0 }}
          />
          <button
            className="primary-btn"
            onClick={() => handleExecute()}
            disabled={working}
            style={{ width: 'auto', minWidth: '130px' }}
          >
            {working ? 'Executing DAG…' : 'Run Command'}
          </button>
        </div>

        <div style={{ marginTop: '12px' }}>
          <p className="muted" style={{ fontSize: '0.85rem', marginBottom: '6px' }}>
            Quick command inspiration (click to fill):
          </p>
          <div className="chip-row">
            {SAMPLE_COMMANDS.map((sample) => (
              <button
                key={sample}
                className="action-btn"
                onClick={() => {
                  setCommand(sample);
                  if (file) handleExecute(sample);
                }}
                type="button"
                style={{ fontSize: '0.82rem', padding: '4px 10px' }}
              >
                {sample}
              </button>
            ))}
          </div>
        </div>
      </section>

      {activeResult ? (
        <section className="card active-result-card">
          <h2>Execution Breakdown & Explanation</h2>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '16px' }}>
            <div style={{ padding: '10px', background: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <strong>🎯 Detected Intent:</strong>
              <div style={{ textTransform: 'capitalize', color: '#0284c7', fontWeight: 600 }}>{activeResult.user_intent}</div>
            </div>
            <div style={{ padding: '10px', background: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <strong>🛡️ Validation Audit:</strong>
              <div style={{ color: activeResult.validation_summary.status === 'PASSED' ? '#16a34a' : '#ea580c', fontWeight: 600 }}>
                {activeResult.validation_summary.status} ({activeResult.validation_summary.critical_issues} issues)
              </div>
            </div>
            <div style={{ padding: '10px', background: '#f8fafc', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <strong>⏱️ Execution Time:</strong>
              <div>{activeResult.duration_ms} ms</div>
            </div>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <strong>⚙️ Decomposed Required Operations:</strong>
            <ul style={{ paddingLeft: '20px', marginTop: '6px' }}>
              {activeResult.required_operations.map((op, i) => (
                <li key={i} style={{ color: '#475569', fontSize: '0.9rem' }}>{op}</li>
              ))}
            </ul>
          </div>

          <div style={{ marginBottom: '16px' }}>
            <strong>🤖 Autonomous Agents Deployed:</strong>
            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginTop: '6px' }}>
              {activeResult.selected_agents.map((agent, i) => (
                <span key={i} style={{ background: '#e0e7ff', color: '#3730a3', padding: '2px 8px', borderRadius: '4px', fontSize: '0.85rem' }}>
                  {agent}
                </span>
              ))}
            </div>
          </div>

          {activeResult.model_selection_summary ? (
            <div style={{ padding: '12px', background: '#f0fdf4', borderRadius: '6px', border: '1px solid #bbf7d0', marginBottom: '16px' }}>
              <strong>🏆 Best Model Selection:</strong>
              <div>
                <strong>Model:</strong> {String(activeResult.model_selection_summary.model_name ?? '')} |{' '}
                <strong>Score:</strong> {Number(activeResult.model_selection_summary.primary_metric_value ?? 0).toFixed(4)}
              </div>
            </div>
          ) : null}

          <div style={{ padding: '16px', background: '#f1f5f9', borderRadius: '8px', marginBottom: '16px' }}>
            <h3 style={{ marginTop: 0 }}>Analytical Finding & Explanation</h3>
            <p style={{ whiteSpace: 'pre-line', margin: 0 }}>{activeResult.final_explanation}</p>
          </div>

          {activeResult.visualization && (activeResult.visualization.data || activeResult.visualization.chart_type) ? (
            <div>
              <h3>Visual Evidence</h3>
              {activeResult.visualization.data ? (
                <PlotlyChart data={activeResult.visualization.data as any} layout={activeResult.visualization.layout} />
              ) : (
                <div style={{ padding: '10px', background: '#e2e8f0', borderRadius: '6px' }}>
                  Chart: {activeResult.visualization.chart_type} ({activeResult.visualization.x} vs {activeResult.visualization.y})
                </div>
              )}
            </div>
          ) : null}
        </section>
      ) : null}

      {history.length > 1 ? (
        <section className="card">
          <h2>Recent Command History</h2>
          <ul className="chat-history">
            {history.slice(1).map((entry, idx) => (
              <li key={idx} className="chat-bubble chat-user" style={{ cursor: 'pointer' }} onClick={() => setActiveResult(entry.result)}>
                <strong>Command:</strong> {entry.command}
                <div className="muted" style={{ fontSize: '0.85rem' }}>Intent: {entry.result.user_intent} | {entry.result.duration_ms} ms</div>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}
