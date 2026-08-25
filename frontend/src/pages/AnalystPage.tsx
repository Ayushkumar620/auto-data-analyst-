import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { PageContainer, PageHeader } from '../components/layout/PageContainer';

const EXAMPLE_PROMPTS = [
  'Analyze my sales data',
  'Why did revenue decline last quarter?',
  'Forecast next quarter performance',
  'Find unusual patterns in transactions',
  'Compare regional performance',
  'Build a churn prediction model',
];

export default function AnalystPage() {
  const [query, setQuery] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Navigate to Command Studio where the actual execution happens
    if (query.trim()) {
      navigate('/chat');
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

        <form onSubmit={handleSubmit}>
          <div className="analyst-input-wrap">
            <textarea
              className="analyst-textarea"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. Analyze my sales data and find what's driving revenue growth..."
              rows={3}
              aria-label="Analysis command"
            />
            <button
              className="primary-btn analyst-submit-btn"
              type="submit"
              disabled={!query.trim()}
            >
              Analyse →
            </button>
          </div>
        </form>

        <div>
          <p className="analyst-examples-label">Try an example:</p>
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
            Intent detection, planning, and execution by specialised AI agents working
            in concert to decompose complex analytical questions.
          </p>
        </div>
        <div className="analyst-info-card">
          <h3>Evidence-Backed Insights</h3>
          <p>
            Every finding is attached to real statistical evidence — no fabricated
            results or hallucinated metrics.
          </p>
        </div>
        <div className="analyst-info-card">
          <h3>Autonomous Model Selection</h3>
          <p>
            The system selects the best ML model (traditional, ANN, or CNN) based on
            your data characteristics and problem type.
          </p>
        </div>
      </div>
    </PageContainer>
  );
}
