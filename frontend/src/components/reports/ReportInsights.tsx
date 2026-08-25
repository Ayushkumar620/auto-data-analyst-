import React from 'react';
import type { ReportInsightItem } from '../../types';

type ReportInsightsProps = {
  insights: ReportInsightItem[];
};

export default function ReportInsights({ insights }: ReportInsightsProps) {
  if (!insights || insights.length === 0) return null;

  return (
    <div>
      <h3 className="section-title" style={{ margin: '0 0 0.85rem' }}>
        Statistical Insights & Findings
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
        {insights.map((item, idx) => (
          <div key={idx} className="glass-card glass-card--padded" style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
            <h4 style={{ margin: 0, fontSize: '0.94rem', fontWeight: 600, color: 'var(--ink)' }}>
              💡 {item.title}
            </h4>
            <p style={{ margin: 0, fontSize: '0.84rem', color: '#475569', lineHeight: '1.45' }}>
              {item.narrative}
            </p>

            {(item.metric || item.evidence) && (
              <div
                style={{
                  marginTop: '0.4rem',
                  paddingTop: '0.4rem',
                  borderTop: '1px solid #f1f5f9',
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: '0.76rem',
                  color: 'var(--muted)',
                  flexWrap: 'wrap',
                  gap: '0.4rem',
                }}
              >
                {item.metric && <span><strong>Metric:</strong> {item.metric}</span>}
                {item.evidence && <span><strong>Source:</strong> {item.evidence}</span>}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

