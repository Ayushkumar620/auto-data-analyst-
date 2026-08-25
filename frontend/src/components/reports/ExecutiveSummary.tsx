import React from 'react';

type ExecutiveSummaryProps = {
  summary: string;
};

export default function ExecutiveSummary({ summary }: ExecutiveSummaryProps) {
  if (!summary) return null;

  return (
    <div
      style={{
        padding: '1.25rem 1.5rem',
        borderRadius: '12px',
        backgroundColor: '#f8fafc',
        borderLeft: '4px solid var(--primary)',
        boxShadow: '0 1px 3px rgba(0,0,0,0.03)',
      }}
    >
      <h3 style={{ margin: '0 0 0.5rem', fontSize: '1rem', fontWeight: 600, color: 'var(--ink)' }}>
        Executive Summary
      </h3>
      <p
        style={{
          margin: 0,
          fontSize: '0.92rem',
          lineHeight: '1.6',
          color: '#334155',
        }}
      >
        {summary}
      </p>
    </div>
  );
}
