import React from 'react';
import type { EvidenceItem } from '../../types';

type ReportEvidenceProps = {
  evidence: EvidenceItem[];
};

export default function ReportEvidence({ evidence }: ReportEvidenceProps) {
  if (!evidence || evidence.length === 0) return null;

  return (
    <div>
      <h3 className="section-title" style={{ margin: '0 0 0.85rem' }}>
        Verifiable Evidence & Provenance
      </h3>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.85rem' }}>
        {evidence.map((ev, idx) => (
          <div
            key={idx}
            className="glass-card glass-card--padded"
            style={{
              borderLeft: '3px solid #6366f1',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.35rem',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span
                style={{
                  fontSize: '0.7rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  padding: '0.12rem 0.4rem',
                  borderRadius: '4px',
                  backgroundColor: '#e0e7ff',
                  color: '#4338ca',
                }}
              >
                {ev.claim_type || 'FACT'}
              </span>

              {ev.confidence !== undefined && (
                <span style={{ fontSize: '0.74rem', fontWeight: 600, color: '#059669' }}>
                  {Math.round(ev.confidence * 100)}% Verified
                </span>
              )}
            </div>

            <p style={{ margin: '0.2rem 0 0', fontSize: '0.82rem', color: 'var(--ink)', fontWeight: 500 }}>
              {ev.claim || (ev as any).statement || 'Verified finding'}
            </p>

            {ev.columns && ev.columns.length > 0 && (
              <span className="muted" style={{ fontSize: '0.74rem', marginTop: '0.2rem' }}>
                Features: {ev.columns.join(', ')}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
