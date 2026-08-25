import React, { useState } from 'react';
import type { EvidenceItem } from '../../types';
import { IconCheck, IconChevronRight } from '../ui/Icons';

type EvidencePanelProps = {
  evidence: EvidenceItem[];
};

export default function EvidencePanel({ evidence }: EvidencePanelProps) {
  const [open, setOpen] = useState(false);

  if (!evidence || evidence.length === 0) return null;

  return (
    <div
      style={{
        marginTop: '0.85rem',
        borderRadius: '10px',
        border: '1px solid rgba(99, 102, 241, 0.2)',
        backgroundColor: 'rgba(248, 250, 252, 0.9)',
        overflow: 'hidden',
      }}
    >
      <button
        type="button"
        onClick={() => setOpen(!open)}
        style={{
          width: '100%',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '0.5rem 0.75rem',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: 'var(--ink)',
          fontSize: '0.82rem',
          fontWeight: 600,
        }}
        aria-expanded={open}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <IconCheck size={14} color="var(--primary)" aria-hidden />
          Verifiable Evidence & Provenance ({evidence.length} {evidence.length === 1 ? 'item' : 'items'})
        </span>
        <span style={{ fontSize: '0.75rem', color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '0.2rem' }}>
          {open ? 'Hide Proof ▲' : 'View Proof ▼'}
        </span>
      </button>

      {open && (
        <div style={{ padding: '0.65rem 0.75rem', borderTop: '1px solid #e2e8f0', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
          {evidence.map((ev, idx) => {
            const claimText = ev.claim || (typeof ev.result === 'string' ? ev.result : JSON.stringify(ev.result || ev));
            const claimType = ev.claim_type || 'FACT';
            const confidence = ev.confidence !== undefined ? Math.round(ev.confidence * 100) : 100;
            const columns = ev.columns && ev.columns.length > 0 ? ev.columns.join(', ') : ev.dataset_name || 'dataset';

            return (
              <div
                key={idx}
                style={{
                  padding: '0.55rem 0.75rem',
                  borderRadius: '8px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #e2e8f0',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                  <span
                    style={{
                      fontSize: '0.68rem',
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      padding: '0.1rem 0.45rem',
                      borderRadius: '4px',
                      backgroundColor: 'rgba(99, 102, 241, 0.1)',
                      color: 'var(--primary)',
                    }}
                  >
                    {claimType}
                  </span>
                  <span className="muted" style={{ fontSize: '0.74rem' }}>
                    Confidence: {confidence}% · Source: <code>{columns}</code>
                  </span>
                </div>
                <p style={{ margin: 0, fontSize: '0.84rem', color: 'var(--ink)' }}>
                  {claimText}
                </p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

