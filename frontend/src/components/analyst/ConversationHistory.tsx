import React from 'react';
import type { AnalystSession } from '../../types';
import { IconBrain, IconFolder } from '../ui/Icons';

type ConversationHistoryProps = {
  sessions: AnalystSession[];
  currentSessionId: string;
  onSelectSession: (sessionId: string) => void;
  onNewSession: () => void;
};

export default function ConversationHistory({
  sessions,
  currentSessionId,
  onSelectSession,
  onNewSession,
}: ConversationHistoryProps) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.65rem',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h4 style={{ margin: 0, fontSize: '0.86rem', fontWeight: 600, color: 'var(--ink)' }}>
          Conversations
        </h4>
        <button
          type="button"
          onClick={onNewSession}
          className="action-btn"
          style={{ padding: '0.25rem 0.55rem', fontSize: '0.74rem' }}
        >
          + New
        </button>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '320px', overflowY: 'auto' }}>
        {sessions.length === 0 ? (
          <p className="muted" style={{ margin: '0.5rem 0', fontSize: '0.78rem' }}>
            No previous sessions recorded.
          </p>
        ) : (
          sessions.map((s) => {
            const isSelected = s.session_id === currentSessionId;
            const firstMsg = s.messages.find((m) => m.role === 'user')?.content || 'Analytical Session';

            return (
              <button
                key={s.session_id}
                type="button"
                onClick={() => onSelectSession(s.session_id)}
                style={{
                  textAlign: 'left',
                  padding: '0.5rem 0.65rem',
                  borderRadius: '8px',
                  border: isSelected ? '1px solid var(--primary)' : '1px solid #e2e8f0',
                  backgroundColor: isSelected ? 'rgba(99, 102, 241, 0.08)' : '#ffffff',
                  cursor: 'pointer',
                  transition: 'all 150ms ease',
                }}
              >
                <p
                  style={{
                    margin: 0,
                    fontSize: '0.8rem',
                    fontWeight: isSelected ? 600 : 500,
                    color: isSelected ? 'var(--primary)' : 'var(--ink)',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                  title={firstMsg}
                >
                  {firstMsg}
                </p>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.2rem' }}>
                  <span className="muted" style={{ fontSize: '0.68rem' }}>
                    {s.dataset_name || 'Dataset'}
                  </span>
                  <span className="muted" style={{ fontSize: '0.68rem' }}>
                    {s.messages.length} msgs
                  </span>
                </div>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}

