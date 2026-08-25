import React from 'react';
import type { ChatMessage as ChatMessageType } from '../../types';
import AnalysisResponseRenderer from './AnalysisResponseRenderer';
import EvidencePanel from './EvidencePanel';
import { IconBrain, IconUser } from '../ui/Icons';

type ChatMessageProps = {
  message: ChatMessageType;
};

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  if (isUser) {
    return (
      <div
        style={{
          display: 'flex',
          justifyContent: 'flex-end',
          margin: '0.75rem 0',
        }}
      >
        <div
          style={{
            maxWidth: '75%',
            padding: '0.75rem 1rem',
            borderRadius: '16px 16px 4px 16px',
            backgroundColor: 'var(--primary)',
            color: '#ffffff',
            boxShadow: '0 2px 8px rgba(79, 70, 229, 0.2)',
          }}
        >
          <p style={{ margin: 0, fontSize: '0.92rem', lineHeight: '1.45', whiteSpace: 'pre-wrap' }}>
            {message.content}
          </p>
          <span
            style={{
              display: 'block',
              textAlign: 'right',
              fontSize: '0.68rem',
              opacity: 0.8,
              marginTop: '0.35rem',
            }}
          >
            {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>
      </div>
    );
  }

  // Assistant response
  return (
    <div
      style={{
        display: 'flex',
        gap: '0.75rem',
        alignItems: 'flex-start',
        margin: '1.25rem 0',
      }}
    >
      <div
        style={{
          width: '34px',
          height: '34px',
          borderRadius: '10px',
          backgroundColor: 'var(--primary-light)',
          color: 'var(--primary)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          marginTop: '2px',
        }}
        aria-hidden="true"
      >
        <IconBrain size={18} />
      </div>

      <div
        className="glass-card glass-card--padded"
        style={{
          flex: 1,
          maxWidth: '100%',
          boxShadow: '0 2px 12px rgba(15, 23, 42, 0.04)',
          border: '1px solid rgba(226, 232, 240, 0.8)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
          <span style={{ fontSize: '0.76rem', fontWeight: 600, color: 'var(--primary)' }}>
            Conversational Analyst
          </span>
          <span className="muted" style={{ fontSize: '0.7rem' }}>
            {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>

        {/* Structured analytical response */}
        <AnalysisResponseRenderer content={message.content} />

        {/* Provenance & Evidence Panel */}
        {message.evidence && message.evidence.length > 0 && (
          <EvidencePanel evidence={message.evidence} />
        )}
      </div>
    </div>
  );
}

