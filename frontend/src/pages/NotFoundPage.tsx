import React from 'react';
import { Link } from 'react-router-dom';
import { PageContainer } from '../components/layout/PageContainer';
import { IconInfo } from '../components/ui/Icons';

export default function NotFoundPage() {
  return (
    <PageContainer>
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '60vh',
          textAlign: 'center',
          padding: '2rem',
        }}
      >
        <div
          style={{
            width: '64px',
            height: '64px',
            borderRadius: '50%',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            color: 'var(--primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: '1.25rem',
          }}
        >
          <IconInfo size={36} />
        </div>

        <h1 style={{ fontSize: '2rem', fontWeight: 700, margin: '0 0 0.5rem', letterSpacing: '-0.02em' }}>
          404 — Page Not Found
        </h1>

        <p className="muted" style={{ maxWidth: '420px', margin: '0 0 1.5rem', fontSize: '0.94rem', lineHeight: '1.5' }}>
          The workspace route you requested does not exist or has been moved.
        </p>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <Link to="/overview" className="primary-btn" style={{ padding: '0.5rem 1.25rem', textDecoration: 'none' }}>
            Back to Overview →
          </Link>
          <Link to="/analyst" className="action-btn" style={{ padding: '0.5rem 1rem', textDecoration: 'none' }}>
            ⚡ Ask AI Analyst
          </Link>
        </div>
      </div>
    </PageContainer>
  );
}
