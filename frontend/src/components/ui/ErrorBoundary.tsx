import React, { Component, ErrorInfo, ReactNode } from 'react';
import { IconAlertTriangle } from './Icons';

type Props = {
  children: ReactNode;
};

type State = {
  hasError: boolean;
  error: Error | null;
};

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log unexpected runtime errors without leaking sensitive data
    console.error('ErrorBoundary caught an error:', error.message, errorInfo.componentStack);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.href = '/overview';
  };

  handleReload = () => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem',
            background: 'var(--bg-gradient, #0f172a)',
            fontFamily: 'var(--font-body, sans-serif)',
          }}
        >
          <div
            className="glass-card glass-card--padded"
            style={{
              maxWidth: '480px',
              width: '100%',
              textAlign: 'center',
              backgroundColor: '#ffffff',
              borderRadius: '16px',
              padding: '2rem',
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
            }}
          >
            <div
              style={{
                width: '56px',
                height: '56px',
                borderRadius: '50%',
                backgroundColor: '#fef2f2',
                color: '#dc2626',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '1rem',
              }}
            >
              <IconAlertTriangle size={32} />
            </div>

            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, margin: '0 0 0.5rem', color: 'var(--ink, #0f172a)' }}>
              Something went wrong
            </h2>

            <p style={{ fontSize: '0.88rem', color: 'var(--muted, #64748b)', margin: '0 0 1.5rem', lineHeight: '1.5' }}>
              An unexpected application error occurred. You can attempt to refresh the page or return to the workspace overview.
            </p>

            <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center' }}>
              <button
                type="button"
                onClick={this.handleReload}
                className="action-btn"
                style={{ padding: '0.5rem 1rem', fontSize: '0.86rem' }}
              >
                ↻ Try Again
              </button>
              <button
                type="button"
                onClick={this.handleReset}
                className="primary-btn"
                style={{ padding: '0.5rem 1.25rem', fontSize: '0.86rem' }}
              >
                Return to Overview →
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
