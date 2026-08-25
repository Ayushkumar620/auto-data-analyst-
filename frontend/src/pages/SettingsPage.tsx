import React, { useState } from 'react';
import { PageContainer, PageHeader, Card } from '../components/layout/PageContainer';
import { useAuth } from '../auth/authContext';
import { useNotification } from '../context/NotificationContext';
import { useDataset } from '../context/DatasetContext';
import { getApiBaseUrl } from '../services/api';
import { IconSettings, IconUser, IconDatabase, IconActivity } from '../components/ui/Icons';

export default function SettingsPage() {
  const { user } = useAuth();
  const { notify } = useNotification();
  const { clearDataset } = useDataset();

  const [reducedMotion, setReducedMotion] = useState<boolean>(() => {
    return localStorage.getItem('prefers_reduced_motion') === 'true';
  });

  const handleToggleMotion = () => {
    setReducedMotion((prev) => {
      const next = !prev;
      localStorage.setItem('prefers_reduced_motion', String(next));
      notify(`Reduced motion ${next ? 'enabled' : 'disabled'}.`, 'info');
      return next;
    });
  };

  const handleClearCache = () => {
    clearDataset();
    localStorage.removeItem('analyst_recent_sessions');
    notify('Local workspace cache and session history cleared.', 'success');
  };

  const apiBase = getApiBaseUrl() || window.location.origin;

  return (
    <PageContainer>
      <PageHeader
        eyebrow="System Configuration"
        title="Settings & Preferences"
        subtitle="Manage workspace preferences, system environment connections, and local storage state."
      />

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        {/* Appearance & Accessibility */}
        <Card>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <IconSettings size={18} color="var(--primary)" aria-hidden />
            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>
              Accessibility & Display Preferences
            </h3>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem 0', borderBottom: '1px solid #f1f5f9' }}>
            <div>
              <p style={{ margin: 0, fontWeight: 600, fontSize: '0.88rem' }}>Reduced Motion</p>
              <p className="muted" style={{ margin: '0.1rem 0 0', fontSize: '0.78rem' }}>
                Minimize transitions and decorative animation effects across the UI.
              </p>
            </div>
            <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
              <input
                type="checkbox"
                checked={reducedMotion}
                onChange={handleToggleMotion}
                style={{ width: '18px', height: '18px' }}
              />
            </label>
          </div>
        </Card>

        {/* API & Backend Environment */}
        <Card>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <IconActivity size={18} color="var(--primary)" aria-hidden />
            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>
              Backend & API Environment
            </h3>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
            <div>
              <span className="muted" style={{ fontSize: '0.76rem', fontWeight: 600, textTransform: 'uppercase' }}>API Base URL:</span>
              <p style={{ margin: '0.2rem 0 0', fontFamily: 'var(--font-mono)', fontSize: '0.84rem', background: '#f8fafc', padding: '0.4rem 0.65rem', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                {apiBase}
              </p>
            </div>

            <div>
              <span className="muted" style={{ fontSize: '0.76rem', fontWeight: 600, textTransform: 'uppercase' }}>Application Mode:</span>
              <p style={{ margin: '0.2rem 0 0', fontFamily: 'var(--font-mono)', fontSize: '0.84rem', background: '#f8fafc', padding: '0.4rem 0.65rem', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                {(import.meta as any).env?.MODE || 'production'}
              </p>
            </div>
          </div>
        </Card>

        {/* User Account Info */}
        <Card>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <IconUser size={18} color="var(--primary)" aria-hidden />
            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>
              Active User Session
            </h3>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div>
              <span className="muted" style={{ fontSize: '0.76rem', fontWeight: 600, textTransform: 'uppercase' }}>Username:</span>
              <p style={{ margin: '0.2rem 0 0', fontSize: '0.88rem', fontWeight: 600 }}>{user?.username || 'Analyst'}</p>
            </div>

            <div>
              <span className="muted" style={{ fontSize: '0.76rem', fontWeight: 600, textTransform: 'uppercase' }}>Email:</span>
              <p style={{ margin: '0.2rem 0 0', fontSize: '0.88rem' }}>{user?.email || 'user@example.com'}</p>
            </div>
          </div>
        </Card>

        {/* Workspace Storage & Cache */}
        <Card>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
            <IconDatabase size={18} color="var(--primary)" aria-hidden />
            <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>
              Workspace State & Storage
            </h3>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <p style={{ margin: 0, fontWeight: 600, fontSize: '0.88rem' }}>Clear Local Session Cache</p>
              <p className="muted" style={{ margin: '0.1rem 0 0', fontSize: '0.78rem' }}>
                Resets the active dataset context and local analyst chat history.
              </p>
            </div>

            <button
              type="button"
              onClick={handleClearCache}
              className="action-btn"
              style={{ color: '#dc2626', borderColor: '#fecaca', padding: '0.35rem 0.85rem', fontSize: '0.82rem' }}
            >
              Clear Cache
            </button>
          </div>
        </Card>
      </div>
    </PageContainer>
  );
}
