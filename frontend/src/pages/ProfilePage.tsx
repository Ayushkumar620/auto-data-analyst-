import React from 'react';
import { useAuth } from '../auth/authContext';

export default function ProfilePage() {
  const { user, logout } = useAuth();

  return (
    <div className="page-stack">
      <header>
        <h1>Profile</h1>
        <p className="muted">Account information</p>
      </header>

      <section className="card">
        <div className="metric-grid" style={{ gridTemplateColumns: '1fr 3fr' }}>
          <article className="metric-tile">
            <p className="metric-label">Username</p>
            <p className="metric-value">{user?.username ?? '—'}</p>
          </article>
          <article className="metric-tile">
            <p className="metric-label">Email</p>
            <p className="metric-value">{user?.email ?? '—'}</p>
          </article>
          <article className="metric-tile">
            <p className="metric-label">Active</p>
            <p className="metric-value">{user?.is_active ? 'Yes' : 'No'}</p>
          </article>
          <article className="metric-tile">
            <p className="metric-label">User ID</p>
            <p className="metric-value">{user?.id ?? '—'}</p>
          </article>
        </div>
        <button className="primary-btn" style={{ width: 'auto', marginTop: '1rem' }} onClick={logout}>
          Log out
        </button>
      </section>
    </div>
  );
}