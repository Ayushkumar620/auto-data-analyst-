import React, { useState } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import { useAuth } from '../auth/authContext';

export default function LoginPage() {
  const { user, login, logout, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: string })?.from ?? '/dashboard';

  const [email, setEmail] = useState('demo@example.com');
  const [password, setPassword] = useState('strongpass123');
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    setError('');
    try {
      await login({ email, password });
      navigate(from, { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed. Please check credentials.');
    } finally {
      setSubmitting(false);
    }
  };

  const fillDemoAccount = () => {
    setEmail('demo@example.com');
    setPassword('strongpass123');
    setError('');
  };

  return (
    <div className="auth-splash">
      <div className="auth-card">
        <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          <div className="logo" style={{ justifyContent: 'center', fontSize: '1.6rem', marginBottom: '0.4rem' }}>
            Auto Data Analyst
          </div>
          <p className="muted" style={{ margin: 0, fontSize: '0.95rem' }}>
            Sign in to access your autonomous AI data studio.
          </p>
        </div>

        {isAuthenticated && user ? (
          <div style={{ background: '#ecfdf5', border: '1px solid #a7f3d0', borderRadius: '10px', padding: '0.9rem', marginBottom: '1.2rem', textAlign: 'center' }}>
            <div style={{ color: '#065f46', fontSize: '0.88rem', fontWeight: 600, marginBottom: '0.5rem' }}>
              ✓ Currently signed in as <strong>{user.username}</strong> ({user.email})
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center' }}>
              <button
                type="button"
                onClick={() => navigate('/dashboard')}
                className="primary-btn"
                style={{ padding: '0.4rem 0.8rem', fontSize: '0.82rem', width: 'auto' }}
              >
                Go to Dashboard →
              </button>
              <button
                type="button"
                onClick={logout}
                className="action-btn"
                style={{ padding: '0.4rem 0.8rem', fontSize: '0.82rem', width: 'auto', background: '#fee2e2', color: '#991b1b', border: '1px solid #fca5a5' }}
              >
                Sign Out
              </button>
            </div>
          </div>
        ) : null}

        {error ? <div className="status-error" style={{ marginBottom: '1.2rem' }}>{error}</div> : null}

        <form onSubmit={handleSubmit}>
          <div className="field">
            <label htmlFor="login-email">Email</label>
            <input
              id="login-email"
              type="email"
              placeholder="name@company.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="field">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <label htmlFor="login-password">Password</label>
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                style={{ background: 'none', border: 'none', color: '#6366f1', fontSize: '0.78rem', cursor: 'pointer', fontWeight: 600 }}
              >
                {showPassword ? 'Hide' : 'Show'}
              </button>
            </div>
            <input
              id="login-password"
              type={showPassword ? 'text' : 'password'}
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>

          <button className="primary-btn" type="submit" disabled={submitting} style={{ marginTop: '0.6rem' }}>
            {submitting ? 'Authenticating…' : 'Sign In to Workspace'}
          </button>
        </form>

        <div style={{ marginTop: '1.2rem', paddingTop: '1.2rem', borderTop: '1px solid #e2e8f0', textAlign: 'center' }}>
          <button
            type="button"
            className="action-btn"
            onClick={fillDemoAccount}
            style={{ width: '100%', fontSize: '0.85rem', padding: '0.6rem', background: '#eef2ff', color: '#4338ca', border: '1px solid #c7d2fe' }}
          >
            ⚡ Auto-Fill Localhost Demo Credentials
          </button>

          <div className="link-row" style={{ marginTop: '1rem' }}>
            Don't have an account? <Link to="/register" style={{ color: '#4f46e5', fontWeight: 700 }}>Create an account</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
