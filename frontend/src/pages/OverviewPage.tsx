import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../auth/authContext';
import { listProjects } from '../services/authService';
import { PageContainer, PageHeader, Card } from '../components/layout/PageContainer';
import EmptyState from '../components/ui/EmptyState';
import { LoadingSpinner } from '../components/ui/LoadingState';
import {
  IconGrid,
  IconDatabase,
  IconFolder,
  IconTrendUp,
} from '../components/ui/Icons';

type Project = {
  id: number;
  name: string;
  description: string | null;
  datasets: unknown[];
};

export default function OverviewPage() {
  const { user } = useAuth();
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    listProjects()
      .then(setProjects)
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load'))
      .finally(() => setLoading(false));
  }, []);

  const totalDatasets = projects.reduce(
    (sum, p) => sum + (p.datasets?.length ?? 0),
    0,
  );

  return (
    <PageContainer>
      <PageHeader
        eyebrow="Welcome back"
        title={`Good to see you, ${user?.username ?? 'Analyst'}`}
        subtitle="Turn your data into evidence-backed decisions."
        actions={
          <div className="page-header-action-row">
            <Link to="/chat" className="primary-btn">
              ⚡ Start Analysis
            </Link>
            <Link to="/upload" className="action-btn">
              Upload Dataset
            </Link>
          </div>
        }
      />

      {/* KPI strip */}
      <div className="kpi-strip">
        <div className="kpi-tile">
          <span className="kpi-icon">
            <IconFolder size={20} aria-hidden />
          </span>
          <div>
            <p className="kpi-value">{loading ? '—' : projects.length}</p>
            <p className="kpi-label">Projects</p>
          </div>
        </div>
        <div className="kpi-tile">
          <span className="kpi-icon">
            <IconDatabase size={20} aria-hidden />
          </span>
          <div>
            <p className="kpi-value">{loading ? '—' : totalDatasets}</p>
            <p className="kpi-label">Datasets</p>
          </div>
        </div>
        <div className="kpi-tile">
          <span className="kpi-icon">
            <IconGrid size={20} aria-hidden />
          </span>
          <div>
            <p className="kpi-value">—</p>
            <p className="kpi-label">Analyses</p>
          </div>
        </div>
        <div className="kpi-tile">
          <span className="kpi-icon">
            <IconTrendUp size={20} aria-hidden />
          </span>
          <div>
            <p className="kpi-value">—</p>
            <p className="kpi-label">Models</p>
          </div>
        </div>
      </div>

      <div className="overview-grid">
        {/* Recent Projects */}
        <Card>
          <h2 className="card-title">Recent Projects</h2>
          {loading ? (
            <LoadingSpinner label="Loading projects…" size={24} />
          ) : error ? (
            <p className="status-error">{error}</p>
          ) : projects.length === 0 ? (
            <EmptyState
              icon={<IconFolder size={36} />}
              title="No projects yet"
              description="Create a project to organise your datasets and analyses."
              action={
                <Link to="/projects" className="action-btn">
                  Create project
                </Link>
              }
            />
          ) : (
            <ul className="overview-project-list">
              {projects.slice(0, 6).map((p) => (
                <li key={p.id}>
                  <Link to={`/projects/${p.id}`} className="overview-project-link">
                    <span className="overview-project-name">{p.name}</span>
                    <span className="overview-project-meta">
                      {p.datasets?.length ?? 0} dataset
                      {p.datasets?.length === 1 ? '' : 's'}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <div style={{ display: 'grid', gap: '1rem' }}>
          {/* System Status */}
          <Card>
            <h2 className="card-title">System Status</h2>
            <div className="status-grid">
              <div className="status-row">
                <span className="status-dot status-dot--ok" aria-label="Online" />
                <span className="status-label">Analysis Engine</span>
                <span className="status-value">Online</span>
              </div>
              <div className="status-row">
                <span className="status-dot status-dot--ok" aria-label="Online" />
                <span className="status-label">Model Registry</span>
                <span className="status-value">Online</span>
              </div>
              <div className="status-row">
                <span className="status-dot status-dot--ok" aria-label="Online" />
                <span className="status-label">Forecasting Engine</span>
                <span className="status-value">Online</span>
              </div>
              <div className="status-row">
                <span className="status-dot status-dot--warn" aria-label="Coming soon" />
                <span className="status-label">Monitoring</span>
                <span className="status-value">Phase 6</span>
              </div>
            </div>
          </Card>

          {/* Quick Actions */}
          <Card>
            <h2 className="card-title">Quick Actions</h2>
            <div className="quick-actions-grid">
              <Link to="/chat" className="quick-action-tile">
                <span className="quick-action-icon">⚡</span>
                <span className="quick-action-label">Command Studio</span>
              </Link>
              <Link to="/upload" className="quick-action-tile">
                <span className="quick-action-icon">📤</span>
                <span className="quick-action-label">Upload Data</span>
              </Link>
              <Link to="/projects" className="quick-action-tile">
                <span className="quick-action-icon">📁</span>
                <span className="quick-action-label">Projects</span>
              </Link>
              <Link to="/analyst" className="quick-action-tile">
                <span className="quick-action-icon">🤖</span>
                <span className="quick-action-label">AI Analyst</span>
              </Link>
            </div>
          </Card>
        </div>
      </div>
    </PageContainer>
  );
}
