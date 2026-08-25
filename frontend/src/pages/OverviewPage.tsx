import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../auth/authContext';
import { listProjects } from '../services/authService';
import { listModels } from '../services/modelService';
import { listReports } from '../services/reportService';
import { PageContainer, PageHeader, Card } from '../components/layout/PageContainer';
import EmptyState from '../components/ui/EmptyState';
import { LoadingSpinner } from '../components/ui/LoadingState';
import {
  IconGrid,
  IconDatabase,
  IconFolder,
  IconTrendUp,
  IconBrain,
  IconActivity,
  IconFileText,
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
  const [modelsCount, setModelsCount] = useState<number>(0);
  const [reportsCount, setReportsCount] = useState<number>(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      listProjects().catch(() => []),
      listModels().catch(() => []),
      listReports().catch(() => []),
    ])
      .then(([projs, mdls, reps]) => {
        setProjects(projs);
        setModelsCount(mdls.length);
        setReportsCount(reps.length);
      })
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load workspace overview'))
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
        subtitle="Turn your data into verified evidence-backed decisions across predictive & MLOps workspaces."
        actions={
          <div className="page-header-action-row">
            <Link to="/analyst" className="primary-btn">
              ⚡ Open AI Analyst
            </Link>
            <Link to="/upload" className="action-btn">
              📤 Upload Dataset
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
            <IconBrain size={20} aria-hidden />
          </span>
          <div>
            <p className="kpi-value">{loading ? '—' : modelsCount}</p>
            <p className="kpi-label">Models Registered</p>
          </div>
        </div>

        <div className="kpi-tile">
          <span className="kpi-icon">
            <IconFileText size={20} aria-hidden />
          </span>
          <div>
            <p className="kpi-value">{loading ? '—' : reportsCount}</p>
            <p className="kpi-label">Executive Reports</p>
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
            <h2 className="card-title">Platform Intelligence Status</h2>
            <div className="status-grid">
              <div className="status-row">
                <span className="status-dot status-dot--ok" aria-label="Online" />
                <span className="status-label">Conversational AI Analyst</span>
                <span className="status-value">Online</span>
              </div>
              <div className="status-row">
                <span className="status-dot status-dot--ok" aria-label="Online" />
                <span className="status-label">Model Registry & Training</span>
                <span className="status-value">Online</span>
              </div>
              <div className="status-row">
                <span className="status-dot status-dot--ok" aria-label="Online" />
                <span className="status-label">Forecasting & What-If</span>
                <span className="status-value">Online</span>
              </div>
              <div className="status-row">
                <span className="status-dot status-dot--ok" aria-label="Online" />
                <span className="status-label">Model Monitoring & Drift</span>
                <span className="status-value">Online</span>
              </div>
              <div className="status-row">
                <span className="status-dot status-dot--ok" aria-label="Online" />
                <span className="status-label">Executive PDF Reports</span>
                <span className="status-value">Online</span>
              </div>
            </div>
          </Card>

          {/* Quick Actions */}
          <Card>
            <h2 className="card-title">Quick Actions</h2>
            <div className="quick-actions-grid">
              <Link to="/analyst" className="quick-action-tile">
                <span className="quick-action-icon">🤖</span>
                <span className="quick-action-label">AI Analyst</span>
              </Link>
              <Link to="/models" className="quick-action-tile">
                <span className="quick-action-icon">🧠</span>
                <span className="quick-action-label">Model Registry</span>
              </Link>
              <Link to="/forecasts" className="quick-action-tile">
                <span className="quick-action-icon">📈</span>
                <span className="quick-action-label">Forecasting</span>
              </Link>
              <Link to="/monitoring" className="quick-action-tile">
                <span className="quick-action-icon">🛡️</span>
                <span className="quick-action-label">Monitoring</span>
              </Link>
              <Link to="/reports" className="quick-action-tile">
                <span className="quick-action-icon">📄</span>
                <span className="quick-action-label">Reports</span>
              </Link>
              <Link to="/datasets" className="quick-action-tile">
                <span className="quick-action-icon">💾</span>
                <span className="quick-action-label">Datasets</span>
              </Link>
            </div>
          </Card>
        </div>
      </div>
    </PageContainer>
  );
}
