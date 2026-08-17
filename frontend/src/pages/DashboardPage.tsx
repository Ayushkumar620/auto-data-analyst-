import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../auth/authContext';
import { listProjects } from '../services/authService';

type Project = { id: number; name: string; description: string | null; datasets: unknown[] };

export default function DashboardPage() {
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

  const totalDatasets = projects.reduce((sum, p) => sum + (p.datasets?.length ?? 0), 0);

  return (
    <div className="page-stack">
      <header>
        <h1>Dashboard</h1>
        <p className="muted">Welcome back, {user?.username ?? '—'}.</p>
      </header>

      <section className="metric-grid">
        <article className="metric-tile">
          <p className="metric-label">Projects</p>
          <p className="metric-value">{projects.length}</p>
        </article>
        <article className="metric-tile">
          <p className="metric-label">Datasets</p>
          <p className="metric-value">{totalDatasets}</p>
        </article>
        <article className="metric-tile">
          <p className="metric-label">Analyses</p>
          <p className="metric-value">0</p>
        </article>
        <article className="metric-tile">
          <p className="metric-label">Unread chats</p>
          <p className="metric-value">0</p>
        </article>
      </section>

      <section className="card">
        <h2>Your projects</h2>
        {loading ? (
          <p className="muted">Loading projects…</p>
        ) : error ? (
          <p className="status-error">{error}</p>
        ) : projects.length === 0 ? (
          <div className="empty-state">
            <p>You don't have any projects yet.</p>
            <Link to="/projects">Create your first project</Link>
          </div>
        ) : (
          <ul className="project-list">
            {projects.map((project) => (
              <li key={project.id} className="project-list-item">
                <Link to={`/projects/${project.id}`}>{project.name}</Link>
                <span className="muted">{project.datasets?.length ?? 0} datasets</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
