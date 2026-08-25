import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { PageContainer, PageHeader, Card } from '../components/layout/PageContainer';
import EmptyState from '../components/ui/EmptyState';
import ErrorState from '../components/ui/ErrorState';
import { LoadingSpinner } from '../components/ui/LoadingState';
import { IconWorkspace, IconFolder, IconDatabase } from '../components/ui/Icons';
import { listProjects, type Project } from '../services/authService';
import { createWorkspace } from '../services/workspaceService';
import type { Workspace } from '../types';

export default function WorkspacesPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [workspaceName, setWorkspaceName] = useState('');
  const [creating, setCreating] = useState(false);
  const [createdWorkspaces, setCreatedWorkspaces] = useState<Workspace[]>(() => {
    try {
      const raw = localStorage.getItem('auto_analyst_local_workspaces');
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  });

  const loadData = async () => {
    setLoading(true);
    setError('');
    try {
      const projs = await listProjects();
      setProjects(projs);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load workspace projects');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleCreateWorkspace = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!workspaceName.trim()) return;

    setCreating(true);
    setError('');
    try {
      const ws = await createWorkspace({
        name: workspaceName.trim(),
        owner: 'user',
      });
      const updated = [ws, ...createdWorkspaces];
      setCreatedWorkspaces(updated);
      localStorage.setItem('auto_analyst_local_workspaces', JSON.stringify(updated));
      setWorkspaceName('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create workspace');
    } finally {
      setCreating(false);
    }
  };

  return (
    <PageContainer>
      <PageHeader
        eyebrow="Organization"
        title="Workspaces"
        subtitle="Manage collaborative environments, dataset isolation, and team projects."
      />

      {/* Create Workspace Panel */}
      <Card>
        <h2 className="section-title" style={{ margin: '0 0 0.5rem' }}>Create New Workspace</h2>
        <form onSubmit={handleCreateWorkspace} style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
          <input
            type="text"
            placeholder="Workspace Name (e.g. Finance Analytics)"
            value={workspaceName}
            onChange={(e) => setWorkspaceName(e.target.value)}
            className="horizon-input"
            style={{ flex: 1, minWidth: '240px', padding: '0.5rem 0.85rem' }}
            required
            disabled={creating}
          />
          <button type="submit" className="primary-btn" disabled={creating || !workspaceName.trim()}>
            {creating ? 'Creating…' : 'Create Workspace'}
          </button>
        </form>
      </Card>

      {error && <ErrorState message={error} />}

      {/* Workspaces List */}
      <div>
        <h2 className="section-title" style={{ margin: '0 0 1rem' }}>Active Workspaces & Projects</h2>

        {loading ? (
          <LoadingSpinner label="Loading workspaces…" size={32} />
        ) : createdWorkspaces.length === 0 && projects.length === 0 ? (
          <EmptyState
            icon={<IconWorkspace size={48} />}
            title="No workspaces yet"
            description="Create a workspace above to organize your projects and datasets."
          />
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1.25rem' }}>
            {/* Display local created workspaces */}
            {createdWorkspaces.map((ws) => (
              <div key={ws.id} className="glass-card glass-card--padded">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.75rem' }}>
                  <div
                    style={{
                      width: '36px',
                      height: '36px',
                      borderRadius: '8px',
                      backgroundColor: 'var(--primary-light)',
                      color: 'var(--primary)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <IconWorkspace size={20} />
                  </div>
                  <div>
                    <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>{ws.name}</h3>
                    <p className="muted" style={{ margin: 0, fontSize: '0.75rem' }}>ID: {ws.id}</p>
                  </div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid #e2e8f0', paddingTop: '0.75rem', marginTop: '0.75rem' }}>
                  <span className="muted" style={{ fontSize: '0.82rem' }}>Owner: {ws.owner}</span>
                  <Link to="/projects" className="action-btn" style={{ padding: '0.25rem 0.6rem', fontSize: '0.78rem', textDecoration: 'none' }}>
                    View Projects →
                  </Link>
                </div>
              </div>
            ))}

            {/* Display existing project-based workspaces */}
            {projects.map((p) => (
              <div key={p.id} className="glass-card glass-card--padded">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.75rem' }}>
                  <div
                    style={{
                      width: '36px',
                      height: '36px',
                      borderRadius: '8px',
                      backgroundColor: '#eff6ff',
                      color: '#1d4ed8',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <IconFolder size={20} />
                  </div>
                  <div>
                    <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>{p.name}</h3>
                    <p className="muted" style={{ margin: 0, fontSize: '0.75rem' }}>
                      {p.datasets?.length || 0} Datasets
                    </p>
                  </div>
                </div>
                {p.description && (
                  <p className="muted" style={{ margin: '0 0 0.75rem', fontSize: '0.84rem' }}>
                    {p.description}
                  </p>
                )}
                <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid #e2e8f0', paddingTop: '0.75rem' }}>
                  <span className="muted" style={{ fontSize: '0.8rem' }}>Project ID: {p.id}</span>
                  <Link to={`/projects/${p.id}`} className="action-btn" style={{ padding: '0.25rem 0.6rem', fontSize: '0.78rem', textDecoration: 'none' }}>
                    Open Project →
                  </Link>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </PageContainer>
  );
}

