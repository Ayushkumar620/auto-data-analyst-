import React, { useEffect, useState } from 'react';
import { createProject, listProjects } from '../services/authService';
import type { Project as ProjectType } from '../types';

export default function ProjectsPage() {
  const [projects, setProjects] = useState<ProjectType[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState('');
  const [form, setForm] = useState({ name: '', description: '' });

  const refresh = async () => {
    setLoading(true);
    setError('');
    try {
      const listed = await listProjects();
      setProjects(listed as ProjectType[]);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load projects');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    setError('');
    try {
      await createProject(form.name, form.description || undefined);
      setForm({ name: '', description: '' });
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not create project');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="page-stack">
      <header>
        <h1>Projects</h1>
        <p className="muted">Organize your datasets and analyses into projects.</p>
      </header>

      <section className="card">
        <h2>New project</h2>
        {error ? <div className="status-error">{error}</div> : null}
        <form onSubmit={handleSubmit} style={{ display: 'grid', gap: '0.75rem' }}>
          <div className="field">
            <label htmlFor="project-name">Name</label>
            <input
              id="project-name"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              required
              disabled={creating}
            />
          </div>
          <div className="field">
            <label htmlFor="project-description">Description (optional)</label>
            <input
              id="project-description"
              value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })}
              disabled={creating}
            />
          </div>
          <button className="primary-btn" type="submit" disabled={creating}>
            {creating ? 'Creating…' : 'Create project'}
          </button>
        </form>
      </section>

      <section className="card">
        <h2>Your projects</h2>
        {loading ? (
          <p className="muted">Loading…</p>
        ) : projects.length === 0 ? (
          <div className="empty-state">No projects yet. Create one above.</div>
        ) : (
          <ul className="project-list">
            {projects.map((project) => (
              <li key={project.id} className="project-list-item">
                <a href={`/projects/${project.id}`}>{project.name}</a>
                <span className="muted">ID: {project.id}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
