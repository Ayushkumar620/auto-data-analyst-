import React, { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { listProjects } from '../services/authService';
import type { Project } from '../types';

export default function ProjectViewPage() {
  const { projectId } = useParams<{ projectId: string }>();
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const listed: Project[] = (await listProjects()) as Project[];
      const found = listed.find((p) => String(p.id) === projectId);
      if (!found) {
        setError('Project not found.');
        setProject(null);
      } else {
        setProject(found);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load project');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [projectId]);

  return (
    <div className="page-stack">
      <header>
        <h1>{project?.name ?? 'Project'}</h1>
        <p className="muted">
          {projectId ? `Project #${projectId}` : ''} — organize and analyze your datasets here.
        </p>
      </header>

      {loading ? (
        <p className="muted">Loading…</p>
      ) : error ? (
        <div className="status-error">{error}</div>
      ) : (
        <section className="card">
          {project?.description ? <p>{project.description}</p> : null}
          <p className="muted">Datasets: {project?.datasets?.length ?? 0}</p>
          <Link to="/upload" className="link-row">
            Upload a dataset to this project
          </Link>
        </section>
      )}
    </div>
  );
}
