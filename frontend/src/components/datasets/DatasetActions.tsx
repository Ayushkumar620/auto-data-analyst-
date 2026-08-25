import React from 'react';
import { useNavigate } from 'react-router-dom';
import type { DatasetProfile } from '../../types';
import { useDataset } from '../../context/DatasetContext';
import { useNotification } from '../../context/NotificationContext';
import { IconAnalyst, IconCheck, IconTrendUp } from '../ui/Icons';

type DatasetActionsProps = {
  profile: DatasetProfile;
  fileName?: string;
  onDelete?: () => void;
};

export default function DatasetActions({ profile, fileName, onDelete }: DatasetActionsProps) {
  const { setDataset, fileName: activeFileName, profile: activeProfile } = useDataset();
  const { notify } = useNotification();
  const navigate = useNavigate();

  const datasetName = fileName || profile.dataset_name;
  const isActive = activeFileName === datasetName || (activeProfile && activeProfile.dataset_name === datasetName);

  const handleSetActive = () => {
    setDataset(profile, datasetName);
    notify(`"${datasetName}" set as active dataset.`, 'success');
  };

  const handleAnalyze = () => {
    setDataset(profile, datasetName);
    navigate('/analyst');
  };

  const handleOpenStudio = () => {
    setDataset(profile, datasetName);
    navigate('/chat');
  };

  return (
    <div style={{ display: 'flex', gap: '0.6rem', flexWrap: 'wrap', alignItems: 'center' }}>
      <button
        type="button"
        className="primary-btn"
        onClick={handleAnalyze}
        style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.5rem 0.9rem', fontSize: '0.85rem' }}
      >
        <IconAnalyst size={16} aria-hidden />
        Analyze with AI
      </button>

      <button
        type="button"
        className="action-btn"
        onClick={handleOpenStudio}
        style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '0.5rem 0.85rem', fontSize: '0.85rem' }}
      >
        ⚡ Command Studio
      </button>

      {isActive ? (
        <span
          className="sidebar-soon-badge"
          style={{
            backgroundColor: '#ecfdf5',
            color: '#059669',
            borderColor: '#a7f3d0',
            padding: '0.4rem 0.75rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.3rem',
            fontSize: '0.8rem',
          }}
        >
          <IconCheck size={14} aria-hidden /> Active Dataset
        </span>
      ) : (
        <button
          type="button"
          className="action-btn"
          onClick={handleSetActive}
          style={{ padding: '0.5rem 0.85rem', fontSize: '0.85rem' }}
        >
          Set Active
        </button>
      )}

      {onDelete && (
        <button
          type="button"
          className="ghost-text-btn"
          onClick={onDelete}
          style={{ color: 'var(--alert)', fontSize: '0.85rem' }}
        >
          Delete
        </button>
      )}
    </div>
  );
}

