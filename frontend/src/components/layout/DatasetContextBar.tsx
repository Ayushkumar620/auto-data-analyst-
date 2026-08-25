import React from 'react';
import { useDataset } from '../../context/DatasetContext';
import { IconDatabase } from '../ui/Icons';

type DatasetContextBarProps = {
  className?: string;
};

export default function DatasetContextBar({ className }: DatasetContextBarProps) {
  const { profile, fileName } = useDataset();

  if (!profile) {
    return (
      <div
        className={`topbar-dataset-bar topbar-dataset-bar--empty${className ? ` ${className}` : ''}`}
        aria-label="No dataset selected"
      >
        <IconDatabase size={14} className="topbar-dataset-icon" aria-hidden />
        <span className="topbar-dataset-name" style={{ color: 'var(--muted)', fontWeight: 500 }}>
          Select a dataset
        </span>
      </div>
    );
  }

  return (
    <div
      className={`topbar-dataset-bar${className ? ` ${className}` : ''}`}
      aria-label={`Active dataset: ${fileName ?? profile.dataset_name}`}
    >
      <IconDatabase size={14} className="topbar-dataset-icon" aria-hidden />
      <span className="topbar-dataset-name">
        {fileName ?? profile.dataset_name}
      </span>
      <span className="topbar-dataset-meta">
        {profile.rows.toLocaleString()} rows · {profile.columns} cols
      </span>
    </div>
  );
}

