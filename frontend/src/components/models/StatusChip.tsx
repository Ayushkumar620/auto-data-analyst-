import React from 'react';
import type { ModelStatus } from '../../types';

type StatusChipProps = {
  status: ModelStatus | string;
  className?: string;
};

export default function StatusChip({ status, className }: StatusChipProps) {
  const normalized = status.toLowerCase();

  let bg = '#f1f5f9';
  let color = '#475569';
  let border = '#cbd5e1';
  let label = status;

  if (normalized === 'active') {
    bg = '#ecfdf5';
    color = '#059669';
    border = '#a7f3d0';
    label = 'Active';
  } else if (normalized === 'staging') {
    bg = '#eff6ff';
    color = '#1d4ed8';
    border = '#bfdbfe';
    label = 'Staging';
  } else if (normalized === 'archived') {
    bg = '#f8fafc';
    color = '#94a3b8';
    border = '#e2e8f0';
    label = 'Archived';
  }

  return (
    <span
      className={`sidebar-soon-badge${className ? ` ${className}` : ''}`}
      style={{
        backgroundColor: bg,
        color,
        borderColor: border,
        textTransform: 'capitalize',
        fontSize: '0.72rem',
        fontWeight: 700,
        padding: '0.2rem 0.55rem',
      }}
    >
      {label}
    </span>
  );
}
