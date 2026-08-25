import React from 'react';
import type { DriftSeverityLevel } from '../../types';
import { IconCheck, IconAlertTriangle, IconInfo } from '../ui/Icons';

type MonitoringStatusBadgeProps = {
  status: DriftSeverityLevel | string;
  className?: string;
};

export default function MonitoringStatusBadge({ status, className }: MonitoringStatusBadgeProps) {
  const norm = (status || 'NONE').toUpperCase();

  let bg = '#ecfdf5';
  let color = '#059669';
  let border = '#a7f3d0';
  let icon = <IconCheck size={13} aria-hidden />;
  let label = 'Healthy';

  if (norm === 'CRITICAL' || norm === 'HIGH') {
    bg = '#fef2f2';
    color = '#dc2626';
    border = '#fecaca';
    icon = <IconAlertTriangle size={13} aria-hidden />;
    label = norm === 'CRITICAL' ? 'Critical Drift' : 'High Drift';
  } else if (norm === 'WARNING' || norm === 'MEDIUM' || norm === 'LOW') {
    bg = '#fffbeb';
    color = '#d97706';
    border = '#fde68a';
    icon = <IconAlertTriangle size={13} aria-hidden />;
    label = norm === 'LOW' ? 'Low Drift' : 'Warning Drift';
  } else if (norm === 'NONE' || norm === 'HEALTHY') {
    bg = '#ecfdf5';
    color = '#059669';
    border = '#a7f3d0';
    icon = <IconCheck size={13} aria-hidden />;
    label = 'Healthy';
  } else {
    bg = '#f8fafc';
    color = '#64748b';
    border = '#cbd5e1';
    icon = <IconInfo size={13} aria-hidden />;
    label = norm;
  }

  return (
    <span
      className={`sidebar-soon-badge${className ? ` ${className}` : ''}`}
      style={{
        backgroundColor: bg,
        color,
        borderColor: border,
        fontSize: '0.72rem',
        fontWeight: 700,
        padding: '0.2rem 0.55rem',
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.3rem',
      }}
    >
      {icon}
      <span>{label}</span>
    </span>
  );
}
