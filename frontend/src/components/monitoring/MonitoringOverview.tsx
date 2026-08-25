import React from 'react';
import type { MonitoringOverviewData } from '../../types';
import { IconActivity, IconBrain, IconCheck, IconAlertTriangle } from '../ui/Icons';

type MonitoringOverviewProps = {
  data: MonitoringOverviewData | null;
  loading: boolean;
};

export default function MonitoringOverview({ data, loading }: MonitoringOverviewProps) {
  return (
    <div className="kpi-strip">
      <div className="kpi-tile">
        <span className="kpi-icon" style={{ backgroundColor: 'rgba(99, 102, 241, 0.1)', color: 'var(--primary)' }}>
          <IconBrain size={20} />
        </span>
        <div>
          <p className="kpi-value">{loading ? '—' : data?.total_models ?? 0}</p>
          <p className="kpi-label">Registered Models</p>
        </div>
      </div>

      <div className="kpi-tile">
        <span className="kpi-icon" style={{ backgroundColor: '#ecfdf5', color: '#059669' }}>
          <IconCheck size={20} />
        </span>
        <div>
          <p className="kpi-value">{loading ? '—' : data?.healthy_models ?? 0}</p>
          <p className="kpi-label">Healthy Models</p>
        </div>
      </div>

      <div className="kpi-tile">
        <span className="kpi-icon" style={{ backgroundColor: '#fffbeb', color: '#d97706' }}>
          <IconAlertTriangle size={20} />
        </span>
        <div>
          <p className="kpi-value">{loading ? '—' : data?.warning_models ?? 0}</p>
          <p className="kpi-label">Warning Models</p>
        </div>
      </div>

      <div className="kpi-tile">
        <span className="kpi-icon" style={{ backgroundColor: '#fef2f2', color: '#dc2626' }}>
          <IconAlertTriangle size={20} />
        </span>
        <div>
          <p className="kpi-value">{loading ? '—' : data?.critical_models ?? 0}</p>
          <p className="kpi-label">Critical Models</p>
        </div>
      </div>

      <div className="kpi-tile">
        <span className="kpi-icon" style={{ backgroundColor: '#eff6ff', color: '#1d4ed8' }}>
          <IconActivity size={20} />
        </span>
        <div>
          <p className="kpi-value">{loading ? '—' : data?.total_runs ?? 0}</p>
          <p className="kpi-label">Monitoring Runs</p>
        </div>
      </div>
    </div>
  );
}
