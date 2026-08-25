import React from 'react';
import type { ReportKPI } from '../../types';

type ReportMetricsProps = {
  kpis: ReportKPI[];
};

export default function ReportMetrics({ kpis }: ReportMetricsProps) {
  if (!kpis || kpis.length === 0) return null;

  return (
    <div>
      <h3 className="section-title" style={{ margin: '0 0 0.85rem' }}>
        Key Analytical Metrics
      </h3>

      <div className="kpi-strip" style={{ margin: 0 }}>
        {kpis.map((kpi, idx) => {
          const isPositive = typeof kpi.change === 'number' && kpi.change >= 0;
          return (
            <div key={idx} className="kpi-tile">
              <span className="kpi-icon" style={{ backgroundColor: 'rgba(99, 102, 241, 0.08)', color: 'var(--primary)' }}>
                📊
              </span>
              <div>
                <p className="kpi-value">
                  {kpi.formatted || (typeof kpi.value === 'number' ? kpi.value.toLocaleString() : String(kpi.value))}
                </p>
                <p className="kpi-label">{kpi.name}</p>
                {kpi.change !== undefined && (
                  <span
                    style={{
                      fontSize: '0.74rem',
                      fontWeight: 600,
                      color: isPositive ? '#059669' : '#dc2626',
                    }}
                  >
                    {kpi.change > 0 ? `+${kpi.change}%` : `${kpi.change}%`}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
