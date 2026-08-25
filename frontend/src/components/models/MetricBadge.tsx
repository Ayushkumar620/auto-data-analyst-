import React from 'react';

type MetricBadgeProps = {
  name: string;
  value: number;
  label?: string;
};

export default function MetricBadge({ name, value, label }: MetricBadgeProps) {
  const isPercentage = ['accuracy', 'f1', 'precision', 'recall', 'r2', 'roc_auc'].some((m) =>
    name.toLowerCase().includes(m),
  );

  const formattedValue = isPercentage
    ? value <= 1.0
      ? `${(value * 100).toFixed(2)}%`
      : `${value.toFixed(2)}%`
    : value.toFixed(4);

  return (
    <div
      style={{
        padding: '0.45rem 0.65rem',
        borderRadius: '8px',
        backgroundColor: '#f8fafc',
        border: '1px solid #e2e8f0',
        display: 'inline-flex',
        flexDirection: 'column',
        minWidth: '90px',
      }}
    >
      <span
        className="metric-label"
        style={{ fontSize: '0.68rem', textTransform: 'uppercase', color: 'var(--muted)' }}
      >
        {label || name}
      </span>
      <span
        style={{
          fontFamily: 'var(--font-heading)',
          fontWeight: 700,
          fontSize: '0.96rem',
          color: 'var(--ink)',
          marginTop: '0.1rem',
        }}
      >
        {formattedValue}
      </span>
    </div>
  );
}
