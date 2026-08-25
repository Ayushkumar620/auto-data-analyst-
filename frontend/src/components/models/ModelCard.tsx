import React from 'react';
import { Link } from 'react-router-dom';
import type { ModelMetadata } from '../../types';
import StatusChip from './StatusChip';
import MetricBadge from './MetricBadge';
import { IconBrain, IconTrendUp } from '../ui/Icons';

type ModelCardProps = {
  model: ModelMetadata;
};

export default function ModelCard({ model }: ModelCardProps) {
  const familyLabel =
    model.model_family === 'traditional_ml'
      ? 'Traditional ML'
      : model.model_family === 'ann'
      ? 'Neural Network (ANN)'
      : model.model_family === 'cnn'
      ? 'Convolutional (CNN)'
      : model.model_family === 'forecasting'
      ? 'Forecasting'
      : model.model_family;

  return (
    <div
      className="glass-card glass-card--padded"
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        transition: 'all 200ms ease',
      }}
    >
      <div>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '0.5rem', marginBottom: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', minWidth: 0 }}>
            <div
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '10px',
                backgroundColor: 'var(--primary-light)',
                color: 'var(--primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
              }}
            >
              <IconBrain size={20} />
            </div>
            <div style={{ minWidth: 0 }}>
              <h3
                style={{
                  margin: 0,
                  fontSize: '0.98rem',
                  fontWeight: 600,
                  color: 'var(--ink)',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}
                title={model.name}
              >
                {model.name}
              </h3>
              <p className="muted" style={{ margin: 0, fontSize: '0.75rem' }}>
                v{model.version} · {model.algorithm}
              </p>
            </div>
          </div>
          <StatusChip status={model.status} />
        </div>

        {/* Badges */}
        <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginBottom: '0.85rem' }}>
          <span
            style={{
              padding: '0.15rem 0.5rem',
              borderRadius: '4px',
              backgroundColor: '#eef2ff',
              color: '#3730a3',
              fontSize: '0.72rem',
              fontWeight: 600,
            }}
          >
            {familyLabel}
          </span>
          <span
            style={{
              padding: '0.15rem 0.5rem',
              borderRadius: '4px',
              backgroundColor: '#f1f5f9',
              color: '#334155',
              fontSize: '0.72rem',
              fontWeight: 500,
              textTransform: 'capitalize',
            }}
          >
            {model.problem_type.replace(/_/g, ' ')}
          </span>
        </div>

        {/* Target and Metrics */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            backgroundColor: 'rgba(248, 250, 252, 0.9)',
            borderRadius: '10px',
            padding: '0.65rem 0.75rem',
            marginBottom: '1rem',
          }}
        >
          <div>
            <span className="metric-label" style={{ fontSize: '0.68rem' }}>Target</span>
            <p style={{ margin: '0.1rem 0 0', fontWeight: 600, fontSize: '0.86rem', color: 'var(--ink)' }}>
              {model.target_column}
            </p>
          </div>
          <MetricBadge
            name={model.primary_metric_name}
            value={model.primary_metric_value}
            label={model.primary_metric_name}
          />
        </div>
      </div>

      {/* Footer Actions */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid #e2e8f0', paddingTop: '0.75rem' }}>
        <span className="muted" style={{ fontSize: '0.76rem' }}>
          {model.feature_columns.length} Features
        </span>
        <Link
          to={`/models/${encodeURIComponent(model.model_id)}`}
          className="primary-btn"
          style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem', textDecoration: 'none' }}
        >
          Inspect Model →
        </Link>
      </div>
    </div>
  );
}
