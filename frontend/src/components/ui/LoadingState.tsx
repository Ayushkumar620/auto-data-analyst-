import React from 'react';
import Spinner from './Spinner';

// ──────────────────────────────────────────────────────────────────────────
// LoadingSpinner
// ──────────────────────────────────────────────────────────────────────────
type LoadingSpinnerProps = { label?: string; size?: number };

export function LoadingSpinner({ label = 'Loading…', size = 32 }: LoadingSpinnerProps) {
  return (
    <div className="loading-spinner-wrap">
      <Spinner size={size} />
      <span className="loading-label">{label}</span>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────
// SkeletonCard
// ──────────────────────────────────────────────────────────────────────────
type SkeletonCardProps = { lines?: number; className?: string };

export function SkeletonCard({ lines = 3, className }: SkeletonCardProps) {
  return (
    <div
      className={`skeleton-card${className ? ` ${className}` : ''}`}
      aria-busy="true"
      aria-label="Loading content"
    >
      <div className="skeleton skeleton-title" />
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className={`skeleton skeleton-line${i === lines - 1 ? ' skeleton-line--short' : ''}`}
        />
      ))}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────
// SkeletonTable
// ──────────────────────────────────────────────────────────────────────────
type SkeletonTableProps = { rows?: number; cols?: number };

export function SkeletonTable({ rows = 5, cols = 4 }: SkeletonTableProps) {
  return (
    <div className="skeleton-table" aria-busy="true" aria-label="Loading table">
      <div className="skeleton-table-header">
        {Array.from({ length: cols }).map((_, i) => (
          <div key={i} className="skeleton skeleton-th" />
        ))}
      </div>
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} className="skeleton-table-row">
          {Array.from({ length: cols }).map((_, c) => (
            <div key={c} className="skeleton skeleton-td" />
          ))}
        </div>
      ))}
    </div>
  );
}
