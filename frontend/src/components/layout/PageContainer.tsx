import React from 'react';

// ──────────────────────────────────────────────────────────────────────────
// PageContainer
// ──────────────────────────────────────────────────────────────────────────
type PageContainerProps = { children: React.ReactNode; className?: string; style?: React.CSSProperties };

export function PageContainer({ children, className, style }: PageContainerProps) {
  return (
    <div className={`page-container${className ? ` ${className}` : ''}`} style={style}>
      {children}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────
// PageHeader
// ──────────────────────────────────────────────────────────────────────────
type PageHeaderProps = {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
};

export function PageHeader({ eyebrow, title, subtitle, actions }: PageHeaderProps) {
  return (
    <div className="page-header">
      <div className="page-header-text">
        {eyebrow && <p className="page-eyebrow">{eyebrow}</p>}
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="page-subtitle">{subtitle}</p>}
      </div>
      {actions && <div className="page-header-actions">{actions}</div>}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────
// SectionHeader
// ──────────────────────────────────────────────────────────────────────────
type SectionHeaderProps = {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
};

export function SectionHeader({ title, subtitle, actions }: SectionHeaderProps) {
  return (
    <div className="section-header">
      <div>
        <h2 className="section-title">{title}</h2>
        {subtitle && <p className="section-subtitle">{subtitle}</p>}
      </div>
      {actions && <div className="section-header-actions">{actions}</div>}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────
// Card
// ──────────────────────────────────────────────────────────────────────────
type CardProps = {
  children: React.ReactNode;
  className?: string;
  padding?: boolean;
};

export function Card({ children, className, padding = true }: CardProps) {
  return (
    <div
      className={`glass-card${padding ? ' glass-card--padded' : ''}${
        className ? ` ${className}` : ''
      }`}
    >
      {children}
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────────
// GlassPanel
// ──────────────────────────────────────────────────────────────────────────
type GlassPanelProps = { children: React.ReactNode; className?: string };

export function GlassPanel({ children, className }: GlassPanelProps) {
  return (
    <div className={`glass-panel${className ? ` ${className}` : ''}`}>
      {children}
    </div>
  );
}
