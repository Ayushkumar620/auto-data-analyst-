import React, { useRef, useState, useEffect } from 'react';
import { IconCheck, IconChevronLeft, IconChevronRight, IconBrain, IconTrendUp, IconActivity } from '../ui/Icons';

export type TimelineStage = {
  id: string;
  stepNumber: number;
  title: string;
  agentName: string;
  status: 'completed' | 'active' | 'pending' | 'warning';
  durationMs?: number;
  summary: string;
  details?: string[];
  metrics?: Record<string, string | number>;
};

type HorizontalAnalysisTimelineProps = {
  stages?: TimelineStage[];
  currentStep?: number;
  userIntent?: string;
  totalDurationMs?: number;
  className?: string;
  onSelectStage?: (stage: TimelineStage) => void;
};

export default function HorizontalAnalysisTimeline({
  stages: customStages,
  currentStep = 5,
  userIntent = 'Automated Analysis',
  totalDurationMs = 350,
  className,
  onSelectStage,
}: HorizontalAnalysisTimelineProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);
  const [selectedStageId, setSelectedStageId] = useState<string | null>(null);

  // Default standard 5-stage pipeline if not provided
  const stages: TimelineStage[] = customStages || [
    {
      id: 'stage-1',
      stepNumber: 1,
      title: 'Intent & Planning',
      agentName: 'IntentAnalyzer',
      status: currentStep >= 1 ? 'completed' : 'active',
      durationMs: 45,
      summary: `Decomposed query into targeted actions: ${userIntent}`,
      details: ['Parsed semantic entities and target measures', 'Generated DAG execution graph with dependency resolution'],
    },
    {
      id: 'stage-2',
      stepNumber: 2,
      title: 'Data Quality & Schema',
      agentName: 'DataValidationAgent',
      status: currentStep >= 2 ? 'completed' : currentStep === 1 ? 'active' : 'pending',
      durationMs: 78,
      summary: 'Verified column semantics, missing values, and data integrity',
      details: ['Checked data types and statistical distributions', 'Verified zero critical schema anomalies'],
    },
    {
      id: 'stage-3',
      stepNumber: 3,
      title: 'Modeling & Forecast',
      agentName: 'AutonomousModelOrchestrator',
      status: currentStep >= 3 ? 'completed' : currentStep === 2 ? 'active' : 'pending',
      durationMs: 120,
      summary: 'Trained models and evaluated baseline vs champion predictors',
      details: ['Evaluated performance across cross-validation splits', 'Selected optimal model architecture based on objective metrics'],
    },
    {
      id: 'stage-4',
      stepNumber: 4,
      title: 'Evidence Verification',
      agentName: 'ValidationAgent',
      status: currentStep >= 4 ? 'completed' : currentStep === 3 ? 'active' : 'pending',
      durationMs: 52,
      summary: 'Audited factual accuracy and statistical confidence',
      details: ['Validated assertions against underlying data facts', 'Ensured no speculative or hallucinated conclusions'],
    },
    {
      id: 'stage-5',
      stepNumber: 5,
      title: 'Executive Synthesis',
      agentName: 'DecisionExplainer',
      status: currentStep >= 5 ? 'completed' : currentStep === 4 ? 'active' : 'pending',
      durationMs: 55,
      summary: 'Synthesized actionable executive findings and recommendations',
      details: ['Generated evidence-backed executive summary', 'Structured charts and KPI comparison breakdown'],
    },
  ];

  const checkScroll = () => {
    const el = scrollRef.current;
    if (el) {
      setCanScrollLeft(el.scrollLeft > 5);
      setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 5);
    }
  };

  useEffect(() => {
    checkScroll();
    window.addEventListener('resize', checkScroll);
    return () => window.removeEventListener('resize', checkScroll);
  }, [stages]);

  const handleScroll = (direction: 'left' | 'right') => {
    const el = scrollRef.current;
    if (el) {
      const scrollAmount = direction === 'left' ? -280 : 280;
      el.scrollBy({ left: scrollAmount, behavior: 'smooth' });
      setTimeout(checkScroll, 320);
    }
  };

  const handleStageClick = (stage: TimelineStage) => {
    setSelectedStageId((prev) => (prev === stage.id ? null : stage.id));
    if (onSelectStage) onSelectStage(stage);
  };

  const activeStageDetails = stages.find((s) => s.id === selectedStageId);

  return (
    <div className={`horizontal-timeline-panel${className ? ` ${className}` : ''}`}>
      {/* Panel Header & Controls */}
      <div className="timeline-header">
        <div className="timeline-header-info">
          <div className="timeline-header-title-row">
            <span className="timeline-icon-wrap" aria-hidden="true">
              <IconBrain size={18} />
            </span>
            <h3 className="timeline-title">Autonomous Execution Pipeline</h3>
            <span className="timeline-stage-count">
              {stages.filter((s) => s.status === 'completed').length} / {stages.length} Completed
            </span>
          </div>
          <p className="timeline-subtitle">
            Step-by-step multi-agent reasoning flow moving from left to right across analytical stages.
          </p>
        </div>

        {/* Left / Right Scroll Buttons */}
        <div className="timeline-controls">
          <button
            type="button"
            className="timeline-nav-btn"
            onClick={() => handleScroll('left')}
            disabled={!canScrollLeft}
            aria-label="Scroll timeline left"
            title="Scroll left"
          >
            <IconChevronLeft size={16} aria-hidden />
          </button>
          <button
            type="button"
            className="timeline-nav-btn"
            onClick={() => handleScroll('right')}
            disabled={!canScrollRight}
            aria-label="Scroll timeline right"
            title="Scroll right"
          >
            <IconChevronRight size={16} aria-hidden />
          </button>
        </div>
      </div>

      {/* Horizontal Scroll Track */}
      <div
        ref={scrollRef}
        className="timeline-scroll-track"
        onScroll={checkScroll}
        tabIndex={0}
        role="region"
        aria-label="Analysis stages timeline from left to right"
      >
        <div className="timeline-cards-row">
          {stages.map((stage, idx) => {
            const isSelected = selectedStageId === stage.id;
            const isCompleted = stage.status === 'completed';
            const isActive = stage.status === 'active';

            return (
              <React.Fragment key={stage.id}>
                {idx > 0 && (
                  <div
                    className={`timeline-connector${isCompleted ? ' timeline-connector--completed' : ''}`}
                    aria-hidden="true"
                  >
                    <span className="timeline-connector-arrow">?</span>
                  </div>
                )}

                <div
                  className={`timeline-card timeline-card--${stage.status}${isSelected ? ' timeline-card--selected' : ''}`}
                  onClick={() => handleStageClick(stage)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleStageClick(stage);
                    }
                  }}
                  aria-pressed={isSelected}
                  aria-label={`Stage ${stage.stepNumber}: ${stage.title}, status: ${stage.status}`}
                >
                  <div className="timeline-card-top">
                    <span className="timeline-step-badge">Stage {stage.stepNumber}</span>
                    <span className={`timeline-status-pill timeline-status-pill--${stage.status}`}>
                      {isCompleted ? (
                        <>
                          <IconCheck size={11} aria-hidden /> Done
                        </>
                      ) : isActive ? (
                        '? Running'
                      ) : (
                        'Pending'
                      )}
                    </span>
                  </div>

                  <h4 className="timeline-card-heading">{stage.title}</h4>
                  <span className="timeline-agent-tag">{stage.agentName}</span>

                  <p className="timeline-card-summary">{stage.summary}</p>

                  <div className="timeline-card-footer">
                    {stage.durationMs !== undefined && (
                      <span className="timeline-duration">?? {stage.durationMs}ms</span>
                    )}
                    <span className="timeline-expand-hint">
                      {isSelected ? 'Hide Details ?' : 'View Details ?'}
                    </span>
                  </div>
                </div>
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* Expanded Stage Details Drawer / Dropdown */}
      {activeStageDetails && (
        <div className="timeline-stage-drawer fade-in">
          <div className="timeline-drawer-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="timeline-drawer-step">Stage {activeStageDetails.stepNumber}</span>
              <h4 className="timeline-drawer-title">{activeStageDetails.title} — {activeStageDetails.agentName}</h4>
            </div>
            <button
              type="button"
              className="ghost-text-btn"
              onClick={() => setSelectedStageId(null)}
              style={{ fontSize: '0.78rem' }}
            >
              Close ?
            </button>
          </div>

          <p style={{ margin: '0.3rem 0 0.6rem', fontSize: '0.88rem', color: 'var(--ink-secondary)' }}>
            {activeStageDetails.summary}
          </p>

          {activeStageDetails.details && activeStageDetails.details.length > 0 && (
            <ul className="timeline-drawer-list">
              {activeStageDetails.details.map((detail, dIdx) => (
                <li key={dIdx} className="timeline-drawer-item">
                  <span className="timeline-drawer-dot" />
                  <span>{detail}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
