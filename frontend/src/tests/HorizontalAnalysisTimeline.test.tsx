import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import HorizontalAnalysisTimeline from '../components/analyst/HorizontalAnalysisTimeline';

describe('HorizontalAnalysisTimeline Component', () => {
  it('renders stages from left to right with step badges and titles', () => {
    const { container } = render(
      <HorizontalAnalysisTimeline
        userIntent="Forecast Quarterly Sales"
        totalDurationMs={450}
      />,
    );

    expect(screen.getByText('Autonomous Execution Pipeline')).toBeInTheDocument();
    expect(screen.getByText('Intent & Planning')).toBeInTheDocument();
    expect(screen.getByText('Data Quality & Schema')).toBeInTheDocument();
    expect(screen.getByText('Modeling & Forecast')).toBeInTheDocument();
    expect(screen.getByText('Evidence Verification')).toBeInTheDocument();
    expect(screen.getByText('Executive Synthesis')).toBeInTheDocument();

    // Verify horizontal scroll region exists
    const track = container.querySelector('.timeline-scroll-track');
    expect(track).toBeInTheDocument();
  });

  it('expands stage details when a card is clicked', () => {
    render(
      <HorizontalAnalysisTimeline
        userIntent="Identify Outliers"
      />,
    );

    const intentCard = screen.getByText('Intent & Planning');
    fireEvent.click(intentCard);

    // Verify stage drawer opens
    expect(screen.getByText(/Intent & Planning — IntentAnalyzer/)).toBeInTheDocument();
    expect(screen.getByText(/Close/)).toBeInTheDocument();

    // Clicking close dismisses drawer
    fireEvent.click(screen.getByText(/Close/));
    expect(screen.queryByText(/Intent & Planning — IntentAnalyzer/)).not.toBeInTheDocument();
  });
});
