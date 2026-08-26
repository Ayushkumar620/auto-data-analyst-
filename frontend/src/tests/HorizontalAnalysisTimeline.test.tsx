import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import HorizontalAnalysisTimeline from '../components/analyst/HorizontalAnalysisTimeline';

describe('HorizontalAnalysisTimeline Component', () => {
  it('renders stages from left to right with step badges, down scroll panel and buttons', () => {
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

    // Verify main horizontal scroll track exists
    const track = container.querySelector('.timeline-scroll-track');
    expect(track).toBeInTheDocument();

    // Verify Down Scroll Panel with Left and Right navigation buttons exists
    expect(screen.getByText('Scroll Left')).toBeInTheDocument();
    expect(screen.getByText('Scroll Right')).toBeInTheDocument();

    // Verify quick jump step pills exist
    expect(screen.getByText('S1: Intent')).toBeInTheDocument();
    expect(screen.getByText('S2: Data')).toBeInTheDocument();
    expect(screen.getByText('S3: Modeling')).toBeInTheDocument();
    expect(screen.getByText('S4: Evidence')).toBeInTheDocument();
    expect(screen.getByText('S5: Executive')).toBeInTheDocument();
  });

  it('expands stage details when a card or down pill is clicked', () => {
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
