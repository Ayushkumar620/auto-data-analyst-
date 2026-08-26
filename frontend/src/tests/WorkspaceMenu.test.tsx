import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import WorkspaceMenu from '../components/analyst/WorkspaceMenu';

describe('WorkspaceMenu Component', () => {
  it('renders menu trigger button with correct accessibility attributes', () => {
    render(
      <WorkspaceMenu
        viewMode="agent"
        onViewModeChange={vi.fn()}
        showExecutionDetails={true}
        onToggleExecutionDetails={vi.fn()}
        showExecutiveReport={true}
        onToggleExecutiveReport={vi.fn()}
        onResetWorkspace={vi.fn()}
      />,
    );

    const button = screen.getByRole('button', { name: /Open workspace menu/i });
    expect(button).toBeInTheDocument();
    expect(button).toHaveAttribute('aria-haspopup', 'menu');
    expect(button).toHaveAttribute('aria-expanded', 'false');
  });

  it('opens glassmorphic menu when clicked and displays options', () => {
    render(
      <WorkspaceMenu
        viewMode="agent"
        onViewModeChange={vi.fn()}
        showExecutionDetails={true}
        onToggleExecutionDetails={vi.fn()}
        showExecutiveReport={true}
        onToggleExecutiveReport={vi.fn()}
        onResetWorkspace={vi.fn()}
      />,
    );

    const button = screen.getByRole('button', { name: /Open workspace menu/i });
    fireEvent.click(button);

    expect(button).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('Agent View')).toBeInTheDocument();
    expect(screen.getByText('Analyst View')).toBeInTheDocument();
    expect(screen.getByText('Show Execution Details')).toBeInTheDocument();
    expect(screen.getByText('Show Executive Report')).toBeInTheDocument();
    expect(screen.getByText('? Reset Workspace')).toBeInTheDocument();
  });

  it('handles switching between Agent View and Analyst View', () => {
    const handleViewChange = vi.fn();
    render(
      <WorkspaceMenu
        viewMode="agent"
        onViewModeChange={handleViewChange}
        showExecutionDetails={true}
        onToggleExecutionDetails={vi.fn()}
        showExecutiveReport={true}
        onToggleExecutiveReport={vi.fn()}
        onResetWorkspace={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /Open workspace menu/i }));
    fireEvent.click(screen.getByText('Analyst View'));

    expect(handleViewChange).toHaveBeenCalledWith('analyst');
  });

  it('toggles execution details and executive report visibility', () => {
    const handleToggleDetails = vi.fn();
    const handleToggleReport = vi.fn();

    render(
      <WorkspaceMenu
        viewMode="agent"
        onViewModeChange={vi.fn()}
        showExecutionDetails={true}
        onToggleExecutionDetails={handleToggleDetails}
        showExecutiveReport={true}
        onToggleExecutiveReport={handleToggleReport}
        onResetWorkspace={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /Open workspace menu/i }));
    fireEvent.click(screen.getByText('Show Execution Details'));
    expect(handleToggleDetails).toHaveBeenCalledWith(false);

    fireEvent.click(screen.getByText('Show Executive Report'));
    expect(handleToggleReport).toHaveBeenCalledWith(false);
  });

  it('closes menu on Escape key press', () => {
    render(
      <WorkspaceMenu
        viewMode="agent"
        onViewModeChange={vi.fn()}
        showExecutionDetails={true}
        onToggleExecutionDetails={vi.fn()}
        showExecutiveReport={true}
        onToggleExecutiveReport={vi.fn()}
        onResetWorkspace={vi.fn()}
      />,
    );

    const button = screen.getByRole('button', { name: /Open workspace menu/i });
    fireEvent.click(button);
    expect(screen.getByText('Agent View')).toBeInTheDocument();

    fireEvent.keyDown(document, { key: 'Escape' });
    expect(screen.queryByText('Agent View')).not.toBeInTheDocument();
  });

  it('confirms before triggering reset workspace', () => {
    const handleReset = vi.fn();
    vi.spyOn(window, 'confirm').mockReturnValue(true);

    render(
      <WorkspaceMenu
        viewMode="agent"
        onViewModeChange={vi.fn()}
        showExecutionDetails={true}
        onToggleExecutionDetails={vi.fn()}
        showExecutiveReport={true}
        onToggleExecutiveReport={vi.fn()}
        onResetWorkspace={handleReset}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /Open workspace menu/i }));
    fireEvent.click(screen.getByText('? Reset Workspace'));

    expect(window.confirm).toHaveBeenCalled();
    expect(handleReset).toHaveBeenCalled();
  });
});
