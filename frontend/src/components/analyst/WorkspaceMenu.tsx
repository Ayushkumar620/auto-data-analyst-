import React, { useState, useRef, useEffect } from 'react';
import { IconMenu, IconCheck, IconActivity, IconBrain, IconFileText, IconSettings } from '../ui/Icons';

export type WorkspaceViewMode = 'agent' | 'analyst';

export type WorkspaceMenuProps = {
  viewMode: WorkspaceViewMode;
  onViewModeChange: (mode: WorkspaceViewMode) => void;
  showExecutionDetails: boolean;
  onToggleExecutionDetails: (show: boolean) => void;
  showExecutiveReport: boolean;
  onToggleExecutiveReport: (show: boolean) => void;
  onResetWorkspace: () => void;
  className?: string;
};

export default function WorkspaceMenu({
  viewMode,
  onViewModeChange,
  showExecutionDetails,
  onToggleExecutionDetails,
  showExecutiveReport,
  onToggleExecutiveReport,
  onResetWorkspace,
  className,
}: WorkspaceMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  // Close on outside click
  useEffect(() => {
    const handleOutsideClick = (event: MouseEvent | TouchEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleOutsideClick);
      document.addEventListener('touchstart', handleOutsideClick);
    }
    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
      document.removeEventListener('touchstart', handleOutsideClick);
    };
  }, [isOpen]);

  // Close on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
        buttonRef.current?.focus();
      }
    };
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
    }
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  const handleSelectView = (mode: WorkspaceViewMode) => {
    onViewModeChange(mode);
    setIsOpen(false);
  };

  const handleToggleDetails = () => {
    onToggleExecutionDetails(!showExecutionDetails);
  };

  const handleToggleReport = () => {
    onToggleExecutiveReport(!showExecutiveReport);
  };

  const handleReset = () => {
    const confirmed = window.confirm(
      'Reset workspace view settings? (Your uploaded datasets and backend analysis results will NOT be deleted.)'
    );
    if (confirmed) {
      onResetWorkspace();
      setIsOpen(false);
    }
  };

  return (
    <div className={`workspace-menu-container${className ? ` ${className}` : ''}`} ref={menuRef}>
      {/* ? Menu Trigger Button */}
      <button
        ref={buttonRef}
        type="button"
        className={`workspace-menu-btn${isOpen ? ' workspace-menu-btn--active' : ''}`}
        onClick={() => setIsOpen((prev) => !prev)}
        aria-label="Open workspace menu"
        aria-haspopup="menu"
        aria-expanded={isOpen}
        title="Workspace controls and display settings"
      >
        <IconMenu size={16} aria-hidden />
        <span>Menu</span>
      </button>

      {/* Glassmorphic Popover Dropdown */}
      {isOpen && (
        <div
          className="workspace-menu-dropdown fade-in"
          role="menu"
          aria-label="Workspace options"
        >
          {/* Section: View Mode */}
          <div className="workspace-menu-group">
            <div className="workspace-menu-label">Display View</div>
            
            <button
              type="button"
              className={`workspace-menu-item${viewMode === 'agent' ? ' workspace-menu-item--active' : ''}`}
              role="menuitem"
              onClick={() => handleSelectView('agent')}
            >
              <div className="workspace-menu-item-left">
                <IconBrain size={15} className="workspace-menu-icon" />
                <div>
                  <div className="workspace-menu-item-title">Agent View</div>
                  <div className="workspace-menu-item-desc">Developer execution graph & agents</div>
                </div>
              </div>
              {viewMode === 'agent' && <IconCheck size={14} className="workspace-menu-check" />}
            </button>

            <button
              type="button"
              className={`workspace-menu-item${viewMode === 'analyst' ? ' workspace-menu-item--active' : ''}`}
              role="menuitem"
              onClick={() => handleSelectView('analyst')}
            >
              <div className="workspace-menu-item-left">
                <IconActivity size={15} className="workspace-menu-icon" />
                <div>
                  <div className="workspace-menu-item-title">Analyst View</div>
                  <div className="workspace-menu-item-desc">Clean, user-facing workflow</div>
                </div>
              </div>
              {viewMode === 'analyst' && <IconCheck size={14} className="workspace-menu-check" />}
            </button>
          </div>

          <div className="workspace-menu-divider" />

          {/* Section: Visibility Toggles */}
          <div className="workspace-menu-group">
            <div className="workspace-menu-label">Visibility Controls</div>

            <button
              type="button"
              className="workspace-menu-item"
              role="menuitemcheckbox"
              aria-checked={showExecutionDetails}
              onClick={handleToggleDetails}
            >
              <div className="workspace-menu-item-left">
                <IconSettings size={15} className="workspace-menu-icon" />
                <span>Show Execution Details</span>
              </div>
              <span className={`workspace-menu-toggle-pill${showExecutionDetails ? ' workspace-menu-toggle-pill--on' : ''}`}>
                {showExecutionDetails ? 'ON' : 'OFF'}
              </span>
            </button>

            <button
              type="button"
              className="workspace-menu-item"
              role="menuitemcheckbox"
              aria-checked={showExecutiveReport}
              onClick={handleToggleReport}
            >
              <div className="workspace-menu-item-left">
                <IconFileText size={15} className="workspace-menu-icon" />
                <span>Show Executive Report</span>
              </div>
              <span className={`workspace-menu-toggle-pill${showExecutiveReport ? ' workspace-menu-toggle-pill--on' : ''}`}>
                {showExecutiveReport ? 'ON' : 'OFF'}
              </span>
            </button>
          </div>

          <div className="workspace-menu-divider" />

          {/* Section: Reset Action */}
          <div className="workspace-menu-group">
            <button
              type="button"
              className="workspace-menu-item workspace-menu-item--danger"
              role="menuitem"
              onClick={handleReset}
            >
              <span>? Reset Workspace</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
