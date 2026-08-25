import React, { useEffect, useRef, useState, useCallback } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import {
  IconHome,
  IconAnalyst,
  IconDatabase,
  IconBarChart,
  IconBrain,
  IconTrendUp,
  IconActivity,
  IconFileText,
  IconFolder,
  IconSettings,
  IconChevronLeft,
  IconChevronRight,
  IconLightbulb,
  IconWorkspace,
} from '../ui/Icons';

// ──────────────────────────────────────────────────────────────────────────
// Nav config
// ──────────────────────────────────────────────────────────────────────────
type NavItem = {
  label: string;
  to: string;
  Icon: React.ComponentType<{
    size?: number;
    className?: string;
    'aria-hidden'?: boolean;
  }>;
  comingSoon?: boolean;
};

type NavGroup = {
  group: string;
  items: NavItem[];
};

const NAV_GROUPS: NavGroup[] = [
  {
    group: 'WORKSPACE',
    items: [
      { label: 'Overview', to: '/overview', Icon: IconHome },
      { label: 'Analyst', to: '/analyst', Icon: IconAnalyst },
      { label: 'Datasets', to: '/datasets', Icon: IconDatabase, comingSoon: true },
      { label: 'Analyses', to: '/analyses', Icon: IconBarChart, comingSoon: true },
    ],
  },
  {
    group: 'INTELLIGENCE',
    items: [
      { label: 'Insights', to: '/upload', Icon: IconLightbulb },
      { label: 'Models', to: '/models', Icon: IconBrain, comingSoon: true },
      { label: 'Forecasts', to: '/forecasts', Icon: IconTrendUp, comingSoon: true },
      { label: 'Monitoring', to: '/monitoring', Icon: IconActivity, comingSoon: true },
    ],
  },
  {
    group: 'OUTPUT',
    items: [
      { label: 'Reports', to: '/reports', Icon: IconFileText, comingSoon: true },
    ],
  },
  {
    group: 'PROJECT',
    items: [
      { label: 'Projects', to: '/projects', Icon: IconFolder },
      { label: 'Workspaces', to: '/workspaces', Icon: IconWorkspace, comingSoon: true },
    ],
  },
  {
    group: 'SYSTEM',
    items: [
      { label: 'Settings', to: '/settings', Icon: IconSettings, comingSoon: true },
    ],
  },
];

const COLLAPSED_KEY = 'sidebar_collapsed';

// ──────────────────────────────────────────────────────────────────────────
// Component
// ──────────────────────────────────────────────────────────────────────────
type SidebarProps = {
  mobileOpen: boolean;
  onMobileClose: () => void;
};

export default function Sidebar({ mobileOpen, onMobileClose }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(
    () => localStorage.getItem(COLLAPSED_KEY) === 'true',
  );

  const navigate = useNavigate();
  void navigate; // suppress unused warning (used in handleNavClick via React Router)

  const toggleCollapsed = useCallback(() => {
    setCollapsed((c) => {
      const next = !c;
      localStorage.setItem(COLLAPSED_KEY, String(next));
      return next;
    });
  }, []);

  // Close mobile drawer on Escape
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && mobileOpen) onMobileClose();
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [mobileOpen, onMobileClose]);

  const handleNavClick = useCallback(() => {
    // Close mobile drawer on navigation
    if (window.innerWidth < 1024) onMobileClose();
  }, [onMobileClose]);

  return (
    <>
      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          className="sidebar-overlay"
          onClick={onMobileClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`sidebar${collapsed ? ' sidebar--collapsed' : ''}${
          mobileOpen ? ' sidebar--mobile-open' : ''
        }`}
        aria-label="Main navigation"
      >
        {/* Brand */}
        <div className="sidebar-brand">
          <div className="sidebar-logo-mark" aria-hidden="true">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <rect width="28" height="28" rx="8" fill="url(#sidebar-logo-grad)" />
              <path
                d="M6 20L11 11L15 17L18 13L22 20"
                stroke="white"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <defs>
                <linearGradient id="sidebar-logo-grad" x1="0" y1="0" x2="28" y2="28">
                  <stop stopColor="#4f46e5" />
                  <stop offset="1" stopColor="#06b6d4" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          {!collapsed && (
            <span className="sidebar-brand-name">Auto Analyst</span>
          )}
        </div>

        {/* Navigation */}
        <nav className="sidebar-nav" aria-label="Application navigation">
          {NAV_GROUPS.map(({ group, items }) => (
            <div key={group} className="sidebar-group">
              {!collapsed && (
                <p className="sidebar-group-label" aria-hidden="true">
                  {group}
                </p>
              )}
              <ul className="sidebar-list" role="list">
                {items.map(({ label, to, Icon, comingSoon }) => (
                  <li key={to}>
                    <NavLink
                      to={to}
                      className={({ isActive }) =>
                        `sidebar-item${isActive ? ' sidebar-item--active' : ''}${
                          comingSoon ? ' sidebar-item--soon' : ''
                        }`
                      }
                      onClick={handleNavClick}
                      aria-label={collapsed ? label : undefined}
                      title={collapsed ? label : undefined}
                    >
                      <span className="sidebar-item-icon">
                        <Icon size={18} aria-hidden />
                      </span>
                      {!collapsed && (
                        <span className="sidebar-item-label">{label}</span>
                      )}
                      {!collapsed && comingSoon && (
                        <span className="sidebar-soon-badge" aria-label="Coming soon">
                          Soon
                        </span>
                      )}
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>

        {/* Collapse toggle — desktop only */}
        <button
          className="sidebar-collapse-btn"
          onClick={toggleCollapsed}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          type="button"
        >
          {collapsed ? (
            <IconChevronRight size={16} aria-hidden />
          ) : (
            <IconChevronLeft size={16} aria-hidden />
          )}
        </button>
      </aside>
    </>
  );
}
