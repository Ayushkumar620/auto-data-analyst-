import React, { useCallback, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../auth/authContext';
import { useDataset } from '../../context/DatasetContext';
import {
  IconMenu,
  IconUser,
  IconChevronRight,
  IconDatabase,
  IconBell,
} from '../ui/Icons';

const ROUTE_LABELS: Record<string, string> = {
  '/overview': 'Overview',
  '/analyst': 'Analyst',
  '/datasets': 'Datasets',
  '/analyses': 'Analyses',
  '/upload': 'Upload',
  '/insights': 'Insights',
  '/models': 'Models',
  '/forecasts': 'Forecasts',
  '/monitoring': 'Monitoring',
  '/reports': 'Reports',
  '/projects': 'Projects',
  '/workspaces': 'Workspaces',
  '/settings': 'Settings',
  '/chat': 'Command Studio',
  '/profile': 'Profile',
  '/dashboard': 'Dashboard',
};

type TopBarProps = {
  onMenuClick: () => void;
};

export default function TopBar({ onMenuClick }: TopBarProps) {
  const { user, logout } = useAuth();
  const { profile, fileName } = useDataset();
  const location = useLocation();
  const navigate = useNavigate();
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const pageLabel = ROUTE_LABELS[location.pathname] ?? 'Auto Data Analyst';

  const handleLogout = useCallback(() => {
    logout();
    setUserMenuOpen(false);
    navigate('/login');
  }, [logout, navigate]);

  const handleProfile = useCallback(() => {
    navigate('/profile');
    setUserMenuOpen(false);
  }, [navigate]);

  // Close dropdown on outside click
  const handleBlur = useCallback((e: React.FocusEvent<HTMLDivElement>) => {
    if (!menuRef.current?.contains(e.relatedTarget as Node)) {
      setUserMenuOpen(false);
    }
  }, []);

  return (
    <header className="topbar" role="banner">
      {/* Left — menu toggle + breadcrumb */}
      <div className="topbar-left">
        <button
          className="topbar-menu-btn"
          onClick={onMenuClick}
          aria-label="Toggle navigation menu"
          type="button"
        >
          <IconMenu size={20} aria-hidden />
        </button>

        <nav className="topbar-breadcrumb" aria-label="Breadcrumb">
          <span className="topbar-brand">Auto Data Analyst</span>
          <IconChevronRight size={14} className="topbar-breadcrumb-sep" aria-hidden />
          <span className="topbar-page-title" aria-current="page">
            {pageLabel}
          </span>
        </nav>
      </div>

      {/* Centre — active dataset indicator */}
      {profile && (
        <div className="topbar-dataset-bar" aria-label="Active dataset">
          <IconDatabase size={14} className="topbar-dataset-icon" aria-hidden />
          <span className="topbar-dataset-name">
            {fileName ?? profile.dataset_name}
          </span>
          <span className="topbar-dataset-meta">
            {profile.rows.toLocaleString()} rows · {profile.columns} cols
          </span>
        </div>
      )}

      {/* Right — notifications + user menu */}
      <div className="topbar-right">
        <button
          className="topbar-icon-btn"
          aria-label="Notifications — coming soon"
          title="Notifications coming in a future release"
          disabled
          type="button"
        >
          <IconBell size={18} aria-hidden />
        </button>

        <div
          className="topbar-user-menu"
          ref={menuRef}
          onBlur={handleBlur}
        >
          <button
            className="topbar-user-btn"
            onClick={() => setUserMenuOpen((o) => !o)}
            aria-label="Open user menu"
            aria-expanded={userMenuOpen}
            aria-haspopup="menu"
            type="button"
          >
            <div className="topbar-avatar" aria-hidden="true">
              {user?.username?.charAt(0).toUpperCase() ?? 'U'}
            </div>
            <span className="topbar-username">{user?.username ?? 'User'}</span>
          </button>

          {userMenuOpen && (
            <div className="topbar-dropdown" role="menu">
              <div className="topbar-dropdown-header">
                <p className="topbar-dropdown-name">{user?.username}</p>
                <p className="topbar-dropdown-email">{user?.email}</p>
              </div>
              <hr className="topbar-dropdown-divider" />
              <button
                className="topbar-dropdown-item"
                onClick={handleProfile}
                role="menuitem"
                type="button"
              >
                <IconUser size={15} aria-hidden />
                Profile
              </button>
              <hr className="topbar-dropdown-divider" />
              <button
                className="topbar-dropdown-item topbar-dropdown-item--danger"
                onClick={handleLogout}
                role="menuitem"
                type="button"
              >
                Sign out
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
