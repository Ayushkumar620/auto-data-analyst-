import React from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '../auth/authContext';

export default function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="app-layout">
      <header className="site-header">
        <div className="logo">Auto Data Analyst</div>
        <nav className="main-nav">
          <NavLink to="/dashboard" className="nav-link">Dashboard</NavLink>
          <NavLink to="/projects" className="nav-link">Projects</NavLink>
          <NavLink to="/upload" className="nav-link">Upload</NavLink>
          <NavLink to="/chat" className="nav-link">Chat</NavLink>
          <NavLink to="/profile" className="nav-link">Profile</NavLink>
        </nav>
        <div className="user-area">
          <span className="user-badge">{user?.username ?? '—'}</span>
          <button className="ghost-btn" onClick={handleLogout}>Logout</button>
        </div>
      </header>
      <main className="site-main">
        <Outlet />
      </main>
    </div>
  );
}