import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import TopBar from './TopBar';
import NotificationToast from '../ui/NotificationToast';

export default function AppShell() {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="app-shell">
      <Sidebar
        mobileOpen={mobileOpen}
        onMobileClose={() => setMobileOpen(false)}
      />
      <div className="app-shell-main">
        <TopBar onMenuClick={() => setMobileOpen((o) => !o)} />
        <main
          className="app-shell-content"
          id="main-content"
          tabIndex={-1}
        >
          <Outlet />
        </main>
      </div>
      <NotificationToast />
    </div>
  );
}
