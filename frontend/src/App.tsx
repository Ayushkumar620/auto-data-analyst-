import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { RequireAuth, RequireGuest } from './auth/ProtectedRoute';
import AppShell from './components/layout/AppShell';

// Auth
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';

// Authenticated pages
import OverviewPage from './pages/OverviewPage';
import AnalystPage from './pages/AnalystPage';
import UploadPage from './pages/UploadPage';
import ChatPage from './pages/ChatPage';
import ProjectsPage from './pages/ProjectsPage';
import ProjectViewPage from './pages/ProjectViewPage';
import ProfilePage from './pages/ProfilePage';
import DashboardPage from './pages/DashboardPage';
import ComingSoonPage from './pages/ComingSoonPage';

export default function App() {
  return (
    <Routes>
      {/* Public */}
      <Route
        path="/login"
        element={
          <RequireGuest>
            <LoginPage />
          </RequireGuest>
        }
      />
      <Route
        path="/register"
        element={
          <RequireGuest>
            <RegisterPage />
          </RequireGuest>
        }
      />

      {/* Authenticated shell */}
      <Route
        element={
          <RequireAuth>
            <AppShell />
          </RequireAuth>
        }
      >
        {/* Core routes */}
        <Route path="/overview" element={<OverviewPage />} />
        <Route path="/analyst" element={<AnalystPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/projects/:projectId" element={<ProjectViewPage />} />
        <Route path="/profile" element={<ProfilePage />} />
        <Route path="/dashboard" element={<DashboardPage />} />

        {/* Coming soon stubs (Phase 2–8) */}
        <Route
          path="/datasets"
          element={
            <ComingSoonPage
              title="Datasets"
              phase="Phase 2"
              description="Browse, manage, and compare all uploaded datasets in one place."
            />
          }
        />
        <Route
          path="/analyses"
          element={
            <ComingSoonPage
              title="Analyses"
              phase="Phase 2"
              description="View all past autonomous analyses, evidence chains, and execution graphs."
            />
          }
        />
        <Route
          path="/models"
          element={
            <ComingSoonPage
              title="Model Registry"
              phase="Phase 3"
              description="Browse trained models, view metrics and loss curves, and run live inference."
            />
          }
        />
        <Route
          path="/forecasts"
          element={
            <ComingSoonPage
              title="Forecasts"
              phase="Phase 5"
              description="Autonomous time-series forecasting and what-if scenario analysis."
            />
          }
        />
        <Route
          path="/monitoring"
          element={
            <ComingSoonPage
              title="Monitoring"
              phase="Phase 6"
              description="Data drift detection and model performance degradation monitoring."
            />
          }
        />
        <Route
          path="/reports"
          element={
            <ComingSoonPage
              title="Reports"
              phase="Phase 7"
              description="Generate executive PDF and slide deck reports from your analyses."
            />
          }
        />
        <Route
          path="/workspaces"
          element={
            <ComingSoonPage
              title="Workspaces"
              phase="Phase 2"
              description="Manage collaborative workspaces and team dataset access."
            />
          }
        />
        <Route
          path="/settings"
          element={
            <ComingSoonPage
              title="Settings"
              phase="Phase 8"
              description="Application settings, preferences, and integrations."
            />
          }
        />
      </Route>

      {/* Legacy redirect — /dashboard keeps working */}
      <Route path="/" element={<Navigate to="/overview" replace />} />

      {/* Catch-all */}
      <Route path="*" element={<Navigate to="/overview" replace />} />
    </Routes>
  );
}
