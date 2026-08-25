import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { RequireAuth, RequireGuest } from './auth/ProtectedRoute';
import AppShell from './components/layout/AppShell';
import ErrorBoundary from './components/ui/ErrorBoundary';

// Auth Pages
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';

// Core Workspace Pages
import OverviewPage from './pages/OverviewPage';
import AnalystPage from './pages/AnalystPage';
import UploadPage from './pages/UploadPage';
import ChatPage from './pages/ChatPage';
import ProjectsPage from './pages/ProjectsPage';
import ProjectViewPage from './pages/ProjectViewPage';
import ProfilePage from './pages/ProfilePage';
import DashboardPage from './pages/DashboardPage';

// Phase 2: Datasets & Workspace Management
import DatasetsPage from './pages/DatasetsPage';
import DatasetWorkspacePage from './pages/DatasetWorkspacePage';
import AnalysesPage from './pages/AnalysesPage';
import AnalysisDetailPage from './pages/AnalysisDetailPage';
import WorkspacesPage from './pages/WorkspacesPage';

// Phase 3: Model Registry & Leaderboard
import ModelRegistryPage from './pages/ModelRegistryPage';
import ModelDetailPage from './pages/ModelDetailPage';

// Phase 5: Autonomous Forecasting & What-If
import ForecastsPage from './pages/ForecastsPage';

// Phase 6: Model Monitoring & Data Drift
import MonitoringPage from './pages/MonitoringPage';
import MonitoringDetailPage from './pages/MonitoringDetailPage';

// Phase 7: Reports & Decision Deliverables
import ReportsPage from './pages/ReportsPage';
import ReportDetailPage from './pages/ReportDetailPage';

// Phase 8: System Settings & Error Handling
import SettingsPage from './pages/SettingsPage';
import NotFoundPage from './pages/NotFoundPage';

export default function App() {
  return (
    <ErrorBoundary>
      <Routes>
        {/* Public Routes */}
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

        {/* Authenticated Application Shell */}
        <Route
          element={
            <RequireAuth>
              <AppShell />
            </RequireAuth>
          }
        >
          {/* Core Navigation */}
          <Route path="/overview" element={<OverviewPage />} />
          <Route path="/analyst" element={<AnalystPage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/projects" element={<ProjectsPage />} />
          <Route path="/projects/:projectId" element={<ProjectViewPage />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="/dashboard" element={<DashboardPage />} />

          {/* Phase 2: Datasets & Workspaces */}
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/datasets/:datasetId" element={<DatasetWorkspacePage />} />
          <Route path="/analyses" element={<AnalysesPage />} />
          <Route path="/analyses/:analysisId" element={<AnalysisDetailPage />} />
          <Route path="/workspaces" element={<WorkspacesPage />} />

          {/* Phase 3: Model Registry */}
          <Route path="/models" element={<ModelRegistryPage />} />
          <Route path="/models/:modelId" element={<ModelDetailPage />} />

          {/* Phase 5: Forecasting & Scenarios */}
          <Route path="/forecasts" element={<ForecastsPage />} />

          {/* Phase 6: Monitoring & Drift */}
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/monitoring/:modelId" element={<MonitoringDetailPage />} />

          {/* Phase 7: Reports & Outputs */}
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/reports/:reportId" element={<ReportDetailPage />} />

          {/* Phase 8: Settings */}
          <Route path="/settings" element={<SettingsPage />} />
        </Route>

        {/* Root Redirect */}
        <Route path="/" element={<Navigate to="/overview" replace />} />

        {/* Catch-all 404 Route */}
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </ErrorBoundary>
  );
}
