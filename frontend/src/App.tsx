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

// Phase 2 Live Pages
import DatasetsPage from './pages/DatasetsPage';
import DatasetWorkspacePage from './pages/DatasetWorkspacePage';
import AnalysesPage from './pages/AnalysesPage';
import AnalysisDetailPage from './pages/AnalysisDetailPage';
import WorkspacesPage from './pages/WorkspacesPage';

// Phase 3 Live Pages
import ModelRegistryPage from './pages/ModelRegistryPage';
import ModelDetailPage from './pages/ModelDetailPage';

// Phase 5 Live Pages
import ForecastsPage from './pages/ForecastsPage';

// Future Phase Placeholders
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
        {/* Core workspace & analytics routes */}
        <Route path="/overview" element={<OverviewPage />} />
        <Route path="/analyst" element={<AnalystPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/projects/:projectId" element={<ProjectViewPage />} />
        <Route path="/profile" element={<ProfilePage />} />
        <Route path="/dashboard" element={<DashboardPage />} />

        {/* Phase 2: Datasets & Workspace Management */}
        <Route path="/datasets" element={<DatasetsPage />} />
        <Route path="/datasets/:datasetId" element={<DatasetWorkspacePage />} />
        <Route path="/analyses" element={<AnalysesPage />} />
        <Route path="/analyses/:analysisId" element={<AnalysisDetailPage />} />
        <Route path="/workspaces" element={<WorkspacesPage />} />

        {/* Phase 3: Model Registry & Leaderboard */}
        <Route path="/models" element={<ModelRegistryPage />} />
        <Route path="/models/:modelId" element={<ModelDetailPage />} />

        {/* Phase 5: Autonomous Forecasting & What-If */}
        <Route path="/forecasts" element={<ForecastsPage />} />

        {/* Coming soon stubs (Phase 6–8) */}
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
