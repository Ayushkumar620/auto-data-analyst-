# Frontend Status — Auto Data Analyst Platform

**Status Date:** 2026-08-25  
**Current Milestone:** Frontend Phase 1 (Foundation & Navigation) — COMPLETE  
**Build Status:** PASSING (`tsc` + `vite build` clean)

---

## 1. Current Shell Architecture

The application now uses a unified, glassmorphic **`AppShell`** layout composed of:

- **Sidebar (`Sidebar.tsx`)**:
  - Desktop: Sticky collapsible navigation (expanded: `240px`, collapsed: `64px`), state persisted in `localStorage` (`sidebar_collapsed`).
  - Mobile (<1024px): Off-canvas drawer with backdrop overlay, Escape key listener, auto-close on route transition.
  - Grouped navigation structure: WORKSPACE, INTELLIGENCE, OUTPUT, PROJECT, SYSTEM.
  - Zero external icon libraries: 100% lightweight custom SVG icons (`Icons.tsx`).
- **TopBar (`TopBar.tsx`)**:
  - Sticky glassmorphic header with mobile hamburger trigger.
  - Active page breadcrumb indicator.
  - Contextual dataset badge (syncs live with `DatasetContext`, shows row/column metrics when a dataset is active).
  - Notifications placeholder (disabled).
  - Accessible user profile dropdown menu with sign-out action.
- **Global Context Providers (`main.tsx`)**:
  1. `AuthProvider` — User session & JWT state.
  2. `NotificationProvider` — Global toast notification system with auto-dismiss and color status channels.
  3. `DatasetProvider` — Active dataset profile & file name persistence for cross-page analytical context.

---

## 2. Navigation & Route Hierarchy

| Route | Label | Group | Status | Component |
|---|---|---|---|---|
| `/overview` | Overview | WORKSPACE | Active (Default) | `OverviewPage.tsx` |
| `/analyst` | Analyst | WORKSPACE | Active (Phase 1 Entry) | `AnalystPage.tsx` |
| `/datasets` | Datasets | WORKSPACE | Planned (Phase 2) | `ComingSoonPage.tsx` |
| `/analyses` | Analyses | WORKSPACE | Planned (Phase 2) | `ComingSoonPage.tsx` |
| `/upload` | Insights | INTELLIGENCE | Active (Existing) | `UploadPage.tsx` |
| `/models` | Models | INTELLIGENCE | Planned (Phase 3) | `ComingSoonPage.tsx` |
| `/forecasts` | Forecasts | INTELLIGENCE | Planned (Phase 5) | `ComingSoonPage.tsx` |
| `/monitoring` | Monitoring | INTELLIGENCE | Planned (Phase 6) | `ComingSoonPage.tsx` |
| `/reports` | Reports | OUTPUT | Planned (Phase 7) | `ComingSoonPage.tsx` |
| `/projects` | Projects | PROJECT | Active (Existing) | `ProjectsPage.tsx` |
| `/projects/:projectId` | Project View | PROJECT | Active (Existing) | `ProjectViewPage.tsx` |
| `/workspaces` | Workspaces | PROJECT | Planned (Phase 2) | `ComingSoonPage.tsx` |
| `/settings` | Settings | SYSTEM | Planned (Phase 8) | `ComingSoonPage.tsx` |
| `/chat` | Command Studio | — | Active (Existing) | `ChatPage.tsx` |
| `/profile` | Profile | — | Active (Existing) | `ProfilePage.tsx` |
| `/dashboard` | Dashboard | — | Active (Legacy fallback) | `DashboardPage.tsx` |

---

## 3. Reusable UI Components

Located in `src/components/`:

- **Layout Primitives (`PageContainer.tsx`)**:
  - `PageContainer`: Unified max-width wrapper with responsive padding.
  - `PageHeader`: Structured header with eyebrow, title, subtitle, and action slot.
  - `SectionHeader`: Section-level header with action support.
  - `Card`: Glassmorphic elevated card with hover lift.
  - `GlassPanel`: Subtle backdrop-blur panel for nested content.
- **UI State Components (`src/components/ui/`)**:
  - `Icons.tsx`: Complete Feather-style SVG icon suite.
  - `Spinner.tsx`: SVG arc animation spinner.
  - `EmptyState.tsx`: Accessible empty state with icon, title, description, and action CTA.
  - `ErrorState.tsx`: Friendly error message with retry button and expandable technical details.
  - `LoadingState.tsx`: `LoadingSpinner`, `SkeletonCard` with shimmer, `SkeletonTable`.
  - `NotificationToast.tsx`: Toast container supporting success, error, warning, and info messages.

---

## 4. Design System & Styling

- **Design Language**: Dark-first glassmorphism with radial background glow and indigo/cyan primary accents.
- **CSS Architecture**:
  - `shell.css`: Layout, navigation, page components, animations, and responsive media queries.
  - `index.css`: Design tokens, typography variables, global form and button primitives.
  - `upload-page.css`: Preserved for existing Upload page functionality.
- **Zero Third-Party UI Frameworks**: Pure CSS custom properties (`var(--primary)`, `var(--panel)`, `var(--ink)`, `var(--shadow-glow)`, etc.).

---

## 5. Responsive Behavior

Tested and verified across key viewport tiers:
- **Desktop (1440px / 1280px)**: Full sidebar layout, multi-column KPI grids, side-by-side overview panels.
- **Laptop / Tablet Landscape (1024px)**: Collapsible sidebar, responsive cards.
- **Tablet Portrait (768px)**: Off-canvas sidebar drawer with backdrop, 2-column KPI strip.
- **Mobile (480px / 375px)**: Single column layouts, stacked headers, touch-friendly touch targets (min 44px), zero horizontal overflow.

---

## 6. Known Issues / Next Phase Priorities

1. **Phase 2 (Workspaces & Dataset Management)**:
   - Build out dedicated `/datasets` and `/analyses` explorers.
   - Wire dataset selector to `DatasetContext` globally.
2. **Phase 3 (Model Registry & Leaderboard)**:
   - Build UI for `/models` connecting to `GET /api/v1/models` and prediction endpoints.
