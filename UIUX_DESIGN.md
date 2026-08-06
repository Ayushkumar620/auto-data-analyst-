# 🎨 Auto Data Analyst Agent (ADAA) — UI/UX Design Specification

> **Design goal:** Every screen designed before development starts, so
> implementation is faster, more consistent, and matches user expectations.

This document defines the complete UI/UX for the ADAA platform. It is the
blueprint to be used in Figma and by developers during implementation.

---

## 1. Design System

### 1.1 Visual Identity
- **Brand color:** Deep indigo `#1a1a2e` with accent gradient (blue → pink → green)
- **Light/Dark mode:** Dark-first ("analyst terminal" feel), with light toggle
- **Typography:** Inter (UI) + JetBrains Mono (data/code)
- **Radius:** 12–16px cards, 8px buttons
- **Spacing:** 4px base grid

### 1.2 UI Components (Shadcn UI)
Buttons, inputs, cards, badges, tables, tabs, modals, toasts, skeletons,
tooltips, dropdowns, data tables, charts wrappers.

### 1.3 Navigation
- **Sidebar** (collapsible): Dashboard, Data, Analysis, Reports, Settings
- **Top bar**: search, notifications, user menu, theme toggle

---

## 2. Screens

### 2.1 Landing Page
- Hero with tagline: *"Upload your data. Ask questions. Get business insights in minutes."*
- Product value props (auto-clean, AI insights, charts, forecasts, reports)
- "Upload a file" CTA + "Get Started" (signup)
- Perks / metrics strip, testimonials, footer

### 2.2 Login & Signup
- **Login:** email + password, "forgot password", social login (optional)
- **Signup:** name/email/password, role selection (Analyst/Viewer)
- Validation, error toasts, loading states

### 2.3 Main Dashboard
- KPI cards: total datasets, analyses run, reports, storage used
- Recent activity feed
- Quick actions: "Upload Data", "New Analysis", "Run Report"
- Charts preview (recent dataset trends)

### 2.4 Dataset Upload Page
- **Drag-and-drop zone** + "Browse files"
- Supported formats badges (CSV, Excel, PDF, JSON, SQL...)
- Progress bar during upload
- Uploaded dataset cards: name, rows, columns, type, date, actions
- **Connect DB** tab (SQL connection form)

### 2.5 Analysis Workspace
- Dataset selector (left panel)
- Command input bar: type a command OR ask a question (chat-style)
- Quick command chips: summary, describe, nulls, correlation, chart, clean, insights, anomalies, forecast, report
- Results canvas (right): renders tables, charts, stats, insights dynamically
- **Tabs:** Overview | Data | Charts | Insights | ML | Report

### 2.6 AI Chat Interface
- Chat panel (right side or embedded)
- Message bubbles: user questions + AI answers (with tables/charts inline)
- Suggested question chips ("Which region generated highest profit?", "Predict next quarter sales")
- Streaming/typing indicator
- Context-aware: knows the active dataset

### 2.7 Reports Page
- List of generated reports (PDF/Excel/PPT)
- Report preview card: title, dataset, date, format badge
- Download / share / delete actions
- "Generate Report" CTA

### 2.8 User Profile & Settings
- Profile info: name, email, avatar, role
- Change password
- Preferences: theme, default view, language
- API keys (optional LLM config)
- Notification settings

### 2.9 Admin Panel
- User management: list, roles, activate/deactivate
- Usage/storage analytics
- System health
- Audit log viewer

### 2.10 Mobile-Responsive Layouts
- Collapsible sidebar → bottom nav
- Stacked cards on small screens
- Touch-friendly buttons & inputs
- Responsive tables (horizontal scroll)

---

## 3. User Flows

### 3.1 First-run (Get Value Fast)
```
Sign up → Upload CSV → Auto-clean → Auto-EDA → See dashboard → Ask a question → Download report
```

### 3.2 Upload → Report
```
Upload → (auto) Cleaning → EDA → Insights → (optional) Forecast → Generate Report → Download
```

### 3.3 Ask a Question
```
Open Chat → Select dataset → Ask in natural language → AI routes to agent → Answer + charts inline
```

---

## 4. Empty / Loading / Error States
- **Empty:** friendly placeholder with CTA (e.g., "No datasets yet — upload one")
- **Loading:** skeletons + spinner
- **Error:** inline error box with retry + suggested actions

---

## 5. Accessibility & Consistency
- WCAG AA contrast
- Keyboard navigation
- Consistent spacing, typography, and color tokens
- Labels + helpful microcopy on all forms

---

## 6. Figma Handoff Checklist
- [ ] Design tokens (colors, typography, spacing) as Figma variables
- [ ] Component library mirrored to Shadcn UI
- [ ] All 10 screens above, desktop + mobile
- [ ] Interactive prototypes for key flows
- [ ] Asset export (icons, logos, chart examples)

---

## 7. Next Steps After UI Approval
- Database schema
- API specification
- Start backend (`backend/`) and frontend (`frontend/`) scaffolding

---

*This UI/UX spec is the blueprint for the Figma design and frontend implementation.*
