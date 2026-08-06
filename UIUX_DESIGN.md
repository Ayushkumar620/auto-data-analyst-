# 🎨 Auto Data Analyst Agent (ADAA) — UI/UX Design Specification

> **Design goal:** Every screen designed before development starts, so
> implementation is faster, more consistent, and matches user expectations.

---

## 1. Design Tokens

### Color Palette
| Token | Color | Hex |
|-------|-------|-----|
| **Primary** | Blue | `#2563EB` |
| **Success** | Green | `#22C55E` |
| **Warning** | Orange | `#F59E0B` |
| **Error** | Red | `#EF4444` |
| **Background** | White | `#FFFFFF` |
| **Dark Mode** | Slate | `#0F172A` |

### Typography
| Style | Font |
|-------|------|
| **Headings** | Inter Bold |
| **Body** | Inter Regular |
| **Code/Data** | JetBrains Mono |

### Component Library
Create reusable components for:
- Buttons
- Cards
- Tables
- Charts
- Inputs
- Modals
- Notifications
- Loaders
- File Upload

> Reusable components ensure **consistency across the application**.

---

## 2. Design Principles

Every page should answer three questions immediately:
1. **What data am I looking at?**
2. **What insights matter most?**
3. **What action should I take next?**

**Avoid clutter.** The AI should highlight the most important findings first,
while still allowing users to explore deeper analyses.

---

## 3. Sidebar Navigation

```
Dashboard
Projects
Upload
Analysis
Charts
AI Chat
Forecasts
Reports
Settings
```

---

## 4. Screens

### Screen 1 – Landing Page
**Purpose:** Explain the product and encourage sign-up.

**Sections:**
- Navigation bar
- Hero section
- Features
- Demo video
- Customer testimonials (later)
- Pricing (later)
- Footer

**Hero section:**
```
----------------------------------------
Auto Data Analyst Agent

Upload Any Dataset
Get Business Insights Instantly

[Try Free]   [Watch Demo]

Image: Dashboard Preview
----------------------------------------
```

---

### Screen 2 – Login / Signup
Simple form:
- Email
- Password
- Login
- Google Login
- GitHub Login

---

### Screen 3 – Dashboard
This is the main workspace.
```
-----------------------------------------------------
Logo     Projects     Search     Profile
-----------------------------------------------------

Recent Projects
+ New Analysis

Quick Actions
Recent Reports
AI Suggestions
Storage Usage
Notifications
```

---

### Screen 4 – Create Project
**Fields:**
- Project Name
- Description
- Industry
- Language
- **Create**

---

### Screen 5 – Upload Dataset
Drag and Drop.
**Supported formats:** CSV, Excel, PDF, SQL, JSON

```
-------------------------
Drop Files Here
or Browse

CSV  Excel  PDF
-------------------------
```

---

### Screen 6 – Dataset Overview
After upload, show:
- Rows
- Columns
- Missing Values
- Duplicate Rows
- Data Types
- Sample Data

```
Rows: 12,000
Columns: 18
Missing Values: 63
Duplicates: 12
```

---

### Screen 7 – AI Analysis
Automatically generated **cards**:
- Data Quality
- Business Overview
- Recommendations
- Potential Problems
- Interesting Patterns

---

### Screen 8 – Charts
**Tabs:**
- Overview
- Distribution
- Correlation
- Time Series
- Categories
- Custom

**Interactive chart types:**
- Bar
- Pie
- Line
- Heatmap
- Histogram
- Box Plot
- Scatter

---

### Screen 9 – AI Chat
**The heart of the application.**

```
Ask anything...

Why are sales dropping?
What products perform best?
Predict next month.
Generate report.
```

> Chat looks like ChatGPT but has **access to the uploaded dataset**.

---

### Screen 10 – Reports
**Generate:**
- PDF
- PowerPoint
- Excel

Preview before download.

---

### Screen 11 – Forecasting
**Cards:**
- Revenue Prediction
- Demand Forecast
- Risk Score
- Growth Projection
- Confidence Interval

---

### Screen 12 – Settings
- Theme
- Language
- API Keys
- Notifications
- Team Members
- Billing (future)

---

### Screen 13 – Admin Panel
- Users
- Projects
- Storage
- Analytics
- System Logs
- Model Usage

---

## 5. What Comes Next (Step 4)

Now that the product is fully planned and the UI is designed, we'll move into
**technical planning** by creating:

- [ ] The complete **database schema**
- [ ] **API specifications**
- [ ] **Folder structure**
- [ ] **Development environment**
- [ ] **GitHub repository structure**
- [ ] **Sprint planning**

After that, we'll start building the **MVP** one module at a time, beginning
with **authentication and project setup**.

---

*This UI/UX spec is the blueprint for the Figma design and frontend implementation.*
