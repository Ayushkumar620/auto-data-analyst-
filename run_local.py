"""Unified Localhost Application Launcher.

Launches both Frontend (React UI) and Backend (FastAPI) on a single port (http://localhost:8000).
Automatically opens the browser to the Login page.
"""
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist"


def build_frontend_if_needed():
    """Ensure frontend production bundle is built."""
    if not (FRONTEND_DIST / "index.html").exists():
        print("📦 Building React frontend production assets (one-time setup)...")
        subprocess.run(["npm", "run", "build"], cwd=str(FRONTEND_DIR), check=True, shell=True)
        print("✅ Frontend build completed successfully.")


def open_browser_delayed(url: str, delay_seconds: float = 1.5):
    """Open default web browser after server starts."""
    import threading

    def _open():
        time.sleep(delay_seconds)
        print(f"\n🌐 Opening Auto Data Analyst in your web browser: {url}")
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


def main():
    print("=" * 70)
    print("✦ AUTO DATA ANALYST — UNIFIED LOCALHOST LAUNCHER")
    print("=" * 70)

    build_frontend_if_needed()

    host = "127.0.0.1"
    port = 8000
    app_url = f"http://localhost:{port}"

    print(f"\n🚀 Starting Unified Server at {app_url}...")
    print(f"👉 Login Page: {app_url}/login")
    print(f"👉 API Docs:   {app_url}/docs")
    print(f"👉 Health:     {app_url}/health\n")

    open_browser_delayed(f"{app_url}/login")

    # Launch uvicorn
    import uvicorn
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
