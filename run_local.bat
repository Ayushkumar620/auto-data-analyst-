@echo off
title Auto Data Analyst - Unified Localhost Server
echo ======================================================================
echo ✦ Launching Auto Data Analyst on http://localhost:8000
echo ======================================================================

if exist .\venv\Scripts\python.exe (
    .\venv\Scripts\python.exe run_local.py
) else (
    python run_local.py
)

pause
