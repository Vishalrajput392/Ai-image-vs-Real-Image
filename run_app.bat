@echo off
title AI vs Real Detector Launcher
cd /d C:\Work\Project
call venv\Scripts\activate.bat

echo Starting Backend Server...
start "" /b venv\Scripts\uvicorn.exe backend.main:app --port 8000

echo Waiting for model to initialize...
timeout /t 3 /nobreak >nul

echo Starting Web UI...
start "" /b venv\Scripts\streamlit.exe run frontend/app.py --server.headless=false
