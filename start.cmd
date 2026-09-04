@echo off
cd /d "%~dp0"
echo Starting Model Compare. Open http://127.0.0.1:5000 once Flask is ready.
"%~dp0..\.venv\Scripts\python.exe" Flask.py
pause
