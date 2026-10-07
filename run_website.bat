@echo off
title VERITAS - SOTA Deepfake Cyber-Forensic Studio
echo ======================================================================
echo    VERITAS // World-Class Deepfake Cyber-Forensic Studio
echo    FastAPI High-Performance Engine + SOTA Multi-Modal Dual Stream
echo ======================================================================
echo.

set PYTHON_EXE="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

if exist %PYTHON_EXE% (
    echo [*] Starting VERITAS Web Server on http://localhost:8000 ...
    start "" http://localhost:8000
    %PYTHON_EXE% -m uvicorn server:app --host 0.0.0.0 --port 8000
) else (
    echo [*] Checking system python...
    start "" http://localhost:8000
    python -m uvicorn server:app --host 0.0.0.0 --port 8000
)

pause
