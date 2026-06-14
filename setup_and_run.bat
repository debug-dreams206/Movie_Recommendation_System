@echo off
REM =============================================================================
REM setup_and_run.bat  —  Windows one-shot setup and run script
REM Double-click this file, or run it from Command Prompt / PowerShell.
REM =============================================================================

echo.
echo ============================================================
echo    Movie Recommendation System -- Setup and Run (Windows)
echo ============================================================
echo.

REM ── 1. Navigate to the script's directory ────────────────────────────────
cd /d %~dp0

REM ── 2. Create virtual environment ────────────────────────────────────────
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM ── 3. Activate venv ─────────────────────────────────────────────────────
call venv\Scripts\activate.bat

REM ── 4. Install dependencies ───────────────────────────────────────────────
echo Installing dependencies...
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo Dependencies installed.

REM ── 5. Run project ────────────────────────────────────────────────────────
echo.
echo Launching main.py ...
echo.
python main.py

pause
