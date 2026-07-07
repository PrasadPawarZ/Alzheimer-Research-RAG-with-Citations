@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Potens AIML RAG Runner

if /I "%~1"=="help" goto SHOW_HELP

echo.
echo ==========================================
echo   Potens AIML RAG - One Click Runner
echo ==========================================
echo.

set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 set "PY_CMD=py -3"

if not defined PY_CMD (
  where python >nul 2>nul
  if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
  echo Python was not found.
  echo Install Python 3.10+ from https://www.python.org/downloads/
  echo Make sure "Add Python to PATH" is selected during installation.
  goto FAIL
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY_CMD% -m venv .venv
  if errorlevel 1 goto FAIL
)

set "PY=.venv\Scripts\python.exe"

if not exist ".venv\.deps-installed" (
  echo Installing dependencies. This can take several minutes on first run...
  "%PY%" -m pip install --upgrade pip
  if errorlevel 1 goto FAIL
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 goto FAIL
  echo installed > ".venv\.deps-installed"
) else (
  echo Dependencies already installed.
)

if not exist ".env" (
  echo Creating .env from .env.example...
  copy ".env.example" ".env" >nul
)

set "NEEDS_KEY=0"
findstr /I /C:"your_groq_key_here" ".env" >nul 2>nul && set "NEEDS_KEY=1"
findstr /I /C:"your_gemini_key_here" ".env" >nul 2>nul && set "NEEDS_KEY=1"

if "%NEEDS_KEY%"=="1" (
  echo.
  echo Your .env still has placeholder API keys.
  echo Add either GROQ_API_KEY or GEMINI_API_KEY before asking questions.
  echo Opening .env in Notepad now.
  start "" notepad ".env"
  echo.
  pause
)

if /I "%~1"=="api" goto RUN_API
if /I "%~1"=="ingest" goto RUN_INGEST

call :MAYBE_INGEST
if errorlevel 1 goto FAIL

echo.
echo Starting Streamlit app...
echo Browser should open automatically. If not, open http://localhost:8501
echo.
"%PY%" -m streamlit run streamlit_app.py --server.headless false
goto END

:RUN_API
call :MAYBE_INGEST
if errorlevel 1 goto FAIL
echo.
echo Starting FastAPI app...
echo Open http://127.0.0.1:8000/docs for API docs.
echo.
"%PY%" -m uvicorn app:app --reload --port 8000
goto END

:RUN_INGEST
call :RUN_INGEST_ONLY
if errorlevel 1 goto FAIL
goto END

:MAYBE_INGEST
if not exist "papers" mkdir "papers"
if not exist "chroma_db" mkdir "chroma_db"

dir /b "papers\*.pdf" "papers\*.txt" >nul 2>nul
if errorlevel 1 (
  echo.
  echo No PDF/TXT files found in the papers folder.
  echo Add research papers to papers\ and run this file again.
  echo The app will still open, but document Q&A needs ingested papers.
  exit /b 0
)

if exist "chroma_db\chroma.sqlite3" (
  echo Vector database found. Skipping ingestion.
  exit /b 0
)

echo Papers found but vector database is missing. Running ingestion...
"%PY%" ingest.py --reset
exit /b %errorlevel%

:RUN_INGEST_ONLY
if not exist "papers" mkdir "papers"
dir /b "papers\*.pdf" "papers\*.txt" >nul 2>nul
if errorlevel 1 (
  echo No PDF/TXT files found in papers\.
  echo Add files first, then run: run.bat ingest
  exit /b 1
)
echo Rebuilding vector database...
"%PY%" ingest.py --reset
exit /b %errorlevel%

:SHOW_HELP
echo Usage:
echo   run.bat          Start Streamlit app
echo   run.bat api      Start FastAPI app
echo   run.bat ingest   Rebuild vector database only
exit /b 0

:FAIL
echo.
echo Something went wrong. Read the message above, fix it, then run this file again.
pause
exit /b 1

:END
echo.
echo App stopped.
pause
