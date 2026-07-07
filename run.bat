@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Potens AIML RAG Runner

if /I "%~1"=="help" goto SHOW_HELP

call :BOOTSTRAP
if errorlevel 1 goto FAIL

if /I "%~1"=="streamlit" goto RUN_STREAMLIT
if /I "%~1"=="api" goto RUN_API
if /I "%~1"=="ingest" goto RUN_INGEST
if /I "%~1"=="reset" goto RUN_RESET
if /I "%~1"=="eval" goto RUN_EVAL

:MENU
cls
echo ==========================================
echo   Potens AIML RAG - One Click Runner
echo ==========================================
echo.
echo 1. Start Streamlit dashboard
echo 2. Start FastAPI server
echo 3. Ingest papers
echo 4. Reset vector DB and re-ingest
echo 5. Run evaluation set
echo 6. Open .env
echo 7. Help
echo 0. Exit
echo.
set /p "CHOICE=Choose an option: "

if "%CHOICE%"=="1" goto RUN_STREAMLIT
if "%CHOICE%"=="2" goto RUN_API
if "%CHOICE%"=="3" goto RUN_INGEST
if "%CHOICE%"=="4" goto RUN_RESET
if "%CHOICE%"=="5" goto RUN_EVAL
if "%CHOICE%"=="6" goto OPEN_ENV
if "%CHOICE%"=="7" goto SHOW_HELP_PAUSE
if "%CHOICE%"=="0" goto END
goto MENU

:BOOTSTRAP
echo.
echo Checking Python environment...
set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 set "PY_CMD=py -3"

if not defined PY_CMD (
  where python >nul 2>nul
  if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
  echo Python was not found.
  echo Install Python 3.10+ and enable "Add Python to PATH".
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY_CMD% -m venv .venv
  if errorlevel 1 exit /b 1
)

set "PY=.venv\Scripts\python.exe"

if not exist ".venv\.deps-installed" (
  echo Installing dependencies. First run can take several minutes...
  "%PY%" -m pip install --upgrade pip
  if errorlevel 1 exit /b 1
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
  echo installed > ".venv\.deps-installed"
) else (
  echo Dependencies already installed.
)

if not exist ".env" (
  echo Creating .env from .env.example...
  copy ".env.example" ".env" >nul
)
exit /b 0

:WARN_KEYS
findstr /I /C:"your_gemini_key_here" ".env" >nul 2>nul
if not errorlevel 1 (
  echo.
  echo No LLM key configured. The app will still run in extractive fallback mode.
)
findstr /I /C:"your_groq_key_here" ".env" >nul 2>nul
if not errorlevel 1 (
  echo Add a Gemini or Groq key in .env if you want generated answers.
)
exit /b 0

:RUN_STREAMLIT
call :WARN_KEYS
call :MAYBE_INGEST
if errorlevel 1 goto FAIL
echo.
echo Starting Streamlit dashboard...
echo Browser should open automatically. If not, open http://localhost:8501
"%PY%" -m streamlit run streamlit_app.py --server.headless false
goto END_PAUSE

:RUN_API
call :WARN_KEYS
call :MAYBE_INGEST
if errorlevel 1 goto FAIL
echo.
echo Starting FastAPI server...
echo Open http://127.0.0.1:8000/docs for API docs.
"%PY%" -m uvicorn app:app --host 127.0.0.1 --port 8000
goto END_PAUSE

:RUN_INGEST
call :REQUIRE_PAPERS
if errorlevel 1 goto FAIL
echo.
echo Ingesting papers...
"%PY%" ingest.py
if errorlevel 1 goto FAIL
goto END_PAUSE

:RUN_RESET
call :REQUIRE_PAPERS
if errorlevel 1 goto FAIL
echo.
echo Resetting vector DB and re-ingesting papers...
"%PY%" ingest.py --reset
if errorlevel 1 goto FAIL
goto END_PAUSE

:RUN_EVAL
echo.
echo Running evaluation set...
"%PY%" eval\run_eval.py
if errorlevel 1 goto FAIL
goto END_PAUSE

:OPEN_ENV
start "" notepad ".env"
goto MENU

:MAYBE_INGEST
if not exist "papers" mkdir "papers"
if not exist "chroma_db" mkdir "chroma_db"
dir /b "papers\*.pdf" "papers\*.txt" >nul 2>nul
if errorlevel 1 (
  echo.
  echo No PDF/TXT files found in papers\. The app will open, but Q&A needs ingested papers.
  exit /b 0
)
if exist "chroma_db\chroma.sqlite3" (
  echo Vector database found. Skipping auto-ingestion.
  exit /b 0
)
echo Papers found but vector database is missing. Running ingestion...
"%PY%" ingest.py --reset
exit /b %errorlevel%

:REQUIRE_PAPERS
if not exist "papers" mkdir "papers"
dir /b "papers\*.pdf" "papers\*.txt" >nul 2>nul
if errorlevel 1 (
  echo No PDF/TXT files found in papers\.
  echo Add research PDFs/TXT files first.
  exit /b 1
)
exit /b 0

:SHOW_HELP
echo Usage:
echo   run.bat             Show menu
echo   run.bat streamlit   Start Streamlit dashboard
echo   run.bat api         Start FastAPI server
echo   run.bat ingest      Ingest papers
echo   run.bat reset       Reset vector DB and re-ingest
echo   run.bat eval        Run evaluation set
exit /b 0

:SHOW_HELP_PAUSE
call :SHOW_HELP
pause
goto MENU

:FAIL
echo.
echo Something went wrong. Read the message above, fix it, then run this file again.
pause
exit /b 1

:END_PAUSE
echo.
echo Task finished or app stopped.
pause

:END
exit /b 0
