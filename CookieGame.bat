@echo off
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" -m cookierun_bot.agents.controller
if errorlevel 1 (
  echo.
  echo Launch failed. Run install.ps1 first, then try CookieGame.bat again.
  pause
)
