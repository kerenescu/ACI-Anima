@echo off
setlocal
cd /d "%~dp0"
set "LAB_PY=%~dp0.venv\Scripts\python.exe"
if exist "%LAB_PY%" goto run
set "LAB_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%LAB_PY%" goto run
set "LAB_PY=python"
:run
"%LAB_PY%" -c "import numpy; from PIL import Image" >nul 2>&1
if errorlevel 1 goto missing
"%LAB_PY%" server.py --open
if errorlevel 1 pause
exit /b
:missing
echo Este necesar Python 3.10 sau mai nou, cu NumPy si Pillow.
echo Din acest folder ruleaza:
echo   python -m venv .venv
echo   .venv\Scripts\python -m pip install -r requirements.txt
echo Apoi deschide din nou START_PROIECT.cmd.
pause
