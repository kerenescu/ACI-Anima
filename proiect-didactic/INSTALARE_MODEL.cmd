@echo off
setlocal
cd /d "%~dp0"
set "LAB_BASE_PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%LAB_BASE_PY%" set "LAB_BASE_PY=python"
if exist ".venv\Scripts\python.exe" goto install
"%LAB_BASE_PY%" -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
".venv\Scripts\python.exe" setup_model.py
if errorlevel 1 goto failed
echo Modelul este pregatit. Inchide serverul vechi si deschide START_PROIECT.cmd.
pause
exit /b 0
:failed
echo Instalarea nu s-a terminat. Detaliile erorii sunt afisate mai sus.
pause
exit /b 1
