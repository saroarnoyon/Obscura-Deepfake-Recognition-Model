@echo off
TITLE Deepfake Detector Launcher
cd /d "%~dp0"

echo Starting Deepfake Recognition System...
python -m streamlit run app.py

if %errorlevel% neq 0 (
    echo Error: App crashed.
    pause
)