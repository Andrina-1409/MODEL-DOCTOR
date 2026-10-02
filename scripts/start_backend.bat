@echo off
cd /d "%~dp0.."
python -m pip install -r requirements.txt
python -m uvicorn src.api:app --reload --port 8000
pause
