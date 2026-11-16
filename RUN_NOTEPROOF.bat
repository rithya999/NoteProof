@echo off
cd /d "%~dp0"
if not exist ".env" (
  echo Please copy .env.example to .env and add your Gemini API key first.
  pause
  exit /b 1
)
python -m pip install -r requirements.txt
python -m streamlit run app.py
pause
