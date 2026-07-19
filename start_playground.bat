@echo off
cd /d "%~dp0"
echo Starting Riffusion Playground...
echo Open: http://127.0.0.1:8501/
echo.
python -m riffusion.streamlit.playground
pause
