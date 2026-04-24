@echo off
setlocal
cd /d "%~dp0"
streamlit run epitope_frontend.py
endlocal
