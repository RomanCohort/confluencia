@echo off
chcp 65001 >nul 2>&1
title Confluencia Platform Launcher

echo ==========================================
echo   Confluencia Platform Launcher
echo ==========================================
echo.
echo Select Module:
echo   1. Drug Module (Small Molecules)
echo   2. circRNA Module (RNA Vaccines)
echo   3. Exit
echo.

set /p choice="Enter choice [1-3]: "

if "%choice%"=="1" (
    echo.
    echo Starting Drug Module...
    echo Location: confluencia-2.0-drug/app_drug.py
    cd confluencia-2.0-drug
    streamlit run app_drug.py
) else if "%choice%"=="2" (
    echo.
    echo Starting circRNA Module...
    echo Location: confluencia_circrna/app.py
    cd confluencia_circrna
    streamlit run app.py
) else if "%choice%"=="3" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice. Please enter 1, 2, or 3.
    pause
    exit /b 1
)