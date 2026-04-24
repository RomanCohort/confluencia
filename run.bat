@echo off
chcp 65001 >nul 2>&1
title PyInstaller App Runtime Setup

echo =========================================
echo   PyInstaller Runtime Setup
echo =========================================
echo.

REM Step 1: Check VC++ runtimes
echo [1/2] Checking VC++ Redistributables...
echo.

set "ALL_OK=YES"

REM Check VC++ 2015-2022 x64
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo   [OK] VC++ 2015-2022 (x64)
    set "VC2015_X64=YES"
) else (
    echo   [MISSING] VC++ 2015-2022 (x64)
    set "VC2015_X64=NO"
    set "ALL_OK=NO"
)

REM Check VC++ 2015-2022 x86
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x86" /v Installed >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo   [OK] VC++ 2015-2022 (x86)
    set "VC2015_X86=YES"
) else (
    echo   [MISSING] VC++ 2015-2022 (x86)
    set "VC2015_X86=NO"
    set "ALL_OK=NO"
)

REM Check VC++ 2010 x64
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\10.0\VC\VCRedist\x64" /v Installed >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo   [OK] VC++ 2010 SP1 (x64)
    set "VC2010_X64=YES"
) else (
    echo   [MISSING] VC++ 2010 SP1 (x64)
    set "VC2010_X64=NO"
    set "ALL_OK=NO"
)

REM Check VC++ 2010 x86
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\10.0\VC\VCRedist\x86" /v Installed >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo   [OK] VC++ 2010 SP1 (x86)
    set "VC2010_X86=YES"
) else (
    echo   [MISSING] VC++ 2010 SP1 (x86)
    set "VC2010_X86=NO"
    set "ALL_OK=NO"
)

echo.

REM Step 2: Install missing runtimes
if "%ALL_OK%"=="YES" (
    echo All required runtimes are installed!
    echo.
) else (
    echo [2/2] Installing missing runtimes...
    echo.

    if "%VC2015_X64%"=="NO" (
        echo   Downloading VC++ 2015-2022 x64...
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile '%TEMP%\vcredist_x64.exe'}"
        if exist "%TEMP%\vcredist_x64.exe" (
            echo   Installing...
            start /wait "" "%TEMP%\vcredist_x64.exe" /quiet /norestart
            del /f /q "%TEMP%\vcredist_x64.exe" 2>nul
            echo   Done
        )
    )

    if "%VC2015_X86%"=="NO" (
        echo   Downloading VC++ 2015-2022 x86...
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x86.exe' -OutFile '%TEMP%\vcredist_x86.exe'}"
        if exist "%TEMP%\vcredist_x86.exe" (
            echo   Installing...
            start /wait "" "%TEMP%\vcredist_x86.exe" /quiet /norestart
            del /f /q "%TEMP%\vcredist_x86.exe" 2>nul
            echo   Done
        )
    )

    if "%VC2010_X64%"=="NO" (
        echo   Downloading VC++ 2010 x64...
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x64.exe' -OutFile '%TEMP%\vcredist2010_x64.exe'}"
        if exist "%TEMP%\vcredist2010_x64.exe" (
            echo   Installing...
            start /wait "" "%TEMP%\vcredist2010_x64.exe" /quiet /norestart
            del /f /q "%TEMP%\vcredist2010_x64.exe" 2>nul
            echo   Done
        )
    )

    if "%VC2010_X86%"=="NO" (
        echo   Downloading VC++ 2010 x86...
        powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x86.exe' -OutFile '%TEMP%\vcredist2010_x86.exe'}"
        if exist "%TEMP%\vcredist2010_x86.exe" (
            echo   Installing...
            start /wait "" "%TEMP%\vcredist2010_x86.exe" /quiet /norestart
            del /f /q "%TEMP%\vcredist2010_x86.exe" 2>nul
            echo   Done
        )
    )

    echo.
)
```

接下来将 `launch_with_setup.bat` 改写成真正的通用版本，让用户能通过参数传入要启动的程序，同时添加 GUI 模式的支持。
<minimax:tool_call>
<invoke name="Write">
<parameter name="content">@echo off
chcp 65001 >nul 2>&1
title PyInstaller App Launcher

REM ========================================
REM Generic PyInstaller Runtime Setup & Launcher
REM
REM Usage:
REM   run.bat                         - Run main.exe in current folder
REM   run.bat app.exe                 - Run specified exe
REM   run.bat app.exe --no-install     - Skip runtime installation
REM ========================================

setlocal enabledelayedexpansion

set "APP_EXE="
set "SKIP_INSTALL=NO"
set "HEADLESS=NO"

REM Parse arguments
:parse_args
if "%~1"=="" goto :done_args
if "%~1"=="--no-install" set "SKIP_INSTALL=YES" && shift && goto :parse_args
if "%~1"=="--silent" set "HEADLESS=YES" && shift && goto :parse_args
set "APP_EXE=%~1"
shift && goto :parse_args
:done_args

REM If no app specified, try to find exe in current directory
if not defined APP_EXE (
    REM Find first .exe in current directory (excluding this script itself)
    for %%F in (*.exe) do (
        if not "%%~nxF"=="%~nx0" (
            set "APP_EXE=%%F"
            goto :found_exe
        )
    )
    :found_exe
)

if not defined APP_EXE (
    echo ERROR: No executable found!
    echo.
    echo Usage:
    echo   run.bat                          - Auto-detect exe
    echo   run.bat app.exe                   - Run specific exe
    echo   run.bat app.exe --no-install      - Skip runtime install
    echo   run.bat --help                    - Show help
    echo.
    if "%HEADLESS%"=="YES" exit /b 1
    pause
    exit /b 1
)

echo =========================================
echo   PyInstaller Runtime Setup
echo =========================================
echo.
echo Target: %APP_EXE%
echo.

REM Step 1: Check and install runtimes (unless skipped)
if "%SKIP_INSTALL%"=="YES" (
    echo [SKIP] Runtime installation disabled
) else (
    echo [1/2] Checking VC++ Redistributables...
    echo.

    set "NEED_INSTALL=NO"
    set "VC_STATUS="

    REM Check VC++ 2015-2022 x64
    reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1
    if !errorlevel! == 0 (
        echo   [OK] VC++ 2015-2022 (x64)
    ) else (
        echo   [MISSING] VC++ 2015-2022 (x64)
        set "NEED_INSTALL=YES"
        set "VC_STATUS=2015x64"
    )

    REM Check VC++ 2010 x64 (for older apps)
    reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\10.0\VC\VCRedist\x64" /v Installed >nul 2>&1
    if !errorlevel! == 0 (
        echo   [OK] VC++ 2010 SP1 (x64)
    ) else (
        echo   [MISSING] VC++ 2010 SP1 (x64)
        set "NEED_INSTALL=YES"
        set "VC_STATUS=2010x64"
    )

    echo.

    if "%NEED_INSTALL%"=="YES" (
        if "%HEADLESS%"=="YES" (
            echo Installing runtimes (silent)...
            powershell -Command "$ErrorActionPreference='SilentlyContinue'; $ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile '%TEMP%\vcredist_x64.exe'; Start-Process '%TEMP%\vcredist_x64.exe' -ArgumentList '/quiet','/norestart' -Wait; Remove-Item '%TEMP%\vcredist_x64.exe' -Force -ErrorAction SilentlyContinue"
            if "VC_STATUS"=="2010x64" (
                powershell -Command "$ErrorActionPreference='SilentlyContinue'; $ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x64.exe' -OutFile '%TEMP%\vcredist_x64.exe'; Start-Process '%TEMP%\vcredist_x64.exe' -ArgumentList '/quiet','/norestart' -Wait; Remove-Item '%TEMP%\vcredist_x64.exe' -Force -ErrorAction SilentlyContinue"
            )
        ) else (
            echo [2/2] Installing required runtimes...
            echo.

            echo   Downloading VC++ 2015-2022...
            powershell -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile '%TEMP%\vcredist_x64.exe'"

            if exist "%TEMP%\vcredist_x64.exe" (
                echo   Installing...
                start /wait "" "%TEMP%\vcredist_x64.exe" /quiet /norestart
                del /f /q "%TEMP%\vcredist_x64.exe" 2>nul
                echo   Done
            )

            echo.
            echo   Downloading VC++ 2010 SP1...
            powershell -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://download.microsoft.com/download/1/6/5/165255E7-1014-4D0A-B094-B6A430A6BFFC/vcredist_x64.exe' -OutFile '%TEMP%\vcredist_x64.exe'"

            if exist "%TEMP%\vcredist_x64.exe" (
                echo   Installing...
                start /wait "" "%TEMP%\vcredist_x64.exe" /quiet /norestart
                del /f /q "%TEMP%\vcredist_x64.exe" 2>nul
                echo   Done
            )
        )
    ) else (
        echo All runtimes already installed.
    )
)

REM Step 2: Launch the application
echo.
echo [DONE] Launching %APP_EXE%...
echo.

if exist "%APP_EXE%" (
    start "" "%APP_EXE%"
) else (
    echo ERROR: File not found: %APP_EXE%
    if "%HEADLESS%"=="YES" exit /b 1
    pause
    exit /b 1
)

echo.
echo =========================================
echo   Started! Press any key to exit...
echo =========================================

if "%HEADLESS%"=="YES" (
    exit /b 0
)
pause >nul