@echo off
REM Run ViennaRNA-based TorusFold proxy experiment
REM Must use Python 3.13 (includes ViennaRNA bindings)

C:\Program Files\Python313\python.exe ^
  D:\IGEM集成方案\manuscripts\scripts\torusfold_proxy_experiment.py ^
  --n-sequences 30 ^
  --n-epochs 80 ^
  --n-seeds 3 ^
  --viennarna ^
  --output-dir D:\IGEM集成方案\manuscripts\figures\results

if %ERRORLEVEL% EQU 0 (
  echo.
  echo Experiment completed successfully!
  echo Check results in: D:\IGEM集成方案\manuscripts\figures\results\
  echo Manuscript TBD values will be printed above.
) else (
  echo.
  echo Experiment failed! Check error messages above.
  exit /b 1
)