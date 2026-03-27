@echo off
setlocal
set PYTHONIOENCODING=utf-8
set PYTHON=C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe

echo ============================================
echo KVTC 5090 GPU Benchmark
echo ============================================

cd /d C:\Users\User
if not exist kvtc (
    git clone https://github.com/OnlyTerp/kvtc.git
)
cd kvtc
git pull

%PYTHON% -m pip install -e . --quiet 2>nul

echo.
echo === TEST 1: TinyLlama 512tok balanced (5.6 bits) ===
%PYTHON% -m src.benchmark_gpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda --seq-len 512 --calibration-samples 5 --sink-tokens 4 --window-tokens 32 --bit-budget-ratio 0.35

echo.
echo === TEST 2: TinyLlama 512tok high quality (8 bits) ===
%PYTHON% -m src.benchmark_gpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda --seq-len 512 --calibration-samples 5 --sink-tokens 4 --window-tokens 32 --bit-budget-ratio 0.5

echo.
echo === TEST 3: TinyLlama 512tok max compress (4 bits) ===
%PYTHON% -m src.benchmark_gpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda --seq-len 512 --calibration-samples 5 --sink-tokens 4 --window-tokens 32 --bit-budget-ratio 0.25

echo.
echo === TEST 4: TinyLlama 1024tok balanced ===
%PYTHON% -m src.benchmark_gpu --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda --seq-len 1024 --calibration-samples 5 --sink-tokens 4 --window-tokens 32 --bit-budget-ratio 0.35

echo.
echo ============================================
echo ALL BENCHMARKS COMPLETE
echo ============================================
echo Copy everything above and send to Terp AI Labs
pause
