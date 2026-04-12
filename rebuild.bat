@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d C:\Users\jwest\stsc
nvcc -O3 -arch=sm_120 -use_fast_math -lineinfo -o coupled_field.exe coupled_field.cu > build_out.txt 2> build_err.txt
echo %ERRORLEVEL% > build_exit.txt
