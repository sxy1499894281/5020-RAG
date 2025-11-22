@echo off
echo ========================================
echo Uploading RAG project to remote server
echo ========================================
echo.

set SERVER=root@connect.bjb1.seetacloud.com
set PORT=53799
set REMOTE_DIR=~/rag-project

echo [1/6] Creating remote directory...
ssh -p %PORT% %SERVER% "mkdir -p %REMOTE_DIR%"

echo [2/6] Uploading source code...
scp -P %PORT% -r src %SERVER%:%REMOTE_DIR%/

echo [3/6] Uploading config files...
scp -P %PORT% -r configs %SERVER%:%REMOTE_DIR%/

echo [4/6] Uploading data directory structure...
ssh -p %PORT% %SERVER% "mkdir -p %REMOTE_DIR%/data"
scp -P %PORT% data\*.jsonl %SERVER%:%REMOTE_DIR%/data/ 2>nul

echo [5/6] Uploading requirements...
scp -P %PORT% requirements.txt %SERVER%:%REMOTE_DIR%/
scp -P %PORT% .env %SERVER%:%REMOTE_DIR%/ 2>nul

echo [6/6] Uploading other files...
scp -P %PORT% .gitignore %SERVER%:%REMOTE_DIR%/ 2>nul

echo.
echo ========================================
echo Upload completed!
echo ========================================
echo.
echo Next steps in SSH session:
echo   1. cd ~/rag-project
echo   2. python3 -m venv .venv
echo   3. source .venv/bin/activate
echo   4. pip install -r requirements.txt
echo.
pause
