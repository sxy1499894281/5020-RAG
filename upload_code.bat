@echo off
echo ========================================
echo Uploading project code to server
echo ========================================

set SERVER=root@connect.bjb1.seetacloud.com
set PORT=53799
set REMOTE_DIR=~/rag-project

echo [1/5] Uploading source code...
scp -P %PORT% -r src %SERVER%:%REMOTE_DIR%/

echo [2/5] Uploading configs...
scp -P %PORT% -r configs %SERVER%:%REMOTE_DIR%/

echo [3/5] Uploading requirements.txt...
scp -P %PORT% requirements.txt %SERVER%:%REMOTE_DIR%/

echo [4/5] Uploading .env...
scp -P %PORT% .env %SERVER%:%REMOTE_DIR%/

echo [5/5] Uploading .gitignore...
scp -P %PORT% .gitignore %SERVER%:%REMOTE_DIR%/

echo.
echo Code upload completed!
pause
