@echo off
echo Uploading data to server...
echo.

set SERVER=root@connect.bjb1.seetacloud.com
set PORT=53799
set REMOTE_DIR=~/rag-project

echo Creating remote data directory...
ssh -p %PORT% %SERVER% "mkdir -p %REMOTE_DIR%/data"

echo Uploading entire data folder...
scp -P %PORT% -r data %SERVER%:%REMOTE_DIR%/

echo.
echo Data upload completed!
echo.
pause
