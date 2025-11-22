@echo off
echo ========================================
echo Uploading ArXiv dataset to server
echo File: D:\arxiv-metadata-oai-snapshot.json
echo Size: 4.62 GB
echo ========================================
echo.
echo This may take several minutes...
echo.

set SERVER=root@connect.bjb1.seetacloud.com
set PORT=53799
set REMOTE_DIR=~/rag-project/data

echo Creating remote data directory...
ssh -p %PORT% %SERVER% "mkdir -p %REMOTE_DIR%"

echo.
echo Starting upload...
scp -P %PORT% D:\arxiv-metadata-oai-snapshot.json %SERVER%:%REMOTE_DIR%/

echo.
echo ========================================
echo Upload completed!
echo ========================================
echo.
pause
