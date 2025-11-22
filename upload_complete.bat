@echo off
echo ========================================
echo Step 1: Create remote directory
echo ========================================
ssh -p 53799 root@connect.bjb1.seetacloud.com "mkdir -p ~/rag-project/data"

echo.
echo ========================================
echo Step 2: Upload dataset (4.62 GB)
echo ========================================
scp -P 53799 D:\arxiv-metadata-oai-snapshot.json root@connect.bjb1.seetacloud.com:~/rag-project/data/

echo.
echo ========================================
echo Upload completed!
echo ========================================
pause
