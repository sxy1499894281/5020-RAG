# SSH服务器配置
$SERVER = "root@connect.bjb1.seetacloud.com"
$PORT = "53799"
$REMOTE_DIR = "~/rag-project"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "上传RAG项目到远程服务器" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 获取当前脚本所在目录
$PROJECT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "项目目录: $PROJECT_DIR" -ForegroundColor Green
Write-Host "目标服务器: $SERVER" -ForegroundColor Green
Write-Host "远程目录: $REMOTE_DIR" -ForegroundColor Green
Write-Host ""

# 在服务器上创建项目目录
Write-Host "[1/6] 创建远程目录..." -ForegroundColor Yellow
ssh -p $PORT $SERVER "mkdir -p $REMOTE_DIR"

# 上传源代码
Write-Host "[2/6] 上传源代码 (src/)..." -ForegroundColor Yellow
scp -P $PORT -r "$PROJECT_DIR\src" "${SERVER}:${REMOTE_DIR}/"

# 上传配置文件
Write-Host "[3/6] 上传配置文件 (configs/)..." -ForegroundColor Yellow
scp -P $PORT -r "$PROJECT_DIR\configs" "${SERVER}:${REMOTE_DIR}/"

# 上传数据文件（如果存在且不太大）
Write-Host "[4/6] 上传数据文件..." -ForegroundColor Yellow
# 只上传必要的数据文件，跳过大文件
scp -P $PORT "$PROJECT_DIR\data\*.jsonl" "${SERVER}:${REMOTE_DIR}/data/" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "   注意: data/*.jsonl 文件不存在或上传失败，稍后需要在服务器上生成" -ForegroundColor Yellow
}

# 上传依赖文件
Write-Host "[5/6] 上传依赖文件..." -ForegroundColor Yellow
scp -P $PORT "$PROJECT_DIR\requirements.txt" "${SERVER}:${REMOTE_DIR}/"
scp -P $PORT "$PROJECT_DIR\.env" "${SERVER}:${REMOTE_DIR}/" 2>$null

# 上传README和文档（如果需要）
Write-Host "[6/6] 上传其他文件..." -ForegroundColor Yellow
scp -P $PORT "$PROJECT_DIR\.gitignore" "${SERVER}:${REMOTE_DIR}/" 2>$null

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "上传完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. cd ~/rag-project" -ForegroundColor White
Write-Host "2. python3 -m venv .venv" -ForegroundColor White
Write-Host "3. source .venv/bin/activate" -ForegroundColor White
Write-Host "4. pip install -r requirements.txt" -ForegroundColor White
Write-Host ""
