# 대시보드 실행 (가장 자주 쓰는 명령)
#
#   PowerShell 에서:  .\scripts\run_dashboard.ps1
#   반복 재생 데모  :  .\scripts\run_dashboard.ps1 -Loop
#   포트 바꾸기     :  .\scripts\run_dashboard.ps1 -Port 8080
#   모델 없이 UI만  :  .\scripts\run_dashboard.ps1 -Run ""
#
# 카메라 주소는 D:\computervision\.env 에서 읽는다. (CAM1_SOURCE)

param(
    [string]$Run  = "baseline_r2p1d",   # runs\ 아래 학습 실행 이름. "" 이면 모델 미로딩 상태
    [int]$Port    = 8000,
    [switch]$Loop                        # 파일 입력을 끝나면 처음부터 다시 재생(데모용)
)

$ErrorActionPreference = "Stop"
Set-Location D:\computervision
$PY = "D:\computervision\.venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    Write-Host "가상환경이 없습니다. README 3.2절의 설치 단계를 먼저 실행하세요." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path ".\.env")) {
    Write-Host ".env 가 없습니다. .env.example 을 복사해서 CAM1_SOURCE 를 채우세요." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host ".env 를 만들었습니다. 카메라 주소를 확인하고 다시 실행하세요." -ForegroundColor Yellow
    exit 1
}

$args = @("-m", "storeguard.server.app", "--port", "$Port")
if ($Run) {
    if (-not (Test-Path ".\runs\$Run\best.pt")) {
        Write-Host "체크포인트가 없습니다: runs\$Run\best.pt" -ForegroundColor Yellow
        Write-Host "모델 미로딩 상태로 띄웁니다. 대시보드에 '탐지 불가' 로 표시됩니다." -ForegroundColor Yellow
    } else {
        $args += @("--run", $Run)
    }
}
if ($Loop) { $args += "--loop" }

Write-Host "브라우저에서 http://127.0.0.1:$Port 를 여세요. 종료는 Ctrl+C." -ForegroundColor Cyan
& $PY @args
