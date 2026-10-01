# storeguard 실행 순서 (Windows PowerShell)
# 한 줄씩 복사해서 실행할 것. 이 스크립트를 통째로 돌리면 학습까지 수 시간 걸린다.

$ErrorActionPreference = "Stop"
Set-Location D:\computervision
$PY = "D:\computervision\.venv\Scripts\python.exe"

Write-Host "=== 0. 환경 확인 ===" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
& $PY -c "import torch,cv2; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'| cv2',cv2.__version__)"
& $PY -m storeguard.tools.selftest

Write-Host "=== 1. 압축 해제 (약 92GB, 1~2시간) ===" -ForegroundColor Cyan
& $PY -m storeguard.data.extract --what all --split all

Write-Host "=== 2. 인덱스 + 데이터 검사 ===" -ForegroundColor Cyan
& $PY -m storeguard.data.build_index --probe-video
& $PY -m storeguard.data.inspect

Write-Host "=== 3. 분할 (촬영 사건 단위) ===" -ForegroundColor Cyan
& $PY -m storeguard.data.split

Write-Host "=== 4. 프레임 캐시 (약 25GB, 약 70분) ===" -ForegroundColor Cyan
& $PY -m storeguard.data.preprocess --workers 4

Write-Host "=== 4b. (선택) 정상 영상 등록 ===" -ForegroundColor Cyan
# 이 데이터셋에는 정상 영상이 없다. 정상 구매/반품/물건 줍기/쪼그려 앉기 등을 모은 폴더가 있으면
# 아래를 실행한다. 그래야 '정상 영상 시간당 오경보'를 실제로 측정할 수 있다.
#   & $PY -m storeguard.data.add_normal --dir D:\computervision\data\normal_raw --split-ratio 0.2
#   & $PY -m storeguard.data.preprocess --workers 4     # 새 영상만 추가로 캐시됨

Write-Host "=== 5. VRAM 실측 ===" -ForegroundColor Cyan
& $PY -m storeguard.tools.vram_probe

Write-Host "=== 6. 학습 ===" -ForegroundColor Cyan
& $PY -m storeguard.train --name baseline_r2p1d
# 중단 후 재개:
#   & $PY -m storeguard.train --name baseline_r2p1d --resume

Write-Host "=== 7. 평가 (test 분할, 모델 선택에 사용 금지) ===" -ForegroundColor Cyan
& $PY -m storeguard.evaluate --run baseline_r2p1d --split test
# 정상 영상을 등록했다면 시간당 오경보도 측정한다:
#   & $PY -m storeguard.evaluate --run baseline_r2p1d --split normal

Write-Host "=== 8. MP4 실시간 재생 시험 ===" -ForegroundColor Cyan
$sample = (Get-ChildItem D:\computervision\data\raw\val\videos\*.mp4 | Select-Object -First 1).FullName
& $PY -m storeguard.infer.runner --source $sample --run baseline_r2p1d --print

Write-Host "=== 9. RTSP 연결 시험 ===" -ForegroundColor Cyan
# 계정 정보는 환경변수로만 전달한다. 화면/로그에는 마스킹되어 나온다.
#   $env:CAM1_RTSP = "rtsp://사용자:비밀번호@192.168.0.10:554/stream2"
#   & $PY -m storeguard.infer.runner --source-env CAM1_RTSP --run baseline_r2p1d --print --seconds 60

Write-Host "=== 10. 대시보드 ===" -ForegroundColor Cyan
# .env.example 을 .env 로 복사해 CAM1_SOURCE 를 채운 뒤:
#   & $PY -m storeguard.server.app --run baseline_r2p1d
# 브라우저에서 http://127.0.0.1:8000
