<#
카메라 모드: IP 카메라(VIGI C440-W 등)의 RTSP 를 받아 rtsp://<이 PC>:9554/cam1 로 나눠 준다.

    $env:HAWKEYE_CAMERA_URL = "rtsp://<계정>:<비밀번호>@192.168.0.50:554/stream1"
    .\run-camera.ps1

카메라 주소·계정은 환경변수로만 넘긴다 (저장소에 남기지 않는다).
VIGI 카메라는 앱/웹에서 RTSP 계정을 먼저 만들어야 한다. stream1=고화질, stream2=저화질.
#>
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$mtx = Join-Path $here "bin\mediamtx.exe"
if (-not (Test-Path $mtx)) { throw "mediamtx.exe 가 없습니다. relay\README.md 대로 bin\ 에 내려받으세요." }
if (-not $env:HAWKEYE_CAMERA_URL) { throw "환경변수 HAWKEYE_CAMERA_URL 에 카메라 RTSP 주소를 넣으세요." }

$env:MTX_PATHS_CAM1_SOURCE = $env:HAWKEYE_CAMERA_URL
$env:MTX_PATHS_CAM1_RTSPTRANSPORT = "tcp"
& $mtx (Join-Path $here "mediamtx.yml")
