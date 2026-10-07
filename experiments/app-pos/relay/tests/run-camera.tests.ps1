# run-camera.ps1 의 순수 함수 시험 (Pester 없이 Windows PowerShell 5.1 에서 바로 실행).
#     powershell -NoProfile -ExecutionPolicy Bypass -File relay\tests\run-camera.tests.ps1
. (Join-Path $PSScriptRoot "..\run-camera.ps1")

$script:failed = 0
function Check([string]$Name, $Actual, $Expected) {
    if ($Actual -ceq $Expected) { Write-Host "ok   $Name" }
    else { Write-Host "FAIL $Name`n  기대: $Expected`n  실제: $Actual"; $script:failed++ }
}

# 주소 만들기: 계정·비밀번호의 특수문자는 % 인코딩해야 주소가 깨지지 않는다.
Check "평범한 비밀번호" (New-CameraUrl "192.168.0.50" 554 "stream1" "admin" "abc123") "rtsp://admin:abc123@192.168.0.50:554/stream1"
Check "특수문자 비밀번호" (New-CameraUrl "192.168.0.50" 554 "stream1" "admin" 'p@ss:w/rd#?$" x') "rtsp://admin:p%40ss%3Aw%2Frd%23%3F%24%22%20x@192.168.0.50:554/stream1"
Check "특수문자 계정" (New-CameraUrl "192.168.0.50" 554 "stream2" "a@b" "pw") "rtsp://a%40b:pw@192.168.0.50:554/stream2"

# 카메라 주소 입력 검사: IP·호스트 이름만 받는다.
Check "IP 허용" (Test-CameraHost "192.168.0.50") $true
Check "호스트 이름 허용" (Test-CameraHost "vigi-cam.local") $true
Check "빈 값 거부" (Test-CameraHost "") $false
Check "rtsp 주소 통째로 거부" (Test-CameraHost "rtsp://192.168.0.50") $false
Check "계정 포함 거부" (Test-CameraHost "admin@192.168.0.50") $false

# 중계기 로그 한 줄 -> 관리자용 안내 (MediaMTX 실제 로그 형식)
Check "비밀번호 틀림" (Get-RelayHint "2026/10/07 10:00:00 WAR [path cam1] [RTSP source] bad status code: 401 (Unauthorized)") "카메라 계정 또는 비밀번호가 틀렸습니다. Ctrl+C 로 끄고 다시 실행해 입력하세요."
Check "연결 안 됨" (Get-RelayHint "WAR [path cam1] [RTSP source] dial tcp 192.168.0.50:554: connectex: A connection attempt failed") "카메라에 연결할 수 없습니다. IP 와 카메라 전원, 같은 공유기에 있는지 확인하세요."
Check "시간 초과" (Get-RelayHint "WAR [path cam1] [RTSP source] dial tcp 192.168.0.50:554: i/o timeout") "카메라에 연결할 수 없습니다. IP 와 카메라 전원, 같은 공유기에 있는지 확인하세요."
Check "코덱" (Get-RelayHint "ERR [WebRTC] [session 1] codecs not supported by client") "브라우저가 카메라 영상 코덱을 재생할 수 없습니다. VIGI 앱 → 영상 설정에서 H.264 로 바꾸세요."
Check "수신 시작" (Get-RelayHint "2026/10/07 14:03:20 INF [path cam1] stream is available and online, 1 track (H264)") "카메라 영상을 받고 있습니다."
Check "그 밖의 줄" (Get-RelayHint "INF [RTSP] listener opened on :9554 (TCP)") $null

Write-Host ""
if ($script:failed) { Write-Host "$script:failed 개 실패"; exit 1 } else { Write-Host "모두 통과" }
