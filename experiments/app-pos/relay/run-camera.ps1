<#
카메라 모드: IP 카메라(VIGI C440-W 등)의 RTSP 를 받아 rtsp://<이 PC>:9554/cam1 로 나눠 준다.

    .\run-camera.ps1                    # 카메라 IP·계정·비밀번호를 물어본다 (비밀번호는 화면에 안 보임)
    .\run-camera.ps1 -Stream stream2    # 저화질 스트림 (stream1=고화질, stream2=저화질)

비밀번호는 어디에도 저장하지 않는다. 다음에 다시 묻지 않도록 마지막 IP·계정만
bin\camera.local.json 에 기억한다 (bin\ 은 git 에서 빠져 있다).
자동 실행용으로 $env:HAWKEYE_CAMERA_URL 에 전체 주소가 있으면 묻지 않고 그 주소를 쓴다.
VIGI 카메라는 앱/웹에서 RTSP 계정을 먼저 만들어야 한다.
#>
param(
    [string]$Stream = "stream1",
    [int]$Port = 554
)

# --- 순수 함수 (tests\run-camera.tests.ps1 에서 시험) --------------------------

function New-CameraUrl([string]$Ip, [int]$Port, [string]$Stream, [string]$User, [string]$Password) {
    # @ : / # ? 같은 문자가 비밀번호에 있으면 주소가 깨지므로 % 인코딩한다 (MediaMTX 가 되돌린다).
    $u = [uri]::EscapeDataString($User)
    $p = [uri]::EscapeDataString($Password)
    "rtsp://${u}:${p}@${Ip}:${Port}/$Stream"
}

function Test-CameraHost([string]$Ip) {
    $Ip -match '^[A-Za-z0-9][A-Za-z0-9.\-]*$'
}

function Get-RelayHint([string]$Line) {
    if ($Line -match '\b401\b|Unauthorized') { return "카메라 계정 또는 비밀번호가 틀렸습니다. Ctrl+C 로 끄고 다시 실행해 입력하세요." }
    if ($Line -match 'i/o timeout|connectex|connection refused|no route to host|network is unreachable') {
        return "카메라에 연결할 수 없습니다. IP 와 카메라 전원, 같은 공유기에 있는지 확인하세요."
    }
    if ($Line -match 'codecs not supported') { return "브라우저가 카메라 영상 코덱을 재생할 수 없습니다. VIGI 앱 → 영상 설정에서 H.264 로 바꾸세요." }
    if ($Line -match 'stream is available and online') { return "카메라 영상을 받고 있습니다." }
    $null
}

function Read-Secret([string]$Prompt) {
    $secure = Read-Host -Prompt $Prompt -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}

function Test-TcpPort([string]$Ip, [int]$Port, [int]$TimeoutMs = 2000) {
    $client = New-Object Net.Sockets.TcpClient
    try { $client.ConnectAsync($Ip, $Port).Wait($TimeoutMs) -and $client.Connected } catch { $false }
    finally { $client.Close() }
}

# 시험에서 함수만 불러갈 때(. .\run-camera.ps1)는 여기서 멈춘다.
if ($MyInvocation.InvocationName -eq '.') { return }

# --- 실행 ---------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$mtx = Join-Path $here "bin\mediamtx.exe"
if (-not (Test-Path $mtx)) { throw "mediamtx.exe 가 없습니다. relay\README.md 대로 bin\ 에 내려받으세요." }

if ($env:HAWKEYE_CAMERA_URL) {
    $url = $env:HAWKEYE_CAMERA_URL
    $where = "HAWKEYE_CAMERA_URL 환경변수의 주소"
} else {
    $memo = Join-Path $here "bin\camera.local.json"
    $last = $null
    if (Test-Path $memo) { try { $last = Get-Content $memo -Raw -Encoding UTF8 | ConvertFrom-Json } catch { } }

    while ($true) {
        $ip = (Read-Host "카메라 IP$(if ($last.ip) { " [$($last.ip)]" })").Trim()
        if (-not $ip) { $ip = $last.ip }
        if (-not (Test-CameraHost $ip)) { Write-Host "  IP 만 입력하세요 (예: 192.168.0.50)." -ForegroundColor Yellow; continue }
        if (Test-TcpPort $ip $Port) { break }
        Write-Host "  ${ip}:$Port 에 연결할 수 없습니다. IP 와 카메라 전원, 같은 공유기에 있는지 확인하세요." -ForegroundColor Yellow
    }
    do {
        $user = (Read-Host "RTSP 계정$(if ($last.user) { " [$($last.user)]" })").Trim()
        if (-not $user) { $user = $last.user }
    } while (-not $user)
    do { $password = Read-Secret "RTSP 비밀번호 (입력해도 화면에 안 보임)" } while (-not $password)

    $url = New-CameraUrl $ip $Port $Stream $user $password
    $password = $null
    @{ ip = $ip; user = $user } | ConvertTo-Json | Set-Content $memo -Encoding UTF8
    $where = "${ip}:$Port/$Stream (계정 $user)"
}

Write-Host ""
Write-Host "카메라 중계를 시작합니다. 끄려면 Ctrl+C" -ForegroundColor Green
Write-Host "  카메라      : $where"
Write-Host "  앱·탐지 코드: rtsp://127.0.0.1:9554/cam1"
Write-Host "  PC 대시보드 : http://127.0.0.1:8000 (경보 서버를 켰을 때)"
Write-Host ""

# 비밀번호가 든 주소는 명령줄이 아닌 환경변수로만 중계기에 넘기고, 끝나면 지운다.
$env:MTX_PATHS_CAM1_SOURCE = $url
$env:MTX_PATHS_CAM1_RTSPTRANSPORT = "tcp"
$url = $null
$shown = @{}
Push-Location $here   # 혹시 생기는 파일이 저장소 루트에 흩어지지 않게
try {
    & $mtx (Join-Path $here "mediamtx.yml") | ForEach-Object {
        $_
        $hint = Get-RelayHint $_
        if ($hint -and -not $shown[$hint]) {   # 중계기가 5초마다 다시 시도해도 안내는 한 번만
            $shown[$hint] = $true
            Write-Host "  >> $hint" -ForegroundColor Cyan
        }
    }
} finally {
    Remove-Item Env:MTX_PATHS_CAM1_SOURCE, Env:MTX_PATHS_CAM1_RTSPTRANSPORT -ErrorAction SilentlyContinue
    Pop-Location
}
