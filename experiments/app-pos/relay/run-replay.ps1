<#
녹화 모드: 영상 파일(없으면 시험 화면)을 rtsp://127.0.0.1:9554/cam1 로 반복 송출한다.

    .\run-replay.ps1                         # 시계가 돌아가는 시험 화면
    .\run-replay.ps1 -Video D:\videos\a.mp4  # 영상 파일 반복

필요: relay\bin\mediamtx.exe (README 참고), ffmpeg (PATH)
#>
param([string]$Video = "")

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$mtx = Join-Path $here "bin\mediamtx.exe"
$conf = Join-Path $here "mediamtx.yml"
if (-not (Test-Path $mtx)) { throw "mediamtx.exe 가 없습니다. relay\README.md 대로 bin\ 에 내려받으세요." }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "ffmpeg 가 PATH 에 없습니다." }

# 경로에 공백이 있어도 한 덩어리로 넘어가게 따옴표로 감싼다.
$relay = Start-Process -FilePath $mtx -ArgumentList "`"$conf`"" -WorkingDirectory $here -PassThru -NoNewWindow
Start-Sleep -Seconds 1
try {
    # H.264 baseline, B-프레임 없음, 1초마다 키프레임: 앱이 중간에 붙어도 바로 화면이 나온다.
    $enc = @("-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-profile:v", "baseline",
             "-pix_fmt", "yuv420p", "-bf", "0", "-g", "15", "-an")
    if ($Video) {
        $in = @("-re", "-stream_loop", "-1", "-i", $Video)
    } else {
        # Windows ffmpeg 는 기본 글꼴 설정이 없어 글꼴 파일을 직접 준다.
        $font = "C\:/Windows/Fonts/arial.ttf"
        $text = "drawtext=fontfile='$font':text='HawkEye cam1 %{localtime}':x=20:y=20:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.5"
        $in = @("-re", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=15", "-vf", $text)
    }
    # ffmpeg 는 진행 상황을 stderr 로 내보내므로, 이를 오류로 보고 멈추지 않게 한다 (Windows PowerShell 5.1).
    $ErrorActionPreference = "Continue"
    & ffmpeg -hide_banner -loglevel warning @in @enc -f rtsp -rtsp_transport tcp "rtsp://127.0.0.1:9554/cam1"
}
finally {
    Stop-Process -Id $relay.Id -ErrorAction SilentlyContinue
}
