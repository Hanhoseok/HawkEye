# relay — RTSP 중계기 (MediaMTX)

카메라 한 대 영상을 **파이프라인과 관리자 앱에 함께** 나눠 준다.
저가 IP 카메라는 동시 접속 수가 적고, 카메라 계정을 앱에 넣지 않기 위해서다.

```
카메라 ─RTSP─▶ MediaMTX (rtsp://<이 PC>:9554/cam1) ─┬─▶ 파이프라인
                                                     └─▶ 관리자 앱 라이브 화면
```

## 설치

1. [MediaMTX 릴리스](https://github.com/bluenviron/mediamtx/releases)에서 `mediamtx_<버전>_windows_amd64.zip` 을 받는다.
   같은 릴리스의 `checksums.sha256` 으로 확인한 뒤 `relay/bin/` 에 푼다 (`bin/` 은 커밋되지 않는다).
   ```bash
   gh release download -R bluenviron/mediamtx --pattern "*windows_amd64.zip" --pattern checksums.sha256 -D bin
   ```
2. ffmpeg 이 PATH 에 있어야 녹화 모드를 쓸 수 있다 (`winget install Gyan.FFmpeg`).

개발 PC 확인 버전: MediaMTX v1.21.1, ffmpeg 8.1.1

## 실행

| 모드 | 명령 | 용도 |
|---|---|---|
| 녹화 | `.\run-replay.ps1` 또는 `.\run-replay.ps1 -Video <파일>` | 개발·녹화 영상 시연. 파일이 없으면 시계가 도는 시험 화면 |
| 카메라 | `$env:HAWKEYE_CAMERA_URL="rtsp://..."; .\run-camera.ps1` | 실제 카메라 |

## 열리는 포트

| 포트 | 용도 | 기본 열림 범위 |
|---|---|---|
| TCP 9554 | RTSP (파이프라인·앱) | 설정 `:9554` (모든 네트워크). PC 안에서만 쓰려면 `MTX_RTSPADDRESS=127.0.0.1:9554` |
| TCP 8889 / UDP 8189 | WebRTC (PC 웹 대시보드) | 127.0.0.1 만 |

브라우저 재생 주소: `http://127.0.0.1:8889/cam1` (대시보드가 `/whep` 로 붙는다).

## 다른 구성 요소에 알려줄 주소

포트는 **9554** 다. RTSP 기본값 8554 는 안드로이드 에뮬레이터(gRPC)가 쓰고 있어서 겹친다.

- 파이프라인: `run_tracking.py --source rtsp://127.0.0.1:9554/cam1`
- 경보 서버: `HAWKEYE_STREAMS=cam1=rtsp://<이 PC IP>:9554/cam1` (앱이 이 주소로 라이브를 연다)
  - 에뮬레이터에서 볼 때: `rtsp://10.0.2.2:9554/cam1`
- 방화벽에서 TCP 9554 를 같은 와이파이에 열어야 폰에서 볼 수 있다.
