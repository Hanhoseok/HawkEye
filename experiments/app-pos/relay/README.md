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

## 카메라 설정 (VIGI 앱 → 영상 설정)

- **영상 코덱: H.264** — H.265 는 브라우저(WebRTC)에 따라 재생되지 않는다.
  실제로 개발 PC 크롬에서 `codecs not supported by client` 로 대시보드 라이브가 안 나왔고, H.264 로 바꾸자 해결됐다.
  대시보드는 이 경우 "카메라 영상 코덱을 H.264 로 설정하세요" 라고 알려 준다.
- **해상도: 1920x1080 권장** — 2560x1440 도 재생되지만, 나중에 탐지 코드가 분석할 때 CPU 부담이 크다.
- RTSP 계정은 VIGI 앱에서 만든다. 비밀번호에 `@ : / # ? $ "` 같은 문자가 있으면 주소에 넣을 때 깨지기 쉬우니 영문·숫자를 권장한다.
- 먼저 PC 에서 카메라가 보이는지 확인한다: `ffplay -rtsp_transport tcp "rtsp://계정:비밀번호@카메라IP:554/stream1"`

## 실행

| 모드 | 명령 | 용도 |
|---|---|---|
| 녹화 | `.\run-replay.ps1` 또는 `.\run-replay.ps1 -Video <파일>` | 개발·녹화 영상 시연. 파일이 없으면 시계가 도는 시험 화면 |
| 카메라 | `.\run-camera.ps1` (저화질: `-Stream stream2`) | 실제 카메라. IP·RTSP 계정·비밀번호를 물어본다 |

카메라 모드 메모:
- 비밀번호는 입력할 때 화면에 안 보이고 어디에도 저장되지 않는다. 특수문자가 있어도 주소에 맞게 바꿔 넘긴다.
- 다음 실행 때 Enter 만 쳐도 되도록 마지막 IP·계정만 `bin\camera.local.json` 에 기억한다 (git 제외).
- 입력한 IP 에 닿지 않으면 다시 묻는다. 중계 중 비밀번호가 틀리거나(401) 카메라가 끊기면 로그 아래에 `>>` 로 원인을 알려 준다.
- 자동 실행용: `$env:HAWKEYE_CAMERA_URL` 에 전체 주소가 있으면 묻지 않는다. 이 값이 남아 있는 창에서는 계속 그 주소를 쓰니, 묻게 하려면 `Remove-Item Env:HAWKEYE_CAMERA_URL`.
- 시험: `powershell -NoProfile -ExecutionPolicy Bypass -File tests\run-camera.tests.ps1`

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

## 렌탈 서버 (Linux) — 폰 송출 · 녹화 · 재생

촬영 폰 2대가 Larix 로 SRT 송출 → MediaMTX → tracking(RTSP) · 관리자 앱 라이브(RTSP) · 녹화 → 재생 API(경보 서버가 대신 받음).
설정 `mediamtx.server.yml`, 실행 `run-server.sh`, 대체 송출 `replay-server.sh`, 한 번에 띄우기 `../deploy/start.sh`.

> **이 서버는 공인 IP 가 네트워크 카드에 바로 붙어 있다.** 0.0.0.0 으로 연 포트는 인터넷에 그대로 노출된다.
> 그래서 RTSP·SRT·경보 서버는 **Tailscale IP** 에만, 재생 API 는 **127.0.0.1** 에만 연다. 스크립트가 0.0.0.0 을 거부한다.

### 처음 한 번

1. Tailscale: 서버·촬영 폰·관리자 폰을 같은 계정(tailnet)으로 로그인.
2. MediaMTX (Linux amd64) 를 받아 체크섬 확인 후 `relay/bin/` 에 푼다 (또는 다른 곳에 두고 `.env` 의 `HAWKEYE_MEDIAMTX`).
   ```bash
   V=v1.21.4; F=mediamtx_${V}_linux_amd64.tar.gz; B=https://github.com/bluenviron/mediamtx/releases/download/$V
   mkdir -p relay/bin && cd relay/bin && curl -fsSLO "$B/$F" && curl -fsSLO "$B/checksums.sha256"
   grep -E "[ *]$F\$" checksums.sha256 | sed 's/ \*/  /' | sha256sum -c - && tar xzf "$F" mediamtx && cd ../..
   ```
3. 계정 파일: `cp deploy/.env.example deploy/.env && chmod 600 deploy/.env` 후 송출·보기 계정과 API 키를 채운다 (영문·숫자).
4. 경보 서버 가상환경: `cd server && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..`
5. 확인: `relay/run-server.sh --check`, `deploy/run-alert.sh --check` (비밀번호는 출력하지 않는다)

### 실행

```bash
deploy/start.sh                 # tmux 세션 'hawkeye': relay · alert · pipeline 창
tmux attach -t hawkeye          # 보기 (나가기: Ctrl+b 다음 d)
tmux kill-session -t hawkeye    # 모두 끄기
```

| 구성 | 주소 | 누가 |
|---|---|---|
| SRT 송출 | `srt://<Tailscale IP>:8890?streamid=publish:cam1:<송출 계정>:<비밀번호>&pkt_size=1316` | 촬영 폰 (cam1 전체, cam2 선반) |
| RTSP | `rtsp://<Tailscale IP>:8554/cam1` | tracking (서버 안 — 계정 없이), 관리자 앱 (보기 계정, 경보 서버가 주소를 내려줌) |
| AI 화면 | `rtsp://<Tailscale IP>:8554/cam1_ai` | 파이프라인이 서버 안에서 계정 없이 송출 (`hawkeye_client.FrameStreamer`, 원본 `cam1` 에는 못 보냄). 파이프라인 없이 시험: `python -m hawkeye_client.stream_demo` |
| 재생 API | `http://127.0.0.1:9996` | 경보 서버만 |
| 경보 서버 | `http://<Tailscale IP>:8000` | 관리자 앱 (API 키), tracking `--alert-server` |
| 녹화 | `~/hawkeye/recordings/<cam>/` | 자동 삭제 없음 — 렌탈 종료 때 `deploy/cleanup.sh` |

### Larix Broadcaster (촬영 폰)

- 연결: 위 SRT 주소 (cam1 폰은 `cam1`, cam2 폰은 `cam2`)
- 영상: 1920x1080, 30fps, **H.264**, **5Mbps 고정**, 키프레임 1초
- 송출 중에는 화면을 켜 두고 앱을 앞에 띄운다 (뒤로 가면 끊김). 충전기 연결.

### 대체 경로 — 미리 녹화한 영상을 실시간처럼 흘리기

폰 기본 카메라로 녹화한 파일을 서버에 올린 뒤:

```bash
relay/replay-server.sh cam1.mp4 cam2.mp4 --offset-ms 1500   # cam2 에서 같은 장면이 1.5초 뒤에 나오면 +1500
relay/replay-server.sh cam1.mp4 cam2.mp4 --dry-run          # 명령만 보기
```

파일을 그대로 복사하지 않고 안드로이드가 확실히 트는 형식(H.264 Main, yuv420p, B-프레임 없음, 1초 키프레임, 5Mbps)으로
다시 인코딩한다 (폰 녹화가 HEVC 일 수 있음). GPU 인코더가 안 되면 libx264 로 바꾼다.

### 보관함 — 미리 찍은 촬영 영상을 앱에서 폴더로 보기

라이브가 안 될 때를 대비한다 (`docs/design.md` §11). 영상은 서버에만 두고 커밋하지 않는다. PC 에서 먼저 줄여(720p H.264) 올리면 업로드가 빠르다.

```bash
cd ~/hawkeye/app-pos/server
HAWKEYE_LIBRARY=~/hawkeye/server-data/library .venv/bin/python -m hawkeye_server.library import   --folder "2차 촬영" --camera "cam1=<올린 폴더>/1(angle1)" --camera "cam2=<올린 폴더>/1(angle0)"   --sync <tracking>/labels/store01_dual/sync.csv      # 3차: store01_shoot3, 4차: store01_shoot4
```

카메라가 1대인 1차(`source/0`)는 `--camera cam1=...` 하나만. `--dry-run` 으로 장면 짝을 먼저 볼 수 있다.

AI 탐지 화면은 PC 에서 추적 기록으로 그린 뒤(`tools/ai_render.py --records <tracking>/outputs/dual_eval_pose3 --video cam1=… --video cam2=… --sync … --out <폴더>`,
opencv-python-headless · numpy · pillow 필요) 올려서 같은 명령에 `--ai` 를 붙여 넣는다 (장면을 새로 만들지 않고 이미 넣은 영상에 붙임).
AI 판정은 추적 도구 판정(`experiments/tracking` 에서 `python workbench/data.py --shoot N`)을 `tools/export_verdicts.py <판정.json> --tracking experiments/tracking --shoot N` 으로 꺼내 (추적 도구의 최신 확인을 거쳐 — 최신이 아닌 짝이 있으면 멈춤)
`import --folder "4차 촬영" --verdicts <판정.json>` 으로 붙인다 (`--camera` 없이).

### 시험

`bash relay/tests/server-scripts.tests.sh` — 계정이 비면 거부, 0.0.0.0 거부, 비밀번호를 출력하지 않음, 시차 옵션, 호환 인코딩 인자.
