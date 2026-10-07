# app-pos — 관리자 앱 · 모의 결제 · 테스트 기록

담당: 한호석

## 하는 일

1. **관리자 앱 (모바일)**: Figma로 화면을 설계한 뒤, 카메라의 RTSP 영상을 모바일에서
   볼 수 있는 앱을 만든다. 알림을 받으면 고객 번호, 확인이 필요한 상품, 관련 영상을 보여주고
   "결제 확인 완료" / "잘못된 알림"으로 처리할 수 있게 한다.
2. **모의 결제 시스템**: 영상 처리 결과와 바로 연결해 시험할 수 있도록 결제 성공·실패를
   만들어내는 모의 계산대를 구현한다.
3. **테스트 기록**: 다른 팀원들의 테스트 결과를 받아 기록·정리하고 다음 테스트 방법을 정한다.

## 폴더 구성 (권장)

```
app-pos/
├─ design/        # Figma 링크, 화면 설계 캡처
├─ app/           # 모바일 앱
├─ pos-mock/      # 모의 결제 시스템
└─ test-log/      # 팀 테스트 결과 기록
```

## 설계

`docs/design.md` — 구성, 사건(case) 규칙, API, 오류 처리.

## 경보 서버 실행

```bash
cd experiments/app-pos/server
python -m venv .venv
.venv\Scripts\activate                 # Windows
pip install -r requirements.txt
python -m pytest -q                     # 테스트
python -m hawkeye_server --host 0.0.0.0 --port 8000
```

설정은 환경변수로 준다 (`hawkeye_server/__main__.py` 참고).

| 변수 | 뜻 | 기본값 |
|---|---|---|
| `HAWKEYE_DB` | SQLite 파일 | `data/hawkeye.db` |
| `HAWKEYE_SNAPSHOTS` | 장면 사진 폴더 | `data/snapshots` |
| `HAWKEYE_API_KEY` | 설정하면 `X-API-Key` 헤더 필요 | 없음 |
| `HAWKEYE_STREAMS` | 앱에 내려줄 RTSP 주소 `cam1=rtsp://...,cam2=...` | 없음 |

`server/data/`(DB·장면 사진)는 사람이 찍힌 사진이 있어 커밋되지 않는다.

## 경보 전송 클라이언트 · 시뮬레이터

`client/hawkeye_client` — 어떤 탐지 방식이든 경보 서버로 경보를 보낼 수 있는 작은 모듈 (표준 라이브러리만 사용).
팀 방식이 정해지면 `final/` 에서 아래처럼 붙인다.

```python
from hawkeye_client import AlertPublisher, build_event

alerts = AlertPublisher("http://127.0.0.1:8000", camera_id="cam1")   # 또는 AlertPublisher.from_env(os.environ)
alerts.publish(build_event(person_id=3, level="HIGH_RISK", taken=2, paid=1,
                           items=[{"name": "과자", "taken": 1, "paid": 0}]), jpeg=jpeg_bytes)
alerts.close()   # 끝날 때 남은 경보를 모두 보낸다
```

파이프라인 없이 앱을 시험·시연할 때는 시뮬레이터를 쓴다.

```bash
cd experiments/app-pos/client
python -m hawkeye_client.simulate --server http://127.0.0.1:8000 --scenario all
# theft(결제 없이 퇴장) / partial(일부만 결제) / paid(결제 후 통과) / came_back(되돌아가 결제)
python -m pytest -q    # 테스트 (서버 venv 로 실행: fastapi 가 필요)
```

## RTSP 중계기

`relay/README.md` — 카메라(또는 녹화 영상) → `rtsp://<PC>:9554/cam1` 로 파이프라인·앱에 나눠 준다.

## 관리자 앱 (안드로이드)

```bash
cd experiments/app-pos/app
echo sdk.dir=C:/Users/<사용자>/AppData/Local/Android/Sdk > local.properties   # 처음 한 번
gradlew.bat :admin:testDebugUnitTest      # 단위 테스트
gradlew.bat :admin:assembleDebug          # APK: admin/build/outputs/apk/debug/admin-debug.apk
adb install -r admin/build/outputs/apk/debug/admin-debug.apk
```

- 앱 설정 화면에서 경보 서버 주소를 넣는다. 에뮬레이터는 `http://10.0.2.2:8000`, 실제 폰은 노트북 IP (같은 와이파이).
- 라이브 화면 주소는 서버의 `HAWKEYE_STREAMS` 에서 받는다.
- 개발 PC 확인 환경: JDK 21, Android SDK 34, AGP 8.5.2, Kotlin 2.0.20

## PC 웹 대시보드

경보 서버를 띄우면 브라우저에서 `http://127.0.0.1:8000` 으로 연다. 왼쪽에 카메라 라이브(크게·전체화면), 오른쪽에 경보 목록·상세·처리.
경보가 오면 화면 위에 알림 띠가 뜨고 소리가 난다. 앱과 같은 서버를 쓰므로 PC·폰 어느 쪽에서 처리해도 서로 반영된다.

라이브를 보려면 서버에 브라우저용 주소를 알려 준다: `HAWKEYE_WEB_STREAMS=cam1=http://127.0.0.1:8889/cam1`

화면 로직 테스트: `cd server && node --test "tests/js/*.test.mjs"`

## PC 한 대로 시험하기 (폰·에뮬레이터 없이)

모든 구성 요소가 이 PC 안(127.0.0.1)에서만 통신한다. 방화벽 설정이 필요 없다.

```powershell
# 1) 중계기 — 카메라 (또는 카메라가 없으면 .\run-replay.ps1 로 시험 화면)
cd experiments\app-pos\relay
$env:MTX_RTSPADDRESS = "127.0.0.1:9554"
.\run-camera.ps1     # 카메라 IP·RTSP 계정·비밀번호를 물어본다 (비밀번호는 화면에 안 보이고 저장 안 됨)

# 2) 경보 서버 + 대시보드 (다른 터미널)
cd experiments\app-pos\server
$env:HAWKEYE_WEB_STREAMS = "cam1=http://127.0.0.1:8889/cam1"
python -m hawkeye_server --port 8000
#   → 브라우저에서 http://127.0.0.1:8000

# 3) 경보 보내기 (다른 터미널, 시뮬레이터)
cd experiments\app-pos\client
python -m hawkeye_client.simulate --scenario all
```

## 한 번에 시험하기 (개발 PC + 안드로이드 에뮬레이터)

```bash
# 1) 중계기 (녹화 모드, 시험 화면)
relay
un-replay.ps1
# 2) 경보 서버 (에뮬레이터용 라이브 주소)
set HAWKEYE_STREAMS=cam1=rtsp://10.0.2.2:9554/cam1
python -m hawkeye_server --port 8000
# 3) 경보 보내기 (시뮬레이터. 실제 탐지 방식은 final/ 통합 때 연결)
python -m hawkeye_client.simulate --server http://127.0.0.1:8000 --scenario all
# 4) 에뮬레이터에 앱 설치·실행
```

## 주의

RTSP 주소, 카메라 계정, API 키는 코드에 넣지 않고 `.env`에 둔다 (커밋되지 않음).

## 실험 기록

| 날짜 | 대상 | 시험 내용 | 결과 | 다음 할 일 |
|---|---|---|---|---|
| | | | | |
