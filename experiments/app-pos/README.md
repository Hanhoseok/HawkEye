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
.venv\Scriptsctivate                 # Windows
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

## 한 번에 시험하기 (개발 PC 한 대)

```bash
# 1) 중계기 (녹화 모드, 시험 화면)
relayun-replay.ps1
# 2) 경보 서버 (에뮬레이터용 라이브 주소)
set HAWKEYE_STREAMS=cam1=rtsp://10.0.2.2:9554/cam1
python -m hawkeye_server --port 8000
# 3) 파이프라인 (tracking 폴더, PR #8 의 전송기)
python run_tracking.py --source rtsp://127.0.0.1:9554/cam1 --alert-server http://127.0.0.1:8000
# 4) 에뮬레이터에 앱 설치·실행
```

## 주의

RTSP 주소, 카메라 계정, API 키는 코드에 넣지 않고 `.env`에 둔다 (커밋되지 않음).

## 실험 기록

| 날짜 | 대상 | 시험 내용 | 결과 | 다음 할 일 |
|---|---|---|---|---|
| | | | | |
