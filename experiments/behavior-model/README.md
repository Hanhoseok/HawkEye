# storeguard — 무인매장 행동 인식 시스템

IP 카메라(RTSP) 또는 MP4 영상에서 매장 내 사람 행동을 인식하고,
**이상행동이 보일 때만** 웹 대시보드에 실시간 알림을 띄우는 캡스톤디자인용 시스템이다.

| 그룹 | 클래스 | 알림 |
|---|---|---|
| 이상행동 (AI-Hub 238-2) | 전도, 파손, 절도 의심 | **알림 발생** |
| 구매행동 (AI-Hub 238-1) | 매장이동, 선택, 시험, 구매, 반품, 비교 | 알림 없음 (상태 표시만) |
| 음성 | 라벨 구간 밖 | - |

구매행동을 함께 학습하는 이유는 오경보 때문이다. 이상행동만 학습한 모델을
정상 구매행동 영상에 돌려 보니 **시간당 52.9건**의 오경보가 났다(4.3절).

결제 승인·거절, POS 연동, 위장 결제 판별은 이번 범위에 없다.
영상만으로 절도를 확정하지 않으며 **"절도 의심 행동"** 으로만 표시한다.

관련 문서
- [AGENTS.md](AGENTS.md) — 작업 규약과 금지 사항
- [docs/architecture.md](docs/architecture.md) — 아키텍처, 입출력, 설정 근거
- [reports/data_report.md](reports/data_report.md) — 데이터 검사 결과 (자동 생성)
- [reports/results_baseline_r2p1d.md](reports/results_baseline_r2p1d.md) — 학습·평가 결과 (자동 생성)
- [docs/colab.md](docs/colab.md) — Colab GPU 로 학습 옮기기 (결제 판단 + 절차)
- [docs/purchase_data_status.md](docs/purchase_data_status.md) — 구매행동(238-1) 데이터 상태와 필요한 조치
- [docs/preprocessing_optimization.md](docs/preprocessing_optimization.md) — 전처리 병목 측정과 GPU 검토 결과

---

## 1. 지금 무엇이 실제로 되는가

이 절은 **실행해서 확인한 것만** 적는다. 확인하지 않은 것은 아래 "아직 안 한 것"에 있다.

| 항목 | 상태 | 근거 |
|---|---|---|
| 환경 구축 (Python 3.11.9 + torch 2.5.1+cu121 + CUDA) | 확인됨 | `python -c "import torch; torch.cuda.is_available()"` → True, GTX 1050 Ti |
| 배포 zip 12개 압축 해제 (92 GB) | 확인됨 | mp4 2,169 + xml 2,169 |
| 데이터 전수 검사 | 확인됨 | `reports/data_report.json` — 손상·누락·중복 **0건** |
| 영상 실측 | 확인됨 | 전부 **3.0 fps / 1920×1080 / 180~181프레임(60초)** |
| take 단위 분할 | 확인됨 | train 1,636 / val 287 / test 241 영상, 누수 검사 통과 |
| 공식 배포 분할의 누수 발견 | 확인됨 | take 3건이 Training/Validation 에 걸쳐 있어 학습에서 제외 |
| 핵심 로직 자체 검증 11항목 | 통과 | `python -m storeguard.tools.selftest` → 11/11 |
| VRAM 실측 | 확인됨 | `reports/vram_probe_r2plus1d_18_16x112.json` |
| MP4 실시간 재생 파이프라인 | 확인됨 | 수신 2.94~3.00 fps, 클립 버퍼 16/16 |
| FastAPI 서버 + 대시보드 + MJPEG 미리보기 | 확인됨 | 브라우저에서 라이브 영상·상태·점수 표시 확인 |
| 모델 미로딩 상태 표시 | 확인됨 | 가중치 없을 때 `MODEL_NOT_LOADED`, 점수·알림 생성 안 함 |
| 이상행동 모델 학습 (6에폭, 약 6.8시간) | 완료 | `runs/baseline_r2p1d/history.json`, best val macro-F1 **0.9799** |
| 시험 분할 평가 (클립 + 사건) | 완료 | `runs/baseline_r2p1d/eval_test.json` — 클립 macro-F1 **0.9570** |
| 학습된 모델로 MP4 실시간 탐지 + 알림 + 영상 저장 | 확인됨 | 전도/파손/절도 각 1건 알림 발생, 스냅샷·클립 파일 생성 확인 |
| 대시보드에서 실제 알림 표시 | 확인됨 | 모델 READY 배지, 실시간 점수, 이벤트 카드 |
| 웹에서 영상 전환 / 클래스 필터 / RTSP 연결 | 확인됨 | 브라우저에서 버튼 클릭으로 2/241 → 4/241 전환, 경로·프로토콜 차단 동작 |
| 구매행동(238-1) 데이터 편입 | 완료 | 영상 680개 추가, 총 2,849개. `docs/purchase_data_status.md` |
| 정상 영상 오경보 측정 | 완료 | 이상행동 모델이 정상 구매행동 영상에서 **시간당 52.9건** |
| **통합 10클래스 모델 학습** | **미완료** | 파이프라인은 검증됨(스모크 통과). A100에서 `notebooks/train.ipynb` 실행 필요 |

### 한눈에 보는 결과 (시험 분할, 배포 Validation 241영상)

| 지표 | 전도 | 파손 | 절도 의심 |
|---|---|---|---|
| 사건 단위 탐지율 | **100 %** (81/81) | **97.5 %** (79/81) | **90.2 %** (74/82) |
| 탐지 지연 중앙값 | 1.67초 | 3.33초 | 4.33초 |
| 클립 F1 | 1.000 | 0.963 | 0.898 |

클립 단위 전체 macro-F1 **0.957**, 실시간 추론 지연 **92.5 ms**, 수신 3 fps.
자세한 수치와 조건은 4절에 있다.

---

## 2. 데이터에서 실제로 확인한 것

원본: `D:\computervision\238-2.실내(편의점, 매장) 사람 이상행동 데이터\01-1.정식개방데이터`

| 확인 항목 | 결과 |
|---|---|
| 영상 ↔ 라벨 연결 규칙 | 확장자 제외 파일명이 **1:1 동일**. 짝 없는 파일 0건 |
| 실제 클래스 | 파일명 코드 `7`=전도, `8`=파손, `12`=절도 / XML 이벤트 라벨 `fall`, `broken`, `theft` |
| 수량 | fall **726**, broken **722**, theft **721** (설명서의 "각 800건"과 다름) |
| 영상 길이 / FPS / 해상도 | 60초 / **3.0 fps** / 1920×1080, 전 파일 동일 |
| 사람 ID | 이벤트 마커 박스의 `ID` 속성(대부분 값 1). **프레임별 추적 ID 아님** |
| 바운딩박스 | `*_start`/`*_end` 마커 박스 + `object_article`(22,478) / `object_tool` / `object_abandon`. **사람 전신 박스 트랙 없음** |
| 키포인트 | 17관절, 영상당 13~131프레임에만 존재(중앙값 63/180). **전 프레임 아님** |
| 행동 시작·종료 시점 | **있음.** `*_start`/`*_end` 트랙이 각각 한 프레임에 마커로 찍혀 구간을 이룸 |
| 이벤트 길이(중앙값) | 전도 17.3초 / 파손 24.0초 / 절도 23.3초 |
| 최단 이벤트 | 전도 4.33초, 절도 2.67초, 파손 12.0초 |
| 정상 구간 | 이벤트 밖 프레임이 평균 64%. 단 **정상으로 검증된 구간이 아님** |
| 다중 행동 구간 | 이벤트 2~3개인 영상 49건. 한 영상에 여러 **종류**가 섞인 경우는 0건 |
| 매장 구분 | 가능. 7종 (DYA/DYB/SYA/SYB/SMA/SMB/SMC) |
| 카메라 구분 | 가능. CA~CH |
| 촬영 세션 구분 | 가능. 날짜+시각+시나리오로 607개 take 로 묶임 |
| 배우 구분 | **불가.** `M1`/`F2` 는 성별+번호일 뿐 개인 식별자가 아님 |
| 누락·손상·중복 | **0건** (2,169개 전부 디코딩 및 XML 파싱 성공) |

### XML 구조에서 주의할 점

`meta/task/labels` 에는 fire/smoke/abandon/fight/weak pedestrian 까지 **8종이 항상 선언**되어 있다.
이것은 CVAT 프로젝트 스키마일 뿐이며 실제 라벨이 아니다. **실제 라벨은 `<track>` 요소에만 있다.**
이 구분을 놓치면 존재하지 않는 클래스를 학습시키게 된다.

### 공식 배포 분할의 누수

같은 촬영 사건(동일 시각·시나리오·배우)이 카메라별로 Training 과 Validation 에 갈라져 있는 경우가
**3건** 있다. 예: `C_3_7_48_BU_DYB_10-17_10-23-53_CA` (Training) 와 `..._CB/CC/CD/CE` (Validation).
그대로 쓰면 학습 데이터가 시험 데이터에 새어 든다. `storeguard.data.split` 이 이 take 의
Training 쪽 영상을 자동으로 제외한다.

### 정상 데이터가 없다는 문제

이 데이터셋에는 **정상 행동 영상이 없다.** 세 클래스 모두 연출된 이상행동 영상이다.
따라서:

- `background_unlabeled` 클래스는 이상행동 영상의 이벤트 구간 밖에서 뽑은 것이고,
  행동 직전의 접근·물색 동작이 섞여 있다. **"정상"이라고 부르지 않는다.**
- **정상 연속 영상에 대한 시간당 오경보는 지금은 측정할 수 없다.** 측정하려면 다음이 필요하다.
  1. 같은 AI-Hub 과제의 **정상 구매행동 카테고리** 영상 (구매·반품·비교·매장이동 등), 또는
  2. 직접 촬영한 소형 매장의 정상 영업 영상 (최소 수 시간 연속).

정상 영상이 생기면 아래 세 줄로 파이프라인에 들어가고, 그때부터 오경보가 **실제로 측정된다.**
정상 영상은 보통 30fps 이므로 전처리가 자동으로 3fps 로 시간 리샘플링한다.

```powershell
& $PY -m storeguard.data.add_normal --dir D:\computervision\data\normal_raw --split-ratio 0.2
& $PY -m storeguard.data.preprocess --workers 4
& $PY -m storeguard.train --name with_normal
& $PY -m storeguard.evaluate --run with_normal --split normal   # ← 시간당 오경보
```

정상 폴더에는 **사람이 정상이라고 확인한 영상만** 넣는다. 라벨이 없다는 이유로 정상 취급하면 안 되고,
대상 외 이상행동(방화·흡연·유기·폭행)을 정상 폴더에 넣어서도 안 된다.

---

## 3. 설치와 실행 (Windows PowerShell)

### 3.0 이 PC에서는 지금 바로 됨

데이터 추출·캐시·학습이 모두 끝나 있고 `.env` 도 만들어져 있다. 아래 한 줄이면 대시보드가 뜬다.

```powershell
cd D:\computervision; .\scripts\run_dashboard.cmd --loop
```

브라우저에서 `http://127.0.0.1:8000` 을 연다. 종료는 `Ctrl+C`.

`--loop` 는 데모용으로 입력 영상을 끝나면 처음부터 다시 재생한다. 기본 입력은 `.env` 의
`CAM1_SOURCE` 이며 지금은 전도 시험 영상이 들어 있어 약 45초 지점에서 알림이 뜬다.

**영상을 바꾸려고 `.env` 를 고칠 필요는 없다.** 대시보드의 카메라 카드에 있는
`◀ 이전` / `다음 ▶` 버튼으로 시험 분할 241개 영상을 넘겨 볼 수 있고,
옆의 드롭다운으로 클래스(전도/파손/절도)만 골라 볼 수도 있다.
같은 카드의 입력칸에 RTSP 주소를 넣고 `RTSP 연결` 을 누르면 실제 IP 카메라로 바로 전환된다.

> **`.ps1` 이 "이 시스템에서 스크립트를 실행할 수 없으므로" 오류로 막히는 경우**
> Windows 의 PowerShell 실행 정책 기본값이 `Restricted` 라서 `.ps1` 파일 실행이 차단된 것이다.
> 프로젝트 문제가 아니다. 위의 `.cmd` 런처는 이 정책과 무관하게 동작하므로 그대로 쓰면 된다.
> `.ps1` 을 꼭 쓰고 싶다면 둘 중 하나를 고른다.
>
> ```powershell
> # (1) 이번 실행만 허용 - 시스템 설정을 바꾸지 않는다
> powershell -ExecutionPolicy Bypass -File .\scripts\run_dashboard.ps1 -Loop
>
> # (2) 내 계정에 한해 영구 허용 - 설정이 바뀌므로 판단해서 쓸 것
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

#### 화면에 나오는 영상 바꾸기

입력은 전부 `D:\computervision\.env` 에서 정해진다. 파일을 고치고 **서버를 껐다 켜면** 반영된다.
(서버는 시작할 때 한 번만 `.env` 를 읽는다.)

```ini
CAMERAS=cam1,cam2,cam3          # 여기 적힌 ID 만 대시보드에 나온다

CAM1_NAME=1번 카메라 - 전도
CAM1_SOURCE=D:\computervision\data\raw\val\videos\<파일명>.mp4
```

기본값은 전도·파손·절도 시험 영상 3개가 카메라 3대로 동시에 나오게 되어 있다.
한 대만 보고 싶으면 `CAMERAS=cam1` 로 줄이면 된다.

다른 영상으로 바꾸려면 `data\raw\val\videos` 에서 파일명 앞부분으로 고른다.
`C_3_7_…` 이 전도, `C_3_8_…` 이 파손, `C_3_12_…` 가 절도다. 각 클래스마다 80개씩 있다.

```powershell
# 시험 분할 영상과 사건 발생 시각을 한 번에 보기
cd D:\computervision
.\.venv\Scripts\python.exe -m storeguard.tools.list_demo_videos --class fall --limit 10
```

카메라를 늘리려면 `CAMERAS` 에 ID 를 추가하고 `CAM4_NAME` / `CAM4_SOURCE` 를 적는다.
카메라 3대를 동시에 돌려도 이 GPU 에서 여유가 있었다. 측정값은 아래와 같다.

| 항목 | 카메라 1대 | 카메라 3대 |
|---|---|---|
| 모델 자체 추론 시간 | 92.5 ms | 104 ms |
| 대기 포함 추론 지연 | 92.5 ms | 107~147 ms |
| 카메라당 추론 횟수 | 초당 0.88회 | 초당 0.88회 |

카메라가 2대 이상이면 화면이 자동으로 격자 배치로 바뀐다.

### 3.1 환경

이 PC에서 실제로 구성해 동작을 확인한 환경이다.

```
OS      Windows 10 Home 19045
CPU     Intel i7-7700 (4C/8T)
RAM     8 GB
GPU     GTX 1050 Ti, VRAM 4 GB, driver 560.94
Python  3.11.9  (D:\computervision\.venv)
PyTorch 2.5.1+cu121
OpenCV  5.0.0
```

### 3.2 설치

```powershell
winget install --id Python.Python.3.11 --scope user
cd D:\computervision
C:\Users\$env:USERNAME\AppData\Local\Programs\Python\Python311\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

GPU 확인:

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

CUDA 가 `False` 로 나오면 CPU 전용 torch 가 깔린 것이다. `pip uninstall torch torchvision` 후
위의 `--index-url` 이 붙은 명령으로 다시 설치한다.

### 3.3 실행 순서

`scripts\quickstart.ps1` 에 같은 내용이 있다. 각 단계가 실제로 끝난 것을 보고 다음으로 넘어간다.

```powershell
cd D:\computervision
$PY = "D:\computervision\.venv\Scripts\python.exe"

# 0) 로직 자체 검증 (torch 불필요, 수 초)
& $PY -m storeguard.tools.selftest

# 1) 압축 해제 (약 92 GB, 1~2시간)
& $PY -m storeguard.data.extract --what all --split all

# 2) 인덱스 + 데이터 검사 리포트
& $PY -m storeguard.data.build_index --probe-video
& $PY -m storeguard.data.inspect

# 3) 분할 (촬영 사건 단위, 누수 검사 포함)
& $PY -m storeguard.data.split

# 4) 프레임 캐시 (약 25 GB, CPU 바운드로 2~3시간)
& $PY -m storeguard.data.preprocess --workers 4

# 5) VRAM 실측
& $PY -m storeguard.tools.vram_probe

# 6) 학습 (중단해도 --resume 으로 이어서)
& $PY -m storeguard.train --name baseline_r2p1d
& $PY -m storeguard.train --name baseline_r2p1d --resume

# 7) 평가 (test 분할. 모델 선택·임계값 조정에 쓰지 않는다)
& $PY -m storeguard.evaluate --run baseline_r2p1d --split test
```

### 3.4 MP4 실시간 시험

```powershell
$sample = (Get-ChildItem D:\computervision\data\raw\val\videos\*.mp4 | Select-Object -First 1).FullName
& $PY -m storeguard.infer.runner --source $sample --run baseline_r2p1d --print
```

### 3.5 RTSP IP 카메라 연결

계정·비밀번호는 **환경변수로만** 넘긴다. 로그·API·대시보드에는 `rtsp://***:***@host/...` 로 나온다.

```powershell
$env:CAM1_RTSP = "rtsp://사용자:비밀번호@192.168.0.10:554/stream2"
& $PY -m storeguard.infer.runner --source-env CAM1_RTSP --run baseline_r2p1d --print --seconds 60
```

제조사별 RTSP 경로 예시는 `.env.example` 에 적어 두었다. 대역폭과 지연을 줄이려면
메인 스트림(1080p)이 아니라 **서브 스트림**(보통 640×360~D1)을 쓴다.
이 시스템은 어차피 3 fps 로 샘플링해 112×112 로 줄이므로 서브 스트림으로 충분하다.

### 3.6 대시보드

```powershell
Copy-Item .env.example .env      # 그 다음 .env 안의 CAM1_SOURCE 를 채운다
& $PY -m storeguard.server.app --run baseline_r2p1d
# 브라우저: http://127.0.0.1:8000
```

대시보드에서 볼 수 있는 것
- 카메라 라이브 미리보기 (MJPEG)
- 연결 상태 / 마지막 프레임 수신 시각 / 수신 FPS / 재접속 횟수
- 모델 로딩 상태 (미로딩이면 빨간 배지로 "탐지 불가")
- 클래스별 현재 탐지 점수
- 새 사건 발생 시 화면 우하단 팝업 알림 (SSE)
- 최근 이벤트 목록 + 대표 이미지 + 저장 영상 링크

---

## 4. 학습·평가 결과

> 이 절의 수치는 `runs/<이름>/` 과 `reports/` 의 실제 산출물에서 나온다.
> 값이 `미측정` 이면 아직 실행하지 않았다는 뜻이며, 추정치를 적지 않는다.

### 4.1 기준 모델 선정 근거

| 후보 | 판단 |
|---|---|
| 단일 프레임 객체 검출 + 규칙 | **탈락.** 넘어짐/앉음, 물건 집기/훔치기를 한 프레임으로 구분할 수 없다 |
| 포즈(키포인트) 기반 | **보조로만.** 파손·절도는 손·상품·집기의 관계가 핵심인데 골격만으로는 사라진다. 게다가 이 데이터는 전 프레임 키포인트가 없다(중앙값 63/180) |
| 사람 검출·추적 + 사람 크롭 분류 | **다음 단계.** 작은 인물 문제를 풀지만 검출기·추적기가 VRAM 4 GB 예산과 지연을 잡아먹고, 이 데이터에는 학습용 사람 박스가 없다 |
| **전체 프레임 RGB 클립 분류 (채택)** | 시간 문맥 + 장면 문맥을 함께 본다. 학습/추론 경로가 같아 실시간 조건을 정확히 맞출 수 있다 |

백본은 **`r2plus1d_18` (Kinetics-400 사전학습, torchvision)** 을 썼다.
(2+1)D 분해로 같은 연산량에서 순수 3D 보다 정확하고, 학습 해상도 112 가 4 GB VRAM 에 맞는다.
처음부터 학습하지 않고 **전이학습**한다.

같은 GPU에서 후보 백본을 실측 비교한 결과다 (배치 8 학습 / 배치 1 추론, cudnn 워밍업 후).

| arch | 학습 처리량 | 추론 지연 | 판단 |
|---|---|---|---|
| r2plus1d_18 | 4.4 clips/s | 75.2 ms | **채택.** 속도 차이가 작고 (2+1)D 가 정확도에 유리 |
| mc3_18 | 5.4 clips/s | 61.2 ms | 더 가볍다. VRAM 이 부족하면 이쪽 |
| r3d_18 | 5.7 clips/s | 58.3 ms | 가장 빠르지만 같은 계열에서 정확도가 낮다고 알려져 있다 |

추론 지연 차이(75 vs 58 ms)는 초당 1회 추론이라는 사용 조건에서 의미가 없었다.
그래서 속도가 아니라 정확도 쪽을 택했다.

### 4.2 학습

실행: `runs/baseline_r2p1d`. 6에폭, 배치 8, AMP, 학습 클립 16,600개. 에폭당 약 68분, 총 약 6.8시간.

| epoch | train acc | val macro-F1 |
|---|---|---|
| 0 | 0.7514 | 0.9606 |
| 1 | 0.9523 | 0.9726 |
| 2 | 0.9716 | 0.9742 |
| 3 | 0.9828 | 0.9776 |
| 4 | 0.9870 | **0.9799** ← best |
| 5 | 0.9898 | 0.9797 |

best 체크포인트는 **epoch 4** (val macro-F1 0.9799). 이후 에폭에서 개선이 없어 4번을 썼다.

### 4.3 시험(test) 분할 결과

test 분할 = 배포 Validation 세트 241영상 / 2,327클립 / 244사건. 모델 선택·임계값 조정에 쓰지 않았다.

**클립 단위** — 정확도 **0.9583**, macro-F1 **0.9570**

| 클래스 | Precision | Recall | F1 | 클립 수 |
|---|---|---|---|---|
| 라벨구간외(background) | 0.9460 | 0.9892 | 0.9671 | 1,205 |
| 전도(fall) | 1.0000 | 1.0000 | 1.0000 | 269 |
| 파손(broken) | 0.9671 | 0.9580 | 0.9625 | 429 |
| 절도 의심(theft) | 0.9598 | 0.8443 | 0.8984 | 424 |

**혼동행렬** (행 = 정답, 열 = 예측)

| 정답＼예측 | background | fall | broken | theft |
|---|---|---|---|---|
| **background** | 1192 | 0 | 1 | 12 |
| **fall** | 0 | 269 | 0 | 0 |
| **broken** | 15 | 0 | 411 | 3 |
| **theft** | 53 | 0 | 13 | 358 |

절도가 가장 약하다. 놓친 424−358=66건 중 53건이 background 로 빠졌다.
절도 행동이 상품을 집는 정상 동작과 시각적으로 가장 비슷하기 때문이다.

**사건 단위** — 실시간과 같은 `AlertEngine` 으로 알림을 만든 뒤 정답 사건과 매칭. 총 4.02시간.

| 클래스 | 정답 사건 | 탐지(TP) | 사건 탐지율 | 오탐(FP) | 중복 알림 | 지연 중앙값 | 지연 p90 |
|---|---|---|---|---|---|---|---|
| 전도 | 81 | 81 | **1.000** | 0 | 0 | 1.67초 | 2.33초 |
| 파손 | 81 | 79 | **0.975** | 6 | 0 | 3.33초 | 5.67초 |
| 절도 의심 | 82 | 74 | **0.902** | 9 | 1 | 4.33초 | 9.13초 |

이상행동 영상 위에서의 시간당 오경보 합계: **3.73건/시간**.
이 수치를 정상 매장의 오경보율로 읽으면 안 된다. 이 영상들은 대부분의 시간에
행동 직전의 접근·물색 동작이 이어져 실제 정상 영업보다 훨씬 어려운 조건이다.

전도가 가장 빠르고 정확한 이유는 두 가지다. 자세 변화가 크고, 클래스별 지속 조건을
전도만 느슨하게(최근 3회 중 2회) 잡아 두었기 때문이다.

**매장별 편차** — 평균 하나로 말하면 안 되는 이유다.

| 매장 | 영상 | 정답 사건 | 사건 탐지율 | 시간당 오경보 |
|---|---|---|---|---|
| DYB (대형유인매장B) | 116 | 116 | 0.957 | 1.55 |
| SMC (소형무인카페C) | 110 | 112 | 0.982 | 3.82 |
| DYA (대형유인매장A) | 15 | 16 | **0.813** | 20.0 |

DYA 는 영상이 15개뿐이라 수치가 흔들린다(오탐 5건 / 0.25시간). 그래도 같은 모델이 매장에 따라
탐지율 0.81~0.98 사이에서 움직인다는 사실은 분명하다. **세 매장 모두 학습에도 등장한다.**
완전히 새로운 매장의 성능은 이보다 낮다고 보는 것이 안전하다.

### 4.4 실시간 파이프라인 실측 (MP4 입력)

| 항목 | 값 |
|---|---|
| 실제 수신 FPS | 2.99 (3fps 소스를 실시간 속도로 재생) |
| 수신 / 폐기 프레임 | 180 / 0 |
| 추론 지연 평균 | **92.5 ms** |
| 추론 지연 p90 | 122.6 ms |
| 모델 로딩 시 워밍업 | 약 3.9~5.8초 (1회) |
| 추론 주기 | 1.0초 (설정값) |

세 클래스 모두 실제로 알림이 발생했고 문구와 대표 이미지를 확인했다.

| 영상 | 정답 시작 | 알림 시각 | 지연 | 점수 | 알림 문구 |
|---|---|---|---|---|---|
| 전도 | 43.7초 | 45.7초 | 2.0초 | 0.982 | 1번 카메라에서 전도 의심 상황이 감지되었습니다. |
| 파손 | 24.7초 | 28.7초 | 4.0초 | 0.862 | 3번 카메라에서 파손 의심 행동이 감지되었습니다. |
| 절도 | 26.3초 | 32.7초 | 6.4초 | 0.838 | 2번 카메라에서 절도 의심 행동이 감지되었습니다. |

대표 이미지(128 KB JPEG)와 알림 전후 영상(20초, 960×540, 3fps mp4)이 실제로 저장되는 것을 확인했다.
대시보드를 3분 이상 연속 구동해 발생한 이벤트 3건 모두 스냅샷과 영상이 저장되는 것도 확인했다.

구현 중 실제로 겪어서 고친 문제 네 가지를 남겨 둔다. 같은 함정에 다시 빠지지 않기 위해서다.

| 증상 | 원인 | 조치 |
|---|---|---|
| 캐시 `.npy` 파일이 전부 `X.npy.part.npy` 로 생성 | `np.save` 는 경로 문자열이 `.npy` 로 끝나지 않으면 확장자를 덧붙인다 | 파일 객체를 넘겨 저장 |
| 알림 영상이 0 바이트 | OpenCV `VideoWriter` 는 **확장자**로 컨테이너를 고른다. 임시 파일이 `.part` 로 끝나 열리지 않았다 | 임시 파일을 `.part.mp4` 로 바꾸고 실패를 로그로 남김 |
| 실시간 추론이 453 ms (벤치마크는 76 ms) | cudnn 자동 알고리즘 선택이 꺼져 있고 워밍업이 없었다 | 모델 로딩 시 `cudnn.benchmark` 를 켜고 더미 클립 8회 워밍업 |
| 대시보드가 1~2분 뒤 멈춤 | 상태 갱신마다 카드 HTML 을 다시 그려 `<img>` 가 새로 생기고, MJPEG 스트림이 2초마다 하나씩 쌓여 서버 스레드풀 고갈 | 카드를 한 번만 만들고 값만 갱신, MJPEG 엔드포인트를 비동기 제너레이터로 변경 |

### 4.5 하드웨어 실측

`reports/vram_probe_r2plus1d_18_16x112.json` (cudnn.benchmark 켜고 워밍업 후 측정)

| 배치 | 학습 peak VRAM | 학습 clips/s | 추론 peak VRAM | 추론 clips/s |
|---|---|---|---|---|
| 4 | 2.51 GB | 4.1 | 0.32 GB | 14.5 |
| 8 | 3.07 GB | **4.4** | 0.51 GB | 14.8 |
| 12 | 3.40 GB | 2.7 (VRAM 압박으로 오히려 하락) | 0.70 GB | 14.9 |

실시간(배치 1) 추론 지연 **76.3 ms** → 초당 13회 추론 가능. 시스템은 초당 1회만 쓴다.

> **주의**: cudnn 자동 알고리즘 선택을 켜고 워밍업을 하지 않으면 배치 1 추론이 441 ms 로
> 5배 이상 느려진다. 그래서 모델 로딩 시 더미 클립으로 워밍업을 수행한다.

전체 수치는 [reports/results_baseline_r2p1d.md](reports/results_baseline_r2p1d.md) 에 자동 생성되어 있다.

### 4.7 정상 영상에서의 오경보 — 통합 모델이 필요한 이유

구매행동 영상은 **정상 매장 영상**이다. 덕분에 그동안 `미측정` 이던 항목을 실제로 쟀다.

이상행동만 학습한 `baseline_r2p1d`(4클래스)를 test 분할의 구매행동 영상 133개(3.42시간)에
그대로 돌린 결과다.

| 클래스 | 오탐 | 시간당 오경보 |
|---|---|---|
| 전도 | 0건 | **0.0** |
| 파손 | 140건 | **40.9** |
| 절도 의심 | 41건 | **12.0** |
| 합계 | 181건 | **52.9** |

**133개 영상 중 104개에서 오경보가 났다.** 구매 22/22, 시험 30/31, 반품 17/19 로 거의 전부다.

원인은 학습 데이터에 있다. 그 모델은 연출된 이상행동 영상만 보고 배웠고, 거기서 "배경"은
같은 영상의 행동 직전 구간뿐이었다. **상품을 집고 계산하는 정상 행동을 한 번도 본 적이 없어서**
손을 뻗는 동작을 파손이나 절도로 읽는다.

구매행동을 별도 클래스로 학습시키면 줄어들 것으로 기대하지만,
**그 개선폭은 아직 측정하지 않았다.** 통합 모델 학습 후 같은 도구로 다시 재서 비교해야 한다.

```powershell
.\.venv\Scripts\python.exe -m storeguard.tools.false_alarm --run unified_r2p1d --compare baseline_r2p1d
```

### 4.6 사건 단위 평가의 매칭 규칙

- 알림(클래스 c, 시각 t)이 같은 클래스 정답 사건의 `[start − 3초, end + 3초]` 안에 들면 매칭.
- **정답 사건 1개당 가장 빠른 알림 1건만 TP.** 이후 매칭되는 알림은 '중복 알림'으로 따로 센다.
  중복을 여러 번 정답으로 세지 않는다.
- 어떤 정답 사건에도 안 붙는 알림은 오탐(FP).
- 탐지 지연 = 알림 시각 − 사건 시작 시각.
- 오프라인 평가와 실시간 추론이 **같은 `AlertEngine` 코드**를 쓴다. 평가용 별도 규칙이 없다.

---

## 5. 아직 하지 않은 것 / 측정할 수 없는 것

정직하게 구분한다.

| 항목 | 상태 | 이유와 필요한 것 |
|---|---|---|
| 정상 매장 연속 영상의 시간당 오경보 | **측정 완료** | 구매행동 영상으로 측정. 이상행동 모델 기준 시간당 52.9건 (4.7절) |
| 통합 10클래스 모델의 성능 | **미측정** | 파이프라인만 검증했다. A100에서 학습해야 수치가 나온다 |
| 통합 모델의 오경보 개선폭 | **미측정** | 학습 후 `storeguard.tools.false_alarm --compare` 로 비교해야 한다 |
| 새 매장 추가 학습 전/후 비교 | **미수행** | 직접 촬영한 소형 매장 영상이 아직 없다 |
| 배우 분리 평가 | **불가** | 배우 코드가 개인 식별자가 아니다 |
| 실제 RTSP IP 카메라 연결 | **미수행** | 이 PC에 접근 가능한 IP 카메라가 없다. 코드 경로는 구현·검토되었으나 실제 카메라로 확인하지 않았다 |
| 사람별 중복 알림 억제 | **구현 안 함** | 추적 ID가 없다. 억제 단위는 카메라 × 클래스다 |
| 사람 바운딩박스 표시 | **구현 안 함** | 전체 클립 분류 모델이라 특정 사람에게 행동을 귀속할 근거가 없다 |
| 모델 점수 보정(calibration) | **미수행** | 현재 점수는 확률이 아니다. UI/문서에 그렇게 명시했다 |
| 다중 카메라 동시 운용 부하 | **미측정** | 코드는 다중 카메라를 지원하지만 1대만 시험했다 |
| `split.mode: store` 매장 분리 평가 | **미수행** | 코드는 있으나 실행하지 않았다. 새 매장 일반화를 보려면 이것부터 돌려야 한다 |

### 다른 매장에서도 같은 성능이 나온다고 가정하지 않는다

이 데이터의 7개 매장은 모두 같은 촬영팀(DF2)이 같은 시나리오로 연출한 것이다.
조명, 매대 배치, 카메라 각도, 배우 행동 양식이 실제 매장과 다르다.

측정된 근거도 있다. 위 매장별 표에서 **같은 모델의 사건 탐지율이 0.813~0.982 로 갈린다.**
그것도 세 매장 모두 학습 데이터에 포함된 상태에서다.

따라서 다음 순서를 전제로 삼아야 한다.
1. `split.mode: store` 로 다시 학습·평가해 **매장 분리 조건에서의 하락 폭**을 먼저 잰다.
2. 실제 설치 매장에서 영상을 모아 **그 매장 데이터로 추가 학습**한다.
3. 추가 학습 전/후 성능을 따로 기록한다(같은 수치로 뭉뚱그리지 않는다).

---

## 6. 배포 시 주의 (외부망 접근)

매장 카메라에 외부에서 접근해야 할 때 **RTSP 포트(554)를 인터넷에 직접 열지 않는다.**
RTSP 는 기본 인증이 약하고 스캐너에 바로 노출된다. 다음 중 하나를 쓴다.

1. **VPN** (WireGuard / OpenVPN) — 매장 공유기에 VPN 서버를 두고 분석 PC가 VPN 으로 붙는다. 권장.
2. **제조사 P2P/클라우드 릴레이** — 간편하지만 제조사 서버를 거치므로 영상이 외부로 나간다.
3. **역방향 터널** (예: Tailscale, Cloudflare Tunnel) — 포트 개방 없이 연결.

어느 쪽이든 카메라 계정은 전용 계정을 만들고, 이 시스템에는 **읽기 전용 권한**만 준다.
`.env` 는 공유·커밋하지 않는다.

---

## 7. 코드 지도

```
storeguard/
  config.py            YAML 설정 로더 + 실행 환경 스냅샷
  naming.py            AI-Hub 파일명 규약 파서
  utils.py             시드/로깅/지표/URL 마스킹
  train.py             학습 (AMP, 불균형 대응, 체크포인트/재개, best 선택)
  evaluate.py          클립 단위 + 사건 단위 평가
  data/
    extract.py         배포 zip 압축 해제
    parse_cvat.py      CVAT XML 파서 (이벤트 구간 + 키포인트 + 객체 박스)
    build_index.py     영상↔라벨 연결, 영상 실측, take 그룹핑
    inspect.py         데이터 검사 리포트 생성
    split.py           take 단위 train/val/test 분할 + 누수 검사
    preprocess.py      프레임 캐시(.npy) 생성
    clips.py           인과적 클립 열거 + 라벨 규칙
    dataset.py         PyTorch Dataset
  models/factory.py    백본 팩토리 (r2plus1d_18 / mc3_18 / r3d_18 / s3d)
  infer/
    video_source.py    RTSP/MP4 수신 스레드 (버퍼, 폐기, 재접속, 상태)
    engine.py          모델 로딩 + 클립 추론 (미로딩 상태 명시)
    alerting.py        알림 상태 기계 (임계값·지속·종료·쿨다운)
    recorder.py        스냅샷 + 알림 전후 영상 (순환 버퍼, 비동기 쓰기)
    runner.py          카메라 1대 파이프라인 조립 + CLI
  server/
    app.py             FastAPI (상태, 이벤트, SSE, MJPEG, 임계값 API)
    db.py              SQLite 이벤트 저장소
    static/index.html  대시보드
  tools/
    selftest.py           torch 없이 도는 핵심 로직 검증 11항목
    vram_probe.py         VRAM/처리량 실측
    list_demo_videos.py   데모용 영상과 사건 발생 시각 찾기 (--env 로 .env 형식 출력)
    make_report.py        학습·평가·실시간 결과를 리포트 한 장으로
  data/add_normal.py      직접 촬영한 정상 영상 등록
  data/extract_paired.py  라벨과 짝 맞는 영상만 선별 추출 (클래스 균형 상한 포함)
  data/framestore.py      .npy / .jpz 두 캐시 형식 읽기
  data/pack_cache.py      캐시를 JPEG 로 포장 (24GB → 5GB)
  server/playlist.py      대시보드 재생목록 (다음 영상 버튼)
  tools/false_alarm.py    정상 영상에서의 시간당 오경보 측정
configs/default.yaml   모든 하이퍼파라미터
scripts/quickstart.ps1   전체 파이프라인 실행 순서
scripts/run_dashboard.cmd 대시보드 실행 (실행 정책 영향 없음)
scripts/run_dashboard.ps1 대시보드 실행 (PowerShell 판)
.env                   카메라 주소 (여기를 고쳐서 영상을 바꾼다)
```
