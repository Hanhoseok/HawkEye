# HawkEye — 무인매장 절도 위험 탐지

2026 2학기 캡스톤

> **위치:** 이 README는 저장소의 `experiments/tracking/` 폴더 문서다 (담당: 왕규원).
> 아래의 설치·실행 명령은 모두 **이 폴더에서** 실행한다 → `cd experiments/tracking`

> CCTV 영상에서 고객을 **매장 입장부터 퇴장까지** 추적하고, 선반에서 상품을 집는 행동을 탐지해
> **미결제 상태로 출구에 접근하는 상황**을 위험으로 표시하는 시스템.
> 사람을 "절도범"으로 확정하는 것이 아니라 위험 상황을 표시하는 것이 목적이다.

## 현재 상태

> **최신 결과 종합: [`docs/results-summary.md`](docs/results-summary.md)** — 지금 상태(카메라 2대, 2차 촬영 38쌍)와 1차 평가(카메라 1대, 40편), 다음 단계를 한 장에.
> **구현 설명: [`docs/architecture.md`](docs/architecture.md)** — 프로그램이 어떻게 돌아가는지 (구조·모델·판단 규칙, 카메라 2대는 §6).

**지금 (2026-11-10)**: 카메라 2대(전체 카메라 + 선반 카메라) 구조로 2차 촬영 38쌍을 평가하고 있다.
기본 설정은 관절 모델 + 발목 + 직접 그린 구역(11-09). 판정은 **31/38** (정상 11/12 · 절도 4/9 · 복합 정상 16/17, 손님 수 38/38),
사람이 적은 결제 라벨을 넣으면 30/38 — 결제가 정확해도 집은 개수를 틀리면 판정이 틀린다.

| 카메라 2대 (2차 촬영 38쌍) | 상태 | 문서 |
|---|---|---|
| 두 영상 함께 처리 · 시차 · 바닥 지도 구역 | ✅ 시차는 32쌍을 사람이 맞춤, 바닥 보정점은 추정값 | `docs/multiview-progress.md`, `docs/zones-store01.md` |
| 손님 수 · 신원 | ✅ 38/38 쌍 (가장자리 변 + 되살아난 번호 다시 판정). 비슷한 옷 2명은 약함 | `docs/multiview-progress.md` 후속 1·2 |
| 두 화면 사람 잇기 | 🔶 0.5초 조각의 55% 가 짝을 맺음. 프레임 단위 정답이 없어 정확도는 못 잼 | `docs/multiview-progress.md` 후속 6 |
| 선반 사건 (손-칸 접촉 · 선반 지도 · 칸 그림 비교) | 🔶 병 무리에서 1개 빠진 것을 놓침(T2) | `docs/multiview-progress.md` 후속 4 |
| 절도 판정 | 🔶 절도 4/9. 판정이 안 나온 쌍 8개(영상 끝) | `docs/results-summary.md` §0 |
| 결제 | 🔶 계산대 체류 = 결제. 결제 라벨 도구(`scripts/label_payments.py`)로 재평가, 실제 결제 연동은 아직 | `docs/multiview-progress.md` 후속 8 |
| 선반 센서 | ⬜ 입력 자리와 가짜 기록 시험만. 실제 센서(`experiments/arduino`) 미연결 | `docs/multiview-design.md` M-8 |
| 실시간 | 🔶 녹화본을 카메라 속도로 흘린 흉내만. 실제 카메라 미확인, 카메라당 초당 4~5장 | `docs/multiview-progress.md` 1-7 |
| 확인 도구 | ✅ **분석 도구** (`python workbench/server.py`) — 갤러리 + 질문별 한 쌍 분석 + 수치 조정 + 정답 적기 + 새 영상 추가. 예전 페이지: 오판 장면 보기 · 구역 편집 (`python scripts/review_serve.py`) | [`workbench/README.md`](workbench/README.md), `docs/multiview-progress.md` 후속 4·7·10·11·12·13 |

아래 표는 층별 상태다 (카메라 1대 기준, 2026-10-14 까지).

| 계층 | 상태 | 문서 |
|---|---|---|
| 영상 입력 | ✅ | |
| 사람 탐지 (YOLO / YOLO-pose) | ✅ | `docs/scale-limits.md` |
| 추적 (ByteTrack / BoT-SORT + ReID) | ✅ | `docs/tracker-comparison.md` |
| 매장 단위 신원 (계층 2) | ✅ tracker 버퍼의 3.3배(10초) 공백 복원 · 천장 시점 분열을 "나중에 합치기"로 해결 | `docs/identity-registry.md` |
| 구역 (SHELF / EXIT) | ✅ | `docs/zones.md` |
| TAKE 후보 (계층 3) | 🔶 MERL test 정밀도 0.75 / 재현율 0.71 / F1 0.73 / 국소화 37% | `docs/merl-evaluation.md` |
| 손님 상태 · 결제 연결 · 출구 판정 | 🔶 경보 두 단계(WARNING → HIGH_RISK) + 판정 직전 신원 확인. 실제 CCTV 2편에서 정답 일치, 계산하고 나가는 경우는 미검증 | `docs/risk-pipeline.md` |
| 경보 전송 (관리자 앱·대시보드) | ✅ app-pos 경보 서버로 판정 + 장면 사진 전송. UCF 039 로 끝까지 확인 | `docs/risk-pipeline.md` §6-B |
| 선반 지도 (물건 수) | 🔶 선반에서 줄어든 개수를 손님 기록에 연결. 매장 영상: 병 8/8, 작은 흰 물건 4/15 (2026-10-14) | `docs/shelf-map.md`, `docs/store-test.md` |
| 집음 / 놓음 / 만짐 판정 | 🔶 손 뻗기 + 선반 변화. 매장 영상 손 뻗기 21/25 | `docs/shelf-map.md` 1-C |
| 결제 (기록 없는 매장) | ✅ 계산대 체류 = 결제, 선반으로 돌아가면 취소 (`--pay-by-dwell`) | `docs/store-test.md` 3차 |
| RETURN (되돌려놓기) | ⬜ 미착수 → 🔶 2026-10-06: 지켜보는 선반에서는 '다시 참'으로 개수에서 뺌. 그 밖은 미착수 | `docs/shelf-map.md` |

**처음 보는 사람이 읽는 순서**: `docs/results-summary.md`(지금 결과) → `docs/architecture.md`(어떻게 돌아가나) → `docs/limitations.md`(무엇이 안 되나).
왜 이런 구조가 됐는지는 `docs/진행경과-정리.md` — 9월 추적기 단계까지 자세히, 그 뒤는 맨 끝 §12 의 날짜표로 각 문서를 가리킨다.
**지금 무엇이 안 되는지는 `docs/limitations.md`** — 부딪힌 한계와 실패를 모두 모아 상태(미해결·우회·해결)로 관리한다.

---

## 설치

Python 3.11 기준. GPU(CUDA)가 없으면 CPU 휠로 설치한다.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate         # macOS / Linux

pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
python -m pytest tests -q          # 설치 확인
```

모델 가중치(`yolov8n.pt`, `yolov8n-pose.pt`, ReID `osnet_x0_25_msmt17.pt`)는 처음 실행할 때 자동으로 받아진다.

> 프로젝트 폴더가 OneDrive 같은 동기화 폴더 안에 있으면, 가상환경은 **폴더 밖**에 만드는 편이 낫다.
> torch 가 수천 개 파일이라 동기화가 매우 느려진다.

---

## 테스트 데이터

영상은 저장소에 포함하지 않는다. 출처마다 라이선스가 달라 재배포하지 않고, 각자 받는다.

| 파일 | 출처 | 용도 | 라이선스 |
|---|---|---|---|
| `store-aisle-detection.mp4` | [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) | 매장 통로 (주 실험, 비스듬한 각도) | 저장소 라이선스 참조 |
| `people-detection.mp4` | 위와 동일 | 사람 교차 실험 | 위와 동일 |
| `ThreePastShop2cor.mpg` | [CAVIAR](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/) | 쇼핑센터 실제 CCTV | CC BY-SA |
| `topdown-plaza.mp4` | [saimj7/People-Counting-in-Real-Time](https://github.com/saimj7/People-Counting-in-Real-Time) `utils/data/tests/test_1.mp4` | 거의 수직 하향 시점 | 저장소 라이선스 참조 |
| MERL Shopping (106개) | [MERL](https://www.merl.com/research/downloads/MERL_Shopping_Dataset) | **라벨 5,377건. 평가의 기준** | **비상업 연구용** |

```bash
# 매장 통로 / 교차 실험
curl -L -o data/videos/store-aisle-detection.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4
curl -L -o data/videos/people-detection.mp4      https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4

# MERL Shopping (영상 1.76GB + 라벨 0.1MB) -> data/merl/ 에 압축 해제
curl -LO https://www.merl.com/pub/tmarks/MERL_Shopping_Dataset/Labels_MERL_Shopping_Dataset.zip
curl -LO https://www.merl.com/pub/tmarks/MERL_Shopping_Dataset/Videos_MERL_Shopping_Dataset.zip
```

`labels/store-aisle.csv` 는 store-aisle 영상에 직접 찍은 행동 라벨(TAKE 19 / BROWSE 14 / RETURN 19)이다.

---

## 1. 빠른 실행

기본값이 `botsort`(외형 ReID 포함)라 별도 인자 없이 그대로 쓰면 된다.
빠른 반복이 필요할 때만 가벼운 tracker 로 바꾼다:

```bash
python run_tracking.py --source data/videos/store-aisle-detection.mp4 --tracker bytetrack
```

| tracker | 매칭 근거 | 재등장 재식별 | 교차 시 ID 유지 | 속도 |
|---|---|---|---|---|
| `bytetrack` | 칼만 예측 + IoU | ❌ | ❌ | 8.3 FPS |
| `botsort` (boxmot 기본값) | 위 + 외형(ReID) | ❌ | ✅ | 5.9 FPS |
| **`botsort` (현재 기본값)** | 위 + 외형(ReID) | ✅ | ✅ | 5.9 FPS |

`--lost-buffer` 는 **재등장 공백보다 커야** 한다. 위 실험에서는 90 이 필요했다.

```bash
# 1) 가상환경 활성화 (위 "설치" 참조)

# 2) 테스트 영상을 data/videos/ 에 넣고
python run_tracking.py --source data/videos/sample.mp4

# 3) 빠르게 확인만 하고 싶을 때
python run_tracking.py --source data/videos/sample.mp4 --max-frames 100 --show

# 4) 계층 분리 규칙 + 파이프라인 스모크 테스트
python -m pytest tests -q
```

결과물은 `outputs/` 에 생긴다.

| 파일 | 내용 |
|---|---|
| `outputs/tracked.mp4` | bbox + track_id + 이동 궤적이 그려진 결과 영상 |
| `outputs/observations.jsonl` | 표준 `TrackObservation` 로그 (한 줄에 하나) |

결과를 track 단위로 요약하려면:

```bash
python scripts/analyze_observations.py outputs/observations.jsonl --min-frames 10
```

구간을 지정해 실험할 때 (Phase 3):

```bash
python run_tracking.py --source data/videos/store-aisle-detection.mp4   --start-frame 1200 --stride 2 --tracker-fps 30 --max-frames 200   --out-dir outputs/store-aisle
```

`--stride` 로 프레임을 건너뛸 때는 `--tracker-fps` 도 함께 맞춘다.
`lost_track_buffer` 가 **프레임 단위**라 실효 fps 가 달라지면 의미가 바뀌기 때문이다.

---

## 2. 구조 — 한 장면이 6개 층을 지난다

> **자세한 구현 설명: [`docs/architecture.md`](docs/architecture.md)** — 모델을 어떻게 쓰는지, 층별 판단 규칙과 실제 기준값, 설정·실행, 파일 지도.

```
영상 한 장면
 ├─ ① 사람 찾기·잇기   YOLO(사람) → BoT-SORT                   → TrackObservation (추적 번호)
 ├─ ② 매장 단위 신원   끊겨도·번호가 바뀌어도 같은 손님          → IdentityObservation (손님 번호)
 ├─ ③ 손 뻗기          선반 구역 앞 멈춤 / 손목-선반 거리        → TakeCandidate
 └─ ④ 선반 지도        YOLO(물건) → 지도 → 가림 → 떠난 뒤 비교   → ShelfVisit (사라짐·생김·변화 없음)
      ⑤ 손님 장부      ③+④ → 집음/놓음/만지기만, 물건 수, 결제(기록 또는 계산대 체류)
      ⑥ 위험 판정      출구 접근 WARNING → 나감 확정 → HIGH_RISK / REVIEW / CLEAR → RiskEvent
      출력             기록 파일 · 결과 영상 · 상황판(--show) · 관리자 앱 경보(app-pos)
```

핵심 규칙 세 가지:

1. **층 사이에는 `src/core/types.py` 의 정해진 형식만 오간다.** 그래서 한 층을 갈아 끼워도 나머지는 그대로다
   (추적기 ByteTrack → BoT-SORT 교체 때 다른 층은 손대지 않음, 일반 YOLO → 상품 모델도 ④의 탐지기만 바꾸면 됨). 형식을 바꾸면 팀에 공유한다.
2. **`src/core/` 와 판단 층 `src/risk/` 는 cv2·torch·ultralytics 를 import 하지 않는다** (`tests/test_contract.py` 가 검사).
   그래서 판단 규칙은 영상 없이 테스트하고, 저장된 기록으로 다시 돌릴 수 있다.
3. **기준값과 모델 경로는 `config.yaml` 에만 있다.** 모델은 공개 사전학습 모델 그대로 쓰고, 판단은 규칙이다.

---

## 3. 디렉터리

```
run_tracking.py            진입점 (config.yaml + 명령행 인자, 카메라 2대는 --dual-config)
config.yaml                모든 설정값 (카메라 1대 기본)
configs/store01_dual.yaml  카메라 2대 설정 — config.yaml 위에 얹는다
configs/store01_dual_judge.yaml  (있을 때만) 분석 도구의 수치 조정에서 적용한 판정 값 — 평가·실행이 함께 읽는다
zones/                     카메라별 구역 파일 (SHELF / CHECKOUT / EXIT / OUTSIDE), store01_dual/ 은 카메라 2대(바닥 지도·칸)
src/
  core/                    types.py(층 사이 형식) · config.py(설정) · protocols.py(인터페이스)
  inputs/                  영상 읽기, 두 카메라 시차(sync.py), 실시간 입력(live_source.py)
  detectors/               ① 사람 탐지(YOLO / YOLO-pose), 선반 물건 탐지기, 사람 아님 영역(masked.py)
  trackers/                ① BoT-SORT / ByteTrack 어댑터, 교체 지점 factory.py
  identity/                ② 매장 단위 신원, 외형 모델 어댑터
  zones/                   구역 읽기, 사람-구역 관계, 바닥 지도(floor.py)
  interaction/             ③ 손 뻗기, 자세 특징
  shelf/                   ④ 선반 지도(monitor.py), 파이프라인 연결(watcher.py), 칸 그림 비교(slot_diff.py)
  fusion/                  ⑦ 두 화면 사람 잇기(cross_view.py), 선반 사건 묶기·센서(shelf_events.py), 합친 판정(judge.py)
  risk/                    ⑤ 손님 장부·결제(customers.py, payments.py), ⑥ 위험 판정(engine.py)
  sinks/                   기록 파일, 결과 영상, 경보 전송(alert_sink.py)
  viz/                     사람 상자, 선반 지도, 상황판
  camera_worker.py         카메라 한 대 몫의 ①~④
  pipeline.py              전체 조립 (카메라 1대)
  multicam.py              전체 조립 (카메라 2대)
scripts/                   평가·실험 도구 — 아래 표
workbench/                 분석 도구 — 기록을 영상 위에 그려 질문별로 보는 웹 화면 (workbench/README.md)
tests/                     층별 규칙 테스트
labels/                    정답표 (store01 1차 촬영, store01_dual 2차 촬영: 정답·시차·결제 라벨·가짜 센서, events/ 는 분석 도구로 적는 집음·짝·경보 대상)
data/videos/, outputs/     입력 영상·결과물 (git 제외)
```

파일별 설명은 `docs/architecture.md` 5절.

### scripts — 쓰임별로

파일은 모두 `scripts/` 한 폴더에 있다. 문서 곳곳의 실행 명령이 이 경로를 쓰고 있어 폴더는 나누지 않고 여기서 분류만 한다.

| 쓰임 | 파일 | 한 줄 |
|---|---|---|
| **카메라 2대 평가** | `dual_eval.py` | 2차 촬영 38쌍: `run`(기록) · `sync`(시차) · `calibrate`(띠 축) · `judge`(판정만 다시) |
| | `sync_pair.py` · `calibrate_floor.py` | 두 화면을 보며 시차 확정 · 바닥 보정점으로 바닥 지도 만들기 |
| | `label_payments.py` | 두 화면을 보며 결제 시각·개수·시차를 적는 도구 (`labels/store01_dual/payments.csv`) |
| | `fake_sensor.py` · `live_sim.py` | 가짜 선반 센서 기록 만들기 · 녹화본을 카메라 속도로 흘리는 실시간 흉내 |
| **확인 페이지** | `review_serve.py` | 오판 장면 보기·구역 편집 페이지를 여는 로컬 서버 (내 컴퓨터에서만 열림) |
| | `review_export.py` · `review_viewer.html` · `review_cases.js` | 오판 장면 보기 — 기록을 다시 판정하며 상자·신원·관절·짝·판정을 원본 영상 위에 그림 |
| | `zone_editor.py` · `zone_editor.html` | 구역 편집 — 영상 위에서 구역을 고치고 초안으로 38쌍 판정을 돌려 비교한 뒤 적용 |
| | `review_swapcheck.py` | 두 손님이 겹친 전후로 신원이 뒤바뀌었는지 몸통 색으로 점검 |
| | `link_diagnose.py` | 두 화면 짝이 왜 끊기나 — 조각마다 이유(`why`) · 위치값 견주기(`fit`) · 해결 후보 견주기(`try`). `--shoot 3` 은 바닥 위 거리 방식: 허용 거리의 근거(`fit`) · 띠/바닥 견주기(`try`) · 두 명 장면에서 맺은 짝이 맞는 사람인가(`check`). 기록만 다시 돌리고 아무것도 고치지 않음 |
| | `floor_check.py` | 3차: 두 카메라의 바닥 지도가 같은 좌표인가 — 타일 줄눈 어긋남(`tiles`) · 같은 사람의 두 카메라 발목 거리(`feet`) |
| **1차 매장 영상 (카메라 1대)** | `store_eval.py` | 1차 촬영 40편 일괄 평가: `run` · `judge` |
| | `transfer_zones.py` · `screen_videos.py` | 카메라가 조금 움직였을 때 구역 옮기기 · 검증 영상 고르기(사람·물건이 얼마나 보이나) |
| **기록으로 다시 돌리기 (영상 없이)** | `replay_identity.py` · `replay_risk.py` | 저장된 추적 기록으로 신원만 · 신원 기록으로 판정만 다시 |
| | `take_sweep.py` · `identity_calibration.py` · `remap_identities.py` | 손 뻗기 기준값 훑기 · 신원 판정 방식 비교 · 두 실행 사이 손님 번호 대응표 |
| **공개 데이터 평가** | `merl_to_labels.py` · `merl_batch.py` · `merl_eval.py` · `merl_feature_check.py` | MERL: 라벨 변환 · 묶어 돌리기 · train/test 평가 · 자세 특징 판별력 |
| | `retailaction_eval.py` · `retailaction_hands.py` · `eval_takes.py` | RetailAction: 자세만으로 집기 · 손 사진으로 집음/놓음 · 손 뻗기 후보를 라벨과 대조 |
| **라벨·구역 준비** | `label_actions.py` · `export_frame.py` · `zone_report.py` | 행동 구간 라벨링 · 격자 입힌 장면 저장 · 신원별 구역 체류 집계 |
| **추적 분석 (초기 단계)** | `analyze_observations.py` · `detect_id_jumps.py` · `find_crossings.py` | 추적 번호 단위 요약 · 번호가 남에게 옮겨 간 지점 · 두 사람이 겹친 구간 |
| **시연·시험용** | `shelf_demo.py` · `make_sample_video.py` | 선반 지도 1단계 시연 · 영상 입출력만 확인하는 합성 영상 |

---

## 4. 문서

| 문서 | 내용 |
|---|---|
| **`docs/results-summary.md`** | **결과 종합 — 팀원 공유용 한 장. 지금 상태(카메라 2대)와 1차 평가** |
| **`docs/architecture.md`** | **구현 설명 — 구조, 모델 사용, 판단 규칙과 기준값, 설정·실행, 파일 지도** |
| **`docs/진행경과-정리.md`** | 9월 추적기 단계 정리 — 왜 ByteTrack에서 BoT-SORT로 갔는지, 수치 포함. 맨 끝 §12 에 그 뒤 경과 날짜표 |
| **`docs/multiview-design.md`** | **카메라 2대 + 실시간 + 선반 센서 구조 설계 (M-1~M-12, 단계별 계획)** |
| `docs/multiview-papers.md` | 위 설계의 근거 논문 6편(P1~P6) |
| **`docs/multiview-progress.md`** | **카메라 2대 1단계 진행 기록 — 2차 촬영 38쌍 결과, 후속 1~13. 맨 위에 지금 상태** |
| **`docs/limitations.md`** | **한계 총정리 — 부딪힌 한계·실패와 현재 상태. 해결돼도 지우지 않는다** |
| **`docs/store-test.md`** | **직접 찍은 무인매장 영상 첫 시험 — 정상 2편으로 설정, 절도 2편으로 시험** |
| **`docs/validation-set.md`** | **검증 영상 세트 — 기능별 고르는 기준, 가진 영상 149편 분류, 촬영할 선반 장면** |
| `docs/phase3-failure-notes.md` | tracker 실패 실험 원본 기록 |
| `docs/tracker-comparison.md` | ByteTrack vs BoT-SORT 9회 실험 비교 |
| `docs/phase5-take-return-survey.md` | TAKE/RETURN 후보 사전조사 |
| **`docs/scale-limits.md`** | **사람이 몇 px 이하면 무너지는가 → 카메라 설치 기준, 화각별 검증** |
| **`docs/identity-registry.md`** | **계층 2 — 매장 단위 신원. tracker 가 끊겨도 person_id 유지** |
| **`docs/zones.md`** | **구역 인프라 — SHELF/EXIT 정의와 판정 방식** |
| **`docs/zones-store01.md`** | **카메라 2대 구역 — 구역마다 코드에서 하는 일, 구역 편집 페이지(`scripts/zone_editor.html`)** |
| **`docs/take-candidates.md`** | **계층 3 — 선반 앞 멈춤으로 TAKE 후보 판정, 그 한계** |
| **`docs/labeling.md`** | **행동 라벨링 — 무엇을 왜 어떻게 찍는가** |
| **`docs/pose-signals.md`** | **Pose 신호 — 손 높이는 되고 팔 뻗음·손 위치는 안 된다** |
| **`docs/merl-evaluation.md`** | **MERL 공개 데이터로 train/test 분리 평가. 특징이 화각에 종속적임을 확인** |
| `docs/retailaction-eval.md` | RetailAction 으로 자세만의 집기 판정 확인 — 선반 구역 없이는 약함 |
| `docs/shelf-map.md` | 선반 지도 — 어느 자리의 물건이 사라졌고 그때 누가 있었나 (1단계) |
| **`docs/risk-pipeline.md`** | **손님 상태 · 결제 연결 · 출구 판정 — 입장부터 경보까지** |
| **`docs/filming-guide.md`** | **실제 무인매장 촬영 설계 — 답사·카메라 2대·시험 선반·손님 카드·정답표** |

## 5. Phase 진행 체크리스트

문서 §6 의 "권장 구현 순서와 확인 항목"을 그대로 옮긴 것이다.

> 9월 초기 단계(영상 입출력 ~ 추적기 교체)의 기록이다. 그 뒤 단계는 여기에 더하지 않았다 —
> 손 뻗기·선반 지도·판정은 위 '현재 상태' 표, 카메라 2대 1단계(1-0~1-7)와 2단계 계획은 `docs/multiview-design.md` §6 을 본다.

- [x] **Phase 0. 영상 I/O** — 원본 영상이 끝까지 처리되고 출력 영상이 생성됨
- [x] **Phase 1. YOLO** — `store-aisle-detection.mp4` 에서 person bbox 확인 (conf 0.8~0.9)
- [x] **Phase 2. ByteTrack** — 약 6.6초 동안 track 1·2 가 동일 ID 유지
- [~] **Phase 3. 실패 실험** — 두 가지 실패를 분리해 확인 (`docs/phase3-failure-notes.md`)
      - **분열**(재등장 시 ID 가 쪼개짐): ByteTrack ❌ → BoT-SORT+ReID 로 해결 ✅
      - **뒤바뀜**(교차 시 ID 가 남에게 옮겨감): ByteTrack ❌ → BoT-SORT 기본값으로 해결 ✅
      - 남은 것: 빠른 이동, **혼잡·유사 복장에서의 오병합(가장 중요)**
- [x] **Phase 4. 인터페이스 고정** — tracker 를 ByteTrack ↔ BoT-SORT 로 바꿔도
      `pipeline.py` / `TrackObservation` / 분석 스크립트가 전혀 바뀌지 않음을 실제 교체로 증명
- [~] **Phase 5. 다음 방법 탐색** — 후보를 실제 영상에서 측정해 비교 (`docs/phase5-take-return-survey.md`)
      - 상품 탐지 ✅ / 손목 keypoint ✅ (이 화각 기준) / **상품 개별 추적 ❌ (동일 상품 간 ID 건너뜀)**
      - 결론: **Zone/거리 규칙을 baseline 으로 채택**, Pose 는 2순위

Phase 3 의 기록 양식은 `docs/phase3-failure-notes.md` 에 만들어 두었다. 실험하면서 표를 채우면 된다.

---

## 6. 환경 메모

- GPU: **Intel Arc 140V** → CUDA 사용 불가. torch **CPU 휠**로 설치했고 `config.yaml` 의 `device: cpu`.
- 느리면: `stride: 2~3` 으로 프레임을 건너뛰거나, `imgsz: 480` 으로 낮추거나, 나중에 YOLO 를 **OpenVINO** 로 export 해 Intel GPU 가속을 검토한다.
- 가상환경은 동기화 폴더(OneDrive 등) 밖에 두는 편이 낫다. torch 를 동기화하면 매우 느려진다.
- 설치 확인된 버전: torch 2.14.0+cpu · ultralytics 8.3.40 · supervision 0.25.1 · opencv 4.10.0
- 측정: 720x404 실영상 / yolov8n / imgsz 640 / CPU → bytetrack 8.3 FPS, botsort(ReID) 5.9 FPS.
- 기본 tracker 는 `botsort` 다. 개발 중 빠른 반복이 필요하면 `--tracker bytetrack`.
- **설치 기준**: 모델 입력 텐서 속 사람 키가 **90px 이상**이어야 정상, 65px 미만이면 조치 필요.
  `텐서 픽셀 = 화면 속 사람 키 × imgsz ÷ 프레임 긴 변`. 자세한 근거는 `docs/scale-limits.md`.
- **`imgsz` 는 프레임의 긴 변까지만 올릴 가치가 있다.** 그 이상은 보간일 뿐이고
  실측에서 끊김이 2배 늘었다(384x288 영상, 22 → 52회).

---

## 7. 지금 구현하지 않은 것 (의도적)

상품별 tracking / TAKE·RETURN 판정 / POS 매칭 / HIGH_RISK 판정 / YOLO 재학습.
여러 문제를 동시에 섞으면 **어느 단계에서 실패했는지 판단할 수 없기 때문**이다. (문서 §5)

> 2026-10-06 갱신: 그 뒤 단계별로 검증하며 **TAKE 판정(후보)·POS 매칭·HIGH_RISK 판정**을 구현했고, 되돌려놓기는 선반 지도로 일부 다룬다.
> **상품별 tracking 과 YOLO 재학습은 여전히 하지 않았다** (상품 개별 추적은 측정 결과 실패 — `docs/limitations.md` '시도했지만 접은 방법').

---

## 8. 실험 기록

무엇을 바꿨고 결과가 어땠는지. 자세한 수치와 근거는 오른쪽 문서에 있다.

| 날짜 | 무엇을 | 결과 | 문서 |
|---|---|---|---|
| 2026-09 | 집기(TAKE) 판정을 MERL 공개 데이터로 평가 (train 1~20번 / test 27~41번 분리) | 정밀도 0.75 / 재현율 0.62 / F1 0.68. 카메라 각도마다 쓸 수 있는 단서가 정반대임을 확인 | `docs/merl-evaluation.md` |
| 2026-09-29 | 천장 시점 신원 쪼개짐 → "나중에 합치기" + 화면 가장자리 규칙 | 재현율 0.62 → **0.71**, F1 0.68 → **0.73**. 여러 사람 영상 4편에서 오병합 없음 | `docs/identity-registry.md` §3-D |
| 2026-09-29 | 손님 장부 · 결제 연결 · 출구 판정 추가 (입장~경보 처음 연결) | store-aisle 에서 결제 조건별 HIGH_RISK / REVIEW / CLEAR 가 의도대로. TAKE 후보 수 ≠ 물건 수라 REVIEW 단계 추가 | `docs/risk-pipeline.md` |
| 2026-09-29 | UCF-Crime 실제 절도 CCTV 3편 (031, 039, 047) | 031: 문이 안 보여 출구 판정 검증 불가. 039: BoT-SORT 는 정답 일치, ByteTrack 은 번호 뒤바뀜으로 오경보. 047(혼잡): 아무도 안 나갔는데 HIGH_RISK 2건 | `docs/risk-pipeline.md` §6 |
| 2026-09-29 | 경보 두 단계 + 판정 직전 신원 확인 + 문 밖(OUTSIDE) 구역 | 039 정답 유지(도둑만 WARNING → HIGH_RISK), 047 오경보 2 → **0** (한 건은 기준에 아슬아슬). 두 영상은 규칙을 만드는 데 쓰여 독립 검증 아님 | `docs/risk-pipeline.md` §5-A·B |

| 2026-10-01 | RetailAction(실제 편의점 10곳, 천장)으로 "선반 구역 없이 자세만으로" 집기 찾기 | 집는 순간 짚기 45% (아무 순간 36%), 행동 80% 잡을 때 아무것도 안 한 손님 40% 헛잡음. **자세만으로는 선반 구역을 대신 못 함** | `docs/retailaction-eval.md` |
| 2026-10-01 | 선반 지도 1단계: 선반 물건 배치를 기억했다가 사람이 다녀간 뒤 어느 자리가 비었는지 찾아 그 사람에게 붙임 (일반 YOLO) | 스톡 영상 2편에서 사라진 물건 자리 모두 정답. 1편은 손님까지 정답, 1편은 팔만 보여 주인 미정 | `docs/shelf-map.md` |
| 2026-10-06 | 위험 판정을 app-pos 경보 서버·대시보드로 전송 (장면 사진 포함) | UCF-Crime 039 전체 파이프라인: 판정 6건 전송, 도둑만 HIGH_RISK 사건·나머지 통과로 대시보드에 표시 | `docs/risk-pipeline.md` §6-B |
| 2026-10-06 | 선반 확인 개수를 손님 기록에 연결 — '몇 번 집었나' 대신 '몇 개 가져갔나' | Pexels 2편 전체 파이프라인: 가져간 병을 손님 1 에게 1개로 붙임, 들었다 놓은 통은 안 셈. CPU 13.8 → 4.8 FPS | `docs/shelf-map.md` 손님 기록 연결 |
| 2026-10-06 | 한계 총정리 문서 — 문서 13개의 한계·실패 기록을 모아 상태로 관리 (해결돼도 지우지 않음) | 항목 69개, 시도했지만 접은 방법 10개 | `docs/limitations.md` |
| 2026-10-06 | 검증 영상 선정 — 사람 추적 / 집기 / 선반 물건 개수별 기준을 정하고 가진 영상 149편을 자동 거름망 + 눈으로 분류 | 개수 세기 검증 가능한 영상은 Pexels 같은 촬영본 3편뿐(새로 추가한 10566653 도 정답 일치). 실제 CCTV(UCF·MERL)는 일반 YOLO 로 상품을 하나씩 못 잡음 → 촬영 장면 10개 제안 | `docs/validation-set.md` |
| 2026-10-07 | 집음 / 놓음 / 만짐 행동 판정 — 손 뻗기 + 선반 변화를 합쳐 손님 행동을 기록 (`actions.jsonl`) | Pexels 3편 모두 '집음 1개' 정답 일치. 놓음·만지기만은 영상이 없어 테스트로만 확인, 방문 안의 들었다 놓기는 안 보임 | `docs/shelf-map.md` 1-C |
| 2026-10-07 | 손에 물건이 있나(손 부분 사진, DINOv2-small)로 집음/놓음 가르기 — RetailAction test | 손 자리 특징 AUC 0.71 (관절만 0.60). 놓음 절반 잡을 때 집음 29% 오판으로 아직 못 씀. 학습 데이터가 늘수록 계속 오름 → train 필요 | `docs/retailaction-eval.md` 2차 |
| 2026-10-07 | 실시간 상황판(`--show`/`--panel`) + UCF 039 재검증, 촬영 설계 개편 | 039: 도둑만 WARNING(75.2초) → HIGH_RISK(80.5초), 4명 CLEAR — 이전과 같음. 촬영 설계: 세트·카메라·진열 조건 + 장면 24개(★★★ 7개) + 정답표 3장 | `docs/filming-guide.md` |
| 2026-10-07 | 촬영 설계를 실제 무인매장 촬영으로 개편 | 카메라 2대(매장 CCTV 그대로 + 우리 카메라), 시험 선반 한 칸, 손님 카드 16장(★★★ 7장), 매장 결제 기록·CCTV 녹화 받기 | `docs/filming-guide.md` |
| 2026-10-14 | 선반 물건 탐지를 필요할 때만 + 선반 지도 화면 표시 | Pexels 3편 판정 같음, 4.8 → 10~11 FPS (탐지 58~81% 건너뜀). 화면에 확정 물건·가림·사라진 자리(TAKEN) 표시 | `docs/shelf-map.md` 속도 개선 |
| 2026-10-14 | 직접 찍은 무인매장 영상 첫 시험 (정상 N01·N01-1 로 설정, 절도 T01·T02 는 설정 그대로 시험) | 정상 2편 CLEAR(결제 연결), T01 HIGH_RISK ✅, **T02 놓침**(선반 유령 '놓음'이 집기를 지움). 영상 끝 규칙·구역 옮기기 도구 추가 | `docs/store-test.md` |
| 2026-10-14 | 같은 매장 40편 평가 — 정상 13편으로 보정·고정, 절도 10·복합 정상 17편 시험 (`scripts/store_eval.py`) | **절도 4/10, 복합 정상 4/17 (거짓 경보 13)**. 원인: 선반 '그대로'가 진짜 집기를 지움(6), 구경도 머묾으로 경보(11). 선반 없이 손 뻗기만이면 절도 9/10 이지만 거짓 경보는 그대로 | `docs/store-test.md` 2차 |
| 2026-10-14 | 기능별 평가 (C 는 추적용, 결제 = 계산대 체류·선반으로 돌아가면 취소) | 추적: N·T 23/23, C 깨끗 9/17(뒤바뀜 5·쪼개짐 3, 재입장 2/4). 손 뻗기 21/25. 선반 사라짐: 병 8/8·흰 상자 4/10·흰 카드 0/5. 절도 4/10 (선반이 못 보는 물건이면 손 뻗기를 지움) | `docs/store-test.md` 3차 |
| 2026-11-03 | 카메라 2대 구조 설계(논문 6편 근거)와 1단계 구현 — 카메라 작업자, 바닥 지도 구역, 두 화면 사람 잇기, 칸 그림 비교, 선반 사건 묶기, 가짜 센서, 실시간 입력. 2차 촬영 38쌍 평가 (`scripts/dual_eval.py`) | 정상 11/12 · 절도 6/9 · 복합 정상 15/17, 손님 수 12/38. '매장 안 바닥 1초 = 손님'으로 손님 수 36/38, 절도는 5/9 (T2 는 우연히 맞았던 것) | `docs/multiview-design.md`, `docs/multiview-progress.md` |
| 2026-11-04 | 재입장·신원 조각 조사 — 카메라별 가장자리 변 + 되살아난 번호 다시 판정. 전체 카메라 관절·팔 뻗기 시험 | 손님 수 36 → **38/38**, 판정 그대로 → 기본 채택. 팔 뻗기는 복합 정상 거짓 경보 4개라 기본 꺼짐 | `docs/multiview-progress.md` 후속 2·3 |
| 2026-11-05 | 오판 장면 보기 페이지 — 기록을 원본 영상 위에 그려 틀린 장면 확인. 겹침 뒤바뀜·가짜 사람 점검, 임시 신원 시험 | T2 는 가림이 아니라 병 무리 중 1개 변화가 기준(8%)에 0.7%p 모자람. 겹친 62곳 모두 번호 그대로. T9 는 맞혔지만 이유가 틀림. 임시 신원은 C11 좋아지고 C14 나빠져 기본 끔 | `docs/multiview-progress.md` 후속 4·5 |
| 2026-11-07 | 두 화면 짝: 선반 앞 후보가 한 명이면 외형 0.85 이상으로 잇기 | 짝 47 → 55%, 판정 그대로, C14·N5 방문 주인 고쳐짐 | `docs/multiview-progress.md` 후속 6 |
| 2026-11-08 ~ 09 | 구역 편집 페이지, 발목 발 점, 관절 모델로 38쌍 다시 돌리기 | 구역을 실제 자리에 맞추면 상자 발 점으로는 27/38, 발목과 함께 31/38. **카메라 2대 기본을 관절 모델 + 발목 + 직접 그린 구역으로 바꿈** — 31/38 (절도 4/9, T8 은 결제 연동 때) | `docs/multiview-progress.md` 후속 7, `docs/zones-store01.md` |
| 2026-11-10 | (한호석) 결제 라벨 도구로 38쌍 결제 시각·개수와 시차 32쌍을 적고 다시 판정 | 계산대 체류 31/38, 결제 라벨 30/38 — 결제가 정확해도 집은 개수를 틀리면 판정이 틀림. 틀린 8쌍 원인 정리 | `docs/multiview-progress.md` 후속 8 |
| 2026-11-10 | 채점 고치기 — `dual_eval.py` 가 정답표(`truth.csv`)를 읽고, 판정 없는 쌍을 따로 세고, 손님별 마지막 판정 기준 점수를 함께 냄 | 점수는 그대로 31/38 · 결제 라벨 30/38 (그 전 화면에는 30 · 29 로 찍힘). 판정이 나와서 맞은 것은 25 · 24, 마지막 판정 기준 32 · 31 (C12) | `docs/multiview-progress.md` 후속 9 |
| 2026-11-10 | 분석 도구 1단계 (`workbench/`) — 확인용 페이지를 하나로 모으기 시작. 갤러리 + 한 쌍 분석(사람 추적 / 두 화면 같은 사람 / 물건 집기 / 결제·판정) | 38쌍 보기 데이터 1분 남짓에 만듦, 갤러리 숫자는 채점과 같음(31/38 · 판정이 나와서 맞음 25 · 판정 없음 8 · 손님 수 38/38). 수치 조정·정답 적기·새 영상 추가는 다음 단계 | [`workbench/README.md`](workbench/README.md), `docs/multiview-progress.md` 후속 10 |
| 2026-11-10 | 분석 도구 2단계 — 수치 조정(판정 값 16개를 초안으로 바꿔 다시 판정, 정상 영상으로만 맞추고 절도·복합 채점은 잠금, 적용) | 정상 12편 초안 판정 약 4초, 38편 약 6초. 예: 손님 조건 1초 → 0초면 손님 수 38 → 30/38 (시험 뒤 되돌림). `--set risk.inside_seconds` 가 무시되던 것도 고침 | [`workbench/README.md`](workbench/README.md), `docs/multiview-progress.md` 후속 11 |
| 2026-11-10 | 분석 도구 3단계 — 정답 적기(집음·놓음, 경보가 나야 하는 사람, 두 화면 같은 사람, 결제)와 그 정답으로 채점 | 도구만 만듦 — **정답은 아직 한 편도 적지 않았다.** 적으면 '물건 집기'·'두 화면 같은 사람'에 맞음·틀림이 붙고, 경보가 다른 사람에게 난 것(A-14)이 드러난다 | [`workbench/README.md`](workbench/README.md), `docs/multiview-progress.md` 후속 12 |
| 2026-11-10 | 분석 도구 4단계 — 새 영상 추가(갤러리에서 두 각도 영상을 올리면 뒤에서 처리해 카드로), 시차를 화면에서 맞추기 | 기존 영상 한 쌍(N1 의 두 파일)을 N13 이라는 새 영상으로 올려 끝까지 돌려 봄: 올리기 → 등록(정답표·추가 기록에 한 줄씩, 있던 줄은 그대로) → 처리 약 3분 20초 → 시차 자동 추정 −100ms(손님 위치 상관 0.78) → 카드 '정상 통과'(원본 N1 과 같은 판정, 집음 1개), 갤러리 묶음 '추가한 영상 1/1' · '2차 촬영 31/38'. 시차를 −67ms 로 고쳐 저장 → manual 로 적히고 다시 판정됨. **시험 뒤 영상·기록·표의 줄을 모두 지웠다 — 지금 추가한 영상은 없다.** 시험 7개 추가, 전체 234 통과 | [`workbench/README.md`](workbench/README.md), `docs/multiview-progress.md` 후속 13 |
| 2026-11-10 | 두 화면 짝이 자주 끊기는 이유 다시 재기 (`scripts/link_diagnose.py`, 사용자가 분석 도구를 써 보고 지적) | 짝 53%, 끊김 114번. 못 맺은 조각의 92% 가 선반 카메라 위치식(상자 가운데 가로 1차, 오차 약 29cm) 탓. 골반 가로·세로 식은 오차 약 8cm, 짝 유지와 함께 76% — **실험만, 적용 안 함** | `docs/multiview-progress.md` 후속 13-1 |
| 2026-11-13 | 3차 촬영: 두 화면 짝을 바닥 위 거리로 맺기 (`fusion.position_mode: floor`, `dual_eval.py --shoot 3` 기본) + 바닥 통일 검증 (`scripts/floor_check.py`) | 39쌍에서 짝 94.5 → 99.6%, 끊김 23 → 2번. 두 명 장면에서 맞는 사람과 맺힌 조각 85 → 97%, 다른 사람과 맺힌 조각은 17개 그대로. 바닥 통일: 타일 줄눈 어긋남 중앙 0.5cm, 같은 사람 발목 거리 중앙 4cm · 90% 8cm (메모지 둘레 50cm 안에서만 확인). **판정 점수는 그대로** — 이 컴퓨터에서 만든 기록으로 영상만 26/39 (팀원 기록은 27/39, 기록 차이). 진행 기록 후속 27 |

**다음에 확인할 것** (2026-10-14 에 적은 것): 계산하고 나가는 손님을 통과시키는가(직접 촬영 필요), 선반 지도 2단계(상품 탐지 모델), 선반 확인 속도.
→ 2026-11-10: 계산하고 나가는 손님은 1차·2차 촬영 정상 영상으로 확인(정상 11/12). 상품 탐지 모델은 아직 안 함. **지금의 다음 단계는 `docs/results-summary.md` §0-5.**
