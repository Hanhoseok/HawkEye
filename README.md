# HawkEye — 무인매장 절도 위험 탐지

2026 2학기 캡스톤

> CCTV 영상에서 고객을 **매장 입장부터 퇴장까지** 추적하고, 선반에서 상품을 집는 행동을 탐지해
> **미결제 상태로 출구에 접근하는 상황**을 위험으로 표시하는 시스템.
> 사람을 "절도범"으로 확정하는 것이 아니라 위험 상황을 표시하는 것이 목적이다.

## 현재 상태

| 계층 | 상태 | 문서 |
|---|---|---|
| 영상 입력 | ✅ | |
| 사람 탐지 (YOLO / YOLO-pose) | ✅ | `docs/scale-limits.md` |
| 추적 (ByteTrack / BoT-SORT + ReID) | ✅ | `docs/tracker-comparison.md` |
| 매장 단위 신원 (계층 2) | ✅ tracker 버퍼의 3.3배(10초) 공백 복원 · 천장 시점 분열을 "나중에 합치기"로 해결 | `docs/identity-registry.md` |
| 구역 (SHELF / EXIT) | ✅ | `docs/zones.md` |
| TAKE 후보 (계층 3) | 🔶 MERL test 정밀도 0.75 / 재현율 0.71 / F1 0.73 / 국소화 37% | `docs/merl-evaluation.md` |
| 손님 상태 · 결제 연결 · 출구 판정 | 🔶 로직 완성, 실제 영상 검증 전 (출구·계산대 있는 영상 필요) | `docs/risk-pipeline.md` |
| RETURN (되돌려놓기) | ⬜ 미착수 | |

**처음 보는 사람은 `docs/진행경과-정리.md` 부터 읽는다.** 왜 이런 구조가 됐는지가 수치와 함께 정리되어 있다.

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

## 2. 계층 구조 — 왜 이렇게 나눴는가

```
[의존 계층: 외부 라이브러리를 아는 곳]        [로직 계층: 라이브러리를 몰라도 되는 곳]

 inputs/video_source.py   (cv2)
 detectors/yolo_detector.py (ultralytics)  ──┐
 trackers/bytetrack_tracker.py (supervision) ─┴─→  TrackObservation  ──→  (향후) TAKE/RETURN
                                                                          상태 관리 / POS / Risk
```

핵심 규칙 세 가지 (05 아키텍처 §3~4):

1. **tracker 뒤의 로직은 ByteTrack 내부 객체를 절대 직접 쓰지 않는다.** 경계는 `src/core/types.py` 의 `TrackObservation` 하나뿐이다.
2. **`src/core/` 는 cv2·torch·ultralytics·supervision 을 import 하지 않는다.** 이 규칙은 `tests/test_contract.py` 가 자동으로 검사한다.
3. **임계값과 모델 경로는 `config.yaml` 에만 있다.** 코드에 하드코딩하지 않는다.

그래서 ByteTrack → BoT-SORT 로 바꿔도 `build_tracker()` 한 곳만 고치면 되고, 뒤쪽 로직은 원칙적으로 그대로다.

### 표준 출력 계약

```jsonc
{
  "track_id": 3,
  "bbox": [412.0, 188.5, 498.2, 402.1],   // [x1, y1, x2, y2]
  "frame": 128,
  "timestamp": "2026-09-20T14:02:11.400000+09:00",
  "score": 0.91,          // 선택적 확장 필드
  "class_name": "person"  // 선택적 확장 필드
}
```

문서에 고정된 필드는 앞의 네 개다. 뒤 두 개는 디버깅용 확장이며, 없어도 뒤쪽 로직이 동작해야 한다.

---

## 3. 디렉터리

```
run_tracking.py            진입점 (config.yaml + 명령행 인자)
config.yaml                임계값 / 모델 경로 / 출력 설정
zones.yaml                 구역 정의 (store-aisle, 비스듬한 각도)
zones_merl.yaml            구역 정의 (MERL, 수직 천장)
src/
  core/types.py            Frame, Detection, TrackObservation  ← 계약
  core/protocols.py        Detector / Tracker 인터페이스
  core/config.py           config.yaml 로더
  inputs/video_source.py   Phase 0 — 프레임 + frame index + ISO8601 timestamp
  detectors/yolo_detector.py   Phase 1 — pretrained YOLO person detection
  trackers/factory.py      tracker 교체 지점 (여기 한 곳만 고치면 됨)
  trackers/bytetrack_tracker.py Phase 2 — ByteTrack 어댑터 (움직임만)
  trackers/botsort_tracker.py   BoT-SORT + ReID 어댑터 (외형 포함)
  identity/registry.py     계층 2 — 매장 단위 신원 (numpy 만 사용, tracker 를 모름)
  identity/embedder.py     ReID 백본 어댑터
  zones/geometry.py        다각형 기하 (외부 라이브러리 없음)
  zones/zone_map.py        구역 로드 + 사람과의 관계 판정
  interaction/detector.py  계층 3 — TAKE 후보 (signal: dwell / hand / raise)
  interaction/pose_features.py  손 높이·팔 뻗음 등 자세 특징
  risk/customers.py        손님 장부 — 집기 행동 · 결제 · 신원 합침 반영
  risk/payments.py         결제(POS) 입력 — 지금은 CSV
  risk/engine.py           출구 판정 — HIGH_RISK / REVIEW / CLEAR
  detectors/yolo_pose_detector.py  bbox + 17 keypoint 를 한 번에
  sinks/video_writer.py    결과 영상 저장
  sinks/observation_log.py TrackObservation JSONL 저장/읽기
  viz/overlay.py           bbox + track_id + 궤적 오버레이
  pipeline.py              전체 조립
scripts/make_sample_video.py  Phase 0 검증용 합성 영상 (사람 없음)
tests/test_contract.py     계층 분리 규칙 자동 검사
data/videos/               입력 영상 (git 제외)
outputs/                   결과물 (git 제외)
```

---

## 4. 문서

| 문서 | 내용 |
|---|---|
| **`docs/진행경과-정리.md`** | **팀원 공유용 전체 정리.** 왜 ByteTrack에서 BoT-SORT로 갔는지, 수치 포함 |
| `docs/phase3-failure-notes.md` | tracker 실패 실험 원본 기록 |
| `docs/tracker-comparison.md` | ByteTrack vs BoT-SORT 9회 실험 비교 |
| `docs/phase5-take-return-survey.md` | TAKE/RETURN 후보 사전조사 |
| **`docs/scale-limits.md`** | **사람이 몇 px 이하면 무너지는가 → 카메라 설치 기준, 화각별 검증** |
| **`docs/identity-registry.md`** | **계층 2 — 매장 단위 신원. tracker 가 끊겨도 person_id 유지** |
| **`docs/zones.md`** | **구역 인프라 — SHELF/EXIT 정의와 판정 방식** |
| **`docs/take-candidates.md`** | **계층 3 — 선반 앞 멈춤으로 TAKE 후보 판정, 그 한계** |
| **`docs/labeling.md`** | **행동 라벨링 — 무엇을 왜 어떻게 찍는가** |
| **`docs/pose-signals.md`** | **Pose 신호 — 손 높이는 되고 팔 뻗음·손 위치는 안 된다** |
| **`docs/merl-evaluation.md`** | **MERL 공개 데이터로 train/test 분리 평가. 특징이 화각에 종속적임을 확인** |
| **`docs/risk-pipeline.md`** | **손님 상태 · 결제 연결 · 출구 판정 — 입장부터 경보까지** |
| **`docs/filming-guide.md`** | **검증용 시나리오 영상 촬영 가이드 — 무엇을 찍고 무엇을 기록하나** |

## 5. Phase 진행 체크리스트

문서 §6 의 "권장 구현 순서와 확인 항목"을 그대로 옮긴 것이다.

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
