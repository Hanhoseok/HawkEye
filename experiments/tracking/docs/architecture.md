# 구현 설명 — tracking 프로그램은 어떻게 돌아가나

팀원 공유용. **구조 → 모델을 어떻게 쓰나 → 판단 로직 → 설정·실행 → 파일 지도** 순서.
결과는 `docs/results-summary.md`, 안 되는 것은 `docs/limitations.md`, 각 층의 설계 이유·실험은 층별 문서(아래 표)에 있다.
숫자는 2026-10-19 코드 기준 실제 설정값이다 (`config.yaml`, `src/core/config.py`).
카메라 2대·실시간·선반 센서 구조는 §6 (2026-11-03 1단계 구현, **2026-11-10 에 그 뒤 바뀐 기본 설정까지 반영**), 설계는 `docs/multiview-design.md`, 근거 논문은 `docs/multiview-papers.md`, 구현 중 생긴 일은 `docs/multiview-progress.md`.
§1~§4 는 카메라 1대 실행 기준이다. 카메라 2대에서 달라지는 규칙(발 점·손님 조건·신원)은 §6 에 따로 적었다.

---

## 1. 전체 흐름 — 한 장면이 6개 층을 지난다

실행 파일은 `run_tracking.py` 하나, 장면마다 순서는 `src/pipeline.py` 가 정한다. 영상 한 장면이 들어오면 위에서 아래로 차례로 지나가고, 이것을 장면마다 반복한다.
①~④(관찰)는 **카메라 작업자**(`src/camera_worker.py`)가 맡고, `pipeline.py` 는 ⑤⑥(판단)과 출력을 맡는다 (2026-11-03, 카메라 2대 준비 1-1).
지금은 카메라 한 대라 작업자도 하나다. 모든 장면·관측에는 카메라 이름(`camera_id`, 기본 `cam0`)이 붙는다.

```
영상 한 장면 (Frame: 번호, 재생 시각, 이미지)
 │
 ├─▶ ① 사람 찾기·잇기     YOLO(사람) → BoT-SORT          →  TrackObservation  (상자 + 추적 번호 t1, t2 …)
 │                                                                │
 │   ② 매장 단위 신원     IdentityRegistry                 →  IdentityObservation (손님 1, 손님 2 …)
 │                                                                │
 │        ┌───────────────────────────────────────────────────────┤
 │        ▼                                                       ▼
 │   ③ 손 뻗기 감지   InteractionDetector  → TakeCandidate   ④ 선반 지도   ShelfWatcher → ShelfVisit
 │      "손님 1, 3~7초 선반에 손을 댐"         ▲                "1단 2번째 병 사라짐, 그때 앞에 손님 1"
 │                                             │
 └──────────────── (물건 탐지는 영상에서 따로) ┘
                    │                                 │
                    └──────────────┬──────────────────┘
                                   ▼
     ⑤ 손님 장부  CustomerBook   집음/놓음/만지기만 판정 · 물건 수 · 결제
                                   ▼
     ⑥ 위험 판정  RiskEngine     출구 접근 WARNING → 나감 확정 → HIGH_RISK / REVIEW / CLEAR  →  RiskEvent
                                   ▼
     출력   기록 파일(JSONL) · 결과 영상 · 상황판(--show) · 관리자 앱 경보(--alert-server)
```

| 층 | 묻는 질문 | 코드 | 설계 문서 |
|---|---|---|---|
| ① | 화면 어디에 사람이 있고, 장면 사이 누가 누구인가 | `src/detectors/`, `src/trackers/` | `tracker-comparison.md`, `scale-limits.md` |
| ② | 끊기거나 번호가 바뀌어도 같은 손님인가 | `src/identity/registry.py` | `identity-registry.md` |
| ③ | 언제 누가 선반에 손을 댔나 | `src/interaction/detector.py`, `src/zones/` | `take-candidates.md`, `merl-evaluation.md` |
| ④ | 선반에서 무엇이 사라지거나 생겼나 | `src/shelf/monitor.py`, `watcher.py` | `shelf-map.md` |
| ⑤ | 그래서 손님이 무엇을 했고 몇 개를 가졌고 결제했나 | `src/risk/customers.py`, `payments.py` | `shelf-map.md` 1-B·1-C, `store-test.md` 3차 |
| ⑥ | 나가는 손님이 문제인가 | `src/risk/engine.py` | `risk-pipeline.md` |

**층 사이 약속**: 층과 층은 `src/core/types.py` 의 정해진 형식만 주고받는다(TrackObservation, IdentityObservation, TakeCandidate, RiskEvent 등).
그래서 한 층을 갈아 끼워도 나머지는 그대로다 — 실제로 추적기를 ByteTrack → BoT-SORT 로 바꿀 때 다른 층은 손대지 않았다.
⑤⑥(판단 층)은 영상·모델 라이브러리를 import 하지 않는다(`tests/test_contract.py` 가 검사) → 영상 없이 테스트하고, 저장된 기록으로 판정만 다시 돌릴 수 있다.

## 2. 모델을 어떻게 쓰나

**모델은 '보는 일'만 한다. 판단은 전부 3절의 규칙이 한다.** 모델은 모두 공개 사전학습 모델을 그대로 쓰고, 학습하지 않았다.

| 모델 | 언제 돌리나 | 넣는 것 | 나오는 것 | 주요 설정 | 쓰는 곳 |
|---|---|---|---|---|---|
| **YOLOv8n** (사람) | 매 장면 | 장면 전체 | 사람 상자 + 점수 | 해상도 640, 점수 0.25 이상, 겹침 제거 0.45, 사람만 | ① |
| **BoT-SORT** + **OSNet x0.25** (외형) | 매 장면 | 사람 상자 + 장면 | 상자마다 추적 번호 | 90장(30fps 기준 3초)까지 사라져도 번호 유지, 외형 거리 0.5 이하만 같은 사람 | ① |
| **OSNet** (신원 계층에서 한 번 더) | 새 추적 번호가 생길 때 + 손님마다 0.5초에 한 번 | 사람 부분 사진 | 외형 숫자 묶음 | 키 65px(모델 입력 기준) 미만은 안 씀, 손님당 사진 5장 보관 | ② |
| **YOLOv8n-pose** | `--detector yolo-pose` 일 때만 (사람 탐지 대신) | 장면 전체 | 사람 상자 + 관절 17개 | 사람 탐지와 같음 | ③ 손목-선반 거리 (천장 카메라용) |
| **YOLOv8n** (물건, 따로 한 벌) | 선반을 지정했을 때(`--shelf`), **필요한 장면만** | 장면 전체 | 물건 상자 + 종류 이름(bottle·cup …) | 해상도 1280, 점수 0.05 이상(약한 탐지까지), 사람·가구 등 제외 | ④ |
| DINOv2-small | 실험에서만 (`scripts/retailaction_hands.py`) | 손 주변 | 특징 숫자 | — | 프로그램에는 아직 없음 |

- **사람과 물건은 해상도를 나눠 찾는다.** 작은 물건을 잡으려고 1280 으로 사람까지 찾으면 가까운 사람이 조각 상자 여러 개로 잡혀 한 명이 손님 5명이 됐다.
- **물건 탐지는 필요할 때만 한다** (가장 무거운 단계): 사람이 선반을 가리는 동안은 건너뛰고(그 동안의 탐지는 어차피 버린다), 아무 일 없으면 3장에 1번만.
  누가 다녀간 뒤 비교할 때는 매 장면. 결과는 같고 속도는 4.8 → 10~11 FPS.
- **카메라 각도는 프로그램이 판단하지 않는다.** 설치할 때 사람이 고른다: 비스듬한 카메라 = 기본(사람 상자 + 머묾 규칙), 천장 카메라 = `--detector yolo-pose --signal reach`.

## 3. 판단 로직 — 층별 규칙과 기준값

예시: 손님 A 가 들어와 병 1개를 집고, 계산대에 서지 않고 나간다.

### ② 같은 손님인가 (`identity/registry.py`)

새 추적 번호가 생길 때마다:

```
키 65px 미만?                 → 보류(-1), 다음 장면에 다시
화면 가장자리(2%)에 걸림?      → 비교하지 않고 새 손님 (몸이 잘린 사진은 누구든 닮아 보임)
그 외 → 지금 화면에 없는 손님들과 외형 비교 (같은 장면에 이미 보이는 손님은 후보에서 뺌)
    유사도 ≥ 0.65                                        → 그 손님
    방금(3초 안) 근처(키 1배 거리 안)에서 사라진 손님이면 → 0.45 만 넘어도 그 손님
    아니면                                                → 새 손님
```

- **나중에 합치기**: 새 손님이 사진을 10 / 20 / 40장(0.5초 간격 → 약 5 / 10 / 20초) 모을 때마다 그 직전에 사라진 손님의 사진 평균과 다시 비교, **0.875 이상이면 합친다.**
  한 화면에 함께 보인 적 있는 두 손님은 합치지 않는다. 합친 기록(`identity_merges.jsonl`)은 ⑤가 받아 두 손님의 기록을 하나로 합친다.
- 30초 넘게 안 보인 손님은 비교 대상에서 뺀다.
- 기준값 근거: MERL train 의 실제 재등장 122건 (다른 사람 오병합을 5% 로 묶을 때).

### ③ 손을 뻗었나 (`interaction/detector.py`)

| | 비스듬한 카메라 (기본 `dwell`) | 천장 카메라 (`reach`) |
|---|---|---|
| 조건 | 몸 상자가 선반 구역과 **5% 이상 겹치고** + 속도가 **초당 키의 0.3배 이하**(멈춤) | 손목이 선반 앞 띠에서 키의 0.05배 이내 |
| 지속 | **3초 이상** (0.5초 끊김은 봐줌) | 0.8초 이상 |
| 결과 | `TakeCandidate`: "손님 1, 3.2~7.1초, 선반 구역 shelf" | 같음 |

- 속도는 0.4초 동안 발 위치(상자 아래 가운데)가 움직인 거리를 **키로 나눈 값**이다. 픽셀로 재면 멀리 있는 사람의 걸음이 느리게 보여 통로를 걷는 사람이 전부 '멈춤'이 됐다.
- 이 층은 **"손을 댔다"까지만** 안다. 집었는지·놓았는지는 ④와 합쳐서 ⑤가 정한다.

### ④ 선반에서 무엇이 바뀌었나 (`shelf/monitor.py`)

```
1. 선반 앞에 아무도 없음 → 0.6초치 탐지를 모아, 60% 이상 장면에 같은 자리에 있던 것만 '확정 물건' (점수 0.15 이상)
   → 높이로 단을 나누고 왼쪽부터 번호:  [1단 1번째 cup, 1단 2번째 bottle …]   = 선반 지도
2. 사람 상자가 선반 영역과 겹침 → 세지 않고 '누가 가렸나'만 기록 (손님 1)
3. 다시 0.6초 비면 → 새 지도를 만들어 예전 지도와 자리로 짝 맞추기
      비용 = 중심 거리 ÷ 물건 크기 (+ 종류 이름이 다르면 0.3),  0.8 넘으면 짝 아님
      짝 없는 예전 물건 = 사라짐 후보,  짝 없는 새 물건 = 생김 후보
4. 약한 탐지로 한 번 더 확인 — 그 자리에 원래 점수의 절반 이상인 같은 크기 상자가 장면의 30% 이상 보이면
      '점수만 내려간 것'이지 사라진 게 아니다 (생김도 마찬가지)
5. 변화가 0.6초 더 유지돼야 확정 (손이 잠깐 가린 깜빡임 거르기)
6. 가린 사람이 1명이면 그 사람에게, 2명 이상이면 '누구 것인지 모름', 아무도 안 보였으면 '모름'
```

- 아무것도 안 바뀐 방문도 '변화 없음' 방문으로 남긴다 — 그 시간의 손 뻗기가 실제로는 아무것도 안 가져갔다는 근거가 된다.
- 같은 종류가 여러 개여도 **자리(위치)** 로 구별한다. 종류 이름은 일반 YOLO 라 자주 틀려 보조로만 쓴다.

### ⑤ 무엇을 했고, 몇 개고, 결제했나 (`risk/customers.py`)

**행동 판정** — ④의 방문과 같은 시간·같은 선반의 ③ 손 뻗기를 맞춘다.

| 선반 결과 | 손 뻗기 | 행동 | 물건 수 |
|---|---|---|---|
| 자리가 비었음 | 상관없음 | **집음** | +비운 개수 |
| 자리가 다시 참 | 상관없음 | **놓음** | −찬 개수 |
| 비기도 하고 차기도 함 | 상관없음 | 자리 바뀜 | 순변화 |
| 그대로 | 있음 | **만지기만** | 0 |
| 그대로 | 없음 | 구경 | 0 |
| 선반 기록 없음 | 있음 | 확인 못 함 | +1 |

손님의 물건 수 = 행동들의 개수 변화 합. 근거(`선반+동작` / `선반만` / `동작만`)를 함께 남긴다(`actions.jsonl`).
선반 앞에 두 사람이 함께 있어 누구 것인지 못 정한 변화는 아무에게도 붙이지 않고, 그 사람들은 손 뻗기 수로 센다.

**결제** — 두 방식 중 하나 (`risk.payment_mode`).

| 방식 | 규칙 |
|---|---|
| `feed` (기본) | 결제 기록 CSV(시각·개수)를 받아, 결제 순간 계산대 구역에 서 있던 손님(3초 안에 있었던 손님 포함) 중 가장 오래 서 있던 사람에게 붙인다 |
| `dwell` (`--pay-by-dwell`, 결제 기록 없는 매장) | **계산대에 1.5초 이상 서 있으면 그때까지 가진 것을 모두 결제.** 그 뒤 몸이 선반 구역과 50% 이상 겹치면(상품 쪽으로 돌아감) **취소**, 다음 계산대 체류에서 다시 결제 |

계산대·출구·문 밖 구역은 **발 위치**(상자 아래 가운데)가 구역 안인지로, 선반 구역은 **몸 상자가 겹친 비율**로 판단한다.

### ⑥ 나가는 손님이 문제인가 (`risk/engine.py`)

```
출구 구역에 0.5초 이상 서 있음 + '이대로 나가면 HIGH_RISK' → WARNING (직원이 먼저 알게)
나감 확정 (다음 중 하나)
   (a) 매장 안에 있던 손님이 문 밖 구역에 0.5초 이상 보임
   (b) 출구 근처(떠난 지 2초 안)에서 사라져 3초 동안 안 나타남
   (c) 녹화 영상이 끝남 — 출구 쪽에서 막 사라졌거나, 출구 쪽 화면 끝에 걸쳐 있는 손님
   → 확정 뒤 매장 안에 다시 나타나면 판정을 되돌림
최종 판정
   물건 0                → CLEAR
   물건 ≥1, 결제 0       → HIGH_RISK
   물건 > 결제 (일부만)  → REVIEW  (개수가 틀릴 수 있어 사람이 확인)
   물건 ≤ 결제           → CLEAR
HIGH_RISK 직전 신원 확인
   매장 안 마지막 5초 동안 '최근 모습 vs 예전 모습' 유사도 최고값 < 0.78
   → 추적 번호가 다른 사람에게 옮겨 붙었을 수 있음 → REVIEW 로 낮춤
```

- 경보를 두 단계로 나눈 이유: 혼잡한 문 앞에서 '출구에 들어섰다' 한 번으로 확정했더니 오경보가 났다(UCF 047).
- 마지막 5초 중 **최고값**을 쓰는 이유: 밝은 문 쪽으로 가면 역광으로 같은 사람도 0.87 → 0.68 로 떨어졌다(UCF 039). 조명 변화는 잠깐, 번호 뒤바뀜은 계속된다.

예시 손님 A: ③ "3~7초 손 뻗음" + ④ "1단 2번째 병 사라짐 → 손님 1" → ⑤ 집음 1개, 결제 0 → 출구 앞 WARNING → 문 밖 → **HIGH_RISK** (유사도 0.9 라 그대로) → 경보.

### 기준값은 어떻게 정했나

| 값 | 근거 |
|---|---|
| 0.65 / 0.875 (같은 손님) | MERL train 재등장 122건 |
| 0.78 (뒤바뀜 의심) | MERL train: 같은 사람 오인 5.6%, 실제 뒤바뀜 82% 잡음 |
| 0.05·0.8초 (천장 손 뻗기) | MERL train 1~20 에서 고르고 test 27~41 로 채점 (F1 0.73) |
| 3초 (사라짐 확정) | UCF 031 에서 탐지가 3.7초 끊긴 실측 |
| 0.6초 (선반 관찰) | Pexels 10566665 에서 1초면 너무 늦음 |
| 약한 탐지 확인 | 매장 정상 영상 N11 (흰 상자 점수 0.16 → 0.12 깜빡임) |
| 65px (사람 최소 크기) | 크기를 줄여 가며 잰 탐지 한계 |

## 4. 설정과 실행

```bash
cd experiments/tracking
python run_tracking.py --source <영상> --zones <구역 파일> \
    --shelf "shelf1=x1,y1,x2,y2@선반구역"   # 선반 지도 켜기 (화면 비율 0~1)
    --pay-by-dwell                           # 결제 기록 없는 매장: 계산대 체류 = 결제
    --show                                   # 실시간 창 + 상황판 (--panel 이면 결과 영상에만)
```

| 옵션 | 뜻 |
|---|---|
| `--payments <csv>` | 결제 기록 (time_sec, items, person_id) |
| `--detector yolo-pose --signal reach` | 천장 카메라용 손 뻗기 |
| `--tracker bytetrack` | 빠른 추적기 (기본 BoT-SORT — 겹친 뒤 번호 뒤바뀜이 적다) |
| `--stride 2` | 2장에 1장만 처리 (1080p CPU 는 약 3 FPS) |
| `--alert-server http://…` | 관리자 앱 경보 서버로 보내기 (API 키는 환경변수 `HAWKEYE_API_KEY`) |
| `--shelf-log` | 물건 탐지 결과 저장 (평가에서 영상 없이 선반 판정 재실행) |
| `--truth 24-76` | 상황판에 정답 구간 표시 (검증용) |

- **구역 파일** (`zones/*.yaml`): 카메라마다 한 번 그린다. 화면 비율 좌표, 구역 종류 `SHELF / CHECKOUT / EXIT / OUTSIDE`.
  같은 매장을 다시 찍었는데 카메라가 조금 움직였으면 `scripts/transfer_zones.py` 로 배경을 맞춰 옮긴다.
- **설정값**은 모두 `config.yaml` (코드에 숫자를 박지 않는다).

**출력** (`outputs/<실행>/`)

| 파일 | 내용 |
|---|---|
| `observations.jsonl` | ① 추적 결과 |
| `identities.jsonl`, `identity_merges.jsonl` | ② 손님 번호, 나중에 합친 기록 |
| `take_candidates.jsonl` | ③ 손 뻗기 |
| `shelf_visits.jsonl` | ④ 선반 방문 (사라짐·생김·변화 없음, 누구) |
| `actions.jsonl` | ⑤ 행동 (집음·놓음·만지기만…, 근거) |
| `payments.jsonl`, `risk_events.jsonl` | ⑤ 결제·취소, ⑥ 판정 |
| `tracked.mp4`, `alerts/` | 결과 영상, HIGH_RISK 장면 사진 |

## 5. 파일 지도

```
run_tracking.py                진입점 (config.yaml + 명령행, 카메라 2대는 --dual-config)
config.yaml                    모든 설정값 (카메라 1대 기본)
configs/store01_dual.yaml      카메라 2대 설정 — config.yaml 위에 얹는다 (§6)
zones/                         카메라별 구역 파일 (store-aisle, merl, pexels_*, store01_* …), store01_dual/ 은 카메라 2대
src/
  core/types.py                층 사이 데이터 형식 (팀 공통 계약 — 바꾸면 팀에 공유)
  core/config.py               설정 로더 (기본값과 근거 주석)
  core/protocols.py            Detector / Tracker 인터페이스
  inputs/video_source.py       영상 읽기 (장면 번호·재생 시각, 깨진 시각 정보 처리, 카메라 이름·시차·실제 시각, rtsp 주소)
  inputs/sync.py               두 카메라 시차 표(sync.csv)와 자동 추정 (카메라 2대 준비)
  camera_worker.py             카메라 한 대 몫의 ①~④ (탐지·추적·신원·손 뻗기·선반)
  detectors/yolo_detector.py   ① 사람 탐지 + 선반 물건 탐지기(build_item_detector)
  detectors/yolo_pose_detector.py  ① 사람 + 관절 17개
  trackers/                    ① BoT-SORT / ByteTrack 어댑터, 교체 지점(factory.py)
  identity/registry.py         ② 매장 단위 신원 (즉시 판정 + 나중에 합치기 + 가장자리 규칙)
  identity/embedder.py         ② 외형 모델(OSNet) 어댑터
  zones/                       구역 읽기, 사람-구역 관계 (겹침 비율·발 위치), floor.py 바닥 지도(호모그래피, 카메라 2대 준비)
  interaction/detector.py      ③ 손 뻗기 (dwell / reach …)
  interaction/pose_features.py ③ 손목-선반 거리, 손 높이
  shelf/monitor.py             ④ 선반 지도 (확정·가림·비교·약한 탐지 확인)
  shelf/watcher.py             ④ 파이프라인 연결 (필요할 때만 물건 탐지)
  risk/customers.py            ⑤ 손님 장부 (행동 판정, 물건 수, 결제·취소, 신원 합침)
  risk/payments.py             ⑤ 결제 기록 입력 (CSV)
  risk/engine.py               ⑥ 위험 판정 (두 단계 경보, 나감 확정, 신원 확인, 영상 끝)
  sinks/                       기록 파일, 결과 영상, 경보 전송(alert_sink.py → app-pos)
  viz/                         화면 표시 (사람 상자, 선반 지도 shelf_overlay, 상황판 live_panel)
  pipeline.py                  전체 조립 (장면마다 작업자 ①~④ → ⑤⑥ → 출력)
  multicam.py                  카메라 2대 조립 (두 영상을 공통 시각 순서로, §6)
  inputs/live_source.py        실시간 입력 (최신 장면만, 끊기면 다시 연결)
  detectors/masked.py          사람 아님 영역과 겹치는 사람 상자 버리기
  shelf/slot_diff.py           칸 그림 비교 (칸 안 그림이 방문 전후로 바뀌었나)
  fusion/cross_view.py         ⑦ 두 화면 사람 잇기
  fusion/shelf_events.py       ⑦ 선반 사건 묶기 (주인 정하기, 센서 입력 자리)
  fusion/judge.py              ⑦ 합치기 + 판정 (실시간·재실행 공용)
scripts/                       평가·실험 도구 — 쓰임별 분류는 README §3 'scripts — 쓰임별로'
tests/                         층별 규칙 테스트 (2026-10-19 148개 → 2026-11-10 테스트 함수 185개)
labels/                        정답표 (store01 1차 촬영, store01_dual 2차 촬영: 정답·시차·결제 라벨·가짜 센서)
docs/                          설계·실험 문서
```

## 6. 카메라 2대 실행 (2026-11-03 1단계, 기본 설정은 2026-11-09)

전체 카메라(angle1, 높은 구석)와 선반 카메라(angle0, 선반 높이)를 함께 돌린다. 설계는 `docs/multiview-design.md`.

```
angle1 전체 카메라 → 작업자: 사람+관절(사람 아님 영역 제외) → 화면 신원 → 발목을 바닥 좌표로(FloorZoneMap) → 선반 앞 머묾 ─┐
angle0 선반 카메라 → 작업자: 사람+관절 → 화면 신원 → 손-칸 접촉(손목이 칸 안) / 선반 지도 + 칸 그림 비교 ───────────────┼→ ⑦ DualJudge
선반 센서(있으면) ───────────────────────────────────────────────────────────────────────────────────────────┘
⑦ 교차 화면 연결(위치 1순위 + 외형, 후보가 한 명이면 외형만으로) → 매장 신원 = 전체 카메라 신원
   선반 사건 묶기(가린 사람의 짝 → 손-칸 접촉 → 선반 앞 한 명) → 전체 카메라 손님의 집기·방문으로
→ ⑤ 손님 장부 → ⑥ 위험 판정 (구역은 모두 바닥 좌표)
```

### 카메라 1대(§3)와 달라지는 규칙 — `configs/store01_dual.yaml`

| 무엇 | 카메라 1대 기본 | 카메라 2대 기본 | 설정 | 왜 |
|---|---|---|---|---|
| 전체 카메라 사람 탐지 | YOLOv8n | **YOLOv8n-pose** (관절) | 카메라 `detector: yolo-pose` | 발목을 쓰려고 (11-09). 선반 카메라는 처음부터 관절 모델 |
| 계산대·출구 판단의 발 점 | 상자 아래 가운데 | **발목** (안 잡히면 상자 아래 가운데, 관측의 약 26%) | `zones.foot_point: ankle` | 상자 아래 가운데는 팔을 뻗거나 다리가 가려지면 실제 발에서 수십 px 벗어난다 (B-15) |
| 구역 | 화면 비율 좌표 | **바닥 cm 좌표** (바닥 지도로 옮김), 구역 편집 페이지에서 실제 자리에 맞춰 그림 | `zones/store01_dual/floor.yaml` | 발이 보이는 전체 카메라가 결제·출구를 맡는다. 구역마다 하는 일은 `zones-store01.md` |
| 손님으로 세는 조건 | 문 밖 구역이 아니면 매장 안 | 발이 **선반 앞·계산대·통로에 1초 이상** | `risk.inside_zones`, `inside_seconds: 1.0` | 문 앞 조각·창밖 행인·외투가 손님이 되던 것 (C-20) |
| 가장자리 규칙 (§3 ②) | 네 변 모두 | 화각 때문에 늘 잘리는 변은 뺌 — 전체 카메라는 위쪽, 선반 카메라는 아래쪽 | 카메라 `identity.edge_sides` | 전체 카메라는 머리가 늘 잘려 새 번호 거의 전부가 대조 없이 새 손님이 됐다 (C-21) |
| 되살아난 추적 번호 | 그 번호의 신원 그대로 | 1초 넘게 끊겼다 되살아나면 **신원 다시 판정** | `identity.track_gap_recheck_seconds: 1.0` | 추적기가 끊긴 번호를 다른 사람에게 되살림 (C-22) |
| 선반 카메라 손 뻗기 | 선반 앞 3초 머묾 | **손목이 칸 안에 0.3초** | `detail_interaction.signal: hand` | 짧은 집기를 잡으려고 (설계 M-6) |
| '변화 없음'으로 손 뻗기 지우기 | 물건 탐지가 그대로면 | 물건 탐지와 **칸 그림이 모두 그대로**일 때만 | `src/shelf/slot_diff.py` | 선반이 못 본 물건 때문에 진짜 집기를 지우던 것 (D-17) |
| 결제 | 기록 또는 계산대 체류 | 계산대 체류 1.5초 (평가에서는 결제 라벨도 가능) | `risk.payment_mode: dwell` | 결제 기록 없음 (E-13) |

재 보고 **기본에서 뺀 것** (설정은 남아 있다): 전체 카메라 팔 뻗기 `fusion.overview_reach_seconds`(복합 정상 거짓 경보), 임시 신원 `identity.tentative_seconds`(C14 나빠짐),
구역 잠깐 벗어남 허용 `risk.zone_grace_seconds`(계산대를 스친 도둑이 결제로 잡힘). 과정은 `multiview-progress.md` 후속 3·5·7.

두 화면 사람 잇기의 기준: 0.5초 조각마다 위치(선반 앞 띠 위 좌표)와 외형(OSNet 평균)으로 헝가리안 짝 맞추기. 선반 앞 후보가 한 명뿐이면 위치가 어긋나도 외형 0.85 이상이면 잇는다(`fusion.solo_appearance_min`).

### 코드와 설정

| 무엇 | 코드 | 설정 |
|---|---|---|
| 두 영상을 공통 시각 순서로 섞어 처리 | `src/multicam.py` (`MultiCameraPipeline`, `build_multicam`) | `configs/store01_dual.yaml` |
| 녹화본 시차 / 실시간 카메라 | `src/inputs/video_source.py` (offset_ms) / `src/inputs/live_source.py` (최신 장면만, 재연결) | 카메라 `source` 가 rtsp·http·숫자면 실시간 |
| 바닥 지도·바닥 구역 | `src/zones/floor.py` (`FloorMap`, `FloorZoneMap`) | `zones/store01_dual/floor.yaml` |
| 칸·선반 지도·띠 축 | `zones/store01_dual/angle0.yaml` | `band_map` 은 `dual_eval.py calibrate` 가 채움 |
| 교차 화면 연결 | `src/fusion/cross_view.py` | `fusion.*` (`FusionConfig`) |
| 선반 사건 묶기·센서 | `src/fusion/shelf_events.py` | `fusion.sensor_path`, `sensor_unit_g` |
| 합치기 + 판정 (실시간·재실행 공용) | `src/fusion/judge.py` (`DualJudge`) | |
| 칸 그림 비교 | `src/shelf/slot_diff.py` | |
| 사람 아님 영역 | `src/detectors/masked.py` | `floor.yaml` 의 `ignore_image` |
| 평가 | `scripts/dual_eval.py` (run / sync / calibrate / judge) — 기본 기록 폴더 `outputs/dual_eval_pose3`, `--out`·`--zones`·`--set 절.키=값` 으로 설정을 바꿔 비교, `judge --payments` 는 결제 라벨을 결제 기록으로 | `labels/store01_dual/` (정답 `truth.csv`·시차 `sync.csv`·결제 라벨 `payments.csv`·가짜 센서) |
| 결제 라벨 적기 | `scripts/label_payments.py` (두 화면을 보며 결제 시각·개수·시차 입력) | `labels/store01_dual/payments.csv` |
| 오판 장면 보기 · 구역 편집 | `scripts/review_serve.py` 로 여는 로컬 페이지 (`review_viewer.html`, `zone_editor.html`) — 기록을 원본 영상 위에 그림 / 구역 초안으로 38쌍 판정을 돌려 비교한 뒤 적용 | `outputs/review/`, `outputs/zone_editor/` (git 제외) |

```bash
python run_tracking.py --dual-config configs/store01_dual.yaml \
    --camera "angle1=../../source/1(angle1)/N01.mp4" --camera "angle0=../../source/1(angle0)/20261028_N1.mp4"
python run_tracking.py --dual-config configs/store01_dual.yaml --camera angle1=rtsp://... --camera angle0=rtsp://...
```

출력(`<out>/`): 카메라별 `angle1/`·`angle0/`(identities, take_candidates, 선반 방문, 칸 변화) + 가운데 `cross_view_links`, `shelf_events`(주인을 어떻게 정했나), `actions`, `payments`, `risk_events`, `appearance`.
