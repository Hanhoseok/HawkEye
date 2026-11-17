# 관찰 결과 저장·재계산과 회귀 검사

작성 2026-11-14. 코드: `src/replay/cache.py`, `scripts/dual_eval.py` (`run --record-cache` · `replay`), `scripts/regress.py`.

## 왜

촬영 한 회(짝 38~39편)를 처음부터 돌리면 CPU 로 약 1시간. 신원 규칙·손 뻗기 기준 하나를 바꿔 볼 때마다 1시간을 쓸 수 없다.
화면(화소)을 실제로 쓰는 계산은 다섯 군데뿐이다 — 사람 탐지, 추적, 외형 벡터, 선반 물건 탐지, 칸 그림 비교.
이 다섯의 결과를 장면 번호로 저장해 두면 나머지(신원·손 뻗기·선반 지도·합치기·판정)는 영상도 모델도 없이 다시 계산된다.

| | 3차 39짝 | 2차 38짝 |
|---|---|---|
| 직접 실행 (영상 + 모델) | 약 1시간 | 약 1시간 |
| 저장분으로 다시 계산 | **12초** | **11초** |
| 저장분 크기 | 68MB | 약 50MB |

검증 (2026-11-14): 저장분으로 다시 계산한 기록(신원·손 뻗기·선반·합치기·위험 사건 등 jsonl 전부)이 직접 실행 기록과
2·3차 모든 짝에서 줄 단위로 같다 (시각 문자열 제외, 문자열 해시 시드 3가지로 반복).

## 쓰는 법

```bash
# 1) 저장하며 직접 실행 (촬영당 한 번, 약 1시간)
python scripts/dual_eval.py --shoot 3 run --record-cache        # → outputs/dual_eval_shoot3/<짝>/cache/*.pkl.gz
python scripts/dual_eval.py --shoot 2 run --record-cache        # → outputs/dual_eval_pose3/...

# 2) 설정을 바꿔 다시 계산 (초 단위) 후 판정
python scripts/dual_eval.py --shoot 3 replay --from dual_eval_shoot3 --out try1 --set identity.late_merge_threshold=0.875
python scripts/dual_eval.py --shoot 3 judge --out try1

# 3) 회귀 검사 — 2·3차 × 모든 판정 방식을 한 번에 (14초)
python scripts/regress.py check
python scripts/regress.py check --set risk.exit_hold_seconds=0                       # 판정 단계 설정
python scripts/regress.py check --replay --set identity.late_merge_threshold=0.875   # 관찰 층 설정
python scripts/regress.py update                                                      # 지금 결과를 기준으로
```

## 무엇을 다시 계산할 수 있나

| 바꾸는 것 | 방법 |
|---|---|
| 위험 판정·결합 (`risk.*`, `fusion.*`) | `judge` 만 다시 (`regress.py check --set …`) |
| 신원·합치기·손 뻗기·바닥 구역 (`identity.*`, `interaction.*`, `detail_interaction.*`, `zones.*`, 바닥 구역 좌표) | 저장분으로 다시 계산 (`replay`, `regress.py check --replay --set …`) |
| 탐지기·추적기·처리 간격·선반 물건 탐지·칸 좌표·사람 아님 영역·영상 파일 | 저장분 무효 — 직접 다시 실행 |

마지막 줄은 저장분의 유효 조건(fingerprint)에 들어 있다. 조건이 다르면 조용히 틀린 결과를 내지 않고 `FingerprintMismatch` 로 멈춘다.
저장분에 없는 장면을 찾아도 빈 값으로 넘어가지 않고 `CacheMiss` 로 멈춘다 (그 짝은 실패로 집계).

외형 벡터는 '요청분'(신원 층이 실제 요청한 것)과 '예비'(모든 추적 상자)를 함께 저장한다. 신원 설정을 바꿔 요청이 달라지면 예비를 쓰고,
쓴 횟수를 실행 메시지에 적는다 (한 번에 뽑는 묶음이 달라 직접 실행과 1e-6 수준 차이가 날 수 있음).

## 회귀 검사

- 기준: `baselines/shoot<N>_<방식>.json` — 짝별 판정·맞음·손님 수 (작은 파일, 커밋함).
  3차 v(영상만)·s(센서 L1·L2)·p(결제 라벨)·ps(센서+결제), 2차 v·p.
- 지금 기준 (2026-11-14): 3차 28·29·29·29 / 39, 2차 32·32 / 38, 손님 수 38/39·38/38.
  → 같은 날 후속 46(포개진 상자 보류 + 관절로 가르기, 3차 기본)로 3차 기준을 갱신: 점수 그대로, **손님 수 39/39**(C03), 판정 목록만 바뀐 짝 C03 · C14. 2차 기준은 그대로.
- 종료 코드: 0 = 같거나 좋아짐, 1 = 나빠진 짝 있음 (맞던 짝이 틀림·손님 수가 틀려짐), 2 = 검사 불완전 (빠진 짝·실패).
- `update` 는 결과가 불완전하면(실패·건너뜀·빈 결과·기준에 있던 짝이 빠짐) 기준을 쓰지 않는다. 일부러 짝을 뺄 때만 `--allow-remove`.
- 반영되지 않을 `--set` 은 거절한다 — 관찰 층 설정인데 `--replay` 가 없거나, 탐지기·추적기 설정이거나.

동작 확인:
- 기본 `check` → 6개 방식 모두 기준과 같음, 종료 0.
- `--set risk.exit_hold_seconds=0` → 3차 C12·C15, 2차 C12 나빠짐, 종료 1 (후속 33 과 같은 짝).
- `--replay --set identity.late_merge_overlap_seconds=0 --set identity.late_merge_threshold=0.875` → 3차 C14 손님 수 틀려짐, 종료 1 (후속 31 과 같은 짝).
- 설정을 바꿔 다시 계산한 뒤 같은 폴더에 기본 설정으로 다시 계산 → 기준과 같음 (이전 실행의 합치기 기록이 남아 섞이지 않음).

## 신원 뒤바뀜 확인 (`--identity-check`)

실제 파이프라인은 판정에 신원 일관성 값을 넣지만 `judge` 는 지금까지 넣지 않았다. 실행 때 `angle1/consistency.jsonl` 로 남기고
`judge --identity-check` / `regress.py check --identity-check` 로 켤 수 있다. 2·3차에서는 켜도 점수가 같다. 기본은 끔 (점수 체계 유지).

## 같이 고친 것

선반 물건 이름 다수결(`src/shelf/monitor.py`)이 동률일 때 `set` 순서에 기대, 실행마다(문자열 해시) 이름이 달라질 수 있었다 —
3차 N05 의 선반 기록이 실행마다 조금 달랐고 재계산이 가끔 `CacheMiss` 로 멈췄다. '먼저 나온 이름'으로 고정.

## 한계

- 저장분은 pickle — 자기가 만든 `outputs/` 아래 파일만 읽는다. 커밋하지 않는다.
- 탐지·추적을 바꾸는 시험(예: ByteTrack, 해상도)은 여전히 직접 실행 1시간.
- 실시간 흉내(`live_sim.py`)는 저장하지 않는다 — 어느 장면을 처리하느냐가 처리 속도에 달려 매번 다르다.
