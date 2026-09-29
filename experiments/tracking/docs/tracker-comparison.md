# ByteTrack vs BoT-SORT(ReID) 비교 실험

> 목적: Phase 3 에서 확인된 **"가림 후 재등장 시 ID switch"** 가 외형(ReID) 기반 tracker 로 해결되는지 검증.
> 전제: **카메라 1대**. 따라서 다중 카메라 ReID 가 아니라 같은 화면 안에서의 단기 재식별만 다룬다.
> 관련: `docs/phase3-failure-notes.md`

## 조건 (양쪽 완전 동일)

| 항목 | 값 |
|---|---|
| 영상 / 구간 | `store-aisle-detection.mp4`, 원본 frame 1200~1598 |
| 샘플링 | `--stride 2 --tracker-fps 30` (200 처리프레임) |
| detector | yolov8n.pt, imgsz 640, conf 0.25, CPU |
| 검증 대상 | frame 1458 에 사라졌다가 1558 에 재등장한 인물 (공백 50 처리프레임 ≈ 1.67초) |

정답은 **track 4개** (원래 3명 + 도중에 새로 들어온 1명).

## 결과

| tracker | lost_buffer | proximity | appearance | 고유 track | 속도 | 판정 |
|---|---|---|---|---|---|---|
| ByteTrack | 30 | — | — | 5 | 6.6 FPS | ❌ ID switch |
| ByteTrack | 90 | — | — | 5 | — | ❌ |
| ByteTrack | 150 | — | — | 5 | — | ❌ |
| BoT-SORT+ReID | 30 | 0.5 | 0.25 | 5 | 4.4 FPS | ❌ |
| BoT-SORT+ReID | 30 | 1.0 | 0.25 | 5 | — | ❌ |
| BoT-SORT+ReID | 30 | 1.0 | 0.5 | 5 | — | ❌ |
| BoT-SORT+ReID | 90 | 0.5 | 0.25 | 5 | — | ❌ |
| BoT-SORT+ReID | 90 | 1.0 | 0.25 | 5 | — | ❌ |
| **BoT-SORT+ReID** | **90** | **1.0** | **0.5** | **4** | **4.5 FPS** | ✅ **재식별 성공** |

## 대조 실험 — 고친 것은 '느슨한 설정'이 아니라 '외형 정보'다

성공 조합과 **완전히 같은 설정**에서 ReID 만 껐을 때:

| 설정 | ReID | 고유 track |
|---|---|---|
| buffer 90 / prox 1.0 / app 0.5 | **OFF** | 5 ❌ |
| buffer 90 / prox 1.0 / app 0.5 | **ON** | 4 ✅ |

→ 설정을 아무리 관대하게 풀어도 외형 정보가 없으면 재식별은 불가능하다.
**"ByteTrack 은 움직임만으로는 복구할 수 없다"는 진단은 설정 문제가 아니라 알고리즘의 한계가 맞다.**

재현: `--tracker botsort --no-reid --lost-buffer 90 --reid-proximity 1.0 --reid-appearance 0.5`

## 핵심 발견: 세 개의 관문을 모두 통과해야 한다

재식별 실패는 원인이 하나가 아니라 **직렬로 놓인 관문 세 개** 때문이었다.
하나라도 막히면 나머지를 아무리 풀어도 실패한다. 실제로 8번의 실패가 모두 이 구조 때문이었다.

```
[1] track 이 아직 살아 있는가?        lost_track_buffer >= 공백 길이
       ↓ 통과해야
[2] 외형을 비교할 자격이 있는가?      IoU 거리 <= proximity_threshold
       ↓ 통과해야
[3] 외형이 충분히 닮았는가?           외형 거리 <= appearance_threshold
```

- **관문 1 — 생존**: 공백 50프레임 > 기본 버퍼 30 → track 이 삭제된다. 삭제되면 비교 대상 자체가 없어
  ReID 를 켜도 의미가 없다. (BoT-SORT 를 기본값으로 처음 돌렸을 때 ByteTrack 과 결과가 **완전히 동일**했던 이유)
- **관문 2 — IoU 게이트**: boxmot `botsort.py:289-296` 은
  `emb_dists[ious_dists > proximity_thresh] = 1.0` 으로 **겹치지 않는 쌍의 외형 정보를 버린다.**
  기본값 0.5 는 "IoU 0.5 이상 겹칠 때만 외형을 본다"는 뜻이라, 오래 사라졌던 대상에는 애초에 적용되지 않는다.
  → **이것이 가장 함정이다.** ReID 를 켰다고 해서 가림 복구가 되는 것이 아니다.
- **관문 3 — 외형 임계값**: 0.25 는 너무 엄격했다. 같은 사람이라도 자세·조명·크기가 달라지면 임베딩 거리가 벌어진다.

## 병합이 올바른지 확인

| | 수정 전 | 수정 후 |
|---|---|---|
| track 3 | 1200~1458 (129 frames) | **1200~1598 (151 frames)** |
| track 5 | 1558~1598 (21 frames) | 사라짐 (track 3 으로 흡수) |
| track 4 | 1558~1598 (21 frames) | **1558~1598 (21 frames) 그대로 유지** |

frame 1590 의 결과 영상에서 모자 쓴 남성이 다시 **ID 3** 으로 표시된다 (수정 전에는 ID 5).
동시에 좌하단에서 새로 진입한 다른 남성은 **ID 4 로 분리 유지**되었다.
즉 "합쳐야 할 것은 합치고, 합치면 안 되는 것은 합치지 않았다."

## 비용과 위험

| 항목 | 내용 |
|---|---|
| 속도 | 약 **30% 느려짐**. ReID 임베딩 추출 비용 |

속도는 동일 조건(같은 구간 100 프레임, `--no-video`, 2회 반복)에서 따로 측정했다.
표 안의 FPS 값들은 영상 저장 여부가 섞여 있어 tracker 간 비교에 쓰면 안 된다.

| tracker | 1회 | 2회 |
|---|---|---|
| bytetrack | 8.47 FPS | 8.18 FPS |
| botsort | 5.76 FPS | 6.02 FPS |
| 의존성 | `boxmot==19.0.0` 추가. 25.x 는 numpy>=2.2 를 강제해 기존 핀과 충돌하므로 19.0.0 고정 |
| 모델 | `osnet_x0_25_msmt17.pt` (경량 ReID 백본) 자동 다운로드 |

### ⚠ 아직 검증되지 않은 위험 — 오병합(false merge)

이번 튜닝은 **단 한 건의 재식별 사건**에 맞춘 것이다. `proximity_threshold: 1.0` 은 IoU 게이트를
완전히 열고 `appearance_threshold: 0.5` 는 외형 판정을 느슨하게 한 값이라,
**비슷한 옷차림의 서로 다른 고객을 같은 사람으로 합칠 위험**이 커졌다.

절도 탐지 맥락에서 오병합은 ID switch 만큼, 혹은 그 이상으로 위험하다.

- ID switch → 미결제 상태가 **사라져서** HIGH_RISK 를 **놓친다** (미탐)
- 오병합 → A 고객의 미결제 상태가 B 고객에게 **옮겨붙어** HIGH_RISK 를 **잘못 띄운다** (오탐)

**따라서 이 설정을 기본값으로 승격하지 않았다.** `config.yaml` 의 기본값은 boxmot 표준값(0.5 / 0.25)으로
두고, 실측 조합을 주석으로만 기록해 두었다.

## 두 번째 영상: 교차 실험 (people-detection.mp4)

상세는 `docs/phase3-failure-notes.md` 실험 #2. 요약하면:

| tracker | 재등장 후 재식별 | 교차 시 ID 유지 | 오병합 |
|---|---|---|---|
| ByteTrack | ❌ | ❌ | 없음 |
| BoT-SORT 기본 (0.5 / 0.25) | ❌ | ✅ | 없음 |
| **BoT-SORT 개방 (1.0 / 0.5)** | ✅ | ✅ | 없음 (이 영상 기준) |

- BoT-SORT 는 **기본 게이트만으로도 교차를 해결한다.** 교차 중에는 bbox 가 겹치므로
  IoU 게이트를 통과하기 때문이다. 즉 기본값은 원래 교차를 겨냥한 설계다.
- 게이트를 연 설정은 교차도 유지하면서 재등장까지 해결했고, 오병합도 일으키지 않았다.
- 단, 이 영상은 동시 등장 1~3명 / 옷차림이 뚜렷이 달라 **오병합 위험의 검증으로는 약하다.**

## 다음에 해야 할 검증

1. **혼잡 + 비슷한 옷차림** 영상으로 오병합 검증 (가장 중요. 현재 증거로는 부족)
2. `appearance_threshold` 0.3 / 0.4 / 0.5 스윕 → 재식별 성공과 오병합이 갈리는 경계 찾기
3. 더 긴 구간(store-aisle 전체 3921 frames)으로 track 수가 실제 인원과 맞는지 확인
4. 속도가 문제되면 OpenVINO 로 detector 를 가속해 ReID 비용을 상쇄

## 재현 명령

```bash
# 실패 재현 (기본값)
python run_tracking.py --source data/videos/store-aisle-detection.mp4 --tracker botsort \
  --start-frame 1200 --stride 2 --tracker-fps 30 --max-frames 200 --out-dir outputs/botsort

# 성공 재현
python run_tracking.py --source data/videos/store-aisle-detection.mp4 --tracker botsort \
  --start-frame 1200 --stride 2 --tracker-fps 30 --max-frames 200 \
  --lost-buffer 90 --reid-proximity 1.0 --reid-appearance 0.5 --out-dir outputs/botsort-fixed

python scripts/analyze_observations.py outputs/botsort-fixed/observations.jsonl --min-frames 10
```
