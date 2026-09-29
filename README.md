# HawkEye — AI 기반 무인매장 절도 위험 탐지 및 대응 시스템

2026 2학기 캡스톤 · 팀 호크아이 (왕규원 · 정준우 · 한호석)

CCTV 영상으로 고객이 상품을 집고 돌려놓는 과정을 기록하고, 계산대 결제 기록과 비교하여
**결제하지 않은 상품을 가진 채 출구에 접근하는 상황**을 관리자에게 알려주는 시스템이다.
사람을 절도범으로 단정하지 않고, 관리자가 확인할 기회를 만드는 것이 목적이다.

## 저장소 구조

지금은 팀원마다 다른 방식으로 개발·시험하고 있어서, **각자 폴더에서 실험**하고
비교 결과 적중률이 좋은 방법만 **`final/`로 모아 최종 시스템**을 만든다.

| 폴더 | 담당 | 내용 |
|---|---|---|
| [`experiments/tracking/`](experiments/tracking/) | 왕규원 | YOLO 탐지 + ByteTrack / BoT-SORT + ReID 추적, 입장~출구 판정 파이프라인 |
| [`experiments/behavior-model/`](experiments/behavior-model/) | 정준우 | AI-Hub 무인매장 정상·이상 행동 데이터로 모델 학습·테스트 |
| [`experiments/app-pos/`](experiments/app-pos/) | 한호석 | 모바일 관리자 앱, 모의 결제 시스템, 테스트 기록 |
| [`final/`](final/) | 공동 | 실험에서 선택된 방법을 통합한 최종 시스템 |
| [`docs/`](docs/) | 공동 | 기능 명세서, 데이터 명세서 |

실행 방법은 각 폴더의 README를 본다.

## 작업 규칙

1. **자기 폴더에서만 작업한다.** 다른 사람 폴더를 고쳐야 하면 담당자에게 먼저 말한다.
2. **main에 직접 push하지 않는다.** 브랜치를 만들고 PR로 합친다.
   - 브랜치 이름: `feature/<폴더>-<내용>` (예: `feature/tracking-risk-engine`, `feature/app-pos-dashboard`)
   - `final/`에 들어가는 PR은 다른 팀원 1명 이상이 확인한 뒤 머지한다.
3. **데이터·영상·가중치·비밀 정보는 커밋하지 않는다.** 이 저장소는 Public이다.
   - AI-Hub, MERL 데이터는 약관상 재배포가 제한된다.
   - 직접 촬영한 영상에는 사람 얼굴이 나온다.
   - 가중치(`.pt`)와 영상은 팀 드라이브로 공유한다.
   - RTSP 주소·카메라 계정·API 키는 `.env`에 둔다.
   - 커밋 전에 `git status`로 올라가는 파일을 확인한다.
4. **실험 결과는 각 폴더 README의 "실험 기록"에 남긴다.** 무엇을 바꿨고 결과(정확도·속도 등)가 어땠는지 적어야 나중에 방법을 비교해서 고를 수 있다.
5. **모듈끼리 주고받는 데이터 형식**은 `experiments/tracking/src/core/types.py`를 기준으로 맞춘다. 형식을 바꿔야 하면 팀에 먼저 공유한다.

## Windows 설치 주의

torch 설치 파일 중에 경로가 매우 긴 것이 있어, Windows 기본 설정(경로 260자 제한)에서는
저장소를 깊은 폴더에 클론하면 `pip install`이 `No such file or directory` 오류로 실패한다.
`experiments/tracking/.venv` 기준으로 **저장소 경로가 88자 이하**여야 한다.

- 짧은 경로에 클론한다 (예: `C:\dev\HawkEye`), 또는
- 관리자 PowerShell에서 긴 경로 지원을 켠다:
  `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`
