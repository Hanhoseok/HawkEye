# 상품 탐지 학습 폴더 (3차 촬영, 상품 3종)

**이 폴더만 보면 다른 컴퓨터에서 학습을 돌릴 수 있게** 만든 것이다. 저장소의 다른 코드는 필요 없다.
무인매장 영상에서 상품 3종(칸타타 · 동과루 · 애플사이다)을 찾는 YOLO 모델을 학습하고, 결과를 원래 컴퓨터로 가져간다.

> **AI 에이전트에게**: 이 문서를 처음부터 끝까지 읽고 '할 일'을 차례대로 한다. '하면 안 되는 것'을 먼저 읽는다.
> 막히면 추측으로 넘어가지 말고 사람에게 묻는다. 끝나면 '끝난 뒤 알릴 것'을 사람에게 그대로 알린다.

## 한눈에

| | |
|---|---|
| 여기 있는 것 | 라벨 1,194개 · 나눔별 목록 · `data.yaml` · 그림 표(`dataset/frames.csv`) · 스크립트 3개 |
| **여기 없는 것** | **그림 1,194장 (약 530MB)** — 얼굴이 나와서 저장소에 올리지 않는다. 사람이 따로 가져와야 한다 |
| 만들어지는 것 | `handoff/<이름>/` — 학습된 모델 `best.pt` 와 점수 `result.json`. **이 폴더를 원래 컴퓨터로 가져간다** |

## 하면 안 되는 것

1. **그림(`dataset/images/`)을 저장소 · 공개 링크 · 외부 서비스에 올리지 않는다.** 얼굴이 나온다. 이 저장소는 공개다.
2. **모델 파일(`*.pt`)과 `runs/` · `handoff/` 를 커밋하지 않는다.** `.gitignore` 로 막혀 있다 — 억지로 넣지 않는다.
3. **나눔을 바꾸지 않는다** (`dataset/*.txt` · `frames.csv` · `labels/`). 영상 단위로 나눠 둔 것이다 — 섞으면 점수가 실제보다 높게 나온다.
4. **시험(test) 점수를 보고 설정을 고르지 않는다.** 설정은 검증(val) 점수로 고르고, 시험 점수는 마지막에 적기만 한다.
5. `main` 에 직접 push 하지 않는다. 이 폴더에서 저장소에 올릴 것은 보통 없다 (결과는 `handoff/` 로 가져간다).

## 할 일

명령은 전부 이 폴더(`experiments/tracking/training/products/`)에서 친다.

### 1. 설치

```
python -m venv .venv                       # 한 번만. Windows: .venv\Scripts\activate · Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

- GPU 를 쓰려면 `requirements.txt` **보다 먼저** 그 컴퓨터에 맞는 PyTorch(CUDA)를 깐다 (https://pytorch.org/get-started/locally/).
- 확인: `python -c "import torch; print(torch.cuda.is_available())"` — `True` 면 GPU 로 돈다. `False` 여도 CPU 로 돌기는 한다 (훨씬 느리다).
- ultralytics 버전은 `8.3.40` 으로 맞춘다. 원래 컴퓨터와 같아야 가져간 모델이 그대로 열린다.

### 2. 그림 가져다 놓기 (사람이 그림을 줘야 한다)

그림은 둘 중 하나로 받는다. 받은 폴더를 `prepare.py` 에 넘기면 `dataset/images/` 에 맞는 이름으로 놓는다.

```
python prepare.py <그림이 있는 폴더>
```

- **(가) 원래 컴퓨터의 `experiments/tracking/outputs/products_rf/images/` 폴더** (1,194장) — USB · 비공개 저장소로 옮긴다.
- **(나) Roboflow 에서 내보낸 폴더** (`train` · `valid` · `test` 가 든 것). 원래 컴퓨터의 `source/labeling_object/` 와 **같은 내보내기**여야 한다
  (파일 이름의 `rf.` 뒤 글자가 같아야 짝을 찾는다 — 새로 내보내면 이름이 달라질 수 있다).

에이전트는 그림을 스스로 구하려 하지 말고 **사람에게 폴더 위치를 묻는다.**

### 3. 검사

```
python check.py --deep
```

`결과: 학습을 시작해도 된다` 가 나와야 한다. `!` 줄이 있으면 고치고 다시 한다. 기대하는 수: 그림 1,194 / 1,194장 · 상자 5,981개 · train 768 · val 133 · test 293.

### 4. 학습

```
python train.py
```

- 기본값: 모델 `yolov8n.pt` · 해상도 1280 · 최대 100바퀴 · 20바퀴 동안 안 나아지면 멈춤 · GPU 가 있으면 GPU.
- `yolov8n.pt` 가 없으면 ultralytics 가 내려받는다 (인터넷 필요).
- 메모리가 모자라 멈추면(`CUDA out of memory`) `--batch 8`, 그래도 안 되면 `--batch 4`.
- 중간에 끊기면 `python train.py --resume <이름>` (이름은 `runs/` 아래 폴더 이름, 기본 `yolov8n_1280`).
- 돌아가는지만 1~2분 안에 보려면: `python train.py --epochs 1 --fraction 0.05 --imgsz 320 --name smoke` (이 결과는 버린다).

걸리는 시간 (어림 — 원래 컴퓨터 CPU 에서만 쟀다): CPU 8스레드에서 한 바퀴 약 9분, 50바퀴 약 7.6시간. GPU 는 재지 않았고 10~40배 빠를 것으로 본다.

### 5. 더 해 볼 것 (시간이 되면 — 하나씩, 이름을 다르게)

| 무엇 | 명령 | 왜 |
|---|---|---|
| 더 큰 모델 | `python train.py --model yolov8s.pt --name yolov8s_1280` | 상품이 작아 큰 모델이 나을 수 있다. 대신 원래 컴퓨터(CPU)에서 돌릴 때 그만큼 느리다 |
| 낮은 해상도 | `python train.py --imgsz 960 --name yolov8n_960` | 실시간으로 돌릴 때 빠르다. 점수가 얼마나 떨어지는지 본다 |

어느 것이 나은지는 `result.json` 의 **검증(val)** 점수로 견준다.

### 6. 끝난 뒤 알릴 것

`train.py` 가 끝나면 점수 요약과 `가져갈 폴더: …/handoff/<이름>` 을 찍는다. 사람에게 이것을 알린다:

1. 가져갈 폴더 위치 (`handoff/<이름>/` — `best.pt` · `result.json` · `results.csv` · `args.yaml`).
2. `train.py` 가 찍은 점수 요약 전부 (검증 · 시험, 전체 · 카메라별).
3. 몇 바퀴 돌았고 왜 멈췄는지 (다 돌았는지 · 안 나아져서 멈췄는지 · 오류), 걸린 시간, GPU 이름.
4. 기본값과 다르게 한 것 (batch 를 줄였다 등)과 겪은 문제.
5. **확인하지 못한 것**을 숨기지 않는다 (예: `--deep` 검사를 안 했다, 중간에 끊겨 이어서 했다).

## 원래 컴퓨터로 가져온 뒤

`handoff/<이름>/` 폴더를 원래 컴퓨터의 `experiments/tracking/training/products/handoff/` 에 둔다 (저장소에는 안 올라간다).
점수를 원래 컴퓨터에서 다시 내 보려면 (그림이 `dataset/images/` 에 있어야 한다):

```
yolo detect val model=handoff/<이름>/best.pt data=dataset/data.yaml split=test imgsz=1280 device=cpu
```

## 데이터 설명

- 출처: 3차 촬영 영상에서 뽑은 장면을 Roboflow 로 찍은 것. `scripts/product_coco.py` 로 이 모양으로 바꿨다 (`--publish` 가 이 폴더의 `dataset/` 을 만든다).
- 종류: `0 cantata`(칸타타 — 빨간 뚜껑 병 캔) · `1 donggwaru`(동과루 — 흰 캔) · `2 applecider`(애플사이다 — 노란 캔).
- 그림 이름: `<영상>_<카메라>_rf<순번>.jpg`. `angle0` = 선반 카메라, `angle1` = 전체 카메라. 전부 1920×1080.
- 라벨: YOLO 형식 (`종류 가운데x 가운데y 가로 세로`, 0~1). 빈 파일 = 상품이 없는 그림 (37장).

| | 학습 train | 검증 val | 시험 test |
|---|---|---|---|
| 그림 | 768 | 133 | 293 |
| 선반 카메라 (angle0) | 411 | 95 | 189 |
| 전체 카메라 (angle1) | 357 | 38 | 104 |
| 영상 | 22편 | 4편 | 8편 |

상자 5,981개: 동과루 2,978 · 칸타타 1,999 · 애플사이다 1,004.

파일:

| | |
|---|---|
| `dataset/data.yaml` | 학습기가 읽는 것. 경로를 안 적어 어느 컴퓨터에서나 이 폴더 기준으로 읽는다 |
| `dataset/data_angle0.yaml` · `data_angle1.yaml` | 카메라별 점수를 낼 때 |
| `dataset/train.txt` · `val.txt` · `test.txt` (+ `_angle0` · `_angle1`) | 나눔별 그림 목록 |
| `dataset/labels/` | 라벨 1,194개 |
| `dataset/frames.csv` | 그림마다 한 줄: 원본 파일 이름 · 영상 · 카메라 · 나눔 · 상자 수 · 파일 크기 · sha1 |
| `dataset/images/` | 비어 있다 — `prepare.py` 가 채운다 |
| `prepare.py` · `check.py` · `train.py` | 그림 가져다 놓기 · 검사 · 학습과 점수 |

## 알아 둘 것 (한계)

- **검증 쪽이 작다.** 4편 · 133장이고, 전체 카메라 검증은 영상 한 편(C13)의 38장이다. 학습이 멈추는 때와 '가장 좋은 모델'이 이 한 편에 맞춰질 수 있다.
- 상품 3종 · 매장 한 곳 · 지금 카메라 자리 · 같은 사람 몇 명이다. 시험 영상도 같은 날 같은 매장이라, 시험 점수가 높아도 '다른 매장 · 다른 카메라 자리에서도 된다'는 뜻이 아니다.
- 라벨을 눈으로 확인한 것은 8장뿐이다. 다른 사람의 검수가 없다. 전체 카메라에서는 뒤쪽 캔의 윗부분에도 상자가 있다.
- 상품이 작다 (선반 카메라에서 캔 가로 26픽셀쯤). 해상도를 낮추면 더 작아진다: 1280 에서 17픽셀 · 960 에서 13픽셀 · 640 에서 9픽셀.
- `prepare.py` · `check.py` · `train.py` 는 원래 컴퓨터(Windows · CPU)에서만 돌려 봤다 — 그림 가져다 놓기(두 가지 폴더 모양) → 검사 → 아주 작은 학습(1바퀴 · 그림 5% · 해상도 320) → 점수 → `handoff/` 까지, 그리고 `--score`. **GPU 에서는 돌려 보지 못했다** — `train.py` 의 GPU 쪽(`--device auto` 가 GPU 를 고르는 것, 기본 batch 16)은 확인이 안 된 부분이다. **`--resume` 도 돌려 보지 못했다.** Linux · Mac 에서도 돌려 보지 못했다.
- 그 작은 학습의 점수는 전부 0 이었다 (그만큼만 학습하면 당연하다). 그래서 **점수가 제대로 나오는 학습은 아직 한 번도 끝까지 가 보지 않았다.**
- `train.py` 는 학습이 끝난 뒤 시험 점수까지 한 번에 낸다. 여러 설정을 돌리면 시험 점수를 여러 번 보게 된다 — 그 가운데 높은 것을 고르면 시험 점수가 부푼다 ('하면 안 되는 것' 4번).
- 전체 기록: `experiments/tracking/docs/limitations.md` D-9, `experiments/tracking/labels/products_shoot3/README.md`.
