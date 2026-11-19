"""3차 평가 저장분에서 '선반 물건 탐지'만 상품 탐지 모델로 바꾼 저장분을 만든다 — 선반 지도에 상품 모델을 끼워 보는 길.

    python scripts/product_shelf_cache.py --model training/products/handoff/products-model-v2/yolov8n_1280_v2.pt \
        --from dual_eval_shoot3 --out dual_eval_shoot3_prodcache
    python scripts/dual_eval.py --shoot 3 replay --from dual_eval_shoot3_prodcache --out dual_eval_shoot3_prod \
        --set shelves.model_path=training/products/handoff/products-model-v2/yolov8n_1280_v2.pt
    python scripts/dual_eval.py --shoot 3 judge --out dual_eval_shoot3_prod

왜 이렇게 하나: 처음부터 다시 돌리면 한 시간이 넘는다 (docs/replay-cache.md). 사람 탐지 · 추적 · 외형 벡터 · 칸 그림 비교는
물건 모델과 상관없으므로 저장분 그대로 두고, 선반 카메라(angle0)의 물건 탐지 결과만 새 모델로 다시 뽑아 넣는다.

- 물건은 **처리한 장면 전부**에서 찾는다. 원래 실행은 필요한 장면에서만 찾는데(가린 동안 · 한가한 동안 건너뜀),
  어느 장면이 필요한지는 선반 지도의 상태 = 물건 탐지 결과에 달려 있어 모델이 바뀌면 달라진다.
- 저장분의 유효 조건(fingerprint)에서 선반 설정만 새 값으로 바꾼다. 영상 · 탐지기 · 추적기 조건은 그대로 두므로
  그쪽이 달라졌으면 replay 가 FingerprintMismatch 로 멈춘다.
- --shelves 키=값 : 선반 설정을 더 바꾼다 (예: map_score=0.25). replay 의 --set shelves.키=값 과 같게 줘야 한다.
  conf · imgsz · model_path 가 같고 이미 만든 저장분이 있으면 물건을 다시 찾지 않고 조건만 고쳐 쓴다.

모델 · 영상 · 저장분은 저장소에 없다 (outputs/ · handoff/ · source/).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DETECTION_KEYS = ("model_path", "imgsz", "conf", "ignore")
"""이 값이 같으면 이미 뽑은 물건 탐지를 다시 쓸 수 있다 (나머지 선반 설정은 탐지 결과를 바꾸지 않는다)."""


def shelf_settings(old: dict, model: str, extra: list[str]) -> dict:
    """저장분의 선반 설정 → 새 설정. model_path 를 바꾸고 --shelves 키=값 을 얹는다."""
    new = dict(old)
    new["model_path"] = model
    for item in extra:
        key, value = item.split("=", 1)
        if key not in new:
            raise SystemExit(f"선반 설정에 없는 키: {key} (있는 것: {', '.join(sorted(new))})")
        new[key] = yaml.safe_load(value)
    return new


def same_detection(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in DETECTION_KEYS)


def detect_all(video: Path, cache, detector, stride: int) -> dict[int, list]:
    """저장분이 처리한 장면 전부에서 물건을 찾는다. 장면 번호가 저장분과 어긋나면 멈춘다."""
    from src.inputs.video_source import VideoSource

    want = [f[0] for f in cache.frames]
    out: dict[int, list] = {}
    with VideoSource(str(video), stride=stride, camera_id=cache.camera) as src:
        for n, frame in enumerate(src):
            if n >= len(want):
                break
            if frame.index != want[n]:
                raise SystemExit(f"{video.name}: 장면 번호가 저장분과 다르다 ({frame.index} ≠ {want[n]}) — 영상이나 처리 간격이 바뀌었다")
            out[frame.index] = list(detector.detect(frame))
    if len(out) != len(want):
        raise SystemExit(f"{video.name}: 장면 {len(want)}장 중 {len(out)}장만 읽혔다")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="상품 탐지 모델 (.pt) — replay 의 --set shelves.model_path 에 같은 글자로 준다")
    ap.add_argument("--from", dest="source", default="dual_eval_shoot3", help="저장분이 있는 실행 폴더 (outputs/ 아래)")
    ap.add_argument("--out", required=True, help="새 저장분을 쓸 폴더 (outputs/ 아래)")
    ap.add_argument("--only", help="짝 이름 패턴 (예: C0*)")
    ap.add_argument("--shelves", action="append", default=[], metavar="키=값", help="선반 설정을 더 바꾼다 (예: map_score=0.25)")
    args = ap.parse_args()

    import dual_eval
    import sync_pair
    from src.core.config import AppConfig
    from src.replay.cache import CameraCache

    dual_eval.set_profile(3)
    base = AppConfig.load(ROOT / "config.yaml")
    src_root, out_root = ROOT / "outputs" / args.source, ROOT / "outputs" / args.out
    if src_root.resolve() == out_root.resolve():
        raise SystemExit("--out 은 --from 과 달라야 한다")
    detector = None
    for pair in dual_eval.pairs(args.only):
        src_dir, out_dir = src_root / pair, out_root / pair
        if not (src_dir / "cache" / "angle0.pkl.gz").exists():
            print(f"{pair}: 저장분 없음 (건너뜀)", flush=True)
            continue
        detail = CameraCache.load(src_dir / "cache" / "angle0.pkl.gz")
        overview = CameraCache.load(src_dir / "cache" / "angle1.pkl.gz")
        new = shelf_settings(detail.fingerprint["shelves"], args.model, args.shelves)
        made = out_dir / "cache" / "angle0.pkl.gz"
        reuse = None
        if made.exists():
            old = CameraCache.load(made)
            if same_detection(old.fingerprint["shelves"], new) and len(old.items) == len(detail.frames):
                reuse = old.items
        started = time.time()
        if reuse is None:
            if detector is None:
                from src.detectors.yolo_detector import build_item_detector

                detector = build_item_detector(new["model_path"], new["imgsz"], new["conf"], base.detector.device, new["ignore"])
            video, _ref = sync_pair.pair_files(pair, 3)
            reuse = detect_all(Path(video), detail, detector, dual_eval.STRIDE["angle0"])
        detail.items = reuse
        for cache in (detail, overview):
            cache.fingerprint = {**cache.fingerprint, "shelves": new}
            cache.save(out_dir / "cache" / f"{cache.camera}.pkl.gz")
        for extra in [src_dir / "meta.json", *src_dir.glob("times_*.json")]:
            shutil.copyfile(extra, out_dir / extra.name)
        boxes = sum(len(v) for v in reuse.values())
        print(f"{pair}: 장면 {len(reuse)}장 · 물건 상자 {boxes}개 · {time.time() - started:.0f}초", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
