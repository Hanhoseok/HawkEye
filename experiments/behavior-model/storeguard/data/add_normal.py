"""정상 영상 추가 도구.

왜 필요한가
  이 데이터셋(238-2 이상행동)에는 **정상 영상이 하나도 없다.** 세 클래스 모두 연출된 이상행동이다.
  따라서 현재의 `background_unlabeled` 는 '이상행동 영상의 이벤트 구간 밖'일 뿐 정상이 아니며,
  "정상 연속 영상의 시간당 오경보"는 측정할 수 없다.

무엇을 넣어야 하나
  아래 중 하나를 폴더에 모아 이 도구로 등록한다.
    1) 같은 AI-Hub 과제의 **정상 구매행동 카테고리** 영상
       (매장이동 / 선택 / 시험 / 구매 / 반품 / 비교)
    2) 직접 촬영한 소형 매장의 정상 영업 영상
       정상 구매, 상품 반환, 물건 줍기, 쪼그려 앉기, 개인 물건을 가방에 넣기 등을 포함할 것.
  주의: 대상 외 이상행동(방화·흡연·유기·폭행 등)을 정상 폴더에 넣지 않는다.
  라벨이 없다는 이유로 정상으로 간주하면 안 된다. **사람이 정상이라고 확인한 영상만** 넣는다.

무엇이 달라지나
  - 등록된 영상은 `class_name: normal_verified`, 이벤트 없음으로 인덱스에 들어간다.
  - 클립 생성기가 영상 전체에서 음성 클립을 뽑아 학습의 배경 클래스에 넣는다(출처는 구분 기록).
  - `storeguard.evaluate --split normal` 로 **정상 영상 시간당 오경보**를 실제로 측정할 수 있다.

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.add_normal ^
        --dir D:\\computervision\\data\\normal_raw --split-ratio 0.2
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.preprocess --workers 4
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.train --name with_normal
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from storeguard.config import Config
from storeguard.data.build_index import probe_video
from storeguard.utils import write_json

NORMAL_CLASS = "normal_verified"


def scan(directory: Path, take_from: str) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(directory.rglob("*.mp4")):
        info = probe_video(p)
        n_frames = int(info.get("video_frames") or 0)
        fps = float(info.get("video_fps") or 0.0)
        take = p.parent.name if take_from == "folder" else p.stem
        rows.append({
            "stem": p.stem,
            "split_src": "normal",
            "has_label": False,
            "has_video": True,
            "class_name": NORMAL_CLASS,
            "class_code": "N",
            "store": p.parent.name,
            "camera": "NA",
            "take_id": f"NORMAL::{take}",
            "video_path": str(p),
            "video_bytes": p.stat().st_size,
            "label_frames": n_frames,
            "events": [],
            "problems": [] if info.get("probe_ok") else ["영상 디코딩 실패"],
            "normal_source": str(directory),
            **info,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--dir", type=Path, required=True, help="정상 영상(mp4)이 든 폴더")
    ap.add_argument("--take-from", choices=["folder", "file"], default="folder",
                    help="촬영 사건 그룹 기준. 같은 장면을 여러 카메라로 찍었으면 folder")
    ap.add_argument("--split-ratio", type=float, default=0.2,
                    help="정상 영상 중 평가(normal 분할)로 뺄 비율. 나머지는 학습에 쓴다")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args(argv)

    cfg = Config.load(a.config)
    index_dir = cfg.path("paths.index")
    clips_path = index_dir / "clips.jsonl"
    splits_path = index_dir / "splits.json"

    if not a.dir.is_dir():
        print(f"폴더가 없다: {a.dir}")
        return 2
    rows = scan(a.dir, a.take_from)
    if not rows:
        print(f"{a.dir} 안에 mp4 가 없다.")
        return 2

    existing = [json.loads(l) for l in clips_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = [r for r in existing if r.get("class_name") != NORMAL_CLASS
            or r.get("normal_source") != str(a.dir)]
    merged = keep + rows
    with open(clips_path, "w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # take 단위로 학습용 / 평가용 분리
    takes = sorted({r["take_id"] for r in rows})
    rng = random.Random(a.seed)
    rng.shuffle(takes)
    n_eval = max(1, round(len(takes) * a.split_ratio))
    eval_takes = set(takes[:n_eval])
    normal_eval = sorted(r["stem"] for r in rows if r["take_id"] in eval_takes)
    normal_train = sorted(r["stem"] for r in rows if r["take_id"] not in eval_takes)

    splits = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.exists() else {
        "train": [], "val": [], "test": [], "meta": {}}
    splits["train"] = sorted(set(splits.get("train", [])) | set(normal_train))
    splits["normal"] = normal_eval
    splits.setdefault("meta", {})["정상영상"] = {
        "폴더": str(a.dir), "영상수": len(rows), "take수": len(takes),
        "학습에_추가": len(normal_train), "평가용_normal_분할": len(normal_eval),
        "총_시간_초": round(sum((r.get("label_frames") or 0) / (r.get("video_fps") or 1)
                              for r in rows), 1),
        "주의": "이 영상들은 사람이 '정상'이라고 확인한 것이어야 한다. "
                "라벨이 없다는 이유로 정상 취급한 영상을 넣으면 평가가 왜곡된다.",
    }
    write_json(splits_path, splits)

    print(json.dumps(splits["meta"]["정상영상"], ensure_ascii=False, indent=2))
    print(f"\n인덱스 갱신: {clips_path}")
    print(f"분할 갱신: {splits_path}  (normal 분할 {len(normal_eval)}개)")
    print("\n다음 단계:")
    print("  1) storeguard.data.preprocess --workers 4   (새 영상 캐시 생성)")
    print("  2) storeguard.train --name with_normal")
    print("  3) storeguard.evaluate --run with_normal --split normal   ← 시간당 오경보 측정")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
