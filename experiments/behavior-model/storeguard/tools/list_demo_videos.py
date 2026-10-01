"""대시보드에 넣을 영상 고르기.

인덱스에서 영상 경로와 **사건 발생 시각**을 뽑아 보여준다.
시각을 알면 데모할 때 언제 알림이 뜨는지 미리 알 수 있다.

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.tools.list_demo_videos
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.tools.list_demo_videos --class fall --limit 10
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.tools.list_demo_videos --store SMC --env
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from storeguard.config import Config

KO = {"fall": "전도", "broken": "파손", "theft": "절도"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--class", dest="cls", choices=["fall", "broken", "theft", "all"],
                    default="all")
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="test",
                    help="기본 test(배포 Validation). 학습에 쓰지 않은 영상이라 데모에 적합하다")
    ap.add_argument("--store", default=None, help="매장 코드로 거르기 (DYA/DYB/SYA/SYB/SMA/SMB/SMC)")
    ap.add_argument("--limit", type=int, default=5, help="클래스당 표시 개수")
    ap.add_argument("--env", action="store_true",
                    help=".env 에 그대로 붙여 넣을 수 있는 형태로 출력")
    a = ap.parse_args(argv)

    cfg = Config.load(a.config)
    index_dir = cfg.path("paths.index")
    fps = float(cfg["data.sample_fps"])

    rows = {}
    for line in (index_dir / "clips.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["stem"]] = r

    splits = json.loads((index_dir / "splits.json").read_text(encoding="utf-8"))
    if a.split == "all":
        stems = [s for k in ("train", "val", "test") for s in splits.get(k, [])]
    else:
        stems = splits.get(a.split, [])

    classes = ["fall", "broken", "theft"] if a.cls == "all" else [a.cls]
    picked: list[dict] = []

    for cls in classes:
        got = [rows[s] for s in stems
               if s in rows and rows[s].get("class_name") == cls
               and (not a.store or rows[s].get("store") == a.store)]
        if not a.env:
            print(f"\n=== {KO.get(cls, cls)} ({cls}) — {a.split} 분할에 {len(got)}개"
                  + (f", 매장 {a.store}" if a.store else ""))
        for r in got[: a.limit]:
            ev = (r.get("events") or [{}])[0]
            s = ev.get("start_frame")
            e = ev.get("end_frame")
            span = ("사건 없음" if s is None
                    else f"사건 {s / fps:.1f}~{(e if e is not None else s) / fps:.1f}초")
            picked.append(r)
            if not a.env:
                print(f"  매장 {r.get('store'):3s}  카메라 {r.get('camera'):2s}  {span}")
                print(f"     {r.get('video_path')}")

    if a.env:
        ids = ",".join(f"cam{i + 1}" for i in range(len(picked)))
        print(f"CAMERAS={ids}")
        for i, r in enumerate(picked, start=1):
            ev = (r.get("events") or [{}])[0]
            s = ev.get("start_frame")
            # 주석은 반드시 별도 줄에 둔다. 값 뒤에 붙이면 경로 일부로 읽힐 수 있다.
            if s is not None:
                print(f"# 알림 예상 {s / fps:.0f}초 부근")
            print(f"CAM{i}_NAME={i}번 카메라 - {KO.get(r['class_name'], r['class_name'])}")
            print(f"CAM{i}_SOURCE={r.get('video_path')}")
    else:
        print("\n.env 에 바로 넣을 형태로 보려면 --env 를 붙일 것.")
        print("값을 바꾼 뒤에는 서버를 껐다 켜야 반영된다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
