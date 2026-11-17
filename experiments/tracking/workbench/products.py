"""상품 라벨 찍기 — 3차 촬영 상품 탐지 데이터의 상자를 읽고 쓴다 (화면은 workbench/static/products.js).

어느 장면을 찍을지는 scripts/product_frames.py 가 정해 둔 표(labels/products_shoot3/frames.csv)를 따른다.
라벨은 그림마다 파일 하나: labels/products_shoot3/boxes/<그림 이름>.json

    {"image": "C01_angle0_00031.jpg", "width": 1920, "height": 1080, "done": true, "skip": false,
     "boxes": [{"cls": 0, "x1": 812, "y1": 401, "x2": 843, "y2": 470, "held": false}, ...]}

  - 파일이 없다 = 아직 안 찍었다. done 이 false = 찍는 중 (고칠 때마다 저절로 저장된다). done 이 true = 다 찍었다.
    **학습에는 done 인 장면만 쓴다** — 덜 찍은 장면은 안 찍힌 상품을 '상품 아님'으로 가르치기 때문이다.
  - done 이고 boxes 가 비었다 = '상품이 안 보인다'고 찍었다 (배경 장면으로 학습에 들어간다).
  - 좌표는 그림 픽셀. cls 는 classes.yaml 의 번호. held = 손에 든 상품 (YOLO 라벨에는 안 실리지만 나중에 따로 세려고 적어 둔다).
  - skip = 이 장면은 학습에서 뺀다 (흐려서 무엇인지 모를 때).
라벨에는 얼굴이 없으므로 저장소에 올린다. 그림은 outputs/ 에만 있다.

초안: 안 찍은 장면을 열면 같은 카메라에서 다 찍은 가장 가까운 장면의 상자를 초안으로 준다 (손에 든 것은 빼고).
선반 위 상품은 장면이 바뀌어도 거의 그 자리라서다. 초안은 사람이 확인(저장)해야 라벨이 된다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import product_frames as pf  # noqa: E402

MIN_SIDE = 3.0
"""이보다 작은 상자(픽셀)는 잘못 누른 것으로 보고 버린다."""
ORDER = {"angle0": 0, "angle1": 1}
CAMERA_NAME = {"angle0": "선반 카메라", "angle1": "전체 카메라"}


def read_classes(path: Path) -> list[dict]:
    """classes.yaml → [{"id", "name", "korean", "lane"}] (번호순)."""
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    korean = raw.get("korean") or {}
    lane = {name: lane for lane, name in (raw.get("lane") or {}).items()}
    return [{"id": int(i), "name": name, "korean": korean.get(name, name), "lane": lane.get(name, "")}
            for i, name in sorted((raw.get("names") or {}).items(), key=lambda kv: int(kv[0]))]


def dump(data: dict) -> str:
    """라벨 파일 글 — 상자 하나가 한 줄 (고친 곳이 git 에서 줄 단위로 보이게)."""
    head = {k: v for k, v in data.items() if k != "boxes"}
    boxes = data["boxes"]
    lines = [json.dumps(head, ensure_ascii=False)[:-1] + ', "boxes": [']
    lines += [" " + json.dumps(b, ensure_ascii=False) + ("," if i < len(boxes) - 1 else "") for i, b in enumerate(boxes)]
    return "\n".join(lines) + "\n]}\n"


class Store:
    def __init__(self, dataset: Path | None = None, images: Path | None = None, batch: int = 1):
        self.dataset = dataset or pf.DATASET
        self.images = images or pf.OUT / "images"
        self.batch = batch
        self._plan: tuple[float, list[dict]] | None = None

    # ------------------------------------------------------------------ 표

    @property
    def boxes_dir(self) -> Path:
        return self.dataset / "boxes"

    def classes(self) -> list[dict]:
        return read_classes(self.dataset / "classes.yaml")

    def frames(self) -> list[dict]:
        """찍을 장면 (묶음 batch). 카메라 → 영상 → 장면 순서 — 옆 장면의 상자를 초안으로 이어 쓰기 좋게."""
        path = self.dataset / "frames.csv"
        if not path.exists():
            raise ValueError(f"{path.name} 없음 — 먼저 python scripts/product_frames.py plan")
        stamp = path.stat().st_mtime
        if self._plan is None or self._plan[0] != stamp:
            rows = [r for r in pf.read_plan(path) if self.batch == 0 or str(r["batch"]) == str(self.batch)]
            rows.sort(key=lambda r: (ORDER.get(r["camera"], 9), r["pair"], int(r["frame"])))
            self._plan = (stamp, rows)
        return self._plan[1]

    def row(self, image: str) -> dict:
        for r in self.frames():
            if r["image"] == image:
                return r
        raise ValueError(f"표에 없는 그림: {image}")

    # ------------------------------------------------------------------ 읽고 쓰기

    def path(self, image: str) -> Path:
        return self.boxes_dir / (Path(image).stem + ".json")

    def load(self, image: str) -> dict | None:
        """찍어 둔 라벨. 없으면 None."""
        path = self.path(self.row(image)["image"])
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"image": image, "width": data.get("width"), "height": data.get("height"),
                "done": bool(data.get("done")), "skip": bool(data.get("skip")), "boxes": data.get("boxes") or []}

    def clean(self, boxes, width: int, height: int) -> list[dict]:
        """화면이 보낸 상자를 그림 안으로 자르고, 너무 작은 것 · 모르는 종류는 막는다."""
        known = {c["id"] for c in self.classes()}
        out = []
        for b in boxes or []:
            try:
                cls = int(b["cls"])
                x1, x2 = sorted((float(b["x1"]), float(b["x2"])))
                y1, y2 = sorted((float(b["y1"]), float(b["y2"])))
            except (KeyError, TypeError, ValueError):
                raise ValueError("상자를 읽지 못했습니다") from None
            if cls not in known:
                raise ValueError(f"모르는 종류 번호: {cls}")
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
            if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
                continue
            out.append({"cls": cls, "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
                        "held": bool(b.get("held"))})
        out.sort(key=lambda b: (b["x1"], b["y1"], b["cls"]))
        return out

    def save(self, image: str, boxes, width, height, skip: bool = False, done: bool = False) -> dict:
        image = self.row(image)["image"]          # 표에 있는 이름만 — 다른 곳에 쓰지 못하게
        try:
            width, height = int(width), int(height)
        except (TypeError, ValueError):
            raise ValueError("그림 크기를 모릅니다") from None
        if width <= 0 or height <= 0:
            raise ValueError("그림 크기를 모릅니다")
        data = {"image": image, "width": width, "height": height, "done": bool(done), "skip": bool(skip),
                "boxes": self.clean(boxes, width, height)}
        path = self.path(image)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(dump(data), encoding="utf-8", newline="\n")
        tmp.replace(path)
        return data

    def clear(self, image: str) -> bool:
        """라벨을 지워 '안 찍은 것'으로 되돌린다."""
        path = self.path(self.row(image)["image"])
        if not path.exists():
            return False
        path.unlink()
        return True

    # ------------------------------------------------------------------ 초안

    def draft(self, image: str) -> dict | None:
        """같은 카메라에서 다 찍은 가장 가까운 장면의 상자 (손에 든 것 · 뺀 장면 · 찍는 중인 장면 제외).

        같은 영상 안에서는 장면 번호가 가까운 것, 없으면 찍는 순서로 가까운 다른 영상의 장면.
        """
        rows = self.frames()
        here = next((i for i, r in enumerate(rows) if r["image"] == image), None)
        if here is None:
            raise ValueError(f"표에 없는 그림: {image}")
        me = rows[here]
        best = None
        for i, r in enumerate(rows):
            if i == here or r["camera"] != me["camera"] or not self.path(r["image"]).exists():
                continue
            same = r["pair"] == me["pair"]
            near = abs(int(r["frame"]) - int(me["frame"])) if same else abs(i - here)
            key = (0 if same else 1, near, i)
            if best is not None and key >= best[0]:
                continue
            src = self.load(r["image"])
            if src and src["done"] and not src["skip"]:
                best = (key, r, src)
        if best is None:
            return None
        boxes = [dict(b) for b in best[2]["boxes"] if not b.get("held")]
        return {"from": best[1]["image"], "boxes": boxes} if boxes else None

    # ------------------------------------------------------------------ 화면에 주는 것

    def listing(self) -> dict:
        frames = []
        for r in self.frames():
            entry = {k: r[k] for k in ("image", "pair", "camera", "reason", "lane", "action", "split")}
            entry["frame"] = int(r["frame"])
            entry["time_sec"] = float(r["time_sec"])
            path = self.path(r["image"])
            entry.update(self.state(json.loads(path.read_text(encoding="utf-8")) if path.exists() else None))
            frames.append(entry)
        have = sum(1 for r in frames if (self.images / r["image"]).exists())
        return {"classes": self.classes(), "cameras": CAMERA_NAME, "frames": frames, "have_images": have,
                "batch": self.batch}

    @staticmethod
    def state(data: dict | None) -> dict:
        """목록에 보이는 한 장의 상태: todo(안 찍음) · doing(찍는 중) · done(끝)."""
        if data is None:
            return {"state": "todo", "n": 0, "held": 0, "skip": False}
        boxes = data.get("boxes") or []
        return {"state": "done" if data.get("done") else "doing", "n": len(boxes),
                "held": sum(1 for b in boxes if b.get("held")), "skip": bool(data.get("skip"))}

    def frame(self, image: str) -> dict:
        row = self.row(image)
        saved = self.load(image)
        return {"image": row["image"], "saved": saved, "draft": None if saved else self.draft(image),
                "exists": (self.images / row["image"]).exists()}
