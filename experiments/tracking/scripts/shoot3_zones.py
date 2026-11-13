"""3차 촬영 선반 카메라(angle0) 구역 — area_setting 기준으로 한 번 그리고, 영상마다 카메라 움직임만큼 옮긴다.

카메라가 영상 사이에 움직였다 (T 영상은 레인 자리에서 약 27px, 레인 하나 폭). 그래서 구역 파일
zones/store01_shoot3/angle0.yaml 은 area_setting 장면 좌표로 적고, sync_floor.py register 가 만든
영상→area_setting 호모그래피(outputs/sync_floor/register.csv)의 역으로 영상 좌표에 옮긴다.

    python scripts/shoot3_zones.py check            # 39편 첫 장면에 레인 구역을 겹친 확인 그림 한 장
    python scripts/shoot3_zones.py check --at take  # 정답(takes.csv) 시각 장면에 — 손이 그 레인에 있는지
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inputs.frame_times import index_at, read_times  # noqa: E402

ZONES = ROOT / "zones" / "store01_shoot3" / "angle0.yaml"
REGISTER = ROOT / "outputs" / "sync_floor" / "register.csv"
VIDEOS = ROOT.parent.parent / "source" / "2(angle0)"
LABELS = ROOT / "labels" / "store01_shoot3"
OUT = ROOT / "outputs" / "zones3"
COLORS = {"L1": (60, 60, 255), "L2": (80, 220, 80), "L3": (0, 220, 255), "shelf": (255, 160, 60)}


def load_zones(path: Path = ZONES) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_homographies(path: Path = REGISTER, camera: str = "angle0") -> dict[str, np.ndarray]:
    """영상 픽셀 → area_setting 픽셀."""
    with path.open(encoding="utf-8", newline="") as f:
        return {r["pair"]: np.array([float(v) for v in r["h"].split()]).reshape(3, 3)
                for r in csv.DictReader(f) if r["camera"] == camera}


def ref_polygons(zones: dict, size: tuple[int, int]) -> dict[str, np.ndarray]:
    """구역 이름 → area_setting 픽셀 다각형 (shelf 는 box 사각형)."""
    w, h = size
    out = {}
    for s in zones.get("shelves", []):
        x0, y0, x1, y1 = s["box"]
        out[s["name"]] = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float) * [w, h]
    for z in zones.get("zones", []):
        out[z["name"]] = np.array(z["polygon"], float) * [w, h]
    return out


def to_clip(poly: np.ndarray, h_clip_to_ref: np.ndarray) -> np.ndarray:
    """area_setting 픽셀 → 그 영상 픽셀."""
    pts = cv2.perspectiveTransform(poly.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(h_clip_to_ref))
    return pts.reshape(-1, 2)


def clip_polygons(pair: str, zones: dict, homs: dict[str, np.ndarray], size=(1920, 1080)) -> dict[str, np.ndarray]:
    return {k: to_clip(v, homs[pair]) for k, v in ref_polygons(zones, size).items()}


def frame_at_sec(path: Path, sec: float) -> np.ndarray | None:
    times = read_times(path)
    want = index_at(times, sec * 1000.0)
    cap = cv2.VideoCapture(str(path))
    i, frame = -1, None
    while i < want and cap.grab():
        i += 1
    ok, f = cap.retrieve()
    cap.release()
    return f if ok else frame


def takes(path: Path = LABELS / "takes.csv") -> dict[str, list[tuple[float, str, str]]]:
    out: dict[str, list] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("time_sec") or "").strip():
                out.setdefault(r["pair"], []).append((float(r["time_sec"]), r["lane"], r["action"]))
    return out


def tile(img: np.ndarray, polys: dict[str, np.ndarray], label: str, focus: str | None = None) -> np.ndarray:
    for name, p in polys.items():
        cv2.polylines(img, [p.astype(np.int32)], True, COLORS.get(name, (255, 255, 255)),
                      3 if name == focus else 1, cv2.LINE_AA)
    x0, y0, x1, y1 = polys["shelf"][:, 0].min(), polys["shelf"][:, 1].min(), polys["shelf"][:, 0].max(), polys["shelf"][:, 1].max()
    pad = 60
    crop = img[max(0, int(y0) - pad):int(y1) + pad, max(0, int(x0) - pad):int(x1) + pad]
    crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(crop, (0, 0), (crop.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(crop, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return crop


def sheet(tiles: list[np.ndarray], cols: int = 6) -> np.ndarray:
    hh = max(t.shape[0] for t in tiles)
    ww = max(t.shape[1] for t in tiles)
    rows = []
    for i in range(0, len(tiles), cols):
        row = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, ww - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles[i:i + cols]]
        row += [np.zeros((hh, ww, 3), np.uint8)] * (cols - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def check(at: str) -> Path:
    zones, homs = load_zones(), load_homographies()
    pairs = sorted(p.stem for p in VIDEOS.glob("*.mp4") if p.stem != "area_setting")
    labeled = takes() if at == "take" else {}
    tiles = []
    for pair in pairs:
        if pair not in homs:
            print(f"  {pair}: 카메라 맞추기 없음 (sync_floor.py register)")
            continue
        polys = clip_polygons(pair, zones, homs)
        moments = [(0.0, None, "first")] if at == "first" else [(t, lane, a) for t, lane, a in labeled.get(pair, [])]
        for t, lane, act in moments:
            f = frame_at_sec(VIDEOS / f"{pair}.mp4", t)
            if f is None:
                continue
            tiles.append(tile(f, polys, f"{pair} {act} {lane or ''} {t:.2f}s" if lane else pair, lane))
        print(f"  {pair}: {len(moments)}장", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"check_{at}.jpg"
    cv2.imwrite(str(out), sheet(tiles))
    print(f"→ {out}")
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["check"])
    ap.add_argument("--at", choices=["first", "take"], default="first", help="첫 장면 / 정답 시각 장면")
    args = ap.parse_args()
    check(args.at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
