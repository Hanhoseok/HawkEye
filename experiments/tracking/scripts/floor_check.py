"""3차 촬영: 두 카메라의 바닥 좌표가 같은 좌표인지 검증한다 (docs/multiview-progress.md 후속 27).

    python scripts/floor_check.py tiles     # area_setting: 두 화면을 위에서 본 그림으로 펴서 타일 줄눈이 얼마나 어긋나나
    python scripts/floor_check.py feet      # 영상마다: 두 카메라에 발목이 같이 보인 순간, 같은 사람의 발이 바닥 위에서 얼마나 떨어져 나오나

두 검사는 서로 다른 것을 본다.
- tiles: 바닥 지도 자체. 보정에 쓴 것은 노란 메모지 4장의 중심뿐이라, 메모지와 상관없는 타일 줄눈이 두 카메라에서 같은 자리에 오는지는
  독립된 확인이다. area_setting 장면 한 장으로 재므로 '그 장면의 카메라 자리'에서만 맞다는 뜻이다.
- feet: 영상마다 카메라가 조금씩 돌아간 것을 되돌리는 단계(sync_floor.py register)와 시차까지 포함한 끝에서 끝 확인. 사람이 실제로 다닌 자리만 본다.
  기록(outputs/dual_eval_shoot3)이 있어야 한다 — python scripts/dual_eval.py --shoot 3 run

아무것도 고치지 않는다. 확인 그림은 outputs/floor_check/ 에.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.zones.floor import FloorMap, detail_floor_map, foot_pixel  # noqa: E402

OUT = ROOT / "outputs" / "floor_check"
ZONES = ROOT / "zones" / "store01_shoot3"
SOURCE = ROOT.parent.parent / "source"
AREA = {"angle0": SOURCE / "2(angle0)" / "area_setting.mp4", "angle1": SOURCE / "2(angle1)" / "area_setting.mp4"}

# ---------------------------------------------------------------------- tiles

SCALE = 4.0
"""위에서 본 그림의 배율 (1cm = 4px)."""
SPAN = (-60.0, 120.0, -100.0, 160.0)
"""위에서 볼 바닥 범위 (x0, x1, y0, y1 cm) — 두 카메라가 함께 보는 계산대·선반 앞."""
WINDOW_CM, STEP_CM, SEARCH_CM = 30.0, 15.0, 12.0
MIN_MATCH = 0.35
"""줄눈 무늬 일치도(정규화 상관)가 이보다 낮은 창은 믿지 않는다 (줄눈이 없거나 한쪽에 물건이 놓인 자리)."""


def topdown(image: np.ndarray, fm: FloorMap) -> np.ndarray:
    h, w = image.shape[:2]
    x0, x1, y0, y1 = SPAN
    floor_px = np.float32([[(fx - x0) * SCALE, (fy - y0) * SCALE] for fx, fy in fm.floor_points])
    image_px = np.float32([[x * w, y * h] for x, y in fm.image_points])
    m = cv2.getPerspectiveTransform(image_px, floor_px)
    return cv2.warpPerspective(image, m, (int((x1 - x0) * SCALE), int((y1 - y0) * SCALE)))


def floor_mask(image: np.ndarray) -> np.ndarray:
    """회색 타일 바닥만 (나무 가구·노란 메모지·검은 물건·화면 밖은 뺀다)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] < 60) & (hsv[..., 2] > 90) & (hsv[..., 2] < 235)
    return cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)


def grout(image: np.ndarray) -> np.ndarray:
    """타일 줄눈(가늘고 어두운 줄)만 남긴 그림."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)


def tile_offsets(top0: np.ndarray, top1: np.ndarray) -> list[dict]:
    """위에서 본 두 그림을 창마다 견줘, 선반 카메라의 줄눈이 전체 카메라의 어디에 오는지 (어긋남 cm)."""
    m0, m1 = floor_mask(top0), floor_mask(top1)
    g0, g1 = grout(top0) * m0, grout(top1) * m1
    both = m0 & m1
    win, step, search = int(WINDOW_CM * SCALE), int(STEP_CM * SCALE), int(SEARCH_CM * SCALE)
    out = []
    for y in range(search, both.shape[0] - win - search, step):
        for x in range(search, both.shape[1] - win - search, step):
            if both[y:y + win, x:x + win].mean() < 0.85:
                continue
            patch = g0[y:y + win, x:x + win]
            if patch.std() < 1.0:
                continue
            area = g1[y - search:y + win + search, x - search:x + win + search]
            _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(area, patch, cv2.TM_CCOEFF_NORMED))
            dx, dy, one_way = _across_the_line(patch, (loc[0] - search) / SCALE, (loc[1] - search) / SCALE)
            out.append({"x": x / SCALE + SPAN[0] + WINDOW_CM / 2, "y": y / SCALE + SPAN[2] + WINDOW_CM / 2,
                        "dx": dx, "dy": dy, "match": float(score), "one_way": one_way, "px": (x + win // 2, y + win // 2)})
    return out


ONE_WAY_RATIO = 0.2
"""창 안 무늬가 한 방향 줄뿐인지 가르는 기준 (기울기 분포의 작은 축 ÷ 큰 축)."""


def _across_the_line(patch: np.ndarray, dx: float, dy: float) -> tuple[float, float, bool]:
    """창 안에 줄눈이 한 방향으로만 지나면 그 줄을 따라 미는 것은 구별할 수 없다 — 줄에 수직인 어긋남만 남긴다."""
    gx, gy = cv2.Sobel(patch, cv2.CV_32F, 1, 0), cv2.Sobel(patch, cv2.CV_32F, 0, 1)
    tensor = np.array([[float((gx * gx).sum()), float((gx * gy).sum())], [float((gx * gy).sum()), float((gy * gy).sum())]])
    values, vectors = np.linalg.eigh(tensor)
    if values[1] <= 0 or values[0] / values[1] >= ONE_WAY_RATIO:
        return dx, dy, False
    ex, ey = vectors[:, 1]
    along = dx * ex + dy * ey
    return float(along * ex), float(along * ey), True


def tiles() -> None:
    import sync_floor

    maps = {cam: FloorMap.load(ZONES / f"floor_{cam}.yaml") for cam in AREA}
    tops = {cam: topdown(sync_floor.median_frame(AREA[cam]), maps[cam]) for cam in AREA}
    rows = tile_offsets(tops["angle0"], tops["angle1"])
    good = [r for r in rows if r["match"] >= MIN_MATCH]
    both = floor_mask(tops["angle0"]) & floor_mask(tops["angle1"])
    print(f"두 카메라가 함께 보는 바닥: 약 {both.sum() / SCALE ** 2 / 10000:.2f}㎡ (area_setting 장면, 가구·메모지 빼고)")
    print(f"창 {WINDOW_CM:.0f}cm × {len(rows)}개 중 줄눈 무늬가 뚜렷한 {len(good)}개 — 선반 카메라 줄눈이 전체 카메라의 어디에 오나")
    for r in rows:
        mark = ("" if r["match"] >= MIN_MATCH else "  (무늬 약함 — 뺌)") + ("  (줄이 한 방향 — 수직 어긋남만)" if r["one_way"] else "")
        print(f"  바닥 ({r['x']:4.0f}, {r['y']:4.0f})cm  가로 {r['dx']:+5.2f}  세로 {r['dy']:+5.2f} cm  일치도 {r['match']:.2f}{mark}")
    if good:
        d = np.hypot([r["dx"] for r in good], [r["dy"] for r in good])
        print(f"어긋남: 중앙 {np.median(d):.1f}cm · 최대 {d.max():.1f}cm  (한 칸 = {1 / SCALE:.2f}cm 라 그보다 잘게는 못 잰다)")
    OUT.mkdir(parents=True, exist_ok=True)
    blend = cv2.addWeighted(tops["angle0"], 0.5, tops["angle1"], 0.5, 0)
    for r in good:
        cv2.circle(blend, r["px"], 4, (0, 200, 0), -1)
        cv2.arrowedLine(blend, r["px"], (int(r["px"][0] + r["dx"] * SCALE * 10), int(r["px"][1] + r["dy"] * SCALE * 10)), (0, 0, 255), 2)
    cv2.imwrite(str(OUT / "tiles.jpg"), np.hstack([tops["angle0"], tops["angle1"], blend]))
    print(f"확인 그림: {OUT / 'tiles.jpg'} (왼쪽 선반 카메라 · 가운데 전체 카메라 · 오른쪽 겹친 것, 화살표 = 어긋남 × 10)")


# ---------------------------------------------------------------------- feet

MATCH_MS = 40.0
"""두 카메라 장면을 같은 때로 보는 시간 차."""
PATCH_CENTER = (32.0, 37.0)
"""메모지 네 장의 가운데 (바닥 cm) — 보정점에서 멀수록 어긋나는지 보려고."""


def ankle_floor(fm: FloorMap, obs, width: int, height: int) -> tuple[float, float] | None:
    """발목이 보이는 사람만: 두 발목 가운데의 바닥 cm."""
    (px, py), ankle = foot_pixel(tuple(obs.bbox), getattr(obs, "keypoints", None), 0.3)
    if not ankle or not fm.on_floor_side(px / width, py / height):
        return None
    return fm.to_floor(px / width, py / height)


def feet_of(pair: str) -> list[tuple[float, float, float]]:
    """그 영상에서 두 카메라에 한 명씩만 보이고 둘 다 발목이 보인 순간들의 (바닥 x, y, 두 카메라 발 거리 cm)."""
    import dual_eval as de
    from sync_pair import SyncTable

    log = de.PairLog(pair)
    zdir = de.zones_dir(pair)
    overview = FloorMap.load(zdir / "floor.yaml")
    detail = detail_floor_map(yaml.safe_load((zdir / "angle0.yaml").read_text(encoding="utf-8")))
    if detail is None:
        raise ValueError(f"{pair}: 선반 카메라 바닥 지도(floor_calibration)가 없음")
    offset = SyncTable.load(de.LABELS / "sync.csv").offset_ms(pair) or 0.0
    m0, m1 = log.meta["angle0"], log.meta["angle1"]
    seen1 = []
    for idx, obs in sorted(log.ids["angle1"].items()):
        people = [o for o in obs if o.person_id > 0]
        if len(people) == 1:
            p = ankle_floor(overview, people[0], m1["width"], m1["height"])
            if p is not None:
                seen1.append((log.ms("angle1", idx), p))
    if not seen1:
        return []
    times = np.array([t for t, _ in seen1])
    out = []
    for idx, obs in sorted(log.ids["angle0"].items()):
        people = [o for o in obs if o.person_id > 0]
        if len(people) != 1:
            continue
        p0 = ankle_floor(detail, people[0], m0["width"], m0["height"])
        if p0 is None:
            continue
        t = log.ms("angle0", idx) + offset
        j = int(np.argmin(np.abs(times - t)))
        if abs(times[j] - t) <= MATCH_MS:
            p1 = seen1[j][1]
            out.append(((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, float(np.hypot(p0[0] - p1[0], p0[1] - p1[1]))))
    return out


def feet(only: str | None) -> None:
    import dual_eval as de

    de.set_profile(3)
    everything, per_pair = [], []
    for pair in de.pairs(only):
        if not (de.OUT / pair / ".done").exists():
            continue
        try:
            rows = feet_of(pair)
        except de.PAIR_ERRORS as e:
            print(f"  (건너뜀 {pair}: {e})")
            continue
        if rows:
            d = np.array([r[2] for r in rows])
            per_pair.append((pair, len(d), float(np.median(d)), float(np.percentile(d, 90))))
            everything += rows
    if not everything:
        print("견줄 순간이 없다 — 기록(outputs/dual_eval_shoot3)이 있는지 확인")
        return
    print("영상마다: 두 카메라에 한 명씩, 둘 다 발목이 보인 순간의 발 거리 (같은 사람, cm)")
    for pair, n, med, p90 in per_pair:
        print(f"  {pair:6s} 표본 {n:4d}  중앙 {med:5.1f}  90% {p90:5.1f}" + ("   ← 큼" if med > 20 else ""))
    d = np.array([r[2] for r in everything])
    print(f"전체 {len(d)}표본: 중앙 {np.median(d):.1f}cm · 75% {np.percentile(d, 75):.1f} · 90% {np.percentile(d, 90):.1f} · 95% {np.percentile(d, 95):.1f}")
    away = np.hypot([r[0] - PATCH_CENTER[0] for r in everything], [r[1] - PATCH_CENTER[1] for r in everything])
    print("메모지(보정점) 가운데에서 떨어진 거리별:")
    for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 250), (250, 1000)):
        part = d[(away >= lo) & (away < hi)]
        if len(part):
            print(f"  {lo:3d} ~ {hi:4d}cm: 표본 {len(part):5d}  중앙 {np.median(part):5.1f}  90% {np.percentile(part, 90):5.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("tiles", "feet"))
    ap.add_argument("--only", help="feet: 짝 이름 무늬 (예: 'N*')")
    args = ap.parse_args()
    if args.what == "tiles":
        tiles()
    else:
        feet(args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
