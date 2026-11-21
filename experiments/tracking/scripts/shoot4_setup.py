"""4차 촬영(2026-11-19, source/3(angle0)·3(angle1)·3(log)) 준비 도구 — 판정 규칙·기준값은 건드리지 않고 세팅만 만든다.

    python scripts/shoot4_setup.py inputs       # 두 카메라 영상 짝 = 정답표 짝 인지 검사
    python scripts/shoot4_setup.py floor        # 3차 바닥 좌표계를 4차 전체 카메라 화면으로 옮김 → zones/store01_shoot4/floor.yaml
    python scripts/shoot4_setup.py check        # 레인·바닥 구역을 영상별 장면에 겹친 확인 그림 → outputs/shoot4_setup/
    python scripts/shoot4_setup.py pay-clock    # (run 뒤) 결제 서버 시계 ↔ 휴대폰 시계 차이 확인

4차는 시험용 세트다 (docs/shoot4-demo-plan.md §3): 구역·설정은 판정 결과를 보기 전에 정하고, 본 뒤에 고치지 않는다.

바닥 (floor): 3차와 같은 매장·같은 타일이라 바닥 좌표계를 새로 정하지 않고 그대로 쓴다. 3차·4차 area_setting(angle1)의
바닥 영역만 SIFT(타일 얼룩 무늬)로 맞춰 바닥 평면 호모그래피(4차 화면 → 3차 화면)를 구하고, 3차 보정점 4개를 4차 화면으로 옮긴다.
바닥 cm 구역(shelf·counter·door·aisle)·띠 축은 3차 파일 그대로 — 가구가 같은 자리에 있으므로 같은 바닥 자리를 가리킨다.

결제 시계 (pay-clock): 결제 기록과 영상의 시각은 둘 다 초 단위이고 두 기기 시계 차이를 재지 않았다. 결제가 이르게 찍히면 손님에게
못 붙으므로(계산대 구역에 있거나 3초 안에 있었던 손님에게만 붙음), 첫 판정 전에 관찰 기록(사람 위치)만으로 확인한다:
시계 차이 후보마다 '결제 시각에 계산대 구역에 사람이 있던 주문 수'를 센다. 판정·채점은 보지 않는다.
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

SHOOT = 4
SOURCE = ROOT.parent.parent / "source"
AREA = {3: SOURCE / "2(angle1)" / "area_setting.mp4", 4: SOURCE / "3(angle1)" / "area_setting.mp4"}
ZONES3 = ROOT / "zones" / "store01_shoot3"
ZONES4 = ROOT / "zones" / "store01_shoot4"
OUT = ROOT / "outputs" / "shoot4_setup"
SIZE = (1920, 1080)
FLOOR_MASK = {
    3: [(620, 560), (1150, 540), (1230, 1080), (480, 1080), (500, 760), (600, 740)],
    4: [(680, 560), (1100, 545), (1180, 1080), (560, 1080), (560, 790), (690, 770)],
}
"""area_setting(angle1) 화면에서 바닥만 보이는 영역 (가구·탁자·신발·사람 자리를 뺀 넉넉한 다각형, 픽셀)."""
MIN_FLOOR_INLIERS = 30
RATIO = 0.8
CLOCK_RANGE = (-5.0, 5.0, 0.5)
"""pay-clock 에서 볼 시계 차이 후보 (시작, 끝, 간격 — 초)."""
ZONE_COLOR = {"shelf": (60, 60, 255), "counter": (0, 200, 255), "door": (80, 220, 80), "aisle": (255, 160, 60),
              "L1": (60, 60, 255), "L2": (80, 220, 80), "L3": (0, 220, 255)}


# ---------------------------------------------------------------------- 바닥 옮기기

def floor_features(image: np.ndarray, polygon) -> tuple[list, np.ndarray | None]:
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.int32(polygon)], 255)
    gray = cv2.createCLAHE(3.0, (8, 8)).apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return cv2.SIFT_create(8000, contrastThreshold=0.01).detectAndCompute(gray, mask)


def floor_homography(img3: np.ndarray, img4: np.ndarray, mask3=None, mask4=None,
                     ratio: float = RATIO) -> tuple[np.ndarray | None, int, int]:
    """바닥 평면 호모그래피 (4차 화면 픽셀 → 3차 화면 픽셀), 인라이어 수, 일치 수."""
    k3, d3 = floor_features(img3, mask3 or FLOOR_MASK[3])
    k4, d4 = floor_features(img4, mask4 or FLOOR_MASK[4])
    if d3 is None or d4 is None or len(k3) < 8 or len(k4) < 8:
        return None, 0, 0
    knn = [x for x in cv2.BFMatcher().knnMatch(d4, d3, k=2) if len(x) == 2]
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < 8:
        return None, 0, len(good)
    src = np.float32([k4[m.queryIdx].pt for m in good])
    dst = np.float32([k3[m.trainIdx].pt for m in good])
    h, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return h, int(inl.sum()) if inl is not None else 0, len(good)


def move_points(points_norm, h_4to3: np.ndarray, size=SIZE) -> list[list[float]]:
    """3차 화면 비율 점 → 4차 화면 비율 점 (바닥 위의 점만 뜻이 있다)."""
    w, h = size
    pts = np.float64(points_norm).reshape(-1, 1, 2) * [w, h]
    out = cv2.perspectiveTransform(pts, np.linalg.inv(h_4to3)).reshape(-1, 2)
    return [[round(float(x) / w, 5), round(float(y) / h, 5)] for x, y in out]


def floor_doc(base: dict, image_points: list[list[float]], inliers: int, matches: int) -> dict:
    """3차 floor.yaml 을 4차용으로: 보정점(화면)만 바꾸고 바닥 cm 구역·띠 축은 그대로."""
    out = {k: v for k, v in base.items() if k not in ("note", "notes")}
    doc = {"camera": out.pop("camera", "angle1"),
           "note": ("4차 촬영(2026-11-19) 전체 카메라(angle1) 바닥 지도와 구역 — **area_setting 장면 기준** (영상마다 register 호모그래피로 옮겨 씀, "
                    "scripts/dual_eval.py --shoot 4).\n"
                    "3차 바닥 좌표계 그대로다: 3차·4차 area_setting 의 바닥 타일 무늬를 맞춰(scripts/shoot4_setup.py floor, "
                    f"인라이어 {inliers}/{matches}) 3차 보정점 4개를 4차 화면으로 옮겼다. 바닥 cm 구역·띠 축은 zones/store01_shoot3/floor.yaml 과 같다 "
                    "(같은 매장·같은 가구 자리). 4차 판정 결과를 보고 고치지 않는다."),
           "notes": {"source": "구역별 설명은 zones/store01_shoot3/floor.yaml 의 notes — 같은 바닥 자리."}}
    doc.update(out)
    doc["calibration"] = {"image": image_points, "floor": base["calibration"]["floor"]}
    return doc


def draw_floor_zones(image: np.ndarray, doc: dict) -> np.ndarray:
    w, h = image.shape[1], image.shape[0]
    img_pts = np.float32(doc["calibration"]["image"]) * [w, h]
    f2i = cv2.getPerspectiveTransform(np.float32(doc["calibration"]["floor"]), np.float32(img_pts))

    def to_img(pts):
        return cv2.perspectiveTransform(np.float64(pts).reshape(-1, 1, 2), f2i).reshape(-1, 2)

    out = image.copy()
    for z in doc.get("zones", []):
        p = np.int32(to_img(z["polygon"]))
        color = ZONE_COLOR.get(z["name"], (255, 255, 255))
        cv2.polylines(out, [p], True, color, 3)
        cv2.putText(out, z["name"], tuple(int(v) for v in p.mean(axis=0)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    for p in img_pts:
        cv2.circle(out, tuple(int(v) for v in p), 8, (255, 0, 255), -1)
    return out


def floor() -> int:
    from sync_floor import median_frame

    img3, img4 = median_frame(AREA[3]), median_frame(AREA[4])
    h, inl, n = floor_homography(img3, img4)
    if h is None or inl < MIN_FLOOR_INLIERS:
        print(f"바닥 맞추기 실패 (인라이어 {inl}/{n}, 최소 {MIN_FLOOR_INLIERS}) — floor.yaml 을 쓰지 않음")
        return 1
    base = yaml.safe_load((ZONES3 / "floor.yaml").read_text(encoding="utf-8"))
    points = move_points(base["calibration"]["image"], h)
    doc = floor_doc(base, points, inl, n)
    ZONES4.mkdir(parents=True, exist_ok=True)
    (ZONES4 / "floor.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    warp = cv2.warpPerspective(img3, np.linalg.inv(h), SIZE)
    cv2.imwrite(str(OUT / "floor_blend.jpg"), cv2.addWeighted(img4, 0.5, warp, 0.5, 0))
    cv2.imwrite(str(OUT / "floor_zones.jpg"), draw_floor_zones(img4, doc))
    print(f"바닥 맞춤: 인라이어 {inl}/{n}")
    print(f"  3차 보정점 {base['calibration']['image']}")
    print(f"  4차 보정점 {points}")
    print(f"→ {ZONES4 / 'floor.yaml'}\n확인 그림: {OUT / 'floor_blend.jpg'} (3차 장면을 4차 화면에 겹침 — 타일 줄눈이 겹쳐야 함), {OUT / 'floor_zones.jpg'}")
    return 0


# ---------------------------------------------------------------------- 확인 그림

def frame_at(path: Path, fraction: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n * fraction)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def check(fraction: float = 0.4) -> int:
    """짝마다 선반 카메라 받침 부분에 레인 구역을, 전체 카메라 장면에 바닥 구역을 겹쳐 모아 붙인 그림."""
    import shoot3_eval
    import shoot3_zones as Z
    import sync_pair

    zones = Z.load_zones(ZONES4 / "angle0.yaml")
    floor_base = yaml.safe_load((ZONES4 / "floor.yaml").read_text(encoding="utf-8"))
    h0, h1 = shoot3_eval.homographies("angle0", SHOOT), shoot3_eval.homographies("angle1", SHOOT)
    box = zones["shelves"][0]["box"]
    lane_tiles, floor_tiles = [], []
    for pair in sync_pair.all_pairs(SHOOT):
        video, ref = sync_pair.pair_files(pair, SHOOT)
        if pair not in h0 or pair not in h1:
            print(f"  {pair}: 카메라 맞추기 없음 — 건너뜀")
            continue
        img = frame_at(video, fraction)
        polys = Z.clip_polygons(pair, zones, h0, SIZE)
        for name, poly in polys.items():
            cv2.polylines(img, [np.int32(poly)], True, ZONE_COLOR.get(name, (255, 160, 60)), 1)
        cx, cy = polys["shelf"].mean(axis=0)
        half_w = (box[2] - box[0]) * SIZE[0] * 0.9
        half_h = (box[3] - box[1]) * SIZE[1] * 1.0
        x0, y0 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
        crop = cv2.resize(img[y0:int(cy + half_h), x0:int(cx + half_w)], (420, 280), interpolation=cv2.INTER_CUBIC)
        cv2.putText(crop, pair, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        lane_tiles.append(crop)
        doc = shoot3_eval.pair_floor(floor_base, h1[pair])
        small = cv2.resize(draw_floor_zones(frame_at(ref, fraction), doc), (480, 270))
        cv2.putText(small, pair, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        floor_tiles.append(small)

    def sheet(tiles: list[np.ndarray], cols: int) -> np.ndarray:
        blank = np.zeros_like(tiles[0])
        rows = [cv2.hconcat((tiles[i:i + cols] + [blank] * cols)[:cols]) for i in range(0, len(tiles), cols)]
        return cv2.vconcat(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "lanes_sheet.jpg"), sheet(lane_tiles, 6), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(str(OUT / "floor_sheet.jpg"), sheet(floor_tiles, 6), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"확인 그림: {OUT / 'lanes_sheet.jpg'}, {OUT / 'floor_sheet.jpg'} ({len(lane_tiles)}짝)")
    return 0


# ---------------------------------------------------------------------- 입력 검사

def inputs() -> int:
    import dual_eval

    dual_eval.set_profile(SHOOT)
    problems = dual_eval.input_problems(None)
    truth = dual_eval.load_truth()
    if problems:
        print("입력이 맞지 않음:")
        for p in problems:
            print(f"  {p}")
        return 1
    sets = {k: sum(v["set"] == k for v in truth.values()) for k in ("normal", "theft", "complex")}
    print(f"입력 확인: 두 카메라 영상 {len(truth)}짝 = 정답표 {len(truth)}짝 (정상 {sets['normal']} · 절도 {sets['theft']} · 복합 {sets['complex']})")
    return 0


# ---------------------------------------------------------------------- 결제 시계

def counter_presence(pair: str) -> list[tuple[float, float]]:
    """그 짝의 '계산대 구역에 사람이 있던' 구간들 (초, 전체 카메라 시각). 관찰 기록(신원·바닥 구역)만 쓴다."""
    import dual_eval
    from src.core.config import AppConfig
    from src.core.types import ZoneType
    from src.zones.floor import FloorMap, FloorZoneMap

    log = dual_eval.PairLog(pair)
    cfg = dual_eval.dual_config(AppConfig.load(ROOT / "config.yaml"), pair)
    fl = FloorMap.load(dual_eval.zones_dir(pair) / "floor.yaml")
    fl.foot_mode, fl.ankle_conf = cfg.zones.foot_point, cfg.zones.ankle_conf
    fz = FloorZoneMap(fl)
    meta = log.meta["angle1"]
    fz.resolve(meta["width"], meta["height"])
    stamps = []
    for idx in sorted(log.ids["angle1"]):
        at_counter = any(h.zone_type == ZoneType.CHECKOUT for o in log.ids["angle1"][idx] if o.person_id > 0
                         for h in fz.evaluate(o.bbox, o.keypoints))
        stamps.append((log.ms("angle1", idx) / 1000.0, at_counter))
    return merge_presence(stamps)


def merge_presence(stamps: list[tuple[float, bool]], gap: float = 0.5) -> list[tuple[float, float]]:
    """(시각, 있음) 표본 → 있던 구간. gap 초 안의 끊김은 잇는다."""
    spans: list[list[float]] = []
    for t, on in stamps:
        if not on:
            continue
        if spans and t - spans[-1][1] <= gap:
            spans[-1][1] = t
        else:
            spans.append([t, t])
    return [(a, b) for a, b in spans]


def clock_table(orders: list, spans: list, presence: dict[str, list[tuple[float, float]]],
                offsets: list[float]) -> list[tuple[float, int, int]]:
    """시계 차이 후보마다 (후보, 결제 시각에 계산대에 사람이 있던 주문 수, 영상에 배정된 주문 수).

    후보마다 주문을 영상에 **다시 배정**한다 (pos_payments.assign — 판정에 들어갈 때와 같은 규칙·같은 유효 구간).
    주문을 한 영상에 미리 고정해 두면, 이웃 영상과 가까운 주문이 엉뚱한 영상의 계산대 기록으로 세어진다.
    """
    import pos_payments as P

    table = []
    for off in offsets:
        rows = [r for r in P.assign(orders, spans, off) if r["pair"]]
        hits = sum(any(a <= float(r["time_sec"]) <= b for a, b in presence.get(r["pair"], [])) for r in rows)
        table.append((off, hits, len(rows)))
    return table


def pick_offset(table: list[tuple]) -> float:
    """0초가 최댓값과 1건 이내면 0. 아니면 최댓값인 후보들의 가운데 값 (후보가 짝수 개면 가운데 둘 중 0 에 가까운 쪽).
    table 의 줄 = (후보, 계산대에 사람이 있던 주문 수, …)."""
    best = max(row[1] for row in table)
    zero = next((row[1] for row in table if abs(row[0]) < 1e-9), None)
    if zero is not None and best - zero <= 1:
        return 0.0
    tops = sorted(row[0] for row in table if row[1] == best)
    mid = tops[(len(tops) - 1) // 2: len(tops) // 2 + 1]
    return float(min(mid, key=abs))


def pay_clock() -> int:
    import dual_eval
    import pos_payments as P

    dual_eval.set_profile(SHOOT)
    spans = P.video_spans()
    orders = P.read_orders(P.LOG)
    missing = [s.pair for s in spans if not (dual_eval.OUT / s.pair / ".done").exists()]
    if missing:
        print(f"실행 기록 없는 짝: {', '.join(missing)} — 먼저 dual_eval.py --shoot 4 run")
        return 1
    presence = {s.pair: counter_presence(s.pair) for s in spans}
    offsets = [round(float(v), 2) for v in np.arange(CLOCK_RANGE[0], CLOCK_RANGE[1] + 1e-9, CLOCK_RANGE[2])]
    table = clock_table(orders, spans, presence, offsets)
    print("시계 차이 후보별 '결제 시각에 계산대 구역에 사람이 있던 주문 수 / 영상에 배정된 주문 수' (차이 = 결제 서버 − 휴대폰):")
    for off, hits, n in table:
        print(f"  {off:+5.1f}초  {hits:3d} / {n:3d}  {'#' * hits}")
    chosen = pick_offset(table)
    print(f"고른 값: {chosen:+.1f}초" + (" (0초가 최댓값과 1건 이내 — 그대로)" if chosen == 0.0 else " — python scripts/pos_payments.py --clock-offset 로 다시 만든다"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pay_clock.txt").write_text("\n".join(f"{off:+.1f}\t{hits}\t{n}" for off, hits, n in table)
                                       + f"\nchosen\t{chosen:+.1f}\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["inputs", "floor", "check", "pay-clock"])
    args = ap.parse_args(argv)
    return {"inputs": inputs, "floor": floor, "check": check, "pay-clock": pay_clock}[args.stage]()


if __name__ == "__main__":
    raise SystemExit(main())
