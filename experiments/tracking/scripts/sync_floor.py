"""3차 촬영 두 각도 시차를 발 위치(공통 바닥 좌표)로 자동 추정한다.

두 카메라 모두 바닥이 보인다. 바닥에 붙인 노란 메모지 4장(area_setting.mp4)을 두 화면에서 찾아 같은 바닥 좌표로 묶고,
두 카메라의 사람 발 위치가 가장 잘 겹치는 시차를 찾는다 (계산: src/inputs/floor_sync.py, 정의: src/inputs/sync.py).

    python scripts/sync_floor.py calibrate          # 메모지 → zones/store01_shoot3/floor_angle{0,1}.yaml + 확인 그림
    python scripts/sync_floor.py extract            # 사람 발(화면 좌표) 뽑기, 캐시 (CPU 약 1시간)
    python scripts/sync_floor.py register           # 짝마다 카메라 위치를 area_setting 에 맞춤 (estimate 가 자동으로 부름)
    python scripts/sync_floor.py audio              # 소리 시차 (GCC-PHAT) 캐시 (estimate 가 자동으로 부름)
    python scripts/sync_floor.py estimate           # 시차 → labels/store01_shoot3/sync.csv (manual 줄은 그대로)
    python scripts/sync_floor.py check              # 짝마다 두 화면 나란히 그림 + 비용 곡선 + 카메라 이동 확인
    python scripts/sync_floor.py estimate --only T01,C04

발 위치만으로는 39쌍 중 8쌍만 믿을 만했다 (사람이 선반·계산대 앞에 오래 서 있어 비용 골이 넓다). 그래서 소리 시차를
두 번째 독립 신호로 더해, 둘이 맞을 때만 믿는다 (floor+audio). 맞지 않으면 floor-low 로 두고 note 에 두 후보를 적는다.
추정값은 sync_pair.py --shoot 3 <짝> 으로 눈으로 확인·수정한다 (저장하면 manual).

메모지 위치 (1920×1080 화면 픽셀, 사람이 찍은 대략 위치 → 둘레의 노란 덩어리 무게중심으로 다듬음):
  P1 문 쪽·가구 쪽, P2 문 쪽·방 쪽, P3 안쪽·가구 쪽, P4 안쪽·방 쪽.
area_setting 은 촬영 40분 전에 찍었고 그 뒤·중간에 카메라가 살짝 움직였다(바닥 자리 40~77px). 그래서 짝마다 첫 장면을
area_setting 에 배경 특징점으로 맞춰(호모그래피) 보정점을 그 영상 화면으로 옮긴다 (register).
바닥 cm 는 실측이 아니다 — 타일(60cm 가정) 간격으로 본 대략값. 두 카메라가 같은 4점·같은 좌표를 쓰므로 시차에는 영향이 거의 없다.
줄자로 잰 값이 생기면 --floor 로 바꾼다 (발 캐시는 화면 좌표라 다시 뽑을 필요 없음).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inputs import audio_sync  # noqa: E402
from src.inputs.floor_sync import FloorSyncResult, Samples, combine, lag_cost, median_distance, search  # noqa: E402
from src.inputs.sync import SyncRow, SyncTable  # noqa: E402
from src.zones.floor import FloorMap, foot_pixel  # noqa: E402

SOURCE = ROOT.parent.parent / "source"
CAMS = {"angle0": SOURCE / "2(angle0)", "angle1": SOURCE / "2(angle1)"}
ZONES = ROOT / "zones" / "store01_shoot3"
LABELS = ROOT / "labels" / "store01_shoot3"
OUT = ROOT / "outputs" / "sync_floor"
FEET = OUT / "feet"
FEET_FIELDS = ["ms", "person", "px", "py", "width", "height", "conf", "ankle", "cut"]

ROUGH = {
    "angle0": [(432, 820), (349, 1025), (1135, 785), (1213, 1037)],
    "angle1": [(1125, 604), (942, 604), (1192, 1070), (847, 1068)],
}
FLOOR_CM = [(0.0, 0.0), (64.0, 0.0), (0.0, 74.0), (64.0, 74.0)]
HZ = {"angle0": 10.0, "angle1": 20.0}
POSE_MODEL = "yolov8n-pose.pt"
POSE_CONF = 0.3
VIEW_MARGIN = 0.01
"""다른 카메라 화면 안으로 볼 여유 (화면 비율)."""


# ---------------------------------------------------------------------- 보정

def median_frame(path: Path, every: int = 15) -> np.ndarray:
    """지나가는 사람을 지우려고 여러 장면의 픽셀 중앙값."""
    cap = cv2.VideoCapture(str(path))
    frames, i = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % every == 0:
            ok, f = cap.retrieve()
            if ok:
                frames.append(f)
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"{path} 를 읽지 못함")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def find_note(b: np.ndarray, rough: tuple[float, float], radius: int = 90, rise: float = 12.0,
              min_area: int = 150) -> dict | None:
    """Lab b(노랑) 채널에서 대략 위치 둘레의 노란 덩어리. 대략 위치에 가장 가까운 덩어리를 고른다
    (가장 큰 것을 고르면 둘레의 나무 가구가 잡힌다). 없으면 None."""
    h, w = b.shape
    x, y = int(round(rough[0])), int(round(rough[1]))
    x0, y0, x1, y1 = max(0, x - radius), max(0, y - radius), min(w, x + radius), min(h, y + radius)
    win = b[y0:y1, x0:x1].astype(float)
    mask = (win > np.median(win) + rise).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    cands = [k for k in range(1, n) if stats[k, 4] >= min_area]
    if not cands:
        return None
    k = min(cands, key=lambda k: np.hypot(cents[k][0] + x0 - x, cents[k][1] + y0 - y))
    bx, by, bw, bh, area = stats[k]
    return {"cx": float(cents[k][0] + x0), "cy": float(cents[k][1] + y0), "left": float(bx + x0), "top": float(by + y0),
            "w": float(bw), "h": float(bh), "area": int(area), "cut": bool(by + y0 + bh >= h - 1)}


def note_centers(image: np.ndarray, rough: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], list[str]]:
    """메모지 중심 (픽셀)과 경고. 화면 아래 끝에 잘린 메모지는 위 모서리 + 반 높이(안 잘린 메모지의 세로/가로 비)로 중심을 잡는다."""
    b = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 2]
    found = [find_note(b, r) for r in rough]
    ratios = [f["h"] / f["w"] for f in found if f and not f["cut"] and f["w"] > 0]
    ratio = float(np.median(ratios)) if ratios else 1.0
    centers, warn = [], []
    for i, (f, r) in enumerate(zip(found, rough), 1):
        if f is None:
            centers.append((float(r[0]), float(r[1])))
            warn.append(f"P{i}: 노란 덩어리 못 찾음 — 대략 위치 {r} 그대로")
        elif f["cut"]:
            centers.append((f["cx"], f["top"] + f["w"] * ratio / 2.0))
            warn.append(f"P{i}: 화면 아래 끝에 잘림 — 위 모서리 + 반 높이로 중심 (세로/가로 {ratio:.2f})")
        else:
            centers.append((f["cx"], f["cy"]))
    return centers, warn


def calibrate(floor_cm: list[tuple[float, float]], points: dict[str, list[tuple[float, float]]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    maps, images = {}, {}
    for cam, folder in CAMS.items():
        img = median_frame(folder / "area_setting.mp4")
        h, w = img.shape[:2]
        centers, warn = note_centers(img, points[cam])
        for line in warn:
            print(f"  {cam} {line}")
        print(f"{cam}: 메모지 중심 " + ", ".join(f"P{i}({x:.1f},{y:.1f})" for i, (x, y) in enumerate(centers, 1)))
        fm = FloorMap(cam, [(x / w, y / h) for x, y in centers], floor_cm,
                      note=("3차 촬영 바닥 지도 (sync_floor.py calibrate). 보정점 = 바닥 노란 메모지 4장 (area_setting.mp4). "
                            "cm 는 실측이 아니라 타일 간격으로 본 대략값 — 두 카메라가 같은 좌표를 쓰므로 시차·교차 화면에는 영향이 작다. "
                            "실측이 생기면 sync_floor.py calibrate --floor 로 바꾼다."))
        fm.save(ZONES / f"floor_{cam}.yaml")
        maps[cam], images[cam] = fm, img
        cv2.imwrite(str(OUT / f"calib_{cam}.jpg"), draw_grid(img.copy(), fm, centers))
    tops = [topdown(images[c], maps[c]) for c in CAMS]
    cv2.imwrite(str(OUT / "calib_topdown.jpg"), np.hstack(tops))
    print(f"→ {ZONES}  확인 그림 {OUT}/calib_*.jpg (격자가 두 화면의 같은 바닥 자리에 와야 한다)")


def draw_grid(img: np.ndarray, fm: FloorMap, centers, step: float = 30.0, span=(-300, 400)) -> np.ndarray:
    h, w = img.shape[:2]

    def px(fx, fy):
        if not fm.image_side_ok(fx, fy):
            return None
        u, v = fm.to_image(fx, fy)
        return (int(u * w), int(v * h)) if -1 < u < 2 and -1 < v < 2 else None

    for a in np.arange(span[0], span[1] + 1, step):
        for line in (((a, span[0]), (a, span[1])), ((span[0], a), (span[1], a))):
            pts = [px(*(np.array(line[0]) + t * (np.array(line[1]) - np.array(line[0])))) for t in np.linspace(0, 1, 60)]
            for p, q in zip(pts, pts[1:]):
                if p and q:
                    color = (0, 0, 255) if a == 0 else (255, 160, 0)
                    cv2.line(img, p, q, color, 1, cv2.LINE_AA)
    for i, (x, y) in enumerate(centers, 1):
        cv2.drawMarker(img, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 28, 2)
        cv2.putText(img, f"P{i}", (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return img


def topdown(img: np.ndarray, fm: FloorMap, span=(-250, 350), scale: float = 1.5) -> np.ndarray:
    """바닥 cm 를 위에서 본 그림 (1cm = scale px)."""
    h, w = img.shape[:2]
    size = int((span[1] - span[0]) * scale)
    floor_px = np.float32([[(fx - span[0]) * scale, (fy - span[0]) * scale] for fx, fy in fm.floor_points])
    img_px = np.float32([[x * w, y * h] for x, y in fm.image_points])
    m = cv2.getPerspectiveTransform(img_px, floor_px)
    out = cv2.warpPerspective(img, m, (size, size))
    for g in np.arange(span[0], span[1] + 1, 60):
        p = int((g - span[0]) * scale)
        cv2.line(out, (p, 0), (p, size), (255, 160, 0), 1)
        cv2.line(out, (0, p), (size, p), (255, 160, 0), 1)
    cv2.putText(out, fm.camera, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out


# ---------------------------------------------------------------------- 발 뽑기

def pairs(only: str | None = None) -> list[str]:
    names = sorted(p.stem for p in CAMS["angle1"].glob("*.mp4")
                   if p.stem != "area_setting" and (CAMS["angle0"] / p.name).exists())
    if only:
        want = {s.strip() for s in only.split(",") if s.strip()}
        names = [n for n in names if n in want]
    return names


def feet_path(pair: str, cam: str) -> Path:
    return FEET / f"{pair}_{cam}.csv"


def extract_one(model, path: Path, hz: float) -> list[dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"{path} 열기 실패")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    every = max(1, int(round(fps / hz)))
    rows, idx = [], -1
    while True:
        if not cap.grab():
            break
        idx += 1
        if idx % every:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        h, w = frame.shape[:2]
        res = model.predict(frame, imgsz=640, conf=POSE_CONF, classes=[0], verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            rows.append({"ms": f"{idx * 1000.0 / fps:.1f}", "person": -1, "px": "", "py": "", "width": w, "height": h,
                         "conf": "", "ankle": "", "cut": ""})
            continue
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            (fx, fy), ankle = foot_pixel(tuple(box), kps[i] if kps is not None else None, 0.3)
            cut = (not ankle) and box[3] >= h - 2
            rows.append({"ms": f"{idx * 1000.0 / fps:.1f}", "person": i, "px": f"{fx:.1f}", "py": f"{fy:.1f}",
                         "width": w, "height": h, "conf": f"{conf:.2f}", "ankle": int(ankle), "cut": int(cut)})
    cap.release()
    return rows


def extract(only: str | None, force: bool) -> None:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / POSE_MODEL))
    FEET.mkdir(parents=True, exist_ok=True)
    todo = [(p, c) for p in pairs(only) for c in CAMS if force or not feet_path(p, c).exists()]
    print(f"발 뽑기 {len(todo)}건", flush=True)
    for k, (pair, cam) in enumerate(todo, 1):
        try:
            rows = extract_one(model, CAMS[cam] / f"{pair}.mp4", HZ[cam])
        except ValueError as e:
            print(f"  [{k}/{len(todo)}] {pair} {cam}: 건너뜀 ({e})", flush=True)
            continue
        tmp = feet_path(pair, cam).with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FEET_FIELDS)
            wr.writeheader()
            wr.writerows(rows)
        tmp.replace(feet_path(pair, cam))
        people = sum(1 for r in rows if int(r["person"]) >= 0)
        print(f"  [{k}/{len(todo)}] {pair} {cam}: 표본 {len({r['ms'] for r in rows})}, 사람 {people}", flush=True)


# ---------------------------------------------------------------------- 추정

REGISTER = OUT / "register.csv"
REG_FIELDS = ["pair", "camera", "h", "inliers", "matches", "drift_px"]
MIN_INLIERS = 100
DRIFT_WARN_PX = 5.0


def register_frames(ref: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray | None, int, int]:
    """영상 장면 픽셀 → area_setting 픽셀 호모그래피 (배경 SIFT + RANSAC). 카메라가 삼각대에서 살짝 돌아간 것을 되돌린다."""
    sift = cv2.SIFT_create(4000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0, 0
    knn = [x for x in cv2.BFMatcher().knnMatch(db, da, k=2) if len(x) == 2]
    good = [m for m, n in knn if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, 0, len(good)
    src = np.float32([kb[m.queryIdx].pt for m in good])
    dst = np.float32([ka[m.trainIdx].pt for m in good])
    h, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    return h, int(inl.sum()) if inl is not None else 0, len(good)


def first_last(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    cap = cv2.VideoCapture(str(path))
    first = last = None
    while cap.grab():
        ok, f = cap.retrieve()
        if ok:
            if first is None:
                first = f
            last = f
    cap.release()
    return first, last


def load_registration() -> dict[tuple[str, str], dict]:
    if not REGISTER.exists():
        return {}
    with REGISTER.open(encoding="utf-8", newline="") as f:
        return {(r["pair"], r["camera"]): r for r in csv.DictReader(f)}


def register(only: str | None, force: bool = False) -> dict[tuple[str, str], dict]:
    """짝마다 카메라 위치를 area_setting 에 맞춘다. area_setting 은 촬영 40분 전에 찍었고 그 사이·중간에 카메라가 움직였다
    (바닥 자리 40~77px). 첫 장면으로 맞추고, 마지막 장면과 메모지 자리가 DRIFT_WARN_PX 넘게 다르면 경고."""
    have = load_registration()
    refs: dict[str, np.ndarray] = {}
    for pair in pairs(only):
        for cam in CAMS:
            if (pair, cam) in have and not force:
                continue
            if cam not in refs:
                refs[cam] = median_frame(CAMS[cam] / "area_setting.mp4")
            first, last = first_last(CAMS[cam] / f"{pair}.mp4")
            if first is None:
                print(f"  {pair} {cam}: 영상 못 읽음")
                continue
            h, inl, n = register_frames(refs[cam], first)
            if h is None or inl < MIN_INLIERS:
                print(f"  {pair} {cam}: 맞추기 실패 (일치 {inl}/{n})")
                have.pop((pair, cam), None)
                continue
            drift = 0.0
            h2, inl2, _ = register_frames(refs[cam], last) if last is not None else (None, 0, 0)
            if h2 is not None and inl2 >= MIN_INLIERS:
                notes = np.float32(ROUGH[cam]).reshape(-1, 1, 2)
                a = cv2.perspectiveTransform(notes, np.linalg.inv(h)).reshape(-1, 2)
                b = cv2.perspectiveTransform(notes, np.linalg.inv(h2)).reshape(-1, 2)
                drift = float(np.max(np.hypot(*(a - b).T)))
            have[(pair, cam)] = {"pair": pair, "camera": cam, "h": " ".join(f"{v:.9g}" for v in h.ravel()),
                                 "inliers": inl, "matches": n, "drift_px": f"{drift:.1f}"}
            print(f"  {pair} {cam}: 일치 {inl}/{n}, 영상 안 이동 {drift:.1f}px"
                  + ("  ← 큼" if drift > DRIFT_WARN_PX else ""), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with REGISTER.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=REG_FIELDS)
        wr.writeheader()
        wr.writerows(have[k] for k in sorted(have))
    return have


def pair_map(base: FloorMap, reg: dict, width: int = 1920, height: int = 1080) -> FloorMap:
    """area_setting 기준 바닥 지도를 이 영상 화면으로 옮긴다: 보정점(area 화면) → 영상 화면 = H⁻¹."""
    h = np.array([float(v) for v in reg["h"].split()]).reshape(3, 3)
    pts = np.float32([[x * width, y * height] for x, y in base.image_points]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(pts, np.linalg.inv(h)).reshape(-1, 2)
    return FloorMap(base.camera, [(float(x) / width, float(y) / height) for x, y in moved], base.floor_points,
                    note=base.note)


def pair_maps(pair: str, base: dict[str, FloorMap], regs: dict) -> dict[str, FloorMap] | None:
    if not all((pair, c) in regs for c in CAMS):
        return None
    return {c: pair_map(base[c], regs[(pair, c)]) for c in CAMS}


def load_samples(pair: str, cam: str, maps: dict[str, FloorMap]) -> Samples:
    """발 캐시(화면 좌표) → 이 짝의 바닥 지도로 바닥 cm. 잘린 사람, 지평선 너머, 다른 카메라 화면 밖은 뺀다."""
    fm = maps[cam]
    other = maps["angle1" if cam == "angle0" else "angle0"]
    by_ms: dict[float, list[tuple[float, float]]] = {}
    with feet_path(pair, cam).open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            ms = float(r["ms"])
            pts = by_ms.setdefault(ms, [])
            if int(r["person"]) < 0 or r["cut"] == "1":
                continue
            x, y = float(r["px"]) / float(r["width"]), float(r["py"]) / float(r["height"])
            if not fm.on_floor_side(x, y):
                continue
            fx, fy = fm.to_floor(x, y)
            if not other.image_side_ok(fx, fy):
                continue
            u, v = other.to_image(fx, fy)
            if not (VIEW_MARGIN <= u <= 1 - VIEW_MARGIN and VIEW_MARGIN <= v <= 1 - VIEW_MARGIN):
                continue
            pts.append((fx, fy))
    return [(ms, np.asarray(by_ms[ms], dtype=float).reshape(-1, 2)) for ms in sorted(by_ms)]


def calib_fingerprint() -> str:
    h = hashlib.sha1()
    for cam in CAMS:
        h.update((ZONES / f"floor_{cam}.yaml").read_bytes())
    return h.hexdigest()[:8]


AUDIO = OUT / "audio.csv"
AUDIO_FIELDS = ["pair", "offset_ms", "peak"]
AUDIO_SR = 16000


def read_audio(path: Path) -> np.ndarray:
    """ffmpeg 로 16kHz 모노. 소리 줄이 없거나 못 읽으면 빈 배열."""
    import subprocess

    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
                              "-f", "s16le", "-"], capture_output=True).stdout
    except OSError as e:   # ffmpeg 없음·실행 불가 → 소리 없이 발 위치만
        print(f"  소리 못 읽음 ({path.name}): {e}")
        return np.zeros(0)
    return np.frombuffer(raw[: len(raw) // 2 * 2], np.int16).astype(np.float64)


def audio(only: str | None, force: bool = False) -> dict[str, tuple[float, float]]:
    """짝마다 소리 시차 (GCC-PHAT) 캐시."""
    have: dict[str, tuple[float, float]] = {}
    if AUDIO.exists():
        with AUDIO.open(encoding="utf-8", newline="") as f:
            have = {r["pair"]: (float(r["offset_ms"]), float(r["peak"])) for r in csv.DictReader(f)}
    for pair in pairs(only):
        if pair in have and not force:
            continue
        a, b = (read_audio(CAMS[c] / f"{pair}.mp4") for c in ("angle0", "angle1"))
        if len(a) == 0 or len(b) == 0:
            print(f"  {pair}: 소리 없음 — 발 위치만 (캐시에 남기지 않음)")
            continue
        off, peak = audio_sync.gcc_phat(a, b, AUDIO_SR)
        have[pair] = (off, peak)
        print(f"  {pair}: 소리 {off:+.0f}ms 피크 {peak:.2f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with AUDIO.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=AUDIO_FIELDS)
        wr.writeheader()
        wr.writerows({"pair": k, "offset_ms": f"{v[0]:.1f}", "peak": f"{v[1]:.3f}"} for k, v in sorted(have.items()))
    return have


def estimate(only: str | None) -> dict[str, FloorSyncResult]:
    base = {cam: FloorMap.load(ZONES / f"floor_{cam}.yaml") for cam in CAMS}
    regs = register(only)
    sounds = audio(only)
    fp = calib_fingerprint()
    table = SyncTable.load(LABELS / "sync.csv")
    found: dict[str, tuple] = {}
    for pair in pairs(only):
        if not all(feet_path(pair, c).exists() for c in CAMS):
            print(f"{pair}: 발 캐시 없음 — extract 먼저")
            continue
        maps = pair_maps(pair, base, regs)
        if maps is None:
            print(f"{pair}: 카메라 위치 맞추기 실패 — 건너뜀")
            continue
        a0, a1 = load_samples(pair, "angle0", maps), load_samples(pair, "angle1", maps)
        res = search(a0, a1)
        a_ms, a_peak = sounds.get(pair, (None, 0.0))
        a_ok = audio_sync.usable(a_ms, a_peak)
        c_a, n_a = lag_cost(a0, a1, a_ms) if a_ms is not None and np.isfinite(a_ms) else (float("inf"), 0)
        found[pair] = (res, a0, a1, a_ms, a_peak, a_ok, c_a, n_a)
    good = [f[0].offset_ms for f in found.values() if f[0].judge()[0]]
    fallback = float(np.median(good)) if good else 0.0
    trusted = 0
    for pair, (res, a0, a1, a_ms, a_peak, a_ok, c_a, n_a) in found.items():
        d = combine(res, a_ms, a_ok, c_a, n_a, fallback)
        trusted += d.ok
        dist = median_distance(a0, a1, d.offset_ms)
        audio_txt = f"소리 {a_ms:+.0f}ms 피크 {a_peak:.2f}" if a_ms is not None and np.isfinite(a_ms) else "소리 없음"
        floor_txt = (f"발 위치 {res.offset_ms:+.0f}ms" if res.offset_ms is not None else "발 위치 없음")
        note = (f"{floor_txt} 비용 {res.cost_cm:.0f}cm 뾰족함 {res.sharpness:.2f} 표본 {res.pairs}; {audio_txt}"
                f" (그 시차 비용 {c_a:.0f}cm); 발 거리 중앙 {dist:.0f}cm; 보정 {fp}")
        if not d.ok:
            note += " — 확인 필요: " + "; ".join(d.why)
        print(f"{'  ' if d.ok else '? '}{pair:6s} {d.offset_ms:+7.0f}ms {d.method:11s} {note}")
        row = table.rows.get(pair)
        if row and row.method == "manual":
            print(f"        (manual {row.offset_ms:+.0f}ms 그대로 둠)")
            continue
        table.upsert(SyncRow(pair, f"2(angle0)/{pair}.mp4", f"2(angle1)/{pair}.mp4", round(d.offset_ms, 1),
                             d.method, "", note))
    table.save(LABELS / "sync.csv")
    print(f"믿을 만함 {trusted}/{len(found)} → {LABELS / 'sync.csv'}")
    return {p: f[0] for p, f in found.items()}


# ---------------------------------------------------------------------- 확인

def frame_at(path: Path, ms_list: list[float], width: int = 640) -> list[np.ndarray | None]:
    """여러 시각의 장면 (순차 읽기 — HEVC 탐색 버그)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    want = sorted((f, i) for i, ms in enumerate(ms_list) if (f := int(round(ms * fps / 1000.0))) >= 0)
    out: list[np.ndarray | None] = [None] * len(ms_list)
    idx, k = -1, 0
    while k < len(want):
        if not cap.grab():
            break
        idx += 1
        while k < len(want) and want[k][0] == idx:
            ok, f = cap.retrieve()
            if ok:
                out[want[k][1]] = cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])))
            k += 1
    cap.release()
    return out


def check(only: str | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = {cam: FloorMap.load(ZONES / f"floor_{cam}.yaml") for cam in CAMS}
    regs = register(only)
    table = SyncTable.load(LABELS / "sync.csv")
    (OUT / "check").mkdir(parents=True, exist_ok=True)
    (OUT / "cost").mkdir(parents=True, exist_ok=True)
    for pair in pairs(only):
        row = table.rows.get(pair)
        maps = pair_maps(pair, base, regs)
        if row is None or maps is None or not all(feet_path(pair, c).exists() for c in CAMS):
            continue
        a0, a1 = load_samples(pair, "angle0", maps), load_samples(pair, "angle1", maps)
        res = search(a0, a1)
        fig, ax = plt.subplots(figsize=(6, 2.6))
        xs = [l for l, c, n in res.curve if np.isfinite(c)]
        ys = [c for l, c, n in res.curve if np.isfinite(c)]
        ax.plot(xs, ys, lw=1)
        ax.axvline(row.offset_ms, color="r", lw=1)
        ax.set_xlabel("offset ms (angle0 + offset = angle1)")
        ax.set_ylabel("cost cm")
        ax.set_title(f"{pair} {row.method} {row.offset_ms:+.0f}ms")
        fig.tight_layout()
        fig.savefig(OUT / "cost" / f"{pair}.png", dpi=90)
        plt.close(fig)
        # 사람이 가장 많이 움직이는 세 시각 (angle1 기준)
        busy = [t for t, p in a1 if len(p)]
        if not busy:
            continue
        picks = [busy[int(len(busy) * q)] for q in (0.2, 0.5, 0.8)]
        f1 = frame_at(CAMS["angle1"] / f"{pair}.mp4", picks)
        f0 = frame_at(CAMS["angle0"] / f"{pair}.mp4", [t - row.offset_ms for t in picks])
        rows_img = []
        for t, a, b in zip(picks, f0, f1):
            if a is None or b is None:
                continue
            a, b = a.copy(), b.copy()
            cv2.putText(a, f"angle0 {(t - row.offset_ms) / 1000:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(b, f"angle1 {t / 1000:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            rows_img.append(np.hstack([a, b]))
        if rows_img:
            cv2.imwrite(str(OUT / "check" / f"{pair}.jpg"), np.vstack(rows_img))
        print(f"{pair:6s} {row.method:10s} {row.offset_ms:+7.0f}ms  → check/{pair}.jpg, cost/{pair}.png")
    print(f"→ {OUT}/check, {OUT}/cost")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["calibrate", "extract", "register", "audio", "estimate", "check"])
    ap.add_argument("--only", help="짝 이름 쉼표 구분 (예: T01,C04)")
    ap.add_argument("--force", action="store_true", help="extract·register: 캐시가 있어도 다시")
    ap.add_argument("--floor", help="calibrate: 메모지 P1~P4 바닥 cm 'x,y;x,y;x,y;x,y' (실측 시)")
    ap.add_argument("--points0", help="calibrate: angle0 메모지 대략 위치 픽셀 'x,y;...' (P1~P4 순)")
    ap.add_argument("--points1", help="calibrate: angle1 메모지 대략 위치 픽셀")
    args = ap.parse_args()

    def parse(spec):
        return [tuple(float(v) for v in p.split(",")) for p in spec.split(";") if p.strip()]

    if args.stage == "calibrate":
        floor_cm = parse(args.floor) if args.floor else FLOOR_CM
        points = {"angle0": parse(args.points0) if args.points0 else ROUGH["angle0"],
                  "angle1": parse(args.points1) if args.points1 else ROUGH["angle1"]}
        if len(floor_cm) != 4 or any(len(v) != 4 for v in points.values()):
            print("점은 4개씩 (P1~P4)")
            return 2
        calibrate(floor_cm, points)
    elif args.stage == "extract":
        extract(args.only, args.force)
    elif args.stage == "register":
        register(args.only, args.force)
    elif args.stage == "audio":
        audio(args.only, args.force)
    elif args.stage == "estimate":
        estimate(args.only)
    else:
        check(args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
