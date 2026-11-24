"""보관함 영상에 'AI 탐지 화면'(사람 상자 · 번호)을 그린다 — 이미 있는 추적 기록을 읽어서 (모델을 다시 돌리지 않음).

추적 기록 = experiments/tracking 의 dual_eval 결과 폴더 (회차마다 추적 도구의 기본 기록:
2차 outputs/dual_eval_pose3, 3차 outputs/dual_eval_shoot3_prod, 4차 outputs/dual_eval_shoot4). 장면 폴더마다
meta.json · times_angle{0,1}.json(3차부터 — 없으면 번호 ÷ fps) · angle{0,1}/identities.jsonl ·
angle{0,1}/identity_merges.jsonl · overview_swaps.jsonl · cross_view_links.jsonl 를 읽기만 한다.

번호는 추적 도구 보기(scripts/review_export.py)와 같은 규칙으로 맞춘다:
- 화면 신원은 끝까지 합친 번호 (identity_merges: from → into, 이어서).
- 전체 카메라는 번호 맞바뀜을 되돌린 기록(overview_swaps)이 있으면 바뀐 때(seen_ms, 없으면 ms)부터 매장 신원으로 바꿔 읽는다.
- 선반 카메라는 짝 기록(cross_view_links — 전체 카메라 번호는 이미 매장 신원)으로 전체 카메라 번호를 붙인다.
  짝의 선반 번호도 끝까지 합친 번호로 본다.

    python tools/ai_render.py --records <outputs/dual_eval_pose3> \\
        --video "cam1=<보관함용 영상>/1(angle1)" --video "cam2=<보관함용 영상>/1(angle0)" \\
        --sync <labels/store01_dual/sync.csv> --out <출력 폴더> [--only C01] [--workers 3]

출력: <출력>/cam1/<같은 파일 이름>.mp4 … → 그대로 보관함 가져오기 (import --ai).
영상은 보관함용 변환본(720p · 30fps)이어야 한다 — 그 위에 그리므로 시간축이 원본과 같다.

필요: opencv-python-headless, numpy, pillow (한글 이름표). ffmpeg.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from hawkeye_server.library import load_sync, scene_key, scene_offsets  # noqa: E402  (표준 라이브러리만 씀)

CAMERA_ANGLE = {"cam1": "angle1", "cam2": "angle0"}
"""앱 카메라 → 추적 기록의 카메라. cam1 = 전체(높은 구석), cam2 = 선반."""
OVERVIEW = "cam1"

# 사람마다 같은 색 (앱 상태색 — 빨강·주황·노랑·초록 — 과 겹치지 않는 차분한 색). RGB.
PALETTE = [(11, 158, 146), (49, 130, 246), (139, 92, 246), (229, 84, 154), (6, 182, 212), (99, 102, 241),
           (101, 163, 13), (100, 116, 139)]
UNLINKED = (148, 163, 184)
FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgunbd.ttf", "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def color_for(person: int | None) -> tuple[int, int, int]:
    return UNLINKED if person is None else PALETTE[(person - 1) % len(PALETTE)]


# --- 추적 기록 읽기 -------------------------------------------------------------------

Box = tuple[float, float, float, float]


@dataclass
class Track:
    """한 카메라의 사람별 위치 (그 카메라 재생 시각 ms → 원본 픽셀 상자)."""

    period_ms: float
    """처리한 장면 사이 간격 (stride / fps)."""
    people: dict[int, tuple[list[float], list[Box]]] = field(default_factory=dict)

    def boxes_at(self, t_ms: float) -> list[tuple[int, Box]]:
        """그 시각의 사람 상자. 앞뒤 기록이 가까우면(처리 간격 3배 이내) 선형 보간, 아니면 기록 뒤 1.5 간격까지만 유지."""
        out = []
        near, hold = 3.0 * self.period_ms, 1.5 * self.period_ms
        for pid, (times, boxes) in self.people.items():
            i = bisect.bisect_right(times, t_ms) - 1
            if i < 0:
                continue
            t0, b0 = times[i], boxes[i]
            if t_ms == t0:
                out.append((pid, b0))
                continue
            if i + 1 < len(times) and times[i + 1] - t0 <= near:
                t1, b1 = times[i + 1], boxes[i + 1]
                a = (t_ms - t0) / (t1 - t0)
                out.append((pid, tuple(p + (q - p) * a for p, q in zip(b0, b1))))  # type: ignore[arg-type]
            elif t_ms - t0 <= hold:
                out.append((pid, b0))
        return sorted(out)


def track_key(person_id: int, track_id: Any) -> int | None:
    """사람 묶음 열쇠. 신원이 정해진 사람(>0)은 그 번호, 아직 안 정해진 관측(-1 등)은 추적 번호마다 따로(음수 열쇠) —
    여러 미배정 사람을 한 사람으로 이어 그리지 않게. 추적 번호도 없으면 None (그리지 않음)."""
    if person_id > 0:
        return person_id
    if track_id is None:
        return None
    return -1 - int(track_id)


def load_track(identities: Path, times_ms: list[float], period_ms: float) -> Track:
    """identities.jsonl (처리 장면마다 사람 하나씩) → Track. 같은 장면 같은 열쇠가 둘이면 점수가 높은 것.
    열쇠는 track_key (미배정 관측은 음수 — 이름표 없이 상자만)."""
    best: dict[tuple[int, int], tuple[float, Box]] = {}
    with open(identities, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            frame = int(r["frame"])
            pid = track_key(int(r["person_id"]), r.get("track_id"))
            if pid is None or not 0 <= frame < len(times_ms):
                continue
            score = float(r.get("score") or 0.0)
            if (pid, frame) not in best or score > best[(pid, frame)][0]:
                best[(pid, frame)] = (score, tuple(float(v) for v in r["bbox"][:4]))  # type: ignore[assignment]
    track = Track(period_ms)
    for (pid, frame), (_, box) in sorted(best.items(), key=lambda kv: (kv[0][0], times_ms[kv[0][1]])):
        times, boxes = track.people.setdefault(pid, ([], []))
        times.append(times_ms[frame])
        boxes.append(box)
    return track


def load_times(record: Path, angle: str, meta: dict[str, Any]) -> list[float]:
    """장면 번호 → 그 카메라 재생 시각 ms. 3차부터는 실제 시각 기록(times_*.json — 장면을 빠뜨린 영상), 없으면(2차) 번호 ÷ fps."""
    path = record / f"times_{angle}.json"
    if path.is_file():
        times = json.loads(path.read_text(encoding="utf-8"))
        if times:
            return [float(t) for t in times]
    fps, frames = float(meta.get("fps") or 0), int(meta.get("frames") or 0)
    if fps <= 0 or frames <= 0:
        raise RuntimeError("장면 시각을 알 수 없습니다 (times 도 fps·frames 도 없음)")
    return [i * 1000.0 / fps for i in range(frames)]


def load_merges(path: Path) -> Callable[[int], int]:
    """화면 신원 합침(identity_merges.jsonl) → 끝까지 합친 번호를 돌려주는 함수 (없으면 그대로)."""
    into: dict[int, int] = {}
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    m = json.loads(line)
                    into[int(m["from"])] = int(m["into"])

    def resolve(pid: int) -> int:
        seen = set()
        while pid in into and pid not in seen:
            seen.add(pid)
            pid = into[pid]
        return pid
    return resolve


@dataclass
class Swaps:
    """전체 카메라 번호 맞바뀜을 되돌린 기록 (overview_swaps.jsonl). 바뀐 때부터 화면 번호 → 매장 신원으로 바꿔 읽는다."""

    rows: list[tuple[float, tuple[int, int], tuple[int, int]]]
    """(바뀐 때 ms — seen_ms, 없으면 ms, (화면 a, 화면 b), (매장 a, 매장 b))"""

    def store_id(self, screen_id: int, t_ms: float) -> int:
        store: dict[int, int] = {}
        for at, (sa, sb), (a, b) in sorted(self.rows, key=lambda r: r[0]):
            if at <= t_ms + 1e-6:
                store[sa], store[sb] = b, a
        return store.get(screen_id, screen_id)


def load_swaps(path: Path) -> Swaps:
    rows = []
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    rows.append((float(s.get("seen_ms", s["ms"])), (int(s["screen"][0]), int(s["screen"][1])),
                                 (int(s["store"][0]), int(s["store"][1]))))
    return Swaps(rows)


@dataclass
class Links:
    """선반 카메라 사람(detail_id) ↔ 전체 카메라 사람(overview_id — 매장 신원). 구간은 공통 시각(= 선반 재생 시각 + 시차) ms."""

    rows: list[tuple[float, float, int, int, float]]
    """(start_ms, end_ms, detail_id, overview_id, score)"""
    resolve_detail: Callable[[int], int] = field(default=lambda pid: pid)
    """짝의 선반 번호도 끝까지 합친 번호로 비교한다."""

    def overview_for(self, detail_id: int, common_ms: float, slack_ms: float = 400.0) -> int | None:
        """그 시각을 덮는 링크 중 점수가 가장 높은 것. 없으면 slack 안의 가장 가까운 것. 그것도 없으면 None.
        detail_id 는 끝까지 합친 선반 번호."""
        best: tuple[float, float, int] | None = None   # (거리, -점수, overview)
        for s, e, d, o, score in self.rows:
            if self.resolve_detail(d) != detail_id:
                continue
            gap = 0.0 if s <= common_ms <= e else min(abs(common_ms - s), abs(common_ms - e))
            if gap > slack_ms:
                continue
            cand = (gap, -score, o)
            if best is None or cand < best:
                best = cand
        return best[2] if best else None


def load_links(path: Path, resolve_detail: Callable[[int], int] = lambda pid: pid) -> Links:
    rows = []
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows.append((float(r["start_ms"]), float(r["end_ms"]), int(r["detail_id"]), int(r["overview_id"]),
                                 float(r.get("score") or 0.0)))
    return Links(rows, resolve_detail)


def person_number(camera: str, pid: int, t_ms: float, offset_ms: float, resolve: Callable[[int], int],
                  swaps: Swaps, links: Links | None) -> int | None:
    """화면 사람 번호 → 이름표 번호 (위 규칙). 미배정 관측(음수 열쇠)이나 선반 카메라에서 짝이 없으면 None (상자만)."""
    if pid <= 0:
        return None
    pid = resolve(pid)
    if camera == OVERVIEW:
        return swaps.store_id(pid, t_ms)
    return links.overview_for(pid, t_ms + offset_ms) if links is not None else None


def probe_pts(video: Path, ffprobe: str = "ffprobe") -> list[float]:
    """영상 장면들의 실제 표시 시각(초, 첫 장면 0 기준, 오름차순)."""
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", str(video)], capture_output=True, text=True, timeout=120)
    pts = sorted(float(x) for x in out.stdout.split() if x not in ("", "N/A"))
    if not pts:
        raise RuntimeError("영상 장면 시각을 읽지 못했습니다")
    return [t - pts[0] for t in pts]


def nominal_fps(pts: list[float], fallback: float) -> float:
    """장면 간격의 중앙값으로 본 fps (빠진 장면이 있어도 원래 fps). 장면이 너무 적으면 fallback."""
    gaps = sorted(b - a for a, b in zip(pts, pts[1:]) if b > a)
    if len(gaps) < 2:
        return fallback
    return round(1.0 / gaps[len(gaps) // 2], 3)


def grid_sources(pts: list[float], fps: float) -> list[int]:
    """고른 간격(fps) 출력 장면마다 보여 줄 원본 장면 번호 — 그 시각에 화면에 떠 있는 장면 (반 장면 반올림).
    장면 간격이 고르지 않아도(빠진 장면) 출력의 k 번째 장면 = 시각 k/fps 라 원본과 같은 순간을 가리킨다."""
    total = int(round(pts[-1] * fps)) + 1
    out, i = [], 0
    for k in range(total):
        t = k / fps + 0.5 / fps
        while i + 1 < len(pts) and pts[i + 1] <= t:
            i += 1
        out.append(i)
    return out


def scale_box(box: Box, sx: float, sy: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)


# --- 장면 짝 맞추기 --------------------------------------------------------------------

@dataclass
class Job:
    scene: str
    camera: str
    video: Path
    record: Path
    out: Path
    offset_ms: float = 0.0
    """선반 카메라의 시차 (공통 시각 = 선반 재생 시각 + offset_ms). 전체 카메라는 0."""


def plan_jobs(records: Path, videos: dict[str, Path], out: Path, sync: list[tuple[str, str, float]] | None,
              only: set[str] | None = None) -> tuple[list[Job], list[str]]:
    """(그릴 일, 건너뛴 까닭들). 장면 이름은 보관함과 같은 scene_key 로 맞춘다."""
    record_of = {scene_key(d.name): d for d in records.iterdir() if d.is_dir() and (d / "meta.json").is_file()}
    files: dict[str, dict[str, Path]] = {}
    for cam, folder in videos.items():
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() == ".mp4" and ".part." not in p.name:
                files.setdefault(scene_key(p.stem), {})[cam] = p
    jobs, skipped = [], []
    for scene in sorted(files, key=lambda s: [int(t) if t.isdigit() else t for t in __import__("re").split(r"(\d+)", s)]):
        if only and scene not in only:
            continue
        rec = record_of.get(scene)
        if rec is None:
            skipped.append(f"{scene}: 추적 기록 없음")
            continue
        offsets = scene_offsets(files[scene], sync) if (sync and len(files[scene]) > 1) else None
        for cam, video in sorted(files[scene].items()):
            angle = CAMERA_ANGLE.get(cam)
            if angle is None or not (rec / angle / "identities.jsonl").is_file():
                skipped.append(f"{scene} {cam}: 이 카메라 기록 없음")
                continue
            jobs.append(Job(scene, cam, video, rec, out / cam / video.name,
                            (offsets or {}).get(cam, 0.0) if cam != OVERVIEW else 0.0))
    return jobs, skipped


# --- 그리기 -----------------------------------------------------------------------------

class Labels:
    """'손님 N' 이름표 그림을 한 번만 만들어 둔다 (한글 글꼴이 없으면 '#N')."""

    def __init__(self, height: int) -> None:
        from PIL import ImageFont

        self.size = max(12, round(height * 0.024))
        self.font = None
        for path in FONT_CANDIDATES:
            if os.path.isfile(path):
                try:
                    self.font = ImageFont.truetype(path, self.size)
                    break
                except OSError:
                    continue
        self.cache: dict[int, Any] = {}

    def get(self, person: int) -> Any:
        """BGRA numpy 그림."""
        if person in self.cache:
            return self.cache[person]
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        text = f"손님 {person}" if self.font else f"#{person}"
        font = self.font or ImageFont.load_default()
        probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        l, t, r, b = probe.textbbox((0, 0), text, font=font)
        pad_x, pad_y = round(self.size * 0.45), round(self.size * 0.25)
        w, h = (r - l) + 2 * pad_x, (b - t) + 2 * pad_y
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=(*color_for(person), 235))
        d.text((pad_x - l, pad_y - t), text, font=font, fill=(255, 255, 255, 255))
        arr = np.array(img)[:, :, [2, 1, 0, 3]].copy()      # RGBA → BGRA
        self.cache[person] = arr
        return arr


def blit(frame: Any, img: Any, x: int, y: int) -> None:
    """BGRA 그림을 프레임(BGR)에 얹는다 (화면 밖은 잘라 냄)."""
    h, w = img.shape[:2]
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, fw), min(y + h, fh)
    if x0 >= x1 or y0 >= y1:
        return
    part = img[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = part[:, :, 3:4].astype("float32") / 255.0
    roi = frame[y0:y1, x0:x1]
    roi[:] = (part[:, :, :3] * alpha + roi * (1 - alpha)).astype("uint8")


def render(job: Job, ffmpeg: str = "ffmpeg") -> str:
    """한 장면 · 한 카메라를 그린다. 결과 한 줄을 돌려준다 (실패하면 RuntimeError)."""
    import cv2

    angle = CAMERA_ANGLE[job.camera]
    meta = json.loads((job.record / "meta.json").read_text(encoding="utf-8"))[angle]
    times = load_times(job.record, angle, meta)
    if len(times) < 2:
        raise RuntimeError("장면 시각 기록이 비었습니다")
    period = float(meta.get("stride") or 1) * 1000.0 / float(meta.get("fps") or 30.0)
    track = load_track(job.record / angle / "identities.jsonl", times, period)
    resolve = load_merges(job.record / angle / "identity_merges.jsonl")
    swaps = load_swaps(job.record / "overview_swaps.jsonl")
    links = load_links(job.record / "cross_view_links.jsonl", resolve) if job.camera != OVERVIEW else None

    cap = cv2.VideoCapture(str(job.video))
    if not cap.isOpened():
        raise RuntimeError("영상을 열지 못했습니다")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pts = probe_pts(job.video)
    fps = nominal_fps(pts, fps)                  # 가변 간격 영상은 평균 fps 가 낮게 나온다 — 장면 간격의 중앙값으로
    # 시간축 확인은 재생 길이로 한다 (60→30fps 변환본은 장면 수가 원본 기록과 다르다)
    if abs(pts[-1] * 1000.0 - times[-1]) > 1500:
        cap.release()
        raise RuntimeError(f"영상 길이({pts[-1]:.1f}s)와 기록 길이({times[-1] / 1000:.1f}s)가 다릅니다")
    sources = grid_sources(pts, fps)
    uneven = sum(1 for a, b in zip(sources, sources[1:]) if b - a != 1)
    sx, sy = w / float(meta.get("width") or w), h / float(meta.get("height") or h)
    labels = Labels(h)
    thick = max(2, round(h / 360))

    job.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = job.out.with_name(job.out.stem + ".part.mp4")
    enc = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-profile:v", "main",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = drawn = 0
    read, src = -1, None
    try:
        for k, want in enumerate(sources):
            while read < want:                       # 그 시각의 원본 장면까지 읽는다 (빠진 장면 자리는 앞 장면 반복)
                ok, img = cap.read()
                if not ok:
                    break
                read, src = read + 1, img
            if src is None:
                break
            frame = src.copy()
            t = k * 1000.0 / fps                     # 출력 k 번째 = 시각 k/fps (원본과 같은 순간)
            for pid, box in track.boxes_at(t):
                person = person_number(job.camera, pid, t, job.offset_ms, resolve, swaps, links)
                x1, y1, x2, y2 = scale_box(box, sx, sy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_for(person)[::-1], thick, lineType=cv2.LINE_AA)
                if person is not None:
                    tag = labels.get(person)
                    ty = y1 - tag.shape[0] - thick if y1 - tag.shape[0] - thick >= 0 else y1 + thick
                    blit(frame, tag, x1, ty)
                drawn += 1
            enc.stdin.write(frame.tobytes())  # type: ignore[union-attr]
            frames += 1
        enc.stdin.close()  # type: ignore[union-attr]
        err = enc.stderr.read().decode("utf-8", "replace")  # type: ignore[union-attr]
        if enc.wait() != 0 or not tmp.is_file():
            raise RuntimeError(f"ffmpeg 실패: {err.strip()[-300:]}")
        os.replace(tmp, job.out)
    except BaseException:
        enc.kill()
        tmp.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
    return f"{job.scene} {job.camera}: {frames}장면, 상자 {drawn}" + (f", 고르지 않은 간격 {uneven}곳 맞춤" if uneven else "")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="보관함 영상에 AI 탐지 화면(사람 상자·번호)을 그린다")
    p.add_argument("--records", required=True, help="추적 기록 폴더 (예: experiments/tracking/outputs/dual_eval_pose3)")
    p.add_argument("--video", action="append", required=True, metavar="CAM=DIR", help="보관함용 영상 폴더 (cam1=…, cam2=…)")
    p.add_argument("--sync", help="카메라 시차 표 — 선반 카메라 번호를 전체 카메라 번호로 맞출 때")
    p.add_argument("--out", required=True)
    p.add_argument("--only", action="append", help="이 장면만 (예: C01). 여러 번")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 3))
    p.add_argument("--force", action="store_true", help="이미 그린 것도 다시")
    a = p.parse_args(argv)
    videos = {}
    for spec in a.video:
        cam, sep, folder = spec.partition("=")
        if not sep or cam not in CAMERA_ANGLE or not Path(folder).is_dir():
            p.error(f"--video 는 cam1=<폴더> 또는 cam2=<폴더>: {spec}")
        videos[cam] = Path(folder)
    sync = load_sync(Path(a.sync)) if a.sync else None
    jobs, skipped = plan_jobs(Path(a.records), videos, Path(a.out), sync, {scene_key(s) for s in a.only} if a.only else None)
    for s in skipped:
        print(f"  건너뜀 — {s}")
    todo = [j for j in jobs if a.force or not j.out.is_file()]
    print(f"[ai_render] 그릴 것 {len(todo)} (이미 있음 {len(jobs) - len(todo)}, 건너뜀 {len(skipped)})")
    failed = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(render, j): j for j in todo}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                print("  " + fut.result(), flush=True)
            except Exception as e:
                failed += 1
                print(f"  실패 — {j.scene} {j.camera}: {e}", flush=True)
    print(f"[ai_render] 끝: 그림 {len(todo) - failed}, 실패 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
