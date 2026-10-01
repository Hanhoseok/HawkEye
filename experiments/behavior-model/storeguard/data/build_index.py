"""영상 ↔ 라벨 인덱스 생성 + 촬영 사건(take) 그룹핑.

산출물
  data/index/clips.jsonl   영상 1개 = 1줄. 파일명 메타 + XML 이벤트 + (선택) 영상 실측 메타
  data/index/takes.json    take_id -> 소속 영상 stem 목록

take(촬영 사건) 정의
  같은 클래스·매장·날짜·시나리오·배우에서, 파일명 시각이 --take-gap 초 이내로 붙어 있는
  영상들을 하나의 take 로 본다. 여러 카메라가 같은 연기를 찍은 것이므로 절대 분할을 가로질러서는 안 된다.

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.build_index --probe-video
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from storeguard import naming
from storeguard.config import Config
from storeguard.data import parse_cvat

RAW = Path(r"D:\computervision\data\raw")
INDEX = Path(r"D:\computervision\data\index")


def probe_video(path: Path) -> dict:
    """cv2 로 실제 프레임수/fps/해상도 확인. 실패해도 예외를 올리지 않는다."""
    import cv2

    info: dict = {"probe_ok": False}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        info["probe_error"] = "open 실패"
        return info
    try:
        info.update(
            video_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            video_fps=round(float(cap.get(cv2.CAP_PROP_FPS)), 4),
            video_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            video_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        ok, _ = cap.read()
        info["first_frame_ok"] = bool(ok)
        info["probe_ok"] = bool(ok)
    except Exception as exc:  # pragma: no cover
        info["probe_error"] = str(exc)
    finally:
        cap.release()
    return info


def scan_cache(cache_dir: Path) -> dict[str, int]:
    """캐시에 있는 stem -> 프레임 수.

    ingest 로 만든 영상은 mp4 를 디스크에 남기지 않는다(처리 후 삭제).
    그래도 캐시가 정상이면 학습에 쓸 수 있으므로 여기서 따로 찾는다.
    """
    from storeguard.data.framestore import available_stems, open_store

    out: dict[str, int] = {}
    if not cache_dir.is_dir():
        return out
    for stem in sorted(available_stems(cache_dir)):
        try:
            out[stem] = open_store(cache_dir, stem).n_frames
        except Exception:
            pass
    return out


def collect(split: str, probe: bool, raw: Path = RAW,
            extra_roots: list[Path] | None = None) -> list[dict]:
    """한 분할의 영상·라벨을 모은다.

    영상은 용량이 커서 다른 드라이브에 둘 수 있다(extra_roots).
    라벨은 작으므로 raw 아래에만 둔다.
    """
    label_dir = raw / split / "labels"
    labels = {p.stem: p for p in label_dir.glob("*.xml")} if label_dir.is_dir() else {}
    videos: dict[str, Path] = {}
    for root in [raw, *(extra_roots or [])]:
        video_dir = root / split / "videos"
        if video_dir.is_dir():
            for p in video_dir.glob("*.mp4"):
                videos.setdefault(p.stem, p)

    rows: list[dict] = []
    for stem in sorted(set(labels) | set(videos)):
        row: dict = {"stem": stem, "split_src": split,
                     "has_label": stem in labels, "has_video": stem in videos}
        problems: list[str] = []
        try:
            row.update(naming.parse_stem(stem).to_dict())
        except naming.NameParseError as exc:
            problems.append(f"파일명 파싱 실패: {exc}")
            row["class_name"] = None

        if stem in labels:
            row["label_path"] = str(labels[stem])
            try:
                ann = parse_cvat.parse(labels[stem])
                row.update(
                    label_frames=ann.n_frames,
                    label_width=ann.width,
                    label_height=ann.height,
                    keypoint_frame_count=len(ann.keypoint_frames),
                    keypoint_frames=ann.keypoint_frames,
                    person_ids=sorted(ann.person_ids),
                    genders=sorted(ann.genders),
                    age_groups=sorted(ann.age_groups),
                    object_counts={k: len(v) for k, v in ann.object_boxes.items()},
                    events=[
                        {"action": e.action, "start_frame": e.start_frame,
                         "end_frame": e.end_frame, "n_frames": e.n_frames}
                        for e in ann.events
                    ],
                )
                problems.extend(ann.problems)
                actions = {e["action"] for e in row["events"]}
                if row.get("class_name") and actions and actions != {row["class_name"]}:
                    problems.append(
                        f"파일명 클래스({row['class_name']})와 XML 이벤트({sorted(actions)}) 불일치")
            except Exception as exc:
                problems.append(f"XML 파싱 실패: {exc}")
        else:
            problems.append("라벨 XML 없음")

        if stem in videos:
            vp = videos[stem]
            row["video_path"] = str(vp)
            row["video_bytes"] = vp.stat().st_size
            if probe:
                row.update(probe_video(vp))
                if row.get("probe_ok") and row.get("label_frames"):
                    if abs(row["video_frames"] - row["label_frames"]) > 1:
                        problems.append(
                            f"프레임 수 불일치: 영상 {row['video_frames']} vs 라벨 {row['label_frames']}")
                if row.get("probe_ok") is False:
                    problems.append(f"영상 디코딩 실패: {row.get('probe_error', '첫 프레임 읽기 실패')}")
        else:
            problems.append("영상 MP4 없음")

        row["problems"] = problems
        rows.append(row)
    return rows


def assign_takes(rows: list[dict], gap_sec: int) -> dict[str, list[str]]:
    """시간 근접 클러스터링으로 take_id 부여. rows 에 'take_id' 를 채워 넣는다."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("class_code"):
            r["take_id"] = f"UNPARSED::{r['stem']}"
            continue
        key = (r["class_code"], r["store"], r["date"], r["scenario"], r["actor"])
        buckets[key].append(r)

    takes: dict[str, list[str]] = defaultdict(list)
    for key, group in buckets.items():
        group.sort(key=lambda r: (r.get("seconds", -1), r["stem"]))
        cluster_start = None
        prev = None
        idx = 0
        for r in group:
            sec = r.get("seconds", -1)
            if prev is None or sec < 0 or prev < 0 or (sec - prev) > gap_sec:
                idx += 1
                cluster_start = sec
            prev = sec
            take_id = "{}_{}_{}_{}_{}#{}".format(*key, idx)
            r["take_id"] = take_id
            r["take_anchor_sec"] = cluster_start
            takes[take_id].append(r["stem"])
    return dict(takes)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--video-root", type=Path, action="append", default=[],
                    help="영상이 있는 추가 위치(여러 번 지정 가능). "
                         "용량이 커서 다른 드라이브에 둔 경우. 예: F:\\storeguard_video")
    ap.add_argument("--cache", type=Path, default=None,
                    help="프레임 캐시 위치(기본: paths.cache). "
                         "mp4 가 없어도 캐시가 있으면 사용 가능으로 표시한다")
    ap.add_argument("--out", type=Path, default=INDEX)
    ap.add_argument("--probe-video", action="store_true",
                    help="cv2 로 모든 영상을 열어 fps/해상도/첫 프레임을 실측(느림)")
    ap.add_argument("--take-gap", type=int, default=90,
                    help="같은 take 로 묶을 파일명 시각 차이(초)")
    a = ap.parse_args(argv)

    extra = list(a.video_root)
    if extra:
        print(f"추가 영상 위치: {[str(p) for p in extra]}")
    cache_dir = a.cache or Config.load().path("paths.cache")
    cache = scan_cache(Path(cache_dir))
    print(f"캐시에 있는 영상: {len(cache)}개 ({cache_dir})")

    rows: list[dict] = []
    for split in ("train", "val"):
        got = collect(split, a.probe_video, a.raw, extra)
        print(f"{split}: {len(got)}건")
        rows.extend(got)

    # mp4 는 없어도 캐시가 정상이면 학습에 쓸 수 있다
    cache_only = 0
    for r in rows:
        n = cache.get(r["stem"])
        r["has_cache"] = n is not None
        if n is not None:
            r["cache_frames"] = n
            if not r.get("has_video"):
                cache_only += 1
                r["problems"] = [p for p in r["problems"] if p != "영상 MP4 없음"]
    if cache_only:
        print(f"mp4 는 없지만 캐시가 있어 사용 가능한 영상: {cache_only}개")

    takes = assign_takes(rows, a.take_gap)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "clips.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (a.out / "takes.json").write_text(
        json.dumps(takes, ensure_ascii=False, indent=2), encoding="utf-8")
    # 클립 생성기가 쓰는 프레임 수 메타를 캐시 실측값으로 갱신한다
    if cache:
        (a.out / "cache_meta.json").write_text(
            json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"cache_meta.json 갱신: {len(cache)}개")

    n_prob = sum(1 for r in rows if r["problems"])
    print(f"총 {len(rows)}건, take {len(takes)}개, 문제 있는 항목 {n_prob}건")
    print(f"저장: {a.out / 'clips.jsonl'}, {a.out / 'takes.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
