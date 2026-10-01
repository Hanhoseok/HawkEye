"""라벨과 짝이 맞는 영상만 골라서 추출한다.

왜 필요한가
  238-1 구매행동 배포본은 일부만 내려받아져 있다. 영상만 있고 라벨이 없는 것,
  라벨만 있고 영상이 없는 것이 섞여 있어 그대로 다 풀면 수백 GB 를 낭비한다.
  또 클래스별 수량이 크게 달라(시험 900 vs 나머지 100 내외) 그대로 쓰면 모델이
  많은 클래스로 치우친다.

무엇을 하나
  1) 이미 풀어 둔 라벨 XML 의 stem 집합을 읽는다.
  2) 영상 zip 의 중앙 디렉터리만 보고(전체를 읽지 않는다) 짝이 맞는 엔트리를 찾는다.
  3) 클래스별 상한(--max-per-class)을 **촬영 사건(take) 단위로** 적용해 고른다.
     같은 take 의 여러 카메라 영상은 함께 뽑거나 함께 버린다(분할 누수 방지).
     매장이 골고루 들어가도록 매장별로 번갈아 뽑는다.
  4) 고른 엔트리만 압축을 푼다.

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.extract_paired --dry-run
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.extract_paired --max-per-class 160
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import zipfile
from pathlib import Path

from storeguard import naming
from storeguard.data.extract import DATASETS, KIND_DIRS, SPLIT_DIRS

RAW = Path(r"D:\computervision\data\raw")


def label_stems(raw: Path) -> dict[str, str]:
    """이미 풀린 라벨 XML 의 stem -> 분할."""
    out: dict[str, str] = {}
    for split in ("train", "val"):
        d = raw / split / "labels"
        if d.is_dir():
            for p in d.glob("*.xml"):
                out[p.stem] = split
    return out


def scan_video_zips(src: Path) -> dict[str, tuple[Path, str, str, int]]:
    """stem -> (zip 경로, zip 내부 이름, 분할, 압축 전 크기). 중앙 디렉터리만 읽는다."""
    found: dict[str, tuple[Path, str, str, int]] = {}
    for split, sub in SPLIT_DIRS.items():
        d = src / sub / KIND_DIRS["videos"]
        if not d.is_dir():
            continue
        for z in sorted(d.glob("*.zip")):
            if z.stat().st_size == 0:
                continue
            try:
                with zipfile.ZipFile(z) as zf:
                    for e in zf.infolist():
                        if e.is_dir() or not e.filename.lower().endswith(".mp4"):
                            continue
                        stem = Path(e.filename).stem
                        found.setdefault(stem, (z, e.filename, split, e.file_size))
            except Exception as exc:
                print(f"[경고] {z.name} 열기 실패: {exc}")
    return found


def choose(stems: list[str], max_per_class: int, seed: int,
           prefer: set[str] | None = None) -> tuple[list[str], dict]:
    """클래스별 상한을 take 단위로 적용. 매장이 고루 들어가게 번갈아 뽑는다.

    prefer 에 '이미 디스크에 풀려 있는 stem' 을 주면 그 take 를 먼저 고른다.
    데이터가 늘어 다시 고를 때 이미 받아 둔 영상을 버리고 새로 받는 낭비를 막는다.
    (실측: 상한 250 재선정 시 새로 풀 용량이 182 GB → 69 GB 로 줄었다)
    """
    prefer = prefer or set()
    parsed = {}
    unparsed = []
    for s in stems:
        try:
            parsed[s] = naming.parse_stem(s)
        except naming.NameParseError:
            unparsed.append(s)

    by_class: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for s, n in parsed.items():
        by_class[n.class_name][naming.take_key(n)].append(s)

    rng = random.Random(seed)
    picked: list[str] = []
    report: dict = {}
    for cls, takes in sorted(by_class.items()):
        n_videos = sum(len(v) for v in takes.values())
        if not max_per_class or n_videos <= max_per_class:
            for v in takes.values():
                picked.extend(v)
            report[cls] = {"영상": n_videos, "take": len(takes), "선택": n_videos,
                           "상한적용": False}
            continue
        # 매장별로 take 를 나눈 뒤 번갈아 꺼내 상한에 도달할 때까지 담는다
        by_store: dict[str, list[str]] = collections.defaultdict(list)
        for tk in takes:
            by_store[parsed[takes[tk][0]].store].append(tk)
        for lst in by_store.values():
            rng.shuffle(lst)
            # 이미 받아 둔 영상이 들어 있는 take 를 앞으로 보낸다(재다운로드 낭비 방지)
            lst.sort(key=lambda tk: 0 if any(s in prefer for s in takes[tk]) else 1)
        order: list[str] = []
        i = 0
        while any(i < len(v) for v in by_store.values()):
            for st in sorted(by_store):
                if i < len(by_store[st]):
                    order.append(by_store[st][i])
            i += 1
        take_sel, cnt = [], 0
        for tk in order:
            if cnt >= max_per_class:
                break
            take_sel.append(tk)
            cnt += len(takes[tk])
        for tk in take_sel:
            picked.extend(takes[tk])
        report[cls] = {"영상": n_videos, "take": len(takes),
                       "선택": sum(len(takes[t]) for t in take_sel),
                       "선택_take": len(take_sel), "상한적용": True}
    return sorted(picked), {"클래스별": report, "파싱실패": len(unparsed)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="238-1", choices=list(DATASETS))
    ap.add_argument("--raw", type=Path, default=RAW,
                    help="라벨 XML 이 있는 곳. 기본 data/raw")
    ap.add_argument("--video-dst", type=Path, default=None,
                    help="영상을 풀 위치(기본: --raw 와 동일). 용량이 큰 영상만 "
                         "다른 드라이브에 두고 싶을 때 쓴다. 예: F:\\storeguard_video")
    ap.add_argument("--max-per-class", type=int, default=160,
                    help="클래스당 최대 영상 수(take 단위 적용). 0이면 제한 없음")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true", help="무엇을 풀지만 보고 실제로는 풀지 않음")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    src = DATASETS[a.dataset]
    labels = label_stems(a.raw)
    print(f"풀린 라벨 XML: {len(labels)}개")
    videos = scan_video_zips(src)
    print(f"영상 zip 안의 MP4: {len(videos)}개")

    paired = sorted(set(labels) & set(videos))
    print(f"짝이 맞는 것: {len(paired)}개")
    # 이미 풀려 있는 영상은 우선 재사용한다 (여러 위치를 모두 본다)
    vdst = a.video_dst or a.raw
    roots = {a.raw, vdst}
    on_disk: set[str] = set()
    for root in roots:
        for split in ("train", "val"):
            d = root / split / "videos"
            if d.is_dir():
                on_disk |= {p.stem for p in d.glob("*.mp4")}
    print(f"이미 풀려 있는 영상: {len(on_disk & set(videos))}개 (우선 재사용)")
    if vdst != a.raw:
        print(f"새 영상을 풀 위치: {vdst}")
    if not paired:
        print("[오류] 짝이 맞는 영상이 없다. 라벨을 먼저 풀 것:")
        print("  python -m storeguard.data.extract --dataset 238-1 --what labels --split all")
        return 2

    picked, report = choose(paired, a.max_per_class, a.seed, prefer=on_disk)
    print("\n클래스별 선택 결과")
    for cls, r in report["클래스별"].items():
        mark = " (상한 적용)" if r["상한적용"] else ""
        print(f"  {cls:10s} 짝맞음 {r['영상']:4d}개 / take {r['take']:3d}개"
              f"  → 선택 {r['선택']:4d}개{mark}")
    new = [s for s in picked if s not in on_disk]
    print(f"총 선택: {len(picked)}개 (이미 있음 {len(picked) - len(new)}, 새로 풀 것 {len(new)})")

    # 새로 풀 용량과 디스크 여유를 비교한다(중간에 디스크가 차는 사고 방지)
    import shutil
    need = sum(videos[s][3] for s in new) / 1024**3
    vdst.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(str(vdst)).free / 1024**3
    print(f"새로 풀 용량 {need:.0f} GB / {vdst.anchor} 여유 {free:.0f} GB")
    if need > free * 0.9:
        print("[경고] 여유 공간이 부족하다. --max-per-class 를 낮추거나 공간을 확보할 것.")
        if not a.dry_run:
            return 2

    by_split = collections.Counter(videos[s][2] for s in picked)
    print(f"분할별: {dict(by_split)}")

    if a.dry_run:
        print("\n--dry-run 이므로 실제로 풀지 않았다.")
        return 0

    by_zip: dict[Path, list[str]] = collections.defaultdict(list)
    for s in picked:
        by_zip[videos[s][0]].append(s)

    total = 0
    for zpath, stems in sorted(by_zip.items(), key=lambda kv: kv[0].name):
        split = videos[stems[0]][2]
        dst = vdst / split / "videos"
        dst.mkdir(parents=True, exist_ok=True)
        print(f"\n[{zpath.name}] {len(stems)}개 -> {dst}", flush=True)
        with zipfile.ZipFile(zpath) as zf:
            for i, s in enumerate(stems, 1):
                inner = videos[s][1]
                out = dst / f"{s}.mp4"
                info = zf.getinfo(inner)
                if out.exists() and not a.overwrite and out.stat().st_size == info.file_size:
                    continue
                tmp = out.with_suffix(".mp4.part")
                with zf.open(inner) as fsrc, open(tmp, "wb") as fdst:
                    shutil.copyfileobj(fsrc, fdst, length=1 << 20)
                tmp.replace(out)
                total += 1
                if i % 25 == 0:
                    print(f"    {i}/{len(stems)}", flush=True)

    out_report = Path(r"D:\computervision\reports") / "extract_paired.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(
        {"dataset": a.dataset, "max_per_class": a.max_per_class,
         "paired": len(paired), "picked": len(picked), **report},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n총 {total}개 새로 풀었다. 보고서: {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
