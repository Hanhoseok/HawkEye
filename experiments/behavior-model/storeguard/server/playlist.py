"""대시보드에서 넘겨 볼 영상 목록.

`.env` 를 고치고 서버를 껐다 켜지 않아도 웹에서 다음 영상으로 넘어갈 수 있게 한다.

구성
  - 인덱스(clips.jsonl)에서 영상 경로와 클래스, 사건 발생 시각을 읽어 목록을 만든다.
  - 기본은 **test 분할**(배포 Validation)이다. 학습에 쓰지 않은 영상이라 데모에 적합하다.
  - 클래스로 거를 수 있다.

안전
  웹에서 임의의 파일 경로를 재생하도록 열어 두면 로컬 파일을 아무거나 읽게 된다.
  그래서 파일 재생은 **인덱스에 등록된 영상**과 설정된 영상 폴더 안으로만 제한한다.
  RTSP 주소는 사용자가 직접 입력한 것만 허용하고, 응답에서는 항상 마스킹한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Item:
    stem: str
    path: str
    class_name: str
    class_ko: str
    store: str
    camera: str
    dataset: str
    n_events: int
    first_event_sec: float | None

    def public(self, idx: int) -> dict:
        return {
            "index": idx, "stem": self.stem, "class_name": self.class_name,
            "class_ko": self.class_ko, "store": self.store, "camera": self.camera,
            "dataset": self.dataset, "n_events": self.n_events,
            "first_event_sec": self.first_event_sec,
        }


class Playlist:
    def __init__(self, cfg, split: str = "test"):
        self.cfg = cfg
        self.split = split
        self.items: list[Item] = []
        self.allowed_dirs: list[Path] = []
        self.error: str | None = None
        self._load()

    def _load(self) -> None:
        index_dir = Path(str(self.cfg["paths.index"]))
        clips = index_dir / "clips.jsonl"
        splits = index_dir / "splits.json"
        if not clips.exists():
            self.error = f"인덱스가 없다: {clips}"
            return
        rows = {}
        for line in clips.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["stem"]] = r

        stems: list[str]
        if splits.exists():
            sp = json.loads(splits.read_text(encoding="utf-8"))
            stems = list(sp.get(self.split) or [])
            if not stems:
                stems = sorted(rows)
        else:
            stems = sorted(rows)

        ko = self.cfg.get("classes.ko", {}) or {}
        fps = float(self.cfg["data.sample_fps"])
        from storeguard.data.clips import event_seconds, load_cache_meta
        cache_meta = load_cache_meta(self.cfg)
        for s in stems:
            r = rows.get(s)
            if not r or not r.get("video_path"):
                continue
            p = Path(r["video_path"])
            if not p.exists():
                continue
            evs = r.get("events") or []
            # 라벨 프레임은 원본 fps 좌표이므로 그대로 나누면 안 된다(구매행동은 10fps).
            first = round(event_seconds(r, evs[0], cache_meta.get(s), fps)[0], 1) if evs else None
            self.items.append(Item(
                stem=s, path=str(p), class_name=r.get("class_name") or "?",
                class_ko=ko.get(r.get("class_name"), r.get("class_name") or "?"),
                store=r.get("store") or "?", camera=r.get("camera") or "?",
                dataset=r.get("dataset") or "?", n_events=len(evs),
                first_event_sec=first,
            ))
        # 클래스가 번갈아 나오도록 섞는다(같은 클래스가 줄줄이 나오면 데모가 지루하다)
        by_cls: dict[str, list[Item]] = {}
        for it in self.items:
            by_cls.setdefault(it.class_name, []).append(it)
        interleaved: list[Item] = []
        i = 0
        while any(v for v in by_cls.values()):
            for cls in sorted(by_cls):
                if i < len(by_cls[cls]):
                    interleaved.append(by_cls[cls][i])
            i += 1
            if i > max((len(v) for v in by_cls.values()), default=0):
                break
        self.items = interleaved or self.items

        raw = Path(str(self.cfg["paths.raw"]))
        self.allowed_dirs = [raw.resolve()]

    # --- 조회 ---
    def classes(self) -> list[str]:
        seen: list[str] = []
        for it in self.items:
            if it.class_name not in seen:
                seen.append(it.class_name)
        return seen

    def filtered(self, class_name: str | None) -> list[int]:
        if not class_name or class_name == "all":
            return list(range(len(self.items)))
        return [i for i, it in enumerate(self.items) if it.class_name == class_name]

    def get(self, idx: int) -> Item | None:
        if 0 <= idx < len(self.items):
            return self.items[idx]
        return None

    def index_of_path(self, path: str) -> int | None:
        try:
            rp = str(Path(path).resolve())
        except Exception:
            return None
        for i, it in enumerate(self.items):
            if str(Path(it.path).resolve()) == rp:
                return i
        return None

    def step(self, current: int | None, delta: int, class_name: str | None) -> int | None:
        """현재 위치에서 delta 만큼 이동한 항목의 인덱스. 목록 끝에서는 처음으로 돈다."""
        pool = self.filtered(class_name)
        if not pool:
            return None
        if current is None or current not in pool:
            return pool[0] if delta >= 0 else pool[-1]
        k = pool.index(current)
        return pool[(k + delta) % len(pool)]

    def is_allowed_file(self, path: str) -> bool:
        """인덱스에 있는 영상이거나, 설정된 원본 폴더 안의 파일만 허용한다."""
        try:
            rp = Path(path).resolve()
        except Exception:
            return False
        if self.index_of_path(path) is not None:
            return True
        return any(str(rp).startswith(str(d)) for d in self.allowed_dirs)
