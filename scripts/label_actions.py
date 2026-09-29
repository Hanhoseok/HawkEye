"""행동 구간 라벨링 도구.

영상을 넘기며 "누가 언제 무엇을 했는지"를 키보드로 찍는다.
결과 CSV 는 TAKE 후보 판정의 정답이 되고, 이후 모든 개선을 숫자로 비교할 수 있게 한다.

    python scripts/label_actions.py data/videos/store-aisle-detection.mp4 \
        --identities outputs/take-base/identities.jsonl --out labels/store-aisle.csv

identities.jsonl 을 주면 person_id 박스가 함께 표시되어 누구를 라벨링하는지 헷갈리지 않는다.

조작키 (화면 HUD 에도 표시됨)
    SPACE   재생 / 일시정지
    d  a    1 프레임 이동 (방향키도 됨)   . ,   10 프레임        ] [   100 프레임
    0~9     라벨링할 person 선택      TAB   다음 person 으로 순환
    t       TAKE 구간 시작/끝
    r       RETURN 구간 시작/끝
    b       BROWSE 구간 시작/끝 (구경만 하고 집지 않음 - 오탐 확인용)
    x       진행 중인 구간 취소
    u       마지막 구간 삭제
    s       저장
    q/ESC   저장하고 종료

BROWSE 를 함께 찍는 이유: 오탐을 세려면 "집지 않았는데 선반 앞에 있었던" 구간이 필요하다.
TAKE 만 찍으면 재현율은 재도 정밀도는 못 잰다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sinks.observation_log import read_identities  # noqa: E402
from src.zones.zone_map import ZoneMap  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
ACTIONS = {ord("t"): "TAKE", ord("r"): "RETURN", ord("b"): "BROWSE"}

# 방향키 코드는 OS/백엔드마다 다르다. Windows 는 waitKeyEx 로만 잡히고 값도 크다.
# (waitKey() & 0xFF 로는 방향키가 전부 0 이 되어 동작하지 않는다)
LEFT_KEYS = {2424832, 65361, 81}
RIGHT_KEYS = {2555904, 65363, 83}


def normalize_key(raw: int) -> int:
    """방향키를 a/d 로 통일한다. 그 외는 하위 8비트."""
    if raw in LEFT_KEYS:
        return ord("a")
    if raw in RIGHT_KEYS:
        return ord("d")
    return raw & 0xFF if raw >= 0 else 255
ACTION_COLOR = {"TAKE": (60, 60, 255), "RETURN": (60, 200, 255), "BROWSE": (160, 160, 160)}


@dataclass
class Segment:
    person_id: int
    action: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float


def color_for(track_id: int) -> tuple[int, int, int]:
    import numpy as np

    hue = int(((track_id * 0.618033988749895) % 1.0) * 179)
    bgr = cv2.cvtColor(np.uint8([[(hue, 200, 255)]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class Labeler:
    def __init__(self, args) -> None:
        self.cap = cv2.VideoCapture(args.source)
        if not self.cap.isOpened():
            raise SystemExit(f"영상을 열 수 없습니다: {args.source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.out_path = Path(args.out)
        self.tolerance = args.tolerance
        self.index = args.start_frame
        self.playing = False
        self.segments: list[Segment] = []
        self.open_segment: dict[tuple[int, str], int] = {}  # (person, action) -> 시작 프레임
        self.selected: int | None = None

        self.by_frame: dict[int, list] = defaultdict(list)
        self.people: list[int] = []
        if args.identities:
            for obs in read_identities(args.identities):
                if obs.person_id > 0:
                    self.by_frame[obs.frame].append(obs)
            self.people = sorted({o.person_id for obs in self.by_frame.values() for o in obs})
            if self.people:
                self.selected = self.people[0]

        self.zone_map = None
        if args.zones and Path(args.zones).exists():
            self.zone_map = ZoneMap.load(args.zones)
            self.zone_map.resolve(self.width, self.height)

        if self.out_path.exists():
            self._load_existing()

    # ------------------------------------------------------------------

    def _load_existing(self) -> None:
        with self.out_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                self.segments.append(
                    Segment(
                        int(row["person_id"]),
                        row["action"],
                        int(row["start_frame"]),
                        int(row["end_frame"]),
                        float(row["start_sec"]),
                        float(row["end_sec"]),
                    )
                )
        print(f"기존 라벨 {len(self.segments)}건을 불러왔습니다: {self.out_path}")

    def save(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["person_id", "action", "start_frame", "end_frame", "start_sec", "end_sec"],
            )
            writer.writeheader()
            for seg in sorted(self.segments, key=lambda s: (s.start_frame, s.person_id)):
                writer.writerow(asdict(seg))
        print(f"저장: {self.out_path} ({len(self.segments)}건)")

    # ------------------------------------------------------------------

    def observations_at(self, index: int) -> list:
        """stride 로 건너뛴 프레임에서도 가까운 관측을 찾는다."""
        for offset in range(0, self.tolerance + 1):
            for idx in (index - offset, index + offset):
                if idx in self.by_frame:
                    return self.by_frame[idx]
        return []

    def read(self, index: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = self.cap.read()
        return img if ok else None

    def render(self, img):
        from src.viz.overlay import draw_zones

        canvas = draw_zones(img, self.zone_map) if self.zone_map else img.copy()

        for obs in self.observations_at(self.index):
            x1, y1, x2, y2 = (int(v) for v in obs.bbox)
            chosen = obs.person_id == self.selected
            color = color_for(obs.person_id)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3 if chosen else 1)
            tag = f"P{obs.person_id}" + (" <" if chosen else "")
            cv2.putText(canvas, tag, (x1, max(12, y1 - 5)), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, tag, (x1, max(12, y1 - 5)), FONT, 0.5, color, 1, cv2.LINE_AA)

        h, w = canvas.shape[:2]
        bar = 66
        cv2.rectangle(canvas, (0, 0), (w, bar), (0, 0, 0), -1)
        sec = self.index / self.fps
        state = "PLAY" if self.playing else "PAUSE"
        head = f"f{self.index}/{self.total}  {sec:6.2f}s  [{state}]  person=P{self.selected}"
        cv2.putText(canvas, head, (8, 18), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 진행 중인 구간
        if self.open_segment:
            parts = [f"{a}@P{p} from f{s}" for (p, a), s in self.open_segment.items()]
            cv2.putText(canvas, "OPEN: " + ", ".join(parts), (8, 36), FONT, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "a/d=step  ./,=10  ]/[=100  TAB=person  t/r/b=label  u=undo s=save q=quit",
                        (8, 36), FONT, 0.40, (170, 170, 170), 1, cv2.LINE_AA)

        # 이 프레임에 걸치는 확정 라벨
        active = [s for s in self.segments if s.start_frame <= self.index <= s.end_frame]
        if active:
            txt = "  ".join(f"P{s.person_id}:{s.action}" for s in active[:6])
            cv2.putText(canvas, "LABELED: " + txt, (8, 56), FONT, 0.45, (120, 255, 120), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, f"total labels: {len(self.segments)}", (8, 56), FONT, 0.45,
                        (150, 150, 150), 1, cv2.LINE_AA)

        # 하단 진행 바 + 라벨 위치 표시
        y = h - 8
        cv2.line(canvas, (0, y), (w, y), (70, 70, 70), 5)
        for s in self.segments:
            x_a = int(s.start_frame / max(1, self.total) * w)
            x_b = max(x_a + 2, int(s.end_frame / max(1, self.total) * w))
            cv2.line(canvas, (x_a, y), (x_b, y), ACTION_COLOR.get(s.action, (200, 200, 200)), 5)
        cv2.line(canvas, (int(self.index / max(1, self.total) * w), y - 6),
                 (int(self.index / max(1, self.total) * w), y + 6), (255, 255, 255), 2)
        return canvas

    # ------------------------------------------------------------------

    def toggle(self, action: str) -> None:
        if self.selected is None:
            print("[warn] person 을 먼저 선택하세요 (0~9 또는 TAB)")
            return
        key = (self.selected, action)
        if key in self.open_segment:
            start = self.open_segment.pop(key)
            a, b = min(start, self.index), max(start, self.index)
            self.segments.append(
                Segment(self.selected, action, a, b, round(a / self.fps, 3), round(b / self.fps, 3))
            )
            print(f"  + P{self.selected} {action} f{a}-f{b} ({(b-a)/self.fps:.2f}초)")
        else:
            self.open_segment[key] = self.index
            print(f"  … P{self.selected} {action} 시작 f{self.index}")

    def handle(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key == ord(" "):
            self.playing = not self.playing
        elif key == ord("d"):
            self.index += 1
        elif key == ord("a"):
            self.index -= 1
        elif key == ord("."):
            self.index += 10
        elif key == ord(","):
            self.index -= 10
        elif key == ord("]"):
            self.index += 100
        elif key == ord("["):
            self.index -= 100
        elif key == 9 and self.people:  # TAB
            cur = self.people.index(self.selected) if self.selected in self.people else -1
            self.selected = self.people[(cur + 1) % len(self.people)]
        elif ord("0") <= key <= ord("9"):
            self.selected = key - ord("0")
        elif key in ACTIONS:
            self.toggle(ACTIONS[key])
        elif key == ord("x"):
            self.open_segment.clear()
            print("  진행 중인 구간 취소")
        elif key == ord("u") and self.segments:
            removed = self.segments.pop()
            print(f"  - 삭제: P{removed.person_id} {removed.action} f{removed.start_frame}-f{removed.end_frame}")
        elif key == ord("s"):
            self.save()
        self.index = max(0, min(self.total - 1, self.index))
        return True

    def run(self) -> None:
        print(__doc__.split("조작키")[1] if "조작키" in __doc__ else "")
        print(f"영상: {self.width}x{self.height}, {self.fps:.1f}fps, {self.total} frames")
        print(f"person 목록: {self.people or '(identities 없음)'}\n")
        window = "label (q=quit)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            img = self.read(self.index)
            if img is None:
                self.playing = False
                self.index = max(0, self.index - 1)
                img = self.read(self.index)
                if img is None:
                    break
            cv2.imshow(window, self.render(img))
            raw = cv2.waitKeyEx(int(1000 / self.fps) if self.playing else 0)
            key = normalize_key(raw)
            if key != 255 and not self.handle(key):
                break
            if self.playing and key == 255:
                self.index += 1
        cv2.destroyAllWindows()
        self.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--identities", help="person_id 박스를 함께 표시할 identities.jsonl")
    parser.add_argument("--zones", default="zones.yaml")
    parser.add_argument("--out", default="labels/labels.csv")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--tolerance", type=int, default=3, help="관측을 찾을 때 허용할 프레임 오차")
    parser.add_argument("--dry-run", action="store_true", help="창을 띄우지 않고 첫 화면만 PNG 로 저장")
    args = parser.parse_args()

    labeler = Labeler(args)
    if args.dry_run:
        img = labeler.read(labeler.index)
        if img is None:
            raise SystemExit("프레임을 읽을 수 없습니다")
        out = Path("outputs/frames/labeler_preview.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), labeler.render(img))
        print(f"미리보기 저장: {out}")
        return
    labeler.run()


if __name__ == "__main__":
    main()
