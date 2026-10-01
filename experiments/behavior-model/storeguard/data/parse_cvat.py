"""CVAT 1.1 XML 라벨 파서.

실제 확인된 구조(2,169개 파일 전수 조사 기준):

    <annotations>
      <version>1.1</version>
      <meta><task>
        <size>180</size>              # 프레임 수
        <start_frame>0</start_frame>
        <stop_frame>179</stop_frame>
        <original_size><width>1920</width><height>1080</height></original_size>
        <labels>...</labels>          # 스키마 정의(8종 이상행동 전부 나열됨, 실제 사용과 무관)
      </task></meta>
      <track id=".." label="fall_start"> <box frame=".." outside="0" xtl=".." .../> </track>
      <track id=".." label="fall_end">   <box .../> </track>
      <track id=".." label="Right foot"> <points frame=".." points="x,y"/> </track>
      <track id=".." label="object_article"> <box ...><attribute name="article ID">1</attribute></box> </track>
    </annotations>

중요한 사실:
- `meta/task/labels` 에는 fire/smoke/abandon/fight/weak pedestrian 까지 8종이 항상 선언되어 있다.
  이것은 CVAT 프로젝트 스키마일 뿐 실제 라벨이 아니다. **실제 라벨은 `track` 에만 있다.**
- `*_start` / `*_end` 트랙은 각각 box 2개를 가진다: outside="0" 인 프레임 1개 + 그 다음 프레임 outside="1".
  즉 시작/종료 **시점**을 한 프레임으로 찍은 마커다. 구간은 [start_frame, end_frame] 으로 구성한다.
- 한 영상에 이벤트가 2~3개 있는 경우가 있다(절도 27건, 전도 20건, 파손 2건).
- 키포인트(17종)는 모든 프레임이 아니라 일부 프레임에만 있다(영상당 13~131 프레임).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

EVENT_PREFIXES = ("fall", "broken", "theft", "fire", "smoke", "abandon", "fight",
                  "select", "test", "buying", "return", "compare")

# 238-1 매장이동은 start/end 쌍이 아니라 `moving` 단일 라벨이며 영상 전체가 해당 행동이다.
WHOLE_VIDEO_LABELS = {"moving"}

KEYPOINT_LABELS = [
    "Pelvis", "Left hip", "Left knee", "Left foot", "Right  hip", "Right knee",
    "Right foot", "Spine naval", "Spine chest", "Neck base", "Center head",
    "Right shoulder", "Right elbow", "Right hand", "Left shoulder", "Left elbow",
    "Left hand",
]
OBJECT_LABELS = (
    # 238-2 이상행동
    "object_article", "object_tool", "object_abandon",
    # 238-1 구매행동 (가이드라인에서 확인)
    "object_daily necessity", "object_groceries",
)


@dataclass
class Box:
    frame: int
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    attrs: dict = field(default_factory=dict)


@dataclass
class Event:
    """[start_frame, end_frame] 폐구간. end 가 없으면 end_frame=None."""
    action: str            # fall / broken / theft ...
    start_frame: int
    end_frame: int | None
    start_box: Box | None = None
    end_box: Box | None = None

    @property
    def n_frames(self) -> int | None:
        if self.end_frame is None:
            return None
        return self.end_frame - self.start_frame + 1


@dataclass
class Annotation:
    stem: str
    n_frames: int
    width: int
    height: int
    events: list[Event]
    keypoint_frames: list[int]           # 키포인트가 존재하는 프레임 번호 (정렬)
    person_ids: set[str]                 # box attribute "ID" 값 모음
    object_boxes: dict[str, list[Box]]   # object_article / object_tool / object_abandon
    genders: set[str]
    age_groups: set[str]
    problems: list[str] = field(default_factory=list)

    def event_frames(self) -> set[int]:
        out: set[int] = set()
        for e in self.events:
            end = e.end_frame if e.end_frame is not None else e.start_frame
            out.update(range(e.start_frame, end + 1))
        return out


def _first_inside_box(track: ET.Element) -> Box | None:
    """outside=='0' 인 첫 box. 마커 트랙은 이것이 실제 시점이다."""
    for b in track.findall("box"):
        if b.get("outside") == "0":
            attrs = {a.get("name"): (a.text or "") for a in b.findall("attribute")}
            return Box(
                frame=int(b.get("frame")),
                xtl=float(b.get("xtl")), ytl=float(b.get("ytl")),
                xbr=float(b.get("xbr")), ybr=float(b.get("ybr")),
                attrs=attrs,
            )
    return None


def parse(path: str | Path) -> Annotation:
    path = Path(path)
    root = ET.parse(path).getroot()
    task = root.find("meta/task")
    if task is None:
        raise ValueError(f"meta/task 없음: {path}")

    n_frames = int(task.findtext("size", "0"))
    osz = task.find("original_size")
    width = int(osz.findtext("width", "0")) if osz is not None else 0
    height = int(osz.findtext("height", "0")) if osz is not None else 0

    starts: dict[str, list[Box]] = {}
    ends: dict[str, list[Box]] = {}
    kp_frames: set[int] = set()
    person_ids: set[str] = set()
    genders: set[str] = set()
    ages: set[str] = set()
    objects: dict[str, list[Box]] = {k: [] for k in OBJECT_LABELS}
    problems: list[str] = []

    whole_video: list[str] = []
    for tr in root.findall("track"):
        label = (tr.get("label") or "").strip()
        if label in WHOLE_VIDEO_LABELS:
            # 매장이동: 영상 전체가 해당 행동. 구간을 [0, n_frames-1] 로 만든다.
            whole_video.append(label)
            for b in tr.findall("box"):
                attrs = {a.get("name"): (a.text or "") for a in b.findall("attribute")}
                person_ids.update(v for k, v in attrs.items() if k == "ID")
                if "gender" in attrs:
                    genders.add(attrs["gender"])
                if "age group" in attrs:
                    ages.add(attrs["age group"])
        elif label.endswith("_start") or label.endswith("_end"):
            action = label.rsplit("_", 1)[0]
            box = _first_inside_box(tr)
            if box is None:
                problems.append(f"{label} 트랙에 outside=0 박스 없음")
                continue
            person_ids.update(v for k, v in box.attrs.items() if k == "ID")
            if "gender" in box.attrs:
                genders.add(box.attrs["gender"])
            if "age group" in box.attrs:
                ages.add(box.attrs["age group"])
            (starts if label.endswith("_start") else ends).setdefault(action, []).append(box)
        elif label in OBJECT_LABELS:
            for b in tr.findall("box"):
                if b.get("outside") != "0":
                    continue
                attrs = {a.get("name"): (a.text or "") for a in b.findall("attribute")}
                objects[label].append(Box(
                    frame=int(b.get("frame")),
                    xtl=float(b.get("xtl")), ytl=float(b.get("ytl")),
                    xbr=float(b.get("xbr")), ybr=float(b.get("ybr")),
                    attrs=attrs,
                ))
        elif label in KEYPOINT_LABELS:
            for p in tr.findall("points"):
                if p.get("outside") == "0":
                    kp_frames.add(int(p.get("frame")))

    events: list[Event] = []
    for action in sorted(set(starts) | set(ends)):
        s = sorted(starts.get(action, []), key=lambda b: b.frame)
        e = sorted(ends.get(action, []), key=lambda b: b.frame)
        if len(s) != len(e):
            problems.append(f"{action}: start {len(s)}개 vs end {len(e)}개 불일치")
        for i, sb in enumerate(s):
            eb = e[i] if i < len(e) else None
            if eb is not None and eb.frame < sb.frame:
                problems.append(f"{action}#{i}: end({eb.frame}) < start({sb.frame})")
                eb = None
            events.append(Event(
                action=action, start_frame=sb.frame,
                end_frame=eb.frame if eb else None,
                start_box=sb, end_box=eb,
            ))
    for action in sorted(set(whole_video)):
        events.append(Event(action=action, start_frame=0,
                            end_frame=max(0, n_frames - 1)))
    events.sort(key=lambda ev: ev.start_frame)

    if not events:
        problems.append("이벤트 트랙 없음")

    return Annotation(
        stem=path.stem, n_frames=n_frames, width=width, height=height,
        events=events, keypoint_frames=sorted(kp_frames), person_ids=person_ids,
        object_boxes=objects, genders=genders, age_groups=ages, problems=problems,
    )


def iter_xml(root: str | Path) -> Iterable[Path]:
    return sorted(Path(root).rglob("*.xml"))
