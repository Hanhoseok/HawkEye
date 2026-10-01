"""클립 목록 생성과 라벨링 규칙.

클립 정의 (실시간과 동일한 인과적 창)
    anchor t 를 '현재 시각'으로 두고, 창은 W = [t - (L-1)*s, t] 이다.
    실시간 추론도 정확히 같은 창을 쓴다(미래 프레임 사용 금지).

라벨 규칙
    inter = |W ∩ event|
    양성  : inter/|W| >= pos_overlap_clip   또는   inter/|event| >= pos_overlap_event
            (두 번째 조건은 전도처럼 짧은 이벤트가 규칙 때문에 통째로 버려지는 것을 막는다)
    배경  : 어떤 이벤트와도 겹치지 않고, 가장 가까운 이벤트 경계에서 bg_margin_frames 이상 떨어짐
    무시  : 그 외(애매 구간). 학습/평가 어디에도 넣지 않는다.

주의
    'background_unlabeled' 는 정상으로 검증된 구간이 아니다. 연출된 이상행동 영상의
    사전 동작이 포함된다. 진짜 정상 데이터는 별도 카테고리 영상을 추가해야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClipSpec:
    stem: str
    anchor: int          # 창의 마지막 프레임 인덱스 (현재 시각)
    start: int           # 창의 첫 프레임 인덱스
    stride: int
    label: str           # 클래스명
    take_id: str
    store: str
    camera: str
    event_index: int     # 양성이면 소속 이벤트 번호, 배경이면 -1

    def frame_indices(self, clip_len: int) -> list[int]:
        return [self.start + i * self.stride for i in range(clip_len)]


def _intervals(events: list[dict], scale: float = 1.0,
               max_frame: int | None = None) -> list[tuple[int, int, int]]:
    """이벤트 구간을 (시작, 끝, 번호) 로 만든다.

    scale 은 **라벨 좌표 → 캐시 좌표** 변환 배율이다.
    라벨의 프레임 번호는 원본 영상 기준이다. 그런데 프레임 캐시는 sample_fps 로
    시간 리샘플링되어 있다. 예를 들어 구매행동 영상은 원본 10fps / 600프레임인데
    3fps 캐시에서는 180프레임이 된다. 이때 라벨 프레임 231 은 캐시 프레임 69 에 해당한다.
    이 변환을 빠뜨리면 이벤트 구간이 영상 뒤쪽으로 3배 밀려 라벨이 통째로 어긋난다.
    """
    out = []
    for i, e in enumerate(events):
        s = int(round(int(e["start_frame"]) * scale))
        en = e["end_frame"]
        en = int(round(int(en) * scale)) if en is not None else s
        if en < s:
            s, en = en, s
        if max_frame is not None:
            s = max(0, min(s, max_frame))
            en = max(0, min(en, max_frame))
        out.append((s, en, i))
    return out


def clips_for_video(row: dict, *, clip_len: int, frame_stride: int,
                    pos_overlap_clip: float, pos_overlap_event: float,
                    bg_margin: int, stride_pos: int, stride_bg: int,
                    max_bg: int, n_frames: int | None = None,
                    max_pos: int = 0) -> list[ClipSpec]:
    """영상 1개에 대한 클립 목록. row 는 clips.jsonl 의 한 줄."""
    label_frames = int(row.get("label_frames") or 0)
    total = int(n_frames or label_frames or 0)
    span = (clip_len - 1) * frame_stride
    if total <= span:
        return []

    # 라벨 좌표(원본 fps) → 캐시 좌표(sample_fps) 변환 배율
    scale = (total / label_frames) if (label_frames and total != label_frames) else 1.0
    evs = _intervals(row.get("events") or [], scale, total - 1)
    cls = row.get("class_name")

    # 사람이 '정상'이라고 확인한 영상(add_normal.py 로 등록)은 영상 전체가 음성이다.
    # 라벨이 없어서 배경인 것과 구분하기 위해 event_index=-2 로 출처를 남긴다.
    if cls == "normal_verified":
        out: list[ClipSpec] = []
        step = max(1, stride_bg)
        for anchor in range(span, total, step):
            out.append(ClipSpec(row["stem"], anchor, anchor - span, frame_stride,
                                "background_unlabeled", row.get("take_id", ""),
                                row.get("store", ""), row.get("camera", ""), -2))
        cap = max_pos or max_bg
        if cap and len(out) > cap:
            k = len(out) / cap
            out = [out[int(i * k)] for i in range(cap)]
        return out

    # 이벤트별로 유효한 anchor 를 모두 모은 뒤 **고르게 솎아낸다.**
    # 고정 간격(stride)만 쓰면 길이가 크게 다른 이벤트에서 수량이 엉망이 된다.
    # 실제로 매장이동(5분 전체가 이벤트)은 영상당 74개, 선택(2초)은 2개가 나와 33배 차이가 났다.
    per_event: dict[int, list[ClipSpec]] = {}
    bg_candidates: list[ClipSpec] = []

    for anchor in range(span, total):
        start = anchor - span
        w_len = anchor - start + 1
        best_ev = -1
        best_ratio = 0.0
        min_dist = 10**9
        for (s, e, idx) in evs:
            inter = max(0, min(anchor, e) - max(start, s) + 1)
            ev_len = e - s + 1
            if inter > 0:
                r_clip = inter / w_len
                r_ev = inter / ev_len
                if r_clip >= pos_overlap_clip or r_ev >= pos_overlap_event:
                    score = max(r_clip, r_ev)
                    if score > best_ratio:
                        best_ratio, best_ev = score, idx
                min_dist = 0
            else:
                d = s - anchor if s > anchor else start - e
                min_dist = min(min_dist, max(0, d))

        if best_ev >= 0:
            per_event.setdefault(best_ev, []).append(
                ClipSpec(row["stem"], anchor, start, frame_stride, cls,
                         row.get("take_id", ""), row.get("store", ""),
                         row.get("camera", ""), best_ev))
        elif min_dist >= bg_margin and min_dist < 10**9:
            if stride_bg <= 1 or (anchor % stride_bg) == 0:
                bg_candidates.append(ClipSpec(row["stem"], anchor, start, frame_stride,
                                              "background_unlabeled", row.get("take_id", ""),
                                              row.get("store", ""), row.get("camera", ""), -1))

    def thin(items: list[ClipSpec], keep: int) -> list[ClipSpec]:
        """고르게 keep 개만 남긴다(앞뒤로 치우치지 않게)."""
        if keep <= 0 or len(items) <= keep:
            return items
        step = len(items) / keep
        return [items[int(i * step)] for i in range(keep)]

    specs: list[ClipSpec] = []
    if per_event:
        budget = max(1, -(-max_pos // len(per_event))) if max_pos else 0
        for idx in sorted(per_event):
            got = thin(per_event[idx], budget) if budget else per_event[idx]
            # stride_pos 는 최소 간격으로만 쓴다(거의 같은 클립이 붙어 나오는 것 방지)
            if stride_pos > 1 and len(got) > 1:
                kept = [got[0]]
                for c in got[1:]:
                    if c.anchor - kept[-1].anchor >= stride_pos:
                        kept.append(c)
                got = kept
            specs.extend(got)
        if max_pos and len(specs) > max_pos:
            specs = thin(sorted(specs, key=lambda c: c.anchor), max_pos)

    specs.extend(thin(bg_candidates, max_bg) if max_bg else bg_candidates)
    specs.sort(key=lambda c: c.anchor)
    return specs


def label_scale(row: dict, cache_frames: int | None) -> float:
    """라벨 좌표 → 캐시 좌표 배율. 원본 fps 와 sample_fps 가 같으면 1.0."""
    lf = int(row.get("label_frames") or 0)
    if not lf or not cache_frames or cache_frames == lf:
        return 1.0
    return cache_frames / lf


def event_seconds(row: dict, event: dict, cache_frames: int | None,
                  sample_fps: float) -> tuple[float, float]:
    """이벤트의 (시작, 종료) 시각(초). 라벨 좌표를 캐시 좌표로 바꾼 뒤 초로 환산한다."""
    sc = label_scale(row, cache_frames)
    s = int(event["start_frame"]) * sc
    e = event.get("end_frame")
    e = (int(e) * sc) if e is not None else s
    return s / sample_fps, e / sample_fps


def load_cache_meta(cfg) -> dict[str, int]:
    """전처리가 기록한 stem -> 캐시 프레임 수. 시간 리샘플링된 영상에서 필수."""
    import json
    from pathlib import Path

    p = Path(str(cfg["paths.index"])) / "cache_meta.json"
    if not p.exists():
        return {}
    try:
        return {k: int(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
    except Exception:
        return {}


def build_clip_list(rows: list[dict], stems: list[str], cfg) -> list[ClipSpec]:
    by_stem = {r["stem"]: r for r in rows}
    cache_meta = load_cache_meta(cfg)
    out: list[ClipSpec] = []
    for stem in stems:
        r = by_stem.get(stem)
        if r is None:
            continue
        out.extend(clips_for_video(
            r,
            n_frames=cache_meta.get(stem),
            clip_len=int(cfg["data.clip_len"]),
            frame_stride=int(cfg["data.frame_stride"]),
            pos_overlap_clip=float(cfg["clips.pos_overlap_clip"]),
            pos_overlap_event=float(cfg["clips.pos_overlap_event"]),
            bg_margin=int(cfg["clips.bg_margin_frames"]),
            stride_pos=int(cfg["clips.stride_pos"]),
            stride_bg=int(cfg["clips.stride_bg"]),
            max_bg=int(cfg["clips.max_bg_per_video"]),
            max_pos=int(cfg.get("clips.max_pos_per_video", 0) or 0),
        ))
    return out
