"""두 화면 짝이 왜 끊기나 — 저장된 기록만 다시 돌려서 잰다 (docs/multiview-progress.md 후속 13-1, 한계 B-14).

    python scripts/link_diagnose.py why     # 0.5초 조각마다 짝이 안 맺힌 이유를 나눠 센다
    python scripts/link_diagnose.py fit     # 선반 카메라의 어떤 값이 '선반 앞 띠 위치'를 잘 맞히나 (정상 영상, 짝 하나씩 빼고 시험)
    python scripts/link_diagnose.py try     # 해결 후보를 실행 중에만 바꿔 끼워 38쌍을 다시 판정해 견준다
    python scripts/link_diagnose.py --shoot 3 why|fit|try|check   # 3차 촬영 (선반 카메라에도 바닥 지도 — 후속 27)
        why   = 3차 기본(바닥 위 거리)의 끊김. 예전 방식은 --set fusion.position_mode=band
        fit   = 같은 사람의 두 카메라 발이 바닥 위에서 얼마나 떨어져 나오나 (정상 영상) → fusion.max_floor_gap_cm·ray_slack_cm 의 근거
        try   = 띠 위치(한 줄) / 바닥 위 거리 / 짝 유지를 견준다 (연결·끊김·채점)
        check = 두 명 이상 장면에서 맺은 짝이 '맞는 사람'인가 (발목이 두 화면 모두 보인 조각을 기준으로)
    python scripts/link_diagnose.py [--shoot 3] gate   # 손님으로 센 신원만 짝 후보로 삼으면(fusion.link_customers_only) 무엇이 달라지나 (후속 31)

아무것도 고치지 않는다 — 설정·기록·코드를 그대로 두고, `try` 의 후보(위치식 바꾸기·짝 유지)는 이 실행 안에서만 바꿔 끼운다.
영상·모델이 필요 없다. 기록(`outputs/dual_eval_pose3`)만 있으면 된다. `why` 약 1분, `fit` 수 초, `try` 는 후보마다 약 1분.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import dual_eval as de  # noqa: E402
from sync_pair import SyncTable  # noqa: E402

from src.core.config import AppConfig  # noqa: E402
from src.fusion import cross_view as cv  # noqa: E402
from src.fusion import judge as jd  # noqa: E402
from src.zones.floor import FloorMap  # noqa: E402

KEYPOINT_CONF = 0.4
"""관절을 '보인다'고 치는 신뢰도 하한."""
HIP, SHOULDER, HEAD = (11, 12), (5, 6), (0, 1, 2, 3, 4)
LINK_GAP = 0.125
"""외형이 없을 때 짝이 맺히는 띠 위치 차이의 한계 — 위치 점수 0.5 = 1 − 차이 / max_band_gap(0.25)."""

_orig_update = cv.CrossViewLinker.update
_orig_detail_step = jd.DualJudge.detail_step


def _mid(keypoints, ids, width: float, height: float) -> tuple[float, float] | None:
    """관절 몇 개의 가운데 (화면 비율). 하나도 안 보이면 None."""
    if keypoints is None or len(keypoints) < 17:
        return None
    pts = [keypoints[i] for i in ids if keypoints[i][2] >= KEYPOINT_CONF and not (keypoints[i][0] == 0 and keypoints[i][1] == 0)]
    if not pts:
        return None
    return float(np.mean([p[0] for p in pts])) / width, float(np.mean([p[1] for p in pts])) / height


def _done_pairs(pattern: str | None = None) -> list[str]:
    return [p for p in de.pairs(pattern) if (de.OUT / p / ".done").exists()]


# ---------------------------------------------------------------------- why: 조각마다 이유


class Recorder:
    """`CrossViewLinker.update` 를 감싸, 선반 카메라 사람 × 조각마다 결과와 이유를 적는다. 짝짓기 자체는 원래 코드가 한다."""

    def __init__(self, sticky: bool = False) -> None:
        self.rows: list[dict] = []
        self.pair = ""
        self.sticky = sticky

    def install(self) -> None:
        recorder = self

        def update(linker, now_ms):
            window = linker.config.window_seconds * 1000.0
            if linker._last_window_ms is None or now_ms - linker._last_window_ms < window:
                return _orig_update(linker, now_ms)
            start, end = linker._last_window_ms, now_ms
            detail = [pid for pid, t in linker._detail.items() if t.last_ms >= start]
            overview = linker.overview_candidates(start)
            prev = dict(linker._current)
            before = {d: recorder._candidates(linker, d, overview, start, end) for d in detail}
            made = _orig_update(linker, now_ms)
            if recorder.sticky:
                recorder._keep(linker, made, detail, overview, prev, start, end)
            linked = {l.detail_id: l for l in made}
            for d in detail:
                recorder.rows.append(recorder._row(linker, d, before[d], linked.get(d), overview, prev.get(d), start))
            return made

        cv.CrossViewLinker.update = update

    @staticmethod
    def _band(track, start: float, end: float, tol: float = 0.0) -> tuple[float | None, int]:
        """그 조각의 띠 위치 평균과, 띠 위치가 있든 없든 그때 표본 수."""
        inside = [s for s in track.samples if start - tol <= s.pts_ms <= end + tol]
        vals = [s.band for s in inside if s.band is not None]
        return (float(np.mean(vals)) if vals else None), len(inside)

    @staticmethod
    def _floor_gap(linker, a, b, start: float, end: float) -> tuple[float | None, int, int]:
        """(같은 때 두 화면 발의 바닥 위 거리 평균 cm, 바닥 좌표로 견준 표본 수, 견줄 수 있었던 표본 수)."""
        tol = linker.config.pair_tolerance_seconds * 1000.0
        bs = [s for s in b.samples if start - tol <= s.pts_ms <= end + tol]
        dists, n = [], 0
        for s in a.samples:
            if not (start <= s.pts_ms <= end) or not bs:
                continue
            near = min(bs, key=lambda x: abs(x.pts_ms - s.pts_ms))
            if abs(near.pts_ms - s.pts_ms) > tol:
                continue
            n += 1
            if s.floor is not None and near.floor is not None:
                dists.append(cv.floor_distance(s, near))
        return (float(np.mean(dists)) if dists else None), len(dists), n

    @staticmethod
    def _point_gap(linker, a, b, start: float, end: float) -> tuple[float | None, int]:
        """(두 화면 모두 발목이 보인 표본만으로 잰 바닥 위 거리 평균 cm, 그 표본 수). 발이 가려 반직선으로 본 표본은 뺀다 —
        반직선까지의 거리는 실제 거리의 하한일 뿐이라, '누가 맞는 짝인가'의 기준으로는 이 값만 쓴다 (check)."""
        tol = linker.config.pair_tolerance_seconds * 1000.0
        bs = [s for s in b.samples if start - tol <= s.pts_ms <= end + tol]
        dists = []
        for s in a.samples:
            if not (start <= s.pts_ms <= end) or not bs or s.floor is None or s.ray is not None:
                continue
            near = min(bs, key=lambda x: abs(x.pts_ms - s.pts_ms))
            if abs(near.pts_ms - s.pts_ms) > tol or near.floor is None or near.ray is not None:
                continue
            dists.append(float(np.hypot(s.floor[0] - near.floor[0], s.floor[1] - near.floor[1])))
        return (float(np.mean(dists)) if dists else None), len(dists)

    def _candidates(self, linker, d: int, overview: list[int], start: float, end: float) -> dict:
        tol = linker.config.pair_tolerance_seconds * 1000.0
        dband, _ = self._band(linker._detail[d], start, end)
        cands = []
        for o in overview:
            pos = linker._position_score(linker._detail[d], linker._overview[o], start, end)
            app = linker._appearance_score(linker._detail[d], linker._overview[o])
            oband, seen = self._band(linker._overview[o], start, end, tol)
            fgap, n_floor, n_all = self._floor_gap(linker, linker._detail[d], linker._overview[o], start, end)
            pgap, n_point = self._point_gap(linker, linker._detail[d], linker._overview[o], start, end)
            cands.append({"o": o, "pos": pos, "app": app, "score": linker._combine(pos, app), "oband": oband, "seen": seen,
                          "fgap": fgap, "n_floor": n_floor, "n_all": n_all, "pgap": pgap, "n_point": n_point})
        return {"dband": dband, "cands": cands}

    def _row(self, linker, d: int, before: dict, link, overview: list[int], prev: int | None, start: float) -> dict:
        row = {"pair": self.pair, "d": d, "start": start, "dband": before["dband"], "n_over": len(overview), "o": None}
        cands = before["cands"]
        row["fgaps"] = sorted(c["fgap"] for c in cands if c["fgap"] is not None)   # 후보마다의 바닥 위 거리 (가까운 순)
        row["overs"] = list(overview)
        row["cands"] = [{k: c[k] for k in ("o", "fgap", "pgap", "n_point", "n_all")} for c in cands]
        if link is not None:
            mine = next((c for c in cands if c["o"] == link.overview_id), None) or {}
            oband = mine.get("oband")
            row.update(fgap=mine.get("fgap"), n_floor=mine.get("n_floor", 0), n_all=mine.get("n_all", 0))
            row.update(reason="linked", o=link.overview_id, pos=link.position, app=link.appearance, oband=oband,
                       rescued=(link.position is not None and link.appearance is not None
                                and linker._combine(link.position, link.appearance) < linker.config.min_score - linker.config.keep_bonus))
            return row
        if not overview:
            return {**row, "reason": "no_overview"}
        scored = [c for c in cands if c["score"] is not None]
        if not scored:
            # 위치를 비교할 수 없음: 전체 카메라 사람이 띠 범위(−0.3~1.3) 밖이거나, 같은 때 표본이 없음
            return {**row, "reason": "off_band" if any(c["seen"] for c in cands) else "no_sample"}
        bonus = linker.config.keep_bonus
        best = max(scored, key=lambda c: c["score"] + (bonus if prev == c["o"] else 0.0))
        score = best["score"] + (bonus if prev == best["o"] else 0.0)
        row.update(pos=best["pos"], app=best["app"], oband=best["oband"], fgap=best["fgap"], n_floor=best["n_floor"],
                   n_all=best["n_all"])
        return {**row, "reason": "low_score" if score < linker.config.min_score else "taken"}

    @staticmethod
    def _keep(linker, made: list, detail: list[int], overview: list[int], prev: dict, start: float, end: float) -> None:
        """후보 '짝 유지': 직전 짝이 아직 보이고, 다른 사람이 그 짝을 가져가지 않았고, 위치가 허용 거리의 두 배 안이면 잇는다.
        (바닥 방식이고 바닥 위 거리를 잴 수 있으면 그 거리로, 아니면 띠 위치로.)"""
        tol = linker.config.pair_tolerance_seconds * 1000.0
        used = {l.overview_id for l in made}
        have = {l.detail_id for l in made}
        for d in detail:
            o = prev.get(d)
            if d in have or o is None or o not in overview or o in used:
                continue
            pos = linker._position_score(linker._detail[d], linker._overview[o], start, end)
            a, _ = Recorder._band(linker._detail[d], start, end)
            b, _ = Recorder._band(linker._overview[o], start, end, tol)
            fgap = Recorder._floor_gap(linker, linker._detail[d], linker._overview[o], start, end)[0]
            if pos is None:
                continue
            if linker.config.position_mode == "floor" and fgap is not None:
                if fgap > 2 * linker.config.max_floor_gap_cm:
                    continue
            elif a is None or b is None or abs(a - b) > 2 * linker.config.max_band_gap:
                continue
            link = cv.Link(start, end, d, o, float(pos), pos, None)
            made.append(link)
            linker.links.append(link)
            linker._remember(d, o, start, end)
            used.add(o)

    # 집계 --------------------------------------------------------------

    def breaks(self) -> tuple[int, int, list[float]]:
        """(중간에 끊겼다 다시 이어진 횟수, 짝이 다른 사람으로 바뀐 횟수, 끊긴 길이 초)."""
        by: dict[tuple, list[dict]] = collections.defaultdict(list)
        for r in self.rows:
            by[(r["pair"], r["d"])].append(r)
        n_breaks, n_switch, lengths = 0, 0, []
        for rows in by.values():
            rows.sort(key=lambda r: r["start"])
            gap, last = 0, None
            for r in rows:
                if r["reason"] != "linked":
                    gap += 1
                    continue
                if last is not None and gap:
                    n_breaks += 1
                    lengths.append(gap * 0.5)
                if last is not None and last != r["o"]:
                    n_switch += 1
                gap, last = 0, r["o"]
        return n_breaks, n_switch, lengths

    def linked_pct(self, prefix: str = "") -> float:
        rows = [r for r in self.rows if r["pair"].startswith(prefix)]
        return 100.0 * sum(r["reason"] == "linked" for r in rows) / max(1, len(rows))


_BASE_SET: list[str] = []
"""명령줄 --set (모든 실행에 얹는다)."""


def _apply(base: AppConfig, fusion: dict) -> None:
    """이 실행에서만 fusion 설정을 바꾼다. position_mode 는 회차별 기본값(3차 = floor, dual_config)보다 나중에 먹도록 --set 과 같은 길로."""
    for key, value in fusion.items():
        setattr(base.fusion, key, value)
    de.OVERRIDES[:] = _BASE_SET + [f"fusion.{key}={fusion[key]}" for key in ("position_mode", "link_customers_only") if key in fusion]


def _judge_all(recorder: Recorder, base: AppConfig, pattern: str | None = None) -> list[dict]:
    table = SyncTable.load(de.LABELS / "sync.csv")
    band, truth = de.load_band_map(), de.load_truth()
    results = []
    for pair in _done_pairs(pattern):
        recorder.pair = pair
        try:
            results.append(de.judge_pair(pair, base, table.offset_ms(pair) or 0.0, band, truth=truth))
        except de.PAIR_ERRORS as e:
            if de.SHOOT != 3:
                raise
            print(f"  (건너뜀 {pair}: {e})")
    return results


REASONS = {"linked": "짝 있음", "low_score": "두 화면의 위치가 안 맞음 (점수 미달)", "taken": "그 짝을 다른 사람이 가져감",
           "no_overview": "전체 카메라에 사람 없음", "off_band": "전체 카메라 사람이 띠 범위 밖", "no_sample": "같은 때 전체 카메라 표본 없음"}


def why() -> None:
    rec = Recorder()
    rec.install()
    _judge_all(rec, AppConfig.load(ROOT / "config.yaml"))
    rows = rec.rows
    print(f"선반 카메라 사람 × 0.5초 조각 {len(rows)}개 ({len({r['pair'] for r in rows})}쌍)")
    count = collections.Counter(r["reason"] for r in rows)
    for key, name in REASONS.items():
        print(f"  {name:38s} {count[key]:5d}  {100 * count[key] / len(rows):5.1f}%")
    print("  세트별 짝 있음: " + " · ".join(f"{s} {rec.linked_pct(s):.0f}%" for s in "NTC"))
    n_breaks, n_switch, lengths = rec.breaks()
    span = f" (길이 중앙 {np.median(lengths):.1f}초, 최장 {max(lengths):.1f}초)" if lengths else ""
    print(f"중간에 끊겼다 다시 이어진 횟수 {n_breaks}{span} · 짝이 다른 사람으로 바뀐 횟수 {n_switch}")
    print(f"짝 있음 중 '후보 한 명이면 외형으로 잇기' 덕인 것: {sum(1 for r in rows if r.get('rescued'))}")

    low = [r for r in rows if r["reason"] == "low_score"]
    pos = np.array([r["pos"] for r in low])
    app = np.array([np.nan if r["app"] is None else r["app"] for r in low])
    gap = np.array([abs(r["dband"] - r["oband"]) for r in low if r["dband"] is not None and r["oband"] is not None])
    if low:
        print(f"\n위치가 안 맞은 조각 {len(low)}개: 위치 점수 중앙 {np.median(pos):.2f} (0 인 것 {int((pos == 0).sum())}), "
              f"외형 중앙 {np.nanmedian(app):.2f}"
              + (", 띠 위치 차이 25·50·75% = " + " · ".join(f"{v:.2f}" for v in np.percentile(gap, [25, 50, 75])) if len(gap) else ""))

    solo = [r for r in rows if r["pair"].startswith("N") and r["n_over"] == 1 and r["dband"] is not None and r.get("oband") is not None]
    print(f"\n정상 영상에서 전체 카메라에 한 명뿐인 조각 {len(solo)}개 — 선반 카메라가 본 띠 위치 구간별 (선반 − 전체)")
    for lo, hi in ((-9, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 9)):
        part = [r for r in solo if lo <= r["dband"] < hi]
        if part:
            err = np.array([r["dband"] - r["oband"] for r in part])
            cut = 100 * np.mean([r["reason"] != "linked" for r in part])
            print(f"  {max(lo, -0.3):4.1f} ~ {min(hi, 1.5):3.1f}: {len(part):3d}개  치우침 {np.median(err):+.2f}  끊김 {cut:3.0f}%")

    with_floor = [r for r in rows if r.get("fgap") is not None]
    if with_floor:
        low_f = np.array([r["fgap"] for r in with_floor if r["reason"] == "low_score"])
        ok_f = np.array([r["fgap"] for r in with_floor if r["reason"] == "linked"])
        print(f"\n바닥 위 거리(두 카메라가 본 발, cm): 짝 있음 {len(ok_f)}조각 중앙 {np.median(ok_f):.0f}"
              + (f" · 위치가 안 맞은 {len(low_f)}조각 중앙 {np.median(low_f):.0f}" if len(low_f) else ""))
    print("\n짝마다 (조각 수 · 짝 있음 · 끊긴 횟수)")
    for pair in dict.fromkeys(r["pair"] for r in rows):
        one = Recorder()
        one.rows = [r for r in rows if r["pair"] == pair]
        print(f"  {pair:6s} {len(one.rows):3d}  {one.linked_pct():3.0f}%  끊김 {one.breaks()[0]}")


# ---------------------------------------------------------------------- fit: 어떤 값이 위치를 잘 맞히나

FEATURES = ("cx", "x1", "x2", "y1", "hipx", "hipy", "shx", "shy", "headx", "heady")


def collect_samples(pattern: str = "N*", skip: tuple[str, ...] = ("N9",)) -> dict[str, np.ndarray]:
    """한 명만 보이는 장면에서 (전체 카메라 띠 위치, 선반 카메라 값들). 열 순서 = ("band",) + FEATURES. N9 는 두 명이라 뺀다."""
    table = SyncTable.load(de.LABELS / "sync.csv")
    floor = FloorMap.load(de.ZONES / "floor.yaml")
    out: dict[str, np.ndarray] = {}
    for pair in _done_pairs(pattern):
        if pair in skip:
            continue
        log = de.PairLog(pair)
        m1, m0 = log.meta["angle1"], log.meta["angle0"]
        offset = table.offset_ms(pair) or 0.0
        times, bands = [], []
        for idx, obs in sorted(log.ids["angle1"].items()):
            obs = [o for o in obs if o.person_id > 0]
            if len(obs) == 1:
                band = floor.band_position(floor.foot_on_floor(obs[0].bbox, m1["width"], m1["height"]))
                if band is not None and -0.3 <= band <= 1.6:
                    times.append(idx * 1000.0 / m1["fps"])
                    bands.append(band)
        if not times:
            continue
        times_a = np.asarray(times)
        rows = []
        w, h = m0["width"], m0["height"]
        for idx, obs in sorted(log.ids["angle0"].items()):
            obs = [o for o in obs if o.person_id > 0]
            if len(obs) != 1:
                continue
            t = idx * 1000.0 / m0["fps"] + offset
            j = int(np.argmin(np.abs(times_a - t)))
            if abs(times_a[j] - t) > 40.0:
                continue
            o = obs[0]
            x1, y1, x2, _ = o.bbox
            keypoints = getattr(o, "keypoints", None)
            nan = (np.nan, np.nan)
            hip, shoulder, head = (_mid(keypoints, ids, w, h) or nan for ids in (HIP, SHOULDER, HEAD))
            rows.append([bands[j], (x1 + x2) / 2 / w, x1 / w, x2 / w, y1 / h, *hip, *shoulder, *head])
        if rows:
            out[pair] = np.asarray(rows)
    return out


def _design(rows: np.ndarray, feats: tuple[str, ...], degree: int) -> np.ndarray:
    cols = [np.ones(len(rows))]
    for f in feats:
        v = rows[:, 1 + FEATURES.index(f)]
        cols += [v ** d for d in range(1, degree + 1)]
    return np.column_stack(cols)


def fit_weights(samples: dict[str, np.ndarray], feats: tuple[str, ...], degree: int = 2) -> np.ndarray:
    rows = np.vstack(list(samples.values()))
    rows = rows[np.all(np.isfinite(_design(rows, feats, 1)), axis=1)]
    return np.linalg.lstsq(_design(rows, feats, degree), rows[:, 0], rcond=None)[0]


def fit() -> None:
    samples = collect_samples()
    total = sum(len(v) for v in samples.values())
    a, b = de.load_band_map()
    everything = np.vstack(list(samples.values()))
    now = np.abs(a * everything[:, 1] + b - everything[:, 0])
    print(f"정상 영상 {len(samples)}쌍, 두 화면에 한 명씩만 보인 표본 {total}개")
    print(f"지금 식 (band_map, 상자 가운데 가로 1차): 오차 중앙 {np.median(now):.3f}  {LINK_GAP} 안 {100 * (now <= LINK_GAP).mean():.0f}%")
    print("\n짝 하나씩 빼고 맞춰 그 짝으로 시험 — 오차 = |선반 카메라로 계산한 띠 위치 − 전체 카메라 띠 위치| (1.0 = 169cm)")
    print(f"  {'선반 카메라에서 쓴 값':28s} 차수  중앙    90%   {LINK_GAP} 안  쓸 수 있는 표본")
    for label, feats, degree in (("상자 가운데 가로", ("cx",), 1), ("상자 가운데 가로", ("cx",), 3),
                                 ("상자 왼쪽·오른쪽·위", ("x1", "x2", "y1"), 2), ("머리 가로·세로", ("headx", "heady"), 2),
                                 ("어깨 가로", ("shx",), 1), ("어깨 가로·세로", ("shx", "shy"), 2),
                                 ("골반 가로", ("hipx",), 1), ("골반 가로·세로", ("hipx", "hipy"), 2)):
        errors, usable = [], 0
        for held, test in samples.items():
            train = {p: v for p, v in samples.items() if p != held}
            ok = np.all(np.isfinite(_design(test, feats, 1)), axis=1)
            usable += int(ok.sum())
            if ok.any():
                errors.append(np.abs(_design(test[ok], feats, degree) @ fit_weights(train, feats, degree) - test[ok][:, 0]))
        e = np.concatenate(errors)
        print(f"  {label:28s} {degree}    {np.median(e):.3f}  {np.percentile(e, 90):.3f}  {100 * (e <= LINK_GAP).mean():4.0f}%   {100 * usable / total:4.0f}%")


# ---------------------------------------------------------------------- try: 해결 후보 견주기


def install_joint_band(order: tuple[tuple[str, tuple[str, ...]], ...]) -> None:
    """후보 '위치식 바꾸기': 선반 카메라 띠 위치를 관절(가로·세로 2차식, 정상 영상으로 맞춤)로 낸다. order 순서로 보이는 것을 쓰고, 다 안 보이면 지금 식."""
    samples = collect_samples()
    weights = {name: fit_weights(samples, feats) for name, feats in order}

    def detail_step(judge, pts_ms, identities, takes, visits, appearance=None, resolve=None, slot_changes=None):
        w = judge.detail_width
        h = w * 9 / 16
        table = {}
        for obs in identities:
            x1, y1, x2, _ = obs.bbox
            keypoints = getattr(obs, "keypoints", None)
            for name, _ in order:
                point = (x1 / w, x2 / w, y1 / h) if name == "box" else _mid(keypoints, HIP if name == "hip" else SHOULDER, w, h)
                if point is not None:
                    table[tuple(obs.bbox)] = float(weights[name] @ np.array([1.0] + [v ** d for v in point for d in (1, 2)]))
                    break
        original = judge.detail_band
        judge.detail_band = lambda bbox: table.get(tuple(bbox), original(bbox))
        try:
            return _orig_detail_step(judge, pts_ms, identities, takes, visits, appearance, resolve, slot_changes)
        finally:
            judge.detail_band = original

    jd.DualJudge.detail_step = detail_step


HIP_ONLY = (("hip", ("hipx", "hipy")),)
CASCADE = (("hip", ("hipx", "hipy")), ("shoulder", ("shx", "shy")), ("box", ("x1", "x2", "y1")))


def try_all() -> None:
    print("기록 다시 판정 — 연결 = 선반 카메라에 사람이 보인 조각 중 짝이 있는 비율, 채점 = 쌍 단위(truth.csv)")
    for label, band, sticky, fusion in (
            ("지금", None, False, {}),
            ("A1 허용 거리 0.25 → 0.5", None, False, {"max_band_gap": 0.5}),
            ("A2 최소 점수 0.5 → 0.35", None, False, {"min_score": 0.35}),
            ("A1 + A2", None, False, {"max_band_gap": 0.5, "min_score": 0.35}),
            ("B 짝 유지", None, True, {}),
            ("C 골반 위치식", HIP_ONLY, False, {}),
            ("C + B", HIP_ONLY, True, {}),
            ("C′ 골반→어깨→상자 위치식", CASCADE, False, {}),
            ("C′ + B", CASCADE, True, {})):
        jd.DualJudge.detail_step = _orig_detail_step
        if band:
            install_joint_band(band)
        rec = Recorder(sticky)
        rec.install()
        base = AppConfig.load(ROOT / "config.yaml")
        for key, value in fusion.items():
            setattr(base.fusion, key, value)
        results = _judge_all(rec, base)
        n_breaks, n_switch, _ = rec.breaks()
        wrong = [r["pair"] for r in results if not r["correct"]]
        people = sum(r["people_ok"] in (True, None) for r in results)
        print(f"  {label:26s} 연결 {rec.linked_pct():3.0f}% (N {rec.linked_pct('N'):.0f} · T {rec.linked_pct('T'):.0f} · C {rec.linked_pct('C'):.0f})  "
              f"끊김 {n_breaks:3d}  짝 바뀜 {n_switch:2d}  채점 {len(results) - len(wrong)}/{len(results)}  손님 수 {people}/{len(results)}  "
              f"틀린 쌍 {','.join(wrong)}", flush=True)


# ---------------------------------------------------------------------- 3차 촬영: 바닥 위 거리

TWO_PEOPLE_N = ("N09", "N9")
"""정상 세트지만 두 명이 나오는 영상 — '한 명 영상' 통계에서 뺀다."""


def fit3() -> None:
    """같은 사람의 두 카메라 발이 바닥 위에서 얼마나 떨어져 나오나 — max_floor_gap_cm 을 정상 영상으로 정하는 근거."""
    base = AppConfig.load(ROOT / "config.yaml")
    _apply(base, {"position_mode": "floor"})
    rec = Recorder()
    rec.install()
    _judge_all(rec, base, "N*")
    solo = [r for r in rec.rows if r["pair"] not in TWO_PEOPLE_N and r["n_over"] == 1]
    seen = [r for r in solo if r.get("n_all")]
    gaps = np.array([r["fgap"] for r in solo if r.get("fgap") is not None])
    used = sum(r.get("n_floor", 0) for r in seen)
    total = sum(r.get("n_all", 0) for r in seen)
    print(f"정상 한 명 영상 {len({r['pair'] for r in solo})}쌍, 전체 카메라에 한 명뿐인 조각 {len(solo)}개")
    print(f"  두 화면 모두 발이 보여 바닥 위 거리로 견준 표본 {100 * used / max(1, total):.0f}% (나머지는 발이 잘려 띠 위치로)")
    print("  같은 사람의 바닥 위 거리(cm): " + " · ".join(f"{q}% {np.percentile(gaps, q):.0f}" for q in (50, 75, 90, 95, 99)))
    for half in (15, 20, 30, 40, 50):
        print(f"    허용 거리 {2 * half:3d}cm (외형 없이 {half}cm 안이면 짝): 같은 사람 조각의 {100 * (gaps <= half).mean():.0f}% 가 들어옴")
    # 발목이 두 화면 모두 보인 조각과, 한쪽이라도 발이 가려 반직선으로 견준 조각을 갈라 본다 → fusion.ray_slack_cm 의 근거
    one = [(r, r["cands"][0]) for r in solo if len(r.get("cands") or []) == 1 and r["cands"][0]["fgap"] is not None]
    seen_feet = np.array([c["pgap"] for _, c in one if c["n_point"] and c["n_point"] == c["n_all"]])
    hidden = np.array([c["fgap"] for _, c in one if not c["n_point"]])
    for name, arr in (("발목이 두 화면 모두 보인 조각", seen_feet), ("발이 가려 반직선으로만 견준 조각", hidden)):
        if len(arr):
            print(f"  {name} {len(arr)}개: " + " · ".join(f"{q}% {np.percentile(arr, q):.0f}" for q in (50, 75, 90, 95, 99)))
    print("  짝마다 중앙 거리: " + " · ".join(
        f"{p} {np.median([r['fgap'] for r in solo if r['pair'] == p and r.get('fgap') is not None] or [np.nan]):.0f}"
        for p in dict.fromkeys(r["pair"] for r in solo)))

    rec2 = Recorder()
    rec2.install()
    _judge_all(rec2, base, "C*,T*," + ",".join(TWO_PEOPLE_N))
    multi = [r for r in rec2.rows if len(r["fgaps"]) >= 2]
    if multi:
        second = np.array([r["fgaps"][1] for r in multi])
        first = np.array([r["fgaps"][0] for r in multi])
        print(f"\n전체 카메라에 두 명 이상인 조각 {len(multi)}개 (복합·절도) — 가장 가까운 사람 / 그다음 사람까지의 바닥 위 거리")
        print("  가장 가까운 사람: " + " · ".join(f"{q}% {np.percentile(first, q):.0f}" for q in (50, 90)))
        print("  그다음 사람     : " + " · ".join(f"{q}% {np.percentile(second, q):.0f}" for q in (5, 10, 25, 50)))
        for half in (15, 20, 30, 40, 50):
            print(f"    허용 거리 {2 * half:3d}cm: 그다음 사람도 {half}cm 안인 조각 {100 * (second <= half).mean():.0f}% (누구인지 위치로 못 가름)")


def try3() -> None:
    print("3차 기록 다시 판정 — 연결 = 선반 카메라에 사람이 보인 조각 중 짝이 있는 비율, 채점 = 쌍 단위(truth.csv)")
    for label, sticky, fusion in (
            ("예전 (띠 위치 한 줄)", False, {"position_mode": "band"}),
            ("바닥 60cm, 가린 발 여유 0", False, {"position_mode": "floor", "max_floor_gap_cm": 60.0, "ray_slack_cm": 0.0}),
            ("바닥 40cm, 여유 25", False, {"position_mode": "floor", "max_floor_gap_cm": 40.0}),
            ("바닥 60cm, 여유 25 (기본)", False, {"position_mode": "floor", "max_floor_gap_cm": 60.0}),
            ("바닥 80cm, 여유 25", False, {"position_mode": "floor", "max_floor_gap_cm": 80.0}),
            ("띠 위치 + 짝 유지", True, {"position_mode": "band"}),
            ("바닥 60cm, 여유 25 + 짝 유지", True, {"position_mode": "floor", "max_floor_gap_cm": 60.0})):
        rec = Recorder(sticky)
        rec.install()
        base = AppConfig.load(ROOT / "config.yaml")
        _apply(base, fusion)
        results = _judge_all(rec, base)
        n_breaks, n_switch, _ = rec.breaks()
        wrong = [r["pair"] for r in results if not r["correct"]]
        people = sum(r["people_ok"] in (True, None) for r in results)
        print(f"  {label:22s} 연결 {rec.linked_pct():3.0f}% (N {rec.linked_pct('N'):.0f} · T {rec.linked_pct('T'):.0f} · C {rec.linked_pct('C'):.0f})  "
              f"끊김 {n_breaks:3d}  짝 바뀜 {n_switch:2d}  채점 {len(results) - len(wrong)}/{len(results)}  손님 수 {people}/{len(results)}  "
              f"틀린 쌍 {','.join(wrong)}", flush=True)


REF_NEAR_CM, REF_FAR_CM = 15.0, 30.0
"""'맞는 짝'을 확실히 아는 조각의 조건: 두 화면 모두 발목이 보인 거리로 한 명이 15cm 안, 나머지 후보는 모두 30cm 밖."""


def sure_pairs(rows: list[dict]) -> dict[tuple, int]:
    """전체 카메라에 후보가 둘 이상인 조각 중 '맞는 짝'이 확실한 것: (쌍, 선반 카메라 사람, 조각 시작) → 전체 카메라 사람.

    발목이 두 화면 모두 보인 표본의 거리(같은 사람이면 중앙 4cm — floor_check feet)로 한 명만 15cm 안이고,
    나머지 후보는 바닥 위 거리(발이 가렸으면 반직선까지 = 실제 거리의 하한)가 30cm 이상일 때만 센다."""
    sure = {}
    for r in rows:
        live = [c for c in r.get("cands") or [] if c["n_all"]]
        if len(live) < 2:
            continue
        near = [c for c in live if c["pgap"] is not None and c["n_point"] >= 2 and c["pgap"] <= REF_NEAR_CM]
        if len(near) != 1:
            continue
        if all(c["fgap"] is not None and c["fgap"] >= REF_FAR_CM for c in live if c["o"] != near[0]["o"]):
            sure[(r["pair"], r["d"], r["start"])] = near[0]["o"]
    return sure


def carried_pairs(rows: list[dict], sure: dict[tuple, int]) -> dict[tuple, int]:
    """확실하지 않은 조각(발이 가렸거나 두 사람이 붙어 있음)의 기준: 같은 선반 카메라 사람의 가장 가까운 '확실한 조각'에서 맺힌
    전체 카메라 사람이 지금도 보이면 그 사람. 추적 번호가 안 바뀌었다는 데 기대므로, 번호가 서로 바뀐 영상에서는 틀릴 수 있다."""
    by: dict[tuple, list[tuple[float, int]]] = collections.defaultdict(list)
    for (pair, d, start), o in sure.items():
        by[(pair, d)].append((start, o))
    carried = {}
    for r in rows:
        key = (r["pair"], r["d"], r["start"])
        marks = by.get((r["pair"], r["d"]))
        if key in sure or not marks or sum(1 for c in r.get("cands") or [] if c["n_all"]) < 2:
            continue
        _, o = min(marks, key=lambda m: abs(m[0] - r["start"]))
        if o in r["overs"]:
            carried[key] = o
    return carried


def score_links(rows: list[dict], ref: dict[tuple, int]) -> tuple[int, int, int]:
    """(기준과 같은 짝, 다른 사람과 맺힘, 짝 없음)."""
    right = wrong = none = 0
    for r in rows:
        o = ref.get((r["pair"], r["d"], r["start"]))
        if o is None:
            continue
        if r["o"] is None:
            none += 1
        elif r["o"] == o:
            right += 1
        else:
            wrong += 1
    return right, wrong, none


def real_switches(rows: list[dict]) -> tuple[int, int]:
    """(직전 짝이 아직 보이는데 다른 사람으로 바뀐 횟수, 직전 짝이 사라져 새 사람과 맺힌 횟수)."""
    by: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[(r["pair"], r["d"])].append(r)
    stay = gone = 0
    for rs in by.values():
        last = None
        for r in sorted(rs, key=lambda r: r["start"]):
            if r["o"] is None:
                continue
            if last is not None and last != r["o"]:
                if last in r["overs"]:
                    stay += 1
                else:
                    gone += 1
            last = r["o"]
    return stay, gone


def check3() -> None:
    """두 명 이상 장면에서 맺은 짝이 '맞는 사람'인가. 기준은 발목이 두 화면 모두 보인 조각(sure_pairs)과 그걸 이어 붙인 조각(carried_pairs)."""
    def run(sticky: bool, fusion: dict) -> Recorder:
        rec = Recorder(sticky)
        rec.install()
        base = AppConfig.load(ROOT / "config.yaml")
        _apply(base, fusion)
        _judge_all(rec, base)
        return rec

    floor60 = run(False, {"position_mode": "floor", "max_floor_gap_cm": 60.0, "ray_slack_cm": 0.0})
    multi =[r for r in floor60.rows if sum(1 for c in r["cands"] if c["n_all"]) >= 2]
    sure = sure_pairs(floor60.rows)
    carried = carried_pairs(floor60.rows, sure)
    print(f"전체 카메라에 후보가 둘 이상인 조각 {len(multi)}개 ({len({r['pair'] for r in multi})}쌍)")
    print(f"  맞는 짝이 확실한 조각 {len(sure)}개 (발목이 두 화면 모두 보이고 한 명만 {REF_NEAR_CM:.0f}cm 안, 나머지는 {REF_FAR_CM:.0f}cm 밖)")
    print(f"  확실한 조각에서 이어 붙여 기준을 준 조각 {len(carried)}개 (발 가림·두 사람이 붙음) · 기준 없음 {len(multi) - len(sure) - len(carried)}개")
    print("  * 확실한 조각은 바닥 방식과 같은 정보로 고른 것이라 바닥 방식에 유리하다 — 바닥 방식의 시험은 '이어 붙인 조각' 쪽이다.")
    print(f"\n  {'방식':22s} {'확실한 조각: 맞음/다른 사람/없음':30s} {'이어 붙인 조각: 맞음/다른 사람/없음':32s} 짝 바뀜(직전 짝 있음/사라짐)")
    for label, sticky, fusion in (
            ("예전 (띠 위치 한 줄)", False, {"position_mode": "band"}),
            ("바닥 60cm, 가린 발 여유 0", False, None),
            ("바닥 40cm, 여유 25", False, {"position_mode": "floor", "max_floor_gap_cm": 40.0}),
            ("바닥 60cm, 여유 15", False, {"position_mode": "floor", "max_floor_gap_cm": 60.0, "ray_slack_cm": 15.0}),
            ("바닥 60cm, 여유 25 (기본)", False, {"position_mode": "floor", "max_floor_gap_cm": 60.0}),
            ("바닥 60cm, 여유 35", False, {"position_mode": "floor", "max_floor_gap_cm": 60.0, "ray_slack_cm": 35.0}),
            ("바닥 80cm, 여유 25", False, {"position_mode": "floor", "max_floor_gap_cm": 80.0}),
            ("바닥 60cm, 여유 25 + 짝 유지", True, {"position_mode": "floor", "max_floor_gap_cm": 60.0})):
        rec = floor60 if fusion is None else run(sticky, fusion)
        a, b = score_links(rec.rows, sure), score_links(rec.rows, carried)
        stay, gone = real_switches(rec.rows)
        print(f"  {label:22s} {a[0]:4d} / {a[1]:3d} / {a[2]:3d} ({100 * a[0] / max(1, sum(a)):3.0f}%)        "
              f"{b[0]:4d} / {b[1]:3d} / {b[2]:3d} ({100 * b[0] / max(1, sum(b)):3.0f}%)          {stay:3d} / {gone:3d}", flush=True)
        n_breaks, _, _ = rec.breaks()
        print(f"      연결 {rec.linked_pct():.1f}% · 끊김 {n_breaks}", flush=True)
        both = {**carried, **sure}
        bad = collections.Counter(r["pair"] for r in rec.rows
                                  if r["o"] is not None and both.get((r["pair"], r["d"], r["start"]), r["o"]) != r["o"])
        if bad:
            print("      다른 사람과 맺힌 조각이 있는 쌍: " + " · ".join(f"{p} {n}" for p, n in sorted(bad.items())))


def gate() -> None:
    """손님으로 센 신원만 짝 후보로 삼으면(`fusion.link_customers_only`) 무엇이 달라지나 — 끔/켬을 같은 기록으로 견준다 (후속 31).

    손님 여부는 그 쌍을 끝까지 판정한 뒤의 손님 장부(seen_inside)로 센다. 2·3차 모두 된다."""
    seen: dict[str, dict[str, set[int]]] = {}
    orig_finish = jd.DualJudge.finish

    def run(on: bool) -> tuple[Recorder, list[dict]]:
        rec = Recorder()
        rec.install()
        key = "on" if on else "off"

        def finish(judge, frame, resolve=None):
            done = orig_finish(judge, frame, resolve)
            seen.setdefault(rec.pair, {})[key] = {pid for pid, c in judge.risk.book.customers.items() if c.seen_inside}
            return done

        jd.DualJudge.finish = finish
        try:
            base = AppConfig.load(ROOT / "config.yaml")
            _apply(base, {"link_customers_only": on})
            return rec, _judge_all(rec, base)
        finally:
            jd.DualJudge.finish = orig_finish

    (off, res_off), (on, res_on) = run(False), run(True)
    people = {r["pair"]: r["people"] for r in res_off}
    solo = {p for p, n in people.items() if p.startswith("N") and n == 1}
    print(f"{de.SHOOT}차 기록 — 조각 = 선반 카메라 사람 × 0.5초. 정상 한 명 영상 {len(solo)}쌍: {' '.join(sorted(solo))}")
    print(f"  {'':18s} {'조각':>5s} {'짝 있음':>7s} {'끊김':>4s} {'짝 바뀜':>6s} {'손님 아닌 번호와 맺힘':>12s}   채점")
    for label, rec, results, key in (("끔 (지금까지)", off, res_off, "off"), ("켬 (손님만 후보)", on, res_on, "on")):
        for name, keep in (("정상 한 명", lambda p: p in solo), ("전체", lambda p: True)):
            part = Recorder()
            part.rows = [r for r in rec.rows if keep(r["pair"])]
            ghost = sum(1 for r in part.rows if r["o"] is not None and r["o"] not in seen[r["pair"]][key])
            n_breaks, n_switch, _ = part.breaks()
            scored = [r for r in results if keep(r["pair"])]
            wrong = [r["pair"] for r in scored if not r["correct"]]
            print(f"  {label:14s} {name:6s} {len(part.rows):5d} {part.linked_pct():6.1f}% {n_breaks:4d} {n_switch:6d} {ghost:12d}   "
                  f"{len(scored) - len(wrong)}/{len(scored)}" + (f"  틀린 쌍 {','.join(wrong)}" if name == "전체" else ""), flush=True)
    before = {(r["pair"], r["d"], r["start"]): r for r in off.rows}
    changed = [(before[k], r) for r in on.rows for k in [(r["pair"], r["d"], r["start"])] if k in before and before[k]["o"] != r["o"]]
    kinds = collections.Counter()
    print(f"\n달라진 조각 {len(changed)}개")
    for a, b in changed:
        cust = seen[a["pair"]]["off"]
        if a["o"] is not None and a["o"] not in cust:
            kind = "가짜 번호 → 손님" if b["o"] is not None else "가짜 번호 → 짝 없음"
        elif b["o"] is None:
            kind = "손님 → 짝 없음 (아직 손님으로 안 세어짐)"
        elif a["o"] is None:
            kind = "짝 없음 → 손님"
        else:
            kind = "손님 → 다른 손님"
        kinds[kind] += 1
        print(f"  {a['pair']:6s} 선반 {a['d']:2d}  {a['start'] / 1000:6.2f}초  {a['o']} → {b['o']}   {kind}")
    for kind, n in kinds.most_common():
        print(f"  {kind}: {n}")
    verdicts = {r["pair"]: r for r in res_off}
    moved = [r["pair"] for r in res_on if (r["verdicts"], r["correct"]) != (verdicts[r["pair"]]["verdicts"], verdicts[r["pair"]]["correct"])]
    print("판정이 달라진 쌍: " + (", ".join(moved) if moved else "없음"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("why", "fit", "try", "check", "gate"))
    ap.add_argument("--shoot", type=int, choices=(2, 3), default=2, help="촬영 회차 (기본 2). 3 = 선반 카메라에도 바닥 지도")
    ap.add_argument("--set", action="append", default=[], metavar="절.키=값", help="why 에서 설정을 바꿔 본다 (예: fusion.position_mode=floor)")
    args = ap.parse_args()
    de.set_profile(args.shoot)
    _BASE_SET[:] = args.set
    de.OVERRIDES[:] = args.set
    if args.what == "check" and args.shoot != 3:
        ap.error("check 는 --shoot 3 에서만 (두 카메라 모두 바닥 지도가 있어야 기준을 만들 수 있다)")
    if args.shoot == 3:
        {"why": why, "fit": fit3, "try": try3, "check": check3, "gate": gate}[args.what]()
    else:
        {"why": why, "fit": fit, "try": try_all, "gate": gate}[args.what]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
