"""두 화면 짝이 왜 끊기나 — 저장된 기록만 다시 돌려서 잰다 (docs/multiview-progress.md 후속 13-1, 한계 B-14).

    python scripts/link_diagnose.py why     # 0.5초 조각마다 짝이 안 맺힌 이유를 나눠 센다
    python scripts/link_diagnose.py fit     # 선반 카메라의 어떤 값이 '선반 앞 띠 위치'를 잘 맞히나 (정상 영상, 짝 하나씩 빼고 시험)
    python scripts/link_diagnose.py try     # 해결 후보를 실행 중에만 바꿔 끼워 38쌍을 다시 판정해 견준다

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
            overview = [pid for pid, t in linker._overview.items() if t.last_ms >= start]
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

    def _candidates(self, linker, d: int, overview: list[int], start: float, end: float) -> dict:
        tol = linker.config.pair_tolerance_seconds * 1000.0
        dband, _ = self._band(linker._detail[d], start, end)
        cands = []
        for o in overview:
            pos = linker._position_score(linker._detail[d], linker._overview[o], start, end)
            app = linker._appearance_score(linker._detail[d], linker._overview[o])
            oband, seen = self._band(linker._overview[o], start, end, tol)
            cands.append({"o": o, "pos": pos, "app": app, "score": linker._combine(pos, app), "oband": oband, "seen": seen})
        return {"dband": dband, "cands": cands}

    def _row(self, linker, d: int, before: dict, link, overview: list[int], prev: int | None, start: float) -> dict:
        row = {"pair": self.pair, "d": d, "start": start, "dband": before["dband"], "n_over": len(overview), "o": None}
        cands = before["cands"]
        if link is not None:
            oband = next((c["oband"] for c in cands if c["o"] == link.overview_id), None)
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
        row.update(pos=best["pos"], app=best["app"], oband=best["oband"])
        return {**row, "reason": "low_score" if score < linker.config.min_score else "taken"}

    @staticmethod
    def _keep(linker, made: list, detail: list[int], overview: list[int], prev: dict, start: float, end: float) -> None:
        """후보 '짝 유지': 직전 짝이 아직 보이고, 다른 사람이 그 짝을 가져가지 않았고, 띠 위치가 허용 거리의 두 배 안이면 잇는다."""
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
            if pos is None or a is None or b is None or abs(a - b) > 2 * linker.config.max_band_gap:
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


def _judge_all(recorder: Recorder, base: AppConfig) -> list[dict]:
    table = SyncTable.load(de.LABELS / "sync.csv")
    band, truth = de.load_band_map(), de.load_truth()
    results = []
    for pair in _done_pairs():
        recorder.pair = pair
        results.append(de.judge_pair(pair, base, table.offset_ms(pair) or 0.0, band, truth=truth))
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
    print(f"중간에 끊겼다 다시 이어진 횟수 {n_breaks} (길이 중앙 {np.median(lengths):.1f}초, 최장 {max(lengths):.1f}초) · 짝이 다른 사람으로 바뀐 횟수 {n_switch}")
    print(f"짝 있음 중 '후보 한 명이면 외형으로 잇기' 덕인 것: {sum(1 for r in rows if r.get('rescued'))}")

    low = [r for r in rows if r["reason"] == "low_score"]
    pos = np.array([r["pos"] for r in low])
    app = np.array([np.nan if r["app"] is None else r["app"] for r in low])
    gap = np.array([abs(r["dband"] - r["oband"]) for r in low if r["dband"] is not None and r["oband"] is not None])
    print(f"\n위치가 안 맞은 조각 {len(low)}개: 위치 점수 중앙 {np.median(pos):.2f} (0 인 것 {int((pos == 0).sum())}), "
          f"외형 중앙 {np.nanmedian(app):.2f}, 띠 위치 차이 25·50·75% = " + " · ".join(f"{v:.2f}" for v in np.percentile(gap, [25, 50, 75])))

    solo = [r for r in rows if r["pair"].startswith("N") and r["n_over"] == 1 and r["dband"] is not None and r.get("oband") is not None]
    print(f"\n정상 영상에서 전체 카메라에 한 명뿐인 조각 {len(solo)}개 — 선반 카메라가 본 띠 위치 구간별 (선반 − 전체)")
    for lo, hi in ((-9, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 9)):
        part = [r for r in solo if lo <= r["dband"] < hi]
        if part:
            err = np.array([r["dband"] - r["oband"] for r in part])
            cut = 100 * np.mean([r["reason"] != "linked" for r in part])
            print(f"  {max(lo, -0.3):4.1f} ~ {min(hi, 1.5):3.1f}: {len(part):3d}개  치우침 {np.median(err):+.2f}  끊김 {cut:3.0f}%")

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("why", "fit", "try"))
    args = ap.parse_args()
    {"why": why, "fit": fit, "try": try_all}[args.what]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
