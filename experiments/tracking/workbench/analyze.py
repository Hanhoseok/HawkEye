"""보기 데이터 → 질문별 요약.

분석 화면은 네 가지 질문(렌즈)으로 나뉜다. 여기서 렌즈마다 다음을 만든다.

- facts : 한눈에 볼 사실. level 은 ok / bad / warn / info. 정답과 견줄 수 있는 것만 ok·bad 를 붙이고,
          정답이 없는 것(두 화면 짝, 집은 시각)은 info·warn 으로만 말한다 — 눈으로 확인하라는 뜻.
- rows  : 타임라인 줄. segs(구간)와 marks(순간). who 는 전체 카메라 손님 번호(색을 정함), tone 은 뜻(ok·bad·warn·take·put·quiet).
- log   : 시간 순서 기록 (누르면 그 시각으로).
facts 의 knob 은 그 사실을 좌우하는 판정 값의 이름(workbench/tune.py KNOBS) — 화면이 '이 기준 바꿔 보기'로 잇는다.

영상·모델 라이브러리를 쓰지 않는다 — 기록만으로 계산하므로 시험할 수 있다 (tests/test_workbench_analyze.py).
입력 data 는 scripts/review_export.py pair_data 가 만든 모양, result 는 scripts/dual_eval.py judge_pair 의 채점 결과.
"""

from __future__ import annotations

from statistics import median

LENSES = ("track", "link", "take", "verdict")
LENS_NAME = {"track": "사람 추적", "link": "두 화면 같은 사람", "take": "물건 집기", "verdict": "결제·판정"}
SET_NAME = {"normal": "정상", "theft": "절도", "complex": "복합"}
HEADLINE = {
    "hit": "절도를 잡음", "miss": "절도인데 놓침", "false_alarm": "정상인데 경보",
    "pass": "정상 통과", "no_verdict": "판정 없음",
}
LEVEL_NAME = {"HIGH_RISK": "경보", "REVIEW": "확인 필요", "CLEAR": "통과", "WARNING": "출구 경고"}
TONE = {"HIGH_RISK": "bad", "REVIEW": "warn", "CLEAR": "ok"}
NO_LINK_TRUTH = "짝이 맞는지는 정답이 없어 채점하지 못함 — 두 화면의 같은 색 상자가 같은 사람인지 눈으로 확인"
NO_TAKE_TRUTH = "누가 언제 몇 개 집었는지 정답이 아직 없어 개수는 채점하지 못함"
ZERO_TAKEN = "정답은 물건을 가져간 영상인데 프로그램은 가져간 물건을 0개로 셈"
GAP_MS = 700.0
"""이보다 오래 안 보이면 타임라인에서 구간을 끊는다."""


def sec(ms: float) -> str:
    return f"{ms / 1000.0:.1f}초"


def frame_ms(track: dict, key) -> float:
    """기록의 장면(track["frames"] 의 열쇠) → 그 카메라 시각(ms, 시차 전).
    3차처럼 장면을 빠뜨린 영상은 실제 시각 표(track["ms"])가 실려 있다. 없으면 번호 ÷ fps."""
    times = track.get("ms")
    if times is not None:
        return times[key]
    return int(key) * 1000.0 / track["fps"]


def presence(track: dict, offset_ms: float = 0.0) -> dict[int, list[list[float]]]:
    """신원별로 화면에 보인 구간 [[시작, 끝], ...] (공통 시각 ms). 나중에 합쳐진 신원은 합친 뒤 번호로."""
    step = track["stride"] * 1000.0 / track["fps"]
    out: dict[int, list[list[float]]] = {}
    for key in sorted(track["frames"], key=int):
        ms = frame_ms(track, key) + offset_ms
        for row in track["frames"][key]:
            rid = row[1]
            if rid <= 0:
                continue
            segs = out.setdefault(rid, [])
            if segs and ms - segs[-1][1] <= GAP_MS:
                segs[-1][1] = ms + step
            else:
                segs.append([ms, ms + step])
    return {rid: [[round(s), round(e)] for s, e in segs] for rid, segs in out.items()}


def _total(segs: list[list[float]]) -> float:
    return sum(e - s for s, e in segs)


def _covered(segs: list[list[float]], cover: list[list[float]]) -> float:
    """segs 중 cover 와 겹치는 시간."""
    total = 0.0
    for s, e in segs:
        for cs, ce in cover:
            total += max(0.0, min(e, ce) - max(s, cs))
    return total


def _merge(segs: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for s, e in sorted(segs):
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _who(pid: int, customers: set[int]) -> str:
    return f"손님 {pid}" if pid in customers else f"손님 아님 #{pid}"


def _duration(data: dict) -> float:
    t1, t0 = data["tracks"]["angle1"], data["tracks"]["angle0"]

    def end(t: dict) -> float:
        return t.get("end_ms", t["frames_total"] * 1000.0 / t["fps"])
    return round(max(end(t1), end(t0) + data["offset_ms"]))


# ---------------------------------------------------------------------- 렌즈별

def lens_track(data: dict, result: dict, customers: set[int]) -> dict:
    over = presence(data["tracks"]["angle1"])
    detail = presence(data["tracks"]["angle0"], data["offset_ms"])
    facts, rows, log = [], [], []
    people, expected = result["people"], result.get("people_expected")
    if expected is None:
        facts.append({"level": "info", "text": f"손님 {people}명으로 셈 (정답표에 손님 수 없음)"})
    else:
        facts.append({"level": "ok" if people == expected else "bad", "knob": None if people == expected else "risk.inside_seconds",
                      "text": f"손님 수: 프로그램 {people}명 · 정답 {expected}명"})
    extra = [pid for pid in over if pid not in customers]
    if extra:
        facts.append({"level": "info",
                      "text": f"전체 카메라에 생긴 신원 {len(over)}개 중 {len(extra)}개는 손님이 아닌 것으로 걸러짐 "
                              "(매장 안 바닥에 충분히 오래 서지 않음 — 외투·문밖 행인·입장 순간 조각)"})
    marks: dict[int, list[dict]] = {}
    for b in data["births"]["angle1"]:
        edge = " · 화면 끝에서 등장" if b.get("edge") else ""
        marks.setdefault(b["pid"], []).append({"ms": b["ms"], "tone": "quiet", "tip": f"새 신원 #{b['pid']}{edge}"})
    for m in data["merges"]["angle1"]:
        tip = f"신원 #{m['from']} 을 #{m['into']} 과 같은 사람으로 합침"
        marks.setdefault(m["into"], []).append({"ms": m["ms"], "tone": "warn", "tip": tip})
        log.append({"ms": m["ms"], "level": "info", "text": tip})
    if data["merges"]["angle1"]:
        facts.append({"level": "info", "text": f"한 사람이 여러 신원으로 쪼개졌다가 나중에 합쳐진 것 {len(data['merges']['angle1'])}건"})
    for r in data["revivals"]["angle1"]:
        if r["before_pid"] > 0 and r["after_pid"] > 0 and r["before_pid"] != r["after_pid"]:
            text = (f"추적 번호 t{r['track']} 가 {r['gap_ms'] / 1000:.1f}초 끊겼다 되살아나 "
                    f"{_who(r['before_pid'], customers)} → {_who(r['after_pid'], customers)} 로 다시 판정됨")
            facts.append({"level": "warn", "text": text, "ms": r["ms"]})
            log.append({"ms": r["ms"], "level": "warn", "text": text})
    for pid in sorted(over, key=lambda p: (p not in customers, over[p][0][0])):
        inside = pid in customers
        rows.append({"label": _who(pid, customers), "group": "전체 카메라", "who": pid if inside else None,
                     "segs": [{"s": s, "e": e, "tone": None if inside else "quiet"} for s, e in over[pid]],
                     "marks": marks.get(pid, [])})
    for pid in sorted(detail, key=lambda p: detail[p][0][0]):
        rows.append({"label": f"선반 화면 {pid}", "group": "선반 카메라", "who": None,
                     "segs": [{"s": s, "e": e, "tone": "plain"} for s, e in detail[pid]], "marks": []})
    for pid in sorted(over, key=lambda p: over[p][0][0]):
        log.append({"ms": over[pid][0][0], "level": "info" if pid in customers else "quiet",
                    "text": f"{_who(pid, customers)} 처음 보임"})
    status = "info" if expected is None else ("ok" if people == expected else "bad")
    return {"status": status, "facts": facts, "rows": rows, "log": sorted(log, key=lambda r: r["ms"])}


def lens_link(data: dict, customers: set[int]) -> dict:
    detail = presence(data["tracks"]["angle0"], data["offset_ms"])
    by_detail: dict[int, list[dict]] = {}
    for link in sorted(data["links"], key=lambda r: r["start"]):
        by_detail.setdefault(link["detail"], []).append(link)
    facts, rows, log = [], [], []
    visible = sum(_total(segs) for segs in detail.values())
    linked = sum(_covered(detail.get(d, []), _merge([[l["start"], l["end"]] for l in links]))
                 for d, links in by_detail.items())
    pct = round(100.0 * linked / visible) if visible else None
    if pct is None:
        facts.append({"level": "info", "text": "선반 카메라에 사람이 보이지 않음"})
    else:
        facts.append({"level": "info", "knob": "fusion.min_score",
                      "text": f"선반 카메라에 사람이 보인 시간의 {pct}% 에 전체 카메라 짝이 있음 "
                              f"({visible / 1000:.1f}초 중 {linked / 1000:.1f}초)"})
    for d in sorted(detail, key=lambda p: detail[p][0][0]):
        links = by_detail.get(d, [])
        segs = [{"s": s, "e": e, "tone": "ghost"} for s, e in detail[d]]
        runs: list[dict] = []
        for l in links:
            segs.append({"s": l["start"], "e": l["end"], "who": l["overview"] if l["overview"] in customers else None,
                         "tone": None if l["overview"] in customers else "warn",
                         "tip": f"선반 화면 {d} ⇄ {_who(l['overview'], customers)} · 점수 {l['score']:.2f}" + _gap_text(l)})
            if runs and runs[-1]["overview"] == l["overview"]:
                runs[-1]["end"] = l["end"]
            else:
                runs.append({"overview": l["overview"], "start": l["start"], "end": l["end"]})
        rows.append({"label": f"선반 화면 {d}", "group": "선반 카메라 사람 → 전체 카메라 짝", "who": None, "segs": segs, "marks": []})
        if not links:
            facts.append({"level": "warn", "text": f"선반 화면 {d} 번 사람은 한 번도 짝을 맺지 못함", "ms": detail[d][0][0]})
            continue
        for run in runs:
            log.append({"ms": run["start"], "level": "info" if run["overview"] in customers else "warn",
                        "text": f"선반 화면 {d} ⇄ {_who(run['overview'], customers)} ({sec(run['start'])}~{sec(run['end'])})"})
        partners = list(dict.fromkeys(run["overview"] for run in runs))
        if len(partners) > 1:
            chain = " → ".join(f"{_who(run['overview'], customers)}({sec(run['start'])})" for run in runs[:4])
            if len(runs) > 4:
                chain += f" … 모두 {len(runs) - 1}번 바뀜"
            facts.append({"level": "warn", "text": f"선반 화면 {d} 번 사람의 짝이 바뀜: {chain}", "ms": runs[1]["start"]})
        outsiders = [p for p in partners if p not in customers]
        if outsiders:
            first = next(run for run in runs if run["overview"] in outsiders)
            facts.append({"level": "warn", "ms": first["start"],
                          "text": f"선반 화면 {d} 번 사람이 손님이 아닌 신원(#{', #'.join(map(str, outsiders))})과 짝을 맺음"})
    facts += _floor_facts(data)
    if not any(f["level"] == "warn" for f in facts) and pct is not None:
        facts.append({"level": "info", "text": NO_LINK_TRUTH})
    return {"status": "warn" if any(f["level"] == "warn" for f in facts) else "info", "pct": pct,
            "facts": facts, "rows": rows, "log": sorted(log, key=lambda r: r["ms"])}


HIDDEN_SHARE = 0.5
"""조각의 표본 중 이만큼 이상이 발목 없이 잰 것이면 '발이 가린 조각'으로 말한다."""


def _gap_text(link: dict) -> str:
    """짝 하나의 바닥 위 거리 (바닥으로 맺는 설정에서만 실려 있다 — scripts/review_export.py _link_gaps)."""
    if link.get("gap") is None:
        return ""
    hidden = " (발이 가려 방향으로만 잼)" if link.get("hidden", 0) >= HIDDEN_SHARE else ""
    return f" · 바닥 위 {link['gap']:.0f}cm{hidden}"


def _floor_facts(data: dict) -> list[dict]:
    """두 화면 짝을 바닥 위 거리로 맺었을 때(3차): 맺은 짝들이 바닥에서 얼마나 가까웠나."""
    floor = data.get("floor")
    measured = [l for l in data["links"] if l.get("gap") is not None]
    if not floor or not measured:
        return []
    seen = [l["gap"] for l in measured if l.get("hidden", 0) < HIDDEN_SHARE]
    hidden = [l["gap"] for l in measured if l.get("hidden", 0) >= HIDDEN_SHARE]
    bits = []
    if seen:
        bits.append(f"발목이 보인 조각 {len(seen)}개는 중앙 {median(seen):.0f}cm")
    if hidden:
        bits.append(f"발이 가려 방향으로만 잰 조각 {len(hidden)}개는 중앙 {median(hidden):.0f}cm")
    facts = [{"level": "info", "knob": "fusion.max_floor_gap_cm",
              "text": "맺은 짝의 두 카메라 발이 바닥 위에서 떨어진 거리: " + " · ".join(bits)
                      + f" (이 거리가 {floor['max_gap']:g}cm 이면 위치 점수 0, 가린 발은 {floor['slack']:g}cm 까지 봐줌)"}]
    far = [l for l in measured if l.get("pos") is not None and l["pos"] <= 0.0]
    if far:
        worst = max(far, key=lambda l: l["gap"])
        facts.append({"level": "warn", "ms": worst["start"], "knob": "fusion.solo_appearance_min",
                      "text": f"바닥 위 위치는 맞지 않는데(위치 점수 0) 이어진 조각 {len(far)}개 — 가장 먼 것 {worst['gap']:.0f}cm. "
                              "비교할 사람이 한 명뿐이라 외형만으로 이은 것"})
    return facts


def lens_take(data: dict, result: dict, customers: set[int], expected: str) -> dict:
    facts, rows, log = [], [], []
    owner_of = {(round(e["start_ms"]), e["kind"]): e for e in data["shelf_events"]}

    def owner_tip(start: float, kind: str) -> tuple[int | None, str]:
        e = owner_of.get((round(start), kind))
        if e is None:
            near = [x for x in data["shelf_events"] if x["kind"] == kind and abs(x["start_ms"] - start) <= 60]
            e = near[0] if near else None
        if e is None or not e.get("owner") or e["owner"] <= 0:
            return None, "주인을 못 정함"
        return e["owner"], f"주인: {_who(e['owner'], customers)} ({e['how']})"

    segs = []
    for t in data["takes0"]:
        who, tip = owner_tip(t["start"], "take")
        segs.append({"s": t["start"], "e": t["end"], "who": who, "tone": None if who else "warn",
                     "tip": f"손목이 '{t['zone']}' 칸 안 {(t['end'] - t['start']) / 1000:.1f}초 · {tip}"})
        log.append({"ms": t["start"], "level": "info" if who else "warn", "text": f"손 접촉: '{t['zone']}' 칸 · {tip}"})
    rows.append({"label": "손 접촉", "group": "선반 카메라", "who": None, "segs": segs, "marks": []})

    segs = []
    for v in data["visits"]:
        who, tip = owner_tip(v["start"], "visit")
        change = v["removed"] or v["added"] or v.get("slot_change")
        tone = "take" if v["removed"] else ("put" if v["added"] else ("take" if v.get("slot_change") else "quiet"))
        what = (f"사라짐 {v['removed']}" if v["removed"] else "") + (f" 생김 {v['added']}" if v["added"] else "")
        if not what and v.get("slot_change"):
            what = f"칸 그림 변화({v['slot_change']})"
        segs.append({"s": v["start"], "e": v["end"], "who": None, "tone": tone,
                     "tip": f"선반 가림 {sec(v['start'])}~{sec(v['end'])} · {what.strip() or '변화 없음'} · {tip}"})
        log.append({"ms": v["end"], "level": "info" if change else "quiet",
                    "text": f"선반이 다시 보임: {what.strip() or '변화 없음'} · {tip}"})
    rows.append({"label": "선반 가림·비교", "group": "선반 카메라", "who": None, "segs": segs, "marks": []})

    slot = data.get("slot_changes") or []
    rows.append({"label": "칸 그림 변화", "group": "선반 카메라", "who": None, "marks": [],
                 "segs": [{"s": c["start"], "e": c["end"], "who": None, "tone": "take" if c["direction"] == "줄어듦" else "warn",
                           "tip": f"'{c['slot']}' 칸 화소 {c['ratio'] * 100:.1f}% 바뀜 · {c['direction']}"} for c in slot]})
    for c in slot:
        log.append({"ms": c["end"], "level": "info", "text": f"칸 그림: '{c['slot']}' {c['ratio'] * 100:.1f}% 바뀜 ({c['direction']})"})

    rows.append({"label": "선반 앞 머묾", "group": "전체 카메라", "marks": [], "who": None,
                 "segs": [{"s": t["start"], "e": t["end"], "who": t["pid"] if t["pid"] in customers else None,
                           "tip": f"{_who(t['pid'], customers)} 선반 앞 {(t['end'] - t['start']) / 1000:.1f}초"}
                          for t in data["takes1"]]})

    actions = [a for a in data["actions"] if a["kind"] != "LOOK"]
    rows.append({"label": "행동 판정", "group": "결론", "marks": [], "who": None,
                 "segs": [{"s": a["start"], "e": max(a["end"], a["start"] + 200), "who": a["pid"] if a["pid"] in customers else None,
                           "tone": {"TAKE": "take", "PUT": "put"}.get(a["kind"]),
                           "tip": f"{_who(a['pid'], customers)}: {a['name']}" + (f" {a['count']:+d}개" if a["count"] else "")
                                  + f" · 근거 {a['evidence']}"} for a in actions]})
    per: dict[int, dict] = {}
    for a in actions:
        p = per.setdefault(a["pid"], {"items": 0, "names": {}})
        p["items"] += a["count"]
        p["names"][a["name"]] = p["names"].get(a["name"], 0) + 1
        log.append({"ms": a["end"], "level": "info", "text": f"행동: {a['text']}"})
    total = 0
    for pid in sorted(per):
        p = per[pid]
        total += p["items"]
        kinds = ", ".join(f"{name} {n}번" for name, n in p["names"].items())
        facts.append({"level": "info", "text": f"{_who(pid, customers)}: 가져간 물건 {p['items']}개로 셈 ({kinds})"})
    if not per:
        facts.append({"level": "info", "text": "선반에서 일어난 행동이 하나도 잡히지 않음"})
    if expected == "HIGH_RISK" and total <= 0:
        facts.insert(0, {"level": "bad", "text": ZERO_TAKEN})
    touches = [a for a in actions if a["kind"] == "TOUCH"]
    if touches:
        note = ""
        if not slot:
            note = " 기준에 못 미친 칸 그림 변화율은 기록에 남지 않아 여기서 볼 수 없음"
        facts.append({"level": "warn" if expected == "HIGH_RISK" else "info", "ms": touches[0]["start"],
                      "text": f"손은 닿았지만 선반이 그대로라고 봐서 '만지기만'이 된 것 {len(touches)}번 "
                              f"— 물건 탐지 전후 개수와 칸 그림이 모두 변화 없음.{note}"})
    blind = [a for a in actions if a["evidence"] == "동작만"]
    if blind:
        facts.append({"level": "warn", "ms": blind[0]["start"],
                      "text": f"선반 근거 없이 동작만으로 센 것 {len(blind)}번 — 실제로 집었는지 확인 못 함 (각각 1개로 셈)"})
    if result.get("unowned_takes"):
        facts.append({"level": "warn", "text": f"주인을 못 정한 손 접촉 {result['unowned_takes']}건"})
    facts.append({"level": "info", "text": NO_TAKE_TRUTH})
    status = "bad" if facts and facts[0]["level"] == "bad" else ("warn" if any(f["level"] == "warn" for f in facts) else "info")
    return {"status": status, "taken": total, "facts": facts, "rows": rows, "log": sorted(log, key=lambda r: r["ms"])}


def lens_verdict(data: dict, result: dict, customers: set[int], dwell_seconds: float | None) -> dict:
    facts, rows, log = [], [], []
    outcome, expected = result["outcome"], result["expected"]
    answer = "미결제로 나감 (경보가 나야 함)" if expected == "HIGH_RISK" else "정상 (경보가 없어야 함)"
    facts.append({"level": "ok" if result["correct"] and not result["no_verdict"] else ("warn" if result["correct"] else "bad"),
                  "text": f"정답: {answer} · 결과: {HEADLINE[outcome]}"})
    finals = [e for e in data["events"] if e["level"] != "WARNING"]
    last_by: dict[int, dict] = {}
    for e in finals:
        last_by[e["pid"]] = e
    for e in finals:
        tone = TONE.get(e["level"], "info")
        replaced = last_by[e["pid"]] is not e
        text = f"{_who(e['pid'], customers)} → {LEVEL_NAME[e['level']]}: {e['reason']}"
        if replaced:
            text += f" (뒤에 다시 판정받아 '{LEVEL_NAME[last_by[e['pid']]['level']]}'(으)로 바뀜)"
        facts.append({"level": tone, "text": text, "ms": e["ms"]})
        log.append({"ms": e["ms"], "level": tone, "text": text})
    if result["no_verdict"]:
        tail = " 정상 영상이라 채점에서는 맞음에 들어가지만 통과까지 확인된 것은 아님." if expected != "HIGH_RISK" else ""
        facts.append({"level": "warn", "knob": "risk.exit_confirm_seconds",
                      "text": "판정이 하나도 없음 — 영상이 끝날 때까지 손님이 나갔다고 확정되지 않음 "
                              f"(끝 장면에 아직 화면에 보임).{tail}"})
    if result["correct"] != result["correct_last"]:
        facts.append({"level": "info", "text": "손님별 마지막 판정만 보면 " + ("맞음" if result["correct_last"] else "틀림")})
    paid = [p for p in data["payments"] if p["method"] in ("dwell", "checkout", "given")]
    cancels = [p for p in data["payments"] if p["method"] == "cancel"]
    unmatched = [p for p in data["payments"] if p["method"] == "unmatched"]
    marks = []
    for p in data["payments"]:
        if p["method"] == "cancel":
            tip, tone = f"결제 취소 — {_who(p['pid'], customers)} 가 계산 뒤 선반으로 돌아감", "warn"
        elif p["method"] == "unmatched":
            tip, tone = f"결제 {p['items']}개 — 계산대에 선 손님을 못 찾음", "bad"
        else:
            how = "계산대에 서 있음" if p["method"] == "dwell" else "결제 기록"
            tip, tone = f"결제 {p['items']}개 — {_who(p['pid'], customers)} ({how})", "ok"
        marks.append({"ms": p["ms"], "tone": tone, "tip": tip, "who": p["pid"] if p["pid"] in customers else None})
        log.append({"ms": p["ms"], "level": "info" if tone == "ok" else "warn", "text": tip})
    rows.append({"label": "결제", "group": "전체 카메라", "who": None, "segs": [], "marks": marks})
    warn_marks = [{"ms": e["ms"], "tone": "warn", "tip": f"{_who(e['pid'], customers)} 출구에 다가섬 — 이대로 나가면 경보",
                   "who": e["pid"]} for e in data["events"] if e["level"] == "WARNING"]
    for m in warn_marks:
        log.append({"ms": m["ms"], "level": "warn", "text": m["tip"]})
    rows.append({"label": "출구 경고", "group": "전체 카메라", "who": None, "segs": [], "marks": warn_marks})
    rows.append({"label": "최종 판정", "group": "결론", "who": None, "segs": [],
                 "marks": [{"ms": e["ms"], "tone": TONE.get(e["level"], "info"),
                            "tip": f"{_who(e['pid'], customers)} → {LEVEL_NAME[e['level']]}", "who": None} for e in finals]})
    bits = [f"결제 {len(paid)}건"] + ([f"취소 {len(cancels)}건"] if cancels else []) + ([f"연결 못 한 결제 {len(unmatched)}건"] if unmatched else [])
    basis = (f"결제 기록이 없어 '계산대에 {dwell_seconds:g}초 이상 서 있으면 결제'로 봄 — 실제 결제가 아님"
             if dwell_seconds else "결제 기록 사용")
    facts.append({"level": "info", "knob": "risk.dwell_pay_seconds" if dwell_seconds else None,
                  "text": f"{' · '.join(bits)}. {basis}"})
    status = "bad" if not result["correct"] else ("warn" if result["no_verdict"] else "ok")
    return {"status": status, "facts": facts, "rows": rows, "log": sorted(log, key=lambda r: r["ms"])}


# ---------------------------------------------------------------------- 사람이 적은 정답으로 채점

def locate(track: dict, point: list[float], ms: float, offset_ms: float = 0.0, window_ms: float = 700.0) -> int | None:
    """그 시각 화면의 그 점에 있던 사람(나중에 합쳐진 신원은 합친 뒤 번호). 점을 품은 상자 중 가장 작은 것.

    정답의 '누가'는 화면 위의 점이다 (workbench/labels.py). 그 시각 가장 가까운 장면부터 window_ms 안에서 찾는다 —
    사람이 잠깐 안 잡힌 장면에 찍어도 앞뒤 장면에서 찾게.
    """
    x, y = point[0] * track["w"], point[1] * track["h"]
    near = sorted(track["frames"], key=lambda k: abs(frame_ms(track, k) + offset_ms - ms))
    for key in near:
        if abs(frame_ms(track, key) + offset_ms - ms) > window_ms:
            break
        inside = [r for r in track["frames"][key] if r[1] > 0 and r[3] <= x <= r[5] and r[4] <= y <= r[6]]
        if inside:
            return min(inside, key=lambda r: (r[5] - r[3]) * (r[6] - r[4]))[1]
    return None


def score_labels(data: dict, customers: set[int]) -> dict | None:
    """사람이 적은 정답(data["labels"])과 프로그램의 판단을 견준다. 정답이 없으면 None.

    take    : 손님마다 '가져간 개수'(집음 − 놓음)가 정답과 같은가. takes_done 일 때만.
    link    : 정답으로 찍은 두 화면 짝마다, 그 시각 프로그램의 짝이 같은 사람인가.
    culprit : 경보(HIGH_RISK)가 정답의 사람에게 났는가.
    """
    labels = data.get("labels")
    if not labels or not (labels["takes"] or labels["links"] or labels["culprits"] or labels["takes_done"]):
        return None
    over, detail, offset = data["tracks"]["angle1"], data["tracks"]["angle0"], data["offset_ms"]
    out = {"takes_done": bool(labels["takes_done"]), "take": None, "link": None, "culprit": None, "events": []}

    actions = [a for a in data["actions"] if a["pid"] in customers]
    for t in labels["takes"]:
        ms = t["time_sec"] * 1000.0
        rid = locate(over, t["who"], ms)
        caught = any(a["pid"] == rid and a["count"] * (1 if t["kind"] == "take" else -1) > 0
                     and a["start"] - 1500 <= ms <= a["end"] + 1500 for a in actions)
        out["events"].append({"ms": ms, "kind": t["kind"], "count": t["count"], "slot": t["slot"],
                              "who": rid if rid in customers else None, "caught": caught})
    if labels["takes_done"]:
        truth: dict[int, int] = {}
        lost = [e for e in out["events"] if e["who"] is None]
        for e in out["events"]:
            if e["who"] is not None:
                truth[e["who"]] = truth.get(e["who"], 0) + e["count"] * (1 if e["kind"] == "take" else -1)
        model: dict[int, int] = {}
        for a in actions:
            model[a["pid"]] = model.get(a["pid"], 0) + a["count"]
        people = sorted(set(truth) | {pid for pid, n in model.items() if n != 0})
        rows = [{"who": pid, "truth": truth.get(pid, 0), "model": model.get(pid, 0),
                 "ok": truth.get(pid, 0) == model.get(pid, 0)} for pid in people]
        out["take"] = {"people": rows, "lost": [e["ms"] for e in lost], "right": sum(r["ok"] for r in rows), "n": len(rows),
                       "ok": all(r["ok"] for r in rows) and not lost,
                       "caught": sum(e["caught"] for e in out["events"]), "events": len(out["events"])}

    if labels["links"]:
        items = []
        for link in labels["links"]:
            ms = link["time_sec"] * 1000.0
            d, want = locate(detail, link["detail"], ms, offset), locate(over, link["overview"], ms)
            got = None
            for seg in data["links"]:
                if seg["detail"] == d and seg["start"] <= ms < seg["end"] + 250:
                    got = seg["overview"]
            found = d is not None and want is not None
            items.append({"ms": ms, "detail": d, "want": want, "got": got, "found": found, "ok": found and got == want})
        out["link"] = {"items": items, "right": sum(i["ok"] for i in items), "n": len(items)}

    if labels["culprits"]:
        who = [locate(over, c["who"], c["time_sec"] * 1000.0) for c in labels["culprits"]]
        alarm = sorted({e["pid"] for e in data["events"] if e["level"] == "HIGH_RISK"})
        found = [w for w in who if w is not None]
        ok = None if not alarm or not found else bool(set(alarm) & set(found))
        out["culprit"] = {"who": who, "alarm": alarm, "ok": ok, "ms": labels["culprits"][0]["time_sec"] * 1000.0}
    return out


def apply_truth(lenses: dict, score: dict, customers: set[int]) -> None:
    """채점 결과를 렌즈의 요약·타임라인·상태에 얹는다 (정답이 있는 것에만 맞음·틀림이 붙는다)."""
    take, link, verdict = lenses["take"], lenses["link"], lenses["verdict"]

    if score["events"]:
        take["rows"].insert(0, {"label": "정답: 집음·놓음", "group": "사람이 적은 정답", "who": None, "segs": [], "marks": [
            {"ms": e["ms"], "tone": "take" if e["kind"] == "take" else "put",
             "tip": ("집음" if e["kind"] == "take" else "놓음") + f" {e['count']}개" + (f" · {e['slot']} 칸" if e["slot"] else "")
                    + (f" · {_who(e['who'], customers)}" if e["who"] is not None else " · 프로그램이 이 사람을 손님으로 못 봄")
                    + ("" if e["caught"] else " · 프로그램 행동과 안 맞음")} for e in score["events"]]})
    t = score["take"]
    if t is not None:
        take["facts"] = [f for f in take["facts"] if f["text"] not in (NO_TAKE_TRUTH, ZERO_TAKEN)]
        head = []
        if t["ok"]:
            head.append({"level": "ok", "text": f"가져간 개수가 정답과 같음 ({t['n']}명)" if t["n"] else "아무도 가져가지 않음 — 정답과 같음"})
        else:
            head.append({"level": "bad", "text": f"가져간 개수가 정답과 다름 — {t['n']}명 중 {t['right']}명 맞음"})
            for r in t["people"]:
                if not r["ok"]:
                    head.append({"level": "bad", "text": f"{_who(r['who'], customers)}: 정답 {r['truth']}개 · 프로그램 {r['model']}개"})
            for ms in t["lost"]:
                head.append({"level": "bad", "ms": ms, "text": "정답에 적은 사람을 프로그램이 그 시각에 손님으로 보지 못함"})
        if t["events"]:
            head.append({"level": "info", "text": f"정답의 집음·놓음 {t['events']}번 중 {t['caught']}번은 프로그램의 행동과 시각·사람이 맞음"})
        take["facts"] = head + take["facts"]
        take["status"] = "ok" if t["ok"] else "bad"
    elif score["events"]:
        take["facts"] = [f for f in take["facts"] if f["text"] != NO_TAKE_TRUTH]
        take["facts"].append({"level": "info", "text": "집음·놓음 정답을 적는 중 — '다 적었음'을 체크하면 개수를 채점함"})

    k = score["link"]
    if k is not None:
        link["rows"].insert(0, {"label": "정답: 같은 사람", "group": "사람이 적은 정답", "who": None, "segs": [], "marks": [
            {"ms": i["ms"], "tone": "ok" if i["ok"] else "bad", "tip": _link_text(i, customers)} for i in k["items"]]})
        link["facts"] = [f for f in link["facts"] if f["text"] != NO_LINK_TRUTH]
        head = [{"level": "ok" if k["right"] == k["n"] else "bad",
                 "text": f"정답으로 찍은 짝 {k['n']}개 " + ("모두 맞음" if k["right"] == k["n"] else f"중 {k['right']}개 맞음")}]
        head += [{"level": "bad", "ms": i["ms"], "text": _link_text(i, customers)} for i in k["items"] if not i["ok"]]
        link["facts"] = head + link["facts"]
        link["status"] = "ok" if k["right"] == k["n"] else "bad"

    c = score["culprit"]
    if c is not None:
        names = ", ".join(_who(w, customers) if w is not None else "못 찾음" for w in c["who"])
        if c["ok"] is None:
            fact = {"level": "warn" if None in c["who"] else "info", "ms": c["ms"],
                    "text": f"경보가 나야 하는 사람: {names}" + (" — 경보가 나지 않음" if not c["alarm"] else "")}
        elif c["ok"]:
            fact = {"level": "ok", "ms": c["ms"], "text": f"경보가 정답의 사람({names})에게 남"}
        else:
            fact = {"level": "bad", "ms": c["ms"],
                    "text": f"경보가 다른 사람에게 남 — 정답 {names} · 경보 {', '.join(_who(p, customers) for p in c['alarm'])}"}
            verdict["status"] = "bad"
        verdict["facts"].insert(1, fact)


def _link_text(item: dict, customers: set[int]) -> str:
    if not item["found"]:
        return "정답으로 찍은 사람을 프로그램이 그 시각에 찾지 못함"
    got = "짝 없음" if item["got"] is None else _who(item["got"], customers)
    return f"선반 화면 {item['detail']} 의 짝: 정답 {_who(item['want'], customers)} · 프로그램 {got}"


# ---------------------------------------------------------------------- 한 짝 전체

def short_reason(data: dict, result: dict) -> str:
    """갤러리 카드에 쓰는 한 줄 이유."""
    if result["no_verdict"]:
        return "영상 끝까지 나감이 확정되지 않음"
    finals = [e for e in data["events"] if e["level"] != "WARNING"]
    alarm = [e for e in finals if e["level"] == "HIGH_RISK"]
    pick = alarm[0] if alarm else finals[-1]
    return pick["reason"].split(" — ")[0]


def analyze(data: dict, result: dict, dwell_seconds: float | None = 1.5) -> dict:
    customers = {int(pid) for pid, c in data["customers"].items() if c["inside"]}
    expected = result["expected"]
    track = lens_track(data, result, customers)
    link = lens_link(data, customers)
    take = lens_take(data, result, customers, expected)
    verdict = lens_verdict(data, result, customers, dwell_seconds)
    lenses = {"track": track, "link": link, "take": take, "verdict": verdict}
    score = score_labels(data, customers)
    if score is not None:
        apply_truth(lenses, score, customers)
    headline = HEADLINE[result["outcome"]]
    if result["outcome"] == "miss" and result["no_verdict"]:
        headline = "절도인데 판정 없음"
    truth = data.get("truth") or {}
    return {
        "pair": data["pair"], "set": truth.get("set"), "set_name": SET_NAME.get(truth.get("set"), "?"),
        "notes": truth.get("notes", ""), "expected": expected, "outcome": result["outcome"], "headline": headline,
        "reason": short_reason(data, result), "correct": result["correct"], "correct_last": result["correct_last"],
        "no_verdict": result["no_verdict"], "verdicts": result["verdicts"],
        "people": result["people"], "people_expected": result.get("people_expected"), "people_ok": result.get("people_ok"),
        "link_pct": link["pct"], "taken": take["taken"], "duration_ms": _duration(data),
        "customers": sorted(customers),
        "labeled": None if score is None else {
            "takes_done": score["takes_done"],
            "take_ok": None if score["take"] is None else score["take"]["ok"],
            "link": None if score["link"] is None else [score["link"]["right"], score["link"]["n"]],
            "culprit_ok": None if score["culprit"] is None else score["culprit"]["ok"],
        },
        "lenses": lenses,
    }


def card(summary: dict) -> dict:
    """갤러리 카드용으로 줄인 것 (타임라인·기록은 뺀다)."""
    keep = ("pair", "set", "set_name", "expected", "outcome", "headline", "reason", "correct", "correct_last",
            "no_verdict", "verdicts", "people", "people_expected", "people_ok", "link_pct", "taken", "duration_ms", "labeled")
    out = {k: summary[k] for k in keep}
    out["status"] = {name: summary["lenses"][name]["status"] for name in LENSES}
    return out
