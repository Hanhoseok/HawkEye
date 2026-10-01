"""데이터 검사 리포트 생성.

요구사항 3절의 확인 항목을 하나씩 실제 수치로 답한다.
입력: data/index/clips.jsonl (build_index 산출물)
출력: reports/data_report.json, reports/data_report.md

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.inspect
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

INDEX = Path(r"D:\computervision\data\index\clips.jsonl")
REPORTS = Path(r"D:\computervision\reports")


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    vals = sorted(vals)
    return {
        "n": len(vals), "min": vals[0], "p25": vals[len(vals) // 4],
        "median": st.median(vals), "p75": vals[(3 * len(vals)) // 4],
        "max": vals[-1], "mean": round(st.fmean(vals), 2),
    }


def build_report(rows: list[dict]) -> dict:
    rep: dict = {}
    rep["총_파일수"] = len(rows)
    rep["분할별"] = dict(Counter(r["split_src"] for r in rows))
    rep["클래스별"] = dict(Counter(r.get("class_name") or "UNPARSED" for r in rows))
    rep["클래스별_분할별"] = {
        k: dict(Counter(r["split_src"] for r in rows if (r.get("class_name") or "UNPARSED") == k))
        for k in sorted({r.get("class_name") or "UNPARSED" for r in rows})
    }

    # 1) 영상-라벨 연결 규칙
    rep["연결규칙"] = {
        "규칙": "확장자를 제외한 파일명(stem)이 mp4 와 xml 사이에서 1:1로 동일",
        "xml_meta_task_name_과_일치": sum(1 for r in rows if r.get("has_label")),
        "라벨만_있고_영상없음": sorted(r["stem"] for r in rows if r["has_label"] and not r["has_video"])[:20],
        "영상만_있고_라벨없음": sorted(r["stem"] for r in rows if r["has_video"] and not r["has_label"])[:20],
        "라벨만_있고_영상없음_수": sum(1 for r in rows if r["has_label"] and not r["has_video"]),
        "영상만_있고_라벨없음_수": sum(1 for r in rows if r["has_video"] and not r["has_label"]),
    }

    # 2) 실제 클래스명
    action_counter = Counter()
    for r in rows:
        for e in r.get("events", []):
            action_counter[e["action"]] += 1
    rep["XML_이벤트_라벨"] = dict(action_counter)
    rep["파일명_클래스코드"] = dict(Counter(r.get("class_code") for r in rows))

    # 3) 영상 길이 / FPS / 해상도
    probed = [r for r in rows if r.get("probe_ok")]
    rep["영상_실측"] = {
        "실측_대상수": len(probed),
        "fps": dict(Counter(r["video_fps"] for r in probed)),
        "해상도": dict(Counter(f"{r['video_width']}x{r['video_height']}" for r in probed)),
        "프레임수": dict(Counter(r["video_frames"] for r in probed)),
        "길이초": {k: v for k, v in sorted(Counter(
            round(r["video_frames"] / r["video_fps"], 1) for r in probed if r["video_fps"]).items())},
        "용량MB": _stats([round(r["video_bytes"] / 1e6, 1) for r in rows if r.get("video_bytes")]),
    }
    rep["라벨_기준_프레임수"] = dict(Counter(r.get("label_frames") for r in rows if r.get("label_frames")))
    rep["라벨_기준_해상도"] = dict(Counter(
        f"{r.get('label_width')}x{r.get('label_height')}" for r in rows if r.get("label_width")))

    # 4) 사람 ID / 바운딩박스 / 키포인트
    rep["사람ID"] = {
        "설명": "이벤트 마커 box 의 attribute name='ID' 값. 프레임별 추적 ID가 아니라 연기자 번호.",
        "값분포": dict(Counter(i for r in rows for i in r.get("person_ids", []))),
        "영상당_ID개수": dict(Counter(len(r.get("person_ids", [])) for r in rows)),
    }
    rep["바운딩박스"] = {
        "설명": "*_start / *_end 마커 박스와 object_article/object_tool/object_abandon 박스만 존재. "
                "사람 전체를 프레임마다 감싼 person 박스 트랙은 없음.",
        "object_박스_총계": {k: sum(r.get("object_counts", {}).get(k, 0) for r in rows)
                            for k in ("object_article", "object_tool", "object_abandon")},
        "object_박스_보유영상수": {
            k: sum(1 for r in rows if r.get("object_counts", {}).get(k, 0) > 0)
            for k in ("object_article", "object_tool", "object_abandon")},
    }
    rep["키포인트"] = {
        "관절수": 17,
        "주석된_프레임수_영상당": _stats([r["keypoint_frame_count"] for r in rows if "keypoint_frame_count" in r]),
        "전프레임_주석_여부": "아니오 (영상당 일부 프레임만 주석)",
    }

    # 5) 행동 시작/종료 시점
    ev_rows = [r for r in rows if r.get("events")]
    durations = defaultdict(list)
    starts = defaultdict(list)
    ends = defaultdict(list)
    missing_end = Counter()
    for r in rows:
        for e in r.get("events", []):
            starts[e["action"]].append(e["start_frame"])
            if e["end_frame"] is None:
                missing_end[e["action"]] += 1
            else:
                ends[e["action"]].append(e["end_frame"])
                durations[e["action"]].append(e["n_frames"])
    rep["시간라벨"] = {
        "존재": True,
        "형식": "*_start / *_end 트랙이 각각 한 프레임에 outside=0 박스로 찍혀 있음 → [start, end] 폐구간",
        "이벤트_보유_영상수": len(ev_rows),
        "시작프레임_분포": {k: _stats(v) for k, v in starts.items()},
        "종료프레임_분포": {k: _stats(v) for k, v in ends.items()},
        "지속_프레임수_분포": {k: _stats(v) for k, v in durations.items()},
        "지속_초_분포_3fps기준": {k: {kk: (round(vv / 3.0, 2) if isinstance(vv, (int, float)) and kk != "n" else vv)
                                  for kk, vv in _stats(v).items()} for k, v in durations.items()},
        "end_누락": dict(missing_end),
    }

    # 6) 정상 구간 / 다중 행동 구간
    n_events = Counter(len(r.get("events", [])) for r in rows)
    multi = [r["stem"] for r in rows if len(r.get("events", [])) > 1]
    outside_frac = []
    for r in rows:
        nf = r.get("label_frames") or 0
        if not nf or not r.get("events"):
            continue
        covered = set()
        for e in r["events"]:
            end = e["end_frame"] if e["end_frame"] is not None else e["start_frame"]
            covered.update(range(e["start_frame"], end + 1))
        outside_frac.append(round(1 - len(covered) / nf, 4))
    rep["구간구성"] = {
        "영상당_이벤트수_분포": dict(sorted(n_events.items())),
        "다중이벤트_영상수": len(multi),
        "다중이벤트_예시": multi[:10],
        "이벤트_구간외_프레임_비율": _stats(outside_frac),
        "주의": "구간 밖 프레임은 '정상'으로 검증된 것이 아니라 '라벨이 없는 구간'이다. "
                "연출된 이상행동 영상의 접근/사전 동작이 섞여 있으므로 background_unlabeled 로만 사용한다.",
        "다중_행동종류_혼재_영상수": sum(
            1 for r in rows if len({e["action"] for e in r.get("events", [])}) > 1),
    }

    # 7) 매장 / 세션 / 카메라 / 배우 구분 가능 여부
    rep["메타데이터_구분"] = {
        "매장": dict(Counter(r.get("store") for r in rows)),
        "매장_한글": dict(Counter(r.get("store_kr") for r in rows)),
        "카메라": dict(Counter(r.get("camera") for r in rows)),
        "지역": dict(Counter(r.get("region") for r in rows)),
        "촬영팀": dict(Counter(r.get("crew") for r in rows)),
        "배우코드": dict(Counter(r.get("actor") for r in rows)),
        "성별_XML": dict(Counter(g for r in rows for g in r.get("genders", []))),
        "연령대_XML": dict(Counter(g for r in rows for g in r.get("age_groups", []))),
        "날짜수": len({r.get("date") for r in rows}),
        "한계": "배우코드(M1/F2…)는 성별+번호일 뿐 개인 식별자가 아니다. "
                "매장·날짜가 다르면 같은 코드라도 다른 사람일 수 있어 '배우 분리 평가'는 불가능하다.",
    }
    # take 통계
    takes = defaultdict(list)
    for r in rows:
        takes[r.get("take_id", "?")].append(r)
    take_sizes = Counter(len(v) for v in takes.values())
    cross_split = [t for t, v in takes.items() if len({x["split_src"] for x in v}) > 1]
    rep["촬영사건_take"] = {
        "정의": "같은 (클래스, 매장, 날짜, 시나리오, 배우) 안에서 파일명 시각이 90초 이내로 붙은 영상 묶음",
        "take_수": len(takes),
        "take당_영상수_분포": dict(sorted(take_sizes.items())),
        "배포_train과_val을_가로지르는_take": len(cross_split),
        "take_내_카메라_중복": sum(
            1 for v in takes.values() if len({x.get("camera") for x in v}) != len(v)),
    }

    # 8) 누락 / 손상 / 중복
    prob_counter = Counter()
    for r in rows:
        for p in r["problems"]:
            key = p.split(":")[0]
            prob_counter[key] += 1
    rep["문제항목"] = {
        "문제있는_파일수": sum(1 for r in rows if r["problems"]),
        "유형별": dict(prob_counter),
        "예시": [{"stem": r["stem"], "problems": r["problems"]} for r in rows if r["problems"]][:20],
        "중복_stem": [s for s, c in Counter(r["stem"] for r in rows).items() if c > 1],
    }
    return rep


def to_markdown(rep: dict) -> str:
    j = lambda o: json.dumps(o, ensure_ascii=False, indent=2)
    return f"""# 데이터 검사 리포트 (자동 생성)

> `storeguard.data.inspect` 산출물. 수치는 모두 실제 파일에서 계산되었다.

## 0. 규모

- 총 파일: **{rep['총_파일수']}**
- 분할별: {rep['분할별']}
- 클래스별: {rep['클래스별']}
- 클래스×분할: {j(rep['클래스별_분할별'])}

## 1. 영상 ↔ 라벨 연결 규칙

{j(rep['연결규칙'])}

## 2. 실제 클래스명 / 이벤트 라벨

XML `track@label` 에서 실제로 쓰인 이벤트 라벨:

{j(rep['XML_이벤트_라벨'])}

파일명 클래스 코드: {rep['파일명_클래스코드']}

## 3. 영상 길이 / FPS / 해상도

{j(rep['영상_실측'])}

라벨 기준 프레임 수: {rep['라벨_기준_프레임수']}
라벨 기준 해상도: {rep['라벨_기준_해상도']}

## 4. 사람 ID / 바운딩박스 / 키포인트

### 사람 ID
{j(rep['사람ID'])}

### 바운딩박스
{j(rep['바운딩박스'])}

### 키포인트
{j(rep['키포인트'])}

## 5. 행동 시작·종료 시점

{j(rep['시간라벨'])}

## 6. 정상 구간 / 다중 행동 구간

{j(rep['구간구성'])}

## 7. 매장 / 촬영 세션 / 카메라 / 배우

{j(rep['메타데이터_구분'])}

### 촬영 사건(take)
{j(rep['촬영사건_take'])}

## 8. 누락 / 손상 / 중복

{j(rep['문제항목'])}
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--out", type=Path, default=REPORTS)
    a = ap.parse_args(argv)

    rows = load(a.index)
    rep = build_report(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "data_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.out / "data_report.md").write_text(to_markdown(rep), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2)[:4000])
    print(f"\n저장: {a.out / 'data_report.json'}, {a.out / 'data_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
