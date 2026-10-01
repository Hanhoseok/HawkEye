"""train / val / test 분할 목록 생성.

원칙 (AGENTS.md 5항)
  - 분할은 **촬영 사건(take) 단위**로 자른다. 같은 take 의 다른 카메라 영상은 같은 분할에 들어간다.
  - test 는 배포된 Validation 세트를 그대로 쓴다. 모델 선택/임계값 튜닝에는 쓰지 않는다.
  - 배포 Training 세트를 take 단위로 train/val 로 나눈다.

mode
  take   : 기본. take 단위 층화 분할(클래스 비율 유지). 매장은 train/val 양쪽에 모두 등장한다.
  store  : 매장 단위 분할. 새 매장 일반화 성능을 보려면 이쪽. 데이터가 줄고 분산이 커진다.

산출물: data/index/splits.json
    {"train": [stem...], "val": [...], "test": [...], "meta": {...}}

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.split
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.data.split --set split.mode=store
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from storeguard.config import Config


def load_rows(index_dir: Path) -> list[dict]:
    p = index_dir / "clips.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def _usable(r: dict) -> bool:
    # mp4 가 없어도 프레임 캐시가 있으면 학습에 쓸 수 있다.
    # (ingest 는 영상을 캐시로 구운 뒤 mp4 를 지운다)
    if not ((r.get("has_video") or r.get("has_cache")) and r.get("class_name")):
        return False
    if r.get("class_name") == "normal_verified":
        # 정상 영상은 라벨 XML 도 이벤트도 없다(add_normal.py 로 등록된 것)
        return True
    return bool(r.get("has_label") and r.get("events"))


def split_takes(rows: list[dict], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    """take 단위 층화 분할. take 의 대표 클래스로 층화한다."""
    by_take: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_take[r["take_id"]].append(r)

    by_class: dict[str, list[str]] = defaultdict(list)
    for take_id, group in by_take.items():
        cls = Counter(r["class_name"] for r in group).most_common(1)[0][0]
        by_class[cls].append(take_id)

    rng = random.Random(seed)
    train_t: set[str] = set()
    val_t: set[str] = set()
    for cls, takes in sorted(by_class.items()):
        takes = sorted(takes)
        rng.shuffle(takes)
        n_val = max(1, round(len(takes) * val_ratio))
        val_t.update(takes[:n_val])
        train_t.update(takes[n_val:])
    return train_t, val_t


def split_stores(rows: list[dict], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    """매장 단위 분할. 매장 수가 적으므로(최대 7) 1~2개를 val 로 뺀다."""
    stores = sorted({r["store"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(stores)
    n_val = max(1, round(len(stores) * val_ratio))
    val_s = set(stores[:n_val])
    return set(stores) - val_s, val_s


def split_pool_three_way(rows: list[dict], val_ratio: float, test_ratio: float,
                         seed: int) -> tuple[set[str], set[str], set[str]]:
    """take 단위 층화 3분할. 배포 분할을 쓸 수 없는 데이터셋에 쓴다."""
    by_take: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_take[r["take_id"]].append(r)
    by_class: dict[str, list[str]] = defaultdict(list)
    for take_id, group in by_take.items():
        cls = Counter(r["class_name"] for r in group).most_common(1)[0][0]
        by_class[cls].append(take_id)

    rng = random.Random(seed)
    train_t: set[str] = set()
    val_t: set[str] = set()
    test_t: set[str] = set()
    for cls, takes in sorted(by_class.items()):
        takes = sorted(takes)
        rng.shuffle(takes)
        n = len(takes)
        n_test = max(1, round(n * test_ratio)) if n >= 3 else 0
        n_val = max(1, round(n * val_ratio)) if n - n_test >= 2 else 0
        test_t.update(takes[:n_test])
        val_t.update(takes[n_test:n_test + n_val])
        train_t.update(takes[n_test + n_val:])
    return train_t, val_t, test_t


def build(cfg: Config) -> dict:
    index_dir = cfg.path("paths.index")
    rows = load_rows(index_dir)
    usable = [r for r in rows if _usable(r)]
    dropped = [r["stem"] for r in rows if not _usable(r)]

    # 데이터셋마다 배포 분할을 믿을 수 있는지가 다르다.
    #   238-2 이상행동 : Training/Validation 이 모두 완전히 내려받아져 있다 → 배포 분할을 쓴다.
    #   238-1 구매행동 : 배포본 일부만 받아져 있어 클래스 대부분이 Validation 쪽에만 있다.
    #                    그대로 두면 학습 데이터가 0이 된다 → take 단위로 직접 3분할한다.
    honor = set(cfg.get("split.honor_deployed_test_datasets", ["238-2"]) or [])
    honored = [r for r in usable if (r.get("dataset") or "238-2") in honor]
    selfsplit = [r for r in usable if (r.get("dataset") or "238-2") not in honor]

    # 배포된 Training / Validation 사이에 같은 take 가 걸쳐 있는 경우가 실제로 3건 있다
    # (동일 시각·시나리오를 카메라별로 갈라 놓음). 그대로 두면 train→test 누수가 된다.
    # test(=배포 Validation)를 보존하고, 겹치는 take 의 train 쪽 영상을 버린다.
    take_splits: dict[str, set[str]] = defaultdict(set)
    for r in honored:
        take_splits[r["take_id"]].add(r["split_src"])
    cross_takes = {t for t, s in take_splits.items() if len(s) > 1}
    cross_dropped = [r["stem"] for r in honored
                     if r["take_id"] in cross_takes and r["split_src"] == "train"]

    pool = [r for r in honored
            if r["split_src"] == "train" and r["take_id"] not in cross_takes]
    test_rows = [r for r in honored if r["split_src"] == "val"]

    mode = cfg.get("split.mode", "take")
    ratio = float(cfg.get("split.val_ratio", 0.15))
    seed = int(cfg.get("split.seed", 1337))

    if mode == "take":
        train_keys, val_keys = split_takes(pool, ratio, seed)
        keyf = lambda r: r["take_id"]
    elif mode == "store":
        train_keys, val_keys = split_stores(pool, ratio, seed)
        keyf = lambda r: r["store"]
    else:
        raise ValueError(f"알 수 없는 split.mode: {mode}")

    train = [r["stem"] for r in pool if keyf(r) in train_keys]
    val = [r["stem"] for r in pool if keyf(r) in val_keys]
    test = [r["stem"] for r in test_rows]

    # 배포 분할을 못 쓰는 데이터셋은 take 단위로 직접 3분할한다.
    self_info = {}
    if selfsplit:
        t_ratio = float(cfg.get("split.test_ratio", 0.2))
        st, sv, ste = split_pool_three_way(selfsplit, ratio, t_ratio, seed)
        train += [r["stem"] for r in selfsplit if r["take_id"] in st]
        val += [r["stem"] for r in selfsplit if r["take_id"] in sv]
        test += [r["stem"] for r in selfsplit if r["take_id"] in ste]
        self_info = {
            "대상_데이터셋": sorted({r.get("dataset") for r in selfsplit}),
            "영상수": len(selfsplit),
            "take수": len({r["take_id"] for r in selfsplit}),
            "train_take": len(st), "val_take": len(sv), "test_take": len(ste),
            "이유": "배포본이 일부만 내려받아져 있어 해당 클래스가 Validation 쪽에만 존재한다. "
                    "배포 분할을 그대로 쓰면 학습 데이터가 0이 되므로 take 단위로 직접 나눈다.",
        }

    train, val, test = sorted(train), sorted(val), sorted(test)
    by_stem = {r["stem"]: r for r in usable}

    def summary(stems: list[str]) -> dict:
        rs = [by_stem[s] for s in stems]
        return {
            "n_videos": len(rs),
            "n_takes": len({r["take_id"] for r in rs}),
            "datasets": dict(Counter(r.get("dataset") or "?" for r in rs)),
            "classes": dict(Counter(r["class_name"] for r in rs)),
            "stores": dict(Counter(r["store"] for r in rs)),
            "n_events": sum(len(r["events"]) for r in rs),
        }

    # 누수 검사
    t_takes = {by_stem[s]["take_id"] for s in train}
    v_takes = {by_stem[s]["take_id"] for s in val}
    te_takes = {by_stem[s]["take_id"] for s in test}
    leaks = {
        "train∩val_take": sorted(t_takes & v_takes),
        "train∩test_take": sorted(t_takes & te_takes),
        "val∩test_take": sorted(v_takes & te_takes),
        "stem_중복": sorted(set(train) & set(val) | set(train) & set(test) | set(val) & set(test)),
    }

    return {
        "train": train, "val": val, "test": test,
        "meta": {
            "mode": mode, "val_ratio": ratio, "seed": seed,
            "test_출처": "배포 Validation 세트 그대로. 모델 선택/임계값 조정 금지.",
            "train_요약": summary(train), "val_요약": summary(val), "test_요약": summary(test),
            "제외된_영상수": len(dropped),
            "제외된_영상_예시": dropped[:20],
            "배포분할_누수_take": sorted(cross_takes),
            "누수로_train에서_제외한_영상": cross_dropped,
            "배포분할_사용_데이터셋": sorted(honor),
            "직접분할": self_info,
            "누수검사": leaks,
            "매장_중복": {
                "train_val_공통": sorted(
                    {by_stem[s]["store"] for s in train} & {by_stem[s]["store"] for s in val}),
                "train_test_공통": sorted(
                    {by_stem[s]["store"] for s in train} & {by_stem[s]["store"] for s in test}),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    cfg = Config.load(a.config, a.overrides)
    out = a.out or (cfg.path("paths.index") / "splits.json")
    res = build(cfg)

    # add_normal.py 로 등록해 둔 정상 영상 분할이 있으면 덮어쓰지 않고 이어받는다.
    if out.exists():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
            if prev.get("normal"):
                res["normal"] = prev["normal"]
                res["meta"]["정상영상"] = prev.get("meta", {}).get("정상영상")
                normal_train = [s for s in prev.get("train", []) if s not in set(res["train"])]
                res["train"] = sorted(set(res["train"]) | set(normal_train))
        except Exception:
            pass
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    m = res["meta"]
    print(json.dumps({k: m[k] for k in ("mode", "train_요약", "val_요약", "test_요약",
                                        "제외된_영상수", "누수검사", "매장_중복")},
                     ensure_ascii=False, indent=2))
    bad = any(v for k, v in m["누수검사"].items())
    print(("[경고] 분할 누수 발견" if bad else "[확인] 분할 누수 없음"))
    print(f"저장: {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
