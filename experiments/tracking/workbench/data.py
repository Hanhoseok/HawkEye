"""기록 → 분석 도구 데이터 (outputs/workbench/).

저장된 기록(2차 outputs/dual_eval_pose3/<짝>/ · 3차 outputs/dual_eval_shoot3_prod/<짝>/ — 상품 모델 기록)을 지금 설정으로 다시 판정하며 보기 데이터를 만들고,
질문별 요약(analyze.py)을 붙여 짝마다 한 파일로 둔다. 영상을 다시 돌리지 않는다 — 짝당 1초 안팎.

    python workbench/data.py              # 바뀐 짝만 다시 만들기 (2차)
    python workbench/data.py --shoot 3    # 3차 촬영
    python workbench/data.py --force      # 전부 다시

촬영 회차마다 영상·기록·구역·정답표·적용한 값이 따로다 (PROFILES). 짝 이름이 겹치므로(C12 는 양쪽에 있다) 만든 것도 따로 둔다
— 2차 outputs/workbench/, 3차 outputs/workbench/shoot3/.
만들어 둔 것은 구역·시차·정답표·설정 파일이나 기록이 바뀌면 그 짝만 다시 만든다(서명으로 확인).
결과와 영상은 저장소에 올리지 않는다 (outputs/·source/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from workbench import labels as label_store  # noqa: E402
from workbench.analyze import analyze, card, sensor_log  # noqa: E402

OUT = ROOT / "outputs" / "workbench"
VERSION = 16
"""보기 데이터의 모양이나 **판정 코드의 결과**가 바뀌면 올린다 — 예전에 만들어 둔 것을 전부 다시 만들게 된다.
8: 3차에서 매장 안 바닥을 밟은 신원만 두 화면 짝 후보 (fusion.link_customers_only, 후속 43) — 짝이 달라진다.
9: 직전 짝 가산점은 바로 앞 조각의 짝에게만 (fusion.keep_bonus_recent_only, 후속 44).
10: 전체 카메라 번호 맞바뀜 되돌리기 (fusion.swap_repair, 후속 49) — 보기 데이터에 swaps 가 생기고 3차 C20 의 번호·짝이 달라진다.
11: 전체 카메라에서 찾은 상품 (overview_items, 후속 52) — 보기만 하고 판정은 그대로다.
12: 전체 카메라의 선반 앞 머묾을 판정에 넣었는지 (overview_dwell, 후속 55) — 뺀 보기는 그 줄에 '판정에 안 씀'이 붙는다.
13: 3차의 기본 기록이 상품 모델 기록으로, 3차 기본 판정이 전체 카메라 개수 · 든 사람 · 머묾 안 씀으로 (후속 56). 보기 31 · 32 의 뜻이 바뀌고 33 은 없어졌다.
(같은 날, 후속 57) 보기는 회차마다 최신 2개만 둔다 — 31 = 3차의 바로 전 판정, 32 는 없앴다. 만드는 내용은 그대로라 번호를 안 올렸다.
14: 전체 카메라의 '선반 앞 머묾' 줄을 없앴다 (코드에서 지움, 후속 58). 손에 든 것을 보면 바로 세는 기록(held)과 그 줄이 생겼다 (후속 59).
    31 = 바로 세기를 끈 판정.
15: 센서 로그의 기록을 보기 데이터에 넣는다 (sensor_log — '물건 집기'에 줄로 그린다). 결제 로그로 판정한 보기는
    '계산대에 서 있으면 결제'라는 문장을 안 쓴다 (후속 63).
16: 3차 보기의 기준이 센서 기록(L1 · L2) + 결제 라벨 판정으로 (후속 64). 31 = 바로 전 판정 = 영상만. 서명에 센서 레인 · 결제 기록 여부를 넣었다."""
RUN_KEY = "current"


ADDED = ROOT / "labels" / "store01_dual" / "added.csv"
BATCH_NAME = {"shoot2": "2차 촬영", "added": "추가한 영상", "shoot3": "3차 촬영", "shoot4": "4차 촬영"}


@dataclass(frozen=True)
class Profile:
    """한 촬영 회차의 자리들. 카메라 자리가 다르면 구역·시차·적용한 판정 값이 모두 따로다."""

    shoot: int
    name: str
    records: Path
    """탐지·추적 기록 (scripts/dual_eval.py run 이 만든 것)."""
    labels: Path
    """정답표·시차·결제·사람이 적은 정답·수치 조정 기록."""
    zones: Path
    out: Path
    """만든 보기 데이터."""
    judge_file: Path
    """수치 조정에서 적용한 값."""
    signature_files: tuple[Path, ...]
    """모든 짝의 결과를 바꿀 수 있는 파일 (설정·구역)."""
    can_add: bool
    """분석 도구에서 새 영상을 추가할 수 있나 (구역을 그대로 쓸 수 있는 회차만)."""
    payment: str = "계산대 체류 = 결제"
    note: str = ""
    """갤러리 아래에 보이는 설명 — 같은 회차의 다른 기록(견줘 보기)이 무엇을 바꾼 것인지."""
    can_tune: bool = True
    """수치 조정을 할 수 있나. 견줘 보기는 못 한다 — 적용한 값 파일과 조정 기록을 그 회차와 함께 쓰기 때문."""
    thumbs: Path | None = None
    """장면 그림을 여기서 가져다 쓴다 (전체 카메라 추적이 같은 기록이면 그림도 같다 — 영상을 다시 읽지 않는다)."""
    overrides: tuple[str, ...] = ()
    """이 보기에서만 얹는 판정 값 (`절.키=값`, dual_eval.py 의 --set 과 같다) — 같은 기록을 다른 설정으로 판정해 견줄 때."""
    sensor_lanes: tuple[str, ...] = ()
    """판정에 넣을 센서 레인 (labels/<회차>/sensor/<짝>.csv). 비면 센서 없이 판정한다."""
    payments: bool = False
    """결제 기록(labels/<회차>/payments.csv)을 결제로 쓴다. False 면 계산대 체류 = 결제."""


PROFILES = {
    2: Profile(2, "2차 촬영", ROOT / "outputs" / "dual_eval_pose3", ROOT / "labels" / "store01_dual",
               ROOT / "zones" / "store01_dual", OUT, ROOT / "configs" / "store01_dual_judge.yaml",
               (ROOT / "config.yaml", ROOT / "zones" / "store01_dual" / "floor.yaml",
                ROOT / "zones" / "store01_dual" / "angle0.yaml"), True),
    # 3차는 영상마다 카메라가 조금씩 움직여, 기본 구역을 영상별 화면으로 옮겨 쓴다 (outputs/sync_floor/register.csv).
    # 새 영상을 넣으려면 그 맞추기부터 해야 해서 도구에서 추가하지 못한다.
    # 3차의 기본 기록은 상품 모델 기록이다 (후속 56 — scripts/dual_eval.py RECORDS 와 같은 폴더). 판정 값도 그쪽 3차 기본 그대로:
    # 전체 카메라 개수 · 든 사람으로 주인 · 선반 앞 머묾 안 씀.
    3: Profile(3, "3차 촬영", ROOT / "outputs" / "dual_eval_shoot3_prod", ROOT / "labels" / "store01_shoot3",
               ROOT / "zones" / "store01_shoot3", OUT / "shoot3", ROOT / "configs" / "store01_shoot3_judge.yaml",
               (ROOT / "config.yaml", *(ROOT / "zones" / "store01_shoot3" / n for n in
                                        ("floor.yaml", "angle0.yaml", "floor_angle0.yaml", "band_map.yaml")),
                ROOT / "outputs" / "sync_floor" / "register.csv"), False,
               # 3차도 최종 판정 흐름으로 본다 (사용자 결정 2026-11-23, 후속 64): 센서 기록(L1 · L2) + 결제 라벨. 3차에는 실제 결제 로그가 없어
               # 사람이 적은 결제 라벨(payments.csv)을 쓴다. 수치 조정의 초안도 같은 입력으로 판정한다 (Builder._build).
               payment="결제 라벨 (사람이 적음)", sensor_lanes=("L1", "L2"), payments=True,
               note="센서 기록(L1 · L2)과 사람이 적은 결제 라벨을 넣은 판정입니다. 센서 L3(애플사이다)은 판정에 안 씁니다."),
    # 4차(2026-11-19)는 최종 판정 흐름 그대로 본다: 상품 모델 기록 + 센서 로그(L1 · L2) + 결제 로그 (사용자 방향, 2026-11-20).
    # 수치 조정은 꺼 둔다. 처음 끈 까닭("초안이 '영상만'으로 돈다")은 이제 맞지 않는다 — 초안도 같은 입력으로 판정한다 (3차로 확인, 후속 64).
    # 4차에서 켜 보지는 않았다.
    4: Profile(4, "4차 촬영", ROOT / "outputs" / "dual_eval_shoot4", ROOT / "labels" / "store01_shoot4",
               ROOT / "zones" / "store01_shoot4", OUT / "shoot4", ROOT / "configs" / "store01_shoot4_judge.yaml",
               (ROOT / "config.yaml", *(ROOT / "zones" / "store01_shoot4" / n for n in
                                        ("floor.yaml", "angle0.yaml", "floor_angle0.yaml", "band_map.yaml")),
                ROOT / "outputs" / "sync_floor_shoot4" / "register.csv"), False,
               payment="결제 로그 (POS)", can_tune=False, sensor_lanes=("L1", "L2"), payments=True,
               note="센서 로그(L1 · L2)와 결제 로그(POS)를 넣은 판정입니다. 센서 L3(애플사이다)은 판정에 안 쓰고 카메라와 결제 로그로 셉니다."),
}
"""열쇠는 보기 번호 — 주소(#/3)와 요청(?shoot=3)에 쓴다. 회차의 기본 기록은 회차 번호 그대로이고,
같은 회차의 다른 기록(견줘 보기)은 다른 번호를 쓴다 (Profile.shoot 는 그대로 회차)."""

MAX_VIEWS = 2
"""한 회차의 보기는 최신 2개까지 (사용자 규칙, 2026-11-20): 지금 기준(#/3)과 바로 전 판정(#/31).
새 판정이 기준이 되면 — 지금 것을 '바로 전'으로 옮기고, 그 전 것은 지운다 (보기 데이터 폴더는 `--prune`). 3차: 2026-11-23 에 한 번 옮겼다 (후속 64).
더 옛 판정은 점수만 본다: `scripts/dual_eval.py judge --out <기록> --set …` (docs/shelf-map.md)."""

SHOOT3_PREV = 31
"""바로 전 판정: 지금 기록(상품 모델)을 **영상만**으로 판정한 것 — 센서 기록과 결제 라벨을 안 넣고 계산대 체류를 결제로 본다
(2026-11-23 후속 64 전까지 3차 보기의 기준이었다). 그 전의 '방문이 끝나야 셈'(fusion.overview_instant 끔) 보기는 규칙대로 지웠다.
정답표 · 시차 · 결제 · 구역 · 적용한 값은 3차 것을 그대로 쓴다 — 여기서 정답을 고치면 3차의 정답이 바뀐다."""
PROFILES[SHOOT3_PREV] = replace(
    PROFILES[3], name="3차 · 바로 전 판정 (영상만)", out=OUT / "shoot3_prev", can_tune=False, thumbs=OUT / "shoot3" / "thumbs",
    payment="계산대 체류 = 결제", sensor_lanes=(), payments=False,
    note="'3차 촬영'과 같은 기록입니다. 센서 기록과 결제 라벨 없이 영상만으로 낸 판정입니다 "
         "(2026-11-23 후속 64 전의 기준). 수치 조정은 3차 촬영에서 합니다.")


def views_of(shoot: int) -> list[int]:
    """그 회차의 보기 번호들 (기본 보기 + 견줘 보기)."""
    return [key for key, p in PROFILES.items() if p.shoot == shoot]


def check_views() -> None:
    """보기 규칙을 어기면 그 자리에서 멈춘다 — 회차마다 MAX_VIEWS 개까지, 만든 것을 두는 폴더는 서로 달라야 한다."""
    for shoot in {p.shoot for p in PROFILES.values()}:
        keys = views_of(shoot)
        if len(keys) > MAX_VIEWS:
            raise RuntimeError(f"{shoot}차 보기가 {len(keys)}개입니다 ({keys}) — 최신 {MAX_VIEWS}개만 둡니다. 가장 옛 것을 지우세요")
    outs = [p.out for p in PROFILES.values()]
    if len(set(outs)) != len(outs):
        raise RuntimeError("두 보기가 같은 폴더에 만듭니다")


check_views()


def stale_outputs() -> list[Path]:
    """지금 어느 보기도 안 쓰는 보기 데이터 폴더 (outputs/workbench/shoot*) — 없앤 옛 보기가 남긴 것."""
    used = {p.out for p in PROFILES.values()}
    return sorted(d for d in OUT.glob("shoot*") if d.is_dir() and d not in used)


def main_views() -> list[int]:
    """회차의 기본 기록인 보기들 (견줘 보기를 뺀 것)."""
    return [key for key, p in PROFILES.items() if key == p.shoot]

ENGINE_LOCK = threading.RLock()
"""판정 코드(scripts/dual_eval.py)는 촬영 회차·기록 폴더를 모듈 값으로 들고 있다 — 한 번에 한 회차만 돌린다.
회차별 Builder 가 이 자물쇠 하나를 함께 쓴다."""


def added_names() -> set[str]:
    """분석 도구에서 추가한 영상의 이름 (workbench/intake.py 가 적는 labels/store01_dual/added.csv)."""
    if not ADDED.exists():
        return set()
    import csv
    with open(ADDED, encoding="utf-8", newline="") as f:
        return {r["pair"] for r in csv.DictReader(f)}


def natural(pair: str) -> tuple:
    """N1, N2, …, N10 순서 (글자 순이면 N10 이 N2 앞에 온다). 세트는 정상 → 절도 → 복합."""
    digits = "".join(c if c.isdigit() else " " for c in pair).split()
    return ("NTC".find(pair[0]) % 9, [int(d) for d in digits])


def totals(cards: list[dict]) -> dict:
    """카드 묶음의 점수 — 갤러리 위쪽 숫자와 수치 조정의 전후 비교가 함께 쓴다 (dual_eval.summarize 와 같은 셈)."""
    right = [c for c in cards if c["correct"]]
    counted = [c for c in cards if c["people_ok"] is not None]
    return {
        "n": len(cards), "correct": len(right),
        "correct_decided": sum(not c["no_verdict"] for c in right),
        "no_verdict": sum(c["no_verdict"] for c in cards),
        "correct_last": sum(c["correct_last"] for c in cards),
        "people_ok": sum(c["people_ok"] for c in counted), "people_n": len(counted),
        "labeled": _labeled(cards),
        "by_set": {s: {"n": sum(c["set"] == s for c in cards), "correct": sum(c["set"] == s and c["correct"] for c in cards)}
                   for s in ("normal", "theft", "complex")},
    }


def _labeled(cards: list[dict]) -> dict:
    """사람이 적은 정답이 있는 영상들의 점수 (정답 적기)."""
    have = [c["labeled"] for c in cards if c.get("labeled")]
    take = [x["take_ok"] for x in have if x["take_ok"] is not None]
    link = [x["link"] for x in have if x["link"] is not None]
    culprit = [x["culprit_ok"] for x in have if x["culprit_ok"] is not None]
    return {"n": len(have), "take_n": len(take), "take_ok": sum(take),
            "link_n": sum(n for _, n in link), "link_ok": sum(r for r, _ in link),
            "culprit_n": len(culprit), "culprit_ok": sum(culprit)}


class Builder:
    """한 촬영 회차의 짝별 데이터를 만들고 상태를 알려 준다. 서버는 build_async 로 뒤에서 돌리고 status 를 화면에 보인다."""

    def __init__(self, key: int = 2) -> None:
        self.profile = PROFILES[key]
        self.key = key
        """보기 번호 (PROFILES 의 열쇠)."""
        self.shoot = self.profile.shoot
        """촬영 회차 — 판정 코드가 구역 · 정답표를 고르는 번호."""
        self.out = self.profile.out
        self.lock = ENGINE_LOCK
        self.state = {"running": False, "done": 0, "total": 0, "current": "", "errors": [], "seconds": 0.0}
        self._ctx = None

    @contextmanager
    def engine(self):
        """판정 코드를 이 회차로 맞추고 쓴다 (자물쇠를 쥔 채). 판정·설정 읽기·시차 추정은 모두 이 안에서."""
        with self.lock:
            de = self.ctx()["dual_eval"]
            if de.SHOOT != self.shoot or de.OUT != self.profile.records:
                de.set_profile(self.shoot)
                de.OUT = self.profile.records      # 같은 회차의 다른 기록 (견줘 보기)
            yield de

    # -- 판정에 쓰는 것들 (한 번 읽어 둔다)
    def ctx(self):
        if self._ctx is None:
            import csv

            import dual_eval
            import review_export
            import sync_pair
            from src.core.config import AppConfig
            from src.inputs.sync import SyncTable
            self._ctx = {"dual_eval": dual_eval, "review_export": review_export, "sync_pair": sync_pair,
                         "AppConfig": AppConfig, "SyncTable": SyncTable, "csv": csv}
        return self._ctx

    def records_dir(self) -> Path:
        return self.profile.records

    def pairs(self) -> list[str]:
        names = [p.name for p in self.records_dir().iterdir() if (p / ".done").exists()] if self.records_dir().exists() else []
        return sorted(names, key=natural)

    def signature(self, pair: str) -> str:
        """이 짝의 결과를 바꿀 수 있는 파일들의 크기·수정 시각."""
        p = self.profile
        files = [*p.signature_files, p.judge_file, self.events_dir() / f"{pair}.json",
                 self.records_dir() / pair / ".done",
                 self.records_dir() / pair / "meta.json",
                 self.records_dir() / pair / "angle1" / "product_items.meta.json"]     # 전체 카메라 상품 (있을 때만)
        if p.sensor_lanes:
            files.append(p.labels / "sensor" / f"{pair}.csv")                          # 센서 로그 (판정에 넣는 회차만)
        h = hashlib.sha1(f"v{VERSION}{'|'.join(p.overrides)}|{','.join(p.sensor_lanes)}|{int(p.payments)}".encode())
        # 짝마다 한 줄씩 있는 표는 그 짝의 줄만 본다 — 한 영상의 시차·결제를 고쳤다고 38쌍을 다 다시 만들지 않게
        for table in (p.labels / "sync.csv", p.labels / "truth.csv", self.payments_file(), p.labels / "added.csv"):
            h.update(self._rows(table).get(pair, "-").encode("utf-8"))
        for f in files:
            st = f.stat() if f.exists() else None
            h.update(f"{f.name}:{st.st_size if st else -1}:{int(st.st_mtime) if st else 0};".encode())
        return h.hexdigest()[:16]

    def events_dir(self) -> Path:
        """사람이 적은 정답 (workbench/labels.py)."""
        return self.profile.labels / "events"

    def payments_file(self) -> Path:
        return self.profile.labels / "payments.csv"

    def _rows(self, path: Path) -> dict[str, str]:
        """표의 줄을 짝 이름(첫 칸)별로 묶은 것. 파일이 그대로면 다시 읽지 않는다."""
        cache = self.__dict__.setdefault("_row_cache", {})
        st = path.stat() if path.exists() else None
        key = (st.st_size, st.st_mtime_ns) if st else None
        if path not in cache or cache[path][0] != key:
            rows: dict[str, str] = {}
            if st:
                for line in path.read_text(encoding="utf-8-sig").splitlines()[1:]:
                    name = line.split(",", 1)[0]
                    rows[name] = rows.get(name, "") + line + "\n"
            cache[path] = (key, rows)
        return cache[path][1]

    def settings_signature(self) -> str:
        """모든 짝에 걸친 것(설정·구역·시차·정답표·적용한 값·기록 목록)의 지문 — 수치 조정의 초안이 어느 기준 위의 것인지."""
        h = hashlib.sha1()
        for pair in self.pairs():
            h.update(self.signature(pair).encode())
        return h.hexdigest()[:12]

    def stale(self) -> list[str]:
        index = self.read_index()
        have = {c["pair"]: c.get("signature") for c in index.get("cards", [])}
        return [p for p in self.pairs()
                if have.get(p) != self.signature(p) or not (self.out / "pairs" / f"{p}.json").exists()]

    def read_index(self) -> dict:
        path = self.out / "index.json"
        if not path.exists():
            return {"cards": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"cards": []}

    # -- 만들기
    def build(self, pairs: list[str] | None = None, force: bool = False) -> dict:
        with self.engine():
            started = time.time()
            todo = self.pairs() if force else self.stale()
            if pairs is not None:
                todo = [p for p in todo if p in pairs]
            self.state.update(running=True, done=0, total=len(todo), current="", errors=[], seconds=0.0)
            try:
                if todo:
                    self._build(todo)
                self._write_index()
            finally:
                self.state.update(running=False, current="", seconds=round(time.time() - started, 1))
            return dict(self.state)

    def build_async(self) -> bool:
        if self.state["running"] or not self.stale():
            return False
        self.state.update(running=True, done=0, total=len(self.stale()), current="준비 중")
        threading.Thread(target=self.build, daemon=True).start()
        return True

    def build_draft(self, out: Path, pairs: list[str], overrides: list[str]) -> list[str]:
        """초안 값(--set 과 같은 '절.키=값')으로 pairs 를 다시 판정해 out 에 둔다 → 실패 목록. 이미 있는 짝은 건너뛴다."""
        with self.engine():
            todo = [p for p in pairs if not (out / "cards" / f"{p}.json").exists()]
            self.state.update(running=True, done=0, total=len(todo), current="", errors=[], seconds=0.0)
            started = time.time()
            try:
                if todo:
                    self._build(todo, out, overrides)
            finally:
                self.state.update(running=False, current="", seconds=round(time.time() - started, 1))
            return list(self.state["errors"])

    def _build(self, todo: list[str], out: Path | None = None, overrides: list[str] | None = None) -> None:
        """engine() 안에서만 부른다."""
        out = out or self.out
        main = out == self.out
        c = self.ctx()
        de, rx = c["dual_eval"], c["review_export"]
        base = c["AppConfig"].load(ROOT / "config.yaml")
        table = c["SyncTable"].load(de.LABELS / "sync.csv")
        band = de.load_band_map()
        with open(de.LABELS / "truth.csv", encoding="utf-8", newline="") as f:
            truth = {r["pair"]: {"people": int(r["people"]), "expected": r["expected"], "notes": r["notes"], "set": r["set"]}
                     for r in c["csv"].DictReader(f)}
        run_dir = self.records_dir().name
        added = added_names() if self.profile.can_add else set()
        home = f"shoot{self.shoot}"
        (out / "pairs").mkdir(parents=True, exist_ok=True)
        (out / "cards").mkdir(parents=True, exist_ok=True)
        sets = [*self.profile.overrides, *(overrides or [])]
        opts = {"set": sets} if sets else {}
        if self.profile.sensor_lanes:
            opts.update(sensor=True, lanes=self.profile.sensor_lanes)
        if self.profile.payments:
            opts["payments"] = True
        for pair in todo:
            self.state["current"] = pair
            try:
                built = rx.pair_data(RUN_KEY, run_dir, opts, pair, base, table, band, truth)
                if built is None:
                    continue
                data, result = built
                data["shoot"] = self.shoot
                data["labels"] = label_store.load(pair, self.events_dir())              # 사람이 적은 정답 (없으면 빈 것)
                data["payment_labels"] = label_store.load_payments(pair, self.payments_file())
                row = table.rows.get(pair)
                data["sync"] = {"offset_ms": row.offset_ms if row else 0.0, "method": row.method if row else "none",
                                "checked_by": row.checked_by if row else "", "note": row.note if row else ""}
                data["sensor_log"] = (sensor_log(self.profile.labels / "sensor" / f"{pair}.csv", data["sync"]["offset_ms"],
                                                 self.profile.sensor_lanes) if self.profile.sensor_lanes else None)   # None = 센서를 판정에 안 넣는 보기
                cfg = de.dual_config(base, pair)   # 방금 판정에 쓴 값 그대로 (적용한 값·초안 포함)
                # 결제 로그로 판정한 보기는 계산대 체류를 결제로 안 본다 (dual_config 는 --payments 를 모른다)
                dwell = cfg.risk.dwell_pay_seconds if cfg.risk.payment_mode == "dwell" and not self.profile.payments else None
                summary = analyze(data, result, dwell)
                data["summary"] = summary
                (out / "pairs" / f"{pair}.json").write_text(
                    json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
                item = card(summary)
                item["signature"] = self.signature(pair)
                item["batch"] = "added" if pair in added else home
                item["sync_method"] = data["sync"]["method"]
                item["thumb"] = self._thumb(pair, data)   # 장면 그림은 기록에만 달려 있어 초안도 같은 것을 쓴다
                item["score"] = {k: result[k] for k in ("correct", "correct_last", "no_verdict", "people_ok",
                                                         "expected_source")}
                (out / "cards" / f"{pair}.json").write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
            except Exception as e:  # noqa: BLE001 — 한 짝이 실패해도 나머지는 만든다. 화면에 그대로 보인다
                self.state["errors"].append(f"{pair}: {e!r}")
            self.state["done"] += 1
            if main:
                self._write_index()   # 만드는 동안에도 다 된 짝부터 갤러리에 보인다

    def _thumb(self, pair: str, data: dict) -> str | None:
        """전체 카메라 영상의 한 장면을 작은 그림으로. 손님이 가장 크게 보이는 때를 고른다."""
        path = self.out / "thumbs" / f"{pair}.jpg"
        done = self.records_dir() / pair / ".done"
        if path.exists() and done.exists() and path.stat().st_mtime >= done.stat().st_mtime:
            return f"thumbs/{pair}.jpg"   # 기록이 그대로면 다시 만들지 않는다 (영상을 처음부터 읽어야 해서 느리다)
        ready = self.profile.thumbs / f"{pair}.jpg" if self.profile.thumbs else None
        if ready is not None and ready.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ready, path)
            return f"thumbs/{pair}.jpg"
        try:
            import cv2
            track = data["tracks"]["angle1"]
            best, best_area = None, 0
            for key, rows in track["frames"].items():
                area = sum((r[5] - r[3]) * (r[6] - r[4]) for r in rows if r[1] > 0)
                if area > best_area:
                    best, best_area = int(key), area
            video = ROOT.parents[1] / data["video"]["angle1"].lstrip("/")
            cap = cv2.VideoCapture(str(video))
            target = best if best is not None else track["frames_total"] // 2
            image = None
            # 이 영상(H.265)은 위치 이동이 어긋나므로 처음부터 차례로 읽는다
            for i in range(target + 1):
                ok = cap.grab()
                if not ok:
                    break
                if i == target:
                    ok, image = cap.retrieve()
            cap.release()
            if image is None:
                return None
            h, w = image.shape[:2]
            image = cv2.resize(image, (480, int(480 * h / w)))
            path.parent.mkdir(parents=True, exist_ok=True)
            ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])   # 한글 경로라 imwrite 대신
            if ok:
                path.write_bytes(buf.tobytes())
            return f"thumbs/{pair}.jpg" if ok else None
        except Exception:  # noqa: BLE001 — 그림이 없어도 카드는 보인다
            return None

    def _write_index(self) -> None:
        cards = []
        for pair in self.pairs():
            path = self.out / "cards" / f"{pair}.json"
            if path.exists():
                cards.append(json.loads(path.read_text(encoding="utf-8")))
        home = f"shoot{self.shoot}"
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "index.json").write_text(json.dumps(
            {"version": VERSION, "built": time.strftime("%Y-%m-%d %H:%M"), "records": self.records_dir().name,
             "payment": self.profile.payment, "cards": cards, "totals": totals(cards),
             "batches": [{"key": key, "name": name, "totals": totals([c for c in cards if c.get("batch", home) == key])}
                         for key, name in BATCH_NAME.items() if any(c.get("batch", home) == key for c in cards)]},
            ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="바뀌지 않은 짝도 다시 만든다")
    ap.add_argument("--only", help="짝 이름 (쉼표)")
    ap.add_argument("--shoot", type=int, choices=sorted(PROFILES), default=2,
                    help=f"보기 번호 — 2 · 3 · 4 는 촬영 회차 (3차 · 4차는 상품 모델 기록, 4차는 센서 + 결제 로그 판정), "
                         f"{SHOOT3_PREV} 은 3차의 바로 전 판정 (기본 2)")
    ap.add_argument("--prune", action="store_true", help="없앤 옛 보기가 남긴 보기 데이터 폴더를 지우고 끝낸다 (다시 만들 수 있는 것만 들어 있다)")
    args = ap.parse_args()
    if args.prune:
        for d in stale_outputs():
            shutil.rmtree(d)
            print("지움:", d)
        return 0
    b = Builder(args.shoot)
    state = b.build(args.only.split(",") if args.only else None, args.force)
    print(f"만든 짝 {state['done']}/{state['total']}, {state['seconds']}초 → {b.out}")
    for e in state["errors"]:
        print("  실패:", e)
    return 1 if state["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
