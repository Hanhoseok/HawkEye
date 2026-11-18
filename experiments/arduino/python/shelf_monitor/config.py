"""config.toml 로드와 검증."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

MAX_LANES = 5


class ConfigError(Exception):
    pass


@dataclass
class LaneConfig:
    lane_id: int
    enabled: bool
    product_id: str
    product_name: str
    thickness_mm: float
    empty_distance_mm: float
    max_count: int
    tca_channel: int = 0
    valid_min_mm: float = 30
    valid_max_mm: float = 1200
    tolerance_mm: float = 15
    hysteresis_mm: float = 10
    stable_time_ms: int = 1200
    min_stable_samples: int = 5
    stable_spread_mm: float = 10
    median_window: int = 5
    max_sample_gap_ms: int = 600
    error_invalid_count: int = 5
    error_invalid_ms: int = 1500
    count_distance_mm: Optional[list[float]] = None
    count_tolerance_mm: Optional[list[float]] = None   # 수량별 허용 오차 (없으면 모두 tolerance_mm)
    shape: str = "box"
    block_mm: float = 0
    """밀판 앞 고정 블록(스페이서) 길이. 첫 상품은 밀판을 (thickness_mm - block_mm) 만 밀고, 둘째부터 thickness_mm 씩 민다.
    실측 거리표(count_distance_mm)가 없는 레인에서만 쓴다 — 보정하면 거리표가 우선한다. 0 = 블록 없음."""

    def first_step_mm(self) -> float:
        """첫 상품이 밀판을 미는 거리 (0개 → 1개 기준 거리 차이)."""
        return self.thickness_mm - self.block_mm

    def reference_mm(self, n: int) -> float:
        """수량 n 일 때 기대되는 센서→밀판 거리."""
        if self.count_distance_mm:
            return float(self.count_distance_mm[n])
        if self.block_mm > 0 and n >= 1:
            return self.empty_distance_mm - self.first_step_mm() - (n - 1) * self.thickness_mm
        return self.empty_distance_mm - n * self.thickness_mm

    def tolerance_for(self, n: int) -> float:
        """수량 n 의 허용 오차. 예: 첫 상품이 밀판을 조금만 미는 레일은 0개·1개만 좁게 준다."""
        if self.count_tolerance_mm:
            return float(self.count_tolerance_mm[n])
        if self.block_mm > 0 and not self.count_distance_mm and n <= 1:
            # 0개·1개 간격이 좁다 → 두 허용 구간과 히스테리시스가 그 간격 안에 들어가게 나눈다
            return min(self.tolerance_mm, max(1.0, (self.first_step_mm() - self.hysteresis_mm - 0.5) / 2))
        return self.tolerance_mm

    def min_spacing_mm(self) -> float:
        refs = [self.reference_mm(n) for n in range(self.max_count + 1)]
        return min((refs[i] - refs[i + 1] for i in range(self.max_count)), default=float("inf"))

    def validate(self) -> list[str]:
        e: list[str] = []
        p = f"lane {self.lane_id}"
        if not 1 <= self.lane_id <= MAX_LANES:
            e.append(f"{p}: lane_id 는 1..{MAX_LANES}")
        if self.max_count < 1:
            e.append(f"{p}: max_count >= 1 이어야 함")
            return e
        if not 0 <= self.tca_channel <= 7:
            e.append(f"{p}: tca_channel 은 0..7")
        if self.count_distance_mm is not None:
            t = self.count_distance_mm
            if len(t) != self.max_count + 1:
                e.append(f"{p}: count_distance_mm 길이는 max_count+1 ({self.max_count + 1}) 이어야 함 (현재 {len(t)})")
                return e
            if any(t[i] <= t[i + 1] for i in range(len(t) - 1)):
                e.append(f"{p}: count_distance_mm 는 수량이 늘수록 엄격히 감소해야 함 (밀판이 센서에 가까워짐)")
                return e
        elif self.thickness_mm <= 0:
            e.append(f"{p}: thickness_mm > 0 이어야 함")
            return e
        elif self.block_mm < 0 or (self.block_mm > 0 and self.first_step_mm() < 5):
            e.append(f"{p}: block_mm 은 0 이상이고 thickness_mm 보다 5mm 넘게 작아야 함 "
                     f"(첫 상품이 밀판을 {self.first_step_mm():g}mm 만 밀면 0개·1개를 구분할 수 없음)")
            return e
        if self.count_tolerance_mm is not None:
            if len(self.count_tolerance_mm) != self.max_count + 1:
                e.append(f"{p}: count_tolerance_mm 길이는 max_count+1 ({self.max_count + 1}) 이어야 함 "
                         f"(현재 {len(self.count_tolerance_mm)})")
                return e
            if any(t <= 0 for t in self.count_tolerance_mm):
                e.append(f"{p}: count_tolerance_mm 값은 모두 > 0")
                return e
        if self.tolerance_mm <= 0 or self.hysteresis_mm < 0:
            e.append(f"{p}: tolerance_mm > 0, hysteresis_mm >= 0 이어야 함")
        # 확정 수량의 유지 구간(tol+hyst)과 이웃 수량의 진입 구간(tol)이 겹치지 않아야 한다.
        for n in range(self.max_count):
            gap = self.reference_mm(n) - self.reference_mm(n + 1)
            need = self.tolerance_for(n) + self.tolerance_for(n + 1) + self.hysteresis_mm
            if need >= gap:
                e.append(
                    f"{p}: {n}개와 {n + 1}개의 허용 오차 합 + hysteresis_mm ({need:g}) 가 두 기준 거리 간격 "
                    f"({gap:g} mm) 이상 → 허용 구간이 겹침. 값을 줄이세요."
                )
        near = self.reference_mm(self.max_count) - self.tolerance_for(self.max_count) - self.hysteresis_mm
        far = self.reference_mm(0) + self.tolerance_for(0) + self.hysteresis_mm
        if near < self.valid_min_mm:
            e.append(f"{p}: max_count 일 때 거리({self.reference_mm(self.max_count):g}) 가 valid_min_mm 에 너무 가까움 "
                     f"→ max_count 또는 empty_distance_mm 확인")
        if far > self.valid_max_mm:
            e.append(f"{p}: empty_distance_mm 가 valid_max_mm 를 넘음")
        if self.valid_min_mm >= self.valid_max_mm:
            e.append(f"{p}: valid_min_mm < valid_max_mm 이어야 함")
        if self.median_window < 1:
            e.append(f"{p}: median_window >= 1")
        if self.min_stable_samples < 2:
            e.append(f"{p}: min_stable_samples >= 2")
        if self.stable_time_ms < 200:
            e.append(f"{p}: stable_time_ms 가 너무 짧음 (>= 200)")
        if self.max_sample_gap_ms < 150:
            e.append(f"{p}: max_sample_gap_ms 가 너무 짧음 (5개 순차 측정 재방문 주기 약 200ms 보다 커야 함)")
        if self.shape not in ("box", "can"):
            e.append(f"{p}: shape 은 box 또는 can")
        return e


@dataclass
class AppConfig:
    source: str = "serial"
    udp_port: int = 4210
    port: str = "auto"
    baud: int = 115200
    device_id: str = ""
    link_stale_s: float = 3.0
    lane_stale_s: float = 3.0
    backlog_limit_ms: int = 1000
    resync_gap_ms: int = 5000
    status_interval_s: float = 5.0
    restock_timeout_s: float = 600
    event_log: str = "logs/events.jsonl"
    status_log: str = "logs/status.jsonl"
    lanes: list[LaneConfig] = field(default_factory=list)
    base_dir: Path = Path(".")
    config_path: Optional[Path] = None      # 보정 결과를 다시 쓸 파일

    def lane(self, lane_id: int) -> Optional[LaneConfig]:
        for l in self.lanes:
            if l.lane_id == lane_id:
                return l
        return None

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.base_dir / p


_LANE_KEYS = {f.name for f in fields(LaneConfig)}


def load_config(path: str | Path, enable_all: bool = False) -> AppConfig:
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    cfg = AppConfig(base_dir=path.resolve().parent, config_path=path.resolve())
    errors: list[str] = []
    n = raw.get("network", {})
    cfg.source = str(n.get("source", cfg.source))
    cfg.udp_port = int(n.get("udp_port", cfg.udp_port))
    if cfg.source not in ("serial", "wifi"):
        errors.append('network.source 는 "serial" 또는 "wifi"')
    if not 1 <= cfg.udp_port <= 65535:
        errors.append("network.udp_port 는 1..65535")
    s = raw.get("serial", {})
    cfg.port = str(s.get("port", cfg.port))
    cfg.baud = int(s.get("baud", cfg.baud))
    cfg.device_id = str(s.get("device_id", ""))
    t = raw.get("timing", {})
    for k in ("link_stale_s", "lane_stale_s", "status_interval_s", "restock_timeout_s"):
        if k in t:
            setattr(cfg, k, float(t[k]))
    for k in ("backlog_limit_ms", "resync_gap_ms"):
        if k in t:
            setattr(cfg, k, int(t[k]))
    o = raw.get("output", {})
    cfg.event_log = str(o.get("event_log", cfg.event_log))
    cfg.status_log = str(o.get("status_log", cfg.status_log))

    defaults = raw.get("lane_defaults", {})
    for k in defaults:
        if k not in _LANE_KEYS:
            errors.append(f"lane_defaults: 알 수 없는 키 '{k}'")
    for i, lane_raw in enumerate(raw.get("lanes", [])):
        merged = {**defaults, **lane_raw}
        unknown = set(merged) - _LANE_KEYS
        if unknown:
            errors.append(f"lanes[{i}]: 알 수 없는 키 {sorted(unknown)}")
            continue
        try:
            lane = LaneConfig(**merged)
        except TypeError as ex:
            errors.append(f"lanes[{i}]: {ex}")
            continue
        if enable_all:
            lane.enabled = True
        cfg.lanes.append(lane)

    ids = [l.lane_id for l in cfg.lanes]
    if len(ids) != len(set(ids)):
        errors.append("lane_id 중복")
    chans = [l.tca_channel for l in cfg.lanes if l.enabled]
    if len(chans) != len(set(chans)):
        errors.append("활성 레인끼리 tca_channel 중복")
    if len(cfg.lanes) > MAX_LANES:
        errors.append(f"레인은 최대 {MAX_LANES}개")
    for lane in cfg.lanes:
        if lane.enabled:
            errors.extend(lane.validate())
    if not any(l.enabled for l in cfg.lanes):
        errors.append("활성화된 레인이 없음")
    if errors:
        raise ConfigError("\n".join(errors))
    cfg.lanes.sort(key=lambda l: l.lane_id)
    return cfg


def _toml_value(v) -> str:
    import json
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)          # TOML 기본 문자열과 이스케이프가 같다
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return str(v)


def update_lane_in_toml(path: str | Path, lane_id: int, values: dict, note: str = "",
                        drop_comments: bool = True) -> Path:
    """config.toml 에서 lane_id 레인 블록의 키만 바꾼다. 다른 레인과 주석은 그대로 둔다.

    바꾸기 전 파일은 config.toml.bak 으로 남기고, 바꾼 뒤 전체를 다시 검사해 실패하면 되돌린다.
    drop_comments: 보정 저장 때는 그 레인 블록의 주석(이전 보정 설명)을 지우고 note 한 줄로 대신한다.
    이름만 바꿀 때는 False 로 두어 주석을 남긴다.
    """
    import re
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines) if l.strip() == "[[lanes]]"]
    block = None
    for k, s in enumerate(starts):
        e = next((j for j in range(s + 1, len(lines)) if lines[j].lstrip().startswith("[")), len(lines))
        if any(re.match(rf"\s*lane_id\s*=\s*{lane_id}\s*(#.*)?$", lines[j]) for j in range(s + 1, e)):
            block = (s, e)
            break
    if block is None:
        raise ConfigError(f"config.toml 에 lane_id = {lane_id} 레인이 없습니다")
    s, e = block
    body = [l for l in lines[s + 1:e] if not (drop_comments and l.lstrip().startswith("#"))]
    while body and not body[-1].strip():
        body.pop()
    done = set()
    for i, l in enumerate(body):
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", l)
        if m and m.group(1) in values:
            key = m.group(1)
            body[i] = f"{key} = {_toml_value(values[key])}"
            done.add(key)
        elif m and m.group(1) == "enabled" and drop_comments:
            body[i] = "enabled = true"
    for key, v in values.items():
        if key not in done:
            body.append(f"{key} = {_toml_value(v)}")
    if note:
        lid = next((i for i, l in enumerate(body) if re.match(r"\s*lane_id\s*=", l)), 0)
        body.insert(lid + 1, f"# {note}")
    new_lines = lines[:s + 1] + body + [""] + lines[e:]
    while len(new_lines) > 1 and not new_lines[-1].strip() and not new_lines[-2].strip():
        new_lines.pop()
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(text, encoding="utf-8")
    path.write_text("\n".join(new_lines).rstrip("\n") + "\n", encoding="utf-8")
    try:
        load_config(path)
    except ConfigError:
        path.write_text(text, encoding="utf-8")
        raise
    return backup
