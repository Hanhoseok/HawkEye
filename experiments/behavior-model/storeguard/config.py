"""YAML 설정 로더.

- 점 표기로 접근: cfg["train.batch_size"] / cfg.get("alert.per_class.fall.threshold")
- `--set a.b=1` 형태의 오버라이드 지원
- 학습 시 사용한 설정 전체를 체크포인트 옆에 스냅샷으로 저장한다(재현용)
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class Config:
    def __init__(self, data: dict):
        self._d = data

    # --- 생성 ---
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG
        data = _load_with_base(path)
        cfg = cls(data)
        for ov in overrides or []:
            if "=" not in ov:
                raise ValueError(f"--set 형식은 key=value: {ov}")
            k, v = ov.split("=", 1)
            cfg.set(k.strip(), _coerce(v.strip()))
        cfg._d.setdefault("_meta", {})["config_path"] = str(path)
        return cfg

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(copy.deepcopy(d))

    # --- 접근 ---
    def __getitem__(self, key: str) -> Any:
        cur: Any = self._d
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"설정 키 없음: {key}")
            cur = cur[part]
        return cur

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        cur = self._d
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = value

    def path(self, key: str) -> Path:
        return Path(str(self[key]))

    @property
    def raw(self) -> dict:
        return self._d

    # --- 저장 ---
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self._d, fh, allow_unicode=True, sort_keys=False)

    def to_json(self) -> str:
        return json.dumps(self._d, ensure_ascii=False, indent=2, default=str)

    # --- 자주 쓰는 파생값 ---
    @property
    def class_names(self) -> list[str]:
        return list(self["classes.names"])

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.class_names)}

    @property
    def action_classes(self) -> list[str]:
        """**알림을 낼** 클래스 (이상행동). 구매행동은 정상이므로 여기 들어가지 않는다.

        모델은 구매행동까지 전부 예측하지만, AlertEngine 은 이 목록만 감시한다.
        """
        alerting = self.get("classes.alerting")
        if alerting:
            return [c for c in alerting if c in self.class_names]
        return [c for c in self.class_names if c != "background_unlabeled"]

    @property
    def normal_classes(self) -> list[str]:
        """알림을 내지 않는 정상 행동 클래스 (구매행동 등). 대시보드에 상태로만 표시한다."""
        alerting = set(self.action_classes)
        return [c for c in self.class_names
                if c != "background_unlabeled" and c not in alerting]

    def alert_params(self, cls: str) -> dict:
        d = dict(self.get("alert.defaults", {}))
        d.update(self.get(f"alert.per_class.{cls}", {}) or {})
        return d

    def device(self) -> str:
        want = str(self.get("runtime.device", "auto"))
        if want != "auto":
            return want
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def _deep_merge(base: dict, over: dict) -> dict:
    """over 를 base 위에 재귀적으로 덮어쓴다. 리스트는 통째로 교체한다."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_with_base(path: Path, _seen: set | None = None) -> dict:
    """`_base: 다른파일.yaml` 을 따라가 재귀적으로 병합한다.

    설정 파일이 여러 개(로컬/Colab 등)로 갈라질 때 값이 서로 어긋나는 것을 막는다.
    바뀌는 값만 적고 나머지는 base 를 그대로 쓴다.
    """
    path = Path(path).resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"설정 _base 순환 참조: {path}")
    _seen.add(path)

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    base_ref = data.pop("_base", None)
    if not base_ref:
        return data
    base_path = (path.parent / str(base_ref)).resolve()
    return _deep_merge(_load_with_base(base_path, _seen), data)


def _coerce(v: str) -> Any:
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.startswith("[") or v.startswith("{"):
        try:
            return json.loads(v)
        except Exception:
            pass
    return v


def env_report() -> dict:
    """재현성 기록용 실행 환경 스냅샷."""
    import platform
    import sys

    rep: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
    }
    try:
        import torch
        rep["torch"] = torch.__version__
        rep["cuda_available"] = torch.cuda.is_available()
        rep["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            rep["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            rep["gpu_total_mem_gb"] = round(props.total_memory / 1024**3, 2)
            rep["gpu_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:
        rep["torch_error"] = str(exc)
    try:
        import cv2
        rep["opencv"] = cv2.__version__
    except Exception as exc:
        rep["opencv_error"] = str(exc)
    return rep
