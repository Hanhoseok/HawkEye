"""가짜 선반 센서 기록 — 센서가 오기 전에 ⑦-b 선반 사건 묶기를 시험한다 (설계 M-8, 1-6).

실제 센서가 없으므로 '센서가 있었다면 이렇게 찍혔을 것'을 정답표(장면표로 눈으로 본 집은 시각)로 만든다.
형식은 실제 센서와 같게 약속한 것: time_sec(기준 카메라 시간축), slot(선반 칸), delta_g(무게 변화), confidence.

    python scripts/fake_sensor.py            # 아래 SPEC 의 짝들에 대해 labels/store01_dual/fake_sensor/<짝>.csv
    python scripts/dual_eval.py judge --only N1,T1,T8 --sensor

센서 하나 무게(g)는 configs/store01_dual.yaml 의 fusion.sensor_unit_g 와 같게 둔다 (여기서는 칸마다 가정값).
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "labels" / "store01_dual" / "fake_sensor"

UNIT_G = {"card": 45.0, "bottle": 520.0, "box": 160.0}
"""칸별 물건 하나 무게 가정값 (실측 아님)."""

SPEC = {
    # 짝: [(시각 초, 칸, 개수 변화)]  — 음수 = 빠짐. 시각은 장면표(1초 간격)로 본 값이라 ±1초 오차가 있다.
    "N1": [(6.5, "card", -1)],
    "T1": [(8.0, "card", -1)],
    "T8": [(6.0, "card", -1), (20.0, "card", -1)],
    "N8-1": [(5.0, "card", -1), (16.0, "card", +1)],   # 결제 뒤 선반으로 돌아가 도로 놓음 (장면표 해석)
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for pair, events in SPEC.items():
        path = OUT / f"{pair}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_sec", "slot", "delta_g", "confidence"])
            for t, slot, n in events:
                w.writerow([t, slot, round(n * UNIT_G[slot], 1), 1.0])
        print(f"{pair}: {len(events)}건 → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
