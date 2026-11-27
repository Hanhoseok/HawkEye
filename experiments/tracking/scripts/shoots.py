"""촬영 회차별 경로 · 값 — 3차부터 (두 카메라 모두 <짝>.mp4, area_setting 장면 기준 구역, 영상마다 카메라 맞추기).

3차와 4차는 같은 매장 · 같은 방식이고 카메라 자리와 날짜만 다르다. 회차마다 달라지는 것(영상 폴더 · 구역 폴더 · 라벨 폴더 ·
바닥 표시)을 여기 한 곳에 둔다. 2차(1(angle*), 이름 규칙이 다름)는 scripts/sync_pair.py 가 따로 다룬다.

    3차 2026-11-11  source/2(angle0) · 2(angle1)   바닥 표시 = 노란 메모지 4장 (64×74cm 로 가정)
    4차 2026-11-19  source/3(angle0) · 3(angle1)   바닥 표시 = 흰 테이프 4장. 실측값이 없어 3차 바닥 좌표로 옮겨 씀
                    (3차 구역 cm 를 그대로 쓴다). 실제 결제 로그 · 센서 로그가 있다 (source/sensor_payment_log).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent.parent / "source"
LANE_KIND = {"L1": "cantata", "L2": "donggwaru", "L3": "applecider"}
"""센서 레인 → 그 레인의 상품 종류 (상품 탐지 모델의 이름). 3차 · 4차 진열이 같다."""
NO_SENSOR = "센서 로그는 4차에만 해당한다 — {shoot}차는 센서 없이 판정한다 (사용자 2026-11-27, 후속 84)"


@dataclass(frozen=True)
class Shoot:
    number: int
    date: str
    angle0: Path
    """선반 카메라 영상 폴더."""
    angle1: Path
    """전체 카메라 영상 폴더."""
    zones: Path
    labels: Path
    sync_out: Path
    """카메라 맞추기(register) · 소리 시차 · 발 캐시 · 확인 그림 (outputs/ 아래, 저장소에 없음)."""
    marker: str
    """바닥 표시 색: yellow(메모지) / white(테이프)."""
    rough: dict = field(default_factory=dict)
    """표시 P1~P4 의 대략 화면 위치 (1920×1080 픽셀). P1 문 쪽·가구 쪽, P2 문 쪽·방 쪽, P3 안쪽·가구 쪽, P4 안쪽·방 쪽."""
    floor_cm: tuple = ()
    """표시 P1~P4 의 바닥 cm."""
    count_source: str = "overview"
    """선반 방문의 개수를 어느 카메라에서 (fusion.count_source): overview = 전체 카메라가 센 개수, shelf = 선반 카메라의 선반 지도.
    **카메라 자리에 맞춘 값** — 카메라를 옮기면 정상(N) 영상으로 다시 고른다 (scripts/count_source_check.py)."""
    top_merge: float = 0.0
    """선반 지도가 같은 물건의 상자를 윗변 거리로 묶는 기준 (shelves.top_merge, 0 = 예전대로 가운데 거리).
    **카메라 자리 · 진열에 맞춘 값** — 4차 0.18: 정상(N) 영상의 상자 윗변 거리 분포에서 골짜기(0.12~0.20)를 골랐다. 다만 켤지 말지는
    정상 영상만으로는 못 갈랐다 (선반 지도가 맞은 정상 영상 11 / 13 → 11 / 13: N10-1 맞아짐 · N04 틀려짐). 42편 판정 점수가 올라서 켰다 (후속 61).
    3차는 꺼 둔다 (켜면 선반 지도가 맞는 영상 34 → 30 / 39)."""
    sensor: bool = False
    """센서 로그를 판정에 넣을 수 있는 회차인가. **4차만** (사용자 2026-11-27, 후속 84 — "센서로그는 4차에만 해당").
    3차의 labels/store01_shoot3/sensor/ 는 라벨 도구의 기록으로 남아 있지만 판정 · 점수 · 회귀 검사 · 분석 도구에는 안 쓴다."""

    def video(self, pair: str) -> tuple[Path, Path]:
        """(선반 카메라 파일, 전체 카메라 파일)."""
        return self.angle0 / f"{pair}.mp4", self.angle1 / f"{pair}.mp4"

    def pairs(self) -> list[str]:
        return sorted(p.stem for p in self.angle1.glob("*.mp4")
                      if p.stem != "area_setting" and (self.angle0 / p.name).exists())


SHOOTS = {
    3: Shoot(3, "2026-11-11", SOURCE / "2(angle0)", SOURCE / "2(angle1)",
             ROOT / "zones" / "store01_shoot3", ROOT / "labels" / "store01_shoot3", ROOT / "outputs" / "sync_floor",
             "yellow",
             {"angle0": [(432, 820), (349, 1025), (1135, 785), (1213, 1037)],
              "angle1": [(1125, 604), (942, 604), (1192, 1070), (847, 1068)]},
             ((0.0, 0.0), (64.0, 0.0), (0.0, 74.0), (64.0, 74.0))),
    4: Shoot(4, "2026-11-19", SOURCE / "3(angle0)", SOURCE / "3(angle1)",
             ROOT / "zones" / "store01_shoot4", ROOT / "labels" / "store01_shoot4", ROOT / "outputs" / "sync_floor_shoot4",
             "white",
             {"angle0": [(322, 833), (245, 1024), (1094, 755), (1153, 905)],
              "angle1": [(1011, 602), (883, 607), (1064, 1034), (868, 1042)]},
             # 테이프의 바닥 cm = 3차 바닥 좌표로 옮긴 값. 3차와 4차 선반 카메라 area_setting 의 바닥 무늬(타일 줄눈)를 특징점으로 맞춰
             # (일치 81/102) 테이프 자리를 3차 화면으로 옮기고 3차 바닥 지도로 cm 를 읽었다. 전체 카메라로 검산: 몇 픽셀 안.
             ((30.8, -11.5), (86.1, -10.6), (11.4, 71.3), (52.8, 72.7)),
             # 4차 자리에서는 전체 카메라가 칸타타 2개 중 1개만 보고 종류를 섞어 본다 — 선반 지도가 맞았다 (정상 영상으로 고름, 후속 61)
             count_source="shelf", top_merge=0.18, sensor=True),
}
LATEST = max(SHOOTS)


def get(number: int) -> Shoot:
    if number not in SHOOTS:
        raise ValueError(f"촬영 회차 {number} 는 없다 — {sorted(SHOOTS)} 중 하나")
    return SHOOTS[number]
