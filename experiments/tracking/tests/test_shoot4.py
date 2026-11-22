"""4차 촬영(2026-11-19)을 붙인 것들 — 회차별 경로(scripts/shoots.py), 결제 소리, 소리만으로 시차, 결제 로그 나누기.

영상 · 로그는 저장소에 없다 (source/). 여기서는 만든 소리와 가짜 로그로 규칙만 본다.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import dual_eval as DE  # noqa: E402
import pos_payments as PP  # noqa: E402
import shoot3_eval as S3  # noqa: E402
import shoot3_zones as Z  # noqa: E402
import shoots  # noqa: E402
import sync_floor as SF  # noqa: E402
import sync_pair as SP  # noqa: E402
from src.inputs import audio_read as AR  # noqa: E402

SR = AR.SR


def tone(hz: float, seconds: float, amp: float = 3000.0) -> np.ndarray:
    return amp * np.sin(2 * np.pi * hz * np.arange(int(seconds * SR)) / SR)


def clip(seconds: float, chimes_at: list[float], seed: int = 0, extra: list[tuple[float, float]] | None = None) -> np.ndarray:
    """잡음 위에 결제 소리(984Hz 0.12초 → 1312Hz 0.12초)를 넣은 소리. extra = [(초, 주파수)] 한 음짜리 다른 소리."""
    x = np.random.default_rng(seed).normal(0.0, 40.0, int(seconds * SR))
    for t in chimes_at:
        a = int(t * SR)
        both = np.concatenate([tone(984.0, 0.12), tone(1312.0, 0.12)])
        x[a:a + len(both)] += both
    for t, hz in extra or []:
        a = int(t * SR)
        x[a:a + int(0.12 * SR)] += tone(hz, 0.12)
    return x


# ---------------------------------------------------------------------- 회차별 경로

def test_shoot_profiles_point_to_their_own_folders():
    s3, s4 = shoots.get(3), shoots.get(4)
    assert (s3.angle0.name, s3.angle1.name) == ("2(angle0)", "2(angle1)")
    assert (s4.angle0.name, s4.angle1.name) == ("3(angle0)", "3(angle1)")
    assert s4.zones.name == s4.labels.name == "store01_shoot4" and s3.sync_out != s4.sync_out
    assert len(s4.rough["angle0"]) == len(s4.rough["angle1"]) == len(s4.floor_cm) == 4
    with pytest.raises(ValueError):
        shoots.get(5)


def test_sync_pair_names_for_shoot4():
    video, ref = SP.pair_files("N10-2", shoot=4)
    assert (SP.rel(video), SP.rel(ref)) == ("3(angle0)/N10-2.mp4", "3(angle1)/N10-2.mp4")


def test_profile_switch_moves_every_path_and_comes_back():
    try:
        DE.set_profile(4)
        assert DE.SHOOT == 4 and DE.OUT.name == "dual_eval_shoot4"
        assert DE.ZONES.name == DE.LABELS.name == "store01_shoot4" and DE.JUDGE_FILE.name == "store01_shoot4_judge.yaml"
        assert S3.BASE.name == "store01_shoot4" and Z.REGISTER.parent.name == "sync_floor_shoot4" and Z.VIDEOS.name == "3(angle0)"
        DE.set_profile(3)   # 분석 도구는 한 프로그램 안에서 회차를 오간다 — 카메라 맞추기 표도 따라와야 한다
        assert DE.OUT.name == "dual_eval_shoot3_prod" and DE.LABELS.name == "store01_shoot3"
        assert S3.BASE.name == "store01_shoot3" and Z.REGISTER.parent.name == "sync_floor" and Z.VIDEOS.name == "2(angle0)"
    finally:
        DE.set_profile(2)
    assert DE.ZONES.name == "store01_dual"


def test_sync_floor_switches_shoot_and_back():
    try:
        SF.set_shoot(4)
        assert SF.CAMS["angle1"].name == "3(angle1)" and SF.MARKER == "white"
        assert SF.REGISTER == SF.OUT / "register.csv" and SF.OUT.name == "sync_floor_shoot4"
    finally:
        SF.set_shoot(3)
    assert SF.CAMS["angle1"].name == "2(angle1)" and SF.MARKER == "yellow" and SF.FLOOR_CM[3] == (64.0, 74.0)


def test_shoot4_zone_files_share_the_shoot3_floor_frame():
    """4차 바닥 지도는 3차 바닥 좌표로 적는다 — 그래야 3차의 구역 cm 를 그대로 쓴다."""
    import yaml

    z3, z4 = shoots.get(3).zones, shoots.get(4).zones
    f3 = yaml.safe_load((z3 / "floor.yaml").read_text(encoding="utf-8"))
    f4 = yaml.safe_load((z4 / "floor.yaml").read_text(encoding="utf-8"))
    assert {z["name"]: z["polygon"] for z in f3["zones"]} == {z["name"]: z["polygon"] for z in f4["zones"]}
    assert f3["band"] == f4["band"]
    for cam in ("angle0", "angle1"):
        cal = yaml.safe_load((z4 / f"floor_{cam}.yaml").read_text(encoding="utf-8"))["calibration"]
        assert [tuple(p) for p in cal["floor"]] == list(shoots.get(4).floor_cm)
    assert f4["calibration"]["image"] == yaml.safe_load((z4 / "floor_angle1.yaml").read_text(encoding="utf-8"))["calibration"]["image"]
    lanes = yaml.safe_load((z4 / "angle0.yaml").read_text(encoding="utf-8"))
    assert [z["name"] for z in lanes["zones"]] == ["L1", "L2", "L3"]
    box = lanes["shelves"][0]["box"]
    for z in lanes["zones"]:   # 레인은 받침 상자 안
        assert all(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in z["polygon"])


def test_unknown_pairs_are_left_out_of_the_score():
    """정답표에도 없고 이름으로도 종류를 모르는 짝(4차 L1)은 정상으로 치지 않고 채점에서 뺀다."""
    assert DE.kind("L1") == "기타" and DE.kind("N01") == "정상"
    rows = []
    for pair, finals, truth in (("N01", [(1, "CLEAR")], None), ("T01", [(1, "HIGH_RISK")], None), ("L1", [(1, "HIGH_RISK")], None),
                                ("S1", [(1, "CLEAR")], {"S1": {"expected": "HIGH_RISK", "people": 1, "set": "theft"}})):
        rows.append({"pair": pair, "kind": DE.kind(pair), **DE.score(pair, finals, 1, truth)})
    s = DE.summarize(rows)
    assert s["unscored"] == ["L1"]
    assert s["전체"]["n"] == 3 and s["전체"]["correct"] == 2 and s["전체"]["wrong"] == ["S1"]   # S1 은 정답표에 있어 채점한다


# ---------------------------------------------------------------------- 결제 소리

def test_chime_is_found_at_its_start():
    got = AR.chimes(clip(20.0, [5.0, 12.5]))
    assert len(got) == 2
    assert got[0] == pytest.approx(5.0, abs=0.05) and got[1] == pytest.approx(12.5, abs=0.05)


def test_single_tones_and_silence_are_not_chimes():
    assert AR.chimes(clip(10.0, [])) == []
    assert AR.chimes(clip(10.0, [], extra=[(3.0, 984.0)])) == []        # 낮은 음만
    assert AR.chimes(clip(10.0, [], extra=[(3.0, 1312.0)])) == []       # 높은 음만
    assert AR.chimes(clip(10.0, [], extra=[(3.0, 3400.0)])) == []       # 다른 삑 소리 (3차 바코드)
    assert AR.chimes(np.zeros(100)) == []


def test_read_audio_gives_empty_for_missing_file(tmp_path, monkeypatch):
    import subprocess

    def boom(*a, **k):
        raise FileNotFoundError("ffmpeg")
    monkeypatch.setattr(subprocess, "run", boom)
    assert len(AR.read_audio(tmp_path / "없는 영상.mp4")) == 0


# ---------------------------------------------------------------------- 소리만으로 시차

def test_sound_decision_prefers_audio_checked_by_chime():
    assert SF.sound_decision(-95.0, 1.9, [-80.0], 1021.0)[:2] == (-95.0, "audio+chime")
    off, method, why = SF.sound_decision(700.0, 1.9, [-80.0, -64.0], 0.0)      # 소리 시차가 결제 소리와 다르면 결제 소리
    assert (off, method) == (-72.0, "chime") and "다름" in why
    assert SF.sound_decision(-22.0, 1.09, [-16.0], 0.0)[:2] == (-16.0, "chime")   # 소리 피크가 약해도 결제 소리가 있으면


def test_sound_decision_without_chime_needs_peak_and_clock():
    assert SF.sound_decision(-72.0, 1.48, [], 1107.0)[:2] == (-72.0, "audio")     # 녹화 시각은 1초 칸 — 1.6초 안이면 통과
    assert SF.sound_decision(-72.0, 1.48, [], 4000.0)[1] == "audio-low"
    assert SF.sound_decision(-36.0, 1.16, [], 0.0)[:2] == (0.0, "audio-low")      # 피크 약함 → 녹화 시각 값으로, 확인 필요
    assert SF.sound_decision(float("nan"), 0.0, [], None)[:2] == (0.0, "audio-low")


def test_chime_offsets_pair_in_order_or_by_nearest():
    assert SF.chime_offsets([5.0, 12.0], [5.05, 12.03], 0.0) == pytest.approx([50.0, 30.0])
    assert SF.chime_offsets([5.0, 12.0], [12.03], 0.0) == pytest.approx([30.0])   # 한쪽이 하나 놓침 — 가까운 것끼리
    assert SF.chime_offsets([5.0], [], 0.0) == []


def test_white_markers_are_used_as_given(tmp_path, monkeypatch):
    """흰 테이프는 자동으로 못 찾는다 — 적어 둔 자리 그대로 바닥 지도를 쓴다."""
    from src.zones.floor import FloorMap

    try:
        SF.set_shoot(4)
        monkeypatch.setattr(SF, "ZONES", tmp_path)
        monkeypatch.setattr(SF, "OUT", tmp_path)
        monkeypatch.setattr(SF, "median_frame", lambda path: np.full((1080, 1920, 3), 128, np.uint8))
        SF.calibrate(SF.FLOOR_CM, SF.ROUGH)
        fm = FloorMap.load(tmp_path / "floor_angle0.yaml")
        assert fm.image_points[0] == pytest.approx((322 / 1920, 833 / 1080), abs=1e-4)
        assert fm.to_floor(*fm.image_points[2]) == pytest.approx(SF.FLOOR_CM[2], abs=0.1)
    finally:
        SF.set_shoot(3)


# ---------------------------------------------------------------------- 결제 로그 나누기

def test_orders_group_lines_and_count_items(tmp_path):
    log = tmp_path / "payment-log.csv"
    with log.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["주문번호", "결제시각", "계산대ID", "품목순번", "바코드", "상품명", "단가", "수량", "품목금액", "주문총액", "주문수량합"])
        w.writerow(["12", "2026-11-19 17:24:55", "X", "1", "111", "칸타타 275ml", "2000", "1", "2000", "5500", "3"])
        w.writerow(["12", "2026-11-19 17:24:55", "X", "2", "222", "동과루", "1500", "2", "3000", "5500", "3"])
        w.writerow(["9", "2026-11-19 17:21:29", "X", "1", "111", "칸타타 275ml", "2000", "1", "2000", "2000", "1"])
    orders = PP.load_orders(log)
    assert [o.number for o in orders] == ["9", "12"]            # 시간 순
    assert orders[1].items == 3 and orders[1].text() == "칸타타 275ml×1, 동과루×2"
    assert PP.lane_of("칸타타 275ml") == "L1" and PP.lane_of("동과루") == "L2" and PP.lane_of("삼다수 생수") == ""


def test_orders_match_chimes_in_time_order():
    assert PP.match_chimes([12.0], [13.7]) == [13.7]
    assert PP.match_chimes([11.0, 29.0], [12.9, 30.7]) == [12.9, 30.7]
    assert PP.match_chimes([11.0, 29.0], [30.7]) == [None, 30.7]        # 첫 주문의 소리를 놓침 — 둘째 것을 당겨 쓰지 않는다
    assert PP.match_chimes([12.0], [11.0]) == [None]                    # 소리가 로그보다 먼저면 짝이 아니다
    assert PP.match_chimes([12.0, 13.0], [13.5]) == [13.5, None]        # 소리 하나를 두 주문에 쓰지 않는다
