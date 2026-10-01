"""torch 없이 돌아가는 핵심 로직 자체 검증.

검증 대상: 설정 로딩, 파일명 파서, 클립 라벨 규칙, 알림 상태 기계, URL 마스킹.

PowerShell 예:
    D:\\computervision\\.venv\\Scripts\\python.exe -m storeguard.tools.selftest
"""
from __future__ import annotations

import sys
import traceback

from storeguard import naming
from storeguard.config import Config
from storeguard.data.clips import clips_for_video
from storeguard.infer.alerting import AlertEngine
from storeguard.utils import mask_url

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
    except AssertionError as exc:
        RESULTS.append((name, False, str(exc) or "assert 실패"))
    except Exception:
        RESULTS.append((name, False, traceback.format_exc(limit=3)))


# ---------- 1. 설정 ----------
def t_config():
    cfg = Config.load()
    assert cfg.class_names[0] == "background_unlabeled", "0번은 배경이어야 한다"
    assert cfg.action_classes == ["fall", "broken", "theft"], cfg.action_classes
    assert cfg["data.sample_fps"] == 3.0, "원천 영상이 3fps 이므로 샘플링도 3fps"
    p = cfg.alert_params("fall")
    assert p["min_hits"] <= cfg.get("alert.defaults.min_hits"), "전도는 지속 조건이 더 느슨해야 한다"
    p2 = cfg.alert_params("theft")
    assert p2["threshold"] >= cfg.get("alert.defaults.threshold"), "절도는 더 보수적이어야 한다"


# ---------- 2. 파일명 ----------
def t_naming():
    n = naming.parse_stem("C_3_7_1_BU_DYA_07-31_15-15-25_CA_RGB_DF2_M1")
    assert n.class_name == "fall", n.class_name
    assert n.store == "DYA" and n.store_kr == "대형유인매장A"
    assert n.camera == "CA" and n.actor_gender == "male"
    assert n.seconds == 15 * 3600 + 15 * 60 + 25
    assert n.dataset == "238-2", n.dataset
    assert naming.parse_stem("C_3_12_1_BU_SMA_09-01_10-00-00_CB_RGB_DF2_F2").class_name == "theft"
    try:
        naming.parse_stem("이상한_이름")
        raise AssertionError("예외가 나야 한다")
    except naming.NameParseError:
        pass


def t_naming_purchase():
    """238-1 구매행동 파일명은 토큰 수가 12/13/16 으로 제각각이다. 실제 파일에서 확인된 형태들."""
    cases = [
        # (파일명, 기대 클래스, 기대 날짜, 기대 시각, 배우 수, modality)
        ("C_1_1_11_BU_DYB_10-09_13-16-40_CE_DF1_F1_F1", "moving", "10-09", "13-16-40", 2, ""),
        ("C_1_1_20_BU_DYB_10-09_15-17-39_CF_RGB_DF1_F2_F2", "moving", "10-09", "15-17-39", 2, "RGB"),
        ("C_1_1_16_BU_DYA_08_19_15_46_22_CF_RGB_DF1_M3_M3", "moving", "08-19", "15-46-22", 2, "RGB"),
        ("C_2_2_1_BU_DYA_08-29_10-32-48_CB_RGB_DF1_M1_F13", "select", "08-29", "10-32-48", 2, "RGB"),
    ]
    for stem, cls, date, time, n_actor, modality in cases:
        n = naming.parse_stem(stem)
        assert n.class_name == cls, (stem, n.class_name)
        assert n.date == date, (stem, n.date)
        assert n.time == time, (stem, n.time)
        assert n.n_actors == n_actor, (stem, n.actors)
        assert n.modality == modality, (stem, n.modality)
        assert n.dataset == "238-1", (stem, n.dataset)
        assert n.crew.startswith("DF"), (stem, n.crew)
    # 구매행동 6종 코드가 모두 매핑되어야 한다
    got = {naming.FILENAME_CLASS_CODE[("2", c)] for c in ("2", "3", "4", "5", "6")}
    assert got == {"select", "test", "buying", "return", "compare"}, got


def t_class_split():
    """알림 대상(이상행동)과 정상 행동(구매행동)이 분리되어야 한다."""
    cfg = Config.load()
    names = cfg.class_names
    assert names[0] == "background_unlabeled"
    for c in ("fall", "broken", "theft", "moving", "select", "test",
              "buying", "return", "compare"):
        assert c in names, c
    assert cfg.action_classes == ["fall", "broken", "theft"], cfg.action_classes
    assert set(cfg.normal_classes) == {"moving", "select", "test", "buying",
                                       "return", "compare"}, cfg.normal_classes
    # 구매행동에는 알림 문구가 없어야 한다(실수로 알림이 나가지 않도록)
    for c in cfg.normal_classes:
        assert cfg.get(f"classes.alert_text.{c}") is None, c


# ---------- 3. 클립 라벨 규칙 ----------
def t_clips_positive_window():
    """이벤트 한가운데 anchor 는 양성이어야 한다."""
    row = {"stem": "v", "class_name": "fall", "label_frames": 180,
           "events": [{"start_frame": 100, "end_frame": 150}],
           "take_id": "t", "store": "S", "camera": "CA"}
    specs = clips_for_video(row, clip_len=16, frame_stride=1, pos_overlap_clip=0.5,
                            pos_overlap_event=0.8, bg_margin=6, stride_pos=1,
                            stride_bg=4, max_bg=8)
    pos = [s for s in specs if s.label == "fall"]
    bg = [s for s in specs if s.label == "background_unlabeled"]
    assert pos, "양성 클립이 하나도 없다"
    # anchor 130 (창 115~130) 은 전부 이벤트 안 → 양성
    assert any(s.anchor == 130 for s in pos), sorted(s.anchor for s in pos)[:5]
    # anchor 50 은 이벤트에서 멀다 → 배경 후보
    assert all(s.anchor < 100 or s.anchor > 150 for s in bg), "배경이 이벤트와 겹쳤다"
    assert len(bg) <= 8, "max_bg 를 넘었다"


def t_clips_short_event():
    """짧은 이벤트(13프레임)도 pos_overlap_event 규칙으로 살아남아야 한다."""
    row = {"stem": "v", "class_name": "fall", "label_frames": 180,
           "events": [{"start_frame": 100, "end_frame": 112}],
           "take_id": "t", "store": "S", "camera": "CA"}
    specs = clips_for_video(row, clip_len=16, frame_stride=1, pos_overlap_clip=0.5,
                            pos_overlap_event=0.8, bg_margin=6, stride_pos=1,
                            stride_bg=4, max_bg=8)
    pos = [s for s in specs if s.label == "fall"]
    assert pos, "짧은 이벤트가 통째로 버려졌다 (전도 놓침 위험)"
    # 규칙상 clip 겹침 50%(=8프레임)는 만족 못해도 event 80%(=11프레임)로 잡혀야 한다
    assert min(s.anchor for s in pos) <= 115, sorted(s.anchor for s in pos)[:5]


def t_clips_causal():
    """창은 anchor 를 마지막 프레임으로 갖는다(미래 프레임 사용 금지)."""
    row = {"stem": "v", "class_name": "theft", "label_frames": 180,
           "events": [{"start_frame": 50, "end_frame": 100}],
           "take_id": "t", "store": "S", "camera": "CA"}
    specs = clips_for_video(row, clip_len=16, frame_stride=1, pos_overlap_clip=0.5,
                            pos_overlap_event=0.8, bg_margin=6, stride_pos=1,
                            stride_bg=4, max_bg=8)
    for s in specs:
        idx = s.frame_indices(16)
        assert max(idx) == s.anchor, "anchor 가 창의 마지막이 아니다"
        assert min(idx) >= 0


def t_clips_fps_rescale():
    """라벨은 원본 fps 좌표, 캐시는 sample_fps 좌표다. 변환이 안 되면 라벨이 통째로 밀린다.

    구매행동 실제 사례: 원본 10fps / 600프레임, 이벤트 231~245.
    3fps 캐시(180프레임)에서는 약 69~74 여야 한다.
    """
    row = {"stem": "v", "class_name": "select", "label_frames": 600,
           "events": [{"start_frame": 231, "end_frame": 245}],
           "take_id": "t", "store": "S", "camera": "CA"}
    specs = clips_for_video(row, n_frames=180, clip_len=16, frame_stride=1,
                            pos_overlap_clip=0.5, pos_overlap_event=0.8,
                            bg_margin=6, stride_pos=1, stride_bg=4, max_bg=99)
    pos = [s for s in specs if s.label == "select"]
    assert pos, "변환 후 양성 클립이 없다"
    anchors = sorted(s.anchor for s in pos)
    # 이벤트(캐시 좌표 69~74)를 포함하는 창의 anchor 는 74 부근에서 시작해야 한다
    assert 70 <= anchors[0] <= 90, anchors[:5]
    assert anchors[-1] < 180, anchors[-5:]
    # 변환을 빼먹으면 anchor 가 231 이상으로 가는데, 캐시가 180프레임이라 아예 못 만든다
    bg = [s for s in specs if s.label == "background_unlabeled"]
    assert bg, "배경 클립이 없다 (변환이 잘못되면 이벤트가 범위 밖으로 나간다)"

    # 배율이 1이면(이상행동처럼 원본 fps == sample_fps) 좌표가 그대로여야 한다
    row2 = dict(row, label_frames=180, class_name="fall",
                events=[{"start_frame": 100, "end_frame": 150}])
    s2 = clips_for_video(row2, n_frames=180, clip_len=16, frame_stride=1,
                         pos_overlap_clip=0.5, pos_overlap_event=0.8,
                         bg_margin=6, stride_pos=1, stride_bg=4, max_bg=99)
    p2 = sorted(s.anchor for s in s2 if s.label == "fall")
    assert p2 and 100 <= p2[0] <= 130, p2[:5]


def t_clips_whole_video_label():
    """매장이동은 영상 전체가 해당 행동이다. 배경 클립이 나오면 안 된다."""
    row = {"stem": "v", "class_name": "moving", "label_frames": 901,
           "events": [{"start_frame": 0, "end_frame": 900}],
           "take_id": "t", "store": "S", "camera": "CA"}
    specs = clips_for_video(row, n_frames=901, clip_len=16, frame_stride=1,
                            pos_overlap_clip=0.5, pos_overlap_event=0.8,
                            bg_margin=6, stride_pos=30, stride_bg=4, max_bg=99)
    assert specs, "클립이 없다"
    labels = {s.label for s in specs}
    assert labels == {"moving"}, labels


def t_clips_ambiguous_dropped():
    """이벤트 경계 근처 애매 구간은 배경으로 넣지 않는다."""
    row = {"stem": "v", "class_name": "broken", "label_frames": 180,
           "events": [{"start_frame": 80, "end_frame": 120}],
           "take_id": "t", "store": "S", "camera": "CA"}
    specs = clips_for_video(row, clip_len=16, frame_stride=1, pos_overlap_clip=0.5,
                            pos_overlap_event=0.8, bg_margin=6, stride_pos=1,
                            stride_bg=4, max_bg=99)
    for s in specs:
        if s.label == "background_unlabeled":
            # 창 전체가 이벤트에서 6프레임 이상 떨어져야 한다
            assert s.anchor < 80 - 6 or s.start > 120 + 6, (s.start, s.anchor)


# ---------- 4. 알림 상태 기계 ----------
def t_alert_needs_persistence():
    """한 번의 높은 점수로는 알림이 나가지 않는다."""
    clock = {"t": 0.0}
    eng = AlertEngine("cam", ["fall"], {"fall": {"threshold": 0.6, "min_hits": 3, "window": 5,
                                                 "cooldown_sec": 10, "end_below_threshold": 2}},
                      clock=lambda: clock["t"])
    started, _ = eng.step({"fall": 0.95}, 1.0)
    assert not started, "1회 고득점으로 알림이 나갔다"
    clock["t"] = 1.0
    started, _ = eng.step({"fall": 0.95}, 2.0)
    assert not started
    clock["t"] = 2.0
    started, _ = eng.step({"fall": 0.95}, 3.0)
    assert len(started) == 1, "3회째에 알림이 나와야 한다"
    assert started[0].action == "fall"


def t_alert_no_duplicate_then_recover():
    """사건 진행 중 중복 알림 없음 → 종료 → 쿨다운 → 재탐지."""
    clock = {"t": 0.0}
    eng = AlertEngine("cam", ["theft"], {"theft": {"threshold": 0.5, "min_hits": 2, "window": 3,
                                                   "cooldown_sec": 5, "end_below_threshold": 2}},
                      clock=lambda: clock["t"])
    n_started = 0
    for i in range(6):                       # 계속 높은 점수
        clock["t"] = i
        s, _ = eng.step({"theft": 0.9}, float(i))
        n_started += len(s)
    assert n_started == 1, f"중복 알림 {n_started}건"

    ended = []
    for i in range(6, 9):                    # 낮은 점수 → 종료
        clock["t"] = i
        _, e = eng.step({"theft": 0.1}, float(i))
        ended += e
    assert len(ended) == 1, "사건이 종료되지 않았다"

    n2 = 0
    for i in range(9, 12):                   # 쿨다운 중(5초) 재알림 금지
        clock["t"] = i
        s, _ = eng.step({"theft": 0.9}, float(i))
        n2 += len(s)
    assert n2 == 0, "쿨다운 중에 알림이 나갔다"

    n3 = 0
    for i in range(14, 18):                  # 쿨다운 종료 후 재탐지
        clock["t"] = i
        s, _ = eng.step({"theft": 0.9}, float(i))
        n3 += len(s)
    assert n3 == 1, f"재탐지가 되지 않았다({n3})"


def t_alert_fall_faster_than_theft():
    """설정값대로 전도가 절도보다 빨리 알림이 떠야 한다."""
    cfg = Config.load()
    clock = {"t": 0.0}
    eng = AlertEngine("cam", ["fall", "theft"],
                      {a: cfg.alert_params(a) for a in ("fall", "theft")},
                      clock=lambda: clock["t"])
    first = {}
    for i in range(12):
        clock["t"] = float(i)
        s, _ = eng.step({"fall": 0.99, "theft": 0.99}, float(i))
        for ev in s:
            first.setdefault(ev.action, i)
    assert "fall" in first and "theft" in first, first
    assert first["fall"] < first["theft"], f"전도가 더 느리다: {first}"


def t_alert_reset_closes():
    """연결 끊김 시 진행 중 사건을 닫는다."""
    clock = {"t": 0.0}
    eng = AlertEngine("cam", ["fall"], {"fall": {"threshold": 0.5, "min_hits": 1, "window": 1,
                                                 "cooldown_sec": 1, "end_below_threshold": 2}},
                      clock=lambda: clock["t"])
    s, _ = eng.step({"fall": 0.9}, 0.0)
    assert s
    closed = eng.reset()
    assert len(closed) == 1 and closed[0].ended_at is not None
    assert not eng.active_events()


# ---------- 5. 비밀정보 마스킹 ----------
def t_mask():
    assert mask_url("rtsp://admin:s3cret@192.168.0.10:554/s1") == "rtsp://***:***@192.168.0.10:554/s1"
    assert mask_url("rtsp://192.168.0.10:554/s1") == "rtsp://192.168.0.10:554/s1"
    assert "s3cret" not in mask_url("rtsp://admin:s3cret@h/p")
    assert mask_url("D:\\a\\b.mp4") == "D:\\a\\b.mp4"


def main() -> int:
    for name, fn in [
        ("설정 로딩과 클래스 순서", t_config),
        ("파일명 파서 (238-2 이상행동)", t_naming),
        ("파일명 파서 (238-1 구매행동, 토큰 수 가변)", t_naming_purchase),
        ("클래스 분리: 알림 대상 vs 정상 행동", t_class_split),
        ("클립: 이벤트 내부 양성 / 배경 분리", t_clips_positive_window),
        ("클립: 짧은 전도 이벤트 보존", t_clips_short_event),
        ("클립: 인과적 창(미래 프레임 미사용)", t_clips_causal),
        ("클립: fps 차이 프레임 좌표 변환", t_clips_fps_rescale),
        ("클립: 매장이동 전체구간 라벨", t_clips_whole_video_label),
        ("클립: 애매 구간 폐기", t_clips_ambiguous_dropped),
        ("알림: 단발 고득점 무시", t_alert_needs_persistence),
        ("알림: 중복 억제 → 종료 → 쿨다운 → 재탐지", t_alert_no_duplicate_then_recover),
        ("알림: 전도가 절도보다 빠름", t_alert_fall_faster_than_theft),
        ("알림: 끊김 시 사건 종료", t_alert_reset_closes),
        ("RTSP 비밀번호 마스킹", t_mask),
    ]:
        check(name, fn)

    ok = sum(1 for _, p, _ in RESULTS if p)
    for name, passed, msg in RESULTS:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        if not passed:
            print("       " + msg.replace("\n", "\n       "))
    print(f"\n{ok}/{len(RESULTS)} 통과")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
