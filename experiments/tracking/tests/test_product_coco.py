"""Roboflow(COCO) 상품 데이터셋을 우리 YOLO 모양으로 바꾸기 (scripts/product_coco.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import product_coco as pc  # noqa: E402

NAMES = {0: "cantata", 1: "donggwaru", 2: "applecider"}
CATS = [{"id": 0, "name": "labeling"}, {"id": 1, "name": "apple soda"}, {"id": 2, "name": "cantata"}, {"id": 3, "name": "donggwaru"}]


def test_file_name_gives_video_and_order():
    assert pc.parse_name("C01_mp4-0005_jpg.rf.4a5Z8wTNqu12oz0W2njS.jpg") == ("C01", 5)
    assert pc.parse_name("N08-1_mp4-0012_jpg.rf.abc.jpg") == ("N08-1", 12)
    assert pc.parse_name("area_setting_mp4-0003_jpg.rf.x.jpg") == ("area_setting", 3)
    assert pc.parse_name("photo.jpg") is None
    assert pc.out_name("C01", "angle1", 5) == "C01_angle1_rf0005.jpg"


def test_classes_are_matched_by_name_not_number():
    cmap = pc.class_map(CATS, NAMES)
    assert cmap == {1: 2, 2: 0, 3: 1}                     # 'apple soda' = applecider, 묶음 이름 'labeling' 은 빠진다
    assert pc.class_map([{"id": 7, "name": "Dong-Gwaru"}], NAMES) == {7: 1}


def test_boxes_are_clamped_and_bad_ones_counted():
    ann = [{"category_id": 2, "bbox": [100, 50, 30, 60]},
           {"category_id": 3, "bbox": [1900, 1060, 40, 40]},      # 그림 밖으로 나감 — 잘린다
           {"category_id": 1, "bbox": [10, 10, 1, 30]},           # 너무 작음
           {"category_id": 0, "bbox": [10, 10, 30, 30]}]          # 못 맞춘 종류
    boxes, dropped = pc.to_boxes(ann, pc.class_map(CATS, NAMES), 1920, 1080)
    assert boxes == [{"cls": 0, "x1": 100.0, "y1": 50.0, "x2": 130.0, "y2": 110.0},
                     {"cls": 1, "x1": 1900.0, "y1": 1060.0, "x2": 1920.0, "y2": 1080.0}]
    assert dropped == {"unknown_class": 1, "tiny": 1}


def test_side_follows_the_video_not_the_frame():
    plan = {"C01": "train", "C03": "test"}
    assert pc.side_of("C03", plan, "train") == "test"
    assert pc.side_of("area_setting", plan, "train") == "train"
    assert pc.side_of("area_setting", plan, "skip") is None


def test_camera_is_the_closer_reference():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(36, 64)).astype(np.float32), rng.normal(size=(36, 64)).astype(np.float32)
    refs = {"angle0": [a], "angle1": [b]}
    camera, margin = pc.pick_camera(a + 0.1 * rng.normal(size=a.shape).astype(np.float32), refs)
    assert camera == "angle0" and margin > 0.5
    camera, margin = pc.pick_camera((a + b) / 2, refs)
    assert margin < pc.MIN_MARGIN                              # 둘과 똑같이 닮음 — 모름


def _dataset(tmp_path: Path) -> Path:
    """그림 6장: C01 은 Roboflow 가 세 쪽에 흩어 놓았고, C03 · area_setting 이 한 장씩, 표에 없는 그림 한 장."""
    src = tmp_path / "rf"
    files = {"train": ["C01_mp4-0000_jpg.rf.a.jpg", "C01_mp4-0000_jpg.rf.b.jpg", "C03_mp4-0001_jpg.rf.c.jpg",
                       "area_setting_mp4-0002_jpg.rf.d.jpg"],
             "valid": ["C01_mp4-0001_jpg.rf.e.jpg"], "test": ["C01_mp4-0002_jpg.rf.f.jpg"]}
    for folder, names in files.items():
        (src / folder).mkdir(parents=True)
        images, anns = [], []
        for i, name in enumerate(names):
            (src / folder / name).write_bytes(b"jpg-" + name.encode())
            images.append({"id": i, "file_name": name, "width": 200, "height": 100})
            if "rf.d" not in name:                              # area_setting 그림은 상자가 없다
                anns.append({"id": i, "image_id": i, "category_id": 2, "bbox": [20, 10, 40, 20]})
        (src / folder / "_annotations.coco.json").write_text(
            json.dumps({"categories": CATS, "images": images, "annotations": anns}), encoding="utf-8")
    (src / "train" / "T02_mp4-0014_jpg.rf.z.jpg").write_bytes(b"jpg")          # 표에 없는 그림
    return src


def test_convert_resplits_by_video_and_names_the_camera(tmp_path):
    src, out = _dataset(tmp_path), tmp_path / "out"
    # 그림 내용 대신 파일 이름으로 카메라를 정한다: rf.b 만 전체 카메라
    stats = pc.convert(src, out, {"C01": "val", "C03": "test"}, NAMES,
                       lambda p: ("angle1" if ".rf.b." in p.name else "angle0", 0.9))
    assert stats["images"] == 6 and stats["boxes"] == 5 and stats["empty"] == 1
    assert stats["not_listed"] == 1 and stats["unknown_videos"] == {"area_setting": 1}
    assert stats["by_split"] == {"val": 4, "test": 1, "train": 1}
    assert stats["moved_side"] == 4                            # C01 의 train 2장 · test 1장, C03 의 train 1장
    assert stats["by_class"] == {"cantata": 5, "donggwaru": 0, "applecider": 0}

    def lines(name):
        return (out / name).read_text(encoding="utf-8").split()

    assert lines("val.txt") == ["./images/C01_angle0_rf0000.jpg", "./images/C01_angle1_rf0000.jpg",
                                "./images/C01_angle0_rf0001.jpg", "./images/C01_angle0_rf0002.jpg"]
    assert lines("val_angle1.txt") == ["./images/C01_angle1_rf0000.jpg"]
    assert lines("train.txt") == ["./images/area_setting_angle0_rf0002.jpg"] and lines("test_angle1.txt") == []
    assert (out / "labels" / "C01_angle1_rf0000.txt").read_text(encoding="utf-8") == "0 0.200000 0.200000 0.200000 0.200000\n"
    assert (out / "labels" / "area_setting_angle0_rf0002.txt").read_text(encoding="utf-8") == ""      # 배경 그림
    assert (out / "images" / "C01_angle1_rf0000.jpg").read_bytes() == b"jpg-C01_mp4-0000_jpg.rf.b.jpg"
    assert "val: val_angle1.txt" in (out / "data_angle1.yaml").read_text(encoding="utf-8")
    table = (out / "frames.csv").read_text(encoding="utf-8").splitlines()
    assert table[0] == ",".join(pc.COLUMNS) and "C01_angle0_rf0002.jpg,test/C01_mp4-0002_jpg.rf.f.jpg,C01,angle0,2,val,test,1" in table


def test_convert_leaves_out_what_it_cannot_place(tmp_path):
    src, out = _dataset(tmp_path), tmp_path / "out"
    # 카메라를 못 가른 그림 · 같은 이름이 되는 그림 · 표에 없는 영상(skip)은 넣지 않고 센다
    stats = pc.convert(src, out, {"C01": "val", "C03": "test"}, NAMES,
                       lambda p: ("angle0", 0.05 if ".rf.c." in p.name else 0.9), unknown="skip")
    assert stats["unsure_camera"] == 1 and stats["clash"] == 1 and stats["skipped_video"] == 1
    assert stats["images"] == 3 and sorted(p.name for p in (out / "images").glob("*.jpg")) == [
        "C01_angle0_rf0000.jpg", "C01_angle0_rf0001.jpg", "C01_angle0_rf0002.jpg"]
    # 다시 돌리면 지난번 것은 지워진다
    pc.convert(src, out, {}, NAMES, lambda p: ("angle0", 0.9), unknown="skip")
    assert list((out / "images").glob("*.jpg")) == [] and list((out / "labels").glob("*.txt")) == []
