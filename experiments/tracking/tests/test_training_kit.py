"""상품 탐지 학습 폴더 (training/products/) — 다른 컴퓨터에서 이 폴더만으로 학습하게 만든 것."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "training" / "products"
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import check as kit_check  # noqa: E402
import prepare as kit_prepare  # noqa: E402
import product_coco as pc  # noqa: E402
import train as kit_train  # noqa: E402

NAMES = {0: "cantata", 1: "donggwaru", 2: "applecider"}


def _converted(tmp_path: Path) -> Path:
    """product_coco 가 바꿔 둔 모양(outputs/products_rf)을 작게 흉내 낸다 — 그림 3장."""
    out = tmp_path / "rf"
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()
    rows = []
    for image, video, camera, side, boxes in (("C01_angle0_rf0000.jpg", "C01", "angle0", "train", 2),
                                              ("C01_angle1_rf0000.jpg", "C01", "angle1", "train", 0),
                                              ("C05_angle0_rf0003.jpg", "C05", "angle0", "val", 1)):
        (out / "images" / image).write_bytes(b"jpg:" + image.encode())
        (out / "labels" / (Path(image).stem + ".txt")).write_text("0 0.5 0.5 0.1 0.1\n" * boxes, encoding="utf-8")
        rows.append({"image": image, "source": f"train/{video}_mp4-{image[-8:-4]}_jpg.rf.{camera}.jpg", "video": video,
                     "camera": camera, "index": int(image[-8:-4]), "split": side, "roboflow_split": "train", "boxes": boxes})
    pc.write_lists(out, rows, NAMES)
    return out


def _kit(tmp_path: Path) -> tuple[Path, Path]:
    out, data = _converted(tmp_path), tmp_path / "kit" / "dataset"
    pc.publish(out, data, NAMES)
    return out, data


def test_publish_leaves_the_images_behind(tmp_path):
    out, data = _kit(tmp_path)
    assert not (data / "images").exists()                               # 그림(얼굴)은 저장소 쪽으로 가지 않는다
    assert sorted(p.name for p in (data / "labels").glob("*.txt")) == [
        "C01_angle0_rf0000.txt", "C01_angle1_rf0000.txt", "C05_angle0_rf0003.txt"]
    yaml = (data / "data.yaml").read_text(encoding="utf-8")
    assert "path:" not in yaml and yaml.startswith("train: train.txt\n")     # 이 컴퓨터의 경로를 적지 않는다
    assert "path:" in (out / "data.yaml").read_text(encoding="utf-8")
    row = kit_check.read_table(data)[0]
    raw = (out / "images" / row["image"]).read_bytes()
    assert int(row["bytes"]) == len(raw) and row["sha1"] == hashlib.sha1(raw).hexdigest()


def test_check_says_what_is_missing(tmp_path):
    out, data = _kit(tmp_path)
    c = kit_check.check(data)
    assert not c["ok"] and len(c["missing_images"]) == 3 and c["boxes"] == 3
    assert c["by_split"] == {"train": 2, "val": 1, "test": 0}
    assert "없는 그림" in kit_check.report(c) and "아직 안 된다" in kit_check.report(c)


def test_prepare_takes_either_folder_shape(tmp_path):
    out, data = _kit(tmp_path)
    # (가) 원래 컴퓨터의 images 폴더
    s = kit_prepare.prepare(out / "images", data)
    assert s == {"placed": 3, "already": 0, "missing": []} and kit_check.check(data, deep=True)["ok"]
    assert kit_prepare.prepare(out / "images", data)["already"] == 3
    # (나) Roboflow 가 내보낸 폴더 — source 칸의 이름으로 찾는다
    for p in (data / "images").glob("*.jpg"):
        p.unlink()
    rf = tmp_path / "export"
    for r in kit_check.read_table(data)[:2]:
        (rf / Path(r["source"]).parent).mkdir(parents=True, exist_ok=True)
        (rf / r["source"]).write_bytes((out / "images" / r["image"]).read_bytes())
    s = kit_prepare.prepare(rf, data)
    assert s["placed"] == 2 and s["missing"] == ["C05_angle0_rf0003.jpg"]


def test_check_catches_a_wrong_image_and_a_leaky_split(tmp_path):
    out, data = _kit(tmp_path)
    kit_prepare.prepare(out / "images", data)
    (data / "images" / "C01_angle0_rf0000.jpg").write_bytes(b"other")                  # 크기가 다른 그림
    assert kit_check.check(data)["wrong_images"] == ["C01_angle0_rf0000.jpg"]
    kit_prepare.prepare(out / "images", data)                                          # 크기가 다르면 다시 가져온다
    same_size = bytearray((data / "images" / "C01_angle0_rf0000.jpg").read_bytes())
    same_size[0] ^= 1
    (data / "images" / "C01_angle0_rf0000.jpg").write_bytes(bytes(same_size))          # 크기는 같고 내용이 다른 그림
    assert kit_check.check(data)["ok"] and kit_check.check(data, deep=True)["wrong_images"] == ["C01_angle0_rf0000.jpg"]
    # 같은 영상이 두 쪽에 있으면 막는다
    table = (data / "frames.csv").read_text(encoding="utf-8").replace("C01,angle1,0,train", "C01,angle1,0,val")
    (data / "frames.csv").write_text(table, encoding="utf-8")
    c = kit_check.check(data)
    assert c["video_overlap"] == ["C01"] and "train.txt" in c["list_mismatch"] and not c["ok"]


def test_committed_dataset_is_consistent():
    """저장소에 올린 dataset/ — 그림은 없으므로 라벨 · 목록 · 표만 본다."""
    c = kit_check.check()
    assert c["images"] == 1194 and c["boxes"] == 5981 and c["by_split"] == {"train": 768, "val": 133, "test": 293}
    assert not (c["missing_labels"] or c["label_mismatch"] or c["list_mismatch"] or c["video_overlap"])
    yaml = (KIT / "dataset" / "data.yaml").read_text(encoding="utf-8")
    assert "path:" not in yaml and "0: cantata" in yaml and "2: applecider" in yaml
    tracked = (KIT / ".gitignore").read_text(encoding="utf-8")
    assert "dataset/images/*" in tracked and "*.pt" in tracked and "handoff/" in tracked


def test_scores_and_epochs_are_read_from_ultralytics_output(tmp_path):
    box = SimpleNamespace(mp=0.81234, mr=0.7, map50=0.75, map=0.5, ap_class_index=[0, 2], p=[0.9, 0.7], r=[0.8, 0.6],
                          ap50=[0.85, 0.65], ap=[0.6, 0.4])
    s = kit_train.scores_of(SimpleNamespace(box=box, names=NAMES))
    assert s["precision"] == 0.8123 and s["map50"] == 0.75 and list(s["classes"]) == ["cantata", "applecider"]
    assert s["classes"]["applecider"] == {"precision": 0.7, "recall": 0.6, "map50": 0.65, "map50_95": 0.4}
    csv_path = tmp_path / "results.csv"
    csv_path.write_text("   epoch,  metrics/mAP50-95(B)\n 1, 0.10\n 2, 0.35\n 3, 0.30\n", encoding="utf-8")
    assert kit_train.epochs_run(csv_path) == {"run": 3, "best": 2}
    assert kit_train.epochs_run(tmp_path / "none.csv") == {"run": 0, "best": None}
    assert kit_train.pick_device("cpu") == "cpu" and kit_train.pick_device("auto") in ("cpu", "0")


def test_scores_read_a_real_ultralytics_result():
    """흉내 낸 객체가 아니라 설치된 ultralytics 의 결과 객체로 — 버전이 바뀌어 이름이 달라지면 여기서 걸린다."""
    import numpy as np
    import pytest

    metrics_module = pytest.importorskip("ultralytics.utils.metrics")
    m = metrics_module.DetMetrics(names=NAMES)
    rng = np.random.default_rng(0)
    # 예측 30개(종류마다 10개) · 정답 90개(종류마다 30개) — 맞은 예측이 정답보다 많을 수 없게 맞춘 지어낸 값
    m.process(rng.random((30, 10)) > 0.4, rng.random(30), np.repeat([0, 1, 2], 10), np.repeat([0, 1, 2], 30))
    s = kit_train.scores_of(m)
    assert list(s["classes"]) == ["cantata", "donggwaru", "applecider"]
    assert 0 < s["map50"] <= 1 and 0 < s["classes"]["donggwaru"]["recall"] <= 1
