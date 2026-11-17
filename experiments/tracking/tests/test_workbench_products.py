"""분석 도구의 상품 라벨 찍기 — 라벨 읽고 쓰기(workbench/products.py)와 YOLO 로 내보내기(scripts/product_frames.py export)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import product_frames as pf  # noqa: E402
from workbench.products import Store  # noqa: E402

CLASSES = "names:\n  0: cantata\n  1: donggwaru\n  2: applecider\nkorean:\n  cantata: 칸타타\nlane:\n  L1: cantata\n"


def frame(pair: str, camera: str, n: int, split: str = "train", batch: int = 1, reason: str = "base") -> dict:
    return {"image": pf.image_name(pair, camera, n), "pair": pair, "camera": camera, "frame": n, "time_sec": f"{n / 30:.2f}",
            "reason": reason, "lane": "", "action": "", "people": "", "split": split, "batch": batch}


@pytest.fixture
def store(tmp_path):
    dataset, images = tmp_path / "labels", tmp_path / "out" / "images"
    dataset.mkdir()
    images.mkdir(parents=True)
    (dataset / "classes.yaml").write_text(CLASSES, encoding="utf-8")
    rows = [frame("N02", "angle1", 10), frame("N01", "angle0", 90), frame("N01", "angle0", 30), frame("N01", "angle0", 60),
            frame("N02", "angle0", 10, split="val"), frame("N03", "angle0", 10, split="test"), frame("N01", "angle0", 45, batch=0)]
    pf.write_plan(rows, dataset / "frames.csv")
    for r in rows:
        (images / r["image"]).write_bytes(b"jpg")
    return Store(dataset, images)


def box(cls=0, x1=100, y1=100, x2=130, y2=170, **more) -> dict:
    return {"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, **more}


def test_frames_are_batch_one_in_labeling_order(store):
    # 카메라 → 영상 → 장면 순서. 묶음 0 (후보일 뿐인 장면)은 안 나온다
    assert [f["image"] for f in store.frames()] == [
        "N01_angle0_00030.jpg", "N01_angle0_00060.jpg", "N01_angle0_00090.jpg", "N02_angle0_00010.jpg", "N03_angle0_00010.jpg",
        "N02_angle1_00010.jpg"]


def test_save_and_load_round_trip(store):
    image = "N01_angle0_00030.jpg"
    assert store.load(image) is None
    store.save(image, [box(1, 300, 200, 330, 260, held=True), box(0)], 1920, 1080)
    got = store.load(image)
    assert got["done"] is False and got["skip"] is False
    assert got["boxes"] == [{"cls": 0, "x1": 100.0, "y1": 100.0, "x2": 130.0, "y2": 170.0, "held": False},
                            {"cls": 1, "x1": 300.0, "y1": 200.0, "x2": 330.0, "y2": 260.0, "held": True}]   # 왼쪽부터
    assert store.listing()["frames"][0] | {"state": "doing", "n": 2, "held": 1} == store.listing()["frames"][0]
    store.save(image, [], 1920, 1080, done=True)                          # 상품이 안 보인다고 찍은 것
    assert store.load(image)["boxes"] == [] and store.listing()["frames"][0]["state"] == "done"
    assert store.clear(image) and store.load(image) is None and not store.clear(image)


def test_save_cuts_boxes_to_the_image_and_drops_slips(store):
    saved = store.save("N01_angle0_00030.jpg", [box(0, 1900, 1050, 2000, 1200), box(0, 50, 50, 51, 120), box(2, 40, 90, 10, 20)],
                       1920, 1080)
    assert saved["boxes"] == [{"cls": 2, "x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 90.0, "held": False},   # 거꾸로 끈 것은 바로잡는다
                              {"cls": 0, "x1": 1900.0, "y1": 1050.0, "x2": 1920.0, "y2": 1080.0, "held": False}]


def test_save_refuses_what_it_does_not_know(store):
    with pytest.raises(ValueError):
        store.save("../../evil.jpg", [], 1920, 1080)                      # 표에 없는 이름 — 다른 곳에 못 쓴다
    with pytest.raises(ValueError):
        store.save("N01_angle0_00045.jpg", [], 1920, 1080)                # 묶음 1 이 아닌 장면
    with pytest.raises(ValueError):
        store.save("N01_angle0_00030.jpg", [box(7)], 1920, 1080)          # 없는 종류
    with pytest.raises(ValueError):
        store.save("N01_angle0_00030.jpg", [{"cls": 0}], 1920, 1080)
    with pytest.raises(ValueError):
        store.save("N01_angle0_00030.jpg", [box()], 0, 0)
    assert not list(store.boxes_dir.glob("*")) if store.boxes_dir.exists() else True


def test_draft_comes_from_the_nearest_finished_frame_of_the_same_camera(store):
    assert store.draft("N01_angle0_00060.jpg") is None                    # 아직 찍은 것이 없다
    store.save("N02_angle1_00010.jpg", [box(0)], 1920, 1080, done=True)   # 다른 카메라 — 자리가 달라 못 쓴다
    assert store.draft("N01_angle0_00060.jpg") is None
    store.save("N02_angle0_00010.jpg", [box(2, 500, 500, 530, 560)], 1920, 1080, done=True)
    assert store.draft("N01_angle0_00060.jpg")["from"] == "N02_angle0_00010.jpg"      # 같은 영상에 없으면 다른 영상
    store.save("N01_angle0_00090.jpg", [box(0), box(1, 300, 200, 330, 260, held=True)], 1920, 1080, done=True)
    store.save("N01_angle0_00030.jpg", [box(1)], 1920, 1080)              # 찍는 중인 것은 초안으로 안 쓴다
    got = store.draft("N01_angle0_00060.jpg")
    assert got["from"] == "N01_angle0_00090.jpg"
    assert [b["cls"] for b in got["boxes"]] == [0]                        # 손에 든 것은 다음 장면에서 자리가 다르다 — 뺀다
    assert store.frame("N01_angle0_00060.jpg")["draft"] == got
    assert store.frame("N01_angle0_00090.jpg")["draft"] is None           # 찍어 둔 것이 있으면 초안은 안 준다


def test_yolo_lines_are_center_and_size_in_fractions():
    assert pf.yolo_lines([box(2, 960, 540, 1920, 1080)], 1920, 1080) == ["2 0.750000 0.750000 0.500000 0.500000"]
    assert pf.yolo_lines([], 1920, 1080) == []


def test_export_writes_only_finished_frames(store, tmp_path):
    out = tmp_path / "out"
    store.save("N01_angle0_00030.jpg", [box(0), box(1, 300, 200, 330, 260, held=True)], 1920, 1080, done=True)
    store.save("N01_angle0_00060.jpg", [box(0)], 1920, 1080)                          # 찍는 중
    store.save("N01_angle0_00090.jpg", [box(0)], 1920, 1080, done=True, skip=True)    # 뺀 것
    store.save("N02_angle0_00010.jpg", [], 1920, 1080, done=True)                     # 상품 없는 장면
    store.save("N03_angle0_00010.jpg", [box(2)], 1920, 1080, done=True)
    (out / "images" / "N03_angle0_00010.jpg").unlink()                                # 그림이 없다
    (out / "labels").mkdir()
    (out / "labels" / "old.txt").write_text("0 0 0 0 0\n")
    st = pf.export(pf.read_plan(store.dataset / "frames.csv"), store.dataset, out)
    assert st["frames"] == 2 and st["boxes"] == 2 and st["held"] == 1 and st["empty"] == 1
    assert (st["doing"], st["skip"], st["no_image"]) == (1, 1, 1)
    assert st["by_class"] == {"cantata": 1, "donggwaru": 1, "applecider": 0}
    assert sorted(p.name for p in (out / "labels").glob("*.txt")) == ["N01_angle0_00030.txt", "N02_angle0_00010.txt"]
    assert (out / "labels" / "N02_angle0_00010.txt").read_text() == ""
    assert (out / "labels" / "N01_angle0_00030.txt").read_text().splitlines()[0].startswith("0 0.059896 0.125000 ")
    assert (out / "train.txt").read_text() == "./images/N01_angle0_00030.jpg\n"
    assert (out / "val.txt").read_text() == "./images/N02_angle0_00010.jpg\n" and (out / "test.txt").read_text() == ""
    assert json.dumps(st["by_split"]) == '{"train": 1, "val": 1, "test": 0}'
    assert "0: cantata" in (out / "data.yaml").read_text()


def test_label_file_has_one_box_per_line(store):
    store.save("N01_angle0_00030.jpg", [box(0), box(1, 300, 200, 330, 260)], 1920, 1080, done=True)
    text = store.path("N01_angle0_00030.jpg").read_text(encoding="utf-8")
    assert len(text.splitlines()) == 4 and json.loads(text)["boxes"][1]["cls"] == 1
    store.save("N01_angle0_00030.jpg", [], 1920, 1080, done=True)
    assert json.loads(store.path("N01_angle0_00030.jpg").read_text(encoding="utf-8"))["boxes"] == []
