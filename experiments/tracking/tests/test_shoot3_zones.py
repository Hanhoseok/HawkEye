import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import shoot3_zones as Z  # noqa: E402


def test_zone_file_has_sensor_lanes():
    zones = Z.load_zones()
    assert [z["name"] for z in zones["zones"]] == ["L1", "L2", "L3"]
    assert zones["reference"] == "area_setting"
    polys = Z.ref_polygons(zones, (1920, 1080))
    assert set(polys) == {"shelf", "L1", "L2", "L3"}
    # 레인은 선반 상자 안, 오른쪽부터 L1·L2·L3
    sx0, sy0 = polys["shelf"].min(axis=0)
    sx1, sy1 = polys["shelf"].max(axis=0)
    for k in ("L1", "L2", "L3"):
        assert (polys[k].min(axis=0) >= [sx0, sy0]).all() and (polys[k].max(axis=0) <= [sx1, sy1]).all()
    assert polys["L1"][:, 0].mean() > polys["L2"][:, 0].mean() > polys["L3"][:, 0].mean()


def test_to_clip_inverts_clip_to_ref_homography():
    # 영상 → area_setting 이 +80px 이동이면, area_setting 의 점은 영상에서 -80px
    h = np.array([[1, 0, 80], [0, 1, 5], [0, 0, 1]], float)
    poly = np.array([[1000, 400], [1100, 400]], float)
    assert np.allclose(Z.to_clip(poly, h), [[920, 395], [1020, 395]])
