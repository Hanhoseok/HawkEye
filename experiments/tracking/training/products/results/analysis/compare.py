import json, sys
from pathlib import Path

h = Path(sys.argv[1])
for name in ("yolov8n_1280", "yolov8s_1280"):
    r = json.loads((h / name / "result.json").read_text(encoding="utf-8"))
    print(name, "epochs", r["epochs"], "sec", r["train_seconds"], "batch", r["batch"], "size MB", round((h / name / "best.pt").stat().st_size / 1e6, 1))
    for key in ("val_all", "val_angle0", "val_angle1", "test_all"):
        s = r["scores"][key]
        print(f"  {key:11s} map50 {s['map50']:.3f} map50-95 {s['map50_95']:.3f} P {s['precision']:.3f} R {s['recall']:.3f}")
        if key == "val_all":
            for c, v in s["classes"].items():
                print(f"     {c:11s} map50 {v['map50']:.3f} map50-95 {v['map50_95']:.3f} P {v['precision']:.3f} R {v['recall']:.3f}")
