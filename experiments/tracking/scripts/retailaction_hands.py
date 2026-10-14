"""RetailAction — '손에 물건이 있나'로 집음 / 놓음 / 만짐을 가를 수 있나 (손 부분 사진 방식 가능성 시험).

집음과 놓음은 팔 동작이 같다. 차이는 손에 물건이 언제 있었나뿐이다.
    집음: 뻗기 전 빈손 -> 거둔 뒤 물건
    놓음: 뻗기 전 물건 -> 거둔 뒤 빈손
그래서 행동 직전·직후 장면에서 **손 주변을 잘라** 비교한다. 손 위치는 데이터에 든 관절 좌표로 찾는다.

이 시험은 '행동이 언제 일어났는지 안다'고 두고(정답 시점 사용), 그 행동의 **종류**만 가른다.
우리 파이프라인에서는 그 시점을 손 뻗기 감지가 준다.

비교하는 근거(특징):
    pose          관절만 — 손-목 거리, 손-접촉점 거리 (전/중/후)        ← 스켈레톤만으로 되나
    pixel_hand    손 주변 작은 흑백 사진(16x16) 전/후                   ← 모델 없이 되나
    dino_hand     손 주변 사진의 DINOv2-small 특징 전/후/차이            ← 이번 방법
    dino_contact  접촉점(선반 쪽) 주변 사진의 DINOv2 특징 전/후/차이      ← 선반 변화 방식의 RetailAction 판
    dino_both     dino_hand + dino_contact

접촉점(손이 닿은 지점) 정답은 '어느 손인지 고르기'와 '선반 쪽 자리' 에만 쓴다. 집음/놓음 구분 정보는 없다.

채점: 행동이 하나인 샘플. 샘플 단위로 5겹 교차검증(같은 샘플의 두 카메라는 같은 겹), 씨앗 5개 평균.
놓음은 49개뿐이라 **가능성 확인**이다 — 오차가 크다.

    python scripts/retailaction_hands.py --start 0 --limit 700 --extract-only     # 나눠서 특징 추출 (CPU 로 한 번에 10분 넘음)
    python scripts/retailaction_hands.py --start 700 --limit 700 --extract-only
    ...
    python scripts/retailaction_hands.py --eval-only                               # 모두 합쳐 채점

라이선스: Standard.AI Dataset License (출처 Standard Cognition). 재식별 금지 — 얼굴·사람 특징은 쓰지 않는다(손·선반 부분만).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

DATA = Path("data/retailaction/test")
OUT = Path("outputs/retailaction_hands")
LABELS = ("take", "put", "touch")
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ts(text: str) -> float:
    """ISO 시각 -> 초. 1970 년 근처라 Windows 의 timestamp() 가 실패하므로 UTC 기준으로 직접 뺀다."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt - EPOCH).total_seconds()


def pt(pose: dict | None, name: str):
    """(행, 열) 비율 -> (x, y) 비율."""
    if not pose or name not in pose or pose[name] is None:
        return None
    r, c = pose[name]
    return np.array([c, r], dtype=float)


def hand_point(pose, side: str):
    return pt(pose, f"{side}_hand") if pt(pose, f"{side}_hand") is not None else pt(pose, f"{side}_wrist")


def torso(pose):
    neck, waist = pt(pose, "neck"), pt(pose, "middle_of_waist")
    return None if neck is None or waist is None else float(np.linalg.norm(neck - waist))


def instances(sample: Path):
    """샘플 하나에서 행동마다, 카메라마다 (전 장면, 후 장면, 손 위치, 접촉점, 자르기 크기, 관절 특징)을 만든다.

    처음엔 행동이 하나인 샘플만 썼다 — 놓음 49건 중 30건이 다른 행동과 같은 샘플에 있어 17건만 남았다.
    """
    meta = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))["content"]
    seg = meta["segment_info"]
    t0, t1 = ts(seg["sampled_at_start"]), ts(seg["sampled_at_end"])
    out = []
    for k, act in enumerate(meta["labels"]["action"]):
        if act["label"] in LABELS:
            out += _instances_for(sample, meta, act, k, t0, t1)
    return out


def _instances_for(sample: Path, meta: dict, act: dict, k: int, t0: float, t1: float):
    out = []
    for cam in ("rank0", "rank1"):
        info = meta["action_cam"].get(cam) or {}
        poses, stamps = info.get("poses") or [], info.get("frame_timestamps") or []
        spatial = (act.get("spatial") or {}).get("action_cam", {}).get(cam)
        if not poses or len(poses) != len(stamps) or not spatial or t1 <= t0:
            continue
        frac = [(ts(s) - t0) / (t1 - t0) for s in stamps]
        P = [(p or {}).get("pose") for p in poses]
        contact = np.array([spatial["x"], spatial["y"]], dtype=float)
        mids = [i for i, f in enumerate(frac) if act["start"] <= f <= act["end"]] or \
               [int(np.argmin([abs(f - (act["start"] + act["end"]) / 2) for f in frac]))]
        # 어느 손인가: 행동 중 접촉점에 가장 가까이 간 손
        best = None
        for side in ("left", "right"):
            d = [np.linalg.norm(hand_point(P[i], side) - contact) for i in mids if hand_point(P[i], side) is not None]
            if d and (best is None or min(d) < best[1]):
                best = (side, min(d))
        if best is None:
            continue
        side = best[0]
        before = [i for i, f in enumerate(frac) if f < act["start"] and hand_point(P[i], side) is not None]
        after = [i for i, f in enumerate(frac) if f > act["end"] and hand_point(P[i], side) is not None]
        if not before or not after:
            continue
        b, a = before[-1], after[0]
        torsos = [torso(p) for p in P if torso(p)]
        if not torsos:
            continue
        scale = float(np.median(torsos))
        mid = mids[len(mids) // 2]
        neck = lambda i: pt(P[i], "neck")
        feats = []
        for i in (b, mid, a):
            h = hand_point(P[i], side)
            n = neck(i)
            feats += [np.linalg.norm(h - n) / scale if h is not None and n is not None else np.nan,
                      np.linalg.norm(h - contact) / scale if h is not None else np.nan]
        feats += [act["end"] - act["start"]]
        out.append({
            "sample": sample.name, "action": f"{sample.name}_{k}", "cam": cam, "label": act["label"],
            "video": str(sample / f"{cam}_video.mp4"), "before": b, "after": a,
            "hand_before": hand_point(P[b], side), "hand_after": hand_point(P[a], side),
            "contact": contact, "crop": scale * 0.9, "pose": np.array(feats, dtype=float),
        })
    return out


def crop(image: np.ndarray, center_xy, size_frac: float) -> np.ndarray:
    h, w = image.shape[:2]
    side = int(np.clip(size_frac * w, 40, 200))
    cx, cy = int(center_xy[0] * w), int(center_xy[1] * h)
    x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
    x2, y2 = min(w, x1 + side), min(h, y1 + side)
    patch = image[y1:y2, x1:x2]
    return cv2.copyMakeBorder(patch, 0, side - patch.shape[0], 0, side - patch.shape[1], cv2.BORDER_CONSTANT)


MAX_TAKE_SAMPLES = 600


def select_samples(seed: int = 0) -> list[Path]:
    """놓음·만짐이 든 샘플은 모두, 집음만 든 샘플은 무작위 600개 (집음 vs 놓음 비교에는 충분, 추출 시간 1/4)."""
    rare, takes = [], []
    for p in sorted(q for q in DATA.iterdir() if q.is_dir()):
        labels = {a["label"] for a in json.loads((p / "metadata.json").read_text(encoding="utf-8"))["content"]["labels"]["action"]}
        (rare if labels & {"put", "touch"} else takes if labels else []).append(p)
    rng = np.random.default_rng(seed)
    pick = [takes[i] for i in sorted(rng.choice(len(takes), size=min(MAX_TAKE_SAMPLES, len(takes)), replace=False))]
    return sorted(rare + pick)


def extract(start: int, limit: int | None) -> dict:
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    def local(frames: list[np.ndarray], points: list[list]) -> np.ndarray:
        """장면 전체를 336x336(24x24 패치, 한 칸 = 원본 약 25px)으로 넣고, 각 점 주변 3x3 패치 토큰의 평균.
        손이 30px 정도라 잘라서 키우면 주변(선반·몸)이 특징을 덮는다 — 그래서 손 자리의 토큰만 쓴다."""
        x = np.stack([(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (336, 336)).astype(np.float32) / 255.0 - MEAN) / STD
                      for f in frames]).transpose(0, 3, 1, 2)
        with torch.no_grad():
            tok = model(pixel_values=torch.from_numpy(x)).last_hidden_state[:, 1:].reshape(len(frames), 24, 24, -1).numpy()
        out = []
        for t, pts in zip(tok, points):
            v = []
            for q in pts:
                cx, cy = int(np.clip(q[0] * 24, 0, 23)), int(np.clip(q[1] * 24, 0, 23))
                v.append(t[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2].reshape(-1, t.shape[-1]).mean(0))
            out.append(np.concatenate(v))
        return np.stack(out)

    def embed(patches: list[np.ndarray]) -> np.ndarray:
        x = np.stack([(cv2.resize(cv2.cvtColor(p, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255.0 - MEAN) / STD
                      for p in patches]).transpose(0, 3, 1, 2)
        with torch.no_grad():
            hs = model(pixel_values=torch.from_numpy(x)).last_hidden_state
        return torch.cat([hs[:, 0], hs[:, 1:].mean(1)], dim=1).numpy()   # CLS + 패치 평균 = 768

    samples = select_samples()
    samples = samples[start:start + limit] if limit else samples[start:]
    rows, crops, locs, examples = [], [], [], []
    for k, sample in enumerate(samples, 1):
        for inst in instances(sample):
            cap = cv2.VideoCapture(inst["video"])
            frames = {}
            for idx in (inst["before"], inst["after"]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, im = cap.read()
                if ok:
                    frames[idx] = im
            cap.release()
            if len(frames) < 2:
                continue
            fb, fa = frames[inst["before"]], frames[inst["after"]]
            c = [crop(fb, inst["hand_before"], inst["crop"]), crop(fa, inst["hand_after"], inst["crop"]),
                 crop(fb, inst["contact"], inst["crop"]), crop(fa, inst["contact"], inst["crop"])]
            rows.append(inst)
            crops.append(c)
            locs.append(local([fb, fa], [[inst["hand_before"], inst["contact"]], [inst["hand_after"], inst["contact"]]]))
            if inst["label"] != "take" or sum(e[0] == "take" for e in examples) < 4:
                examples.append((inst["label"], c))
        if k % 200 == 0:
            print(f"  ... {k}/{len(samples)} 샘플, 사례 {len(rows)}", flush=True)

    print(f"사례 {len(rows)}개 - DINOv2 특징 추출 중", flush=True)
    flat = [p for c in crops for p in c]
    feats = np.concatenate([embed(flat[i:i + 64]) for i in range(0, len(flat), 64)]).reshape(len(rows), 4, -1)
    pixel = np.array([[cv2.resize(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), (16, 16)).astype(np.float32).ravel() / 255.0
                       for p in c[:2]] for c in crops])
    data = {
        "sample": np.array([r["sample"] for r in rows]), "action": np.array([r["action"] for r in rows]),
        "cam": np.array([r["cam"] for r in rows]), "local": np.stack(locs),
        "label": np.array([r["label"] for r in rows]), "pose": np.stack([r["pose"] for r in rows]),
        "dino": feats, "pixel": pixel,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"features_{start:05d}.npz", **data)
    save_examples(examples, start)
    return data


def save_examples(examples, start: int = 0):
    """확인용: 종류별 몇 개의 손(전/후)·접촉점(전/후) 사진."""
    rows = []
    for label, c in examples[:12]:
        tiles = [cv2.resize(p, (120, 120)) for p in c]
        strip = cv2.hconcat(tiles)
        cv2.putText(strip, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        rows.append(strip)
    if rows:
        cv2.imwrite(str(OUT / f"examples_{start:05d}.jpg"), cv2.vconcat(rows))


def feature_sets(d: dict) -> dict:
    hb, ha, cb, ca = (d["dino"][:, i] for i in range(4))
    half = d["local"].shape[-1] // 2
    lhb, lcb = d["local"][:, 0, :half], d["local"][:, 0, half:]
    lha, lca = d["local"][:, 1, :half], d["local"][:, 1, half:]
    pose = np.nan_to_num(d["pose"], nan=np.nanmedian(d["pose"]))
    return {
        "pose": pose,
        "pixel_hand": np.concatenate([d["pixel"][:, 0], d["pixel"][:, 1], d["pixel"][:, 1] - d["pixel"][:, 0]], 1),
        "dino_hand": np.concatenate([hb, ha, ha - hb], 1),
        "dino_contact": np.concatenate([cb, ca, ca - cb], 1),
        "dino_both": np.concatenate([hb, ha, ha - hb, cb, ca, ca - cb], 1),
        "local_hand": np.concatenate([lhb, lha, lha - lhb], 1),
        "local_contact": np.concatenate([lcb, lca, lca - lcb], 1),
        "local_both": np.concatenate([lhb, lha, lha - lhb, lcb, lca, lca - lcb], 1),
    }


def evaluate(d: dict, seeds: int = 5) -> list[dict]:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.array([LABELS.index(l) for l in d["label"]])
    groups = d["sample"]                 # 같은 샘플(행동·카메라)은 같은 겹에
    units = d["action"]                  # 채점은 행동 단위 (두 카메라 확률 평균)
    samples = np.unique(units)
    sample_label = {s: y[units == s][0] for s in samples}
    results = []
    for name, X in feature_sets(d).items():
        aucs, recalls, confs = [], [], []
        for seed in range(seeds):
            prob = np.zeros((len(y), len(LABELS)))
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, te in cv.split(X, y, groups):
                steps = [StandardScaler()]
                if X.shape[1] > 64:
                    steps.append(PCA(n_components=64, random_state=seed))
                steps.append(LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000))
                clf = make_pipeline(*steps).fit(X[tr], y[tr])
                p = np.zeros((len(te), len(LABELS)))
                p[:, clf.classes_] = clf.predict_proba(X[te])
                prob[te] = p
            # 샘플 단위: 두 카메라 확률 평균
            sp = np.stack([prob[units == s].mean(0) for s in samples])
            sy = np.array([sample_label[s] for s in samples])
            tp = (sy == 0) | (sy == 1)
            score = sp[tp, 1] / np.clip(sp[tp, 0] + sp[tp, 1], 1e-9, None)   # 놓음 쪽 확률 (집음 vs 놓음만)
            truth = sy[tp] == 1
            aucs.append(roc_auc_score(truth, score))
            thr = np.quantile(score[~truth], 0.95)          # 집음을 놓음으로 잘못 보는 비율 5% 에서
            recalls.append(float(np.mean(score[truth] > thr)))
            confs.append(np.array([[np.sum((sy == i) & (sp.argmax(1) == j)) for j in range(3)] for i in range(3)]))
        results.append({
            "features": name, "dims": X.shape[1],
            "auc_take_vs_put": (float(np.mean(aucs)), float(np.std(aucs))),
            "put_recall_at_5pct": (float(np.mean(recalls)), float(np.std(recalls))),
            "confusion": np.mean(confs, axis=0),
            "n": {LABELS[i]: int(np.sum(np.array(list(sample_label.values())) == i)) for i in range(3)},
        })
    return results


def report(results: list[dict]) -> str:
    n = results[0]["n"]
    lines = [f"행동: 집음 {n['take']} / 놓음 {n['put']} / 만짐 {n['touch']}  (5겹 교차검증 x 씨앗 5개 평균 ± 표준편차)", "",
             "| 근거 | 집음 vs 놓음 AUC | 집음 오판 5% 에서 놓음 잡는 비율 |", "|---|---|---|"]
    for r in results:
        a, s = r["auc_take_vs_put"]
        rc, rs = r["put_recall_at_5pct"]
        lines.append(f"| {r['features']} ({r['dims']}) | {a:.3f} ± {s:.3f} | {rc:.0%} ± {rs:.0%} |")
    lines.append("")
    for r in results:
        c = r["confusion"]
        lines.append(f"{r['features']} 혼동표(행=정답 take/put/touch, 열=예측): "
                     + " / ".join("[" + ", ".join(f"{v:.0f}" for v in row) + "]" for row in c))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="몇 번째 샘플부터 (나눠 돌리기)")
    parser.add_argument("--limit", type=int, help="몇 개 샘플만")
    parser.add_argument("--extract-only", action="store_true", help="특징만 뽑고 채점은 나중에")
    args = parser.parse_args()
    if not args.eval_only:
        extract(args.start, args.limit)
        if args.extract_only:
            return
    # 나눠 뽑은 특징을 모두 합쳐 채점
    parts = [dict(np.load(f, allow_pickle=False)) for f in sorted(OUT.glob("features_*.npz"))]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    text = report(evaluate(d))
    print("\n" + text)
    (OUT / "results.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
