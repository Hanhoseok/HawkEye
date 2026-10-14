"""선반 지도 그리기 — 확정된 물건 자리, 방금 사라진/생긴 자리, 선반 상태를 영상 위에 표시한다.

    선반 테두리 색 = 상태:  초록 ok(지도 확정) / 주황 blocked(사람이 가림) / 노랑 checking(다녀간 뒤 비교 중) / 회색 learning
    물건 자리      = 얇은 초록 상자 + '단-번째' (예: 1-2)
    방금 사라진 자리 = 굵은 빨강 + TAKEN, 방금 생긴 자리 = 굵은 파랑 + PUT  (변화 확정 뒤 몇 초간)

영상 위 글자는 OpenCV 라 영어로 쓴다. 한글 설명은 상황판(live_panel)에 나온다.
"""

from __future__ import annotations

import cv2

STATUS_COLORS = {  # BGR
    "ok": (90, 200, 90), "blocked": (0, 150, 255), "checking": (0, 215, 255), "learning": (160, 160, 160),
}
RECENT_MS = 4000.0


def _label(image, text: str, org, color, scale: float = 0.45) -> None:
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _box(image, bbox, color, thickness: int) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_shelves(image, watcher, now_ms: float):
    """watcher(ShelfWatcher)의 선반마다 지도와 상태를 그린 새 이미지를 돌려준다."""
    canvas = image.copy()
    for _, monitor in watcher.monitors:
        status = monitor.status
        color = STATUS_COLORS[status]
        x1, y1, x2, y2 = (int(v) for v in monitor.box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if status == "blocked":
            who = ",".join(f"P{p}" for p in monitor.blocking if p > 0) or "?"
            text = f"{monitor.name}: blocked by {who}"
        elif status == "checking":
            text = f"{monitor.name}: checking"
        elif status == "learning":
            text = f"{monitor.name}: learning"
        else:
            text = f"{monitor.name}: {len(monitor.state or [])} items"
        # 구역 이름과 겹치지 않게 선반 상자 아래에 (화면 밖이면 상자 안 아래쪽)
        h = canvas.shape[0]
        ty = y2 + 18 if y2 + 18 < h - 4 else y2 - 6
        _label(canvas, text, (x1 + 4, ty), color, 0.5)

        # 확정된 물건 자리 (가린 동안에도 기억하고 있는 지도를 보여 준다)
        for slot in monitor.state or []:
            _box(canvas, slot.item.bbox, (90, 200, 90), 1)
            sx, sy = int(slot.item.bbox[0]), int(slot.item.bbox[1])
            _label(canvas, f"{slot.row}-{slot.index}", (sx + 2, sy + 12), (90, 200, 90), 0.38)

        # 방금 확정된 변화
        if monitor.changes and now_ms - monitor.changes[-1].end_ms <= RECENT_MS:
            change = monitor.changes[-1]
            for slot in change.removed:
                _box(canvas, slot.item.bbox, (0, 0, 255), 3)
                _label(canvas, "TAKEN", (int(slot.item.bbox[0]), int(slot.item.bbox[3]) + 14), (0, 0, 255), 0.5)
            for slot in change.added:
                _box(canvas, slot.item.bbox, (255, 120, 0), 3)
                _label(canvas, "PUT", (int(slot.item.bbox[0]), int(slot.item.bbox[3]) + 14), (255, 120, 0), 0.5)
    return canvas


def shelf_lines(watcher) -> list[str]:
    """상황판용 한 줄 요약 (한글)."""
    lines = []
    for _, monitor in watcher.monitors:
        n = len(monitor.state or [])
        status = monitor.status
        if status == "blocked":
            who = ", ".join(f"손님 {p}" for p in monitor.blocking if p > 0) or "누군가"
            lines.append(f"{monitor.name}: 물건 {n}개 기억 · {who} 가림")
        elif status == "checking":
            lines.append(f"{monitor.name}: 다녀간 뒤 비교 중")
        elif status == "learning":
            lines.append(f"{monitor.name}: 처음 배치 확인 중")
        else:
            lines.append(f"{monitor.name}: 물건 {n}개 확정")
    return lines
