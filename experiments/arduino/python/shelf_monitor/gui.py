"""로컬 창(tkinter): 스프링 푸셔 레일 5개를 위에서 내려다본 모습으로 표시.

그림 구성 (첨부 사진의 레일 기준, 위쪽 = 레일 뒤쪽 = 센서 위치, 아래쪽 = 앞쪽 투명 스토퍼)
  - 검은 레일과 롤러, 양옆의 은색 와이어 가이드
  - 맨 뒤 브래킷의 VL53L0X 센서와 측정 빔(빨간 점선)
  - 밀판: 필터된 실시간 거리 위치에 그림 (확정 전 움직임도 보임)
  - 상품: "확정 수량" 만큼 앞 스토퍼부터 쌓아 그림 (캔은 원, 상자는 사각형)
  - 오른쪽 눈금: 수량별 기준 거리, 왼쪽 연녹색 막대: 허용 오차(tolerance) 구간
"""
from __future__ import annotations

import queue
import time
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable, Optional

from .app import Monitor
from .config import AppConfig, MAX_LANES
from .sinks import STATE_KO, describe

LANE_COLORS = ["#d62828", "#2a9d8f", "#f4a261", "#4ea8de", "#9b5de5"]
STATE_COLORS = {
    "INITIALIZING": "#7a869a",
    "STABLE": "#2e9e5b",
    "CHANGING": "#e0a100",
    "UNCERTAIN": "#e07a00",
    "OUT_OF_RANGE": "#8e44ad",
    "SENSOR_ERROR": "#d0342c",
}
BG = "#efe9e1"
PLATE_MM = 14      # 화면 표시용 밀판 두께
STOP_MM = 12       # 화면 표시용 앞 스토퍼 두께


def _darker(hex_color: str, f: float = 0.65) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"#{int(r * f):02x}{int(g * f):02x}{int(b * f):02x}"


def _lighter(hex_color: str, f: float = 0.45) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"#{int(r + (255 - r) * f):02x}{int(g + (255 - g) * f):02x}{int(b + (255 - b) * f):02x}"


class RailWindow:
    def __init__(self, cfg: AppConfig, monitor: Monitor, q: queue.Queue, pump: Callable[[], None],
                 on_close: Callable[[], None]):
        self.cfg, self.monitor, self.q, self.pump, self.on_close = cfg, monitor, q, pump, on_close
        self.root = tk.Tk()
        self.root.title("스프링 푸셔 진열대 모니터")
        self.root.geometry("1320x900")
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        fam = "Malgun Gothic"
        self.f_small = tkfont.Font(family=fam, size=8)
        self.f_norm = tkfont.Font(family=fam, size=10)
        self.f_bold = tkfont.Font(family=fam, size=11, weight="bold")
        self.f_big = tkfont.Font(family=fam, size=26, weight="bold")
        self.f_title = tkfont.Font(family=fam, size=14, weight="bold")

        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        log_frame = tk.Frame(self.root, bg=BG)
        log_frame.pack(fill="x", padx=10, pady=(0, 8))
        tk.Label(log_frame, text="최근 이벤트 (확정 이벤트는 logs/events.jsonl 에 저장)", bg=BG,
                 font=self.f_norm, anchor="w").pack(fill="x")
        self.log = tk.Listbox(log_frame, height=7, font=self.f_norm, activestyle="none")
        self.log.pack(fill="x")
        self._shown_seq = 0
        # 화면 전용 상태 (판단 로직과 무관): 밀판 위치 평활값, 상태 표시 지연
        self._view: dict[int, dict] = {}

        self.buttons: dict[int, tk.Button] = {}
        self.button_windows: dict[int, int] = {}
        for lane_id in range(1, MAX_LANES + 1):
            if lane_id in monitor.lanes:
                b = tk.Button(self.root, font=self.f_norm, width=12,
                              command=lambda i=lane_id: self._toggle_restock(i))
                self.buttons[lane_id] = b
                self.button_windows[lane_id] = self.canvas.create_window(0, 0, window=b)

    # ------------------------------------------------------------------
    def _close(self) -> None:
        self.on_close()
        self.root.destroy()

    def _toggle_restock(self, lane_id: int) -> None:
        lane = self.monitor.lanes[lane_id]
        self.monitor.set_restock(lane_id, not lane.restock, time.monotonic(), time.time())

    def run(self) -> None:
        self._loop()
        self.root.mainloop()

    def _loop(self) -> None:
        self.pump()
        self._redraw()
        self._update_log()
        self.root.after(100, self._loop)

    def _update_log(self) -> None:
        new = [r for r in self.monitor.recent if r["pc_event_seq"] > self._shown_seq]
        for rec in reversed(new):
            t = rec["pc_time"][11:19]
            self.log.insert(0, f"{t}  {describe(rec)}")
            colors = {"ADD": "#1d6f42", "REMOVE": "#a12622", "RESYNC": "#6a3fa0",
                      "SENSOR_ERROR": "#c0392b", "INIT": "#1f5fa8"}
            self.log.itemconfig(0, fg=colors.get(rec["event"], "#333333"))
            self._shown_seq = rec["pc_event_seq"]
        while self.log.size() > 200:
            self.log.delete(200)

    # ------------------------------------------------------------------
    # 화면 안정화: 수량 판단은 그대로 두고, 그리는 값만 부드럽게 만든다
    SNAP_MM = 20           # 이보다 크게 움직이면 실제 이동으로 보고 바로 따라감
    SMOOTH = 0.25          # 작은 흔들림 평활 계수 (100ms 마다, 시상수 약 0.35초)
    LABEL_HOLD_S = 1.0     # 상태 표시는 이 시간 이상 유지될 때만 바꿈

    def _smooth_mm(self, lane_id: int, d: Optional[float]) -> Optional[float]:
        v = self._view.setdefault(lane_id, {})
        if d is None:
            return v.get("mm")
        mm = v.get("mm")
        if mm is None or abs(d - mm) > self.SNAP_MM:
            v["mm"] = d
        else:
            v["mm"] = mm + self.SMOOTH * (d - mm)
        return v["mm"]

    def _stable_state(self, lane_id: int, snap: dict, now: float) -> str:
        key = "RESTOCK" if snap["restock"] else snap["state"]
        if snap["pending_resync"] and not snap["error"] and not snap["restock"]:
            key += "+RESYNC"
        v = self._view.setdefault(lane_id, {})
        shown = v.get("state")
        urgent = key.startswith(("SENSOR_ERROR", "RESTOCK")) or (shown or "").startswith("RESTOCK")
        if shown is None or key == shown or urgent:
            v["state"], v["cand"] = key, None
        else:
            cand = v.get("cand")
            if cand is None or cand[0] != key:
                v["cand"] = (key, now)
            elif now - cand[1] >= self.LABEL_HOLD_S:
                v["state"], v["cand"] = key, None
        return v["state"]

    # ------------------------------------------------------------------
    def _redraw(self) -> None:
        c = self.canvas
        c.delete("dyn")
        w = max(c.winfo_width(), 900)
        h = max(c.winfo_height(), 600)
        self._draw_header(w)
        now = time.monotonic()
        snaps = {s["lane_id"]: s for s in self.monitor.snapshots(now)}
        for lane_id, s in snaps.items():
            s["view_mm"] = self._smooth_mm(lane_id, s["display_mm"])
            s["view_state"] = self._stable_state(lane_id, s, now)
        col_w = w / MAX_LANES
        top, info_h = 120, 150
        bottom = h - info_h - 34
        for i in range(MAX_LANES):
            lane_id = i + 1
            cx = col_w * i + col_w / 2
            self._draw_lane(lane_id, cx, top, bottom, snaps.get(lane_id), LANE_COLORS[i])
            self._draw_info(lane_id, cx, bottom + 42, col_w, snaps.get(lane_id), LANE_COLORS[i])

    def _draw_header(self, w: float) -> None:
        c, m = self.canvas, self.monitor
        c.create_rectangle(0, 0, w, 54, fill="#23272e", outline="", tags="dyn")
        c.create_text(16, 27, text="스프링 푸셔 진열대 모니터", anchor="w", fill="white",
                      font=self.f_title, tags="dyn")
        if m.link_ok:
            link_txt, link_col = f"● 수신 중  {m.port}", "#4cd07d"
        elif m.connected:
            link_txt, link_col = f"● 데이터 없음 (stale)  {m.port}", "#f0b429"
        else:
            link_txt, link_col = f"● 연결 안 됨  {m.link_note}", "#ff6b6b"
        c.create_text(300, 18, text=link_txt, anchor="w", fill=link_col, font=self.f_norm, tags="dyn")
        st = m.stats()
        boards = "  ".join(
            f"{'●' if not d.stale else '○'}{d.device_id}" + (f"({d.info['ip']})" if d.info.get("ip") else "")
            for d in sorted(m.devices.values(), key=lambda d: d.device_id))
        info = (f"보드 {st['devices']}대 {boards or '-'}   "
                f"누락 {st['missing']}  중복 {st['duplicates']}  재부팅 {st['reboots']}  파싱오류 {m.parse_errors}")
        if m.backlog:
            info += "  [수신 지연]"
        c.create_text(300, 38, text=info, anchor="w", fill="#c9d1d9", font=self.f_small, tags="dyn")
        if m.device_config_error:
            c.create_text(w - 16, 27, text=f"펌웨어 설정 오류: {m.device_config_error}", anchor="e",
                          fill="#ff6b6b", font=self.f_norm, width=w * 0.35, tags="dyn")
        c.create_text(w - 16, 44, text="화면 위 = 레일 뒤(센서) · 아래 = 앞(고객 쪽)", anchor="e", fill="#8b949e",
                      font=self.f_small, tags="dyn")

    def _draw_lane(self, lane_id: int, cx: float, top: float, bottom: float,
                   snap: Optional[dict], color: str) -> None:
        c = self.canvas
        lane_cfg = self.cfg.lane(lane_id)
        active = snap is not None
        rail_fill = "#262626" if active else "#8d8d8d"
        roller_a, roller_b = ("#6e6e6e", "#3e3e3e") if active else ("#b5b5b5", "#a0a0a0")

        y0, y1 = top, bottom
        # 와이어 가이드 (사진의 은색 U자 와이어)
        for dx in (-62, 62):
            c.create_line(cx + dx, y0 - 40, cx + dx, y1 + 28, fill="#b9bec4", width=4, tags="dyn")
            c.create_line(cx + dx - 1, y0 - 40, cx + dx - 1, y1 + 28, fill="#f4f6f8", width=1, tags="dyn")
        c.create_line(cx - 62, y0 - 40, cx - 48, y0 - 52, cx + 48, y0 - 52, cx + 62, y0 - 40,
                      fill="#b9bec4", width=3, smooth=True, tags="dyn")
        # 레일 본체 + 롤러
        c.create_rectangle(cx - 50, y0 - 10, cx + 50, y1 + 22, fill="#111111" if active else "#7a7a7a",
                           outline="", tags="dyn")
        c.create_rectangle(cx - 40, y0, cx + 40, y1 + 12, fill=rail_fill, outline="", tags="dyn")
        y = y0 + 4
        k = 0
        while y < y1 + 10:
            c.create_line(cx - 37, y, cx + 37, y, fill=roller_a if k % 2 == 0 else roller_b, width=2, tags="dyn")
            y += 6
            k += 1
        # 뒤 브래킷 + 센서
        c.create_rectangle(cx - 56, y0 - 36, cx + 56, y0 - 8, fill="#161616", outline="#000000", tags="dyn")
        sensor_ok = active and not snap["error"]
        c.create_rectangle(cx - 17, y0 - 33, cx + 17, y0 - 12, fill="#1e2a55",
                           outline="#5b7be0" if sensor_ok else "#d0342c", width=2, tags="dyn")
        c.create_oval(cx - 3, y0 - 17, cx + 3, y0 - 11, fill="#ffcc33" if sensor_ok else "#555", outline="",
                      tags="dyn")
        c.create_text(cx, y0 - 26, text="ToF", fill="white", font=self.f_small, tags="dyn")

        if lane_cfg is None or not active:
            c.create_rectangle(cx - 58, (y0 + y1) / 2 - 22, cx + 58, (y0 + y1) / 2 + 22,
                               fill="#f5f5f5", outline="#999", tags="dyn")
            c.create_text(cx, (y0 + y1) / 2, text="비활성\n(센서 미설치)" if lane_cfg else "미설정",
                          justify="center", fill="#555", font=self.f_norm, tags="dyn")
            return

        refs = [lane_cfg.reference_mm(n) for n in range(lane_cfg.max_count + 1)]
        th = lane_cfg.thickness_mm
        # 앞 끝 위치: 상품 1개가 밀판 바로 앞에 놓였을 때 그 앞면. 빈 밀판이 그보다 덜 나가면
        # 그 사이는 밀판이 들어가지 못하는 고정 블록(첫 상품만 밀판을 조금 미는 레일)
        stop_mm = max(refs[0], refs[1] + th) if len(refs) > 1 else refs[0]
        total_mm = stop_mm + PLATE_MM + STOP_MM
        scale = (y1 - y0) / total_mm

        def ym(mm: float) -> float:
            return y0 + mm * scale

        y_stop = ym(stop_mm + PLATE_MM)
        if stop_mm - refs[0] > 5:
            yb0 = ym(refs[0] + PLATE_MM)
            c.create_rectangle(cx - 40, yb0, cx + 40, y_stop, fill="#55595f", outline="#3a3d42", tags="dyn")
            yy = yb0 + 6
            while yy < y_stop - 2:
                c.create_line(cx - 38, yy + 6, cx - 26, yy, fill="#6d7178", tags="dyn")
                c.create_line(cx + 26, yy + 6, cx + 38, yy, fill="#6d7178", tags="dyn")
                yy += 8
            c.create_text(cx - 54, (yb0 + y_stop) / 2, text="블록", anchor="e", fill="#7d7468",
                          font=self.f_small, tags="dyn")
        # 허용 오차 구간(왼쪽)과 수량 눈금(오른쪽)
        for n, ref in enumerate(refs):
            hi = n == snap["count"]
            c.create_rectangle(cx - 47, ym(ref - lane_cfg.tolerance_for(n)), cx - 43, ym(ref + lane_cfg.tolerance_for(n)),
                               fill="#7fd99a" if hi else "#3f6b4d", outline="", tags="dyn")
            c.create_line(cx + 42, ym(ref), cx + 52, ym(ref), fill="#2e9e5b" if hi else "#9a9a9a",
                          width=2 if hi else 1, tags="dyn")
            c.create_text(cx + 72, ym(ref), text=str(n), fill="#1d6f42" if hi else "#8a8073",
                          font=self.f_bold if hi else self.f_small, tags="dyn")
        # 앞 스토퍼 (투명 아크릴)
        c.create_rectangle(cx - 48, y_stop, cx + 48, y_stop + STOP_MM * scale + 6, fill="#d6ecf3",
                           outline="#8fb9c7", width=2, tags="dyn")
        c.create_text(cx, y_stop + STOP_MM * scale + 18, text="앞 (고객 쪽)", fill="#7d7468",
                      font=self.f_small, tags="dyn")

        # 상품: 확정 수량만큼. 실시간 밀판보다 뒤에 있어야 할 상품은 흐리게(확정 전 변화)
        count = snap["count"]
        d = snap["view_mm"]
        live_front = ym(d + PLATE_MM) if d is not None and not snap["error"] else None
        if count is not None:
            for k in range(count):
                yb = y_stop - k * th * scale
                yt = y_stop - (k + 1) * th * scale
                ghost = live_front is not None and yt < live_front - 0.4 * th * scale
                if ghost:   # 확정 수량에는 있지만 밀판이 이미 그 자리에 와 있음 → 점선 윤곽만
                    c.create_rectangle(cx - 33, yt + 2, cx + 33, yb - 2, outline=_lighter(color, 0.2), dash=(3, 3),
                                       width=2, tags="dyn")
                elif lane_cfg.shape == "can":
                    r = min((yb - yt) / 2 - 1, 34)
                    ym_ = (yb + yt) / 2
                    c.create_oval(cx - r, ym_ - r, cx + r, ym_ + r, fill=color, outline=_darker(color), width=2,
                                  tags="dyn")
                    c.create_oval(cx - r * 0.72, ym_ - r * 0.72, cx + r * 0.72, ym_ + r * 0.72,
                                  outline=_lighter(color), width=2, tags="dyn")
                    c.create_oval(cx - 3, ym_ - r * 0.5, cx + 3, ym_ - r * 0.5 + 6, fill=_lighter(color, 0.7),
                                  outline="", tags="dyn")
                else:
                    c.create_rectangle(cx - 35, yt + 1, cx + 35, yb - 1, fill=color, outline=_darker(color),
                                       width=2, tags="dyn")
                    c.create_line(cx - 30, yt + 4, cx + 30, yt + 4, fill=_lighter(color), tags="dyn")

        # 밀판: 실시간(필터) 거리, 작은 흔들림만 평활
        state = snap["view_state"].split("+")[0]
        if d is not None and not snap["error"]:
            yb = ym(d)
            yf = ym(d + PLATE_MM)
            c.create_line(cx, y0 - 12, cx, yb - 6, fill="#ff4040", dash=(4, 3), width=2, tags="dyn")
            outline = STATE_COLORS.get(state, "#888") if state != "STABLE" else "#000000"
            c.create_rectangle(cx - 45, yb, cx + 45, yf, fill="#0e0e0e", outline=outline, width=2, tags="dyn")
            c.create_rectangle(cx - 42, yf - 3, cx + 42, yf + 3, fill="#dff1f7", outline="#9cc7d6", tags="dyn")
            c.create_rectangle(cx - 9, yb - 7, cx + 9, yb, fill="#f2f2f2", outline="#aaaaaa", tags="dyn")  # 표적
            c.create_text(cx - 54, yb, text=f"{d:.0f}", anchor="e", fill="#5a5148", font=self.f_small, tags="dyn")

        # 상태 오버레이
        mid = (y0 + y1) / 2
        if snap["restock"]:
            c.create_rectangle(cx - 58, y0 + 6, cx + 58, y0 + 30, fill="#2f6fd6", outline="", tags="dyn")
            obs = snap["restock_observed"]
            c.create_text(cx, y0 + 18, text=f"재입고 중 (관측 {obs if obs is not None else '?'}개)",
                          fill="white", font=self.f_small, tags="dyn")
        if snap["error"]:
            self._badge(cx, mid, f"센서 오류\n{snap['error_kind']}", "#d0342c")
        elif not snap["link_ok"]:
            self._badge(cx, mid, "보드 데이터 없음\n(재고 0 아님)", "#c27c0e")
        elif snap["stale"]:
            self._badge(cx, mid, "측정 없음 (stale)", "#c27c0e")
        elif state == "OUT_OF_RANGE":
            self._badge(cx, mid, "수량 범위 밖\n보정 확인", "#8e44ad")

    def _badge(self, cx: float, cy: float, text: str, color: str) -> None:
        self.canvas.create_rectangle(cx - 60, cy - 24, cx + 60, cy + 24, fill=color, outline="white", width=2,
                                     tags="dyn")
        self.canvas.create_text(cx, cy, text=text, justify="center", fill="white", font=self.f_norm, tags="dyn")

    def _draw_info(self, lane_id: int, cx: float, y: float, col_w: float, snap: Optional[dict], color: str) -> None:
        c = self.canvas
        lane_cfg = self.cfg.lane(lane_id)
        x0, x1 = cx - col_w / 2 + 8, cx + col_w / 2 - 8
        c.create_rectangle(x0, y, x1, y + 128, fill="white", outline="#d8d0c4", tags="dyn")
        c.create_rectangle(x0, y, x0 + 6, y + 128, fill=color if snap else "#bbbbbb", outline="", tags="dyn")
        name = lane_cfg.product_name if lane_cfg else "-"
        title = f"레인 {lane_id} · {name}"
        room = (x1 - 116) - (x0 + 14) if snap is not None else (x1 - x0 - 20)   # 오른쪽 상태 표시 자리 제외
        while len(title) > 6 and self.f_bold.measure(title) > room:
            title = title[:-2] + "…"
        c.create_text(x0 + 14, y + 14, text=title, anchor="w", font=self.f_bold, tags="dyn")
        if snap is None:
            c.create_text(x0 + 14, y + 44, text="비활성", anchor="w", fill="#888", font=self.f_norm, tags="dyn")
            return
        cnt = snap["count"]
        stale = snap["stale"]
        c.create_text(x0 + 14, y + 52, text="—" if cnt is None else str(cnt), anchor="w",
                      fill="#999999" if stale else "#1b1b1b", font=self.f_big, tags="dyn")
        c.create_text(x0 + 64, y + 58, text=f"/ {lane_cfg.max_count}개" + (" (stale)" if stale else ""),
                      anchor="w", fill="#666", font=self.f_norm, tags="dyn")
        state, _, resync = snap["view_state"].partition("+")
        label = "재입고 중" if state == "RESTOCK" else STATE_KO.get(state, state)
        if resync:
            label += " · 재동기화 대기"
        col = "#2f6fd6" if state == "RESTOCK" else STATE_COLORS.get(state, "#888")
        c.create_oval(x1 - 112, y + 8, x1 - 102, y + 18, fill=col, outline="", tags="dyn")
        c.create_text(x1 - 98, y + 13, text=label, anchor="w", fill=col, font=self.f_small, width=96, tags="dyn")
        vm = snap["view_mm"]
        dist = "-" if vm is None else f"{vm:.0f} mm"
        est_v = None if vm is None else self.monitor.lanes[lane_id].continuous_count(vm)
        est = "-" if est_v is None else f"{max(est_v, 0):.1f}"
        c.create_text(x0 + 14, y + 86, text=f"거리 {dist} · 연속추정 {est}", anchor="w", fill="#555",
                      font=self.f_small, tags="dyn")
        since = snap["seconds_since_valid"]
        c.create_text(x0 + 14, y + 102, text=f"센서 {snap['sensor_status']} · 마지막 정상 "
                                               f"{'-' if since is None else f'{since:.1f}초 전'}",
                      anchor="w", fill="#777", font=self.f_small, tags="dyn")
        b = self.buttons.get(lane_id)
        if b is not None:
            b.configure(text="재입고 종료" if snap["restock"] else "재입고 시작",
                        bg="#2f6fd6" if snap["restock"] else "#f0f0f0",
                        fg="white" if snap["restock"] else "black")
            c.coords(self.button_windows[lane_id], x1 - 52, y + 54)
            c.tag_raise(self.button_windows[lane_id])
