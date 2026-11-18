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
from tkinter import ttk
from datetime import datetime
from typing import Callable, Optional

from .app import Monitor
from .calibration import DONE, MEASURING, CalibrationError, LevelCapture, compute_settings
from .config import AppConfig, ConfigError, MAX_LANES, load_config, update_lane_in_toml
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
        self.dialogs: dict[int, "CalibrationDialog"] = {}
        self.product_dialogs: dict[int, "ProductDialog"] = {}
        self.pending_calib: dict[int, tuple] = {}     # 레인 → (보정 계산 결과, 측정값) — 재입고 종료 때 저장

        self.buttons: dict[int, tk.Button] = {}
        self.button_windows: dict[int, int] = {}
        self.name_buttons: dict[int, tk.Button] = {}
        self.name_button_windows: dict[int, int] = {}
        self.name_dialogs: dict[int, "NameDialog"] = {}
        for lane_id in range(1, MAX_LANES + 1):
            if lane_id in monitor.lanes:
                b = tk.Button(self.root, font=self.f_norm, width=12,
                              command=lambda i=lane_id: self._toggle_restock(i))
                self.buttons[lane_id] = b
                self.button_windows[lane_id] = self.canvas.create_window(0, 0, window=b)
                nb = tk.Button(self.root, text="이름 수정", font=self.f_small, width=12,
                               command=lambda i=lane_id: self._open_name_dialog(i))
                self.name_buttons[lane_id] = nb
                self.name_button_windows[lane_id] = self.canvas.create_window(0, 0, window=nb)

    # ------------------------------------------------------------------
    def _close(self) -> None:
        self.on_close()
        self.root.destroy()

    def _toggle_restock(self, lane_id: int) -> None:
        lane = self.monitor.lanes[lane_id]
        if lane.restock:
            self.finish_restock(lane_id)
            return
        # 재입고 시작 = 재입고 모드(중간 이벤트 억제) + 짧은 보정 창
        self.pending_calib.pop(lane_id, None)
        self.monitor.set_restock(lane_id, True, time.monotonic(), time.time())
        if lane_id not in self.dialogs:
            self.dialogs[lane_id] = CalibrationDialog(self, lane_id)

    def _open_name_dialog(self, lane_id: int) -> None:
        dlg = self.name_dialogs.get(lane_id)
        if dlg is not None:
            dlg.top.lift()
            return
        self.name_dialogs[lane_id] = NameDialog(self, lane_id)

    def finish_restock(self, lane_id: int) -> None:
        """재입고 종료: 측정한 보정 결과가 있으면 상품 정보 입력 창을 띄우고, 없으면 바로 끝낸다."""
        dlg = self.dialogs.get(lane_id)
        if dlg is not None:
            dlg.close()
        pending = self.pending_calib.get(lane_id)
        if pending is None:
            self.monitor.set_restock(lane_id, False, time.monotonic(), time.time())
            return
        existing = self.product_dialogs.get(lane_id)
        if existing is not None:
            existing.top.lift()
            return
        self.product_dialogs[lane_id] = ProductDialog(self, lane_id, *pending)

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
            banner = "재입고 중 · 보정 측정 완료" if lane_id in self.pending_calib else \
                f"재입고 중 (관측 {obs if obs is not None else '?'}개)"
            c.create_text(cx, y0 + 18, text=banner, fill="white", font=self.f_small, tags="dyn")
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
            c.coords(self.button_windows[lane_id], x1 - 52, y + 70)
            c.tag_raise(self.button_windows[lane_id])
        nb = self.name_buttons.get(lane_id)
        if nb is not None:
            c.coords(self.name_button_windows[lane_id], x1 - 52, y + 40)
            c.tag_raise(self.name_button_windows[lane_id])


# ---------------------------------------------------------------------------
# 재입고 + 짧은 보정 창
#   재입고 시작 → CalibrationDialog: 빈 → 1개 → 2개 측정, 결과는 RailWindow.pending_calib 에 보관
#   재입고 종료 → (측정 결과가 있으면) ProductDialog: 상품 이름·ID·최대 수량 입력 → 저장·적용
# ---------------------------------------------------------------------------
class CalibrationDialog:
    """빈 상태 → 1개 → 2개 위치를 재서 상품 지름과 0개·1개 기준을 계산한다. 저장은 재입고 종료 때 한다."""

    STEPS = [
        ("1/3  빈 레인", "레인의 상품을 모두 빼고, 밀판이 앞으로 나와 멈출 때까지 기다린 뒤 누르세요.", "비웠음 → 측정"),
        ("2/3  상품 1개", "상품을 1개 넣고 손을 뗀 뒤 누르세요.", "1개 넣음 → 측정"),
        ("3/3  상품 2개", "상품을 1개 더 넣어 2개로 만들고 손을 뗀 뒤 누르세요.", "2개 넣음 → 측정"),
    ]
    TIMEOUT_S = 30

    def __init__(self, window: "RailWindow", lane_id: int):
        self.win, self.lane_id = window, lane_id
        self.monitor = window.monitor
        self.lane_cfg = window.cfg.lane(lane_id)
        self.step = 0
        self.results = []
        self.capture: Optional[LevelCapture] = None
        self.capture_started = 0.0
        self.live: Optional[float] = None
        self.closed = False

        top = self.top = tk.Toplevel(window.root)
        top.title(f"레인 {lane_id} 재입고 · 짧은 보정")
        top.configure(bg=BG, padx=18, pady=14)
        top.resizable(False, False)
        top.protocol("WM_DELETE_WINDOW", self.close)
        f = window.f_norm
        self.title = tk.Label(top, font=window.f_title, bg=BG, anchor="w")
        self.title.pack(fill="x")
        self.text = tk.Label(top, font=f, bg=BG, anchor="w", justify="left", wraplength=440)
        self.text.pack(fill="x", pady=(6, 8))
        self.bar = ttk.Progressbar(top, length=440, maximum=100)
        self.bar.pack(fill="x")
        self.status = tk.Label(top, font=window.f_small, bg=BG, anchor="w", fg="#555", wraplength=440,
                               justify="left")
        self.status.pack(fill="x", pady=(4, 8))
        self.buttons = tk.Frame(top, bg=BG)
        self.buttons.pack(fill="x", pady=(6, 0))
        self.primary = tk.Button(self.buttons, font=f, width=16, command=self._on_primary)
        self.primary.pack(side="left")
        tk.Button(self.buttons, font=f, text="처음부터", command=self._restart).pack(side="left", padx=6)
        self.secondary = tk.Button(self.buttons, font=f, text="보정 없이 재입고만", command=self._skip)
        self.secondary.pack(side="right")

        self.monitor.sample_listeners.append(self._on_sample)
        self._show_step()
        self._tick()

    # ------------------------------------------------------------------
    def _on_sample(self, lane_id: int, t_ms: int, d: float) -> None:
        if lane_id != self.lane_id:
            return
        self.live = d
        if self.capture is not None:
            self.capture.feed(t_ms, d)

    def _show_step(self, note: str = "") -> None:
        title, text, button = self.STEPS[self.step]
        self.title.configure(text=f"레인 {self.lane_id} · {title}")
        self.text.configure(text=text + (f"\n\n{note}" if note else ""), fg="#b03a2e" if note else "#1b1b1b")
        self.primary.configure(text=button, state="normal")
        self.secondary.configure(text="보정 없이 재입고만", command=self._skip)
        self.bar["value"] = 0
        self.capture = None

    def _on_primary(self) -> None:
        if self.step >= len(self.STEPS):
            self.win.finish_restock(self.lane_id)       # 측정 완료 후: 바로 재입고 종료 + 상품 정보 입력
            return
        self.capture = LevelCapture(median_window=5)
        self.capture_started = time.monotonic()
        self.primary.configure(state="disabled")

    def _restart(self) -> None:
        self.step, self.results = 0, []
        self.win.pending_calib.pop(self.lane_id, None)
        self._show_step()

    def _skip(self) -> None:
        self.win.pending_calib.pop(self.lane_id, None)
        self.close()

    def _tick(self) -> None:
        if self.closed:
            return
        live = "-" if self.live is None else f"{self.live:.0f} mm"
        if self.capture is not None:
            state, frac = self.capture.progress()
            waited = time.monotonic() - self.capture_started
            if state == DONE:
                self._step_done(self.capture.result)
            elif state == MEASURING:
                self.bar["value"] = 40 + 60 * frac
                self.status.configure(text=f"측정 중… (현재 {live})")
            else:
                self.bar["value"] = 40 * frac
                hint = ""
                if self.live is None:
                    hint = " · 이 레인 데이터가 들어오지 않습니다 (보드·Wi-Fi 확인)"
                elif waited > self.TIMEOUT_S:
                    hint = " · 값이 계속 흔들립니다. 손을 떼고 밀판이 멈출 때까지 기다리세요"
                self.status.configure(text=f"값이 안정되길 기다리는 중… (현재 {live}){hint}")
        elif self.step < len(self.STEPS):
            self.status.configure(text=f"현재 거리 {live}")
        self.top.after(150, self._tick)

    def _step_done(self, res) -> None:
        self.capture = None
        if self.step == 1 and self.results[0].median - res.median < 3:
            self._show_step(f"빈 상태({self.results[0].median:.0f}mm)와 거리가 거의 같습니다 ({res.median:.0f}mm). "
                            "상품이 밀판을 밀었는지 확인하고 다시 측정하세요.")
            return
        self.results.append(res)
        self.step += 1
        if self.step < len(self.STEPS):
            self._show_step()
            return
        try:
            calib = compute_settings(*self.results, valid_min_mm=self.lane_cfg.valid_min_mm)
        except CalibrationError as e:
            self.step, self.results = 1, self.results[:1]
            self._show_step(str(e))
            return
        self.win.pending_calib[self.lane_id] = (calib, list(self.results))
        self.title.configure(text=f"레인 {self.lane_id} · 측정 완료")
        self.text.configure(text="\n".join(calib_summary(calib, self.results)) +
                            "\n\n상품을 마저 채운 뒤 레인의 '재입고 종료' 버튼을 누르면 상품 이름을 입력하고 저장합니다."
                            "\n이 창은 닫아도 측정 결과는 유지됩니다.", fg="#1b1b1b")
        self.bar["value"] = 100
        self.status.configure(text="")
        self.primary.configure(text="지금 재입고 종료", state="normal")
        self.secondary.configure(text="닫기 (재입고 계속)", command=self.close)

    def close(self) -> None:
        """창만 닫는다. 재입고 모드와 측정 결과(있으면)는 유지되고, '재입고 종료' 버튼으로 끝낸다."""
        if self.closed:
            return
        self.closed = True
        if self._on_sample in self.monitor.sample_listeners:
            self.monitor.sample_listeners.remove(self._on_sample)
        self.win.dialogs.pop(self.lane_id, None)
        try:
            self.top.destroy()
        except tk.TclError:
            pass


def calib_summary(c, results) -> list[str]:
    e, o, t = results
    lines = [f"빈 레인 {e.median:.0f} mm · 1개 {o.median:.0f} mm · 2개 {t.median:.0f} mm",
             f"상품 지름(진열 방향 길이) {c.thickness_mm:g} mm · 0개↔1개 간격 {c.gap01_mm:g} mm",
             f"허용 오차: 0개 ±{c.count_tolerance_mm[0]:g} · 1개 ±{c.count_tolerance_mm[1]:g}"
             f" · 2개 이상 ±{c.count_tolerance_mm[2]:g} mm"]
    return lines + [f"⚠ {w}" for w in c.warnings]


class ProductDialog:
    """재입고 종료 시: 측정한 보정값과 함께 상품 이름·ID·최대 수량을 저장하고 적용한다."""

    def __init__(self, window: "RailWindow", lane_id: int, calib, results):
        self.win, self.lane_id, self.calib, self.results = window, lane_id, calib, results
        self.lane_cfg = window.cfg.lane(lane_id)
        self.closed = False
        top = self.top = tk.Toplevel(window.root)
        top.title(f"레인 {lane_id} 재입고 종료 · 상품 정보")
        top.configure(bg=BG, padx=18, pady=14)
        top.resizable(False, False)
        top.protocol("WM_DELETE_WINDOW", self.close)
        f = window.f_norm
        tk.Label(top, text=f"레인 {lane_id} · 상품 정보 입력", font=window.f_title, bg=BG, anchor="w").pack(fill="x")
        tk.Label(top, text="\n".join(calib_summary(calib, results)), font=window.f_small, bg=BG, anchor="w",
                 justify="left", wraplength=440, fg="#444").pack(fill="x", pady=(6, 10))
        form = tk.Frame(top, bg=BG)
        form.pack(fill="x")
        self.name_var = tk.StringVar(value=self.lane_cfg.product_name)
        self.id_var = tk.StringVar(value=self.lane_cfg.product_id)
        self.max_var = tk.IntVar(value=max(1, min(self.lane_cfg.max_count, calib.max_count_geometry)))
        self.name_entry = tk.Entry(form, textvariable=self.name_var, font=f, width=32)
        rows = [("상품 이름", self.name_entry),
                ("상품 ID", tk.Entry(form, textvariable=self.id_var, font=f, width=32)),
                (f"최대 수량 (1~{calib.max_count_geometry})",
                 tk.Spinbox(form, from_=1, to=calib.max_count_geometry, textvariable=self.max_var, font=f, width=6))]
        for r, (label, widget) in enumerate(rows):
            tk.Label(form, text=label, font=f, bg=BG, anchor="w").grid(row=r, column=0, sticky="w", pady=3)
            widget.grid(row=r, column=1, sticky="w", padx=8, pady=3)
        tk.Label(top, text="저장하면 이 레인 설정이 바뀌고 재입고 모드가 끝납니다. 안정된 수량으로 RESYNC 됩니다.",
                 font=window.f_small, bg=BG, fg="#555", anchor="w", wraplength=440, justify="left").pack(fill="x", pady=(8, 6))
        buttons = tk.Frame(top, bg=BG)
        buttons.pack(fill="x")
        tk.Button(buttons, text="저장하고 재입고 종료", font=f, command=self.save).pack(side="left")
        tk.Button(buttons, text="취소 (재입고 계속)", font=f, command=self.close).pack(side="left", padx=6)
        tk.Button(buttons, text="보정 버리고 재입고 종료", font=f, command=self.discard).pack(side="right")
        self.name_entry.focus_set()
        self.name_entry.select_range(0, "end")

    def save(self) -> None:
        from tkinter import messagebox
        name, pid = self.name_var.get().strip(), self.id_var.get().strip()
        try:
            max_count = int(self.max_var.get())
        except (tk.TclError, ValueError):
            max_count = 0
        c = self.calib
        if not name or not pid:
            messagebox.showerror("입력 확인", "상품 이름과 상품 ID 를 입력하세요.", parent=self.top)
            return
        if not 1 <= max_count <= c.max_count_geometry:
            messagebox.showerror("입력 확인", f"최대 수량은 1~{c.max_count_geometry} 사이여야 합니다.", parent=self.top)
            return
        refs, tols = c.for_max_count(max_count)
        values = {"product_id": pid, "product_name": name, "thickness_mm": c.thickness_mm,
                  "empty_distance_mm": c.empty_distance_mm, "max_count": max_count,
                  "count_distance_mm": refs, "count_tolerance_mm": tols,
                  "hysteresis_mm": c.hysteresis_mm, "median_window": c.median_window}
        e, o, t = self.results
        note = (f"자동 보정 {datetime.now():%Y-%m-%d %H:%M} · 빈 {e.median:.0f} / 1개 {o.median:.0f} / 2개 {t.median:.0f} mm"
                f" · 지름 {c.thickness_mm:g} · 0↔1 간격 {c.gap01_mm:g}")
        path = self.win.cfg.config_path
        try:
            if path is None:
                raise ConfigError("설정 파일 경로를 알 수 없습니다")
            update_lane_in_toml(path, self.lane_id, values, note=note)
            new_lane = load_config(path, enable_all=True).lane(self.lane_id)
        except (ConfigError, OSError) as ex:
            messagebox.showerror("저장 실패", f"설정을 저장하지 못했습니다. 이전 설정을 유지합니다.\n\n{ex}", parent=self.top)
            return
        new_lane.enabled = True
        self.win.pending_calib.pop(self.lane_id, None)
        self.win.monitor.recalibrate_lane(new_lane, time.monotonic(), time.time(),
                                          detail={"thickness_mm": c.thickness_mm, "gap01_mm": c.gap01_mm})
        self.close()

    def discard(self) -> None:
        self.win.pending_calib.pop(self.lane_id, None)
        self.win.monitor.set_restock(self.lane_id, False, time.monotonic(), time.time())
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.win.product_dialogs.pop(self.lane_id, None)
        try:
            self.top.destroy()
        except tk.TclError:
            pass


class NameDialog:
    """상품 이름·ID 만 바꾼다. 보정값과 현재 수량은 그대로 두고 config.toml 에 저장한다."""

    def __init__(self, window: "RailWindow", lane_id: int):
        self.win, self.lane_id = window, lane_id
        lane_cfg = window.cfg.lane(lane_id)
        self.closed = False
        top = self.top = tk.Toplevel(window.root)
        top.title(f"레인 {lane_id} 상품 이름 수정")
        top.configure(bg=BG, padx=18, pady=14)
        top.resizable(False, False)
        top.protocol("WM_DELETE_WINDOW", self.close)
        f = window.f_norm
        tk.Label(top, text=f"레인 {lane_id} · 상품 이름 수정", font=window.f_title, bg=BG, anchor="w").pack(fill="x")
        form = tk.Frame(top, bg=BG)
        form.pack(fill="x", pady=(8, 4))
        self.name_var = tk.StringVar(value=lane_cfg.product_name)
        self.id_var = tk.StringVar(value=lane_cfg.product_id)
        self.name_entry = tk.Entry(form, textvariable=self.name_var, font=f, width=32)
        for r, (label, widget) in enumerate([("상품 이름", self.name_entry),
                                             ("상품 ID", tk.Entry(form, textvariable=self.id_var, font=f, width=32))]):
            tk.Label(form, text=label, font=f, bg=BG, anchor="w").grid(row=r, column=0, sticky="w", pady=3)
            widget.grid(row=r, column=1, sticky="w", padx=8, pady=3)
        tk.Label(top, text="보정값과 현재 수량은 바뀌지 않습니다. 이후 기록되는 이벤트에 새 이름이 들어갑니다.",
                 font=window.f_small, bg=BG, fg="#555", anchor="w", wraplength=420, justify="left").pack(fill="x", pady=(4, 8))
        buttons = tk.Frame(top, bg=BG)
        buttons.pack(fill="x")
        tk.Button(buttons, text="저장", font=f, width=10, command=self.save).pack(side="left")
        tk.Button(buttons, text="취소", font=f, width=8, command=self.close).pack(side="left", padx=6)
        top.bind("<Return>", lambda _e: self.save())
        top.bind("<Escape>", lambda _e: self.close())
        self.name_entry.focus_set()
        self.name_entry.select_range(0, "end")

    def save(self) -> None:
        from tkinter import messagebox
        name, pid = self.name_var.get().strip(), self.id_var.get().strip()
        if not name or not pid:
            messagebox.showerror("입력 확인", "상품 이름과 상품 ID 를 입력하세요.", parent=self.top)
            return
        path = self.win.cfg.config_path
        try:
            if path is None:
                raise ConfigError("설정 파일 경로를 알 수 없습니다")
            update_lane_in_toml(path, self.lane_id, {"product_name": name, "product_id": pid}, drop_comments=False)
        except (ConfigError, OSError) as ex:
            messagebox.showerror("저장 실패", f"저장하지 못했습니다. 이전 이름을 유지합니다.\n\n{ex}", parent=self.top)
            return
        self.win.monitor.rename_product(self.lane_id, name, pid, time.time())
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.win.name_dialogs.pop(self.lane_id, None)
        try:
            self.top.destroy()
        except tk.TclError:
            pass
