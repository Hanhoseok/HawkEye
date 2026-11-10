"use strict";
/* 정답 적기 — 한 쌍 분석 화면(Player, app.js)에 덧붙는 다섯 번째 탭.
   영상을 멈추고 버튼(또는 키)을 누른 뒤 화면에서 사람을 누르면, 그 시각과 화면 위 점이 정답으로 저장된다.
   '누가'는 프로그램의 손님 번호가 아니라 점으로 적는다 — 정답이 프로그램의 판단에 기대지 않게 (workbench/labels.py).
   저장하면 서버가 그 짝을 다시 채점해 요약을 돌려주고, 다른 탭의 점·요약이 바로 바뀐다. */

const PAY_NAME = { paid: "결제 완료", failed: "결제 실패", incomplete: "미완료", cancelled: "취소", none: "결제 시도 없음" };
const NAME_KEY = "wb.label.name";
const cloneJSON = v => JSON.parse(JSON.stringify(v));
const round4 = v => Math.round(v * 10000) / 10000;

Player.prototype.initLabels = function () {
  const L = this.data.labels || {};
  let name = "";
  try { name = localStorage.getItem(NAME_KEY) || ""; } catch (e) { /* 이름 없이도 적을 수 있다 */ }
  this.labels = { checked_by: L.checked_by || name, takes: cloneJSON(L.takes || []), culprits: cloneJSON(L.culprits || []),
    links: cloneJSON(L.links || []), takes_done: Boolean(L.takes_done) };
  this.pays = cloneJSON(this.data.payment_labels || []);
  this.pick = null; this.saveState = ""; this.saveError = ""; this.saveTimer = null; this.paysDirty = false;
};

Player.prototype.labelCount = function () {
  const L = this.labels;
  return L.takes.length + L.culprits.length + L.links.length;
};

/* -- 그 시각 그 점의 사람 (프로그램이 누구로 봤나 — 참고로만 보여 준다) */
Player.prototype.personAt = function (cam, point, ms) {
  const tr = this.data.tracks[cam], x = point[0] * tr.w, y = point[1] * tr.h;
  // 채점(workbench/analyze.py locate)과 같게: 그 시각에서 가까운 장면부터 앞뒤 0.7초 안에서 점을 품은 가장 작은 상자
  const step = 1000 * tr.stride / tr.fps;
  for (let n = 0; n * step <= 700; n++) {
    for (const d of n ? [-n * step, n * step] : [0]) {
      let best = null;
      for (const r of this.rowsAt(cam, ms + d)) {
        if (r[1] > 0 && r[3] <= x && x <= r[5] && r[4] <= y && y <= r[6]) {
          const area = (r[5] - r[3]) * (r[6] - r[4]);
          if (!best || area < best.area) best = { rid: r[1], area };
        }
      }
      if (best) return best.rid;
    }
  }
  return null;
};
Player.prototype.seenAs = function (cam, point, ms) {
  const rid = this.personAt(cam, point, ms);
  if (rid == null) return "프로그램은 여기서 사람을 못 봄";
  return "프로그램: " + (cam === "angle1" ? this.who(rid) : `선반 화면 ${rid}`);
};

/* -- 타임라인 */
Player.prototype.labelRows = function () {
  const L = this.labels, own = "사람이 적은 정답", ref = "프로그램이 본 것 (참고)";
  const rows = [
    { label: "집음·놓음", group: own, who: null, segs: [], marks: L.takes.map(t => ({ ms: t.time_sec * 1000, tone: t.kind,
      tip: (t.kind === "take" ? "집음" : "놓음") + ` ${t.count}개` + (t.slot ? ` · ${t.slot} 칸` : "") })) },
    { label: "경보 대상", group: own, who: null, segs: [], marks: L.culprits.map(c => ({ ms: c.time_sec * 1000, tone: "bad", tip: "경보가 나야 하는 사람" })) },
    { label: "같은 사람", group: own, who: null, segs: [], marks: L.links.map(k => ({ ms: k.time_sec * 1000, tone: "plain", tip: "두 화면의 같은 사람" })) },
    { label: "결제", group: own, who: null, segs: [], marks: this.pays.filter(p => p.time_sec != null).map(p => ({ ms: p.time_sec * 1000,
      tone: p.result === "paid" ? "ok" : "warn", tip: `${PAY_NAME[p.result]} ${p.items}개` })) },
  ];
  const take = this.s.lenses.take.rows, verdict = this.s.lenses.verdict.rows;
  for (const name of ["손 접촉", "선반 가림·비교", "칸 그림 변화", "행동 판정"]) {
    const r = take.find(x => x.label === name);
    if (r) rows.push({ ...r, group: ref });
  }
  const pay = verdict.find(x => x.label === "결제");
  if (pay) rows.push({ ...pay, label: "결제로 본 때", group: ref });
  return rows;
};

/* -- 화면 누르기 */
Player.prototype.startPick = function (kind) {
  this.pause();
  this.pick = { kind, step: kind === "link" ? "detail" : "overview", ms: this.t };
  this.showPick();
};
Player.prototype.showPick = function () {
  const p = this.pick;
  const text = !p ? "" : p.kind === "link"
    ? (p.step === "detail" ? "① 여기 선반 카메라에서 사람을 누르세요 (Esc 취소)" : "② 여기 전체 카메라에서 같은 사람을 누르세요 (Esc 취소)")
    : p.kind === "culprit" ? "경보가 나야 하는 사람을 누르세요 (Esc 취소)"
      : `${p.kind === "take" ? "집은" : "놓은"} 사람을 누르세요 (Esc 취소)`;
  for (const [msg, step] of [[this.msgA, "overview"], [this.msgB, "detail"]]) {
    const on = Boolean(p) && p.step === step;
    msg.hidden = !on; msg.textContent = on ? text : "";
    msg.parentElement.classList.toggle("picking", on);
  }
  this.draw();
};
Player.prototype.clickVideo = function (cam, e) {
  const p = this.pick;
  if (this.lens !== "label" || !p || (p.step === "overview") !== (cam === "angle1")) return;
  const r = e.currentTarget.getBoundingClientRect();
  if (!(r.width > 0 && r.height > 0)) return;
  const point = [round4(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))), round4(Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)))];
  const time_sec = Math.round(p.ms / 10) / 100;
  if (p.kind === "link" && p.step === "detail") { p.detail = point; p.step = "overview"; return this.showPick(); }
  if (p.kind === "link") this.labels.links.push({ time_sec, detail: p.detail, overview: point, note: "" });
  else if (p.kind === "culprit") this.labels.culprits.push({ time_sec, who: point, note: "" });
  else this.labels.takes.push({ time_sec, kind: p.kind, count: 1, slot: "", who: point, note: "" });
  this.pick = null;
  this.showPick();
  this.labelsChanged();
};
Player.prototype.labelKey = function (e) {
  const k = e.key.toLowerCase();
  if (e.key === "Escape" && this.pick) { this.pick = null; this.showPick(); }
  else if (k === "t") this.startPick("take");
  else if (k === "r") this.startPick("put");
  else if (k === "a") this.startPick("culprit");
  else if (k === "l") this.startPick("link");
  else if (k === "p") this.addPayment();
};
Player.prototype.addPayment = function () {
  this.pays = this.pays.filter(p => p.result !== "none");
  this.pays.push({ time_sec: Math.round(this.t / 10) / 100, items: 1, result: "paid", checked_by: this.labels.checked_by, note: "" });
  this.paysDirty = true;
  this.labelsChanged();
};

/* -- 바뀌면: 정렬, 다시 그리기, 저장 예약 */
Player.prototype.labelsChanged = function () {
  const L = this.labels;
  for (const key of ["takes", "culprits", "links"]) L[key].sort((a, b) => a.time_sec - b.time_sec);
  this.pays.sort((a, b) => (a.time_sec == null ? 1e9 : a.time_sec) - (b.time_sec == null ? 1e9 : b.time_sec));
  this.saveState = "저장 중…"; this.saveError = "";
  if (this.lens === "label") { this.drawTimeline({ rows: this.labelRows() }); this.drawLabelPanel(); }
  this.drawLensBar();
  this.draw();
  clearTimeout(this.saveTimer);
  this.saveTimer = setTimeout(() => this.saveLabels(), 400);
};
Player.prototype.saveLabels = async function () {
  const body = { pair: this.s.pair, labels: this.labels, payments: this.paysDirty ? this.pays : null };
  try {
    const res = await postJSON("/api/wb/label", body);
    this.s = res.summary; this.data.summary = res.summary; this.data.labels = res.labels; this.data.payment_labels = res.payment_labels;
    this.paysDirty = false;
    this.saveState = "저장됨"; this.saveError = "";
    app.index = null;   // 갤러리의 점수도 달라졌다
    this.drawLensBar();
  } catch (e) {
    this.saveState = ""; this.saveError = e.message;
  }
  if (this.lens === "label") this.drawLabelState();
};
Player.prototype.drawLabelState = function () {
  const el = document.getElementById("lab-state");
  if (!el) return;
  el.className = "lab-state" + (this.saveError ? " err" : "");
  if (this.saveError) el.replaceChildren(`저장하지 못함: ${this.saveError} `, h("button", { class: "jump", onclick: () => this.saveLabels() }, "다시 저장"));
  else el.replaceChildren(this.saveState);
};

/* -- 오른쪽: 정답 목록 */
Player.prototype.drawLabelPanel = function () {
  const L = this.labels, slots = (this.data.zones.slots || []).map(s => s.name);
  const at = t => h("button", { class: "jump", "data-tip": "이 장면으로", onclick: () => this.seek(t * 1000) }, `▶ ${t.toFixed(2)}`);
  const del = (list, i, pays) => h("button", { class: "del", "aria-label": "지우기", "data-tip": "지우기",
    onclick: () => { list.splice(i, 1); if (pays) this.paysDirty = true; this.labelsChanged(); } }, "×");
  const add = (text, key, fn) => h("button", { class: "btn", onclick: fn }, text, " ", h("kbd", {}, key));
  const none = text => h("div", { class: "muted small" }, text);
  const nameBox = h("input", { class: "lab-name", placeholder: "적는 사람", value: L.checked_by, "aria-label": "적는 사람",
    onchange: () => { L.checked_by = nameBox.value.trim(); try { localStorage.setItem(NAME_KEY, L.checked_by); } catch (e) { /* 무시 */ } if (this.labelCount() || L.takes_done) this.labelsChanged(); } });
  const done = h("input", { type: "checkbox", onchange: () => { L.takes_done = done.checked; this.labelsChanged(); } });
  done.checked = L.takes_done;

  this.facts.replaceChildren(
    h("div", { class: "lab-head" }, h("h2", {}, "정답 적기"), nameBox),
    h("div", { id: "lab-state", class: "lab-state" }),
    h("p", { class: "muted small" }, "영상을 그 순간에 멈추고 버튼을 누른 뒤 화면에서 사람을 누릅니다. 상자는 참고용이고, 정답은 누른 점으로 저장됩니다."),

    h("div", { class: "lab-sec" },
      h("h3", {}, h("span", {}, "집음·놓음"), add("집음", "T", () => this.startPick("take")), add("놓음", "R", () => this.startPick("put"))),
      L.takes.length ? L.takes.map((t, i) => {
        const kind = h("select", { "aria-label": "종류", onchange: () => { t.kind = kind.value; this.labelsChanged(); } },
          [["take", "집음"], ["put", "놓음"]].map(([v, n]) => h("option", { value: v, selected: v === t.kind }, n)));
        const count = h("input", { type: "number", min: 1, max: 20, value: String(t.count), "aria-label": "개수",
          onchange: () => { t.count = Math.min(20, Math.max(1, Math.round(Number(count.value) || 1))); this.labelsChanged(); } });
        const slot = h("select", { "aria-label": "칸", onchange: () => { t.slot = slot.value; this.labelsChanged(); } },
          [h("option", { value: "", selected: !t.slot }, "칸 모름"), ...slots.map(s => h("option", { value: s, selected: s === t.slot }, `${s} 칸`))]);
        return h("div", { class: "lab-row" }, at(t.time_sec), kind, count, "개", slot,
          h("span", { class: "who" }, this.seenAs("angle1", t.who, t.time_sec * 1000)), del(L.takes, i));
      }) : none("아직 없습니다."),
      h("label", { class: "lab-done" }, done, h("span", {}, "이 영상의 집음·놓음을 빠짐없이 적었음",
        h("div", { class: "muted small" }, "체크해야 개수를 채점합니다. 아무도 안 집은 영상이면 비워 둔 채 체크합니다.")))),

    h("div", { class: "lab-sec" },
      h("h3", {}, h("span", {}, "경보가 나야 하는 사람"), add("지정", "A", () => this.startPick("culprit"))),
      L.culprits.length ? L.culprits.map((c, i) => h("div", { class: "lab-row" }, at(c.time_sec),
        h("span", { class: "who" }, this.seenAs("angle1", c.who, c.time_sec * 1000)), del(L.culprits, i)))
        : none("절도 영상에서 물건을 들고 나간 사람을 한 번 누릅니다.")),

    h("div", { class: "lab-sec" },
      h("h3", {}, h("span", {}, "두 화면 같은 사람"), add("짝", "L", () => this.startPick("link"))),
      L.links.length ? L.links.map((k, i) => h("div", { class: "lab-row" }, at(k.time_sec),
        h("span", { class: "who" }, `${this.seenAs("angle0", k.detail, k.time_sec * 1000)} ⇄ ${this.seenAs("angle1", k.overview, k.time_sec * 1000).replace("프로그램: ", "")}`),
        del(L.links, i))) : none("두 화면에 같은 사람이 함께 보이는 순간에 선반 카메라, 전체 카메라 순서로 누릅니다. 선반 앞에 있을 때 몇 번이면 됩니다.")),

    h("div", { class: "lab-sec" },
      h("h3", {}, h("span", {}, "결제"), add("지금 결제", "P", () => this.addPayment())),
      this.pays.length ? this.pays.map((p, i) => {
        const items = h("input", { type: "number", min: 0, max: 50, value: String(p.items), "aria-label": "품목 수",
          onchange: () => { p.items = Math.min(50, Math.max(0, Math.round(Number(items.value) || 0))); this.paysDirty = true; this.labelsChanged(); } });
        const result = h("select", { "aria-label": "결과", onchange: () => { p.result = result.value; this.paysDirty = true; this.labelsChanged(); } },
          Object.entries(PAY_NAME).filter(([v]) => v !== "none" || p.result === "none").map(([v, n]) => h("option", { value: v, selected: v === p.result }, n)));
        return h("div", { class: "lab-row" }, p.time_sec == null ? h("span", { class: "jump" }, "—") : at(p.time_sec),
          p.result === "none" ? null : [items, "개"], result,
          h("span", { class: "who" }, p.checked_by ? `적은 사람 ${p.checked_by}` : ""), del(this.pays, i, true));
      }) : none("결제 정답이 없습니다."),
      h("div", { class: "muted small" }, "결제 정답은 labels/store01_dual/payments.csv 에 저장됩니다 (예전 결제 라벨 도구와 같은 파일).")),
    this.syncSection());
  this.drawLabelState();
};

/* -- 두 영상의 시차: 같은 순간(손이 칸에 닿음, 문 열림)이 두 화면에 함께 보이게 맞춘다 */
Player.prototype.syncSection = function () {
  const sync = this.data.sync || { method: "none", note: "" };
  const how = { manual: "사람이 맞춘 값", trajectory: "자동 추정 — 확인 필요", "trajectory-fallback": "추정이 약해 다른 영상들의 중앙값 — 확인 필요", none: "시차 없음" }[sync.method] || sync.method;
  const input = h("input", { type: "number", step: "1", value: String(Math.round(this.offset)), "aria-label": "시차 (ms)", style: "width:76px",
    onchange: () => this.setOffset(Number(input.value) || 0) });
  const nudge = (text, d, tip) => h("button", { class: "btn", "data-tip": tip, onclick: () => this.setOffset(this.offset + d) }, text);
  const changed = Math.round(this.offset) !== Math.round(this.savedOffset);
  return h("div", { class: "lab-sec", id: "lab-sync" },
    h("h3", {}, h("span", {}, "두 영상의 시차"), changed ? h("button", { class: "btn primary", onclick: () => this.saveOffset() }, "시차 저장") : null),
    h("div", { class: "lab-row" },
      nudge("◀◀", 100, "선반 카메라를 0.1초 더 이른 장면으로"), nudge("◀", 33, "선반 카메라를 한 장면 더 이른 장면으로"),
      input, "ms",
      nudge("▶", -33, "선반 카메라를 한 장면 더 늦은 장면으로"), nudge("▶▶", -100, "선반 카메라를 0.1초 더 늦은 장면으로")),
    h("div", { class: "muted small" }, changed ? `저장된 값 ${Math.round(this.savedOffset)}ms 에서 바꾸는 중 — 두 화면을 보며 맞춘 뒤 저장하세요. ` : `${how}. `,
      "같은 순간(손이 칸에 닿음, 문 열림)이 두 화면에 함께 보이면 맞은 것입니다. 저장하면 이 영상을 다시 판정합니다."),
    this.syncError ? h("div", { class: "lab-state err" }, this.syncError) : null);
};
Player.prototype.setOffset = function (ms) {
  this.pause();
  this.offset = Math.max(-30000, Math.min(30000, Math.round(ms)));
  this.syncB(true);
  this.draw();
  const old = document.getElementById("lab-sync");
  if (old) old.replaceWith(this.syncSection());
};
Player.prototype.saveOffset = async function () {
  this.syncError = "";
  try {
    await postJSON("/api/wb/sync", { pair: this.s.pair, offset_ms: this.offset, checked_by: this.labels.checked_by });
    app.index = null;
    route();   // 시차가 바뀌면 손 접촉·선반 방문의 시각이 모두 달라진다 — 이 영상을 다시 불러온다
  } catch (e) {
    this.syncError = "저장하지 못함: " + e.message;
    const old = document.getElementById("lab-sync");
    if (old) old.replaceWith(this.syncSection());
  }
};

/* -- 영상 위에 정답 표시 (그 시각 앞뒤 0.5초) */
Player.prototype.drawTruth = function () {
  const lens = this.lens, L = this.labels;
  if (lens === "track") return;
  const near = t => Math.abs(t * 1000 - this.t) <= 500;
  const ring = (p, point, color, text) => {
    const x = point[0] * p.w, y = point[1] * p.h, { ctx, u } = p;
    ctx.save();
    ctx.beginPath(); ctx.arc(x, y, 14 * u, 0, Math.PI * 2);
    ctx.lineWidth = 3 * u; ctx.strokeStyle = "#fff"; ctx.stroke();
    ctx.lineWidth = 1.5 * u; ctx.strokeStyle = color; ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, 3 * u, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
    ctx.restore();
    if (text) this.label(p, text, x + 18 * u, y - 9 * u, color, false);
  };
  const scale = (canvas, tr) => {
    const ctx = canvas.getContext("2d");
    ctx.setTransform(canvas.width / tr.w, 0, 0, canvas.height / tr.h, 0, 0);
    ctx.textBaseline = "top";
    const u = tr.w / (canvas.clientWidth || 1);
    ctx.font = `600 ${12 * u}px system-ui, "Malgun Gothic", sans-serif`;
    return { ctx, u, w: tr.w, h: tr.h };
  };
  const pA = scale(this.cA, this.data.tracks.angle1), pB = scale(this.cB, this.data.tracks.angle0);
  if (lens === "label" || lens === "take") {
    for (const t of L.takes) if (near(t.time_sec)) ring(pA, t.who, t.kind === "take" ? "#c2255c" : "#1971c2", `정답: ${t.kind === "take" ? "집음" : "놓음"} ${t.count}개`);
  }
  if (lens === "label" || lens === "verdict") {
    for (const c of L.culprits) if (near(c.time_sec)) ring(pA, c.who, "#b42318", "정답: 경보 대상");
  }
  if (lens === "label" || lens === "link") {
    for (const k of L.links) if (near(k.time_sec)) { ring(pB, k.detail, "#2f5bd8", "정답: 같은 사람"); ring(pA, k.overview, "#2f5bd8", "정답: 같은 사람"); }
  }
  if (this.pick && this.pick.detail) ring(pB, this.pick.detail, "#2f5bd8", "① 이 사람");
};
