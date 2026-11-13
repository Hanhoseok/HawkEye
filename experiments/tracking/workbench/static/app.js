"use strict";
/* HawkEye 분석 도구 — 갤러리(#/<회차>)와 한 쌍 분석(#/<회차>/pair/<짝>/<렌즈>), 수치 조정(#/<회차>/tune).
   촬영 회차(2차·3차)마다 영상·기록·정답이 따로라 주소 맨 앞에 회차가 온다. 회차 없는 예전 주소는 마지막에 본 회차로 연다.
   데이터는 workbench/data.py 가 만든 outputs/workbench/ (목록 index.json, 짝별 pairs/<짝>.json — 3차는 shoot3/ 아래).
   짝별 파일의 summary.lenses[렌즈] 에 요약(facts)·타임라인 줄(rows)·기록(log)이 들어 있어, 여기서는 그리기만 한다. */

const LENSES = [
  { key: "track", name: "사람 추적", ask: "사람을 놓치지 않고 같은 번호로 따라가나" },
  { key: "link", name: "두 화면 같은 사람", ask: "두 카메라에서 같은 사람을 같은 사람으로 잇나" },
  { key: "take", name: "물건 집기", ask: "누가 무엇을 집었는지 맞게 세나" },
  { key: "verdict", name: "결제·판정", ask: "결제와 나감을 보고 맞게 판정하나" },
];
const LEGEND = {
  label: [["take", "집음"], ["put", "놓음"], ["bad", "경보 대상"], ["plain", "같은 사람"], ["ok", "결제"], ["quiet", "프로그램이 본 것은 참고용"]],
  track: [["who", "색 상자 = 손님 (번호마다 다른 색)"], ["quiet", "회색 = 손님이 아닌 것으로 거른 신원"], ["plain", "선반 카메라 사람"]],
  link: [["who", "두 화면에서 같은 색 = 같은 사람으로 이음"], ["ghost", "가는 줄 = 보였지만 짝 없음"], ["warn", "손님이 아닌 신원과 이음"]],
  take: [["take", "사라짐·집음"], ["put", "생김·놓음"], ["quiet", "변화 없음"], ["who", "색 = 그 손님 것으로 봄"], ["warn", "주인 못 정함"]],
  verdict: [["ok", "통과·결제"], ["warn", "확인 필요·출구 경고·취소"], ["bad", "경보"]],
};
const PALETTE = ["#4c6ef5", "#f76707", "#0ca678", "#d6336c", "#7048e8", "#f08c00", "#1098ad", "#74b816"];
const GRAY = "#a8a79f";
const ZONE_NAME = { shelf: "선반 앞", counter: "계산대", door: "출구", outside: "문 밖", aisle: "통로" };
const ZONE_COLOR = { SHELF: "#f08c00", CHECKOUT: "#0ca678", EXIT: "#e03131", OUTSIDE: "#868e96", OTHER: "#4c6ef5" };
const LEVEL_NAME = { HIGH_RISK: "경보", REVIEW: "확인 필요", CLEAR: "통과" };
const OUTCOME_TONE = { hit: "ok", pass: "ok", miss: "bad", false_alarm: "bad", no_verdict: "warn" };

const view = document.getElementById("view");
const crumbs = document.getElementById("crumbs");
const context = document.getElementById("context");
const tipBox = document.getElementById("tip");
const shootsNav = document.getElementById("shoots");
const app = { index: null, shoot: null, filter: "all", batch: null, player: null, poll: null };
const SHOOT_KEY = "wb.shoot";
const at = path => `#/${app.shoot}${path || ""}`;                       // 지금 촬영 회차 안의 주소
const api = path => `/api/wb/${path}?shoot=${app.shoot}`;               // 지금 촬영 회차의 요청

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.style.cssText = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid);
  return el;
}
const sec = ms => (ms / 1000).toFixed(1) + "초";
const getJSON = async url => {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url} → ${r.status}`);
  return r.json();
};

/* ------------------------------------------------------------------ 말풍선 */
document.addEventListener("mousemove", e => {
  const el = e.target.closest ? e.target.closest("[data-tip]") : null;
  if (!el) { tipBox.hidden = true; return; }
  tipBox.textContent = el.dataset.tip;
  tipBox.hidden = false;
  const w = tipBox.offsetWidth, x = Math.min(e.clientX + 12, window.innerWidth - w - 8);
  tipBox.style.left = x + "px";
  tipBox.style.top = (e.clientY + 16) + "px";
});

/* ------------------------------------------------------------------ 길 찾기 */
async function route() {
  clearInterval(app.poll);
  if (app.player) { app.player.destroy(); app.player = null; }
  tipBox.hidden = true;
  const [path, qs] = location.hash.replace(/^#\/?/, "").split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  const query = Object.fromEntries(new URLSearchParams(qs || ""));
  let shoot = /^\d+$/.test(parts[0] || "") ? Number(parts.shift()) : null;
  if (shoot == null) { try { shoot = Number(localStorage.getItem(SHOOT_KEY)) || null; } catch (e) { /* 서버가 고른다 */ } }
  if (shoot !== app.shoot) {   // 다른 촬영으로 — 고른 것들을 처음으로
    app.shoot = shoot; app.index = null; app.batch = null; app.filter = "all";
    add.open = false; add.el = null; add.f = null;
  }
  if (app.shoot == null) {     // 처음 연 것: 서버가 기록이 있는 가장 최근 촬영을 고른다
    try { await loadIndex(); } catch (e) { return showError(e); }
  }
  const tab = parts[0] === "tune" ? "tune" : "gallery";
  for (const a of document.querySelectorAll(".tabs a")) {
    a.setAttribute("aria-current", String(a.dataset.tab === tab));
    a.setAttribute("href", at(a.dataset.tab === "tune" ? "/tune" : ""));
  }
  if (parts[0] === "pair" && parts[1]) showPair(parts[1], parts[2], query.draft);
  else if (parts[0] === "tune") showTune(query);
  else showGallery();
}
window.addEventListener("hashchange", route);

async function loadIndex() {
  const index = await getJSON("/api/wb/index" + (app.shoot ? `?shoot=${app.shoot}` : ""));
  app.index = index; app.shoot = index.shoot;
  try { localStorage.setItem(SHOOT_KEY, String(index.shoot)); } catch (e) { /* 기억 못 해도 된다 */ }
  context.textContent = index.cards && index.cards.length
    ? `기록 ${index.records} · 결제: ${index.payment} · 만든 때 ${index.built}` : "";
  const shoots = (index.shoots || []).filter(s => s.n || s.shoot === index.shoot);
  shootsNav.replaceChildren(...(shoots.length > 1 ? shoots.map(s => h("a", {
    href: `#/${s.shoot}`, "aria-current": String(s.shoot === index.shoot), "data-tip": `${s.name} 영상 ${s.n}쌍`,
  }, s.name, h("span", { class: "n" }, String(s.n)))) : []));
  return index;
}

/* ------------------------------------------------------------------ 갤러리 */
const FILTERS = [
  ["all", "전체", () => true],
  ["wrong", "틀린 것", c => !c.correct],
  ["silent", "판정 없음", c => c.no_verdict],
  ["labeled", "정답 적은 것", c => Boolean(c.labeled)],
  null,
  ["normal", "정상", c => c.set === "normal"],
  ["theft", "절도", c => c.set === "theft"],
  ["complex", "복합", c => c.set === "complex"],
];

async function showGallery() {
  crumbs.replaceChildren();
  view.replaceChildren(h("div", { class: "empty" }, "불러오는 중"));
  let index;
  try { index = await loadIndex(); } catch (e) { return showError(e); }
  drawGallery(index);
  galleryPoll(index);
}

// 뒤에서 일이 도는 동안(보기 데이터 만들기, 추가한 영상 처리)만 주기적으로 다시 읽는다
function galleryPoll(index) {
  clearInterval(app.poll);
  const busy = i => i.build.running || (i.pending || []).some(p => ["queued", "running", "finishing"].includes(p.state));
  if (!busy(index)) return;
  app.poll = setInterval(async () => {
    const fresh = await loadIndex().catch(() => null);
    if (!fresh) return;
    if (!busy(fresh)) clearInterval(app.poll);
    drawGallery(fresh);
  }, 1500);
}

function drawBuilding(build) {
  const el = document.getElementById("building");
  if (!el) return;
  const pct = build.total ? Math.round(100 * build.done / build.total) : 0;
  el.replaceChildren(
    `기록을 읽어 보기 데이터를 만드는 중 — ${build.done}/${build.total}` + (build.current ? ` (${build.current})` : ""),
    h("div", { class: "bar" }, h("i", { style: `width:${pct}%` })));
}

function drawGallery(index) {
  const all = index.cards || [], batches = index.batches || [], pending = index.pending || [];
  const batch = batches.find(x => x.key === app.batch) || batches[0] || null, home = `shoot${index.shoot}`;
  const cards = batch && batches.length > 1 ? all.filter(c => (c.batch || home) === batch.key) : all;
  const t = batch && batches.length > 1 ? batch.totals : index.totals;
  const kids = [];
  if (index.build.running) kids.push(h("div", { class: "building", id: "building" }));
  if (index.build.errors && index.build.errors.length) kids.push(h("div", { class: "errors" }, "만들지 못한 짝:\n" + index.build.errors.join("\n")));
  if (!all.length && !pending.length && !add.open) {
    kids.push(h("div", { class: "empty" }, index.build.running ? "처음 여는 중이라 잠시 걸립니다 (짝마다 몇 초)." :
      (index.can_add ? "아직 영상이 없습니다. 아래에서 두 각도 영상을 추가하세요."
        : `${index.shoot_name}의 기록이 없습니다. 먼저 python scripts/dual_eval.py --shoot ${index.shoot} run 으로 영상을 처리해 주세요.`)));
  }
  if (batches.length > 1) {
    kids.push(h("div", { class: "filters batches" }, batches.map(x => h("button", {
      class: "chip", "aria-pressed": String(x.key === batch.key), onclick: () => { app.batch = x.key; drawGallery(index); },
    }, x.name, h("span", { class: "n" }, String(x.totals.n))))));
  }
  if (cards.length) {
    const theft = t.by_set.theft, normal = t.by_set.normal, complex = t.by_set.complex;
    kids.push(h("div", { class: "stats" },
      stat("맞은 영상", t.correct, t.n, `판정이 실제로 나와서 맞은 것은 ${t.correct_decided}`),
      stat("절도를 잡음", theft.correct, theft.n, `정상 ${normal.correct}/${normal.n} · 복합 ${complex.correct}/${complex.n}`),
      stat("판정이 안 나온 영상", t.no_verdict, t.n, "영상이 끝날 때 손님이 아직 화면에 있음"),
      stat("손님 수 맞음", t.people_ok, t.people_n, "정답표의 손님 수와 비교"),
      labelStat(t.labeled, t.n)));
    kids.push(h("div", { class: "filters" }, FILTERS.map(f => f === null ? h("span", { class: "gap" }) :
      h("button", { class: "chip", "aria-pressed": String(app.filter === f[0]), onclick: () => { app.filter = f[0]; drawGallery(index); } },
        f[1], h("span", { class: "n" }, String(cards.filter(f[2]).length))))));
  }
  if (add.open && index.can_add) kids.push(addForm(index));
  const test = FILTERS.find(f => f && f[0] === app.filter)[2];
  const shown = cards.filter(test);
  kids.push(h("div", { class: "grid" }, index.can_add ? addCard(index) : null, pending.map(x => pendingCard(x)), shown.map(c => card(c, index))));
  if (cards.length && !shown.length) kids.push(h("div", { class: "empty" }, "여기에 해당하는 영상이 없습니다."));
  if (!index.can_add && all.length) {
    kids.push(h("p", { class: "gallery-note" }, `${index.shoot_name}은 영상마다 카메라가 조금씩 움직여 영상별로 구역을 옮겨 씁니다. ` +
      `그래서 여기서는 새 영상을 추가하지 못합니다 — 영상을 넣으려면 카메라 맞추기(scripts/sync_floor.py)와 scripts/dual_eval.py --shoot ${index.shoot} run 을 먼저 합니다. ` +
      "위 점수는 이 컴퓨터에서 만든 기록으로 다시 판정한 것입니다."));
  }
  view.replaceChildren(...kids);
  if (index.build.running) drawBuilding(index.build);
}

function labelStat(L, total) {
  if (!L || !L.n) {
    return h("div", { class: "stat" }, h("div", { class: "k" }, "사람이 적은 정답"),
      h("div", { class: "v" }, "0", h("small", {}, ` / ${total}`)),
      h("div", { class: "s" }, "영상을 열어 '정답 적기'로 적으면 집은 개수·두 화면 짝이 채점됩니다"));
  }
  const bits = [];
  if (L.take_n) bits.push(`집은 개수 ${L.take_ok}/${L.take_n}`);
  if (L.link_n) bits.push(`두 화면 짝 ${L.link_ok}/${L.link_n}`);
  if (L.culprit_n) bits.push(`경보 대상 ${L.culprit_ok}/${L.culprit_n}`);
  return h("div", { class: "stat" }, h("div", { class: "k" }, "사람이 적은 정답"),
    h("div", { class: "v" }, String(L.n), h("small", {}, ` / ${total} 영상`)),
    h("div", { class: "s" }, bits.join(" · ") || "채점할 만큼 적지 않음"));
}

function stat(k, a, b, s) {
  return h("div", { class: "stat" }, h("div", { class: "k" }, k),
    h("div", { class: "v" }, String(a), h("small", {}, ` / ${b}`)), h("div", { class: "s" }, s));
}

function card(c, index) {
  const tone = OUTCOME_TONE[c.outcome];
  const meta = [`손님 ${c.people}명`];
  if (c.link_pct != null) meta.push(`짝 ${c.link_pct}%`);
  meta.push(`물건 ${c.taken}개`);
  return h("a", { class: "card", href: at(`/pair/${encodeURIComponent(c.pair)}`) },
    h("div", { class: "thumb", style: c.thumb ? `background-image:url("${index.data}${c.thumb}")` : "" },
      h("span", { class: `pill ${tone}` }, c.headline)),
    h("div", { class: "body" },
      h("div", { class: "name" }, c.pair, h("span", { class: "tag" }, c.set_name),
        c.batch === "added" && c.sync_method !== "manual"
          ? h("span", { class: "pill warn", "data-tip": "두 영상의 시차가 자동 추정값입니다 — 영상을 열어 '정답 적기'에서 확인하세요" }, "시차 확인") : null),
      h("div", { class: "why" }, c.reason),
      h("div", { class: "meta" }, meta.join(" · "),
        h("span", { class: "lens-dots", "data-tip": LENSES.map(l => `${l.name}: ${statusWord(c.status[l.key])}`).join("  ·  ") },
          LENSES.map(l => h("span", { class: `dot ${c.status[l.key]}` }))))));
}
const statusWord = s => ({ ok: "맞음", bad: "틀림", warn: "확인할 것 있음", info: "정답 없음 (눈으로 확인)" }[s] || s);

function showError(e) {
  view.replaceChildren(h("div", { class: "errors" }, "불러오지 못했습니다.\n" + (e && e.message ? e.message : String(e))));
}

/* ------------------------------------------------------------------ 한 쌍 분석 */
async function showPair(pair, lens, draft) {
  view.replaceChildren(h("div", { class: "empty" }, "불러오는 중"));
  let index, data, meta = null;
  try {
    index = app.index || await loadIndex();
    const base = draft ? `${index.data}drafts/${encodeURIComponent(draft)}/` : index.data;
    data = await getJSON(`${base}pairs/${encodeURIComponent(pair)}.json`);
    if (draft) meta = await getJSON(`${base}meta.json`);
  } catch (e) {
    crumbs.replaceChildren("›", h("b", {}, pair));
    return showError(e);
  }
  const s = data.summary;
  if (!LENSES.some(l => l.key === lens) && !(lens === "label" && !draft)) {
    lens = (LENSES.find(l => s.lenses[l.key].status === "bad") || { key: "verdict" }).key;
  }
  crumbs.replaceChildren("›", h("b", {}, pair), meta ? h("span", { class: "pill warn" }, "초안 값") : "");
  app.player = new Player(index, data, lens, meta);
}

class Player {
  constructor(index, data, lens, draft) {
    this.index = index; this.data = data; this.s = data.summary; this.lens = lens; this.draft = draft;
    this.offset = data.offset_ms; this.savedOffset = data.offset_ms; this.duration = this.s.duration_ms; this.t = 0; this.playing = false;
    this.customers = new Set(this.s.customers);
    this.colors = new Map(this.s.customers.map((pid, i) => [pid, PALETTE[i % PALETTE.length]]));
    this.keys = {};
    this.keyMs = {};   // 기록이 있는 장면의 그 카메라 시각 — 3차는 실제 시각 표(tracks.ms), 2차는 번호 ÷ fps
    for (const cam of ["angle1", "angle0"]) {
      const tr = data.tracks[cam];
      this.keys[cam] = Object.keys(tr.frames).map(Number).sort((a, b) => a - b);
      this.keyMs[cam] = this.keys[cam].map(k => tr.ms ? tr.ms[k] : k * 1000 / tr.fps);
    }
    this.itemMs = data.shelf_items.map(r => r.length > 2 ? r[2] : r[0] * 1000 / data.tracks.angle0.fps);
    this.floor = data.floor ? this.floorSetup(data.floor) : null;
    this.showZones = lens === "verdict" || lens === "take";
    this.initLabels();
    this.build();
    this.onKey = e => this.key(e);
    document.addEventListener("keydown", this.onKey);
    this.onResize = () => this.draw();
    window.addEventListener("resize", this.onResize);
    this.raf = requestAnimationFrame(() => this.tick());
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    document.removeEventListener("keydown", this.onKey);
    window.removeEventListener("resize", this.onResize);
    for (const v of [this.vA, this.vB]) { v.pause(); v.removeAttribute("src"); v.load(); }
  }

  color(pid) { return this.colors.get(pid) || GRAY; }
  who(pid) { return this.customers.has(pid) ? `손님 ${pid}` : `손님 아님 #${pid}`; }

  /* -- 화면 뼈대 */
  build() {
    const s = this.s, cards = this.index.cards || [], i = cards.findIndex(c => c.pair === s.pair);
    const go = d => { const c = cards[i + d]; if (c) location.hash = at(`/pair/${encodeURIComponent(c.pair)}`); };
    this.vA = h("video", { src: this.data.video.angle1, preload: "auto", muted: true, playsinline: true });
    this.vB = h("video", { src: this.data.video.angle0, preload: "auto", muted: true, playsinline: true });
    this.cA = h("canvas", { onclick: e => this.clickVideo("angle1", e) });
    this.cB = h("canvas", { onclick: e => this.clickVideo("angle0", e) });
    this.msgA = h("div", { class: "pickmsg", hidden: true }); this.msgB = h("div", { class: "pickmsg", hidden: true });
    this.offB = h("div", { class: "off", hidden: true }, "이 시각에는 선반 카메라 영상이 없습니다");
    for (const [v, name] of [[this.vA, "전체 카메라"], [this.vB, "선반 카메라"]]) {
      v.addEventListener("error", () => v.parentElement.append(h("div", { class: "off" }, `${name} 영상을 열 수 없습니다: ${v.getAttribute("src")}`)));
      v.addEventListener("seeked", () => this.draw());
      v.addEventListener("loadeddata", () => this.draw());
    }
    this.vA.addEventListener("ended", () => this.pause());
    this.playBtn = h("button", { class: "btn play", onclick: () => this.toggle() }, "재생");
    this.timeEl = h("span", { class: "time" });
    this.speed = h("select", { "aria-label": "재생 속도", onchange: () => this.setRate() },
      [["0.25", "0.25배"], ["0.5", "0.5배"], ["1", "1배"]].map(([v, n]) => h("option", { value: v, selected: v === "0.5" }, n)));
    this.zoneBox = h("input", { type: "checkbox", onchange: () => { this.showZones = this.zoneBox.checked; this.draw(); } });
    this.zoneBox.checked = this.showZones;
    this.lensBar = h("div", { class: "lenses", role: "tablist" });
    this.legend = h("div", { class: "legend" });
    this.timeline = h("div", { class: "timeline" });
    this.facts = h("div", { class: "box" });
    this.logBox = h("div", { class: "box" });
    this.cF = h("canvas", {});
    this.floorBox = h("div", { class: "box floormap", hidden: true },
      h("h2", {}, "바닥 지도 — 두 카메라가 본 발의 자리"), this.cF,
      h("div", { class: "keys" }, h("span", {}, "● 전체 카메라가 본 발"), h("span", {}, "◆ 선반 카메라가 본 발"),
        h("span", {}, "점선 = 발목이 안 보여 방향만 앎"), h("span", {}, "■ 바닥 메모지(보정점)")),
      h("div", { class: "cap" }, "위에서 내려다본 그림. 좌우는 전체 카메라 화면과 같고 아래쪽이 전체 카메라 쪽입니다. " +
        "같은 사람이면 ●와 ◆가 겹치거나, 점선 위에 놓입니다. 숫자는 지금 장면에서 둘 사이의 거리."));
    view.replaceChildren(
      h("div", { class: "head" },
        h("h1", {}, s.pair), h("span", { class: "tag" }, s.set_name),
        h("span", { class: `pill ${OUTCOME_TONE[s.outcome]}` }, s.headline),
        h("span", { class: "muted" }, s.reason),
        this.draft ? null : h("div", { class: "nav" },
          h("button", { class: "btn", disabled: i <= 0, onclick: () => go(-1) }, "‹ 이전 영상"),
          h("button", { class: "btn", disabled: i < 0 || i >= cards.length - 1, onclick: () => go(1) }, "다음 영상 ›"))),
      h("p", { class: "scenario" }, s.notes ? `정답표 메모: ${s.notes}` : ""),
      this.draft ? h("div", { class: "draftbar" },
        h("span", {}, "초안 값으로 다시 판정한 결과입니다 — ", h("b", {}, this.draft.changes.join(" · "))),
        h("a", { class: "btn", href: at(`/pair/${encodeURIComponent(s.pair)}/${this.lens}`) }, "지금 값으로 보기"),
        h("a", { class: "btn", href: at("/tune") }, "수치 조정으로")) : "",
      this.lensBar,
      h("div", { class: "work" },
        h("div", {},
          h("div", { class: "videos" },
            h("div", { class: "screen" }, this.vA, this.cA, h("span", { class: "label" }, "전체 카메라"), this.msgA),
            h("div", { class: "screen" }, this.vB, this.cB, h("span", { class: "label" }, "선반 카메라"), this.offB, this.msgB)),
          h("div", { class: "transport" },
            this.playBtn,
            h("button", { class: "btn", "aria-label": "한 장면 뒤로", onclick: () => this.step(-1) }, "‹"),
            h("button", { class: "btn", "aria-label": "한 장면 앞으로", onclick: () => this.step(1) }, "›"),
            this.timeEl, this.speed,
            h("label", {}, this.zoneBox, "구역 표시"),
            h("span", { class: "keys" }, "스페이스 재생 · ←→ 한 장면 · Shift+←→ 1초 · 1~4 질문 바꾸기")),
          this.legend, this.timeline),
        h("div", { class: "side" }, this.floorBox, this.facts, this.logBox)));
    this.setRate();
    this.setLens(this.lens, true);
  }

  setLens(lens, first) {
    this.lens = lens;
    if (!first) {
      this.showZones = lens === "verdict" || lens === "take";
      this.zoneBox.checked = this.showZones;
      history.replaceState(null, "", at(`/pair/${encodeURIComponent(this.s.pair)}/${lens}`) + (this.draft ? `?draft=${this.draft.draft}` : ""));
      const back = view.querySelector(".draftbar a");
      if (back) back.setAttribute("href", at(`/pair/${encodeURIComponent(this.s.pair)}/${lens}`));
    }
    this.floorBox.hidden = !(this.floor && lens === "link");
    if (lens !== "label" && this.pick) { this.pick = null; this.showPick(); }
    this.drawLensBar();
    this.legend.replaceChildren(...LEGEND[lens].map(([tone, text]) => h("span", {},
      h("i", { style: tone === "who" ? `background:${PALETTE[0]}` : "", class: tone === "who" ? "" : `tl-seg ${tone}`, }), text)));
    for (const i of this.legend.querySelectorAll("i.tl-seg")) i.style.position = "static";
    this.logBox.hidden = lens === "label";
    if (lens === "label") {
      this.drawTimeline({ rows: this.labelRows() });
      this.drawLabelPanel();
    } else {
      this.drawTimeline(this.s.lenses[lens]);
      this.drawFacts(this.s.lenses[lens]);
    }
    this.draw();
  }

  drawLensBar() {
    const L = this.s.lenses, lens = this.lens;
    const tabs = LENSES.map((l, n) => h("button", {
      class: "lens", role: "tab", "aria-selected": String(l.key === lens), "data-tip": `${l.ask} — ${statusWord(L[l.key].status)}`,
      onclick: () => this.setLens(l.key),
    }, h("span", { class: `dot ${L[l.key].status}` }), l.name, h("kbd", {}, String(n + 1))));
    if (!this.draft) {
      tabs.push(h("span", { class: "lens-gap" }), h("button", {
        class: "lens write", role: "tab", "aria-selected": String(lens === "label"),
        "data-tip": "영상을 보며 정답을 적습니다 — 적은 것으로 위 질문들이 채점됩니다", onclick: () => this.setLens("label"),
      }, "정답 적기", this.labelCount() ? h("span", { class: "count" }, String(this.labelCount())) : null, h("kbd", {}, "5")));
    }
    this.lensBar.replaceChildren(...tabs);
  }

  /* -- 타임라인 */
  pct(ms) { return Math.max(0, Math.min(100, 100 * ms / this.duration)); }

  drawTimeline(L) {
    const kids = [];
    let group = null;
    for (const row of L.rows) {
      if (row.group !== group) { group = row.group; kids.push(h("div", { class: "tl-group" }, group)); }
      const track = h("div", { class: "tl-track" });
      for (const g of row.segs) {
        const who = g.who != null ? g.who : (g.tone ? null : row.who);
        track.append(h("div", {
          class: "tl-seg" + (g.tone ? ` ${g.tone}` : ""), "data-tip": g.tip || `${row.label} ${sec(g.s)}~${sec(g.e)}`, "data-ms": g.s,
          style: `left:${this.pct(g.s)}%;width:${Math.max(0.3, this.pct(g.e) - this.pct(g.s))}%;` + (who != null ? `background:${this.color(who)}` : ""),
        }));
      }
      for (const m of row.marks) {
        track.append(h("div", { class: `tl-mark ${m.tone || ""}`, "data-tip": `${sec(m.ms)} · ${m.tip}`, "data-ms": m.ms, style: `left:${this.pct(m.ms)}%` }));
      }
      kids.push(h("div", { class: "tl-row" },
        h("div", { class: "tl-label", "data-tip": row.label },
          row.who != null ? h("i", { style: `background:${this.color(row.who)}` }) : null, row.label),
        track));
    }
    if (!L.rows.length) kids.push(h("div", { class: "tl-empty" }, "표시할 것이 없습니다."));
    const ticks = h("div", { class: "tl-ticks" });
    const every = this.duration > 40000 ? 10000 : (this.duration > 16000 ? 5000 : 2000);
    for (let ms = 0; ms <= this.duration; ms += every) ticks.append(h("span", { style: `left:${this.pct(ms)}%` }, `${ms / 1000}초`));
    kids.push(h("div", { class: "tl-ruler" }, h("div"), ticks));
    this.headEl = h("div", { class: "tl-head" });
    kids.push(this.headEl);
    this.timeline.replaceChildren(...kids);
    const seekAt = e => {
      const track = this.timeline.querySelector(".tl-track") || ticks;
      const r = track.getBoundingClientRect();
      this.seek(this.duration * Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)));
    };
    this.timeline.onmousedown = e => {
      const hit = e.target.closest(".tl-mark");
      if (hit) return this.seek(Number(hit.dataset.ms));
      if (!e.target.closest(".tl-track, .tl-ticks")) return;
      seekAt(e);
      const move = ev => seekAt(ev), up = () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
      window.addEventListener("mousemove", move); window.addEventListener("mouseup", up);
    };
    this.placeHead();
  }

  placeHead() {
    const track = this.timeline.querySelector(".tl-track") || this.timeline.querySelector(".tl-ticks");
    if (!track || !this.headEl) return;
    const a = this.timeline.getBoundingClientRect(), b = track.getBoundingClientRect();
    this.headEl.style.left = (b.left - a.left + b.width * this.t / this.duration - 1) + "px";
  }

  /* -- 오른쪽: 요약과 기록 */
  drawFacts(L) {
    const lens = LENSES.find(l => l.key === this.lens);
    this.facts.replaceChildren(h("h2", {}, lens.ask), ...L.facts.map(f => h("div", { class: `fact ${f.level}` },
      h("span", { class: `dot ${f.level}` }),
      h("div", {}, h("span", { class: "t" }, f.text),
        f.ms != null ? [" ", h("button", { class: "jump", onclick: () => this.seek(f.ms - 300) }, `▶ ${sec(f.ms)}`)] : null,
        f.knob ? [" ", h("a", { class: "jump", href: at(`/tune?focus=${f.knob}`) }, "이 기준 바꿔 보기")] : null))));
    this.logItems = L.log.map(r => {
      const el = h("button", { onclick: () => this.seek(r.ms - 300) }, h("span", { class: "ms" }, sec(r.ms)), h("span", { class: r.level }, r.text));
      el.dataset.ms = r.ms;
      return el;
    });
    this.logBox.replaceChildren(h("h2", {}, "시간 순서 기록 — 누르면 그 장면으로"),
      this.logItems.length ? h("div", { class: "log" }, this.logItems) : h("div", { class: "tl-empty" }, "기록이 없습니다."));
    this.nowLog = null;
  }

  /* -- 재생 */
  setRate() { this.vA.playbackRate = this.vB.playbackRate = Number(this.speed.value); }
  toggle() { this.playing ? this.pause() : this.play(); }
  play() {
    if (this.t >= this.duration - 50 || this.vA.ended) this.seek(0);
    this.playing = true; this.playBtn.textContent = "멈춤";
    this.vA.play().catch(() => this.pause());
  }
  pause() {
    this.playing = false; this.playBtn.textContent = "재생";
    this.vA.pause(); this.vB.pause();
  }
  seek(ms) {
    this.t = Math.max(0, Math.min(this.duration, ms));
    const a = this.t / 1000;
    if (Number.isFinite(this.vA.duration)) this.vA.currentTime = Math.min(a, Math.max(0, this.vA.duration - 0.03));
    this.syncB(true);
    this.draw();
  }
  step(dir, ms) {
    this.pause();
    this.seek(this.t + dir * (ms || 1000 / 15));
  }
  syncB(force) {
    const target = (this.t - this.offset) / 1000, dur = this.vB.duration;
    const out = !Number.isFinite(dur) || target < 0 || target > dur;
    this.offB.hidden = !out || !Number.isFinite(dur);
    if (out) { if (!this.vB.paused) this.vB.pause(); return; }
    const diff = Math.abs(this.vB.currentTime - target);
    if (this.playing) {
      if (this.vB.paused) this.vB.play().catch(() => {});
      if (diff > 0.15) this.vB.currentTime = target;
    } else {
      if (!this.vB.paused) this.vB.pause();
      if (force || diff > 0.03) this.vB.currentTime = target;
    }
  }
  tick() {
    if (this.playing) {
      this.t = Math.min(this.duration, this.vA.currentTime * 1000);
      this.syncB(false);
      this.draw();
    }
    this.raf = requestAnimationFrame(() => this.tick());
  }
  key(e) {
    if ((e.target.matches && e.target.matches("input, select, textarea")) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " ") { e.preventDefault(); this.toggle(); }
    else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      this.step(e.key === "ArrowLeft" ? -1 : 1, e.shiftKey ? 1000 : 0);
    } else if ("1234".includes(e.key) && e.key.length === 1) this.setLens(LENSES[Number(e.key) - 1].key);
    else if (e.key === "5" && !this.draft) this.setLens("label");
    else if (this.lens === "label") this.labelKey(e);
  }

  /* -- 그 시각의 기록 찾기 */
  rowsAt(cam, at) {
    const now = at == null ? this.t : at;
    const tr = this.data.tracks[cam], local = cam === "angle0" ? now - this.offset : now;
    const ms = this.keyMs[cam];
    let lo = 0, hi = ms.length - 1;
    if (hi < 0) return [];
    while (lo < hi) { const mid = (lo + hi) >> 1; if (ms[mid] < local) lo = mid + 1; else hi = mid; }
    const i = lo > 0 && Math.abs(ms[lo - 1] - local) <= Math.abs(ms[lo] - local) ? lo - 1 : lo;
    return Math.abs(ms[i] - local) <= tr.stride * 1500 / tr.fps ? tr.frames[this.keys[cam][i]] : [];
  }
  linkOf(detail) {
    let best = null;
    for (const l of this.data.links) if (l.detail === detail && l.start <= this.t && this.t < l.end + 250) best = l;
    return best;
  }
  linkTo(overview) {
    let best = null;
    for (const l of this.data.links) if (l.overview === overview && l.start <= this.t && this.t < l.end + 250) best = l;
    return best;
  }
  itemsOf(pid) {
    let n = 0;
    for (const a of this.data.actions) if (a.pid === pid && a.end <= this.t) n += a.count;
    return n;
  }
  paidOf(pid) {
    let n = 0;
    for (const p of this.data.payments) {
      if (p.pid !== pid || p.ms > this.t) continue;
      if (p.method === "cancel") n = 0;
      else if (p.method !== "unmatched") n += p.items;
    }
    return n;
  }
  verdictOf(pid) {
    let last = null;
    for (const e of this.data.events) if (e.pid === pid && e.level !== "WARNING" && e.ms <= this.t) last = e;
    return last;
  }

  /* -- 그리기 */
  draw() {
    if (!this.cA.isConnected) return;
    this.timeEl.textContent = `${(this.t / 1000).toFixed(2)} / ${(this.duration / 1000).toFixed(1)}초`;
    this.placeHead();
    if (this.logItems && this.logItems.length) {
      let now = null;
      for (const el of this.logItems) if (Number(el.dataset.ms) <= this.t + 350) now = el;
      if (now !== this.nowLog) {
        if (this.nowLog) this.nowLog.classList.remove("now");
        if (now) {
          now.classList.add("now");
          // 기록 칸 안에서만 굴린다 — scrollIntoView 는 페이지까지 굴려서, 기록 칸이 화면 아래에 있으면 영상이 밀려 올라갔다
          const box = now.parentElement, a = now.getBoundingClientRect(), b = box.getBoundingClientRect();
          if (a.top < b.top) box.scrollTop -= b.top - a.top;
          else if (a.bottom > b.bottom) box.scrollTop += a.bottom - b.bottom;
        }
        this.nowLog = now;
      }
    }
    this.drawOverview();
    this.drawDetail();
    this.drawTruth();
    if (this.floor && this.lens === "link") this.drawFloor();
  }

  pen(canvas, track) {
    const css = canvas.clientWidth || 1, dpr = window.devicePixelRatio || 1;
    const w = Math.round(css * dpr), hgt = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== hgt) { canvas.width = w; canvas.height = hgt; }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, w, hgt);
    ctx.setTransform(w / track.w, 0, 0, hgt / track.h, 0, 0);
    const u = track.w / css;   // 화면 1px 이 영상 몇 px 인가 — 굵기·글자 크기를 화면 기준으로 맞춘다
    ctx.lineJoin = "round";
    ctx.textBaseline = "top";
    ctx.font = `600 ${12 * u}px system-ui, "Malgun Gothic", sans-serif`;
    return { ctx, u, w: track.w, h: track.h };
  }
  label(p, text, x, y, color, above) {
    const { ctx, u } = p, pad = 4 * u, hgt = 17 * u, wid = ctx.measureText(text).width + pad * 2;
    let ty = above ? y - hgt : y;
    if (ty < 0) ty = y;
    const tx = Math.max(0, Math.min(p.w - wid, x));
    ctx.fillStyle = color;
    ctx.fillRect(tx, ty, wid, hgt);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, tx + pad, ty + 2.5 * u);
  }
  box(p, r, color, dashed, width) {
    const { ctx, u } = p;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = (width || 2.5) * u;
    if (dashed) ctx.setLineDash([6 * u, 5 * u]);
    ctx.strokeRect(r[3], r[4], r[5] - r[3], r[6] - r[4]);
    ctx.restore();
  }
  poly(p, pts, color, text, fill) {
    const { ctx, u } = p;
    ctx.save();
    ctx.beginPath();
    pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.closePath();
    if (fill) { ctx.globalAlpha = fill; ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1; }
    ctx.strokeStyle = color; ctx.lineWidth = 1.5 * u; ctx.setLineDash([5 * u, 4 * u]); ctx.stroke();
    if (text) {
      const cx = pts.reduce((a, q) => a + q[0], 0) / pts.length, cy = pts.reduce((a, q) => a + q[1], 0) / pts.length;
      ctx.font = `500 ${11 * u}px system-ui, "Malgun Gothic", sans-serif`;
      const wid = ctx.measureText(text).width;
      ctx.globalAlpha = 0.75; ctx.fillStyle = "#000";
      ctx.fillRect(cx - wid / 2 - 3 * u, cy - 8 * u, wid + 6 * u, 16 * u);
      ctx.globalAlpha = 1; ctx.fillStyle = "#fff";
      ctx.fillText(text, cx - wid / 2, cy - 5.5 * u);
    }
    ctx.restore();
  }
  dotAt(p, x, y, color, r) {
    const { ctx, u } = p;
    ctx.beginPath(); ctx.arc(x, y, (r || 4) * u, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    ctx.lineWidth = 1.5 * u; ctx.strokeStyle = "#fff"; ctx.stroke();
  }
  foot(r) {
    const kp = r[9] || [];
    // 판정(src/zones/floor.py foot_pixel)과 같은 조건: 점수 0.3 이상, (0, 0) 아님(자세 모델이 못 찾은 표시), 상자 안(높이 5% 여유)
    const pad = 0.05 * (r[6] - r[4]);
    const pts = [[kp[18], kp[19], kp[20]], [kp[21], kp[22], kp[23]]].filter(q => q[2] >= 30 && (q[0] || q[1])
      && r[3] - pad <= q[0] && q[0] <= r[5] + pad && r[4] - pad <= q[1] && q[1] <= r[6] + pad);
    if (pts.length) return [pts.reduce((a, q) => a + q[0], 0) / pts.length, pts.reduce((a, q) => a + q[1], 0) / pts.length, true];
    return [(r[3] + r[5]) / 2, r[6], false];
  }

  drawOverview() {
    const tr = this.data.tracks.angle1, p = this.pen(this.cA, tr), lens = this.lens, z = this.data.zones;
    if (this.showZones) {
      const names = lens === "take" ? ["shelf"] : (lens === "verdict" ? ["counter", "door", "outside", "shelf"] : z.overview.map(q => q.name));
      for (const q of z.overview) if (names.includes(q.name)) this.poly(p, q.poly, ZONE_COLOR[q.type] || "#868e96", ZONE_NAME[q.name] || q.name, 0.08);
    }
    for (const r of this.rowsAt("angle1")) {
      const rid = r[1], cust = this.customers.has(rid);
      if (!cust) {
        this.box(p, r, GRAY, true, 1.5);
        if (lens === "track") this.label(p, rid > 0 ? `손님 아님 #${rid}` : "미정", r[3], r[4], GRAY, true);
        continue;
      }
      const color = this.color(rid);
      this.box(p, r, color);
      let text = `손님 ${rid}`;
      if (lens === "track") text += ` · 추적 t${r[2]}`;
      else if (lens === "link") { const l = this.linkTo(rid); text += l ? ` ⇄ 선반 화면 ${l.detail}` : " · 짝 없음"; }
      else if (lens === "take") text += ` · 물건 ${this.itemsOf(rid)}`;
      else if (lens === "label") text += "";
      else {
        const v = this.verdictOf(rid);
        text += ` · 물건 ${this.itemsOf(rid)} · 결제 ${this.paidOf(rid)}` + (v ? ` · ${LEVEL_NAME[v.level] || v.level}` : "");
      }
      this.label(p, text, r[3], r[4], color, true);
      if (lens === "verdict" || lens === "track") { const f = this.foot(r); this.dotAt(p, f[0], f[1], color, 4.5); }
      else if (lens === "link" && this.floor) this.footMark(p, r, color);
    }
  }
  /* 두 화면 짝에 쓰인 발 자리 (바닥으로 맺는 3차): 발목이 보이면 그 점, 안 보이면 상자 아래 가운데에서 아래로 화살표 */
  footMark(p, r, color) {
    const f = r[10];
    if (!f) return;
    const [x, y, ankle] = this.foot(r), { ctx, u } = p;
    if (f.length === 4 || !ankle) {
      const bx = (r[3] + r[5]) / 2, by = Math.min(r[6], p.h - 14 * u);
      ctx.save();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 4 * u; ctx.setLineDash([5 * u, 4 * u]);
      ctx.beginPath(); ctx.moveTo(bx, by - 12 * u); ctx.lineTo(bx, by + 12 * u); ctx.stroke();
      ctx.strokeStyle = color; ctx.lineWidth = 2 * u; ctx.stroke();
      ctx.restore();
      this.dotAt(p, bx, by - 12 * u, color, 3.5);
    } else this.dotAt(p, x, y, color, 4.5);
  }

  drawDetail() {
    const tr = this.data.tracks.angle0, p = this.pen(this.cB, tr), lens = this.lens, z = this.data.zones;
    if (lens === "take" && this.showZones) {
      const active = new Set(this.data.takes0.filter(t => t.start <= this.t && this.t <= t.end + 150).map(t => t.zone));
      for (const s of z.slots) this.poly(p, s.poly, active.has(s.name) ? "#ff3b6b" : "#ffd43b", `${s.name} 칸`, active.has(s.name) ? 0.28 : 0);
      const items = this.itemsAt();
      for (const it of items) {
        p.ctx.strokeStyle = "#69db7c"; p.ctx.lineWidth = 1.5 * p.u;
        p.ctx.strokeRect(it[1], it[2], it[3] - it[1], it[4] - it[2]);
      }
    }
    for (const r of this.rowsAt("angle0")) {
      const rid = r[1], l = rid > 0 ? this.linkOf(rid) : null;
      const cust = l && this.customers.has(l.overview), color = cust ? this.color(l.overview) : GRAY;
      this.box(p, r, color, !cust);
      let text = rid > 0 ? `선반 화면 ${rid}` : "미정";
      if (lens === "link") text += l ? ` = ${this.who(l.overview)} · 점수 ${l.score.toFixed(2)}` + (l.gap != null ? ` · ${Math.round(l.gap)}cm` : "") : " · 짝 없음";
      else if (cust) text += ` = 손님 ${l.overview}`;
      this.label(p, text, r[3], r[4], color, true);
      if (lens === "link" && this.floor) this.footMark(p, r, color);
      if (lens === "take") {
        const kp = r[9] || [];
        for (const i of [12, 15]) if (kp[i + 2] >= 30) this.dotAt(p, kp[i], kp[i + 1], "#ff3b6b", 5);
      }
    }
  }
  itemsAt() {
    const local = this.t - this.offset, ms = this.itemMs;
    let best = -1;
    for (let i = 0; i < ms.length; i++) { if (ms[i] <= local) best = i; else break; }
    if (best < 0 || local - ms[best] > 1000) return [];
    return this.data.shelf_items[best][1];
  }

  /* -- 바닥 지도 (3차: 두 화면 짝을 바닥 위 거리로 맺을 때). 줄 끝 r[10] = 발의 바닥 좌표 [x, y] 또는 [x, y, dx, dy](발목이 안 보임) */
  floorSetup(F) {
    // 선반 카메라에 사람이 보인 자리를 중심으로 본다 — 짝을 맺는 곳이 거기다. 지평선 가까이 튄 점은 뺀다 (2~98%)
    const to = q => [q[0] * F.right[0] + q[1] * F.right[1], q[0] * F.down[0] + q[1] * F.down[1]];
    const xs = [], ys = [];
    for (const rows of Object.values(this.data.tracks.angle0.frames)) {
      for (const r of rows) if (r[1] > 0 && r[10]) { const q = to(r[10]); xs.push(q[0]); ys.push(q[1]); }
    }
    for (const m of F.marks || []) { const q = to(m); xs.push(q[0]); ys.push(q[1]); }
    if (xs.length < 4) return null;
    xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
    const cut = (v, f) => v[Math.min(v.length - 1, Math.max(0, Math.round(f * (v.length - 1))))];
    const pad = 50;
    return { F, to, x0: cut(xs, 0.02) - pad, x1: cut(xs, 0.98) + pad, y0: cut(ys, 0.02) - pad, y1: cut(ys, 0.98) + pad };
  }
  /* 두 발 자리 사이의 가장 가까운 두 점과 거리 (src/fusion/cross_view.py floor_distance 와 같은 식) */
  floorGap(a, b) {
    const onRay = (q, o) => {   // 점 q 에서 반직선 o 위 가장 가까운 점
      const t = Math.max(0, (q[0] - o[0]) * o[2] + (q[1] - o[1]) * o[3]);
      return [o[0] + t * o[2], o[1] + t * o[3]];
    };
    const dist = (q, w) => Math.hypot(q[0] - w[0], q[1] - w[1]);
    if (a.length === 2 && b.length === 2) return { d: dist(a, b), p: a, q: b };
    if (a.length === 2) { const q = onRay(a, b); return { d: dist(a, q), p: a, q }; }
    if (b.length === 2) { const p = onRay(b, a); return { d: dist(p, b), p, q: b }; }
    const det = a[2] * -b[3] - a[3] * -b[2];
    if (Math.abs(det) > 1e-9) {
      const t = ((b[0] - a[0]) * -b[3] - (b[1] - a[1]) * -b[2]) / det, s = (a[2] * (b[1] - a[1]) - a[3] * (b[0] - a[0])) / det;
      if (t >= 0 && s >= 0) { const x = [a[0] + t * a[2], a[1] + t * a[3]]; return { d: 0, p: x, q: x }; }
    }
    const q1 = onRay(a, b), p2 = onRay(b, a);
    return dist(a, q1) <= dist(p2, b) ? { d: dist(a, q1), p: a, q: q1 } : { d: dist(p2, b), p: p2, q: b };
  }
  drawFloor() {
    const V = this.floor, F = V.F, canvas = this.cF;
    const css = canvas.clientWidth || 300, dpr = window.devicePixelRatio || 1;
    const scale = css / (V.x1 - V.x0), hgt = Math.max(140, Math.min(360, (V.y1 - V.y0) * scale));
    if (canvas.style.height !== hgt + "px") canvas.style.height = hgt + "px";
    const w = Math.round(css * dpr), hh = Math.round(hgt * dpr);
    if (canvas.width !== w || canvas.height !== hh) { canvas.width = w; canvas.height = hh; }
    const ctx = canvas.getContext("2d"), style = getComputedStyle(document.documentElement);
    const ink = style.getPropertyValue("--text2").trim() || "#555", faint = style.getPropertyValue("--line-strong").trim() || "#bbb";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, css, hgt);
    const oy = (hgt - (V.y1 - V.y0) * scale) / 2;
    const px = q => { const m = V.to(q); return [(m[0] - V.x0) * scale, oy + (m[1] - V.y0) * scale]; };
    ctx.lineJoin = "round"; ctx.textBaseline = "middle";
    ctx.font = '500 11px system-ui, "Malgun Gothic", sans-serif';
    for (const z of F.zones) {   // 바닥 구역
      const pts = z.poly.map(px), color = ZONE_COLOR[z.type] || "#868e96";
      ctx.beginPath(); pts.forEach((q, i) => i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])); ctx.closePath();
      ctx.globalAlpha = z.name === "aisle" ? 0.04 : 0.1; ctx.fillStyle = color; ctx.fill();
      ctx.globalAlpha = 0.55; ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([4, 3]); ctx.stroke(); ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      if (z.name !== "aisle") {
        const cx = Math.max(24, Math.min(css - 24, pts.reduce((a, q) => a + q[0], 0) / pts.length));
        const cy = Math.max(9, Math.min(hgt - 9, pts.reduce((a, q) => a + q[1], 0) / pts.length));
        ctx.globalAlpha = 0.75; ctx.fillStyle = color; ctx.textAlign = "center"; ctx.fillText(ZONE_NAME[z.name] || z.name, cx, cy); ctx.globalAlpha = 1;
      }
    }
    ctx.fillStyle = "#f2c94c"; ctx.strokeStyle = "#8a6d00"; ctx.lineWidth = 1;
    for (const m of F.marks || []) { const q = px(m); ctx.fillRect(q[0] - 3, q[1] - 3, 6, 6); ctx.strokeRect(q[0] - 3, q[1] - 3, 6, 6); }
    const bar = 50 * scale;   // 길이 눈금
    ctx.strokeStyle = ink; ctx.fillStyle = ink; ctx.lineWidth = 1.5; ctx.textAlign = "left";
    ctx.beginPath(); ctx.moveTo(8, hgt - 9); ctx.lineTo(8 + bar, hgt - 9); ctx.moveTo(8, hgt - 13); ctx.lineTo(8, hgt - 5);
    ctx.moveTo(8 + bar, hgt - 13); ctx.lineTo(8 + bar, hgt - 5); ctx.stroke();
    ctx.fillText("50cm", 13 + bar, hgt - 9);

    const ray = (f, color) => {   // 발목이 안 보임: 발이 있을 방향
      if (f.length !== 4) return;
      const a = px(f), b = px([f[0] + f[2] * 400, f[1] + f[3] * 400]);
      ctx.save(); ctx.strokeStyle = color; ctx.globalAlpha = 0.7; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.restore();
    };
    const placed = [];   // 이미 그린 이름표·표식 자리 — 새 이름표는 이것들을 피해 놓는다
    const tag = (text, x, y, color) => {
      const wid = ctx.measureText(text).width + 8, tall = 16;
      const tries = [[10, -20], [10, 6], [-wid - 10, -20], [-wid - 10, 6], [10, -40], [10, 26], [-wid - 10, -40], [-wid - 10, 26],
        [-wid / 2, -44], [-wid / 2, 30], [-wid / 2, -64], [-wid / 2, 50]];
      let spot = null;
      for (const [dx, dy] of tries) {
        const tx = Math.max(2, Math.min(css - wid - 2, x + dx)), ty = Math.max(2, Math.min(hgt - tall - 2, y + dy));
        if (!spot) spot = [tx, ty];
        if (!placed.some(r => tx < r[0] + r[2] && r[0] < tx + wid && ty < r[1] + r[3] && r[1] < ty + tall)) { spot = [tx, ty]; break; }
      }
      placed.push([spot[0], spot[1], wid, tall]);
      ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.globalAlpha = 0.6;   // 이름표가 가리키는 자리
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(Math.max(spot[0], Math.min(spot[0] + wid, x)), spot[1] + tall / 2); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = color; ctx.fillRect(spot[0], spot[1], wid, tall);
      ctx.fillStyle = "#fff"; ctx.textAlign = "left"; ctx.fillText(text, spot[0] + 4, spot[1] + tall / 2 + 0.5);
    };
    const over = this.rowsAt("angle1").filter(r => r[1] > 0 && r[10]);
    const detail = this.rowsAt("angle0").filter(r => r[1] > 0 && r[10]);
    for (const r of over) ray(r[10], this.customers.has(r[1]) ? this.color(r[1]) : GRAY);
    const marks = [], tags = [], paired = new Set();
    for (const r of detail) {
      const l = this.linkOf(r[1]) || this.linkOf(r[0]);
      const partner = l ? over.find(o => o[1] === l.overview || o[0] === l.overview) : null;
      const color = l && this.customers.has(l.overview) ? this.color(l.overview) : GRAY;
      ray(r[10], color);
      const q = px(r[10]);
      marks.push({ q, color, detail: true });
      if (partner) {   // 맺은 짝: 두 발 사이의 가장 가까운 두 점을 잇고, 이름표 하나에 거리까지
        const g = this.floorGap(r[10], partner[10]), a = px(g.p), b = px(g.q);
        ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        paired.add(partner);
        tags.push([`${this.who(l.overview)} ⇄ 선반 화면 ${r[1]} · ${Math.round(g.d)}cm`, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, color]);
      } else tags.push([`선반 화면 ${r[1]}` + (l ? ` = ${this.who(l.overview)}` : " · 짝 없음"), q[0], q[1], color]);
    }
    for (const r of over) {
      const q = px(r[10]), cust = this.customers.has(r[1]), color = cust ? this.color(r[1]) : GRAY;
      marks.push({ q, color, small: !cust });
      if (cust && !paired.has(r)) tags.push([`손님 ${r[1]}`, q[0], q[1], color]);
    }
    for (const m of marks) {
      const [x, y] = m.q, s = 6;
      placed.push([x - 8, y - 8, 16, 16]);
      ctx.beginPath();
      if (m.detail) { ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath(); }
      else ctx.arc(x, y, m.small ? 4 : 6, 0, Math.PI * 2);
      ctx.fillStyle = m.detail ? "#fff" : m.color; ctx.fill();
      ctx.strokeStyle = m.detail ? m.color : "#fff"; ctx.lineWidth = m.detail ? 2.5 : 1.5; ctx.stroke();
    }
    for (const t of tags) tag(...t);
    if (!over.length && !detail.length) {
      ctx.fillStyle = faint; ctx.textAlign = "center"; ctx.fillText("이 장면에는 사람이 없습니다", css / 2, hgt / 2);
    }
  }
}

