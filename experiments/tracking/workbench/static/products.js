"use strict";
/* 상품 라벨 찍기 (#/products, #/products/<그림 이름>). app.js 의 h·getJSON·view·crumbs·at 과 tune.js 의 postJSON 을 쓴다.
   서버 쪽은 workbench/products.py — 장면 목록(GET /api/wb/products), 한 장(GET …/frame), 저장(POST …/save), 지우기(POST …/clear).
   3차 촬영 그림만 있다 (어느 촬영 탭에서 열어도 같은 것).

   한 장의 흐름: 안 찍음 → (하나라도 고치면 저절로 저장) 찍는 중 → Enter 로 '다 찍음'. 학습에는 다 찍은 장면만 들어간다.
   안 찍은 장면을 열면 옆 장면의 상자가 초안(점선)으로 깔린다 — 확인하고 고친 뒤 Enter. */

const PROD_COLOR = ["#ff5b5b", "#4dabf7", "#ffd43b", "#69db7c", "#da77f2"];
const PROD_REASON = { base: "기본 장면", take: "꺼냄 근처", scan: "계산 삑 근처" };
const PROD_STATE = { todo: "안 찍음", doing: "찍는 중", done: "끝" };
const PROD_MIN = 4;          // 이보다 작게 끈 것은 상자로 안 친다 (그림 픽셀)
const PROD_ZOOM_MAX = 24;

const prod = {
  list: null, filter: { state: "all", camera: "all", reason: "all" },
  cur: null, boxes: [], isDraft: false, draftFrom: "", done: false, skip: false, sel: -1, cls: 0,
  img: null, views: {}, view: null, drag: null, mouse: null, space: false, undo: [],
  dirty: false, timer: null, saving: null, status: "", error: "", token: 0, cache: new Map(),
  cv: null, side: null, title: null, observer: null,
};

const prodStem = image => image.replace(/\.jpg$/, "");
const prodHref = image => at("/products/" + encodeURIComponent(prodStem(image)));
const prodName = f => `${f.pair} · ${prod.list.cameras[f.camera] || f.camera} · ${f.time_sec.toFixed(1)}초`;
const prodKids = (...kids) => kids.flat().filter(k => k != null && k !== false);   // replaceChildren 은 null 을 글자로 찍는다
const prodClass = id => prod.list.classes.find(c => c.id === id) || { id, korean: `종류 ${id}`, lane: "" };

async function showProducts(stem) {
  crumbs.replaceChildren();
  if (!prod.list) view.replaceChildren(h("div", { class: "empty" }, "불러오는 중"));
  try {
    await prodFlush();
    if (!prod.list || !stem) prod.list = await getJSON("/api/wb/products");
  } catch (e) { return showError(e); }
  if (stem) return prodOpen(stem + ".jpg");
  prod.token++; prod.cur = null; prod.cv = null;
  drawProdList();
}

/* ------------------------------------------------------------------ 목록 */
function prodShown() {
  const f = prod.filter;
  return prod.list.frames.filter(x => (f.state === "all" || x.state === f.state || (f.state === "left" && x.state !== "done"))
    && (f.camera === "all" || x.camera === f.camera) && (f.reason === "all" || x.reason === f.reason));
}
function prodNav() {   // 앞뒤로 넘길 때의 순서 — 카메라 · 까닭 고르기만 따른다 (상태는 찍으면서 바뀌므로)
  const f = prod.filter;
  return prod.list.frames.filter(x => (f.camera === "all" || x.camera === f.camera) && (f.reason === "all" || x.reason === f.reason));
}

function drawProdList() {
  const L = prod.list, all = L.frames, done = all.filter(f => f.state === "done");
  const count = pred => all.filter(pred).length;
  const boxes = done.reduce((a, f) => a + f.n, 0), held = done.reduce((a, f) => a + f.held, 0);
  const chip = (group, key, name, n) => h("button", {
    class: "chip", "aria-pressed": String(prod.filter[group] === key),
    onclick: () => { prod.filter[group] = key; drawProdList(); },
  }, name, n == null ? null : h("span", { class: "n" }, String(n)));
  const shown = prodShown(), next = prodNav().find(f => f.state !== "done") || all.find(f => f.state !== "done");

  const groups = [];
  for (const cam of Object.keys(L.cameras)) {
    const rows = shown.filter(f => f.camera === cam);
    if (!rows.length) continue;
    const pairs = [...new Set(rows.map(f => f.pair))];
    groups.push(h("div", { class: "box prod-group" },
      h("h2", null, `${L.cameras[cam]} — ${rows.filter(f => f.state === "done").length}/${rows.length}장 끝`),
      pairs.map(pair => h("div", { class: "prow" },
        h("span", { class: "pname" }, pair, h("span", { class: "muted" }, " " + rows.find(f => f.pair === pair).split)),
        h("span", { class: "ptiles" }, rows.filter(f => f.pair === pair).map(f => h("a", {
          class: `ptile ${f.state}${f.skip ? " skip" : ""} ${f.reason}`, href: prodHref(f.image),
          "data-tip": `${prodName(f)} · ${PROD_REASON[f.reason]}${f.lane ? " " + f.lane : ""} · ${f.skip ? "학습에서 뺌" : PROD_STATE[f.state]}`
            + (f.state === "todo" ? "" : ` · 상자 ${f.n}개${f.held ? ` (손에 든 것 ${f.held})` : ""}`),
        }, f.time_sec.toFixed(1))))))));
  }

  view.replaceChildren(...prodKids(
    h("div", { class: "stats" },
      stat("다 찍은 장면", String(done.length), String(all.length), `찍는 중 ${count(f => f.state === "doing")}장 · 학습에서 뺀 것 ${count(f => f.skip)}장`),
      prodStat("찍은 상자", String(boxes), `손에 든 것 ${held}개`),
      prodStat("학습 · 검증 · 시험", ["train", "val", "test"].map(s => done.filter(f => f.split === s).length).join(" · "),
        `전체 ${["train", "val", "test"].map(s => count(f => f.split === s)).join(" · ")}장 (영상 단위로 나눔)`)),
    L.have_images < all.length ? h("div", { class: "errors" },
      `그림이 ${all.length}장 중 ${L.have_images}장만 있습니다. 먼저 그림을 뽑으세요:  python scripts/product_frames.py extract`) : null,
    h("div", { class: "filters" },
      chip("state", "all", "전체", all.length), chip("state", "left", "남은 것", count(f => f.state !== "done")),
      chip("state", "doing", "찍는 중", count(f => f.state === "doing")), chip("state", "done", "끝", done.length),
      h("span", { class: "gap" }),
      chip("camera", "all", "두 카메라"), Object.entries(L.cameras).map(([k, name]) => chip("camera", k, name, count(f => f.camera === k))),
      h("span", { class: "gap" }),
      chip("reason", "all", "모든 장면"), Object.entries(PROD_REASON).map(([k, name]) => chip("reason", k, name, count(f => f.reason === k))),
      h("span", { class: "gap" }),
      next ? h("a", { class: "btn primary", href: prodHref(next.image) }, "이어서 찍기") : h("span", { class: "pill ok" }, "전부 끝")),
    h("div", { class: "prod-key small muted" },
      "칸 하나가 한 장 (숫자 = 영상 속 시각, 초). ",
      h("i", { class: "ptile todo" }, " "), " 안 찍음 ", h("i", { class: "ptile doing" }, " "), " 찍는 중 ", h("i", { class: "ptile done" }, " "), " 끝 ",
      h("i", { class: "ptile done skip" }, " "), " 학습에서 뺌 · 굵은 테두리 = 꺼냄 근처"),
    groups.length ? groups : h("div", { class: "empty" }, "고른 조건에 맞는 장면이 없습니다"),
    h("p", { class: "gallery-note" }, `라벨은 ${L.labels_dir}/ 에 그림마다 파일 하나로 저장됩니다 (저장소에 올림 — 얼굴 없음). `
      + "학습용으로 내보내기:  python scripts/product_frames.py export")));
}
const prodStat = (k, v, s) => h("div", { class: "stat" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v), h("div", { class: "s" }, s));

/* ------------------------------------------------------------------ 한 장 열기 */
function prodImage(url) {
  if (!prod.cache.has(url)) {
    prod.cache.set(url, new Promise(resolve => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => { prod.cache.delete(url); resolve(null); };
      img.src = url;
    }));
    while (prod.cache.size > 6) prod.cache.delete(prod.cache.keys().next().value);   // 큰 그림이라 몇 장만 쥐고 있는다
  }
  return prod.cache.get(url);
}

async function prodOpen(image) {
  const entry = prod.list.frames.find(f => f.image === image);
  if (!entry) return showError(new Error(`표에 없는 그림: ${image}`));
  const token = ++prod.token;
  Object.assign(prod, { cur: entry, boxes: [], isDraft: false, draftFrom: "", done: false, skip: false, sel: -1, img: null,
    drag: null, mouse: null, undo: [], dirty: false, status: "", error: "" });
  drawProdEditor();
  let info, img;
  try {
    [info, img] = await Promise.all([getJSON("/api/wb/products/frame?image=" + encodeURIComponent(image)), prodImage(prod.list.images + image)]);
  } catch (e) { if (token === prod.token) showError(e); return; }
  if (token !== prod.token) return;
  if (info.saved) Object.assign(prod, { boxes: info.saved.boxes, done: info.saved.done, skip: info.saved.skip });
  else if (info.draft) Object.assign(prod, { boxes: info.draft.boxes, isDraft: true, draftFrom: info.draft.from });
  prod.img = img;
  if (!img) prod.error = "그림이 없습니다 — python scripts/product_frames.py extract";
  prodResize();
  drawProdSide();
  const order = prodNav(), i = order.indexOf(entry);
  if (i >= 0 && order[i + 1]) prodImage(prod.list.images + order[i + 1].image);   // 다음 장을 미리 읽어 둔다
}

function drawProdEditor() {
  const f = prod.cur;
  view.classList.add("wide");
  crumbs.replaceChildren(h("a", { href: at("/products") }, "상품 라벨"), "›", h("b", null, prodName(f)));
  prod.cv = h("canvas", { class: "prod-canvas", tabindex: "0" });
  prod.side = h("aside", { class: "side prod-side" });
  const stage = h("div", { class: "prod-stage" }, prod.cv);
  view.replaceChildren(h("div", { class: "prod" }, stage, prod.side));
  const cv = prod.cv;
  cv.addEventListener("pointerdown", prodDown);
  cv.addEventListener("pointermove", prodMove);
  cv.addEventListener("pointerup", prodUp);
  cv.addEventListener("pointercancel", prodUp);
  cv.addEventListener("pointerleave", () => { prod.mouse = null; prodDraw(); });
  cv.addEventListener("wheel", prodWheel, { passive: false });
  cv.addEventListener("contextmenu", e => e.preventDefault());
  if (prod.observer) prod.observer.disconnect();
  prod.observer = new ResizeObserver(() => { if (prod.cv === cv) prodResize(); });
  prod.observer.observe(stage);
  drawProdSide();
}

/* ------------------------------------------------------------------ 보는 자리 (확대 · 이동) */
function prodFit() {
  const cv = prod.cv, img = prod.img;
  const s = Math.min(cv.clientWidth / img.naturalWidth, cv.clientHeight / img.naturalHeight);
  return { s, ox: (cv.clientWidth - img.naturalWidth * s) / 2, oy: (cv.clientHeight - img.naturalHeight * s) / 2 };
}
function prodResize() {
  const cv = prod.cv;
  if (!cv || !cv.isConnected) return;
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(1, Math.round(cv.clientWidth * dpr));
  cv.height = Math.max(1, Math.round(cv.clientHeight * dpr));
  if (prod.img) prod.view = prod.views[prod.cur.camera] || prodFit();   // 확대해 둔 자리는 같은 카메라의 다음 장에서도 그대로
  prodDraw();
}
function prodAt(e) {   // 캔버스 안에서의 자리 (화면 픽셀). offsetX 는 화면 배율에 따라 어긋날 수 있어 직접 뺀다
  const r = prod.cv.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}
const prodToImg = (x, y) => [(x - prod.view.ox) / prod.view.s, (y - prod.view.oy) / prod.view.s];
function prodZoom(factor, x, y) {
  const v = prod.view, fit = prodFit().s;
  const s = Math.min(PROD_ZOOM_MAX, Math.max(fit * 0.8, v.s * factor));
  prod.view = { s, ox: x - (x - v.ox) * s / v.s, oy: y - (y - v.oy) * s / v.s };
  prod.views[prod.cur.camera] = prod.view;
  prodDraw();
}
function prodWheel(e) {
  if (!prod.img) return;
  e.preventDefault();
  prodZoom(e.deltaY < 0 ? 1.25 : 0.8, ...prodAt(e));
}

/* ------------------------------------------------------------------ 그리기 */
function prodDraw() {
  const cv = prod.cv;
  if (!cv || !cv.isConnected) return;
  const ctx = cv.getContext("2d"), dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!prod.img || !prod.view) {
    ctx.fillStyle = "#cfcfc8"; ctx.font = "13px system-ui"; ctx.textAlign = "center";
    ctx.fillText(prod.error || "그림을 읽는 중", W / 2, H / 2);
    return;
  }
  const v = prod.view;
  ctx.imageSmoothingEnabled = v.s < 2.5;
  ctx.drawImage(prod.img, v.ox, v.oy, prod.img.naturalWidth * v.s, prod.img.naturalHeight * v.s);
  ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.font = "600 11px system-ui";
  const rect = b => [v.ox + b.x1 * v.s, v.oy + b.y1 * v.s, (b.x2 - b.x1) * v.s, (b.y2 - b.y1) * v.s];
  prod.boxes.forEach((b, i) => {
    const [x, y, w, t] = rect(b), color = PROD_COLOR[b.cls % PROD_COLOR.length], sel = i === prod.sel;
    if (b.held) { ctx.fillStyle = color + "33"; ctx.fillRect(x, y, w, t); }
    ctx.setLineDash(prod.isDraft ? [5, 4] : []);
    if (sel) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 4; ctx.strokeRect(x, y, w, t); }
    ctx.strokeStyle = color; ctx.lineWidth = sel ? 2 : 1.5;
    ctx.strokeRect(x, y, w, t);
    ctx.setLineDash([]);
    if (w >= 12 || sel) {
      const text = String(b.cls + 1) + (b.held ? " 손" : ""), tw = ctx.measureText(text).width + 6;
      ctx.fillStyle = color; ctx.fillRect(x - 1, y - 14, tw, 14);
      ctx.fillStyle = "#111"; ctx.fillText(text, x + 2, y - 1);
    }
    if (sel) {
      ctx.fillStyle = "#fff";
      for (const [hx, hy] of [[x, y], [x + w, y], [x, y + t], [x + w, y + t]]) ctx.fillRect(hx - 3, hy - 3, 6, 6);
    }
  });
  const d = prod.drag;
  if (d && d.type === "new") {
    const color = PROD_COLOR[prod.cls % PROD_COLOR.length];
    ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    ctx.strokeRect(...rect({ x1: Math.min(d.ax, d.bx), y1: Math.min(d.ay, d.by), x2: Math.max(d.ax, d.bx), y2: Math.max(d.ay, d.by) }));
  } else if (prod.mouse && !d) {   // 십자선 — 작은 상품의 모서리를 맞추기 쉽게
    ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(prod.mouse[0] + 0.5, 0); ctx.lineTo(prod.mouse[0] + 0.5, H);
    ctx.moveTo(0, prod.mouse[1] + 0.5); ctx.lineTo(W, prod.mouse[1] + 0.5);
    ctx.stroke();
  }
}

/* ------------------------------------------------------------------ 마우스 */
function prodHit(x, y) {
  // 고른 상자의 테두리 근처면 크기 바꾸기, 아니면 그 점을 품은 가장 작은 상자 (손에 든 상품이 선반 상품 위에 겹칠 때)
  const [ix, iy] = prodToImg(x, y), s = prod.view.s;
  const b = prod.boxes[prod.sel];
  if (b) {
    const tx = Math.min(6, (b.x2 - b.x1) * s / 4) / s, ty = Math.min(6, (b.y2 - b.y1) * s / 4) / s, out = 6 / s;
    if (ix >= b.x1 - out && ix <= b.x2 + out && iy >= b.y1 - out && iy <= b.y2 + out) {
      const edges = { l: ix <= b.x1 + tx, r: ix >= b.x2 - tx, t: iy <= b.y1 + ty, b: iy >= b.y2 - ty };
      if (edges.l || edges.r || edges.t || edges.b) return { type: "size", i: prod.sel, edges };
    }
  }
  let best = -1, area = Infinity;
  prod.boxes.forEach((k, i) => {
    const a = (k.x2 - k.x1) * (k.y2 - k.y1);
    if (ix >= k.x1 && ix <= k.x2 && iy >= k.y1 && iy <= k.y2 && a < area) { best = i; area = a; }
  });
  return best >= 0 ? { type: "move", i: best } : null;
}
function prodCursor(hit) {
  if (!hit) return "crosshair";
  if (hit.type === "move") return "move";
  const e = hit.edges;
  if ((e.l && e.t) || (e.r && e.b)) return "nwse-resize";
  if ((e.r && e.t) || (e.l && e.b)) return "nesw-resize";
  return e.l || e.r ? "ew-resize" : "ns-resize";
}
function prodDown(e) {
  if (!prod.img) return;
  prod.cv.focus({ preventScroll: true });
  try { prod.cv.setPointerCapture(e.pointerId); } catch (err) { /* 못 잡아도 끌기는 된다 */ }
  const [x, y] = prodAt(e);
  if (e.button !== 0 || prod.space) {   // 가운데 · 오른쪽 단추, 또는 스페이스를 누른 채: 그림 옮기기
    prod.drag = { type: "pan", x, y, ox: prod.view.ox, oy: prod.view.oy };
    prod.cv.style.cursor = "grabbing";
    return;
  }
  const [ix, iy] = prodToImg(x, y), hit = prodHit(x, y);
  if (hit) {
    prod.sel = hit.i;
    prod.drag = { ...hit, ix, iy, orig: { ...prod.boxes[hit.i] }, moved: false };
  } else {
    prod.sel = -1;
    const cx = Math.min(prod.img.naturalWidth, Math.max(0, ix)), cy = Math.min(prod.img.naturalHeight, Math.max(0, iy));
    prod.drag = { type: "new", ax: cx, ay: cy, bx: cx, by: cy };
  }
  drawProdSide(); prodDraw();
}
function prodMove(e) {
  if (!prod.img) return;
  const [x, y] = prodAt(e), d = prod.drag, W = prod.img.naturalWidth, H = prod.img.naturalHeight;
  prod.mouse = [x, y];
  if (!d) { prod.cv.style.cursor = prod.space ? "grab" : prodCursor(prodHit(x, y)); return prodDraw(); }
  const [ix, iy] = prodToImg(x, y);
  if (d.type === "pan") {
    prod.view = { s: prod.view.s, ox: d.ox + x - d.x, oy: d.oy + y - d.y };
    prod.views[prod.cur.camera] = prod.view;
  } else if (d.type === "new") {
    d.bx = Math.min(W, Math.max(0, ix)); d.by = Math.min(H, Math.max(0, iy));
  } else {
    const b = prod.boxes[d.i], o = d.orig;
    let dx = ix - d.ix, dy = iy - d.iy;
    if (!d.moved && Math.hypot(dx, dy) * prod.view.s < 3) return;   // 누르기만 한 것은 고르기
    if (!d.moved) { prodSnap(); d.moved = true; }
    if (d.type === "move") {
      dx = Math.min(W - o.x2, Math.max(-o.x1, dx)); dy = Math.min(H - o.y2, Math.max(-o.y1, dy));
      Object.assign(b, { x1: o.x1 + dx, x2: o.x2 + dx, y1: o.y1 + dy, y2: o.y2 + dy });
    } else {
      if (d.edges.l) b.x1 = Math.min(W, Math.max(0, o.x1 + dx));
      if (d.edges.r) b.x2 = Math.min(W, Math.max(0, o.x2 + dx));
      if (d.edges.t) b.y1 = Math.min(H, Math.max(0, o.y1 + dy));
      if (d.edges.b) b.y2 = Math.min(H, Math.max(0, o.y2 + dy));
    }
  }
  prodDraw();
}
function prodUp(e) {
  const d = prod.drag;
  prod.drag = null;
  if (!d) return;
  try { prod.cv.releasePointerCapture(e.pointerId); } catch (err) { /* 이미 풀렸다 */ }
  prod.cv.style.cursor = "crosshair";
  if (d.type === "new") {
    const box = { cls: prod.cls, x1: Math.min(d.ax, d.bx), y1: Math.min(d.ay, d.by), x2: Math.max(d.ax, d.bx), y2: Math.max(d.ay, d.by), held: false };
    if (box.x2 - box.x1 >= PROD_MIN && box.y2 - box.y1 >= PROD_MIN) {
      prodSnap();
      prod.boxes.push(box);   // 새 상자는 고른 채로 두지 않는다 — 다음 종류 키를 눌렀을 때 방금 그린 상자가 바뀌면 안 된다
      prodTouched();
    }
  } else if (d.type !== "pan" && d.moved) {
    const b = prod.boxes[d.i];
    Object.assign(b, { x1: Math.min(b.x1, b.x2), x2: Math.max(b.x1, b.x2), y1: Math.min(b.y1, b.y2), y2: Math.max(b.y1, b.y2) });
    if (b.x2 - b.x1 < PROD_MIN || b.y2 - b.y1 < PROD_MIN) Object.assign(b, d.orig);   // 찌그러뜨린 것은 되돌린다
    prodTouched();
  }
  drawProdSide(); prodDraw();
}

/* ------------------------------------------------------------------ 고치기 · 저장 */
function prodSnap() {
  prod.undo.push(JSON.stringify({ boxes: prod.boxes, sel: prod.sel, skip: prod.skip }));
  if (prod.undo.length > 100) prod.undo.shift();
}
function prodTouched() {   // 무엇이든 고쳤다 — 초안은 이제 내 라벨이고, 조금 뒤 저절로 저장한다
  prod.isDraft = false; prod.dirty = true; prod.error = "";
  prod.status = "고침";
  clearTimeout(prod.timer);
  prod.timer = setTimeout(prodFlush, 600);
  drawProdSide(); prodDraw();
}
async function prodFlush(keepalive) {
  clearTimeout(prod.timer); prod.timer = null;
  if (!prod.dirty || !prod.cur || !prod.img) return prod.saving;
  prod.dirty = false;
  const entry = prod.cur;
  const body = JSON.stringify({ image: entry.image, boxes: prod.boxes, width: prod.img.naturalWidth, height: prod.img.naturalHeight,
    skip: prod.skip, done: prod.done });
  const send = async () => {
    const r = await fetch("/api/wb/products/save", { method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: Boolean(keepalive) });
    const out = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(out.error || `저장 → ${r.status}`);
    Object.assign(entry, { state: out.state, n: out.n, held: out.held, skip: out.skip });
    if (prod.cur === entry && !prod.dirty) { prod.status = "저장됨"; prod.error = ""; }
  };
  prod.status = "저장 중";
  prod.saving = (prod.saving || Promise.resolve()).then(send).catch(err => {   // 저장은 한 번에 하나씩, 보낸 순서대로
    if (prod.cur === entry) { prod.error = "저장 못 함: " + err.message; prod.dirty = true; }
  }).then(() => { if (prod.cur === entry) drawProdSide(); });
  return prod.saving;
}
window.addEventListener("pagehide", () => { prodFlush(true); });

function prodSetClass(id) {
  prod.cls = id;
  const b = prod.boxes[prod.sel];
  if (b && b.cls !== id) { prodSnap(); b.cls = id; return prodTouched(); }
  drawProdSide();
}
function prodDelete() {
  if (prod.sel < 0) return;
  prodSnap();
  prod.boxes.splice(prod.sel, 1); prod.sel = -1;
  prodTouched();
}
function prodHeld() {
  const b = prod.boxes[prod.sel];
  if (!b) return;
  prodSnap(); b.held = !b.held;
  prodTouched();
}
function prodNudge(dx, dy, grow) {
  const b = prod.boxes[prod.sel];
  if (!b) return;
  prodSnap();
  const W = prod.img.naturalWidth, H = prod.img.naturalHeight;
  if (grow) {   // Shift: 오른쪽 · 아래 변만 움직여 크기를 바꾼다
    b.x2 = Math.min(W, Math.max(b.x1 + PROD_MIN, b.x2 + dx)); b.y2 = Math.min(H, Math.max(b.y1 + PROD_MIN, b.y2 + dy));
  } else {
    dx = Math.min(W - b.x2, Math.max(-b.x1, dx)); dy = Math.min(H - b.y2, Math.max(-b.y1, dy));
    b.x1 += dx; b.x2 += dx; b.y1 += dy; b.y2 += dy;
  }
  prodTouched();
}
function prodUndo() {
  const last = prod.undo.pop();
  if (!last) return;
  Object.assign(prod, JSON.parse(last));
  prodTouched();
}
function prodClearAll() {
  if (!prod.boxes.length) return;
  prodSnap(); prod.boxes = []; prod.sel = -1;
  prodTouched();
}
function prodSkip() {
  prodSnap(); prod.skip = !prod.skip;
  prodTouched();
}
async function prodGo(step) {
  const order = prodNav(), i = order.indexOf(prod.cur);
  const next = order[(i < 0 ? 0 : i + step)];
  if (next) location.hash = prodHref(next.image);
}
async function prodDone() {   // 다 찍음 — 저장하고 아직 안 끝난 다음 장으로
  if (!prod.img) return;
  prod.done = true; prod.isDraft = false; prod.dirty = true;
  await prodFlush();
  if (prod.error) return drawProdSide();
  const order = prodNav(), i = order.indexOf(prod.cur);
  const next = order.slice(i + 1).find(f => f.state !== "done") || order.find(f => f.state !== "done");
  location.hash = next ? prodHref(next.image) : at("/products");
}
async function prodForget() {
  if (!confirm("이 장면의 라벨을 지우고 '안 찍음'으로 되돌릴까요?")) return;
  clearTimeout(prod.timer); prod.dirty = false;
  const entry = prod.cur;
  try {
    await prod.saving;
    await postJSON("/api/wb/products/clear", { image: entry.image });
  } catch (e) { prod.error = e.message; return drawProdSide(); }
  Object.assign(entry, { state: "todo", n: 0, held: 0, skip: false });
  prodOpen(entry.image);
}

document.addEventListener("keydown", e => {
  if (!prod.cv || !prod.cv.isConnected || !prod.cur) return;
  if (e.target.closest && e.target.closest("input, select, textarea")) return;
  const k = e.key, cmd = e.ctrlKey || e.metaKey;
  if (cmd && k.toLowerCase() === "z") { e.preventDefault(); return prodUndo(); }
  if (cmd || e.altKey) return;
  if (k === " ") { e.preventDefault(); prod.space = true; return; }
  if (/^[1-9]$/.test(k)) {
    const c = prod.list.classes[Number(k) - 1];
    if (c) { e.preventDefault(); prodSetClass(c.id); }
    return;
  }
  const arrows = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
  if (arrows[k]) {
    if (prod.sel < 0) return;
    e.preventDefault();
    return prodNudge(arrows[k][0], arrows[k][1], e.shiftKey);
  }
  const keys = {
    Enter: prodDone, Delete: prodDelete, Backspace: prodDelete, h: prodHeld, x: prodSkip,
    a: () => prodGo(-1), d: () => prodGo(1),
    f: () => { delete prod.views[prod.cur.camera]; if (prod.img) { prod.view = prodFit(); prodDraw(); } },
    Escape: () => { prod.drag = null; prod.sel = -1; drawProdSide(); prodDraw(); },
    Tab: () => { if (prod.boxes.length) { prod.sel = (prod.sel + (e.shiftKey ? -1 : 1) + prod.boxes.length + (prod.sel < 0 && e.shiftKey ? 1 : 0)) % prod.boxes.length; drawProdSide(); prodDraw(); } },
  };
  const run = keys[k] || keys[k.toLowerCase()];
  if (run) { e.preventDefault(); run(); }
});
document.addEventListener("keyup", e => { if (e.key === " ") prod.space = false; });

/* ------------------------------------------------------------------ 옆 칸 */
function drawProdSide() {
  const side = prod.side, f = prod.cur;
  if (!side || !side.isConnected || !f) return;
  const order = prodNav(), i = order.indexOf(f), left = order.filter(x => x.state !== "done").length;
  const sel = prod.boxes[prod.sel];
  const state = prod.isDraft ? ["warn", `초안 ${prod.boxes.length}개 — 확인 필요`]
    : prod.skip ? ["info", "학습에서 뺌"]
      : prod.done ? ["ok", "끝"] : f.state === "todo" && !prod.dirty && prod.status === "" ? ["info", "안 찍음"] : ["warn", "찍는 중"];
  side.replaceChildren(...prodKids(
    h("div", { class: "box" },
      h("div", { class: "lab-head" }, h("h2", null, prodName(f)), h("span", { class: `pill ${state[0]}` }, state[1])),
      h("div", { class: "small muted" }, `${PROD_REASON[f.reason]}${f.lane ? ` (${f.lane}${f.action === "return" ? " 되돌림" : ""})` : ""} · ${f.split} · `
        + (i >= 0 ? `${i + 1}/${order.length}번째 · 남은 것 ${left}장` : "지금 고른 조건 밖")),
      prod.isDraft ? h("div", { class: "small prod-note" }, `점선은 옆 장면(${prodStem(prod.draftFrom)})에서 가져온 초안입니다. 없어진 상품은 지우고 자리를 맞춘 뒤 Enter.`) : null,
      h("div", { class: "prod-bar" },
        h("button", { class: "btn", disabled: i <= 0, onclick: () => prodGo(-1), "data-tip": "앞 장 (A)" }, "◀"),
        h("button", { class: "btn primary", disabled: !prod.img, onclick: prodDone, "data-tip": "이 장을 '끝'으로 저장하고 남은 다음 장으로 (Enter)" },
          prod.boxes.length ? "다 찍음 → 다음" : "상품 안 보임 → 다음"),
        h("button", { class: "btn", disabled: i < 0 || i >= order.length - 1, onclick: () => prodGo(1), "data-tip": "다음 장 (D)" }, "▶")),
      h("div", { class: `lab-state${prod.error ? " err" : ""}` }, prod.error || prod.status || " ")),
    h("div", { class: "box" },
      h("h2", null, "종류 (새 상자 · 고른 상자에 적용)"),
      prod.list.classes.map((c, n) => h("button", {
        class: "prod-class", "aria-pressed": String(prod.cls === c.id), onclick: () => prodSetClass(c.id),
      }, h("i", { style: `background:${PROD_COLOR[c.id % PROD_COLOR.length]}` }), h("kbd", null, String(n + 1)),
      h("span", null, c.korean), h("span", { class: "muted small" }, c.lane),
      h("span", { class: "count" }, String(prod.boxes.filter(b => b.cls === c.id).length))))),
    h("div", { class: "box" },
      h("h2", null, sel ? `고른 상자 — ${prodClass(sel.cls).korean}` : `상자 ${prod.boxes.length}개`),
      sel ? [
        h("div", { class: "small muted" }, `가로 ${Math.round(sel.x2 - sel.x1)} × 세로 ${Math.round(sel.y2 - sel.y1)} 픽셀`),
        h("label", { class: "lab-done" }, h("input", { type: "checkbox", checked: sel.held, onchange: prodHeld }), "손에 든 상품 (H)"),
        h("button", { class: "btn", onclick: prodDelete }, "이 상자 지우기 (Del)"),
      ] : h("div", { class: "small muted" }, "상자를 누르면 고릅니다. 빈 곳을 끌면 새 상자."),
      h("div", { class: "prod-bar" },
        h("button", { class: "btn", disabled: !prod.undo.length, onclick: prodUndo }, "되돌리기"),
        h("button", { class: "btn", disabled: !prod.boxes.length, onclick: prodClearAll }, "전부 지우기"),
        h("button", { class: "btn", "aria-pressed": String(prod.skip), onclick: prodSkip, "data-tip": "흐려서 무엇인지 모를 때 — 학습에 안 쓴다 (X)" },
          prod.skip ? "학습에 다시 넣기" : "학습에서 빼기")),
      f.state !== "todo" ? h("button", { class: "jump", onclick: prodForget }, "라벨을 지우고 안 찍음으로") : null),
    h("div", { class: "box small" },
      h("h2", null, "찍는 기준"),
      h("ul", { class: "plain" },
        h("li", null, "보이는 상품은 전부 찍는다. 안 찍은 상품은 '상품 아님'으로 배운다."),
        h("li", null, "절반 넘게 가린 것은 안 찍는다."),
        h("li", null, "상자는 보이는 부분에 딱 맞게."),
        h("li", null, "손에 든 것은 H 로 표시."),
        h("li", null, "흐려서 모르겠으면 그 장면을 뺀다 (X).")),
      h("h2", null, "단축키"),
      h("ul", { class: "plain" },
        h("li", null, "1 · 2 · 3 종류 / 끌기 새 상자 / Del 지우기"),
        h("li", null, "화살표 1픽셀 옮기기 (Shift: 크기)"),
        h("li", null, "휠 확대 / 오른쪽 단추 끌기 · Space+끌기 이동 / F 전체 보기"),
        h("li", null, "Enter 다 찍음 → 다음 / A · D 앞뒤 / Ctrl+Z 되돌리기")))));
}
