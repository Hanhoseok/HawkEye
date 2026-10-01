// HawkEye PC 관제 대시보드. 안드로이드 앱과 같은 서버 API·WebSocket 을 쓴다 (docs/design.md §5).
import {
  applyCase, listCases, latestUpdatedAt, alertFor, missedAlert, levelLabel, stateLabel, resolutionLabel, clockTime,
} from "./store.js";
import { playWhep } from "./whep.js";

const $ = (id) => document.getElementById(id);
const cases = new Map();
let filter = "active";
let selected = null;
let draftNote = ""; // 상세가 다시 그려져도(갱신 수신) 쓰던 메모가 지워지지 않게 따로 둔다
let apiKey = safeGet("hawkeye.apiKey") || "";

function safeGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function safeSet(k, v) { try { localStorage.setItem(k, v); } catch { /* 사생활 보호 모드 */ } }

// --- 서버 호출 ---------------------------------------------------------------
async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (apiKey) headers["X-API-Key"] = apiKey;
  const r = await fetch(path, { ...options, headers });
  if (r.status === 401) { setConnection("bad", "API 키가 맞지 않습니다"); throw new Error("401"); }
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r;
}

async function refresh() {
  const since = latestUpdatedAt(cases);
  const r = await api("/api/cases" + (since ? `?updated_since=${encodeURIComponent(since)}` : ""));
  const fresh = [];
  for (const c of await r.json()) {
    const isNew = !cases.has(c.id);
    if (applyCase(cases, c) && isNew) fresh.push(c);
  }
  render();
  return fresh;
}

// --- 실시간 연결 (끊기면 1·2·4…30초 간격으로 다시 붙고, 놓친 사건을 다시 받는다) -------
let backoff = 1000;
let reconnecting = false;

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws${apiKey ? `?key=${encodeURIComponent(apiKey)}` : ""}`);
  // 다시 붙는 중에는 '연결 끊김' 표시를 유지한다 (끊긴 것을 관리자가 알아야 한다).
  if (!reconnecting) setConnection("wait", "연결 중…");
  ws.onopen = async () => {
    backoff = 1000;
    setConnection("ok", "경보 서버 연결됨");
    const announce = reconnecting;
    reconnecting = true;
    try {
      const missed = await refresh();
      if (announce) missed.map(missedAlert).filter(Boolean).forEach(showAlert);
    } catch { /* refresh 실패는 다음 재연결 때 다시 */ }
  };
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type !== "case.updated") return;
    if (applyCase(cases, m.case)) render();
    const a = alertFor(m);
    if (a?.type === "show") showAlert(a);
    if (a?.type === "clear") clearAlert(a.caseId);
    if (selected === m.case.id) openDetail(m.case.id);
  };
  ws.onclose = (e) => {
    setConnection("bad", e.code === 1008 ? "API 키가 맞지 않습니다" : "연결 끊김 — 다시 연결합니다");
    setTimeout(connect, backoff);
    backoff = Math.min(backoff * 2, 30000);
  };
}

function setConnection(kind, text) {
  const el = $("connection");
  el.className = `pill pill-${kind}`;
  el.textContent = text;
}

// --- 경보 알림 띠 ---------------------------------------------------------------
let bannerCase = null;

function showAlert(a) {
  const c = cases.get(a.caseId);
  bannerCase = a.caseId;
  const b = $("banner");
  b.className = "banner" + (c?.level === "REVIEW" ? " review" : c?.level === "WARNING" ? " warn" : "");
  $("bannerTitle").textContent = a.title;
  $("bannerText").textContent = a.text;
  b.hidden = false;
  beep();
}

function clearAlert(caseId) {
  if (bannerCase === caseId) $("banner").hidden = true;
}

let audio;
function beep() {
  try {
    audio = audio || new AudioContext();
    const o = audio.createOscillator(), g = audio.createGain();
    o.frequency.value = 880; g.gain.value = 0.08;
    o.connect(g).connect(audio.destination);
    o.start(); o.stop(audio.currentTime + 0.25);
  } catch { /* 소리 실패는 무시 */ }
}

// --- 목록 · 상세 ----------------------------------------------------------------
const time = clockTime;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
const badge = (level) => `<span class="badge ${esc(level)}">${esc(levelLabel(level))}</span>`;

function render() {
  const rows = listCases(cases, filter);
  const active = listCases(cases, "active").length;
  $("activeCount").textContent = active;
  $("activeCount").hidden = active === 0;
  $("empty").hidden = rows.length > 0;
  $("empty").textContent = filter === "active" ? "확인할 경보가 없습니다" : "경보 기록이 없습니다";
  $("list").innerHTML = rows.map((c) => `
    <li data-id="${c.id}" class="${c.id === selected ? "sel" : ""}">
      ${badge(c.level)}
      <div><div class="who">손님 ${c.person_id} · ${esc(c.camera_id)}</div>
        <div class="sub">감지 ${c.taken} / 결제 ${c.paid} · 미결제 ${c.unpaid} · ${esc(stateLabel(c.state, c.resolution))}</div></div>
      <div class="time">${time(c.updated_at)}</div>
    </li>`).join("");
}

async function openDetail(id) {
  if (id !== selected) draftNote = "";
  selected = id;
  render();
  const el = $("detail");
  el.hidden = false;
  try {
    const d = await (await api(`/api/cases/${id}`)).json();
    const lastShot = [...d.events].reverse().find((e) => e.has_snapshot);
    const items = d.items?.length
      ? `<table class="items"><tr><th>품목</th><th>감지</th><th>결제</th></tr>${d.items.map((i) =>
          `<tr><td>${esc(i.name)}</td><td>${i.taken}</td><td>${i.paid}</td></tr>`).join("")}</table>` : "";
    el.innerHTML = `
      <h2>${badge(d.level)} 손님 ${d.person_id} <span class="state">${esc(stateLabel(d.state, d.resolution))}</span></h2>
      ${lastShot ? `<img id="shot" alt="경보 장면">` : ""}
      <div class="counts"><div><small>감지</small><b>${d.taken}</b></div><div><small>결제</small><b>${d.paid}</b></div>
        <div><small>미결제</small><b class="${d.unpaid > 0 ? "bad" : ""}">${d.unpaid}</b></div></div>
      ${items}
      ${d.reason ? `<div>사유: ${esc(d.reason)}</div>` : ""}
      ${d.identity_check != null ? `<div class="muted">신원 일치도 ${Number(d.identity_check).toFixed(2)} (낮으면 추적 번호가 다른 사람에게 옮겨 붙었을 수 있음)</div>` : ""}
      <textarea id="note" rows="2" placeholder="메모 (선택)"></textarea>
      <div class="actions"><button class="paid" data-r="paid_confirmed">결제 확인 완료</button><button data-r="false_alarm">잘못된 알림</button></div>
      <ul class="history">
        ${d.resolutions.map((r) => `<li>${time(r.at)} ${esc(resolutionLabel(r.resolution))}${r.note ? " — " + esc(r.note) : ""}</li>`).join("")}
        ${d.events.map((e) => `<li>${badge(e.level)} 영상 ${esc(e.timestamp ?? "-")} · 수신 ${time(e.received_at)}</li>`).join("")}
      </ul>`;
    $("note").value = draftNote;
    if (lastShot) {
      const blob = await (await api(`/api/events/${lastShot.id}/snapshot.jpg`)).blob();
      const img = $("shot");
      if (img) img.src = URL.createObjectURL(blob);
    }
  } catch (e) {
    el.innerHTML = `<div class="muted">불러오지 못했습니다 (${esc(e.message)})</div>`;
  }
}

async function resolve(resolution) {
  const note = draftNote.trim();
  draftNote = "";
  if ($("note")) $("note").value = "";
  await api(`/api/cases/${selected}/resolution`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(note ? { resolution, note } : { resolution }),
  });
  // 결과는 WebSocket 으로 돌아와 목록·상세가 갱신된다.
}

// --- 라이브 ---------------------------------------------------------------------
async function startLive() {
  const status = $("videoStatus");
  let cfg;
  try { cfg = await (await api("/api/config")).json(); } catch { setTimeout(startLive, 3000); return; }
  const cam = cfg.streams.find((s) => s.web_url);
  if (!cam) {
    status.hidden = false;
    status.textContent = "브라우저용 영상 주소가 없습니다 (서버 HAWKEYE_WEB_STREAMS 설정)";
    return;
  }
  $("liveName").textContent = `라이브 · ${cam.camera_id}`;
  const retry = (why) => {
    status.hidden = false;
    status.textContent = `${why} — 다시 연결합니다`;
    setTimeout(startLive, 3000);
  };
  try {
    status.hidden = false;
    status.textContent = "영상 연결 중…";
    await playWhep($("video"), `${cam.web_url.replace(/\/$/, "")}/whep`, () => retry("영상이 끊겼습니다"));
    $("video").onplaying = () => { status.hidden = true; };
  } catch (e) {
    retry(e.message);
  }
}

// --- 이벤트 연결 ----------------------------------------------------------------
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
  filter = t.dataset.filter;
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === t));
  render();
}));
$("list").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-id]");
  if (li) openDetail(Number(li.dataset.id));
});
$("detail").addEventListener("click", (e) => {
  const r = e.target.closest("button[data-r]")?.dataset.r;
  if (r) resolve(r).catch((err) => alert(`저장하지 못했습니다: ${err.message}`));
});
$("detail").addEventListener("input", (e) => { if (e.target.id === "note") draftNote = e.target.value; });
$("bannerOpen").addEventListener("click", () => { if (bannerCase) openDetail(bannerCase); $("banner").hidden = true; });
$("bannerClose").addEventListener("click", () => { $("banner").hidden = true; });
$("fullscreen").addEventListener("click", () => document.querySelector(".video-box").requestFullscreen?.());
$("keyButton").addEventListener("click", () => {
  const k = prompt("경보 서버 API 키 (서버에 설정하지 않았으면 비워 두세요)", apiKey);
  if (k === null) return;
  apiKey = k.trim();
  safeSet("hawkeye.apiKey", apiKey);
  location.reload();
});

refresh().catch(() => {});
connect();
startLive();
