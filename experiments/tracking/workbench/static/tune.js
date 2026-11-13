"use strict";
/* 수치 조정 화면 (#/tune). app.js 의 h·getJSON·view·crumbs·OUTCOME_TONE 을 쓴다 (부르는 때에는 이미 있다).
   서버 쪽은 workbench/tune.py — 값 목록(GET /api/wb/tune), 초안 판정(run), 전후 비교(result), 적용(apply). */

const tune = { page: null, values: {}, result: null, error: "", notice: "", busy: "", timer: null, focus: null };
const tuneKey = () => `wb.tune.values.${app.shoot}`;   // 초안 값은 촬영 회차마다 따로 기억한다

// 모든 저장·판정 요청에 지금 촬영 회차를 함께 보낸다
const postJSON = async (url, body) => {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ shoot: app.shoot, ...body }) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `${url} → ${r.status}`);
  return data;
};

function knobShow(k, v) {
  if (k.kind === "bool") return v ? "켬" : "끔";
  if (k.kind === "choice") return (k.choices.find(c => c.value === v) || { name: v }).name;
  return `${Number(v)}${k.unit}`;
}
function tuneChanged() {
  const out = {};
  for (const k of tune.page.knobs) if (k.key in tune.values && tune.values[k.key] !== k.value) out[k.key] = tune.values[k.key];
  return out;
}
function tuneSave() { try { sessionStorage.setItem(tuneKey(), JSON.stringify(tune.values)); } catch (e) { /* 저장 못 해도 화면은 돈다 */ } }

async function showTune(query) {
  crumbs.replaceChildren();
  view.replaceChildren(h("div", { class: "empty" }, "불러오는 중"));
  tune.focus = query.focus || null;
  tune.error = ""; tune.notice = "";
  try {
    tune.page = await getJSON(api("tune"));
    if (!app.index) await loadIndex();
  } catch (e) { return showError(e); }
  tune.result = null; tune.busy = "";
  try { tune.values = JSON.parse(sessionStorage.getItem(tuneKey()) || "{}"); } catch (e) { tune.values = {}; }
  for (const key of Object.keys(tune.values)) if (!tune.page.knobs.some(k => k.key === key)) delete tune.values[key];
  drawTune();
  await tuneRefresh();
  if (tune.page.job.running) tunePoll();
  if (tune.focus) {
    const el = document.getElementById("knob-" + tune.focus);
    if (el) { el.scrollIntoView({ block: "center" }); el.classList.add("focus"); }
  }
}

async function tuneRefresh() {
  const changed = tuneChanged();
  if (!Object.keys(changed).length) { tune.result = null; drawTuneSide(); return; }
  try { tune.result = await postJSON("/api/wb/tune/result", { values: changed }); }
  catch (e) { tune.result = null; tune.error = e.message; }
  drawTuneSide();
}

function tuneSet(key, value) {
  const k = tune.page.knobs.find(x => x.key === key);
  if (value === k.value) delete tune.values[key]; else tune.values[key] = value;
  tune.error = ""; tune.notice = "";
  tuneSave();
  drawKnob(k);
  clearTimeout(tune.timer);
  tune.timer = setTimeout(tuneRefresh, 250);
  drawTuneSide();
}

function drawTune() {
  const p = tune.page, groups = [];
  for (const k of p.knobs) {
    let g = groups.find(x => x.name === k.group);
    if (!g) groups.push(g = { name: k.group, knobs: [] });
    g.knobs.push(k);
  }
  view.replaceChildren(
    h("div", { class: "head" }, h("h1", {}, "수치 조정"), h("span", { class: "tag" }, p.shoot_name)),
    h("p", { class: "scenario" },
      "판정 값을 바꿔 저장된 기록으로 다시 판정해 봅니다. 영상을 다시 돌리지 않아 몇 초면 됩니다. 큰 규칙은 그대로 두고 기준값만 바꿉니다. " +
      `여기서 바꾸는 값은 ${p.shoot_name}에만 적용됩니다 — 카메라 자리가 다른 촬영은 값을 따로 둡니다.`),
    h("div", { class: "tune" },
      h("div", { class: "tune-main" },
        groups.map(g => h("section", { class: "box knobs" }, h("h2", {}, g.name), g.knobs.map(k => h("div", { class: "knob", id: "knob-" + k.key })))),
        h("section", { class: "box" }, h("h2", {}, "여기서 못 바꾸는 값"),
          h("p", { class: "muted small" }, `아래 값은 탐지·추적 단계에서 쓰여 기록에 이미 들어가 있습니다. 바꾸려면 영상을 다시 돌려야 합니다 (${p.pairs.all}쌍에 약 2시간).`),
          h("ul", { class: "plain" }, p.rerun_only.map(t => h("li", {}, t))),
          p.shoot === 2
            ? h("p", { class: "muted small" }, "구역(계산대·출구·선반 앞)은 아직 예전 ",
              h("a", { class: "link", href: "/experiments/tracking/scripts/zone_editor.html" }, "구역 편집 페이지"), "에서 고칩니다.")
            : h("p", { class: "muted small" }, "이 촬영의 구역은 기본 구역 파일(zones/store01_shoot3/)을 영상마다 그 영상의 카메라 자리로 옮겨 씁니다. " +
              "구역 편집 페이지는 2차 촬영 구역만 엽니다."))),
      h("aside", { class: "tune-side", id: "tune-side" })));
  for (const k of p.knobs) drawKnob(k);
  drawTuneSide();
}

function drawKnob(k) {
  const el = document.getElementById("knob-" + k.key);
  if (!el) return;
  const value = k.key in tune.values ? tune.values[k.key] : k.value, changed = value !== k.value;
  let control;
  if (k.kind === "bool") {
    const box = h("input", { type: "checkbox", onchange: () => tuneSet(k.key, box.checked) });
    box.checked = Boolean(value);
    control = h("label", { class: "switch" }, box, value ? "켬" : "끔");
  } else if (k.kind === "choice") {
    control = h("select", { onchange: e => tuneSet(k.key, e.target.value) },
      k.choices.map(c => h("option", { value: c.value, selected: c.value === value }, c.name)));
  } else {
    const input = h("input", { type: "number", min: k.min, max: k.max, step: k.step, value: String(value), "aria-label": k.name,
      onchange: () => {
        let v = Number(input.value);
        if (!Number.isFinite(v)) v = k.value;
        v = Math.min(k.max, Math.max(k.min, Math.round(v / k.step) * k.step));
        tuneSet(k.key, Number(v.toFixed(4)));
      } });
    const range = h("input", { type: "range", min: k.min, max: k.max, step: k.step, value: String(value), "aria-label": k.name + " 막대",
      oninput: () => { input.value = range.value; }, onchange: () => tuneSet(k.key, Number(Number(range.value).toFixed(4))) });
    control = h("div", { class: "num" }, range, input, h("span", { class: "unit" }, k.unit));
  }
  const keep = el.classList.contains("focus");
  el.className = "knob" + (changed ? " changed" : "") + (keep ? " focus" : "");
  el.replaceChildren(
    h("div", { class: "knob-text" },
      h("div", { class: "knob-name" }, k.name),
      h("div", { class: "knob-help" }, k.help)),
    h("div", { class: "knob-ctl" }, control,
      h("div", { class: "knob-now" },
        changed ? [h("b", {}, `${knobShow(k, k.value)} → ${knobShow(k, value)}`), " ",
          h("button", { class: "jump", onclick: () => tuneSet(k.key, k.value) }, "되돌리기")]
          : `지금 ${knobShow(k, k.value)}` + (k.value !== k.factory ? ` (코드 기본 ${knobShow(k, k.factory)})` : ""))));
}

function delta(a, b, goodUp) {
  if (a === b) return h("span", {}, `${a}`);
  const good = goodUp ? b > a : b < a;
  return h("span", { class: good ? "up" : "down" }, `${a} → ${b}`);
}

function compareBox(part, draft, scope) {
  const b = part.before, a = part.after, kids = [];
  const line = (name, x, y, n, goodUp) => h("div", { class: "cmp" }, h("span", {}, name), h("span", {}, delta(x, y, goodUp), n != null ? ` / ${n}` : ""));
  if (scope === "normal") {
    kids.push(line("경보 없이 통과", b.correct, a.correct, part.n, true));
  } else {
    kids.push(line("맞은 영상", b.correct, a.correct, part.n, true));
    kids.push(line("절도를 잡음", b.by_set.theft.correct, a.by_set.theft.correct, a.by_set.theft.n, true));
    kids.push(line("복합 정상", b.by_set.complex.correct, a.by_set.complex.correct, a.by_set.complex.n, true));
    kids.push(line("판정이 나와서 맞음", b.correct_decided, a.correct_decided, part.n, true));
  }
  kids.push(line("판정 없음", b.no_verdict, a.no_verdict, part.n, false));
  kids.push(line("손님 수 맞음", b.people_ok, a.people_ok, a.people_n, true));
  if (!part.rows.length) kids.push(h("p", { class: "muted small" }, "달라진 영상이 없습니다."));
  else {
    kids.push(h("p", { class: "muted small" }, `달라진 영상 ${part.rows.length}편 — 누르면 초안 값으로 본 장면이 열립니다`));
    kids.push(h("div", { class: "changes" }, part.rows.map(r => {
      const bits = [];
      if (r.before.headline !== r.after.headline) bits.push(`${r.before.headline} → ${r.after.headline}`);
      else if (r.diff.includes("verdicts")) bits.push(`판정 ${r.before.verdicts.join(",") || "없음"} → ${r.after.verdicts.join(",") || "없음"}`);
      if (r.diff.includes("people")) bits.push(`손님 ${r.before.people} → ${r.after.people}명`);
      if (r.diff.includes("taken")) bits.push(`물건 ${r.before.taken} → ${r.after.taken}개`);
      if (r.diff.includes("link_pct")) bits.push(`짝 ${r.before.link_pct} → ${r.after.link_pct}%`);
      const lens = r.diff.includes("people") ? "track" : (r.diff.includes("verdicts") || r.diff.includes("outcome") ? "verdict" : (r.diff.includes("taken") ? "take" : "link"));
      return h("a", { class: `change ${r.effect}`, href: at(`/pair/${encodeURIComponent(r.pair)}/${lens}?draft=${draft}`) },
        h("b", {}, r.pair), h("span", { class: "tag" }, r.set_name), h("span", {}, bits.join(" · ")));
    })));
  }
  return kids;
}

function drawTuneSide() {
  const side = document.getElementById("tune-side");
  if (!side || !tune.page) return;
  const p = tune.page, changed = tuneChanged(), n = Object.keys(changed).length, r = tune.result;
  const others = p.pairs.all - p.pairs.normal, running = Boolean(tune.busy);
  const kids = [];
  if (tune.error) kids.push(h("div", { class: "errors" }, tune.error));
  if (tune.notice) kids.push(h("div", { class: "building" }, tune.notice));

  kids.push(h("div", { class: "box" }, h("h2", {}, n ? `바꾼 값 ${n}개` : "바꾼 값 없음"),
    n ? [h("ul", { class: "plain" }, p.knobs.filter(k => k.key in changed).map(k => h("li", {}, `${k.name}: ${knobShow(k, k.value)} → ${knobShow(k, changed[k.key])}`))),
      h("button", { class: "btn", onclick: () => { tune.values = {}; tuneSave(); for (const k of p.knobs) drawKnob(k); tuneRefresh(); } }, "모두 되돌리기")]
      : h("p", { class: "muted small" }, "왼쪽에서 값을 바꾸면 여기에 모입니다.")));

  kids.push(h("div", { class: "box step" }, h("h2", {}, `① 정상 영상 ${p.pairs.normal}편으로 확인`),
    h("p", { class: "muted small" }, "값은 정상 영상만 보며 정합니다. 몇 번이든 해도 됩니다."),
    r && r.normal ? compareBox(r.normal, r.draft, "normal") : null,
    h("button", { class: "btn primary", disabled: !n || running, onclick: () => tuneRun("normal") },
      tune.busy === "normal" ? "판정 중…" : (r && r.normal ? "다시 확인" : "확인하기"))));

  const opened = r && r.opened && r.all;
  kids.push(h("div", { class: "box step" + (opened ? "" : " locked") }, h("h2", {}, `② 절도·복합 ${others}편 채점`),
    opened ? compareBox(r.all, r.draft, "all") : [
      h("p", { class: "muted small" }, "채점용 영상입니다. 이 결과를 보면서 값을 고치면 이 영상들에만 맞는 값이 됩니다. " +
        "정상 영상으로 값을 정한 뒤 여세요. 연 것은 기록에 남습니다."),
      h("p", { class: "small" }, `지금까지 연 횟수 ${p.opens}번`),
      h("button", { class: "btn", disabled: !(r && r.normal) || running, onclick: () => tuneRun("all") },
        tune.busy === "all" ? "판정 중…" : "채점 열기")]));

  kids.push(h("div", { class: "box step" }, h("h2", {}, "③ 적용"),
    h("p", { class: "muted small" }, `적용하면 ${p.judge_file} 에 적히고, 이 촬영의 평가(dual_eval)·갤러리가 이 값을 씁니다. 되돌리려면 값을 원래대로 놓고 다시 적용합니다.` +
      (r && r.normal && !opened ? " 채점을 열지 않고 적용해도 갤러리에서 결과가 보이므로 연 것으로 기록됩니다." : "")),
    h("button", { class: "btn", disabled: !(r && r.normal) || running, onclick: tuneApply }, tune.busy === "apply" ? "적용 중…" : "이 값으로 적용")));

  kids.push(h("div", { class: "box" }, h("h2", {}, "기록"),
    p.log.length ? h("div", { class: "tune-log" }, p.log.map(row => h("div", { class: "logrow" },
      h("div", {}, h("span", { class: `pill ${row.kind === "apply" ? "ok" : "info"}` }, row.kind === "apply" ? "적용" : "채점 열기"), " ",
        h("span", { class: "muted small" }, row.time)),
      h("div", { class: "small" }, row.changes.join(" · ")),
      h("div", { class: "muted small" }, [row.normal ? `정상 ${row.normal.before} → ${row.normal.after}` : "",
        row.all ? `전체 ${row.all.before} → ${row.all.after}` : ""].filter(Boolean).join(" · ")))))
      : h("p", { class: "muted small" }, "아직 채점을 열거나 적용한 적이 없습니다.")));
  side.replaceChildren(...kids);
}

async function tuneRun(scope) {
  tune.error = ""; tune.notice = ""; tune.busy = scope;
  drawTuneSide();
  try { await postJSON("/api/wb/tune/run", { values: tuneChanged(), scope }); }
  catch (e) { tune.busy = ""; tune.error = e.message; return drawTuneSide(); }
  tunePoll();
}

function tunePoll() {
  clearInterval(app.poll);
  app.poll = setInterval(async () => {
    const s = await getJSON(api("status")).catch(() => null);
    if (!s || s.job.running) return;
    clearInterval(app.poll);
    const scope = tune.busy;
    tune.busy = "";
    if (s.job.error) tune.error = "판정하지 못했습니다: " + s.job.error;
    if (scope === "all") tune.page = await getJSON(api("tune")).catch(() => tune.page);
    await tuneRefresh();
  }, 500);
}

async function tuneApply() {
  tune.error = ""; tune.notice = ""; tune.busy = "apply";
  drawTuneSide();
  try {
    await postJSON("/api/wb/tune/apply", { values: tuneChanged() });
    tune.values = {}; tuneSave();
    tune.page = await getJSON(api("tune"));
    app.index = null;
    tune.result = null;
    tune.notice = "적용했습니다. 갤러리를 새 값으로 다시 만드는 중입니다 (몇 초).";
    drawTune();
  } catch (e) { tune.error = e.message; }
  tune.busy = "";
  drawTuneSide();
}
