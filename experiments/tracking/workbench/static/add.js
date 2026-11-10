"use strict";
/* 새 영상 추가 — 갤러리의 첫 카드. 두 각도 영상을 올리면 서버가 뒤에서 처리해 카드로 올린다 (workbench/intake.py).
   같은 매장·같은 카메라 위치에서 찍은 영상만 받는다 — 구역과 바닥 보정을 그대로 쓰기 때문. */

const add = { open: false, el: null, busy: false, set: "normal", nameEdited: false };
const ADD_SETS = [["normal", "정상", "N"], ["theft", "절도", "T"], ["complex", "복합", "C"]];
const WAIT_TEXT = { queued: "처리 대기 중", running: "처리 중", finishing: "마무리 중 (시차 추정·요약)", failed: "처리하지 못함", idle: "아직 처리하지 않음" };

function addCard(index) {
  return h("button", { class: "card add", onclick: () => { add.open = !add.open; drawGallery(app.index || index); } },
    h("span", { class: "plus" }, add.open ? "−" : "+"), h("b", {}, add.open ? "추가 닫기" : "새 영상 추가"),
    h("span", { class: "small" }, "두 각도 영상을 올리면 처리해서 여기에 올립니다"));
}

function waitClock(seconds) {
  return seconds >= 60 ? `${Math.floor(seconds / 60)}분 ${seconds % 60}초` : `${seconds}초`;
}

function pendingCard(p) {
  const active = ["queued", "running", "finishing"].includes(p.state);
  const retry = async () => {
    try { await postJSON("/api/wb/add/retry", { pair: p.pair }); } catch (e) { /* 아래에서 다시 읽으면 상태가 보인다 */ }
    const index = await loadIndex().catch(() => null);
    if (index) { drawGallery(index); galleryPoll(index); }
  };
  return h("div", { class: "card wait" },
    h("div", { class: "thumb" }, WAIT_TEXT[p.state] + (p.state === "running" || p.state === "finishing" ? ` · ${waitClock(p.seconds)}` : "")),
    h("div", { class: "body" },
      h("div", { class: "name" }, p.pair, p.set_name ? h("span", { class: "tag" }, p.set_name) : null,
        h("span", { class: `pill ${p.state === "failed" ? "bad" : (active ? "info" : "warn")}` }, active ? "추가 중" : (p.state === "failed" ? "실패" : "대기"))),
      h("div", { class: "why" }, p.state === "failed" ? p.error
        : p.state === "running" ? "사람 탐지·추적 중입니다. 영상 길이의 8~10배쯤 걸립니다."
          : p.state === "idle" ? "서버가 꺼졌다 켜져 처리가 멈췄습니다." : "곧 갤러리에 올라옵니다."),
      (p.notes || []).map(n => h("div", { class: "muted small" }, n)),
      p.state === "failed" || p.state === "idle" ? h("button", { class: "btn", onclick: retry }, p.state === "failed" ? "다시 처리" : "처리 시작") : null,
      p.state === "failed" && p.log ? h("div", { class: "muted small", "data-tip": p.log }, "기록 끝부분 보기 (올려 두세요)") : null));
}

/* 폼은 한 번 만들어 계속 쓴다 — 갤러리가 다시 그려져도 고른 파일과 적은 글이 남게 */
function addForm(index) {
  if (add.el) { addSync(index); return add.el; }
  const f = add.f = {};
  f.sets = h("div", { class: "seg", role: "radiogroup", "aria-label": "어떤 영상인가" }, ADD_SETS.map(([key, name]) =>
    h("button", { class: "chip", type: "button", role: "radio", "data-set": key, onclick: () => { add.set = key; add.nameEdited = false; f.alarmEdited = false; addSync(app.index); } }, name)));
  f.name = h("input", { type: "text", maxlength: "20", "aria-label": "이름", oninput: () => { add.nameEdited = true; } });
  f.a1 = h("input", { type: "file", accept: "video/*,.mp4,.mov,.m4v,.mkv,.avi", "aria-label": "전체 카메라 영상" });
  f.a0 = h("input", { type: "file", accept: "video/*,.mp4,.mov,.m4v,.mkv,.avi", "aria-label": "선반 카메라 영상" });
  f.people = h("input", { type: "number", min: "1", max: "9", value: "1", "aria-label": "손님 수" });
  f.alarm = h("input", { type: "checkbox", onchange: () => { f.alarmEdited = true; } });
  f.notes = h("input", { type: "text", maxlength: "200", placeholder: "예: 병 1개 집고 계산 없이 나감", "aria-label": "메모" });
  f.msg = h("div", { class: "msg" });
  f.go = h("button", { class: "btn primary", onclick: addSubmit }, "추가하고 처리 시작");
  add.el = h("section", { class: "addbox" },
    h("h2", {}, "새 영상 추가"),
    h("p", { class: "muted small" }, "같은 매장, 같은 카메라 위치에서 찍은 두 각도 영상만 넣을 수 있습니다. 구역과 바닥 보정을 그대로 쓰기 때문입니다. " +
      "처리는 영상 길이의 8~10배쯤 걸립니다 (20초 영상이면 약 3분)."),
    h("div", { class: "addgrid" },
      h("div", { class: "field" }, h("span", {}, "어떤 영상인가요"), f.sets, h("small", {}, "정상 = 계산하고 나감 · 절도 = 계산 없이 들고 나감 · 복합 = 여러 명·재입장 등")),
      h("label", { class: "field" }, h("span", {}, "이름"), f.name, h("small", {}, "세트 글자(N·T·C)와 번호. 그대로 두면 다음 번호가 붙습니다")),
      h("label", { class: "field" }, h("span", {}, "전체 카메라 영상"), f.a1, h("small", {}, "매장 전체가 보이는 높은 카메라")),
      h("label", { class: "field" }, h("span", {}, "선반 카메라 영상"), f.a0, h("small", {}, "선반과 손이 보이는 카메라")),
      h("label", { class: "field" }, h("span", {}, "손님 수"), f.people, h("small", {}, "영상에 나오는 손님 (문밖 행인 제외)")),
      h("label", { class: "field" }, h("span", {}, "경보가 나야 하나요"), h("span", { class: "switch" }, f.alarm, "물건을 들고 계산 없이 나가는 사람이 있음"),
        h("small", {}, "이것과 손님 수가 정답표에 적힙니다")),
      h("label", { class: "field" }, h("span", {}, "메모 (선택)"), f.notes)),
    h("div", { class: "addbar" }, f.go, h("button", { class: "btn", onclick: () => { add.open = false; drawGallery(app.index); } }, "닫기"), f.msg));
  addSync(index);
  return add.el;
}

function addSync(index) {
  const f = add.f;
  if (!f) return;
  for (const b of f.sets.children) b.setAttribute("aria-pressed", String(b.dataset.set === add.set));
  if (!add.nameEdited && index && index.suggest) f.name.value = index.suggest[add.set] || "";
  if (!f.alarmEdited) f.alarm.checked = add.set === "theft";
}

function addMsg(text, err) {
  add.f.msg.textContent = text;
  add.f.msg.className = "msg" + (err ? " err" : "");
}

function addUpload(name, cam, file, onProgress) {
  return new Promise((resolve, reject) => {
    const dot = file.name.lastIndexOf("."), ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
    const x = new XMLHttpRequest();
    x.open("POST", `/api/wb/upload?name=${encodeURIComponent(name)}&cam=${cam}&ext=${encodeURIComponent(ext)}`);
    x.upload.onprogress = e => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    x.onload = () => {
      if (x.status === 200) return resolve();
      let message = `올리지 못했습니다 (${x.status})`;
      try { message = JSON.parse(x.responseText).error || message; } catch (e) { /* 그대로 */ }
      reject(new Error(message));
    };
    x.onerror = () => reject(new Error("올리지 못했습니다 — 서버가 켜져 있는지 확인해 주세요"));
    x.send(file);
  });
}

async function addSubmit() {
  const f = add.f;
  if (add.busy) return;
  const name = f.name.value.trim(), a1 = f.a1.files[0], a0 = f.a0.files[0];
  if (!/^[NTC]\d{1,3}(-[0-9A-Za-z]{1,12})?$/.test(name)) return addMsg("이름은 N·T·C 중 한 글자와 번호입니다 (예: N13, T11, C22-1)", true);
  if (name[0] !== ADD_SETS.find(s => s[0] === add.set)[2]) return addMsg(`이름의 첫 글자가 고른 세트와 다릅니다 (${ADD_SETS.find(s => s[0] === add.set)[1]} = ${ADD_SETS.find(s => s[0] === add.set)[2]})`, true);
  if (!a1 || !a0) return addMsg("두 영상을 모두 골라 주세요", true);
  if (a1.name === a0.name && a1.size === a0.size) return addMsg("두 칸에 같은 파일을 골랐습니다", true);
  add.busy = true; f.go.disabled = true;
  try {
    await addUpload(name, "angle1", a1, p => addMsg(`전체 카메라 영상 올리는 중 ${Math.round(p * 100)}%`));
    await addUpload(name, "angle0", a0, p => addMsg(`선반 카메라 영상 올리는 중 ${Math.round(p * 100)}%`));
    addMsg("영상을 확인하는 중");
    await postJSON("/api/wb/add", { name, people: Number(f.people.value) || 1, alarm: f.alarm.checked, notes: f.notes.value,
      angle1_file: a1.name, angle0_file: a0.name });
    add.open = false; add.el = null; add.f = null; add.nameEdited = false;
    app.batch = "added";
    const index = await loadIndex();
    drawGallery(index);
    galleryPoll(index);
  } catch (e) {
    addMsg(e.message, true);
    f.go.disabled = false;
  }
  add.busy = false;
}
