// PC 대시보드 화면 로직. 안드로이드 앱의 CaseStore·NotificationPolicy 와 같은 규칙이다.
// DOM 을 쓰지 않는 순수 함수만 둔다 (node --test 로 시험).

/** 더 최신(updated_at) 갱신만 반영한다. WebSocket 과 '놓친 사건 다시 받기'가 순서 없이 와도 안전. */
export function applyCase(cases, c) {
  const old = cases.get(c.id);
  if (old && old.updated_at >= c.updated_at) return false;
  cases.set(c.id, c);
  return true;
}

/** filter: "active" = 미처리만, "all" = 경보 없이 통과한 손님(pass)만 빼고 전부. 최신순. */
export function listCases(cases, filter) {
  return [...cases.values()]
    .filter((c) => (filter === "active" ? c.state === "active" : c.state !== "pass"))
    .sort((a, b) => (a.updated_at < b.updated_at ? 1 : a.updated_at > b.updated_at ? -1 : 0));
}

export function latestUpdatedAt(cases) {
  let latest = null;
  for (const c of cases.values()) if (latest === null || c.updated_at > latest) latest = c.updated_at;
  return latest;
}

export function titleFor(level) {
  return { HIGH_RISK: "미결제 퇴장", REVIEW: "확인 필요", WARNING: "출구 접근 · 미결제 의심" }[level] ?? "알림";
}

const summary = (c) => `손님 ${c.person_id} · 감지 ${c.taken} / 결제 ${c.paid}`;

/** 서버 메시지 하나에 대한 화면 알림. 띄울지는 서버의 notify 를 따른다. */
export function alertFor(message) {
  const c = message.case;
  if (message.notify) return { type: "show", caseId: c.id, title: titleFor(c.level), text: summary(c) };
  if (c.state === "resolved" || c.state === "auto_cleared") return { type: "clear", caseId: c.id };
  return null;
}

/** 연결이 끊긴 동안 생겨서 다시 받아온 사건. 아직 처리 안 된 것만 알린다. */
export function missedAlert(c) {
  if (c.state !== "active") return null;
  return { type: "show", caseId: c.id, title: titleFor(c.level), text: `${summary(c)} (연결이 끊긴 동안 발생)` };
}

export function levelLabel(level) {
  return { HIGH_RISK: "위험", REVIEW: "확인 필요", WARNING: "주의", CLEAR: "정상" }[level] ?? level;
}

export function resolutionLabel(r) {
  return { paid_confirmed: "결제 확인 완료", false_alarm: "잘못된 알림" }[r] ?? "-";
}

export function stateLabel(state, resolution) {
  switch (state) {
    case "active": return "미처리";
    case "resolved": return `처리됨 · ${resolutionLabel(resolution)}`;
    case "auto_cleared": return "자동 해제 (결제·복귀)";
    case "pass": return "통과";
    default: return state;
  }
}

/** 서버 시각(ISO8601, UTC)을 이 PC 시간대의 HH:MM:SS 로. */
export function clockTime(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
