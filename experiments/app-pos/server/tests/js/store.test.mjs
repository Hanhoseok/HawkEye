// PC 대시보드 화면 로직 (앱의 CaseStore·NotificationPolicy 와 같은 규칙).
// 실행: node --test tests/js
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  applyCase, listCases, latestUpdatedAt, alertFor, missedAlert, levelLabel, stateLabel,
} from "../../hawkeye_server/dashboard/store.js";

const c = (id, o = {}) => ({
  id, person_id: id, camera_id: "cam1", level: "WARNING", state: "active",
  taken: 1, paid: 0, unpaid: 1, resolution: null, updated_at: "2026-10-01T10:00:00.000000+00:00", ...o,
});

test("새 사건을 추가하고 확인 필요 목록에 보인다", () => {
  const m = new Map();
  assert.equal(applyCase(m, c(1)), true);
  assert.deepEqual(listCases(m, "active").map((x) => x.id), [1]);
});

test("더 최신 갱신만 반영하고 늦게 온 옛 갱신은 무시한다", () => {
  const m = new Map();
  applyCase(m, c(1, { level: "HIGH_RISK", updated_at: "2026-10-01T10:00:05.000000+00:00" }));
  assert.equal(applyCase(m, c(1, { level: "WARNING", updated_at: "2026-10-01T10:00:00.000000+00:00" })), false);
  assert.equal(m.get(1).level, "HIGH_RISK");
});

test("목록은 최신순이고, 전체에서는 경보 없이 통과한 손님을 숨긴다", () => {
  const m = new Map();
  applyCase(m, c(1, { updated_at: "2026-10-01T10:00:00.000000+00:00" }));
  applyCase(m, c(2, { state: "resolved", updated_at: "2026-10-01T10:00:09.000000+00:00" }));
  applyCase(m, c(3, { state: "pass", level: "CLEAR", updated_at: "2026-10-01T10:00:05.000000+00:00" }));
  assert.deepEqual(listCases(m, "all").map((x) => x.id), [2, 1]);
  assert.deepEqual(listCases(m, "active").map((x) => x.id), [1]);
});

test("놓친 사건을 받을 기준 시각은 가장 최신 갱신 시각", () => {
  const m = new Map();
  assert.equal(latestUpdatedAt(m), null);
  applyCase(m, c(1, { updated_at: "2026-10-01T10:00:00.000000+00:00" }));
  applyCase(m, c(2, { updated_at: "2026-10-01T10:00:09.000000+00:00" }));
  assert.equal(latestUpdatedAt(m), "2026-10-01T10:00:09.000000+00:00");
});

test("notify 면 알림을 띄운다", () => {
  assert.deepEqual(alertFor({ notify: true, case: c(7, { level: "HIGH_RISK", taken: 2, paid: 1 }) }),
    { type: "show", caseId: 7, title: "미결제 퇴장", text: "손님 7 · 감지 2 / 결제 1" });
});

test("처리되거나 자동 해제되면 알림을 지운다", () => {
  assert.deepEqual(alertFor({ notify: false, case: c(3, { state: "resolved" }) }), { type: "clear", caseId: 3 });
  assert.deepEqual(alertFor({ notify: false, case: c(3, { state: "auto_cleared" }) }), { type: "clear", caseId: 3 });
  assert.equal(alertFor({ notify: false, case: c(3) }), null);
});

test("연결이 끊긴 동안 생긴 미처리 사건만 다시 알린다", () => {
  assert.equal(missedAlert(c(8, { level: "REVIEW" })).title, "확인 필요");
  assert.equal(missedAlert(c(8, { state: "resolved" })), null);
});

test("단계·상태 문구는 앱과 같다", () => {
  assert.deepEqual(["HIGH_RISK", "REVIEW", "WARNING", "CLEAR"].map(levelLabel), ["위험", "확인 필요", "주의", "정상"]);
  assert.equal(stateLabel("resolved", "false_alarm"), "처리됨 · 잘못된 알림");
  assert.equal(stateLabel("auto_cleared", null), "자동 해제 (결제·복귀)");
});

test("시각은 이 PC 시간대의 시:분:초 (관제 화면용 짧은 형식)", async () => {
  const { clockTime } = await import("../../hawkeye_server/dashboard/store.js");
  const iso = "2026-10-01T11:47:02.123456+00:00";
  const d = new Date(iso);
  const pad = (n) => String(n).padStart(2, "0");
  assert.equal(clockTime(iso), `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`);
  assert.match(clockTime(iso), /^\d{2}:\d{2}:\d{2}$/);
});

test("코덱을 못 트는 브라우저면 원인과 해결을 알려주고 자주 재시도하지 않는다", async () => {
  const { liveError } = await import("../../hawkeye_server/dashboard/store.js");
  // MediaMTX 실제 응답 (2026-10-06 확인): 400 + {"status":"error","error":"codecs not supported by client"}
  const e = liveError(400, '{"status":"error","error":"codecs not supported by client"}');
  assert.match(e.message, /코덱/);
  assert.match(e.message, /H\.264/);
  assert.equal(e.retryMs, 15000);
});

test("카메라 영상이 아직 없으면(404) 중계기 연결을 확인하라고 하고 금방 다시 시도한다", async () => {
  const { liveError } = await import("../../hawkeye_server/dashboard/store.js");
  const e = liveError(404, '{"status":"error","error":"no stream is available on path \'cam1\'"}');
  assert.match(e.message, /카메라 영상이 아직 없습니다/);
  assert.equal(e.retryMs, 3000);
});

test("그 밖의 오류는 중계기가 준 이유를 그대로 보여준다", async () => {
  const { liveError } = await import("../../hawkeye_server/dashboard/store.js");
  assert.match(liveError(500, '{"error":"boom"}').message, /500.*boom/);
  assert.match(liveError(502, "not json").message, /502/);
});
