package com.hawkeye.admin.domain

import com.hawkeye.admin.data.CaseDetail
import com.hawkeye.admin.data.CaseSummary
import com.hawkeye.admin.data.EventRecord
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.Instant

class AlertTimesTest {
    private fun case(id: Int = 1, camera: String = "cam1", level: String = "HIGH_RISK", state: String = "active",
                     created: String = "2026-11-24T05:32:10Z", updated: String = created) =
        CaseSummary(id = id, runId = "r", cameraId = camera, personId = id, level = level, state = state,
            createdAt = created, updatedAt = updated)

    private fun ev(level: String, ts: String?, received: String) =
        EventRecord(id = 1, level = level, timestamp = ts, receivedAt = received)

    @Test fun lateDeliveryUsesSceneTime() {
        // 장면 14:32:08 (+09:00), 서버는 40초 늦게 받음
        val d = CaseDetail(case(), listOf(ev("HIGH_RISK", "2026-11-24T14:32:08.000+09:00", "2026-11-24T05:32:48Z")), emptyList())
        val a = AlertTimes.anchor(d)!!
        assertEquals(Instant.parse("2026-11-24T05:32:08Z"), a.at)
        assertTrue(a.exact)
    }

    @Test fun escalatedCaseUsesLastAlert() {
        val d = CaseDetail(case(), listOf(
            ev("WARNING", "2026-11-24T05:31:52Z", "2026-11-24T05:31:53Z"),
            ev("HIGH_RISK", "2026-11-24T05:32:08Z", "2026-11-24T05:32:09Z"),
            ev("CLEAR", "2026-11-24T05:33:00Z", "2026-11-24T05:33:01Z"),
        ), emptyList())
        assertEquals(Instant.parse("2026-11-24T05:32:08Z"), AlertTimes.anchor(d)!!.at)
    }

    @Test fun relativeTimestampFallsBackToReceivedAndIsMarked() {
        val d = CaseDetail(case(), listOf(ev("REVIEW", "12.40", "2026-11-24T05:18:41Z")), emptyList())
        val a = AlertTimes.anchor(d)!!
        assertEquals(Instant.parse("2026-11-24T05:18:41Z"), a.at)
        assertFalse(a.exact)
    }

    @Test fun nothingReadable() {
        assertNull(AlertTimes.anchor(CaseDetail(case(), emptyList(), emptyList())))
        assertNull(AlertTimes.anchor(CaseDetail(case(), listOf(ev("HIGH_RISK", null, "")), emptyList())))
    }

    @Test fun candidatesKeepLateDeliveriesAcrossMidnight() {
        val from = Instant.parse("2026-11-23T15:00:00Z")   // 서울 11월 24일 0시
        val cs = listOf(
            case(1),                                                     // 들어감
            case(2, camera = "cam2"),                                    // 다른 카메라
            case(3, level = "CLEAR", state = "pass"),                    // 경보 없던 손님 (통과)
            case(4, created = "2026-11-22T05:00:00Z"),                   // 막대 시작 전에 끝난 사건 (장면은 받은 시각보다 앞섬)
            case(5, created = "2026-11-24T15:20:00Z"),                   // 다음 날 0시 20분에 받음 -> 장면이 이날 23:55 일 수 있어 조회
            case(6, created = "2026-11-23T10:00:00Z", updated = "2026-11-24T01:00:00Z"), // 어제 생겨 오늘 단계 상승
        )
        assertEquals(listOf(1, 5, 6), AlertTimes.candidates(cs, "cam1", from).map { it.id })
    }

    @Test fun autoClearedWarningStillMarked() {
        // 주의 경보 뒤 CLEAR → 서버가 level=CLEAR, state=auto_cleared 로 바꿈. 원래 주의 경보 시각은 막대에 남아야 함
        val c = case(7, level = "CLEAR", state = "auto_cleared", created = "2026-11-24T05:05:00Z", updated = "2026-11-24T05:06:00Z")
        assertEquals(listOf(7), AlertTimes.candidates(listOf(c), "cam1", Instant.parse("2026-11-23T15:00:00Z")).map { it.id })
        val d = CaseDetail(c, listOf(
            ev("WARNING", "2026-11-24T05:04:58Z", "2026-11-24T05:05:00Z"),
            ev("CLEAR", "2026-11-24T05:05:58Z", "2026-11-24T05:06:00Z"),
        ), emptyList())
        assertEquals(Instant.parse("2026-11-24T05:04:58Z"), AlertTimes.anchor(d)!!.at)
    }

    @Test fun clearOnlyCaseIsNotAnAlert() {
        val d = CaseDetail(case(), listOf(ev("CLEAR", "2026-11-24T05:05:58Z", "2026-11-24T05:06:00Z")), emptyList())
        assertNull(AlertTimes.anchor(d))
    }

    @Test fun lateAlertLandsOnItsSceneDay() {
        // 장면 23:55 (서울 11월 24일), 다음 날 0:20 에 처음 받음 -> 11월 24일 막대에 찍혀야 함
        val d = CaseDetail(case(created = "2026-11-24T15:20:00Z"),
            listOf(ev("HIGH_RISK", "2026-11-24T23:55:00+09:00", "2026-11-24T15:20:00Z")), emptyList())
        val a = AlertTimes.anchor(d)!!
        val from = Instant.parse("2026-11-23T15:00:00Z")
        val to = Instant.parse("2026-11-24T15:00:00Z")
        assertEquals(1, DayBar.marks(listOf(a.at), from, to).size)
    }
}
