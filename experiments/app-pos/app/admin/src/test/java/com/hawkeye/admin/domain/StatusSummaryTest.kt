package com.hawkeye.admin.domain

import com.hawkeye.admin.data.ApiException
import com.hawkeye.admin.data.CaseSummary
import com.hawkeye.admin.data.Item
import com.hawkeye.admin.data.RecordingSpan
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.IOException
import java.net.SocketTimeoutException
import java.time.Instant
import java.time.ZoneId

class StatusSummaryTest {
    private val seoul = ZoneId.of("Asia/Seoul")
    private val now = Instant.parse("2026-11-24T05:33:00Z") // 서울 14:33
    private val done = SyncState.Done()

    private fun case(
        id: Int, level: String, state: String = "active", person: Int = id, taken: Int = 2, paid: Int = 0,
        updated: String = "2026-11-24T05:32:08Z", created: String = updated, items: List<Item>? = null, camera: String = "cam1",
    ) = CaseSummary(
        id = id, runId = "r", cameraId = camera, personId = person, level = level, state = state,
        taken = taken, paid = paid, unpaid = taken - paid, items = items, updatedAt = updated, createdAt = created,
    )

    @Test fun riskiestActiveCaseWins() {
        val s = StatusSummaries.of(
            listOf(
                case(1, "WARNING", updated = "2026-11-24T05:40:00Z"),
                case(2, "HIGH_RISK", person = 3, items = listOf(Item("과자", 1, 0), Item("음료", 1, 0))),
                case(3, "REVIEW"),
            ),
            offline = false, sync = done, now = now, zone = seoul,
        )
        assertEquals(Tone.DANGER, s.tone)
        assertEquals("손님 3이 결제 없이 나갔어요", s.title)
        assertEquals("과자 1, 음료 1을 결제하지 않았어요. cam1 전체", s.detail)
        assertEquals("14:32", s.meta)
        assertEquals(StatusAction.OPEN_CASE, s.action)
        assertEquals(2, s.caseId)
    }

    @Test fun sameLevelPicksMostRecent() {
        val s = StatusSummaries.of(
            listOf(case(1, "REVIEW", updated = "2026-11-24T05:10:00Z"), case(2, "REVIEW", updated = "2026-11-24T05:20:00Z")),
            offline = false, sync = done, now = now, zone = seoul,
        )
        assertEquals(2, s.caseId)
    }

    @Test fun reviewAndWarningSentences() {
        val r = StatusSummaries.of(listOf(case(5, "REVIEW", taken = 3, paid = 2)), false, done, now, seoul)
        assertEquals(Tone.REVIEW, r.tone)
        assertEquals("손님 5가 3개 중 2개만 결제했어요", r.title)
        val w = StatusSummaries.of(listOf(case(7, "WARNING")), false, done, now, seoul)
        assertEquals(Tone.WARN, w.tone)
        assertEquals("손님 7이 결제 전에 출구로 갔어요", w.title)
    }

    @Test fun handledCasesLeaveCalmCardWithTodayCount() {
        val s = StatusSummaries.of(
            listOf(
                case(1, "HIGH_RISK", state = "resolved"),
                case(2, "REVIEW", state = "auto_cleared"),
                case(3, "CLEAR", state = "pass"),                                    // 통과 손님은 세지 않음
                case(4, "WARNING", state = "resolved", created = "2026-11-23T05:00:00Z"), // 어제
            ),
            offline = false, sync = done, now = now, zone = seoul,
        )
        assertEquals(Tone.CALM, s.tone)
        assertEquals("오늘 받은 경보 2건", s.meta)
        assertEquals(StatusAction.LIVE, s.action)
    }

    @Test fun noCasesAtAll() {
        val s = StatusSummaries.of(emptyList(), false, done, now, seoul)
        assertEquals("새 경보가 오면 여기에 보여요.", s.detail)
    }

    @Test fun offlineComesFirst() {
        val s = StatusSummaries.of(listOf(case(1, "HIGH_RISK")), offline = true, sync = done, now = now, zone = seoul)
        assertEquals(Tone.OFF, s.tone)
        assertEquals(StatusAction.RECONNECT, s.action)
    }

    @Test fun beforeFirstSyncNothingIsClaimed() {
        val s = StatusSummaries.of(emptyList(), offline = false, sync = SyncState.NotYet, now = now, zone = seoul)
        assertEquals("경보 목록을 확인하고 있어요", s.title)
        val s2 = StatusSummaries.of(emptyList(), offline = false, sync = SyncState.Syncing, now = now, zone = seoul)
        assertEquals("확인 중", s2.label)
    }

    @Test fun failedSyncIsNotCalm() {
        val s = StatusSummaries.of(emptyList(), offline = false, sync = SyncState.Failed("HTTP 500"), now = now, zone = seoul)
        assertEquals(Tone.OFF, s.tone)
        assertEquals(StatusAction.RECONNECT, s.action)
        assertTrue(s.detail.contains("HTTP 500"))
    }

    @Test fun knownRiskShowsEvenWhileSyncing() {
        val s = StatusSummaries.of(listOf(case(1, "HIGH_RISK")), offline = false, sync = SyncState.Syncing, now = now, zone = seoul)
        assertEquals(Tone.DANGER, s.tone)
    }

    @Test fun badgeShowsPlusWhenServerListWasCut() {
        assertEquals(null, StatusSummaries.badgeText(0, done))
        assertEquals("3", StatusSummaries.badgeText(3, done))
        assertEquals("1000+", StatusSummaries.badgeText(1000, SyncState.Done(truncated = true)))
        assertEquals("99+", StatusSummaries.badgeText(150, done))
    }

    @Test fun unpaidWithoutItems() {
        assertEquals("미결제 2개예요.", StatusSummaries.unpaidPhrase(case(1, "HIGH_RISK")))
    }

    @Test fun activeCountIgnoresClearAndHandled() {
        val n = StatusSummaries.activeCount(listOf(case(1, "HIGH_RISK"), case(2, "CLEAR"), case(3, "REVIEW", state = "resolved")))
        assertEquals(1, n)
    }

    @Test fun koreanParticleFollowsSpokenNumber() {
        assertEquals("손님 3이", Korean.subject("손님 3"))
        assertEquals("손님 5가", Korean.subject("손님 5"))
        assertEquals("손님 10이", Korean.subject("손님 10"))
        assertEquals("손님 12가", Korean.subject("손님 12"))
        assertEquals("음료 1을", Korean.objectOf("음료 1"))
        assertEquals("음료 2를", Korean.objectOf("음료 2"))
        assertEquals("과자를", Korean.objectOf("과자"))
        assertEquals("물건을", Korean.objectOf("물건"))
    }
}

class DayBarTest {
    private val from = Instant.parse("2026-11-23T15:00:00Z") // 서울 11월 24일 0시
    private val to = Instant.parse("2026-11-24T15:00:00Z")

    @Test fun spanBecomesFractionOfDay() {
        val seg = DayBar.segments(listOf(RecordingSpan("2026-11-24T03:00:00Z", 3600.0)), from, to)
        assertEquals(1, seg.size)
        assertEquals(0.5f, seg[0].start, 0.001f)          // 서울 12시
        assertEquals(13f / 24f, seg[0].end, 0.001f)
    }

    @Test fun spanCrossingMidnightIsClipped() {
        val seg = DayBar.segments(listOf(RecordingSpan("2026-11-23T14:30:00Z", 3600.0)), from, to)
        assertEquals(0f, seg[0].start, 0.0001f)
        assertEquals(0.5f / 24f, seg[0].end, 0.001f)
    }

    @Test fun spansOutsideOrUnreadableAreSkipped() {
        val seg = DayBar.segments(
            listOf(RecordingSpan("2026-11-22T03:00:00Z", 60.0), RecordingSpan("엉뚱한 값", 60.0)), from, to,
        )
        assertTrue(seg.isEmpty())
    }

    @Test fun marksAndCountsInsideSpan() {
        val t1 = Instant.parse("2026-11-24T05:32:08Z")
        val t2 = Instant.parse("2026-11-24T05:18:40Z")
        val t3 = Instant.parse("2026-11-25T01:00:00Z") // 다음 날
        assertEquals(2, DayBar.marks(listOf(t1, t2, t3), from, to).size)
        val span = RecordingSpan("2026-11-24T05:20:00Z", 25 * 60.0)
        assertEquals(1, DayBar.countIn(listOf(t1, t2, t3), span))
    }

    @Test fun fractionIsClamped() {
        assertEquals(0f, DayBar.fraction(from.minusSeconds(10), from, to), 0f)
        assertEquals(1f, DayBar.fraction(to.plusSeconds(10), from, to), 0f)
        assertFalse(DayBar.fraction(from.plusSeconds(10), from, to) == 0f)
    }
}

class SettingsCheckTest {
    @Test fun failureReasons() {
        assertEquals(CheckResult.BadKey, SettingsCheck.classify(ApiException(401, "x")))
        assertEquals(CheckResult.BadKey, SettingsCheck.classify(ApiException(403, "x")))
        assertEquals(CheckResult.Failed("서버 오류 (HTTP 500)"), SettingsCheck.classify(ApiException(500, "x")))
        assertEquals(CheckResult.Unreachable, SettingsCheck.classify(SocketTimeoutException("timeout")))
        assertEquals(CheckResult.Unreachable, SettingsCheck.classify(IOException("refused")))
        assertTrue(SettingsCheck.classify(IllegalArgumentException("bad url")) is CheckResult.Failed)
    }
}
