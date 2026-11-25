package com.hawkeye.admin.domain

import com.hawkeye.admin.data.CaseSummary
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.Instant
import java.time.ZoneId

class ReconcileTest {
    private fun case(id: Int, state: String, updated: String, level: String = "HIGH_RISK") =
        CaseSummary(id = id, runId = "r", cameraId = "cam1", personId = id, level = level, state = state,
            updatedAt = updated, createdAt = updated)

    @Test fun caseMissingFromFullActiveListIsRechecked() {
        val local = listOf(case(1, "active", "2026-11-24T05:00:00Z"), case(2, "active", "2026-11-24T05:10:00Z"))
        val server = listOf(case(2, "active", "2026-11-24T05:10:00Z"))
        assertEquals(listOf(1), Reconcile.staleActive(local, server, truncated = false))
    }

    @Test fun truncatedListProvesNothing() {
        val local = listOf(case(1, "active", "2026-11-24T05:00:00Z"))
        assertTrue(Reconcile.staleActive(local, emptyList(), truncated = true).isEmpty())
    }

    /**
     * 끊긴 동안: 사건 A 를 다른 폰에서 처리 -> 그 뒤 다른 사건 100건 이상 갱신 -> 재연결.
     * 증분 응답(최신 100건)에도 미처리 응답에도 A 가 없다. 대조로 A 를 다시 확인하면 카드와 배지가 바로잡힌다.
     */
    @Test fun handledWhileOfflineIsCorrectedAfterRecheck() {
        val store = CaseStore()
        store.apply(case(1, "active", "2026-11-24T05:00:00Z"))
        (100 until 220).map {
            val m = it - 100
            case(it, "auto_cleared", "2026-11-24T06:%02d:%02dZ".format(m / 60, m % 60), level = "WARNING")
        }.sortedByDescending { it.updatedAt }.take(100).forEach { store.apply(it) }
        val serverActive = emptyList<CaseSummary>()
        val now = Instant.parse("2026-11-24T07:00:00Z")
        val zone = ZoneId.of("Asia/Seoul")
        val before = StatusSummaries.of(store.list(CaseFilter.ALL), false, SyncState.Done(), now, zone)
        assertEquals(Tone.DANGER, before.tone)

        val stale = Reconcile.staleActive(store.list(CaseFilter.ACTIVE), serverActive, truncated = false)
        assertEquals(listOf(1), stale)
        store.apply(case(1, "resolved", "2026-11-24T05:30:00Z"))
        val after = StatusSummaries.of(store.list(CaseFilter.ALL), false, SyncState.Done(), now, zone)
        assertEquals(Tone.CALM, after.tone)
        assertEquals(0, StatusSummaries.activeCount(store.list(CaseFilter.ALL)))
    }

    @Test fun recheckDoesNotOverwriteNewerSocketUpdate() {
        val store = CaseStore()
        store.apply(case(1, "active", "2026-11-24T05:40:00Z", level = "HIGH_RISK"))
        store.apply(case(1, "active", "2026-11-24T05:00:00Z", level = "WARNING"))
        assertEquals("HIGH_RISK", store.get(1)!!.level)
    }

    @Test fun removedCaseDisappears() {
        val store = CaseStore()
        store.apply(case(1, "active", "2026-11-24T05:00:00Z"))
        assertTrue(store.remove(1))
        assertTrue(store.list(CaseFilter.ALL).isEmpty())
    }
}
