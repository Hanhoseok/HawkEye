package com.hawkeye.admin.domain

import com.hawkeye.admin.data.CaseSummary
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

fun case(
    id: Int,
    state: String = "active",
    level: String = "WARNING",
    updatedAt: String = "2026-10-01T10:00:00.000000+00:00",
) = CaseSummary(
    id = id, runId = "r", cameraId = "cam1", personId = id, level = level, state = state,
    taken = 1, paid = 0, unpaid = 1, reason = "", identityCheck = null, items = null,
    resolution = null, hasSnapshot = false, updatedAt = updatedAt, lastEventId = id,
)

class CaseStoreTest {
    @Test fun addsAndListsActiveCases() {
        val s = CaseStore()
        s.apply(case(1))
        assertEquals(listOf(1), s.list(CaseFilter.ACTIVE).map { it.id })
    }

    @Test fun newerUpdateReplacesOlderOne() {
        val s = CaseStore()
        s.apply(case(1, level = "WARNING", updatedAt = "2026-10-01T10:00:00.000000+00:00"))
        s.apply(case(1, level = "HIGH_RISK", updatedAt = "2026-10-01T10:00:05.000000+00:00"))
        assertEquals("HIGH_RISK", s.get(1)!!.level)
    }

    @Test fun staleUpdateIsIgnored() {
        val s = CaseStore()
        s.apply(case(1, level = "HIGH_RISK", updatedAt = "2026-10-01T10:00:05.000000+00:00"))
        s.apply(case(1, level = "WARNING", updatedAt = "2026-10-01T10:00:00.000000+00:00"))
        assertEquals("HIGH_RISK", s.get(1)!!.level)
    }

    @Test fun listIsNewestFirst() {
        val s = CaseStore()
        s.apply(case(1, updatedAt = "2026-10-01T10:00:00.000000+00:00"))
        s.apply(case(2, updatedAt = "2026-10-01T10:00:09.000000+00:00"))
        s.apply(case(3, updatedAt = "2026-10-01T10:00:03.000000+00:00"))
        assertEquals(listOf(2, 3, 1), s.list(CaseFilter.ALL).map { it.id })
    }

    @Test fun activeFilterHidesHandledCasesAndAllHidesSilentPasses() {
        val s = CaseStore()
        s.apply(case(1, state = "active"))
        s.apply(case(2, state = "resolved"))
        s.apply(case(3, state = "auto_cleared"))
        s.apply(case(4, state = "pass", level = "CLEAR"))
        assertEquals(listOf(1), s.list(CaseFilter.ACTIVE).map { it.id })
        assertEquals(setOf(1, 2, 3), s.list(CaseFilter.ALL).map { it.id }.toSet())
    }

    @Test fun latestUpdatedAtIsUsedToFetchMissedCases() {
        val s = CaseStore()
        assertNull(s.latestUpdatedAt())
        s.apply(case(1, updatedAt = "2026-10-01T10:00:00.000000+00:00"))
        s.apply(case(2, updatedAt = "2026-10-01T10:00:09.000000+00:00"))
        assertEquals("2026-10-01T10:00:09.000000+00:00", s.latestUpdatedAt())
    }
}
