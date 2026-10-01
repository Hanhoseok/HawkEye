package com.hawkeye.admin.domain

import org.junit.Assert.assertEquals
import org.junit.Test

class BackoffTest {
    @Test fun doublesUntilCapAndResets() {
        val b = Backoff(baseMs = 1_000, maxMs = 8_000)
        assertEquals(listOf(1_000L, 2_000L, 4_000L, 8_000L, 8_000L), List(5) { b.next() })
        b.reset()
        assertEquals(1_000L, b.next())
    }
}
