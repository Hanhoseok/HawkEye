package com.hawkeye.admin.domain

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LinksTest {
    @Test fun firstConnectIsNotOffline() {
        assertFalse(Links.offline(connected = false, disconnected = false, everConnected = false))
    }

    /** 동기화 끝 → 끊김 → 다시 붙는 중: 연결될 때까지 끊김 (정상 카드가 나오면 안 됨) */
    @Test fun reconnectingAfterDropStaysOffline() {
        assertTrue(Links.offline(connected = false, disconnected = true, everConnected = true))
        assertTrue(Links.offline(connected = false, disconnected = false, everConnected = true))
        assertFalse(Links.offline(connected = true, disconnected = false, everConnected = true))
        val s = StatusSummaries.of(emptyList(), Links.offline(false, false, true), SyncState.Done(), java.time.Instant.now())
        assertTrue(s.tone == Tone.OFF)
    }

    @Test fun firstConnectFailureIsOffline() {
        assertTrue(Links.offline(connected = false, disconnected = true, everConnected = false))
    }
}
